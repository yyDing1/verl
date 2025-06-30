# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import logging
import os

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh

from verl import DataProto
from verl.third_party.vllm import customized_vllm
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer, omega_conf_to_dataclass
from verl.utils.debug import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"])
    return device_mesh


class GenerativeRewardModelWorker(Worker, DistProfilerExtension):
    def __init__(self, config):
        Worker.__init__(self)
        profiler_config = omega_conf_to_dataclass(config.get("profiler", {}), ProfilerConfig)
        DistProfilerExtension.__init__(self, DistProfiler(rank=self.rank, config=profiler_config))

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=get_nccl_backend())
        self.config = config

    def _build_genrm(self, config):
        from torch.distributed.device_mesh import init_device_mesh
        from verl.utils.model import get_generation_config

        infer_tp = config.vllm_infer.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"genrm world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        genrm_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])

        log_gpu_memory_usage("Before building GenRM", logger=logger)
        use_shm = config.model.get("use_shm", False)
        trust_remote_code = config.model.get("trust_remote_code", False)
        local_path = copy_to_local(config.model.path, use_shm=use_shm)
        input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer, use_shm=use_shm)
        self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=trust_remote_code)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        from .genrm_sharding_manager import VLLMShardingManager
        from .genrm_vllm_spmd import vLLMSyncInfer

        genrm = vLLMSyncInfer(model_path=local_path, config=config.vllm_infer)

        log_gpu_memory_usage("After building GenRM", logger=logger)

        genrm_sharding_manager = VLLMShardingManager(
            inference_engine=genrm.inference_engine,
            device_mesh=genrm_device_mesh,
        )
        log_gpu_memory_usage("After building sharding manager", logger=logger)

        return genrm, genrm_sharding_manager

    def _switch_genrm_chat_template(self, data: DataProto):
        src_max_length = data.batch["attention_mask"].shape[-1]
        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        for i in range(data.batch.batch_size[0]):
            # extract question
            prompt_ids = data.batch["promts"][i]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data.batch["attention_mask"][i][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode & remove bos / eos
            prompt = src_tokenizer.decode(valid_prompt_ids)
            prompt = prompt.replace(src_tokenizer.bos_token, "")
            response = src_tokenizer.decode(valid_response_ids)
            response = response.replace(src_tokenizer.eos_token, "")

            chat_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat_messages, add_generation_prompt=False, tokenize=False)
            # TODO HERE

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.genrm, self.genrm_sharding_manager = self._build_genrm(config=self.config)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        data = data.to(get_device_id())

        prompts = self._switch_genrm_chat_template(data)
        prompts.batch = prompts.batch.to(get_device_id())
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        timing_generate = {}
        with self.genrm_sharding_manager:
            log_gpu_memory_usage("After entering GenRM sharding manager", logger=logger)

            prompts = self.genrm_sharding_manager.preprocess_data(prompts)
            with simple_timer("generate_sequences", timing_generate):
                output = self.genrm.generate_sequences(prompts=prompts)

            log_gpu_memory_usage("After rollout generation", logger=logger)
            output = self.genrm_sharding_manager.postprocess_data(output)

        timing_generate.update(self.rollout_sharding_manager.timing)
        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        # clear kv cache
        get_torch_device().empty_cache()
        return output
