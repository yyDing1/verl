# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
The main entry point to run the GenRM PPO algorithm
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
from verl.utils.debug.performance import reduce_timing
from verl.utils.device import get_device_id, get_device_name, get_nccl_backend, get_torch_device, is_cuda_available, is_npu_available
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
        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        genrm_raw_prompt_ids = []

        for i in range(data.batch.batch_size[0]):
            # extract response
            response_ids = data.batch["responses"][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode & remove bos / eos
            prompt = data.non_tensor_batch["extra_info"][0]["question"]
            response = src_tokenizer.decode(valid_response_ids)
            response = response.replace(src_tokenizer.eos_token, "")

            chat_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat_messages, add_generation_prompt=False, tokenize=False)
            if self.rank == 0 and i == 0:
                print(f"Switch template. chat: {prompt_with_chat_template}")

            max_prompt_length = self.config.vllm_infer.prompt_length
            raw_prompt_ids = self.tokenizer.encode(prompt_with_chat_template, add_special_tokens=False)
            if len(raw_prompt_ids) > max_prompt_length:
                raw_prompt_ids = raw_prompt_ids[: max_prompt_length]  # Force to truncate Right
            genrm_raw_prompt_ids.append(raw_prompt_ids)

        return DataProto.from_dict(non_tensors={"raw_prompt_ids": genrm_raw_prompt_ids})

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
        timing_generate = {}
        with self.genrm_sharding_manager:
            log_gpu_memory_usage("After entering GenRM sharding manager", logger=logger)

            prompts = self.genrm_sharding_manager.preprocess_data(prompts)
            with simple_timer("generate_sequences", timing_generate):
                output = self.genrm.generate_sequences(prompts=prompts)

            log_gpu_memory_usage("After GenRM generation", logger=logger)
            output = self.genrm_sharding_manager.postprocess_data(output)

        timing_generate.update(self.rollout_sharding_manager.timing)
        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate = reduce_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        # clear kv cache
        get_torch_device().empty_cache()
        return output
