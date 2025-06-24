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
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import omega_conf_to_dataclass
from verl.utils.debug import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage
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

        infer_tp = config.vllm_infer.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f"genrm world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        genrm_device_mesh = init_device_mesh(device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"])

        log_gpu_memory_usage("Before building GenRM", logger=logger)
        local_path = copy_to_local(config.model.path, use_shm=config.model.get("use_shm", False))

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

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))
        self.genrm, self.genrm_sharding_manager = self._build_genrm(config=self.config)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="brown")
    def compute_rm_score(self, data: DataProto):
        data = data.to(get_device_id())
        breakpoint()

        return
