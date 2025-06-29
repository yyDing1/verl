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

import logging
import os

from torch.distributed.device_mesh import DeviceMesh

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device
from verl.utils.torch_functional import check_device_is_available
from verl.workers.sharding_manager.base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class VLLMShardingManager(BaseShardingManager):
    @check_device_is_available()
    def __init__(self, inference_engine: LLM, device_mesh: DeviceMesh = None):
        self.inference_engine = inference_engine
        self.device_mesh = device_mesh

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = get_torch_device().get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    @GPUMemoryLogger(role="genrm vllm sharding_manager", logger=logger)
    def __enter__(self):
        get_torch_device().empty_cache()
        self.inference_engine.wake_up()

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="genrm vllm sharding_manager", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        self.inference_engine.sleep(level=1)

    @GPUMemoryLogger(role="genrm vllm sharding_manager", logger=logger)
    def preprocess_data(self, data: DataProto) -> DataProto:
        if self.tp_size == 1:
            return data

        group = vllm_ps.get_tensor_model_parallel_group().device_group
        all_gather_data_proto(data=data, process_group=group)
        return data

    @GPUMemoryLogger(role="genrm vllm sharding_manager", logger=logger)
    def postprocess_data(self, data: DataProto) -> DataProto:
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]
