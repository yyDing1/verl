import logging
import os

import numpy as np
import torch
import torch.distributed
from omegaconf import DictConfig
from vllm import LLM, SamplingParams

from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class vLLMSyncInfer:
    def __init__(self, model_path: str, config: DictConfig):
        self.config = config
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=config.get("tensor_model_parallel_size", 1),
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=config.get("max_num_batched_tokens", 8192),
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=True,
            seed=config.get("seed", 0),
        )
        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,
            max_tokens=config.response_length,
        )
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

    @GPUMemoryLogger(role="GenRM infer spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        non_tensor_batch = prompts.non_tensor_batch
        raw_prompt_ids = non_tensor_batch["raw_prompt_ids"]

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        outputs = self.inference_engine.generate(
            prompts=vllm_inputs,  # because we have already convert it to prompt token id
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        response = []
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response_text = output.outputs[sample_id].text
                response.append(response_text)

        return DataProto(non_tensor_batch={"genrm_response": response})
