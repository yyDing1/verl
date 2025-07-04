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

from concurrent.futures import ThreadPoolExecutor
from time import sleep

import requests

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

BASE_URL = "http://localhost:30000"
API_KEY = "EMPTY"
MAX_RETRIES = 3
BASE_DELAY = 2
MAX_WORKERS = 32
MODEL_NAME = "Qwen/Qwen3-8B"
QUERY_TEMPLATE_PRM_GT = (
    "The following is a math problem with its ground truth answer, along with an AI solution (split into paragraphs, enclosed with tags and indexed from 0):\n\n"
    "[Math Problem]\n\n"
    "{problem}\n\n"
    "[Ground Truth]\n\n"
    "{ground_truth}\n\n"
    "[AI Solution]\n\n"
    "{tagged_response}\n\n"
    "Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes 'not found').\n\n"
    "Please reason step by step, put your final answer (i.e., the index) in \\boxed{{}}."
)

def get_response(problem, solution_str, ground_truth):
    prompt = QUERY_TEMPLATE_PRM_GT.format(problem=problem, solution=solution_str)
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(MAX_RETRIES):
        try:
            headers = {"Content-Type": "application/json"}
            chat_url = f"{BASE_URL}/v1/chat/completions"
            data = {"model": MODEL_NAME, "messages": messages}
            output = requests.post(chat_url, headers=headers, json=data, timeout=30)
            response = output.json()["choices"][0]["message"]["content"]
            return response
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print("Exception: ", repr(e))
                delay = BASE_DELAY * (2**attempt)
                print(f"Retrying in {delay} seconds...")
                sleep(delay)
            else:
                print(f"Failed after {MAX_RETRIES} attempts. Error: {e}")

    raise ConnectionRefusedError(f"Failed to run the model for {prompt}!")


def compute_reward(response):
    reward_score = 0.0
    try:
        boxed_result = last_boxed_only_string(response)
        if boxed_result is not None:
            result = remove_boxed(boxed_result)
            reward_score = float(result == "True")
    except Exception as e:
        print(e)
    return reward_score


from recipe.genrm.baselines_func.reward_function import compute_score as compute_score_rule

def compute_score_rule_with_genrm(solution_str, ground_truth, extra_info, **kwargs):
    question, split = extra_info["question"], extra_info["split"]
    rule_result = compute_score_rule(solution_str, ground_truth)
    if rule_result["acc"] == 0:
        rule_result[""]
        pass

def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos
        ):
            future = executor.submit(compute_score, data_source, solution_str, ground_truth, extra_info)
            futures.append(future)

        results = [future.result() for future in futures]

    return results
