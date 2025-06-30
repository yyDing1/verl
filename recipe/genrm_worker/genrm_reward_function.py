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

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
import numpy as np


def compute_score_for_each_critics(genrm_critic):
    reward_score = 0.0
    try:
        boxed_result = last_boxed_only_string(genrm_critic)
        if boxed_result is not None:
            result = remove_boxed(boxed_result)
            reward_score = float(result == "True")
    except Exception as e:
        print(e)
    return reward_score


def default_compute_score(data_source, solution_str, ground_truth, genrm_critics=None, extra_info=None):
    # if genrm is unavailable, compute rule-based reward
    if genrm_critics is None:
        final_score = default_compute_score(data_source, solution_str, ground_truth)
    else:
        scores = [compute_score_for_each_critics(critic["response"]) for critic in genrm_critics]
        final_score = np.mean(scores)
    return {
        "score": final_score,
    }
