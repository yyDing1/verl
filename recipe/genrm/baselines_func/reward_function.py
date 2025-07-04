# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
from typing import Optional

from verl.utils.reward_score.math_dapo import last_boxed_only_string, remove_boxed, normalize_final_answer


def verify(
    solution_str: str, gt: str,
) -> tuple[bool, str]:
    boxed_answer = last_boxed_only_string(solution_str)
    if boxed_answer is not None:
        extracted_answer = remove_boxed(boxed_answer)
    else:
        extracted_answer = "[INVALID]"

    pred = normalize_final_answer(extracted_answer)
    gt = normalize_final_answer(gt)
    return (pred == gt), pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    **kwargs,
) -> float:
    # Limit solution length for efficiency
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Verify the solution
    correct, pred = verify(solution_str, ground_truth)

    reward = 1.0 if correct else -1.0
    acc = correct

    return {
        "score": reward,
        "acc": acc,
        "pred": pred,
    }