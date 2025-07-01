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
Preprocess the dataset to parquet format
"""

import argparse
import os
from functools import partial

from datasets import concatenate_datasets, load_dataset

from verl.utils.hdfs_io import copy, makedirs


def example_map_fn(example, idx, process_fn, data_source, ability, split):
    prompt, ground_truth = process_fn(example)
    data = {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"split": split, "index": idx, "raw_info": example},
    }
    return data


def build_processbench(process=False, use_gt=False):

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
    QUERY_TEMPLATE_PRM_NOGT = (
        "The following is a math problem and an AI solution (split into paragraphs, enclosed with tags and indexed from 0):\n\n"
        "[Math Problem]\n\n"
        "{problem}\n\n"
        "[AI Solution]\n\n"
        "{tagged_response}\n\n"
        "Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes 'not found').\n\n"
        "Please reason step by step, put your final answer (i.e., the index) in \\boxed{{}}."
    )
    QUERY_TEMPLATE_ORM_GT = (
        "The following is a math problem with its ground truth answer, along with an AI solution:\n\n"
        "[Math Problem]\n\n"
        "{problem}\n\n"
        "[Ground Truth]\n\n"
        "{ground_truth}\n\n"
        "[AI Solution]\n\n"
        "{tagged_response}\n\n"
        "Your task is to review and critique the solution step by step, and output whether the AI solution is correct.\n\n"
        "Please reason step by step, put your final answer (i.e., 'True' or 'False') in \\boxed{{}}."
    )
    QUERY_TEMPLATE_ORM_NOGT = (
        "The following is a math problem and an AI solution:\n\n"
        "[Math Problem]\n\n"
        "{problem}\n\n"
        "[AI Solution]\n\n"
        "{tagged_response}\n\n"
        "Your task is to review and critique the solution step by step, and output whether the AI solution is correct.\n\n"
        "Please reason step by step, put your final answer (i.e., 'True' or 'False') in \\boxed{{}}."
    )

    def process_processbench(example):
        problem = example["problem"]
        problem_ground_truth = example["ground_truth"]
        tagged_response = ""
        for sdx, step in enumerate(example["steps"]):
            tagged_response += f'<paragraph_{sdx}>\n{step}\n</paragraph_{sdx}>\n\n'
        tagged_response = tagged_response.strip()
        if process:
            if use_gt:
                query_prompt = QUERY_TEMPLATE_PRM_GT.format(problem=problem, ground_truth=problem_ground_truth, tagged_response=tagged_response)
            else:
                query_prompt = QUERY_TEMPLATE_PRM_NOGT.format(problem=problem, tagged_response=tagged_response)
            ground_truth = int(example['label'])
        else:
            if use_gt:
                query_prompt = QUERY_TEMPLATE_ORM_GT.format(problem=problem, ground_truth=problem_ground_truth, tagged_response=tagged_response)
            else:
                query_prompt = QUERY_TEMPLATE_ORM_NOGT.format(problem=problem, tagged_response=tagged_response)
            ground_truth = int(example['label'] == -1)
        return query_prompt, ground_truth

    data_path = "/mnt/hdfs/resources/datasets/ProcessBench_with_GT/"
    data_source = f"ProcessBench_{'PRM' if process else 'ORM'}_{'GT' if use_gt else 'NOGT'}"
    print(f"Loading the {data_path} dataset ...", flush=True)

    datasets = []
    for config in ["gsm8k", "math", "olympiadbench", "omnimath"]:
        dataset = load_dataset(data_path, split=config)
        map_fn = partial(
            example_map_fn, process_fn=process_processbench, data_source=f"{data_source}/{config}", ability="Math-Reward", split="test"
        )
        dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
        datasets.append(dataset)

    datasets = concatenate_datasets(datasets)
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/r1")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    final_dataset = concatenate_datasets(
        [
            build_processbench(process=True, use_gt=True),
            build_processbench(process=True, use_gt=False),
            build_processbench(process=False, use_gt=True),
            build_processbench(process=False, use_gt=False),
        ]
    )
    final_dataset.to_parquet(os.path.join(args.local_dir, "processbench_full.parquet"))

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
