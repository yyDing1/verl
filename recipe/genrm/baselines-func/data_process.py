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
    question, prompt, ground_truth = process_fn(example)
    data = {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"split": split, "index": idx, "question": question},
    }
    return data


def build_math500_dataset():
    def process_math500(example):
        question, ground_truth = example["problem"], str(example["answer"])
        prompt = question.strip() + "\n\n" + "Please reason step by step, and put your final answer within \\boxed{}."
        return question, prompt, ground_truth

    data_source = "HuggingFaceH4/MATH-500"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="test")
    map_fn = partial(example_map_fn, process_fn=process_math500, data_source=data_source, ability="Math", split="test")
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_aime2024_dataset():
    def process_aime2024(example):
        question, ground_truth = example["Problem"], str(example["Answer"])
        prompt = question.strip() + "\n\n" + "Please reason step by step, and put your final answer within \\boxed{}."
        return question, prompt, ground_truth

    data_source = "Maxwell-Jia/AIME_2024"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    map_fn = partial(example_map_fn, process_fn=process_aime2024, data_source=data_source, ability="Math", split="test")
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_cnmo2024_dataset():
    def process_cnmo2024(example):
        question, ground_truth = example["question"], example["answer"][1:-1]
        prompt = question.strip() + "\n\n" + "Please reason step by step, and put your final answer within \\boxed{}."
        return question, prompt, ground_truth

    data_source = "opencompass/LiveMathBench"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    dataset_en = load_dataset(data_source, "v202412_CNMO_en", split="test")
    map_fn_en = partial(example_map_fn, process_fn=process_cnmo2024, data_source="opencompass/cnmo2024_en", ability="Math", split="test")
    dataset_en = dataset_en.map(map_fn_en, with_indices=True, remove_columns=dataset_en.column_names)

    dataset_zh = load_dataset(data_source, "v202412_CNMO_cn", split="test")
    map_fn_zh = partial(example_map_fn, process_fn=process_cnmo2024, data_source="opencompass/cnmo2024_zh", ability="Math", split="test")
    dataset_zh = dataset_zh.map(map_fn_zh, with_indices=True, remove_columns=dataset_zh.column_names)

    dataset = concatenate_datasets([dataset_en, dataset_zh])
    return dataset

def build_dapo_train_dataset():
    def process_dapo(example):
        question, ground_truth = example["prompt"], example["solution"]
        prompt = question.strip() + "\n\n" + "Please reason step by step, and put your final answer within \\boxed{}."
        return question, prompt, ground_truth

    data_source = "open-r1/DAPO-Math-17k-Processed"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, "all", split="train")
    map_fn = partial(example_map_fn, process_fn=process_dapo, data_source=data_source, ability="Math", split="train")
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


TASK2DATA = {
    "aime2024": build_aime2024_dataset,
    # "math500": build_math500_dataset,
    # "cnmo24": build_cnmo2024_dataset,
}
SUPPORTED_TASKS = TASK2DATA.keys()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/genrm")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--tasks", default="all")

    args = parser.parse_args()

    if args.tasks.lower() == "all":
        args.tasks = SUPPORTED_TASKS
    else:
        args.tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        for task in args.tasks:
            if task not in SUPPORTED_TASKS:
                raise NotImplementedError(f"{task} has not been supported.")

    train_dataset = build_dapo_train_dataset()
    test_datasets = []
    for task in args.tasks:
        test_datasets.append(TASK2DATA[task]())
    test_dataset = concatenate_datasets(test_datasets)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
