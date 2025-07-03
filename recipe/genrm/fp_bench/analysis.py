from datasets import load_dataset, concatenate_datasets
from collections import defaultdict
from tabulate import tabulate


data_source = "/mnt/hdfs/resources/datasets/ProcessBench_with_GT"
dataset = load_dataset(data_source)
dataset_list = [dataset[ds_name] for ds_name in ["gsm8k", "math", "olympiadbench", "omnimath"]]
dataset = concatenate_datasets(dataset_list)

def metric_single(dataset):
    postive_cnt, negative_cnt, false_positive_cnt = 0, 0, 0
    for line in dataset:
        if line["final_answer_correct"]:
            postive_cnt += 1
            if line["label"] != -1:
                false_positive_cnt += 1
        else:
            negative_cnt += 1
    return {
        "Positive": postive_cnt,
        "Negative": negative_cnt,
        "False Positive": false_positive_cnt,
        "FP Ratio": false_positive_cnt / postive_cnt,
    }

bucket = defaultdict(list)
for line in dataset:
    bucket[line["generator"]].append(line)

table_headers = ["Generator", "Positive", "Negative", "False Positive", "FP Ratio"]
float_fmt = (None,) * 4 + (".2%",)
table_rows = []
for generator, subdataset in bucket.items():
    metric = metric_single(subdataset)
    table_rows.append([generator, metric["Positive"]] + [metric[k] for k in table_headers[1:]])
table_rows.sort(key=lambda x: x[0])
print(tabulate(table_rows, headers=table_headers, tablefmt="grid", floatfmt=float_fmt))

