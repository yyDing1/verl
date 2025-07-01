from datasets import load_dataset
from collections import defaultdict, Counter
import argparse
import re
import numpy as np
from tabulate import tabulate


def extract_answer(solution_text: str):
    boxed_pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(boxed_pattern, solution_text)
    if matches:
        return matches[-1].strip()
    return None

def calculate_metric(data_source, samples):
    results = {}

    # ProcessBench Metrics
    error_data = [e for e in samples if e['extra_info']['raw_info']['label'] != -1]
    correct_data = [e for e in samples if e['extra_info']['raw_info']['label'] == -1]
    if "PRM" in data_source:
        err_acc = np.mean([e['extracted_final_answer'] != -1 for e in error_data])
        err_loc_acc = np.mean([e['extracted_final_answer'] == e['extra_info']['raw_info']['label']for e in error_data])
        corr_acc = np.mean([e['extracted_final_answer'] == -1 for e in correct_data])
        pb_f1 = 2 * err_loc_acc * corr_acc / (err_loc_acc + corr_acc)
    else:
        err_acc = np.mean([e['extracted_final_answer'] == False for e in error_data])
        err_loc_acc = None
        corr_acc = np.mean([e['extracted_final_answer'] == True for e in correct_data])
        pb_f1 = None
    results.update(
        {
            "Err (PB)": err_acc,
            "Err_Loc (PB)": err_loc_acc,
            "Corr (PB)": corr_acc,
            "F1 (PB)": pb_f1,
        }
    )

    # FalsePositiveBench Metrics
    positive_data = [e for e in samples if e["extra_info"]["raw_info"]["final_answer_correct"]]
    if "PRM" in data_source:
        pred_true_fp_cnt = len([e for e in positive_data if e['extracted_final_answer'] != -1 and e["extra_info"]["raw_info"]["label"] != -1])
        pred_fp_cnt = len([e for e in positive_data if e['extracted_final_answer'] != -1])
        true_fp_cnt = len([e for e in positive_data if e["extra_info"]["raw_info"]["label"] != -1])
        precision = pred_true_fp_cnt / pred_fp_cnt
        recall = pred_true_fp_cnt / true_fp_cnt
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        pred_true_fp_cnt = len([e for e in positive_data if e['extracted_final_answer'] == False and e["extra_info"]["raw_info"]["label"] != -1])
        pred_fp_cnt = len([e for e in positive_data if e['extracted_final_answer'] == False])
        true_fp_cnt = len([e for e in positive_data if e["extra_info"]["raw_info"]["label"] != -1])
        precision = pred_true_fp_cnt / pred_fp_cnt
        recall = pred_true_fp_cnt / true_fp_cnt
        f1_score = 2 * precision * recall / (precision + recall)
    results.update(
        {
            "Precision (FPB)": precision,
            "Recall (FPB)": recall,
            "F1 (FPB)": f1_score,
        }
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", default="~/data/processbench_orm_format")
    args = parser.parse_args()

    eval_samples = load_dataset("parquet", data_files=args.parquet_path, split="train")

    # split samples
    samples_bucket = defaultdict(list)
    for item in eval_samples:
        source_type = item["data_source"].split("/")[0]
        responses = item["responses"]
        extracted_answers = [extract_answer(response) for response in responses]
        valid_answers = [answer for answer in extracted_answers if answer is not None]
        maj_answers = Counter(valid_answers).most_common(1)[0][0] if len(valid_answers) > 0 else None
        try:
            maj_answers = eval(maj_answers)
        except:
            maj_answers = None
        item["extracted_final_answer"] = maj_answers
        samples_bucket[source_type].append(item)

    table_headers = ["Prompt Type", "Count", "Err (PB)", "Err_Loc (PB)", "Corr (PB)", "F1 (PB)", "Precision (FPB)", "Recall (FPB)", "F1 (FPB)"]
    table_rows = []
    for source_type, samples in samples_bucket.items():
        metric = calculate_metric(source_type, samples)
        table_rows.append([source_type, len(samples)] + [metric[k] for k in table_headers[2:]])
    print(tabulate(table_rows, headers=table_headers, tablefmt="grid", floatfmt=".2%"))