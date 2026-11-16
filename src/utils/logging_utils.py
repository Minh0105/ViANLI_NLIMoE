import os
import csv
from typing import Dict, Any, List
import pandas as pd

SUMMARY_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "summary.csv")

def append_to_summary(record: Dict[str, Any], csv_path: str = None) -> None:
    """
    Ghi một dòng kết quả thực nghiệm vào summary.csv (19 cột chuẩn).
    """
    if csv_path is None:
        csv_path = SUMMARY_CSV_PATH

    columns = [
        "experiment_name", "model", "backbone", "router_type", "routing_type",
        "threshold", "top_k", "num_experts", "seed", "val_accuracy",
        "val_macro_f1", "test_accuracy", "test_macro_f1", "avg_experts_per_sample",
        "routing_entropy", "train_time", "inference_time", "peak_gpu_memory", "notes"
    ]
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        row = {col: record.get(col, "") for col in columns}
        writer.writerow(row)
    print(f"[Logging] Saved experiment record to {csv_path}")

def save_predictions(
    save_path: str,
    uids: List[Any],
    gold_labels: List[int],
    pred_labels: List[int],
    probabilities: List[List[float]],
    label_map: Dict[int, str] = None
) -> None:
    """
    Lưu dự đoán ra file CSV theo chuẩn:
    uid, gold_label, pred_label, prob_entailment, prob_neutral, prob_contradiction
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    records = []
    for idx, (uid, gold, pred, probs) in enumerate(zip(uids, gold_labels, pred_labels, probabilities)):
        gold_str = label_map.get(gold, str(gold)) if label_map else str(gold)
        pred_str = label_map.get(pred, str(pred)) if label_map else str(pred)
        
        row = {
            "uid": uid if uid is not None else idx,
            "gold_label": gold_str,
            "pred_label": pred_str,
            "prob_entailment": probs[0] if len(probs) > 0 else 0.0,
            "prob_neutral": probs[1] if len(probs) > 1 else 0.0,
            "prob_contradiction": probs[2] if len(probs) > 2 else 0.0,
        }
        records.append(row)
        
    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"[Logging] Saved {len(records)} predictions to {save_path}")
