from typing import Dict, List, Union
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

def compute_nli_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    label_names: List[str] = None
) -> Dict[str, float]:
    """
    Tính toán các chỉ số đánh giá chuẩn cho bài toán NLI 3 nhãn.
    Primary metric: Accuracy
    Secondary metric: Macro-F1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1
    }

    if label_names:
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(label_names))), zero_division=0)
        for idx, name in enumerate(label_names):
            metrics[f"f1_{name.lower()}"] = float(f1[idx])

    return metrics
