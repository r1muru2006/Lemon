from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_threshold_for_max_fpr(
    y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.01
) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    thresholds = np.unique(y_score)[::-1]
    best_threshold = 0.5
    best_recall = -1.0

    negatives = max((y_true == 0).sum(), 1)

    for thr in thresholds:
        pred = (y_score >= thr).astype(int)
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())

        fpr = fp / negatives
        recall = tp / max(tp + fn, 1)

        if fpr <= max_fpr and recall > best_recall:
            best_recall = recall
            best_threshold = float(thr)

    return best_threshold


def binary_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        metrics["pr_auc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update(
        {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "fpr": float(fp / max(fp + tn, 1)),
            "tpr": float(tp / max(tp + fn, 1)),
        }
    )

    return metrics


def precision_recall_points(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, list]:
    p, r, t = precision_recall_curve(y_true, y_score)
    return {
        "precision": [float(x) for x in p],
        "recall": [float(x) for x in r],
        "thresholds": [float(x) for x in t],
    }
