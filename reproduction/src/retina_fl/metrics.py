from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def classification_metrics(
    targets: list[int] | np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    y_true = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = probabilities.argmax(axis=1)
    labels = np.arange(len(class_names))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class: dict[str, dict[str, float | int | None]] = {}
    for index, name in enumerate(class_names):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        per_class[name] = {
            "support": int((y_true == index).sum()),
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
        }
    binary = label_binarize(y_true, classes=labels)
    if len(class_names) == 2 and binary.shape[1] == 1:
        binary = np.column_stack([1 - binary[:, 0], binary[:, 0]])
    try:
        auroc = float(roc_auc_score(binary, probabilities, average="macro", multi_class="ovr"))
    except ValueError:
        auroc = None
    try:
        auprc = float(average_precision_score(binary, probabilities, average="macro"))
    except ValueError:
        auprc = None
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_ovr_auroc": auroc,
        "macro_ovr_auprc": auprc,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = (probabilities >= threshold).astype(int)
    per_class: dict[str, dict[str, float | int | None]] = {}
    for index, name in enumerate(class_names):
        truth = y_true[:, index]
        prediction = y_pred[:, index]
        tp = int(((truth == 1) & (prediction == 1)).sum())
        fn = int(((truth == 1) & (prediction == 0)).sum())
        fp = int(((truth == 0) & (prediction == 1)).sum())
        tn = int(((truth == 0) & (prediction == 0)).sum())
        try:
            auroc = float(roc_auc_score(truth, probabilities[:, index]))
        except ValueError:
            auroc = None
        try:
            auprc = float(average_precision_score(truth, probabilities[:, index]))
        except ValueError:
            auprc = None
        per_class[name] = {
            "support": int(truth.sum()),
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "auroc": auroc,
            "auprc": auprc,
        }
    return {
        "n": int(len(y_true)),
        "decision_threshold": float(threshold),
        "exact_match_accuracy": float((y_true == y_pred).all(axis=1).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
    }

