"""
Standardized evaluation metrics for binary interaction classification.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Computes standard classification metrics for binary predictions.
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_probs: Predicted probabilities in [0, 1].
        threshold: Classification decision boundary.
        
    Returns:
        Dictionary containing roc_auc, pr_auc, f1, acc, precision, recall, and cm.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=float)
    y_pred = (y_probs >= threshold).astype(int)

    # Robust ROC-AUC / PR-AUC calculation
    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_true, y_probs))
    except ValueError:
        pr_auc = float("nan")

    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "cm": cm,
    }


def format_metrics_report(
    title: str,
    metrics: Dict[str, Any],
    details: Optional[Dict[str, str]] = None,
) -> str:
    """
    Formats metrics dictionary into a readable markdown/text report block.
    """
    lines = [
        "=" * 70,
        f"🏥 {title.upper()}",
        "=" * 70,
    ]
    if details:
        for k, v in details.items():
            lines.append(f"{k}: {v}")
        lines.append("")

    lines.extend([
        "🏆 FINAL TEST SET METRICS:",
        "-" * 70,
        f"ROC-AUC (Main Metric) : {metrics['roc_auc']:.4f}",
        f"PR-AUC                : {metrics['pr_auc']:.4f}",
        f"F1-Score              : {metrics['f1']:.4f}",
        f"Accuracy              : {metrics['acc']:.4f}",
        f"Precision             : {metrics['precision']:.4f}",
        f"Recall                : {metrics['recall']:.4f}",
        "",
        "🧩 CONFUSION MATRIX:",
        f"                     Predicted Safe (0) | Predicted Toxic (1)",
        f"Actual Safe (0)    : {metrics['cm'][0, 0]:<18} | {metrics['cm'][0, 1]}",
        f"Actual Toxic (1)   : {metrics['cm'][1, 0]:<18} | {metrics['cm'][1, 1]}",
        "=" * 70,
        "",
    ])
    return "\n".join(lines)
