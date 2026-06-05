# ============================================================
# auditflow/evaluation/metrics.py
# Comprehensive evaluation with audit trail + report figures
# ============================================================

import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)

from auditflow.core.logger import get_logger


def evaluate_classification(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[List] = None,
) -> Dict[str, float]:
    """
    Compute all classification metrics with audit trail.

    Parameters
    ----------
    y_true : True labels.
    y_pred : Predicted labels.
    y_prob : Predicted probabilities (for ROC-AUC).
    labels : Class labels for display.

    Returns
    -------
    dict of metric name → value
    """
    audit = get_logger()
    avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, average=avg, zero_division=0), 4),
    }

    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                probs = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                metrics["roc_auc"] = round(roc_auc_score(y_true, probs), 4)
            else:
                metrics["roc_auc"] = round(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"), 4
                )
        except (ValueError, IndexError):
            pass

    # Determine quality
    f1 = metrics["f1_score"]
    if f1 >= 0.9:
        quality = "Excellent"
    elif f1 >= 0.7:
        quality = "Good"
    elif f1 >= 0.5:
        quality = "Fair"
    else:
        quality = "Poor"

    audit.log_decision(
        module="evaluation.metrics",
        action="evaluate_classification",
        rationale=f"Classification evaluation: F1={f1:.4f} ({quality}). "
                  f"Accuracy={metrics['accuracy']:.4f}, "
                  f"Precision={metrics['precision']:.4f}, "
                  f"Recall={metrics['recall']:.4f}."
                  f"{' ROC-AUC=' + str(metrics.get('roc_auc', 'N/A')) if 'roc_auc' in metrics else ''}",
        details=metrics,
    )

    return metrics


def evaluate_regression(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all regression metrics with audit trail.
    """
    audit = get_logger()

    metrics = {
        "r2": round(r2_score(y_true, y_pred), 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "mape": round(
            np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100, 2
        ),
    }

    r2 = metrics["r2"]
    quality = "Excellent" if r2 >= 0.9 else "Good" if r2 >= 0.7 else "Fair" if r2 >= 0.5 else "Poor"

    audit.log_decision(
        module="evaluation.metrics",
        action="evaluate_regression",
        rationale=f"Regression evaluation: R²={r2:.4f} ({quality}). "
                  f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
                  f"MAPE={metrics['mape']:.2f}%.",
        details=metrics,
    )

    return metrics


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot an annotated confusion matrix, auto-stored in the report.
    """
    audit = get_logger()
    cm = confusion_matrix(y_true, y_pred)
    display_labels = labels or sorted(np.unique(y_true))

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=display_labels, yticklabels=display_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()

    # Store for report
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    audit.store_figure(title, base64.b64encode(buf.read()).decode("utf-8"))

    if save_path:
        fig.savefig(save_path)
    return fig


def plot_roc_curve(
    y_true: pd.Series,
    y_prob: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curve for binary classification, auto-stored in the report.
    """
    audit = get_logger()

    probs = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4C72B0", linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#4C72B0")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    audit.store_figure(title, base64.b64encode(buf.read()).decode("utf-8"))

    if save_path:
        fig.savefig(save_path)
    return fig
