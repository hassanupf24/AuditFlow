# ============================================================
# auditflow/models/explainer.py
# Model explainability with audit trail
# ============================================================

"""
Built-in explainability for every trained model:
  - Feature importance (native for tree models, permutation for others)
  - Importance bar chart auto-stored in the report
"""

import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, List, Optional

from sklearn.inspection import permutation_importance

from auditflow.core.logger import get_logger


def explain_model(
    model: Any,
    feature_names: List[str],
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.

    For tree-based models (RF, GBM, DT), uses native feature_importances_.
    For other models, uses permutation importance on the test set.

    Parameters
    ----------
    model         : Fitted sklearn estimator.
    feature_names : List[Any] of feature names.
    X_test        : Test features (required for permutation importance).
    y_test        : Test labels (required for permutation importance).
    top_n         : Number of top features to return.

    Returns
    -------
    pd.DataFrame with columns: feature, importance (sorted descending)
    """
    audit = get_logger()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        method = "native (tree-based)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
        if len(importances) != len(feature_names):
            importances = importances[: len(feature_names)]
        method = "coefficient magnitude"
    elif X_test is not None and y_test is not None:
        perm_result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
        )
        importances = perm_result.importances_mean
        method = "permutation importance"
    else:
        audit.log_decision(
            module="models.explainer",
            action="explain_skipped",
            rationale="Could not extract feature importance: model has no "
            "feature_importances_ or coef_, and no test set provided "
            "for permutation importance.",
        )
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names[: len(importances)],
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    top_features = importance_df.head(5)["feature"].tolist()

    audit.log_decision(
        module="models.explainer",
        action="explain_model",
        rationale=f"Extracted feature importance using {method}. "
        f"Top 5 features: {top_features}. These features contribute "
        f"most to the model's predictions.",
        details={
            "method": method,
            "top_features": importance_df.head(10).to_dict(orient="records"),
        },
    )

    return importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Feature Importance",
    top_n: int = 15,
    save_path: Optional[str] = None,
) -> Any:
    """
    Horizontal bar chart of feature importance, auto-stored in the report.
    """
    audit = get_logger()
    data = importance_df.head(top_n).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(data) * 0.35)))

    colors = plt.cm.RdYlBu(np.linspace(0.2, 0.8, len(data)))  # type: ignore
    ax.barh(data["feature"], data["importance"], color=colors)
    ax.set_xlabel("Importance")
    ax.set_title(title, fontweight="bold", fontsize=13)
    plt.tight_layout()

    # Store in audit logger for the report
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    audit.store_figure(title, b64)

    if save_path:
        fig.savefig(save_path)
    return fig
