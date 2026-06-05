# ============================================================
# auditflow/models/__init__.py
# ============================================================

from auditflow.models.classical import ModelTrainer, get_model
from auditflow.models.explainer import explain_model, plot_feature_importance

__all__ = [
    "ModelTrainer",
    "get_model",
    "explain_model",
    "plot_feature_importance",
]
