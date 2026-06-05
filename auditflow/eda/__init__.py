# ============================================================
# auditflow/eda/__init__.py
# ============================================================

from auditflow.eda.profiler import profile
from auditflow.eda.visualizer import (
    plot_distributions,
    plot_correlations,
    plot_categoricals,
    plot_target_vs_features,
    plot_class_balance,
    plot_missing_map,
)

__all__ = [
    "profile",
    "plot_distributions",
    "plot_correlations",
    "plot_categoricals",
    "plot_target_vs_features",
    "plot_class_balance",
    "plot_missing_map",
]
