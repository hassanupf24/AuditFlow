# ============================================================
# auditflow/eda/visualizer.py
# Auto-visualization with audit trail + report integration
# ============================================================

import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Tuple, Optional, Tuple, List, Optional

from auditflow.core.logger import get_logger

# ── Global style ──────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def _fig_to_base64(fig: Any) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _store_figure(fig: Any, name: str) -> None:
    """Store a figure in the audit logger for the HTML report."""
    audit = get_logger()
    b64 = _fig_to_base64(fig)
    audit.store_figure(name, b64)


# ─────────────────────────────────────────────
# Distribution Grid
# ─────────────────────────────────────────────
def plot_distributions(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    n_cols: int = 3,
    kde: bool = True,
    save_path: Optional[str] = None,
) -> 'Any':
    """
    KDE + histogram grid for all numeric columns.
    Auto-stored in the audit logger for the HTML report.
    """
    num_cols = cols or df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        return plt.figure()

    n_rows = -(-len(num_cols) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.histplot(df[col].dropna(), kde=kde, ax=axes[i], color="#4C72B0")
        mean_val = df[col].mean()
        median_val = df[col].median()
        axes[i].axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Mean: {mean_val:.1f}",
        )
        axes[i].axvline(
            median_val,
            color="orange",
            linestyle="--",
            linewidth=1,
            label=f"Median: {median_val:.1f}",
        )
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Numeric Feature Distributions", fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _store_figure(fig, "Numeric Distributions")
    if save_path:
        fig.savefig(save_path)
    return fig


# ─────────────────────────────────────────────
# Correlation Heatmap
# ─────────────────────────────────────────────
def plot_correlations(
    df: pd.DataFrame,
    method: str = "pearson",
    figsize: Tuple[Any, ...] = (10, 8),
    save_path: Optional[str] = None,
) -> 'Any':
    """
    Annotated correlation heatmap (lower triangle).
    """
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) < 2:
        return plt.figure()

    corr = df[num_cols].corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(
        f"Correlation Matrix ({method.capitalize()})", fontweight="bold", fontsize=13
    )
    plt.tight_layout()
    _store_figure(fig, f"Correlation Heatmap ({method})")
    if save_path:
        fig.savefig(save_path)
    return fig


# ─────────────────────────────────────────────
# Categorical Bar Charts
# ─────────────────────────────────────────────
def plot_categoricals(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    top_n: int = 15,
    n_cols: int = 2,
    save_path: Optional[str] = None,
) -> 'Any':
    """
    Horizontal bar chart of value counts for each categorical column.
    """
    cat_cols = cols or df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return plt.figure()

    n_rows = -(-len(cat_cols) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        counts = df[col].value_counts().nlargest(top_n)
        axes[i].barh(counts.index.astype(str), counts.values, color="#DD8452")
        axes[i].invert_yaxis()
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel("Count")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Categorical Feature Distributions", fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _store_figure(fig, "Categorical Distributions")
    if save_path:
        fig.savefig(save_path)
    return fig


# ─────────────────────────────────────────────
# Target vs Feature Box Plots
# ─────────────────────────────────────────────
def plot_target_vs_features(
    df: pd.DataFrame,
    target: str,
    feature_cols: Optional[List[str]] = None,
    n_cols: int = 3,
    save_path: Optional[str] = None,
) -> 'Any':
    """
    Box plots of each numeric feature grouped by the target variable.
    """
    feature_cols = feature_cols or [
        c for c in df.select_dtypes(include=np.number).columns if c != target
    ]
    if not feature_cols:
        return plt.figure()

    n_rows = -(-len(feature_cols) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        sns.boxplot(data=df, x=target, y=col, ax=axes[i], palette="Set2")
        axes[i].set_title(f"{col} vs {target}", fontweight="bold")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Features by Target: '{target}'", fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _store_figure(fig, f"Features vs {target}")
    if save_path:
        fig.savefig(save_path)
    return fig


# ─────────────────────────────────────────────
# Class Balance
# ─────────────────────────────────────────────
def plot_class_balance(
    df: pd.DataFrame,
    target: str,
    figsize: Tuple[Any, ...] = (7, 4),
    save_path: Optional[str] = None,
) -> 'Any':
    """
    Bar chart with percentage annotations showing class distribution.
    """
    counts = df[target].value_counts()
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        counts.index.astype(str),
        counts.values,
        color=sns.color_palette("Set2", len(counts)),
    )

    for bar, val in zip(bars, counts.values):
        pct = val / counts.sum() * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{pct:.1f}%",
            ha="center",
            fontsize=10,
        )

    ax.set_title(f"Class Balance — '{target}'", fontweight="bold")
    ax.set_ylabel("Count")
    plt.tight_layout()
    _store_figure(fig, f"Class Balance ({target})")
    if save_path:
        fig.savefig(save_path)
    return fig


# ─────────────────────────────────────────────
# Missing Data Map
# ─────────────────────────────────────────────
def plot_missing_map(
    df: pd.DataFrame,
    figsize: Tuple[Any, ...] = (14, 5),
    save_path: Optional[str] = None,
) -> 'Any':
    """
    Binary heatmap: dark = missing, light = present.
    """
    if df.isnull().sum().sum() == 0:
        return plt.figure()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis", ax=ax)
    ax.set_title("Missing Data Map (dark = missing)", fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    _store_figure(fig, "Missing Data Map")
    if save_path:
        fig.savefig(save_path)
    return fig
