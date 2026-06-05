# ============================================================
# auditflow/eda/profiler.py
# Statistical profiler with smart alerts
# ============================================================

"""
Generates a comprehensive statistical profile of a DataFrame and
auto-detects common issues:
  - High cardinality columns
  - Class imbalance
  - Multicollinearity (|r| > 0.9)
  - Zero-variance (constant) columns
  - Highly skewed distributions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from auditflow.core.logger import get_logger


def profile(
    df: pd.DataFrame,
    target: Optional[str] = None,
    correlation_threshold: float = 0.9,
) -> Dict[str, Any]:
    """
    Generate a comprehensive data profile with smart alerts.

    Parameters
    ----------
    df                    : DataFrame to profile.
    target                : Target column name (for class balance analysis).
    correlation_threshold : Flag pairs with |correlation| above this.

    Returns
    -------
    dict with keys:
        "shape"       : (rows, cols)
        "dtypes"      : {col: dtype}
        "numeric"     : summary stats DataFrame
        "categorical" : {col: {n_unique, top_values}}
        "missing"     : {col: {count, pct}}
        "correlations": correlation matrix
        "alerts"      : list of smart alert strings
    """
    audit = get_logger()

    result = {
        "shape": df.shape,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "alerts": [],
    }

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # ── Numeric summary ─────────────────────────────────────
    if num_cols:
        desc = df[num_cols].describe().round(4)
        desc.loc["skewness"] = df[num_cols].skew().round(4)
        desc.loc["kurtosis"] = df[num_cols].kurt().round(4)
        result["numeric"] = desc

    # ── Categorical summary ──────────────────────────────────
    cat_summary = {}
    for col in cat_cols:
        n_unique = df[col].nunique()
        top_vals = df[col].value_counts().head(5).to_dict()
        cat_summary[col] = {"n_unique": n_unique, "top_values": top_vals}
    result["categorical"] = cat_summary

    # ── Missing values ───────────────────────────────────────
    missing = {}
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            missing[col] = {
                "count": null_count,
                "pct": round(null_count / len(df) * 100, 2),
            }
    result["missing"] = missing

    # ── Correlations ─────────────────────────────────────────
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        result["correlations"] = corr
    else:
        result["correlations"] = pd.DataFrame()

    # ── SMART ALERTS ─────────────────────────────────────────
    alerts = []

    # Alert: zero-variance columns
    for col in num_cols:
        if df[col].nunique() <= 1:
            alerts.append(
                f"⚠️ ZERO VARIANCE: Column '{col}' has only 1 unique value. "
                f"It carries no information — consider dropping it."
            )

    # Alert: high cardinality categoricals
    for col, info in cat_summary.items():
        if info["n_unique"] > 50:
            alerts.append(
                f"⚠️ HIGH CARDINALITY: Column '{col}' has {info['n_unique']} unique values. "
                f"One-hot encoding would create {info['n_unique']} columns — "
                f"consider target encoding or embeddings instead."
            )

    # Alert: multicollinearity
    if len(num_cols) >= 2:
        corr_matrix = result["correlations"]
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_pairs = [
            (col_a, col_b, upper.loc[col_a, col_b])
            for col_a in upper.index
            for col_b in upper.columns
            if abs(upper.loc[col_a, col_b]) > correlation_threshold
        ]
        for col_a, col_b, r in high_pairs:
            alerts.append(
                f"⚠️ MULTICOLLINEARITY: '{col_a}' and '{col_b}' have correlation "
                f"r={r:.3f} (>{correlation_threshold}). Consider dropping one "
                f"for linear models."
            )

    # Alert: class imbalance
    if target and target in df.columns:
        counts = df[target].value_counts()
        minority_pct = counts.min() / counts.sum() * 100
        if minority_pct < 10:
            alerts.append(
                f"🔴 SEVERE CLASS IMBALANCE: Minority class in '{target}' is only "
                f"{minority_pct:.1f}%. Use SMOTE, class_weight='balanced', "
                f"and evaluate with F1/PR-AUC instead of accuracy."
            )
        elif minority_pct < 30:
            alerts.append(
                f"🟡 CLASS IMBALANCE: Minority class in '{target}' is "
                f"{minority_pct:.1f}%. Consider oversampling or class_weight='balanced'."
            )

    # Alert: highly skewed columns
    for col in num_cols:
        skew = df[col].skew()
        if abs(skew) > 2.0:
            direction = "right" if skew > 0 else "left"
            alerts.append(
                f"📐 SKEWED: '{col}' has skewness={skew:.2f} ({direction}-skewed). "
                f"Consider log-transform before regression models."
            )

    # Alert: high missing
    for col, info in missing.items():
        if info["pct"] > 30:
            alerts.append(
                f"🔴 HIGH MISSING: '{col}' has {info['pct']}% missing values. "
                f"Consider dropping this column or using KNN imputation."
            )

    result["alerts"] = alerts

    # ── Log to audit trail ───────────────────────────────────
    audit.log_decision(
        module="eda.profiler",
        action="profile",
        rationale=f"Generated data profile: {df.shape[0]} rows × {df.shape[1]} columns. "
                  f"{len(num_cols)} numeric, {len(cat_cols)} categorical columns. "
                  f"{len(missing)} columns with missing values. "
                  f"{len(alerts)} smart alerts raised.",
        details={
            "rows": df.shape[0],
            "columns": df.shape[1],
            "numeric_cols": len(num_cols),
            "categorical_cols": len(cat_cols),
            "columns_with_nulls": len(missing),
            "alerts_count": len(alerts),
        },
    )

    # Print alerts
    if alerts:
        print("\n" + "=" * 60)
        print("🔔 AUDITFLOW SMART ALERTS")
        print("=" * 60)
        for alert in alerts:
            print(f"  {alert}")
        print("=" * 60 + "\n")

    return result
