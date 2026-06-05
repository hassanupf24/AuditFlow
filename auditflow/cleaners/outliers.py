# ============================================================
# auditflow/cleaners/outliers.py
# Outlier detection & handling with audit trail
# ============================================================

import numpy as np
import pandas as pd
from typing import Any, Tuple, Optional, Dict, List, Optional

from auditflow.core.logger import get_logger


def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Detect outliers and return a diagnostic report.

    Parameters
    ----------
    method         : 'iqr' (Interquartile Range) or 'zscore'.
    iqr_multiplier : Fence multiplier for IQR method (default 1.5).
    z_threshold    : Z-score threshold (default 3.0 = ~0.3% of normal dist).

    Returns
    -------
    dict — {column_name: {"count": N, "pct": X, "lower": L, "upper": U}}
    """
    audit = get_logger()
    target_cols = columns or df.select_dtypes(include=np.number).columns.tolist()
    report = {}

    for col in target_cols:
        series = df[col].dropna()

        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
        else:  # zscore
            mean, std = series.mean(), series.std()
            lower = mean - z_threshold * std
            upper = mean + z_threshold * std

        outlier_mask = (series < lower) | (series > upper)
        count = outlier_mask.sum()

        report[col] = {
            "count": int(count),
            "pct": round(count / len(series) * 100, 2),
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
        }

    total_outliers = sum(r["count"] for r in report.values())
    audit.log_decision(
        module="cleaners.outliers",
        action="detect_outliers",
        rationale=f"Scanned {len(target_cols)} numeric columns for outliers using "
        f"'{method}' method. Found {total_outliers} total outlier values.",
        details={
            "method": method,
            "columns_scanned": len(target_cols),
            "total_outliers": total_outliers,
            "per_column": report,
        },
    )

    return report


def handle_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "clip",
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Handle outliers using IQR fences.

    Parameters
    ----------
    method         : How to handle detected outliers:
        'clip'  — Clamp values to fence boundaries (recommended; preserves rows)
        'drop'  — Remove rows containing outliers
        'flag'  — Add boolean '<col>_is_outlier' columns
    iqr_multiplier : 1.5 = standard, 3.0 = conservative.

    Returns
    -------
    pd.DataFrame
    """
    audit = get_logger()
    df = df.copy()
    before_shape = df.shape
    target_cols = columns or df.select_dtypes(include=np.number).columns.tolist()

    total_affected = 0
    for col in target_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
        total_affected += outlier_count

        if method == "clip":
            df[col] = df[col].clip(lower, upper)
        elif method == "drop":
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        elif method == "flag":
            df[f"{col}_is_outlier"] = ~df[col].between(lower, upper)

        if outlier_count > 0:
            if method == "clip":
                rationale = (
                    f"Capped {outlier_count} outlier values to [{lower:.2f}, {upper:.2f}]. "
                    f"Clipping preserves all rows while reducing extreme value distortion."
                )
            elif method == "drop":
                rationale = (
                    f"Removed rows with {outlier_count} outlier values "
                    f"outside [{lower:.2f}, {upper:.2f}]."
                )
            else:
                rationale = (
                    f"Flagged {outlier_count} outlier values "
                    f"outside [{lower:.2f}, {upper:.2f}]."
                )

            audit.log_decision(
                module="cleaners.outliers",
                action=f"outlier_{method}",
                column=col,
                rationale=rationale,
                details={
                    "method": method,
                    "outlier_count": int(outlier_count),
                    "lower_bound": round(float(lower), 4),
                    "upper_bound": round(float(upper), 4),
                    "iqr_multiplier": iqr_multiplier,
                },
                before_shape=before_shape,
                after_shape=df.shape,
            )

    if total_affected == 0:
        audit.log_decision(
            module="cleaners.outliers",
            action="outlier_none",
            rationale=f"No outliers detected in {len(target_cols)} columns "
            f"using IQR method (multiplier={iqr_multiplier}).",
            before_shape=before_shape,
            after_shape=df.shape,
        )

    return df
