# ============================================================
# auditflow/cleaners/types.py
# Smart type inference & casting with audit trail
# ============================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from auditflow.core.logger import get_logger


def auto_cast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically infer and cast column types.

    Smart detection:
      - Numeric strings ("123", "45.6") → float64 / int64
      - Date-like strings ("2024-01-15", "Jan 15, 2024") → datetime64
      - Boolean-like columns (only True/False/0/1/yes/no) → bool
      - Low-cardinality object columns (< 20 unique / < 5% of rows) → category

    Every conversion is logged with rationale.

    Returns
    -------
    pd.DataFrame with optimized dtypes
    """
    audit = get_logger()
    df = df.copy()

    for col in df.columns:
        original_dtype = str(df[col].dtype)

        # Skip already-typed numeric/datetime columns
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        series = df[col].dropna()
        if len(series) == 0:
            continue

        # Try numeric conversion
        numeric_converted = pd.to_numeric(series, errors="coerce")
        if numeric_converted.notna().mean() > 0.9:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            new_dtype = str(df[col].dtype)
            audit.log_decision(
                module="cleaners.types",
                action="auto_cast_numeric",
                column=col,
                rationale=f"Column '{col}' contains >90% parseable numeric values. "
                          f"Cast from {original_dtype} → {new_dtype}.",
                details={"from": original_dtype, "to": new_dtype},
            )
            continue

        # Try datetime conversion
        try:
            date_sample = pd.to_datetime(series.head(100), errors="coerce", infer_datetime_format=True)
            if date_sample.notna().mean() > 0.8:
                df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                audit.log_decision(
                    module="cleaners.types",
                    action="auto_cast_datetime",
                    column=col,
                    rationale=f"Column '{col}' contains date-like strings. "
                              f"Cast from {original_dtype} → datetime64.",
                    details={"from": original_dtype, "to": "datetime64"},
                )
                continue
        except Exception:
            pass

        # Try boolean detection
        unique_lower = set(series.astype(str).str.strip().str.lower().unique())
        bool_values = {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}
        if unique_lower.issubset(bool_values) and len(unique_lower) <= 4:
            bool_map = {"true": True, "yes": True, "1": True, "t": True, "y": True,
                        "false": False, "no": False, "0": False, "f": False, "n": False}
            df[col] = df[col].astype(str).str.strip().str.lower().map(bool_map)
            audit.log_decision(
                module="cleaners.types",
                action="auto_cast_boolean",
                column=col,
                rationale=f"Column '{col}' contains only boolean-like values "
                          f"({unique_lower}). Cast to bool.",
                details={"from": original_dtype, "to": "bool", "unique_values": list(unique_lower)},
            )
            continue

        # Try category (low cardinality)
        n_unique = df[col].nunique()
        cardinality_ratio = n_unique / max(len(df), 1)
        if n_unique < 20 and cardinality_ratio < 0.05:
            df[col] = df[col].astype("category")
            audit.log_decision(
                module="cleaners.types",
                action="auto_cast_category",
                column=col,
                rationale=f"Column '{col}' has low cardinality ({n_unique} unique values, "
                          f"{cardinality_ratio:.1%} of rows). Cast to category for memory efficiency.",
                details={"from": original_dtype, "to": "category", "n_unique": n_unique},
            )

    return df


def cast_columns(
    df: pd.DataFrame,
    int_cols: Optional[List[str]] = None,
    float_cols: Optional[List[str]] = None,
    date_cols: Optional[List[str]] = None,
    date_format: Optional[str] = None,
    bool_cols: Optional[List[str]] = None,
    category_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Explicitly cast columns to target dtypes with audit trail.

    Parameters
    ----------
    date_format : strftime format string (e.g. '%Y-%m-%d'). None = infer.

    Returns
    -------
    pd.DataFrame
    """
    audit = get_logger()
    df = df.copy()

    casts = [
        (int_cols, "Int64", "int"),
        (float_cols, "float64", "float"),
        (bool_cols, "bool", "bool"),
        (category_cols, "category", "category"),
    ]

    for cols, target_dtype, label in casts:
        for col in (cols or []):
            if col not in df.columns:
                continue
            original = str(df[col].dtype)
            try:
                if label == "int":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                elif label == "float":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    df[col] = df[col].astype(target_dtype)
                audit.log_decision(
                    module="cleaners.types",
                    action=f"cast_{label}",
                    column=col,
                    rationale=f"Explicitly cast '{col}' from {original} → {target_dtype}.",
                    details={"from": original, "to": target_dtype},
                )
            except (ValueError, TypeError) as e:
                audit.log_decision(
                    module="cleaners.types",
                    action="cast_failed",
                    column=col,
                    rationale=f"Failed to cast '{col}' to {target_dtype}: {e}",
                    details={"from": original, "to": target_dtype, "error": str(e)},
                )

    for col in (date_cols or []):
        if col not in df.columns:
            continue
        original = str(df[col].dtype)
        df[col] = pd.to_datetime(df[col], format=date_format, errors="coerce")
        audit.log_decision(
            module="cleaners.types",
            action="cast_datetime",
            column=col,
            rationale=f"Parsed '{col}' as datetime "
                      f"(format={'inferred' if not date_format else date_format}).",
            details={"from": original, "to": "datetime64", "format": date_format or "inferred"},
        )

    return df
