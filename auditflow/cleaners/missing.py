# ============================================================
# auditflow/cleaners/missing.py
# Smart missing value handling with auto-strategy selection
# ============================================================

"""
The Smart Defaults Engine for missing values:
  - Profiles each column's distribution (skewness, % missing, correlations)
  - Selects the optimal imputation strategy per column automatically
  - Logs the rationale for every choice

Strategy selection logic (when strategy="auto"):
  - < 5% missing AND column is not critical → drop rows
  - Numeric + skewness > 1.0 → median (robust to skew)
  - Numeric + skewness <= 1.0 → mean
  - Categorical → mode (most frequent)
  - > 40% missing → drop column entirely
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Dict, List, Optional

from auditflow.core.logger import get_logger


def audit_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a diagnostic report of missing values per column.

    Returns a DataFrame with: null_count, null_pct, dtype, recommended_strategy.

    The recommended_strategy column contains AuditFlow's Smart Default
    suggestion — what it would do if you set strategy="auto".
    """
    audit = get_logger()

    report = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "null_count": df.isnull().sum(),
        "null_pct": (df.isnull().mean() * 100).round(2),
    })

    # Add smart recommendations
    recommendations = []
    for col in df.columns:
        null_pct = report.loc[col, "null_pct"]
        if null_pct == 0:
            recommendations.append("no_action")
        elif null_pct > 40:
            recommendations.append("drop_column")
        elif pd.api.types.is_numeric_dtype(df[col]):
            skew = df[col].skew()
            if abs(skew) > 1.0:
                recommendations.append(f"median (skewness={skew:.2f})")
            else:
                recommendations.append(f"mean (skewness={skew:.2f})")
        else:
            recommendations.append("mode")

    report["recommended_strategy"] = recommendations
    report = report[report["null_count"] > 0].sort_values("null_pct", ascending=False)

    audit.log_decision(
        module="cleaners.missing",
        action="audit_missing",
        rationale=f"Profiled missing values: {len(report)} columns have nulls out of "
                  f"{len(df.columns)} total. "
                  f"Highest: {report.index[0]} ({report.iloc[0]['null_pct']}%)"
                  if len(report) > 0 else "No missing values found.",
        details={
            "columns_with_nulls": len(report),
            "total_columns": len(df.columns),
            "total_null_cells": int(df.isnull().sum().sum()),
        },
    )

    return report


def _select_auto_strategy(series: pd.Series, null_pct: float) -> tuple:
    """
    Smart Default Engine: determine the best imputation strategy for a column.

    Returns (strategy_name: str, rationale: str)
    """
    if null_pct > 40:
        return "drop_column", (
            f"Column has {null_pct:.1f}% missing values (>40%). "
            f"Imputation would fabricate too much data — dropping column."
        )

    if pd.api.types.is_numeric_dtype(series):
        skew = series.skew()
        if null_pct < 5:
            return "median", (
                f"Only {null_pct:.1f}% missing. Using median "
                f"(skewness={skew:.2f}) as a safe default for low-null numeric columns."
            )
        if abs(skew) > 1.0:
            return "median", (
                f"Column is {'right' if skew > 0 else 'left'}-skewed "
                f"(skewness={skew:.2f}). Median is more robust than mean "
                f"for skewed distributions."
            )
        else:
            return "mean", (
                f"Column is approximately symmetric (skewness={skew:.2f}). "
                f"Mean is appropriate for normally-distributed data."
            )
    else:
        return "most_frequent", (
            f"Categorical column with {null_pct:.1f}% missing. "
            f"Filling with the most frequent value (mode)."
        )


def impute(
    df: pd.DataFrame,
    strategy: str = "auto",
    columns: Optional[List[str]] = None,
    knn_neighbors: int = 5,
) -> pd.DataFrame:
    """
    Impute missing values with full audit trail.

    Parameters
    ----------
    strategy : Imputation strategy:
        'auto'           — Smart Default: auto-selects per column (recommended)
        'mean'           — Fill with column mean (numeric only)
        'median'         — Fill with column median (numeric only)
        'most_frequent'  — Fill with mode
        'knn'            — K-Nearest Neighbor imputation
        'drop'           — Drop rows with any null in target columns
    columns  : Columns to impute. None = all columns with nulls.
    knn_neighbors : K for KNN imputation.

    Returns
    -------
    pd.DataFrame — with missing values handled
    """
    audit = get_logger()
    df = df.copy()
    before_shape = df.shape
    target_cols = columns or df.columns[df.isnull().any()].tolist()

    if not target_cols:
        audit.log_decision(
            module="cleaners.missing",
            action="impute_skip",
            rationale="No missing values found. Skipping imputation.",
            before_shape=before_shape,
            after_shape=df.shape,
        )
        return df

    if strategy == "drop":
        before_rows = len(df)
        df.dropna(subset=target_cols, inplace=True)
        dropped = before_rows - len(df)
        audit.log_decision(
            module="cleaners.missing",
            action="impute_drop",
            rationale=f"Dropped {dropped} rows containing NaN in columns: {target_cols}. "
                      f"Remaining: {len(df)} rows.",
            details={"dropped_rows": dropped, "columns": target_cols},
            before_shape=before_shape,
            after_shape=df.shape,
        )
        return df

    if strategy == "knn":
        num_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            imp = KNNImputer(n_neighbors=knn_neighbors)
            df[num_cols] = imp.fit_transform(df[num_cols])
            audit.log_decision(
                module="cleaners.missing",
                action="impute_knn",
                rationale=f"Applied KNN imputation (k={knn_neighbors}) to {len(num_cols)} "
                          f"numeric columns. KNN uses similar rows to estimate missing values, "
                          f"preserving multivariate relationships.",
                details={"k": knn_neighbors, "columns": num_cols},
                before_shape=before_shape,
                after_shape=df.shape,
            )
        return df

    if strategy == "auto":
        # Smart Default Engine: per-column strategy selection
        cols_to_drop = []
        for col in target_cols:
            null_pct = df[col].isnull().mean() * 100
            if null_pct == 0:
                continue

            auto_strategy, rationale = _select_auto_strategy(df[col], null_pct)

            if auto_strategy == "drop_column":
                cols_to_drop.append(col)
                audit.log_decision(
                    module="cleaners.missing",
                    action="drop_column",
                    column=col,
                    rationale=rationale,
                    details={"null_pct": round(null_pct, 2)},
                    before_shape=df.shape,
                )
            elif auto_strategy in ("mean", "median"):
                fill_val = df[col].mean() if auto_strategy == "mean" else df[col].median()
                df[col] = df[col].fillna(fill_val)
                audit.log_decision(
                    module="cleaners.missing",
                    action=f"impute_{auto_strategy}",
                    column=col,
                    rationale=rationale,
                    details={
                        "strategy": auto_strategy,
                        "fill_value": round(float(fill_val), 4),
                        "null_pct": round(null_pct, 2),
                    },
                    before_shape=before_shape,
                    after_shape=df.shape,
                )
            elif auto_strategy == "most_frequent":
                fill_val = df[col].mode()[0]
                df[col] = df[col].fillna(fill_val)
                audit.log_decision(
                    module="cleaners.missing",
                    action="impute_mode",
                    column=col,
                    rationale=rationale,
                    details={
                        "strategy": "most_frequent",
                        "fill_value": str(fill_val),
                        "null_pct": round(null_pct, 2),
                    },
                    before_shape=before_shape,
                    after_shape=df.shape,
                )

        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

        return df

    # Fixed strategy (mean, median, most_frequent)
    num_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in target_cols if not pd.api.types.is_numeric_dtype(df[c])]

    if num_cols and strategy in ("mean", "median", "most_frequent"):
        imp = SimpleImputer(strategy=strategy)
        df[num_cols] = imp.fit_transform(df[num_cols])
        audit.log_decision(
            module="cleaners.missing",
            action=f"impute_{strategy}",
            rationale=f"Imputed {len(num_cols)} numeric columns using '{strategy}' strategy.",
            details={"strategy": strategy, "columns": num_cols},
            before_shape=before_shape,
            after_shape=df.shape,
        )

    if cat_cols:
        imp_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imp_cat.fit_transform(df[cat_cols])
        audit.log_decision(
            module="cleaners.missing",
            action="impute_mode",
            rationale=f"Imputed {len(cat_cols)} categorical columns using mode (most frequent).",
            details={"strategy": "most_frequent", "columns": cat_cols},
            before_shape=before_shape,
            after_shape=df.shape,
        )

    return df
