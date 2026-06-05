# ============================================================
# auditflow/features/temporal.py
# DateTime feature expansion with audit trail
# ============================================================

import numpy as np
import pandas as pd
from typing import Optional

from auditflow.core.logger import get_logger


def expand_datetime(
    df: pd.DataFrame,
    col: str,
    drop_original: bool = True,
    add_cyclical: bool = True,
) -> pd.DataFrame:
    """
    Decompose a datetime column into numeric calendar features.

    Generates: year, month, day, weekday, hour, quarter, is_weekend.
    Optionally adds cyclical sin/cos encoding so that Dec and Jan
    are numerically adjacent (critical for distance-based models).

    Parameters
    ----------
    col            : Datetime column name.
    drop_original  : Remove the original datetime column.
    add_cyclical   : Add sin/cos encoded features for month, hour, weekday.

    Returns
    -------
    pd.DataFrame
    """
    audit = get_logger()
    df = df.copy()
    dt = pd.to_datetime(df[col])

    new_cols = []

    df[f"{col}_year"]     = dt.dt.year;      new_cols.append(f"{col}_year")
    df[f"{col}_month"]    = dt.dt.month;     new_cols.append(f"{col}_month")
    df[f"{col}_day"]      = dt.dt.day;       new_cols.append(f"{col}_day")
    df[f"{col}_weekday"]  = dt.dt.weekday;   new_cols.append(f"{col}_weekday")
    df[f"{col}_hour"]     = dt.dt.hour;      new_cols.append(f"{col}_hour")
    df[f"{col}_quarter"]  = dt.dt.quarter;   new_cols.append(f"{col}_quarter")
    df[f"{col}_is_weekend"] = (dt.dt.weekday >= 5).astype(int)
    new_cols.append(f"{col}_is_weekend")

    if add_cyclical:
        df[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
        df[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
        df[f"{col}_hour_sin"]  = np.sin(2 * np.pi * dt.dt.hour / 24)
        df[f"{col}_hour_cos"]  = np.cos(2 * np.pi * dt.dt.hour / 24)
        df[f"{col}_dow_sin"]   = np.sin(2 * np.pi * dt.dt.weekday / 7)
        df[f"{col}_dow_cos"]   = np.cos(2 * np.pi * dt.dt.weekday / 7)
        new_cols.extend([
            f"{col}_month_sin", f"{col}_month_cos",
            f"{col}_hour_sin", f"{col}_hour_cos",
            f"{col}_dow_sin", f"{col}_dow_cos",
        ])

    if drop_original:
        df.drop(columns=[col], inplace=True)

    audit.log_decision(
        module="features.temporal",
        action="expand_datetime",
        column=col,
        rationale=f"Expanded datetime column '{col}' into {len(new_cols)} features. "
                  f"{'Cyclical sin/cos encoding added so Dec(12) and Jan(1) are adjacent.' if add_cyclical else ''} "
                  f"{'Original column dropped.' if drop_original else 'Original column kept.'}",
        details={
            "source_column": col,
            "new_features": new_cols,
            "cyclical": add_cyclical,
            "dropped_original": drop_original,
        },
    )

    return df
