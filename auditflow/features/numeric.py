# ============================================================
# auditflow/features/numeric.py
# Numeric feature engineering with audit trail
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, PolynomialFeatures, LabelEncoder,
)
from typing import Dict, List, Optional, Tuple

from auditflow.core.logger import get_logger
from auditflow.core.registry import TransformerRegistry


def scale(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "standard",
    registry: Optional[TransformerRegistry] = None,
) -> pd.DataFrame:
    """
    Scale numeric columns with audit trail.

    Parameters
    ----------
    columns  : Columns to scale. None = all numeric.
    method   : 'standard' (z-score, mean=0 std=1) or 'minmax' ([0,1]).
    registry : If provided, the fitted scaler is stored for later reuse.

    Returns
    -------
    pd.DataFrame
    """
    audit = get_logger()
    df = df.copy()
    target_cols = columns or df.select_dtypes(include=np.number).columns.tolist()

    if not target_cols:
        return df

    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df[target_cols] = scaler.fit_transform(df[target_cols])

    if registry:
        registry.register(
            name=f"scaler_{method}",
            transformer=scaler,
            columns=target_cols,
            module="features.numeric",
            params={"method": method},
        )

    audit.log_decision(
        module="features.numeric",
        action=f"scale_{method}",
        rationale=f"Applied {method} scaling to {len(target_cols)} columns. "
                  f"{'Z-score normalization (mean=0, std=1)' if method == 'standard' else 'Min-max rescaling to [0, 1]'}. "
                  f"Scaling ensures features contribute equally to distance-based algorithms.",
        details={"method": method, "columns": target_cols},
    )

    return df


def add_interactions(
    df: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    operations: List[str] = None,
) -> pd.DataFrame:
    """
    Create pairwise interaction features with audit trail.

    Parameters
    ----------
    pairs      : List of (col_a, col_b) tuples.
    operations : Subset of ["multiply", "ratio", "add", "subtract"].
                 Default: ["multiply", "ratio"].
    """
    audit = get_logger()
    df = df.copy()
    ops = operations or ["multiply", "ratio"]
    new_cols = []

    for a, b in pairs:
        if "multiply" in ops:
            name = f"{a}_x_{b}"
            df[name] = df[a] * df[b]
            new_cols.append(name)
        if "ratio" in ops:
            name = f"{a}_ratio_{b}"
            df[name] = df[a] / (df[b] + 1e-9)
            new_cols.append(name)
        if "add" in ops:
            name = f"{a}_plus_{b}"
            df[name] = df[a] + df[b]
            new_cols.append(name)
        if "subtract" in ops:
            name = f"{a}_minus_{b}"
            df[name] = df[a] - df[b]
            new_cols.append(name)

    audit.log_decision(
        module="features.numeric",
        action="add_interactions",
        rationale=f"Created {len(new_cols)} interaction features from {len(pairs)} pairs. "
                  f"Operations: {ops}. Interaction features help linear models capture "
                  f"nonlinear relationships between variables.",
        details={"pairs": [list(p) for p in pairs], "operations": ops, "new_columns": new_cols},
    )

    return df


def add_polynomial(
    df: pd.DataFrame,
    cols: List[str],
    degree: int = 2,
    scale_first: bool = True,
) -> pd.DataFrame:
    """
    Generate polynomial features up to `degree` with audit trail.

    Parameters
    ----------
    degree      : Polynomial degree (2 = x², x·y for each pair).
    scale_first : Standardize before generating polynomials to avoid overflow.
    """
    audit = get_logger()
    df = df.copy()
    X = df[cols].values

    if scale_first:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(cols)

    # Only keep new features (not the originals)
    new_names = [n for n in feature_names if n not in cols]
    new_indices = [list(feature_names).index(n) for n in new_names]
    poly_df = pd.DataFrame(
        X_poly[:, new_indices], columns=new_names, index=df.index
    )
    df = pd.concat([df, poly_df], axis=1)

    audit.log_decision(
        module="features.numeric",
        action="add_polynomial",
        rationale=f"Generated {len(new_names)} polynomial features (degree={degree}) "
                  f"from {len(cols)} input columns. "
                  f"{'Inputs were standardized first to prevent overflow.' if scale_first else ''}",
        details={
            "input_columns": cols,
            "degree": degree,
            "new_features_count": len(new_names),
            "scaled_first": scale_first,
        },
    )

    return df


def encode_categoricals(
    df: pd.DataFrame,
    ordinal_cols: Optional[Dict[str, List[str]]] = None,
    ohe_cols: Optional[List[str]] = None,
    label_encode_cols: Optional[List[str]] = None,
    drop_first: bool = True,
) -> pd.DataFrame:
    """
    Encode categorical variables with audit trail.

    Parameters
    ----------
    ordinal_cols      : {col: [ordered categories from low to high]}
    ohe_cols          : Columns for one-hot encoding.
    label_encode_cols : Columns for integer label encoding.
    drop_first        : Drop first dummy to avoid multicollinearity.
    """
    audit = get_logger()
    df = df.copy()

    # Ordinal encoding
    for col, order in (ordinal_cols or {}).items():
        cat = pd.CategoricalDtype(categories=order, ordered=True)
        df[col] = df[col].astype(cat).cat.codes
        audit.log_decision(
            module="features.numeric",
            action="encode_ordinal",
            column=col,
            rationale=f"Ordinal encoded '{col}' using order: {order}. "
                      f"Preserves the natural ranking between categories.",
            details={"column": col, "order": order},
        )

    # One-hot encoding
    if ohe_cols:
        before_cols = set(df.columns)
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=drop_first, dtype=int)
        new_cols = sorted(set(df.columns) - before_cols)
        audit.log_decision(
            module="features.numeric",
            action="encode_onehot",
            rationale=f"One-hot encoded {len(ohe_cols)} columns, creating {len(new_cols)} "
                      f"binary features. drop_first={drop_first} to avoid the dummy variable trap.",
            details={"source_columns": ohe_cols, "new_columns": new_cols, "drop_first": drop_first},
        )

    # Label encoding
    le = LabelEncoder()
    for col in (label_encode_cols or []):
        df[col] = le.fit_transform(df[col].astype(str))
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        audit.log_decision(
            module="features.numeric",
            action="encode_label",
            column=col,
            rationale=f"Label encoded '{col}' to integers. "
                      f"Mapping: {mapping}.",
            details={"column": col, "mapping": {str(k): int(v) for k, v in mapping.items()}},
        )

    return df
