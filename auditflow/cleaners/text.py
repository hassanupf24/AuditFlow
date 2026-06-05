# ============================================================
# auditflow/cleaners/text.py
# Text normalization pipeline with audit trail
# ============================================================

import re
import pandas as pd
from typing import Dict, List, Optional

from auditflow.core.logger import get_logger


def clean_text(
    df: pd.DataFrame,
    columns: List[str],
    lowercase: bool = True,
    remove_html: bool = True,
    remove_urls: bool = True,
    remove_punctuation: bool = True,
    remove_digits: bool = False,
    strip_whitespace: bool = True,
) -> pd.DataFrame:
    """
    Apply a configurable NLP pre-cleaning pipeline to text columns.

    Each step is individually configurable and logged. This does NOT
    tokenize or stem — that's handled in the features module.

    Parameters
    ----------
    columns            : Text columns to clean.
    lowercase          : Convert to lowercase.
    remove_html        : Strip HTML tags.
    remove_urls        : Remove HTTP/HTTPS URLs.
    remove_punctuation : Remove all non-word, non-space characters.
    remove_digits      : Remove all digit characters.
    strip_whitespace   : Collapse multiple spaces into one.

    Returns
    -------
    pd.DataFrame
    """
    audit = get_logger()
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        original_sample = df[col].dropna().head(3).tolist()
        steps_applied = []

        s = df[col].astype(str)

        if remove_html:
            s = s.str.replace(r"<[^>]+>", " ", regex=True)
            steps_applied.append("remove_html")

        if remove_urls:
            s = s.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
            steps_applied.append("remove_urls")

        if lowercase:
            s = s.str.lower()
            steps_applied.append("lowercase")

        if remove_punctuation:
            s = s.str.replace(r"[^\w\s]", " ", regex=True)
            steps_applied.append("remove_punctuation")

        if remove_digits:
            s = s.str.replace(r"\d+", " ", regex=True)
            steps_applied.append("remove_digits")

        if strip_whitespace:
            s = s.str.replace(r"\s+", " ", regex=True).str.strip()
            steps_applied.append("strip_whitespace")

        df[col] = s
        cleaned_sample = df[col].dropna().head(3).tolist()

        audit.log_decision(
            module="cleaners.text",
            action="clean_text",
            column=col,
            rationale=f"Applied {len(steps_applied)} text cleaning steps to '{col}': "
                      f"{', '.join(steps_applied)}.",
            details={
                "steps": steps_applied,
                "sample_before": original_sample[:2],
                "sample_after": cleaned_sample[:2],
            },
        )

    return df
