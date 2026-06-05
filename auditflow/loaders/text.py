# ============================================================
# auditflow/loaders/text.py
# Text corpus loader with audit trail
# ============================================================

import json
import pandas as pd
from pathlib import Path
from typing import Optional

from auditflow.core.logger import get_logger


def load_text_corpus(
    filepath: str,
    text_col: str = "text",
    label_col: Optional[str] = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    Load a text corpus from CSV, JSONL, or plain .txt files.

    Format detection:
      - .jsonl → one JSON object per line
      - .txt   → each line becomes one row in column 'text'
      - .csv   → standard pandas read

    Parameters
    ----------
    filepath  : Path to the text file.
    text_col  : Name of the text column (for CSV format).
    label_col : Name of the label column (optional).
    encoding  : File encoding.

    Returns
    -------
    pd.DataFrame
    """
    audit = get_logger()
    suffix = Path(filepath).suffix.lower()

    if suffix == ".jsonl":
        with open(filepath, encoding=encoding) as f:
            records = [json.loads(line) for line in f if line.strip()]
        df = pd.DataFrame(records)
        fmt = "JSONL"
    elif suffix == ".txt":
        with open(filepath, encoding=encoding) as f:
            lines = [line.strip() for line in f if line.strip()]
        df = pd.DataFrame({text_col: lines})
        fmt = "TXT"
    else:
        df = pd.read_csv(filepath, encoding=encoding)
        fmt = "CSV"

    audit.log_decision(
        module="loaders.text",
        action="load_text_corpus",
        rationale=f"Loaded text corpus from '{filepath}' (format: {fmt}). "
        f"{len(df)} samples, columns: {list(df.columns)}.",
        details={
            "filepath": filepath,
            "format_detected": fmt,
            "samples": len(df),
            "columns": list(df.columns),
        },
        after_shape=df.shape,
    )

    return df
