# ============================================================
# auditflow/loaders/tabular.py
# CSV, Excel, Parquet loaders — all auto-logged
# ============================================================

import pandas as pd
from typing import List, Optional, Union

from auditflow.core.logger import get_logger


def load_csv(
    filepath: str,
    sep: str = ",",
    encoding: str = "utf-8",
    low_memory: bool = False,
    parse_dates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load a CSV / TSV file into a DataFrame.

    All metadata is automatically logged to the audit trail:
    file path, shape, dtypes, and date columns parsed.

    Parameters
    ----------
    filepath    : Path to the CSV / TSV file.
    sep         : Delimiter character. Use '\\t' for TSV files.
    encoding    : File encoding (try 'latin-1' for legacy files).
    low_memory  : Set True for very large files (>500MB).
    parse_dates : Column names to parse as datetime.

    Returns
    -------
    pd.DataFrame
    """
    audit = get_logger()

    df = pd.read_csv(
        filepath,
        sep=sep,
        encoding=encoding,
        low_memory=low_memory,
        parse_dates=parse_dates or [],
    )

    audit.log_decision(
        module="loaders.tabular",
        action="load_csv",
        rationale=f"Loaded CSV from '{filepath}'. "
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns. "
        f"Columns: {list(df.columns)}.",
        details={
            "filepath": filepath,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "separator": sep,
            "encoding": encoding,
            "date_columns_parsed": parse_dates or [],
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        },
        after_shape=df.shape,
    )

    return df


def load_excel(
    filepath: str,
    sheet_name: Union[str, int] = 0,
    header: int = 0,
) -> pd.DataFrame:
    """
    Load a single sheet from an Excel workbook (.xlsx / .xls).

    Parameters
    ----------
    sheet_name : Sheet name (str) or index (int, 0-based).
    header     : Row index to use as column names.

    Returns
    -------
    pd.DataFrame
    """
    audit = get_logger()

    df = pd.read_excel(filepath, sheet_name=sheet_name, header=header)

    audit.log_decision(
        module="loaders.tabular",
        action="load_excel",
        rationale=f"Loaded Excel sheet '{sheet_name}' from '{filepath}'. "
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns.",
        details={
            "filepath": filepath,
            "sheet_name": str(sheet_name),
            "rows": df.shape[0],
            "columns": df.shape[1],
        },
        after_shape=df.shape,
    )

    return df


def load_parquet(filepath: str) -> pd.DataFrame:
    """
    Load a Parquet file into a DataFrame.

    Parquet is a columnar format that's ~10x faster to read than CSV
    and preserves dtypes perfectly. Preferred for large datasets.

    Parameters
    ----------
    filepath : Path to the .parquet file.

    Returns
    -------
    pd.DataFrame
    """
    audit = get_logger()

    df = pd.read_parquet(filepath)

    audit.log_decision(
        module="loaders.tabular",
        action="load_parquet",
        rationale=f"Loaded Parquet from '{filepath}'. "
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns.",
        details={
            "filepath": filepath,
            "rows": df.shape[0],
            "columns": df.shape[1],
        },
        after_shape=df.shape,
    )

    return df
