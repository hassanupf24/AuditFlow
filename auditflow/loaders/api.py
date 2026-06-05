# ============================================================
# auditflow/loaders/api.py
# REST API loader with audit trail
# ============================================================

import requests
import pandas as pd
from typing import Any, Tuple, Optional, Dict, Optional

from auditflow.core.logger import get_logger


def load_from_api(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
    auth: Optional[Tuple[Any, ...]] = None,
    data_key: Optional[str] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch JSON from a REST API and normalize into a DataFrame.

    Parameters
    ----------
    url       : Full endpoint URL.
    params    : Query-string parameters.
    headers   : HTTP headers (e.g. Authorization).
    auth      : Tuple (username, password) for HTTP Basic Auth.
    data_key  : Key to extract if JSON is nested (e.g. "results").
    timeout   : Request timeout in seconds.

    Returns
    -------
    pd.DataFrame
    """
    audit = get_logger()

    response = requests.get(
        url,
        params=params,
        headers=headers,
        auth=auth,
        timeout=timeout,
    )
    response.raise_for_status()
    raw = response.json()

    records = raw[data_key] if data_key else raw
    df = pd.json_normalize(records)

    audit.log_decision(
        module="loaders.api",
        action="load_from_api",
        rationale=f"Fetched data from API: {url}. "
        f"Response status: {response.status_code}. "
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns.",
        details={
            "url": url,
            "status_code": response.status_code,
            "data_key": data_key,
            "rows": df.shape[0],
            "columns": df.shape[1],
        },
        after_shape=df.shape,
    )

    return df
