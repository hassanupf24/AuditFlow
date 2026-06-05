# ============================================================
# auditflow/core/registry.py
# Transformer Registry — stores fitted transformers for reuse
# ============================================================

"""
When AuditFlow fits a scaler, encoder, or vectorizer on training data,
it stores the fitted object here. This enables:
  1. Applying the same transform to test/production data
  2. Saving/loading the entire pipeline state
  3. Auditing which transformers were used on which columns
"""

import json
import threading
import joblib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class TransformerRecord:
    """Metadata wrapper around a fitted transformer."""

    def __init__(
        self,
        name: str,
        transformer: Any,
        columns: List[str],
        module: str,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.transformer = transformer
        self.columns = columns
        self.module = module
        self.params = params or {}
        self.fitted_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Metadata dict (without the transformer object itself)."""
        return {
            "name": self.name,
            "type": type(self.transformer).__name__,
            "columns": self.columns,
            "module": self.module,
            "params": self.params,
            "fitted_at": self.fitted_at,
        }


class TransformerRegistry:
    """
    Central store for all fitted transformers in the pipeline.

    Usage:
        from auditflow.core.registry import TransformerRegistry

        registry = TransformerRegistry()
        registry.register("salary_scaler", scaler, columns=["salary", "age"],
                          module="features.numeric")
        scaler = registry.get("salary_scaler").transformer
    """

    def __init__(self) -> None:
        self._records: Dict[str, TransformerRecord] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        transformer: Any,
        columns: List[str],
        module: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a fitted transformer.

        Parameters
        ----------
        name        : Unique identifier (e.g. "salary_scaler").
        transformer : The fitted sklearn transformer object.
        columns     : Columns this transformer was fit on.
        module      : AuditFlow module that created it (e.g. "features.numeric").
        params      : Transformer hyperparameters for documentation.
        """
        with self._lock:
            self._records[name] = TransformerRecord(
                name=name,
                transformer=transformer,
                columns=columns,
                module=module,
                params=params,
            )

    def get(self, name: str) -> Optional[TransformerRecord]:
        """Retrieve a transformer record by name."""
        with self._lock:
            return self._records.get(name)

    def list_all(self) -> List[Dict[str, Any]]:
        """List[Any] metadata for all registered transformers."""
        with self._lock:
            return [r.to_dict() for r in self._records.values()]

    def save(self, directory: str) -> None:
        """
        Persist all transformers and metadata to disk.

        Creates:
          - {directory}/registry_metadata.json
          - {directory}/{name}.pkl for each transformer
        """
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata = []
        with self._lock:
            for name, record in self._records.items():
                pkl_path = out_dir / f"{name}.pkl"
                joblib.dump(record.transformer, pkl_path)
                meta = record.to_dict()
                meta["pkl_file"] = f"{name}.pkl"
                metadata.append(meta)

        meta_path = out_dir / "registry_metadata.json"
        meta_path.write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )

    def load(self, directory: str) -> None:
        """
        Load transformers and metadata from disk.

        Parameters
        ----------
        directory : Path to the directory saved by .save()
        """
        out_dir = Path(directory)
        meta_path = out_dir / "registry_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No registry found at {meta_path}")

        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        with self._lock:
            for meta in metadata:
                pkl_path = out_dir / meta["pkl_file"]
                transformer = joblib.load(pkl_path)
                self._records[meta["name"]] = TransformerRecord(
                    name=meta["name"],
                    transformer=transformer,
                    columns=meta["columns"],
                    module=meta["module"],
                    params=meta.get("params", {}),
                )

    def reset(self) -> None:
        """Clear all records."""
        with self._lock:
            self._records.clear()
