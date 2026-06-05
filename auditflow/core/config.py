# ============================================================
# auditflow/core/config.py
# YAML-based pipeline configuration
# ============================================================

"""
Defines the PipelineConfig dataclass that drives config-based pipelines.
Users write a YAML file; AuditFlow parses it into this structured config.

Example YAML:
    data:
      source: "data/sales.csv"
      format: "csv"

    cleaning:
      missing:
        strategy: "auto"
      outliers:
        method: "clip"
        multiplier: 1.5

    features:
      datetime_expand: ["order_date"]
      interactions: [["price", "quantity"]]

    model:
      task: "classification"
      target: "churn"
      models: ["rf", "gbm", "logistic"]
      cv_folds: 5

    report:
      output: "report.html"
      include_audit_trail: true
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Any, Dict, List, Optional, Tuple, Union

import yaml


@dataclass
class DataConfig:
    """Configuration for data loading."""

    source: str = ""
    format: str = "csv"  # csv, excel, parquet, api
    sheet_name: Union[str, int] = 0  # Excel sheet
    encoding: str = "utf-8"
    sep: str = ","
    date_columns: List[str] = field(default_factory=list)
    # API-specific
    api_headers: Dict[str, str] = field(default_factory=dict)
    api_data_key: Optional[str] = None


@dataclass
class CleaningConfig:
    """Configuration for data cleaning."""

    missing: Dict[str, Any] = field(
        default_factory=lambda: {
            "strategy": "auto",  # auto | mean | median | mode | knn | drop
            "knn_neighbors": 5,
        }
    )
    outliers: Dict[str, Any] = field(
        default_factory=lambda: {
            "method": "clip",  # clip | drop | flag | none
            "multiplier": 1.5,
            "columns": None,  # None = all numeric
        }
    )
    drop_high_null_cols: float = 0.5  # Drop columns with > this % null
    drop_high_null_rows: float = 0.7  # Drop rows with > this % null
    text_columns: List[str] = field(default_factory=list)
    text_options: Dict[str, bool] = field(
        default_factory=lambda: {
            "lowercase": True,
            "remove_html": True,
            "remove_urls": True,
            "remove_punctuation": True,
        }
    )


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    datetime_expand: List[str] = field(default_factory=list)
    cyclical_encoding: bool = True
    interactions: List[Tuple[str, str]] = field(default_factory=list)
    interaction_ops: List[str] = field(default_factory=lambda: ["multiply", "ratio"])
    polynomial_cols: List[str] = field(default_factory=list)
    polynomial_degree: int = 2
    scale_method: Optional[str] = "standard"  # standard | minmax | None
    scale_columns: Optional[List[str]] = None  # None = all numeric
    ordinal_cols: Dict[str, List[str]] = field(default_factory=dict)
    ohe_cols: List[str] = field(default_factory=list)
    label_encode_cols: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Configuration for model training."""

    task: str = "classification"  # classification | regression
    target: str = ""
    models: List[str] = field(default_factory=lambda: ["rf"])
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    model_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    output: str = "auditflow_report.html"
    title: str = "AuditFlow Analysis Report"
    include_audit_trail: bool = True
    include_figures: bool = True
    include_metrics: bool = True


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration.

    Can be loaded from a YAML file or constructed programmatically.
    """

    data: DataConfig = field(default_factory=DataConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    @classmethod
    def from_yaml(cls, filepath: str) -> "PipelineConfig":
        """
        Load configuration from a YAML file.

        Parameters
        ----------
        filepath : Path to the YAML config file.

        Returns
        -------
        PipelineConfig

        Raises
        ------
        FileNotFoundError : If the YAML file doesn't exist.
        ValueError        : If required fields are missing.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        config = cls()

        # Parse data section
        if "data" in raw:
            d = raw["data"]
            config.data = DataConfig(
                source=d.get("source", ""),
                format=d.get("format", "csv"),
                sheet_name=d.get("sheet_name", 0),
                encoding=d.get("encoding", "utf-8"),
                sep=d.get("sep", ","),
                date_columns=d.get("date_columns", []),
                api_headers=d.get("api_headers", {}),
                api_data_key=d.get("api_data_key"),
            )

        # Parse cleaning section
        if "cleaning" in raw:
            c = raw["cleaning"]
            config.cleaning = CleaningConfig(
                missing=c.get("missing", config.cleaning.missing),
                outliers=c.get("outliers", config.cleaning.outliers),
                drop_high_null_cols=c.get("drop_high_null_cols", 0.5),
                drop_high_null_rows=c.get("drop_high_null_rows", 0.7),
                text_columns=c.get("text_columns", []),
                text_options=c.get("text_options", config.cleaning.text_options),
            )

        # Parse features section
        if "features" in raw:
            fe = raw["features"]
            interactions_raw = fe.get("interactions", [])
            interactions = [tuple(pair) for pair in interactions_raw]
            config.features = FeatureConfig(
                datetime_expand=fe.get("datetime_expand", []),
                cyclical_encoding=fe.get("cyclical_encoding", True),
                interactions=interactions,
                interaction_ops=fe.get("interaction_ops", ["multiply", "ratio"]),
                polynomial_cols=fe.get("polynomial_cols", []),
                polynomial_degree=fe.get("polynomial_degree", 2),
                scale_method=fe.get("scale_method", "standard"),
                scale_columns=fe.get("scale_columns"),
                ordinal_cols=fe.get("ordinal_cols", {}),
                ohe_cols=fe.get("ohe_cols", []),
                label_encode_cols=fe.get("label_encode_cols", []),
            )

        # Parse model section
        if "model" in raw:
            m = raw["model"]
            config.model = ModelConfig(
                task=m.get("task", "classification"),
                target=m.get("target", ""),
                models=m.get("models", ["rf"]),
                cv_folds=m.get("cv_folds", 5),
                test_size=m.get("test_size", 0.2),
                random_state=m.get("random_state", 42),
                model_kwargs=m.get("model_kwargs", {}),
            )

        # Parse report section
        if "report" in raw:
            r = raw["report"]
            config.report = ReportConfig(
                output=r.get("output", "auditflow_report.html"),
                title=r.get("title", "AuditFlow Analysis Report"),
                include_audit_trail=r.get("include_audit_trail", True),
                include_figures=r.get("include_figures", True),
                include_metrics=r.get("include_metrics", True),
            )

        return config

    def to_yaml(self, filepath: str) -> None:
        """Export the current configuration to a YAML file."""
        import dataclasses

        data = dataclasses.asdict(self)
        Path(filepath).write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

    def validate(self) -> List[str]:
        """
        Validate the config and return a list of warnings/errors.

        Returns
        -------
        List[str] — empty if valid, otherwise contains error messages.
        """
        errors = []
        if not self.data.source:
            errors.append("data.source is required — specify a file path or URL.")
        if self.model.target == "":
            errors.append("model.target is required — specify the target column name.")
        if self.model.task not in ("classification", "regression"):
            errors.append(
                f"model.task must be 'classification' or 'regression', "
                f"got '{self.model.task}'."
            )
        return errors
