# ============================================================
# auditflow/api/schemas.py
# Pydantic schemas for strict runtime validation
# ============================================================

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple, Union


class DataConfigSchema(BaseModel):
    source: str = Field(..., description="Data source path or URL")
    format: str = Field(default="csv")
    sheet_name: Union[str, int] = Field(default=0)
    encoding: str = Field(default="utf-8")
    sep: str = Field(default=",")
    date_columns: List[str] = Field(default_factory=list)
    api_headers: Dict[str, str] = Field(default_factory=dict)
    api_data_key: Optional[str] = None


class CleaningConfigSchema(BaseModel):
    missing: Dict[str, Any] = Field(
        default_factory=lambda: {"strategy": "auto", "knn_neighbors": 5}
    )
    outliers: Dict[str, Any] = Field(
        default_factory=lambda: {"method": "clip", "multiplier": 1.5, "columns": None}
    )
    drop_high_null_cols: float = Field(default=0.5)
    drop_high_null_rows: float = Field(default=0.7)
    text_columns: List[str] = Field(default_factory=list)
    text_options: Dict[str, bool] = Field(
        default_factory=lambda: {
            "lowercase": True,
            "remove_html": True,
            "remove_urls": True,
            "remove_punctuation": True,
        }
    )


class FeatureConfigSchema(BaseModel):
    datetime_expand: List[str] = Field(default_factory=list)
    cyclical_encoding: bool = Field(default=True)
    interactions: List[Tuple[str, str]] = Field(default_factory=list)
    interaction_ops: List[str] = Field(default_factory=lambda: ["multiply", "ratio"])
    polynomial_cols: List[str] = Field(default_factory=list)
    polynomial_degree: int = Field(default=2)
    scale_method: Optional[str] = Field(default="standard")
    scale_columns: Optional[List[str]] = None
    ordinal_cols: Dict[str, List[str]] = Field(default_factory=dict)
    ohe_cols: List[str] = Field(default_factory=list)
    label_encode_cols: List[str] = Field(default_factory=list)


class ModelConfigSchema(BaseModel):
    task: str = Field(default="classification")
    target: str = Field(..., description="Target variable to predict")
    models: List[str] = Field(default_factory=lambda: ["rf"])
    cv_folds: int = Field(default=5)
    test_size: float = Field(default=0.2)
    random_state: int = Field(default=42)
    model_kwargs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class ReportConfigSchema(BaseModel):
    output: str = Field(default="auditflow_report.html")
    title: str = Field(default="AuditFlow Analysis Report")
    include_audit_trail: bool = Field(default=True)
    include_figures: bool = Field(default=True)
    include_metrics: bool = Field(default=True)


class PipelineRunRequest(BaseModel):
    data: DataConfigSchema
    cleaning: CleaningConfigSchema = Field(default_factory=CleaningConfigSchema)
    features: FeatureConfigSchema = Field(default_factory=FeatureConfigSchema)
    model: ModelConfigSchema
    report: ReportConfigSchema = Field(default_factory=ReportConfigSchema)
