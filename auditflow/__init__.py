# ============================================================
# AuditFlow — A Self-Documenting Data Analysis Framework
# ============================================================

"""
AuditFlow: Every pipeline step logs why, tracks what changed,
and auto-generates professional reports.

Quick Start:
    from auditflow import Pipeline
    result = Pipeline.from_yaml("config.yaml").run()

    # Or step by step:
    from auditflow.core import AuditLogger
    from auditflow.loaders import load_csv
    from auditflow.cleaners import impute, handle_outliers

    audit = AuditLogger()
    df = load_csv("data.csv")
    df = impute(df, strategy="auto")
    audit.generate_report("report.html")
"""

__version__ = "0.1.0"
__author__ = "Hassan Gasim"

from auditflow.core.logger import AuditLogger, get_logger
from auditflow.core.config import PipelineConfig
from auditflow.pipeline.runner import Pipeline

__all__ = [
    "AuditLogger",
    "get_logger",
    "PipelineConfig",
    "Pipeline",
    "__version__",
]
