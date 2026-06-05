# ============================================================
# tests/test_pipeline.py
# Integration test: full pipeline on synthetic data
# ============================================================

import numpy as np
import pandas as pd
import pytest
import os
from pathlib import Path

from auditflow.core.logger import AuditLogger, get_logger
from auditflow.core.config import PipelineConfig, DataConfig, ModelConfig, ReportConfig


@pytest.fixture(autouse=True)
def reset_logger():
    """Reset singleton between tests."""
    AuditLogger.reset_instance()
    yield
    AuditLogger.reset_instance()


@pytest.fixture
def synthetic_csv(tmp_path):
    """Create a synthetic CSV dataset for testing."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "feature_a": np.random.normal(50, 10, n),
        "feature_b": np.random.normal(100, 20, n),
        "category": np.random.choice(["A", "B", "C"], n),
        "target": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    })
    # Add some missing values
    df.loc[5:10, "feature_a"] = np.nan
    df.loc[15:18, "category"] = np.nan

    filepath = tmp_path / "test_data.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)


class TestPipelineConfig:
    """Tests for pipeline configuration."""

    def test_default_config(self):
        """Default config should have sensible defaults."""
        cfg = PipelineConfig()
        assert cfg.model.task == "classification"
        assert cfg.model.cv_folds == 5
        assert cfg.cleaning.missing["strategy"] == "auto"

    def test_from_yaml(self, tmp_path):
        """Should load config from YAML correctly."""
        yaml_content = """
data:
  source: "test.csv"
  format: csv
model:
  task: classification
  target: churn
  models: [rf, logistic]
"""
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml_content)

        cfg = PipelineConfig.from_yaml(str(yaml_path))
        assert cfg.data.source == "test.csv"
        assert cfg.model.target == "churn"
        assert "rf" in cfg.model.models

    def test_validate(self):
        """Validation should catch missing required fields."""
        cfg = PipelineConfig()
        errors = cfg.validate()
        assert len(errors) > 0  # source and target are missing

    def test_to_yaml(self, tmp_path):
        """Should export config to YAML."""
        cfg = PipelineConfig()
        cfg.data.source = "test.csv"
        path = tmp_path / "export.yaml"
        cfg.to_yaml(str(path))
        assert path.exists()


class TestEndToEndPipeline:
    """Integration test for the full pipeline."""

    def test_full_pipeline(self, synthetic_csv, tmp_path):
        """Full pipeline should run without errors and produce a report."""
        import matplotlib
        matplotlib.use("Agg")

        report_path = str(tmp_path / "test_report.html")

        config = PipelineConfig(
            data=DataConfig(source=synthetic_csv),
            model=ModelConfig(
                target="target",
                task="classification",
                models=["logistic", "rf"],
                cv_folds=3,
            ),
            report=ReportConfig(output=report_path),
        )

        from auditflow.pipeline.runner import Pipeline
        pipeline = Pipeline(config)
        result = pipeline.run()

        # Verify outputs
        assert result.df_raw is not None
        assert result.df_clean is not None
        assert result.comparison is not None
        assert len(result.model_results) == 2  # logistic + rf
        assert result.best_model is not None
        assert result.report_path is not None
        assert Path(result.report_path).exists()

        # Report should be a non-empty HTML file
        html = Path(result.report_path).read_text(encoding="utf-8")
        assert "<html" in html
        assert "AuditFlow" in html

        # Audit trail JSON should exist
        assert result.audit_json_path is not None
        assert Path(result.audit_json_path).exists()

    def test_audit_trail_populated(self, synthetic_csv, tmp_path):
        """Pipeline should log multiple decisions."""
        import matplotlib
        matplotlib.use("Agg")

        config = PipelineConfig(
            data=DataConfig(source=synthetic_csv),
            model=ModelConfig(
                target="target", task="classification",
                models=["rf"], cv_folds=3,
            ),
            report=ReportConfig(output=str(tmp_path / "report.html")),
        )

        from auditflow.pipeline.runner import Pipeline
        Pipeline(config).run()

        audit = get_logger()
        assert len(audit.events) > 10  # Should have many logged decisions
        assert len(audit.figures) > 0   # Should have generated charts
