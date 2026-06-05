# ============================================================
# auditflow/pipeline/runner.py
# Config-driven pipeline orchestration
# ============================================================

"""
The Pipeline class reads a YAML config and executes all steps:
  load → clean → profile → engineer features → split → train → evaluate → report

Each step is optional and fully audited.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from sklearn.model_selection import train_test_split

from auditflow.core.logger import AuditLogger, get_logger
from auditflow.core.config import PipelineConfig
from auditflow.core.registry import TransformerRegistry


@dataclass
class PipelineResult:
    """Container for all pipeline outputs."""
    df_raw: Optional[pd.DataFrame] = None
    df_clean: Optional[pd.DataFrame] = None
    df_featured: Optional[pd.DataFrame] = None
    profile: Optional[Dict] = None
    model_results: Optional[List] = field(default_factory=list)
    comparison: Optional[pd.DataFrame] = None
    best_model: Optional[Any] = None
    report_path: Optional[str] = None
    audit_json_path: Optional[str] = None


class Pipeline:
    """
    Config-driven analysis pipeline with full audit trail.

    Usage (from YAML):
        pipeline = Pipeline.from_yaml("config.yaml")
        result = pipeline.run()

    Usage (programmatic):
        from auditflow.core.config import PipelineConfig, DataConfig, ModelConfig
        config = PipelineConfig(
            data=DataConfig(source="data.csv"),
            model=ModelConfig(target="churn", models=["rf", "gbm"]),
        )
        result = Pipeline(config).run()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.audit = get_logger()
        self.registry = TransformerRegistry()

    @classmethod
    def from_yaml(cls, filepath: str) -> "Pipeline":
        """Create a Pipeline from a YAML config file."""
        config = PipelineConfig.from_yaml(filepath)
        return cls(config)

    def run(self) -> PipelineResult:
        """
        Execute the full pipeline.

        Steps (all optional based on config):
          1. Load data
          2. Clean data (missing values, outliers, type casting)
          3. Profile data (EDA + smart alerts)
          4. Engineer features (encoding, scaling, datetime, interactions)
          5. Split data
          6. Train & compare models
          7. Evaluate best model
          8. Generate report

        Returns
        -------
        PipelineResult with all intermediate and final artifacts
        """
        result = PipelineResult()
        cfg = self.config

        # Reset logger for a fresh run
        self.audit.reset()

        self.audit.log_decision(
            module="pipeline.runner",
            action="pipeline_start",
            rationale=f"Starting AuditFlow pipeline. Data source: {cfg.data.source}. "
                      f"Target: {cfg.model.target}. Task: {cfg.model.task}.",
            details={
                "source": cfg.data.source,
                "target": cfg.model.target,
                "task": cfg.model.task,
            },
        )

        # ── Step 1: Load ─────────────────────────────────────
        with self.audit.track("Data Loading"):
            df = self._load_data()
            result.df_raw = df.copy()

        # ── Step 2: Clean ────────────────────────────────────
        with self.audit.track("Data Cleaning"):
            df = self._clean_data(df)
            result.df_clean = df.copy()

        # ── Step 3: Profile ──────────────────────────────────
        with self.audit.track("Exploratory Data Analysis"):
            result.profile = self._profile_data(df)

        # ── Step 4: Feature Engineering ──────────────────────
        with self.audit.track("Feature Engineering"):
            df = self._engineer_features(df)
            result.df_featured = df.copy()

        # ── Step 5-7: Split, Train, Evaluate ─────────────────
        if cfg.model.target and cfg.model.target in df.columns:
            with self.audit.track("Model Training & Evaluation"):
                comparison, model_results, best = self._train_and_evaluate(df)
                result.comparison = comparison
                result.model_results = model_results
                result.best_model = best

        # ── Step 8: Report ───────────────────────────────────
        with self.audit.track("Report Generation"):
            report_path = self.audit.generate_report(
                output_path=cfg.report.output,
                title=cfg.report.title,
                show_audit_trail=cfg.report.include_audit_trail,
            )
            result.report_path = report_path

            # Also export JSON audit trail
            from auditflow.pipeline.audit import AuditTrail
            trail = AuditTrail(self.audit)
            json_path = str(Path(cfg.report.output).with_suffix(".audit.json"))
            trail.export_json(json_path)
            result.audit_json_path = json_path

        print(f"\n✅ Pipeline complete! Report saved to: {report_path}")
        print(self.audit.summary())

        return result

    # ── Private step methods ─────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        """Load data based on config."""
        from auditflow.loaders import load_csv, load_excel

        cfg = self.config.data
        fmt = cfg.format.lower()

        if fmt == "csv":
            return load_csv(
                cfg.source, sep=cfg.sep, encoding=cfg.encoding,
                parse_dates=cfg.date_columns or None,
            )
        elif fmt == "excel":
            return load_excel(cfg.source, sheet_name=cfg.sheet_name)
        elif fmt == "api":
            from auditflow.loaders import load_from_api
            return load_from_api(
                cfg.source, headers=cfg.api_headers,
                data_key=cfg.api_data_key,
            )
        else:
            return load_csv(cfg.source)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data based on config."""
        from auditflow.cleaners import impute, handle_outliers, auto_cast

        cfg = self.config.cleaning

        # Auto type casting
        df = auto_cast(df)

        # Missing value handling
        missing_cfg = cfg.missing
        strategy = missing_cfg.get("strategy", "auto")
        df = impute(df, strategy=strategy)

        # Outlier handling
        outlier_cfg = cfg.outliers
        method = outlier_cfg.get("method", "clip")
        if method != "none":
            multiplier = outlier_cfg.get("multiplier", 1.5)
            columns = outlier_cfg.get("columns")
            df = handle_outliers(
                df, method=method, iqr_multiplier=multiplier,
                columns=columns,
            )

        # Text cleaning
        if cfg.text_columns:
            from auditflow.cleaners import clean_text
            df = clean_text(df, columns=cfg.text_columns, **cfg.text_options)

        return df

    def _profile_data(self, df: pd.DataFrame) -> Dict:
        """Profile data and generate visualizations."""
        from auditflow.eda import profile, plot_distributions, plot_correlations
        from auditflow.eda import plot_class_balance, plot_missing_map

        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend for pipeline

        target = self.config.model.target
        prof = profile(df, target=target)

        # Generate key visualizations
        try:
            plot_distributions(df)
        except Exception:
            pass

        try:
            plot_correlations(df)
        except Exception:
            pass

        if target and target in df.columns:
            try:
                plot_class_balance(df, target)
            except Exception:
                pass

        if df.isnull().sum().sum() > 0:
            try:
                plot_missing_map(df)
            except Exception:
                pass

        import matplotlib.pyplot as plt
        plt.close("all")

        return prof

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features based on config."""
        from auditflow.features import (
            expand_datetime, add_interactions, encode_categoricals, scale,
        )

        cfg = self.config.features
        target = self.config.model.target

        # DateTime expansion
        for col in cfg.datetime_expand:
            if col in df.columns:
                df = expand_datetime(df, col, add_cyclical=cfg.cyclical_encoding)

        # Interaction features
        if cfg.interactions:
            valid_pairs = [
                (a, b) for a, b in cfg.interactions
                if a in df.columns and b in df.columns
            ]
            if valid_pairs:
                df = add_interactions(df, valid_pairs, cfg.interaction_ops)

        # Categorical encoding
        if cfg.ordinal_cols or cfg.ohe_cols or cfg.label_encode_cols:
            df = encode_categoricals(
                df,
                ordinal_cols=cfg.ordinal_cols or None,
                ohe_cols=cfg.ohe_cols or None,
                label_encode_cols=cfg.label_encode_cols or None,
            )
        else:
            # Auto one-hot encode remaining object columns (except target)
            obj_cols = [
                c for c in df.select_dtypes(include=["object", "category"]).columns
                if c != target
            ]
            if obj_cols:
                df = encode_categoricals(df, ohe_cols=obj_cols)

        return df

    def _train_and_evaluate(self, df: pd.DataFrame):
        """Split, train, evaluate, and explain models."""
        from auditflow.models import ModelTrainer, explain_model, plot_feature_importance
        from auditflow.evaluation import (
            evaluate_classification, evaluate_regression,
            plot_confusion_matrix, plot_roc_curve,
        )

        import matplotlib
        matplotlib.use("Agg")

        cfg = self.config.model
        target = cfg.target

        # Prepare X and y
        y = df[target]
        X = df.drop(columns=[target])

        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])

        # Handle any remaining NaN
        X = X.fillna(0)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state,
            stratify=y if cfg.task == "classification" else None,
        )

        self.audit.log_decision(
            module="pipeline.runner",
            action="train_test_split",
            rationale=f"Split data: {len(X_train)} train / {len(X_test)} test "
                      f"({cfg.test_size:.0%} test). "
                      f"{'Stratified by target.' if cfg.task == 'classification' else ''}",
            details={
                "train_size": len(X_train),
                "test_size": len(X_test),
                "features": X.shape[1],
            },
        )

        # Train and compare
        trainer = ModelTrainer(task=cfg.task, cv_folds=cfg.cv_folds)
        comparison = trainer.compare(
            X_train, y_train, X_test, y_test,
            models=cfg.models,
            model_kwargs=cfg.model_kwargs,
        )

        # Get best model
        best_result = trainer.get_best_model()

        # Explain best model
        if best_result:
            importance_df = explain_model(
                best_result.model,
                feature_names=best_result.feature_names,
                X_test=X_test,
                y_test=y_test,
            )
            if not importance_df.empty:
                plot_feature_importance(
                    importance_df,
                    title=f"Feature Importance ({best_result.model_name})",
                )

            # Classification-specific plots
            if cfg.task == "classification":
                y_pred = best_result.model.predict(X_test)
                plot_confusion_matrix(y_test, y_pred)

                if hasattr(best_result.model, "predict_proba"):
                    try:
                        y_prob = best_result.model.predict_proba(X_test)
                        if len(np.unique(y_test)) == 2:
                            plot_roc_curve(y_test, y_prob)
                    except Exception:
                        pass

        import matplotlib.pyplot as plt
        plt.close("all")

        return comparison, trainer.results, best_result
