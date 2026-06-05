# ============================================================
# auditflow/models/classical.py
# Unified ML model training with audit trail
# ============================================================

"""
Provides a factory-pattern ModelTrainer that supports:
  - Logistic Regression, Random Forest, Gradient Boosting, SVM,
    Ridge, Lasso, ElasticNet, KNN, Decision Tree
  - Unified train/predict/compare API
  - Cross-validation with detailed logging
  - Auto-comparison of multiple models
"""

import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from auditflow.core.logger import get_logger

# ── Model Factory ────────────────────────────────────────────

MODEL_REGISTRY = {
    "classification": {
        "logistic": lambda **kw: LogisticRegression(max_iter=1000, **kw),
        "rf": lambda **kw: RandomForestClassifier(
            n_estimators=100, random_state=42, **kw
        ),
        "gbm": lambda **kw: GradientBoostingClassifier(
            n_estimators=100, random_state=42, **kw
        ),
        "svm": lambda **kw: SVC(probability=True, random_state=42, **kw),
        "dt": lambda **kw: DecisionTreeClassifier(random_state=42, **kw),
        "knn": lambda **kw: KNeighborsClassifier(**kw),
    },
    "regression": {
        "ridge": lambda **kw: Ridge(**kw),
        "lasso": lambda **kw: Lasso(**kw),
        "elasticnet": lambda **kw: ElasticNet(**kw),
        "rf": lambda **kw: RandomForestRegressor(
            n_estimators=100, random_state=42, **kw
        ),
        "gbm": lambda **kw: GradientBoostingRegressor(
            n_estimators=100, random_state=42, **kw
        ),
        "svm": lambda **kw: SVR(**kw),
        "dt": lambda **kw: DecisionTreeRegressor(random_state=42, **kw),
        "knn": lambda **kw: KNeighborsRegressor(**kw),
    },
}


def get_model(name: str, task: str = "classification", **kwargs: Any) -> Any:
    """
    Factory function to create a scikit-learn model by short name.

    Parameters
    ----------
    name : One of: logistic, rf, gbm, svm, dt, knn, ridge, lasso, elasticnet
    task : 'classification' or 'regression'

    Returns
    -------
    Unfitted scikit-learn estimator
    """
    registry = MODEL_REGISTRY.get(task, {})
    if name not in registry:
        available = list(registry.keys())
        raise ValueError(
            f"Unknown model '{name}' for task='{task}'. " f"Available: {available}"
        )
    return registry[name](**kwargs)


@dataclass
class ModelResult:
    """Container for a single model's training results."""

    model_name: str
    model: Any
    task: str
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    train_score: float
    test_score: float
    test_metrics: Dict[str, float]
    train_time_sec: float
    feature_names: List[str]


class ModelTrainer:
    """
    Unified model training interface with audit trail.

    Usage:
        trainer = ModelTrainer(task="classification")
        result = trainer.train(X_train, y_train, X_test, y_test, model_name="rf")

        # Or compare multiple models at once:
        results = trainer.compare(X_train, y_train, X_test, y_test,
                                  models=["logistic", "rf", "gbm"])
    """

    def __init__(self, task: str = "classification", cv_folds: int = 5):
        self.task = task
        self.cv_folds = cv_folds
        self.results: List[ModelResult] = []

    def _get_scoring(self) -> str:
        return "accuracy" if self.task == "classification" else "r2"

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "rf",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ModelResult:
        """
        Train a single model with cross-validation and full metrics.

        Returns
        -------
        ModelResult with all metrics and the fitted model.
        """
        audit = get_logger()
        model_kwargs = model_kwargs or {}

        model = get_model(model_name, self.task, **model_kwargs)
        scoring = self._get_scoring()

        # Cross-validation
        start = time.time()
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=self.cv_folds,
            scoring=scoring,
        ).tolist()

        # Fit on full training set
        model.fit(X_train, y_train)
        train_time = time.time() - start

        # Evaluate
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        if self.task == "classification":
            train_score = accuracy_score(y_train, y_train_pred)
            test_score = accuracy_score(y_test, y_test_pred)
            avg = "binary" if len(np.unique(y_train)) == 2 else "weighted"
            test_metrics = {
                "accuracy": round(test_score, 4),
                "precision": round(
                    precision_score(y_test, y_test_pred, average=avg, zero_division=0),
                    4,
                ),
                "recall": round(
                    recall_score(y_test, y_test_pred, average=avg, zero_division=0), 4
                ),
                "f1": round(
                    f1_score(y_test, y_test_pred, average=avg, zero_division=0), 4
                ),
            }
        else:
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)
            test_metrics = {
                "r2": round(test_score, 4),
                "mae": round(mean_absolute_error(y_test, y_test_pred), 4),
                "rmse": round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4),
            }

        cv_mean = round(np.mean(cv_scores), 4)
        cv_std = round(np.std(cv_scores), 4)

        feature_names = list(X_train.columns) if hasattr(X_train, "columns") else []

        result = ModelResult(
            model_name=model_name,
            model=model,
            task=self.task,
            cv_scores=[round(s, 4) for s in cv_scores],
            cv_mean=cv_mean,
            cv_std=cv_std,
            train_score=round(train_score, 4),
            test_score=round(test_score, 4),
            test_metrics=test_metrics,
            train_time_sec=round(train_time, 2),
            feature_names=feature_names,
        )
        self.results.append(result)

        # Determine if overfitting
        overfit_gap = train_score - test_score
        overfit_note = ""
        if overfit_gap > 0.1:
            overfit_note = (
                f" ⚠️ Possible overfitting detected: train-test gap = {overfit_gap:.3f}. "
                f"Consider regularization or reducing model complexity."
            )

        audit.log_decision(
            module="models.classical",
            action=f"train_{model_name}",
            rationale=f"Trained {model_name} ({self.task}). "
            f"CV score: {cv_mean:.4f} ±{cv_std:.4f} ({self.cv_folds}-fold). "
            f"Test score: {test_score:.4f}. "
            f"Training time: {train_time:.2f}s.{overfit_note}",
            details={
                "model": model_name,
                "task": self.task,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "test_metrics": test_metrics,
                "train_time_sec": round(train_time, 2),
                "train_score": round(train_score, 4),
                "overfit_gap": round(overfit_gap, 4),
            },
        )

        # Store metrics for the report
        audit.store_metrics(f"{model_name} ({self.task})", test_metrics)

        return result

    def compare(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        models: Optional[List[str]] = None,
        model_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """
        Train and compare multiple models.

        Parameters
        ----------
        models       : List[Any] of model names to compare (default: all available).
        model_kwargs : {model_name: {param: value}} per-model hyperparameters.

        Returns
        -------
        pd.DataFrame — comparison table sorted by test score (descending).
        """
        audit = get_logger()
        models = models or list(MODEL_REGISTRY[self.task].keys())
        model_kwargs = model_kwargs or {}

        for name in models:
            kwargs = model_kwargs.get(name, {})
            try:
                self.train(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    model_name=name,
                    model_kwargs=kwargs,
                )
            except Exception as e:
                audit.log_decision(
                    module="models.classical",
                    action=f"train_{name}_failed",
                    rationale=f"Model '{name}' failed to train: {e}",
                    details={"model": name, "error": str(e)},
                )

        # Build comparison table
        rows = []
        for r in self.results:
            row = {
                "model": r.model_name,
                "cv_mean": r.cv_mean,
                "cv_std": r.cv_std,
                "test_score": r.test_score,
                "train_time_sec": r.train_time_sec,
            }
            row.update(r.test_metrics)
            rows.append(row)

        comparison = pd.DataFrame(rows).sort_values("test_score", ascending=False)

        best = comparison.iloc[0]
        audit.log_decision(
            module="models.classical",
            action="compare_models",
            rationale=f"Compared {len(models)} models. Best: {best['model']} "
            f"(test score={best['test_score']:.4f}). "
            f"Runner-up: {comparison.iloc[1]['model'] if len(comparison) > 1 else 'N/A'}.",
            details={
                "models_compared": models,
                "best_model": best["model"],
                "best_score": best["test_score"],
                "comparison_table": comparison.to_dict(orient="records"),
            },
        )

        return comparison

    def get_best_model(self) -> Optional[ModelResult]:
        """Return the best-performing model result."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.test_score)
