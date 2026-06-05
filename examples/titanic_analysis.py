# ============================================================
# examples/titanic_analysis.py
# End-to-end Titanic analysis using AuditFlow's Python API
# ============================================================

"""
This example demonstrates AuditFlow's full capabilities:
  1. Loading the Titanic dataset (auto-downloaded from the web)
  2. Smart cleaning with auto-strategy selection
  3. EDA profiling with smart alerts
  4. Feature engineering (encoding, interactions)
  5. Model comparison (Logistic, RF, GBM)
  6. Explainability (feature importance)
  7. Auto-generated HTML report with full audit trail

Run:
    cd auditflow/
    pip install -e .
    python examples/titanic_analysis.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Add parent to path so we can import auditflow
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── AuditFlow Imports ────────────────────────────────────────
from auditflow.core.logger import AuditLogger, get_logger
from auditflow.cleaners import impute, handle_outliers
from auditflow.cleaners.types import auto_cast
from auditflow.eda import profile, plot_distributions, plot_correlations
from auditflow.eda import plot_class_balance
from auditflow.features.numeric import encode_categoricals, scale, add_interactions
from auditflow.models import ModelTrainer, explain_model, plot_feature_importance
from auditflow.evaluation import (
    evaluate_classification, plot_confusion_matrix, plot_roc_curve,
)
from auditflow.pipeline.audit import AuditTrail


def create_titanic_dataset() -> pd.DataFrame:
    """
    Create a representative Titanic dataset for demonstration.
    Uses seaborn's built-in dataset if available, otherwise creates synthetic.
    """
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
        # Rename to match classic Titanic column names
        df = df.rename(columns={
            "survived": "Survived",
            "pclass": "Pclass",
            "sex": "Sex",
            "age": "Age",
            "sibsp": "SibSp",
            "parch": "Parch",
            "fare": "Fare",
            "embarked": "Embarked",
            "class": "Class",
            "who": "Who",
            "adult_male": "AdultMale",
            "deck": "Deck",
            "embark_town": "EmbarkTown",
            "alive": "Alive",
            "alone": "Alone",
        })
        return df
    except Exception:
        # Synthetic fallback
        np.random.seed(42)
        n = 891
        df = pd.DataFrame({
            "Survived": np.random.choice([0, 1], n, p=[0.62, 0.38]),
            "Pclass": np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
            "Sex": np.random.choice(["male", "female"], n, p=[0.65, 0.35]),
            "Age": np.random.normal(30, 14, n).clip(0.5, 80),
            "SibSp": np.random.choice([0, 1, 2, 3, 4], n, p=[0.68, 0.23, 0.05, 0.02, 0.02]),
            "Parch": np.random.choice([0, 1, 2, 3], n, p=[0.76, 0.13, 0.08, 0.03]),
            "Fare": np.random.exponential(32, n),
            "Embarked": np.random.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09]),
        })
        # Add some missing values (realistic)
        age_mask = np.random.random(n) < 0.20
        df.loc[age_mask, "Age"] = np.nan
        emb_mask = np.random.random(n) < 0.02
        df.loc[emb_mask, "Embarked"] = np.nan
        return df


def main():
    print("=" * 60)
    print("🚢 TITANIC SURVIVAL ANALYSIS — AuditFlow Demo")
    print("=" * 60)

    # Initialize the audit logger (singleton)
    audit = get_logger()
    audit.reset()  # Fresh start

    # ── Step 1: Load Data ────────────────────────────────────
    with audit.track("Data Loading"):
        df = create_titanic_dataset()
        audit.log_decision(
            module="loaders.tabular",
            action="load_dataset",
            rationale=f"Loaded Titanic dataset: {df.shape[0]} rows × {df.shape[1]} columns. "
                      f"Columns: {list(df.columns)}.",
            details={"rows": df.shape[0], "columns": df.shape[1]},
            after_shape=df.shape,
        )
    print(f"\n📦 Loaded: {df.shape}")

    # ── Step 2: Clean Data ───────────────────────────────────
    with audit.track("Data Cleaning"):
        # Drop columns that leak or aren't useful
        drop_cols = [c for c in ["Alive", "Who", "AdultMale", "Deck", "EmbarkTown", "Class"]
                     if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            audit.log_decision(
                module="cleaners.missing",
                action="drop_columns",
                rationale=f"Dropped {len(drop_cols)} redundant/leaky columns: {drop_cols}. "
                          f"'Alive' leaks the target, 'Deck' has >70% missing.",
                details={"dropped": drop_cols},
            )

        # Smart imputation
        df = impute(df, strategy="auto")

        # Outlier handling
        df = handle_outliers(df, method="clip", iqr_multiplier=1.5)

    print(f"🧹 Cleaned: {df.shape}")

    # ── Step 3: EDA ──────────────────────────────────────────
    with audit.track("Exploratory Data Analysis"):
        prof = profile(df, target="Survived")
        plot_distributions(df)
        plot_correlations(df)
        plot_class_balance(df, "Survived")

    import matplotlib.pyplot as plt
    plt.close("all")

    # ── Step 4: Feature Engineering ──────────────────────────
    with audit.track("Feature Engineering"):
        # Encode categoricals
        df = encode_categoricals(
            df,
            ohe_cols=["Sex", "Embarked"] if "Embarked" in df.columns else ["Sex"],
            label_encode_cols=["Alone"] if "Alone" in df.columns else [],
        )

        # Add interaction features
        if "Pclass" in df.columns and "Fare" in df.columns:
            df = add_interactions(df, [("Pclass", "Fare")])

    print(f"⚙️ Engineered: {df.shape}")

    # ── Step 5: Train & Compare Models ───────────────────────
    with audit.track("Model Training & Evaluation"):
        target = "Survived"
        y = df[target]
        X = df.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y,
        )

        audit.log_decision(
            module="pipeline.runner",
            action="train_test_split",
            rationale=f"Split: {len(X_train)} train / {len(X_test)} test (20% test). "
                      f"Stratified by target. {X.shape[1]} features.",
            details={"train": len(X_train), "test": len(X_test), "features": X.shape[1]},
        )

        trainer = ModelTrainer(task="classification", cv_folds=5)
        comparison = trainer.compare(
            X_train, y_train, X_test, y_test,
            models=["logistic", "rf", "gbm"],
        )

        print("\n📊 Model Comparison:")
        print(comparison.to_string(index=False))

        # Explain best model
        best = trainer.get_best_model()
        if best:
            importance_df = explain_model(
                best.model, best.feature_names, X_test, y_test,
            )
            plot_feature_importance(
                importance_df,
                title=f"Feature Importance ({best.model_name})",
            )

            # Classification plots
            y_pred = best.model.predict(X_test)
            plot_confusion_matrix(y_test, y_pred)

            if hasattr(best.model, "predict_proba"):
                y_prob = best.model.predict_proba(X_test)
                try:
                    plot_roc_curve(y_test, y_prob)
                except Exception:
                    pass

    plt.close("all")

    # ── Step 6: Generate Report ──────────────────────────────
    report_path = audit.generate_report(
        output_path="titanic_report.html",
        title="Titanic Survival Analysis — AuditFlow Report",
    )

    # Export JSON audit trail
    trail = AuditTrail(audit)
    trail.export_json("titanic_audit_trail.json")

    print(f"\n✅ Report saved to: {report_path}")
    print(f"📋 Audit trail saved to: titanic_audit_trail.json")
    print(audit.summary())


if __name__ == "__main__":
    main()
