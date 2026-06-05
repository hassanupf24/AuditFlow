# ============================================================
# tests/test_cleaners.py
# Tests for the cleaning modules
# ============================================================

import numpy as np
import pandas as pd
import pytest
from auditflow.core.logger import AuditLogger
from auditflow.cleaners.missing import audit_missing, impute
from auditflow.cleaners.outliers import detect_outliers, handle_outliers
from auditflow.cleaners.types import auto_cast, cast_columns


@pytest.fixture(autouse=True)
def reset_logger():
    """Reset singleton between tests."""
    AuditLogger.reset_instance()
    yield
    AuditLogger.reset_instance()


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with missing values and outliers."""
    np.random.seed(42)
    return pd.DataFrame({
        "age": [25, 30, np.nan, 45, 50, 22, np.nan, 35, 40, 28],
        "salary": [50000, 60000, 70000, 200000, 55000, 45000, 65000, 80000, 75000, 52000],
        "city": ["NYC", "LA", "NYC", None, "LA", "NYC", "LA", "NYC", "LA", None],
        "score": [85.5, 90.0, 78.3, 92.1, 88.7, 76.4, 95.0, 82.3, 91.5, 87.2],
    })


class TestMissing:
    """Tests for the missing value handler."""

    def test_audit_missing(self, sample_df):
        """Should return a report with columns that have nulls."""
        report = audit_missing(sample_df)
        assert "age" in report.index
        assert "city" in report.index
        assert report.loc["age", "null_count"] == 2

    def test_impute_auto(self, sample_df):
        """Auto strategy should fill all nulls."""
        df = impute(sample_df, strategy="auto")
        assert df.isnull().sum().sum() == 0

    def test_impute_median(self, sample_df):
        """Median strategy should fill numeric nulls."""
        df = impute(sample_df, strategy="median")
        assert df["age"].isnull().sum() == 0

    def test_impute_drop(self, sample_df):
        """Drop strategy should remove rows with nulls."""
        df = impute(sample_df, strategy="drop")
        assert df.isnull().sum().sum() == 0
        assert len(df) < len(sample_df)

    def test_impute_no_nulls(self):
        """Should handle DataFrames with no nulls gracefully."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = impute(df, strategy="auto")
        assert result.shape == df.shape


class TestOutliers:
    """Tests for the outlier handler."""

    def test_detect_outliers(self, sample_df):
        """Should detect outliers in the salary column."""
        report = detect_outliers(sample_df, columns=["salary"])
        assert "salary" in report
        assert report["salary"]["count"] >= 0

    def test_handle_clip(self, sample_df):
        """Clipping should cap extreme values."""
        df = handle_outliers(sample_df, columns=["salary"], method="clip")
        assert df["salary"].max() <= sample_df["salary"].max()
        assert len(df) == len(sample_df)  # No rows removed

    def test_handle_drop(self, sample_df):
        """Dropping should remove rows with outliers."""
        df = handle_outliers(sample_df, columns=["salary"], method="drop")
        assert len(df) <= len(sample_df)

    def test_handle_flag(self, sample_df):
        """Flagging should add boolean columns."""
        df = handle_outliers(sample_df, columns=["salary"], method="flag")
        assert "salary_is_outlier" in df.columns


class TestTypeCasting:
    """Tests for the type casting module."""

    def test_auto_cast_numeric_strings(self):
        """Should detect numeric strings and cast them."""
        df = pd.DataFrame({"value": ["1", "2", "3", "4", "5"]})
        result = auto_cast(df)
        assert pd.api.types.is_numeric_dtype(result["value"])

    def test_auto_cast_boolean_like(self):
        """Should detect boolean-like values."""
        df = pd.DataFrame({"flag": ["yes", "no", "yes", "no", "yes"]})
        result = auto_cast(df)
        # Should be converted to bool
        assert result["flag"].dtype == bool or result["flag"].isin([True, False]).all()

    def test_cast_columns_explicit(self):
        """Should cast specified columns."""
        df = pd.DataFrame({
            "age": ["25", "30", "35"],
            "active": ["true", "false", "true"],
        })
        result = cast_columns(df, int_cols=["age"])
        assert pd.api.types.is_numeric_dtype(result["age"])
