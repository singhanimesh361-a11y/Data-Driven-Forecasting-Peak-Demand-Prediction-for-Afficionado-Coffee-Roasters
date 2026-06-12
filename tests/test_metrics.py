"""Unit tests for evaluation metrics and ModelEvaluator.

Covers MAE, RMSE, MAPE (including zero-handling), Peak Error Rate,
Coverage Rate, walk-forward splits, and grade thresholds.
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    ModelEvaluator,
    coverage_rate,
    mean_absolute_error,
    mean_absolute_percentage_error,
    peak_error_rate,
    root_mean_squared_error,
)


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------


class TestMAE:
    """Mean Absolute Error tests."""

    def test_mae_basic(self) -> None:
        """[1,2,3] vs [1,2,4] -> MAE = 1/3."""
        result = mean_absolute_error([1, 2, 3], [1, 2, 4])
        assert result == pytest.approx(1.0 / 3.0)

    def test_mae_perfect(self) -> None:
        """Perfect predictions give MAE = 0."""
        result = mean_absolute_error([5, 10, 15], [5, 10, 15])
        assert result == pytest.approx(0.0)

    def test_mae_empty_raises(self) -> None:
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError):
            mean_absolute_error([], [])


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------


class TestRMSE:
    """Root Mean Squared Error tests."""

    def test_rmse_basic(self) -> None:
        """Hand-calculated: [1,2,3] vs [1,2,4] -> RMSE = sqrt(1/3)."""
        result = root_mean_squared_error([1, 2, 3], [1, 2, 4])
        expected = np.sqrt(1.0 / 3.0)
        assert result == pytest.approx(expected)

    def test_rmse_perfect(self) -> None:
        """Perfect predictions give RMSE = 0."""
        result = root_mean_squared_error([3, 6, 9], [3, 6, 9])
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MAPE
# ---------------------------------------------------------------------------


class TestMAPE:
    """Mean Absolute Percentage Error tests."""

    def test_mape_basic(self) -> None:
        """[100, 200] vs [110, 190] -> MAPE = mean(10/100, 10/200)*100 = 7.5%."""
        result = mean_absolute_percentage_error([100, 200], [110, 190])
        expected = (10.0 / 100.0 + 10.0 / 200.0) / 2.0 * 100.0  # 7.5
        assert result == pytest.approx(expected)

    def test_mape_zero_handling(self) -> None:
        """Rows with y_true == 0 are excluded; remaining rows used."""
        # y_true=[0, 100], y_pred=[10, 110] -> only row 1 used: |10/100|*100 = 10%
        result = mean_absolute_percentage_error([0, 100], [10, 110])
        assert result == pytest.approx(10.0)

    def test_mape_all_zeros_raises(self) -> None:
        """All-zero y_true should raise ValueError."""
        with pytest.raises(ValueError):
            mean_absolute_percentage_error([0, 0, 0], [1, 2, 3])


# ---------------------------------------------------------------------------
# Peak Error Rate
# ---------------------------------------------------------------------------


class TestPeakErrorRate:
    """Peak Error Rate (PER) tests."""

    def test_peak_error_rate_no_missed(self) -> None:
        """All peaks correctly predicted -> PER = 0.0."""
        y_true = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        # Threshold at 75th percentile = 77.5; peaks: 80, 90, 100
        # Predictions above threshold for those peaks
        y_pred = np.array([10, 20, 30, 40, 50, 60, 70, 85, 95, 105])
        result = peak_error_rate(y_true, y_pred, threshold_percentile=75)
        assert result == pytest.approx(0.0)

    def test_peak_error_rate_all_missed(self) -> None:
        """All peaks missed -> PER = 1.0."""
        y_true = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        # All predictions at 0
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        result = peak_error_rate(y_true, y_pred, threshold_percentile=75)
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Coverage Rate
# ---------------------------------------------------------------------------


class TestCoverageRate:
    """Prediction interval coverage tests."""

    def test_coverage_rate_full(self) -> None:
        """All observations within bounds -> coverage = 1.0."""
        y_true = np.array([5, 10, 15])
        lower = np.array([0, 5, 10])
        upper = np.array([10, 15, 20])
        assert coverage_rate(y_true, lower, upper) == pytest.approx(1.0)

    def test_coverage_rate_none(self) -> None:
        """No observations within bounds -> coverage = 0.0."""
        y_true = np.array([100, 200, 300])
        lower = np.array([0, 0, 0])
        upper = np.array([10, 10, 10])
        assert coverage_rate(y_true, lower, upper) == pytest.approx(0.0)

    def test_coverage_rate_partial(self) -> None:
        """2 of 4 within bounds -> coverage = 0.5."""
        y_true = np.array([5, 50, 10, 60])
        lower = np.array([0, 0, 0, 0])
        upper = np.array([20, 20, 20, 20])
        assert coverage_rate(y_true, lower, upper) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Walk-forward splits
# ---------------------------------------------------------------------------


class TestWalkForwardSplits:
    """Walk-forward time-series cross-validation tests."""

    @pytest.fixture()
    def timeseries_df(self) -> pd.DataFrame:
        """60-day time-series for two stores."""
        dates = pd.date_range("2025-01-01", periods=60, freq="D")
        rows = []
        for store in ["s1", "s2"]:
            for d in dates:
                rows.append({"store_id": store, "date": d, "demand": 100})
        return pd.DataFrame(rows)

    def test_walk_forward_split_count(self, timeseries_df: pd.DataFrame) -> None:
        """n_splits=5 should produce exactly 5 (train, test) tuples."""
        evaluator = ModelEvaluator()
        splits = list(evaluator.walk_forward_splits(timeseries_df, n_splits=5, test_size=7))
        assert len(splits) == 5

    def test_walk_forward_no_overlap(self, timeseries_df: pd.DataFrame) -> None:
        """Train max-date must be strictly before test min-date."""
        evaluator = ModelEvaluator()
        for train_df, test_df in evaluator.walk_forward_splits(
            timeseries_df, n_splits=3, test_size=7
        ):
            train_max = pd.to_datetime(train_df["date"]).max()
            test_min = pd.to_datetime(test_df["date"]).min()
            assert train_max < test_min, "Train and test overlap!"


# ---------------------------------------------------------------------------
# Grade thresholds
# ---------------------------------------------------------------------------


class TestGrade:
    """MAPE grading thresholds."""

    def test_grade_pass(self) -> None:
        """MAPE = 10 -> PASS."""
        assert ModelEvaluator.grade(10.0) == "PASS"

    def test_grade_pass_boundary(self) -> None:
        """MAPE = 15 -> PASS (boundary inclusive)."""
        assert ModelEvaluator.grade(15.0) == "PASS"

    def test_grade_review(self) -> None:
        """MAPE = 17 -> REVIEW."""
        assert ModelEvaluator.grade(17.0) == "REVIEW"

    def test_grade_review_boundary(self) -> None:
        """MAPE = 20 -> REVIEW (boundary inclusive)."""
        assert ModelEvaluator.grade(20.0) == "REVIEW"

    def test_grade_fail(self) -> None:
        """MAPE = 25 -> FAIL."""
        assert ModelEvaluator.grade(25.0) == "FAIL"
