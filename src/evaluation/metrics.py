"""Evaluation metrics and model comparison utilities for ADIP.

Provides point-forecast accuracy metrics, probabilistic coverage checks,
a walk-forward cross-validation splitter, and a multi-store leaderboard.
"""

import logging
import warnings
from typing import Generator, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE).

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        MAE as a non-negative float.

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Inputs must not be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE).

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        RMSE as a non-negative float.

    Raises:
        ValueError: If inputs are empty or have mismatched lengths.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Inputs must not be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute Mean Absolute Percentage Error (MAPE).

    Rows where ``y_true == 0`` are excluded to avoid division by zero.
    A warning is logged if any rows are dropped.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        MAPE as a percentage (e.g. 12.5 means 12.5 %).

    Raises:
        ValueError: If inputs are empty, mismatched, or all zeros.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Inputs must not be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    mask = y_true != 0.0
    n_zeros = int((~mask).sum())
    if n_zeros > 0:
        logger.warning(
            "MAPE: excluding %d rows where y_true == 0 (%.1f%% of data)",
            n_zeros,
            100.0 * n_zeros / len(y_true),
        )
    if not mask.any():
        raise ValueError("All y_true values are zero; MAPE is undefined.")

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def peak_error_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_percentile: float = 75.0,
) -> float:
    """Compute Peak Error Rate (PER).

    A *peak* is any row where ``y_true`` is at or above the given
    percentile.  A peak is *missed* (FN) when the corresponding
    prediction is below the threshold.

    PER = FN / (TP + FN)

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.
        threshold_percentile: Percentile (0–100) defining a peak.

    Returns:
        PER in [0, 1].  0.0 means all peaks were detected.

    Raises:
        ValueError: If inputs are empty or mismatched.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Inputs must not be empty.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    threshold = float(np.percentile(y_true, threshold_percentile))
    peak_mask = y_true >= threshold
    total_peaks = int(peak_mask.sum())
    if total_peaks == 0:
        return 0.0  # No peaks to miss

    # TP: predicted value also >= threshold; FN: predicted value < threshold
    tp = int((peak_mask & (y_pred >= threshold)).sum())
    fn = total_peaks - tp
    return float(fn / (tp + fn))


def coverage_rate(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of observations falling within the prediction interval.

    Args:
        y_true: Ground-truth values.
        lower: Lower bound of prediction interval.
        upper: Upper bound of prediction interval.

    Returns:
        Coverage rate in [0, 1].

    Raises:
        ValueError: If inputs are empty or mismatched.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty.")
    if not (len(y_true) == len(lower) == len(upper)):
        raise ValueError("All inputs must have the same length.")

    within = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(within))


# ---------------------------------------------------------------------------
# Model Evaluator
# ---------------------------------------------------------------------------


class ModelEvaluator:
    """Comprehensive model evaluation and walk-forward validation.

    Wraps scalar metrics into a single ``evaluate`` call and supports
    multi-store leaderboard generation and walk-forward time-series
    cross-validation.

    Args:
        date_col: Name of the date column used for temporal splitting.
        target_col: Name of the target column.
        store_col: Name of the store identifier column.

    Attributes:
        date_col: Date column name.
        target_col: Target column name.
        store_col: Store identifier column name.
    """

    def __init__(
        self,
        date_col: str = "date",
        target_col: str = "demand",
        store_col: str = "store_id",
    ) -> None:
        self.date_col = date_col
        self.target_col = target_col
        self.store_col = store_col

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lower_95: Optional[np.ndarray] = None,
        upper_95: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Compute all scalar metrics for a single evaluation set.

        Args:
            y_true: Ground-truth values.
            y_pred: Predicted values.
            lower_95: Optional lower bound of 95 % prediction interval.
            upper_95: Optional upper bound of 95 % prediction interval.

        Returns:
            Dictionary with keys ``mae``, ``rmse``, ``mape``,
            ``peak_error_rate``, and optionally ``coverage_95``.
        """
        results: dict[str, float] = {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred),
            "peak_error_rate": peak_error_rate(y_true, y_pred),
        }
        if lower_95 is not None and upper_95 is not None:
            results["coverage_95"] = coverage_rate(y_true, lower_95, upper_95)
        return results

    def evaluate_all_stores(
        self,
        df: pd.DataFrame,
        pred_col: str = "predicted_value",
    ) -> pd.DataFrame:
        """Produce a per-store leaderboard sorted by MAPE ascending.

        Args:
            df: DataFrame containing ``store_col``, ``target_col``, and
                ``pred_col`` columns.
            pred_col: Name of the prediction column.

        Returns:
            Leaderboard DataFrame with columns: ``store_id``, ``mae``,
            ``rmse``, ``mape``, ``peak_error_rate``, ``grade``.
        """
        rows: list[dict] = []
        for store_id, grp in df.groupby(self.store_col):
            y_true = grp[self.target_col].values
            y_pred = grp[pred_col].values
            metrics = self.evaluate(y_true, y_pred)
            metrics[self.store_col] = store_id
            metrics["grade"] = self.grade(metrics["mape"])
            rows.append(metrics)

        leaderboard = pd.DataFrame(rows)
        col_order = [self.store_col, "mae", "rmse", "mape", "peak_error_rate", "grade"]
        leaderboard = leaderboard[[c for c in col_order if c in leaderboard.columns]]
        leaderboard = leaderboard.sort_values("mape", ascending=True).reset_index(drop=True)
        return leaderboard

    def walk_forward_splits(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        test_size: int = 7,
    ) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """Yield walk-forward (train, test) splits for time-series CV.

        Each split advances the train/test boundary by ``test_size`` days.
        The first split uses ``len(unique_dates) - n_splits * test_size``
        dates for training.

        Args:
            df: Full DataFrame with ``date_col``.
            n_splits: Number of train/test folds to produce.
            test_size: Number of days in each test fold.

        Yields:
            (train_df, test_df) tuples.

        Raises:
            ValueError: If not enough dates for the requested splits.
        """
        dates = sorted(pd.to_datetime(df[self.date_col]).unique())
        total_dates = len(dates)
        required = n_splits * test_size
        if total_dates < required + 1:
            raise ValueError(
                f"Need at least {required + 1} unique dates for "
                f"{n_splits} splits × {test_size}-day test windows, "
                f"got {total_dates}."
            )

        df = df.copy()
        df["_parsed_date"] = pd.to_datetime(df[self.date_col])

        for i in range(n_splits):
            test_end_idx = total_dates - (n_splits - 1 - i) * test_size
            test_start_idx = test_end_idx - test_size
            train_end_date = dates[test_start_idx - 1]
            test_start_date = dates[test_start_idx]
            test_end_date = dates[test_end_idx - 1]

            train_df = df[df["_parsed_date"] <= train_end_date].drop(columns=["_parsed_date"])
            test_df = df[
                (df["_parsed_date"] >= test_start_date)
                & (df["_parsed_date"] <= test_end_date)
            ].drop(columns=["_parsed_date"])
            yield train_df, test_df

    @staticmethod
    def grade(mape: float) -> str:
        """Assign a letter grade based on MAPE thresholds.

        Args:
            mape: Mean Absolute Percentage Error (as percentage, e.g. 12.0).

        Returns:
            ``'PASS'`` if mape ≤ 15, ``'REVIEW'`` if mape ≤ 20, else ``'FAIL'``.
        """
        if mape <= 15.0:
            return "PASS"
        if mape <= 20.0:
            return "REVIEW"
        return "FAIL"
