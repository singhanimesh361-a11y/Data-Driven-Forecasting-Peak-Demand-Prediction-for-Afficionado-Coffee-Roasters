"""Unit tests for the FeaturePipeline.

Covers lag correctness, NaN propagation, group isolation, calendar
features, edge cases, and encoder consistency.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.feature_pipeline import FeaturePipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Multi-store daily DataFrame with 30 days per store."""
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    rows = []
    for store in ["store_3", "store_5"]:
        for d in dates:
            rows.append(
                {
                    "store_id": store,
                    "date": d,
                    "demand": np.random.default_rng(42).integers(50, 200),
                }
            )
    df = pd.DataFrame(rows)
    # Deterministic demand per (store, day-index) for reproducibility
    rng = np.random.default_rng(42)
    df["demand"] = rng.integers(50, 200, size=len(df))
    return df


@pytest.fixture()
def pipeline() -> FeaturePipeline:
    """Default daily FeaturePipeline."""
    return FeaturePipeline(target_col="demand", freq="D")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLagFeatures:
    """Tests for lag feature correctness."""

    def test_lag_correctness(self, sample_df: pd.DataFrame, pipeline: FeaturePipeline) -> None:
        """Spot-check row 8 (0-indexed) lag_1d equals row 7 demand within same group."""
        result = pipeline.fit_transform(sample_df)
        for store in sample_df["store_id"].unique():
            grp = result[result["store_id"] == store].reset_index(drop=True)
            # lag_1d at index 8 should equal demand at index 7
            assert grp.loc[8, "lag_1d"] == pytest.approx(grp.loc[7, "demand"]), (
                f"lag_1d mismatch for {store} at row 8"
            )
            # lag_7d at index 8 should equal demand at index 1
            assert grp.loc[8, "lag_7d"] == pytest.approx(grp.loc[1, "demand"]), (
                f"lag_7d mismatch for {store} at row 8"
            )

    def test_no_nan_propagation(self, sample_df: pd.DataFrame, pipeline: FeaturePipeline) -> None:
        """After the warmup period (7 rows), lag features must not be NaN."""
        result = pipeline.fit_transform(sample_df)
        for store in sample_df["store_id"].unique():
            grp = result[result["store_id"] == store].reset_index(drop=True)
            warmup_done = grp.iloc[7:]
            assert warmup_done["lag_1d"].isna().sum() == 0, "lag_1d has NaN after warmup"
            assert warmup_done["lag_7d"].isna().sum() == 0, "lag_7d has NaN after warmup"


class TestGroupIsolation:
    """Tests for cross-group leakage prevention."""

    def test_group_isolation(self, pipeline: FeaturePipeline) -> None:
        """Lag from store_3 must not leak into store_5."""
        dates = pd.date_range("2025-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "store_id": ["store_3"] * 10 + ["store_5"] * 10,
                "date": list(dates) * 2,
                "demand": list(range(100, 110)) + list(range(200, 210)),
            }
        )
        result = pipeline.fit_transform(df)
        store5 = result[result["store_id"] == "store_5"].reset_index(drop=True)

        # First lag_1d for store_5 should be NaN (not store_3's last value)
        assert pd.isna(store5.loc[0, "lag_1d"]), (
            "store_5 row 0 lag_1d should be NaN, not leaking from store_3"
        )
        # Second lag_1d for store_5 should be 200, not 109
        assert store5.loc[1, "lag_1d"] == pytest.approx(200.0)


class TestCalendarFeatures:
    """Tests for calendar feature generation."""

    def test_weekend_flag(self, sample_df: pd.DataFrame, pipeline: FeaturePipeline) -> None:
        """Saturday and Sunday should be flagged as weekend."""
        result = pipeline.fit_transform(sample_df)
        result["_dt"] = pd.to_datetime(result["date"])
        saturdays = result[result["_dt"].dt.dayofweek == 5]
        sundays = result[result["_dt"].dt.dayofweek == 6]
        weekdays = result[result["_dt"].dt.dayofweek < 5]

        assert (saturdays["is_weekend"] == 1).all(), "Saturdays should be weekend"
        assert (sundays["is_weekend"] == 1).all(), "Sundays should be weekend"
        assert (weekdays["is_weekend"] == 0).all(), "Weekdays should not be weekend"

    def test_cyclical_encoding_range(self, sample_df: pd.DataFrame, pipeline: FeaturePipeline) -> None:
        """sin_dow and cos_dow must be in [-1, 1]."""
        result = pipeline.fit_transform(sample_df)
        assert result["sin_dow"].between(-1, 1).all(), "sin_dow out of range"
        assert result["cos_dow"].between(-1, 1).all(), "cos_dow out of range"


class TestEdgeCases:
    """Tests for edge-case inputs."""

    def test_empty_dataframe(self, pipeline: FeaturePipeline) -> None:
        """Pipeline should handle an empty DataFrame gracefully."""
        empty = pd.DataFrame(columns=["store_id", "date", "demand"])
        # fit_transform on empty should either return empty or raise cleanly
        try:
            result = pipeline.fit_transform(empty)
            assert len(result) == 0
        except (ValueError, KeyError):
            pass  # Acceptable to raise on empty input

    def test_single_row(self, pipeline: FeaturePipeline) -> None:
        """Pipeline should not crash on a single-row DataFrame."""
        df = pd.DataFrame(
            {
                "store_id": ["store_1"],
                "date": [pd.Timestamp("2025-06-01")],
                "demand": [100],
            }
        )
        result = pipeline.fit_transform(df)
        assert len(result) == 1
        assert pd.isna(result.iloc[0]["lag_1d"]), "Single row lag_1d should be NaN"


class TestEncoderConsistency:
    """Tests for entity encoder stability."""

    def test_store_encoder_consistency(self, sample_df: pd.DataFrame, pipeline: FeaturePipeline) -> None:
        """Same store_id must always map to the same encoded value."""
        result = pipeline.fit_transform(sample_df)
        enc_map = result.groupby("store_id")["store_id_enc"].nunique()
        for store, n_unique in enc_map.items():
            assert n_unique == 1, f"{store} mapped to {n_unique} different encoded values"

    def test_transform_encoder_matches_fit(self, sample_df: pd.DataFrame, pipeline: FeaturePipeline) -> None:
        """Encoded values from transform() must match those from fit_transform()."""
        fit_result = pipeline.fit_transform(sample_df)
        # Use a subset for transform (dates within fit range)
        subset = sample_df[sample_df["date"] <= sample_df["date"].max()].copy()
        transform_result = pipeline.transform(subset)

        # Verify store_id_enc values match between fit and transform
        fit_map = fit_result.drop_duplicates("store_id").set_index("store_id")["store_id_enc"]
        transform_map = transform_result.drop_duplicates("store_id").set_index("store_id")["store_id_enc"]
        for store in fit_map.index:
            if store in transform_map.index:
                assert fit_map[store] == transform_map[store], (
                    f"Encoder mismatch for {store}: fit={fit_map[store]}, "
                    f"transform={transform_map[store]}"
                )
