"""Feature engineering pipeline for ADIP time-series models.

Provides lag, rolling, calendar, and entity features with proper
group isolation and data leakage prevention.
"""

import logging

import holidays
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Time-series feature engineering pipeline.

    Generates lag, rolling, calendar, and entity features for daily
    or hourly forecasting models. Processes each (store_id, [category])
    group independently to avoid cross-group lag contamination.

    Attributes:
        target_col: Column name of the forecast target.
        freq: 'D' for daily, 'H' for hourly models.
        store_encoder: Fitted LabelEncoder for store_id.
        category_encoder: Fitted LabelEncoder for product_category.
        last_fit_date: Last date seen during fit, used for leakage check.
    """

    def __init__(self, target_col: str, freq: str = "D"):
        if freq not in ("D", "H"):
            raise ValueError(f"freq must be 'D' or 'H', got '{freq}'")
        self.target_col = target_col
        self.freq = freq
        self.store_encoder = LabelEncoder()
        self.category_encoder = LabelEncoder()
        self.last_fit_date = None
        self._is_fitted = False

    def _add_lag_features(self, group: pd.DataFrame) -> pd.DataFrame:
        target = group[self.target_col]
        group["lag_1d"] = target.shift(1)
        group["lag_7d"] = target.shift(7)
        group["lag_14d"] = target.shift(14)
        group["lag_30d"] = target.shift(30)
        if self.freq == "H":
            group["lag_24h"] = target.shift(24)
        return group

    def _add_rolling_features(self, group: pd.DataFrame) -> pd.DataFrame:
        target = group[self.target_col]
        group["roll_mean_3d"] = target.shift(1).rolling(window=3, min_periods=1).mean()
        group["roll_mean_7d"] = target.shift(1).rolling(window=7, min_periods=3).mean()
        group["roll_std_7d"] = target.shift(1).rolling(window=7, min_periods=3).std()
        group["ewma_7d"] = target.shift(1).ewm(span=7, adjust=False).mean()
        return group

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "date" in df.columns:
            dt = pd.to_datetime(df["date"])
        elif "datetime" in df.columns:
            dt = pd.to_datetime(df["datetime"])
        else:
            raise ValueError("DataFrame must have 'date' or 'datetime' column")

        if self.freq == "H" and "hour" in df.columns:
            df["hour_of_day"] = df["hour"].astype(int)
        elif self.freq == "H":
            df["hour_of_day"] = dt.dt.hour

        df["day_of_week"] = dt.dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["month"] = dt.dt.month
        df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Holiday features
        us_holidays = holidays.US(years=dt.dt.year.unique())
        df["is_holiday"] = dt.dt.date.apply(lambda x: int(x in us_holidays))

        # Calculate days to next holiday
        holiday_dates = [h for h in us_holidays.keys()]
        if holiday_dates:

            def days_to_next(d):
                future = [h for h in holiday_dates if h > d]
                if future:
                    return (min(future) - d).days
                return 30  # Cap if no future holidays in list

            df["days_to_holiday"] = dt.dt.date.apply(days_to_next)
        else:
            df["days_to_holiday"] = 30

        return df

    def _add_entity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "store_id" in df.columns:
            df["store_id_enc"] = self.store_encoder.transform(df["store_id"])
        if "product_category" in df.columns:
            df["category_enc"] = self.category_encoder.transform(df["product_category"])
        return df

    def _get_group_cols(self, df: pd.DataFrame) -> list:
        group_cols = ["store_id"]
        if "product_category" in df.columns:
            group_cols.append("product_category")
        return group_cols

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit pipeline on df and return feature-enriched DataFrame.

        Args:
            df: Input DataFrame with target_col, store_id, and date columns.

        Returns:
            Feature-enriched DataFrame with no data leakage.
        """
        df = df.copy().sort_values(["store_id", "date"] if "date" in df.columns else ["store_id"])

        # Fit encoders
        if "store_id" in df.columns:
            self.store_encoder.fit(df["store_id"])
        if "product_category" in df.columns:
            self.category_encoder.fit(df["product_category"])

        # Track last fit date for leakage prevention
        if "date" in df.columns:
            self.last_fit_date = pd.to_datetime(df["date"]).max()

        # Process lag and rolling features per group
        group_cols = self._get_group_cols(df)
        groups = []
        for name, group in df.groupby(group_cols, sort=False):
            group = group.sort_values("date" if "date" in group.columns else group.index.name)
            group = self._add_lag_features(group)
            group = self._add_rolling_features(group)
            groups.append(group)
        df = pd.concat(groups, ignore_index=True)

        # Add calendar features
        df = self._add_calendar_features(df)

        # Add entity features
        df = self._add_entity_features(df)

        self._is_fitted = True
        logger.info(f"FeaturePipeline fit_transform complete: {len(df)} rows, {len(df.columns)} cols")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transforms to new data.

        Args:
            df: New DataFrame to transform.

        Returns:
            Feature-enriched DataFrame.

        Raises:
            RuntimeError: If pipeline has not been fitted.
            ValueError: If df contains future timestamps beyond fit range.
        """
        if not self._is_fitted:
            raise RuntimeError("FeaturePipeline must be fit before transform.")

        df = df.copy()

        # Check for future timestamp leakage
        if self.last_fit_date is not None and "date" in df.columns:
            max_date = pd.to_datetime(df["date"]).max()
            if max_date > self.last_fit_date:
                raise ValueError(
                    f"DataFrame contains dates ({max_date}) beyond last fit date "
                    f"({self.last_fit_date}). Re-fit pipeline with updated data."
                )

        group_cols = self._get_group_cols(df)
        groups = []
        for name, group in df.groupby(group_cols, sort=False):
            group = group.sort_values("date" if "date" in group.columns else group.index.name)
            group = self._add_lag_features(group)
            group = self._add_rolling_features(group)
            groups.append(group)
        df = pd.concat(groups, ignore_index=True)

        df = self._add_calendar_features(df)
        df = self._add_entity_features(df)

        logger.info(f"FeaturePipeline transform complete: {len(df)} rows")
        return df

    def get_feature_names(self) -> list[str]:
        """Return list of all feature column names generated by this pipeline."""
        features = [
            "lag_1d",
            "lag_7d",
            "lag_14d",
            "lag_30d",
            "roll_mean_3d",
            "roll_mean_7d",
            "roll_std_7d",
            "ewma_7d",
            "day_of_week",
            "is_weekend",
            "month",
            "sin_dow",
            "cos_dow",
            "is_holiday",
            "days_to_holiday",
            "store_id_enc",
        ]
        if self.freq == "H":
            features.extend(["lag_24h", "hour_of_day"])
        return features
