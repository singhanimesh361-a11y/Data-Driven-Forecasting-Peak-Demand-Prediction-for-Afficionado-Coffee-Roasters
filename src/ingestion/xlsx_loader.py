import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

KNOWN_COLUMNS = [
    "transaction_id",
    "year",
    "transaction_time",
    "transaction_qty",
    "unit_price",
    "store_id",
    "store_location",
    "product_id",
    "product_category",
    "product_type",
    "product_detail",
]

KNOWN_CATEGORIES = [
    "Coffee",
    "Tea",
    "Bakery",
    "Drinking Chocolate",
    "Coffee beans",
    "Branded",
    "Loose Tea",
    "Flavours",
    "Packaged Chocolate",
]

VALID_STORE_IDS = {3, 5, 8}


def parse_fractional_time(frac: float, year: int) -> datetime:
    """Convert Excel fractional-day float to a full Python datetime.

    The fractional time represents a fraction of a year. For example,
    0.29596 in year 2025 converts to approximately April 19, 2025 07:06 AM.

    Args:
        frac: Fractional day value from Excel (0.0 to ~1.0).
        year: The year for the base date.

    Returns:
        A datetime object representing the full date and time.

    Raises:
        ValueError: If frac is negative or year is not a valid year.
    """
    if frac < 0:
        raise ValueError(f"Fractional time cannot be negative: {frac}")
    if year < 1900 or year > 2100:
        raise ValueError(f"Year out of reasonable range: {year}")

    base_date = datetime(year, 1, 1)

    # Handle edge case: frac >= 1.0 means it wraps into next year
    if frac >= 1.0:
        logger.warning(f"Fractional time >= 1.0 ({frac}), wrapping into next year")

    # frac represents fraction of year: frac * 365 gives day offset
    # The fractional part of the day offset gives the time
    total_days = frac * 365
    full_days = int(total_days)
    fractional_day = total_days - full_days

    # Convert fractional day to hours, minutes, seconds
    total_seconds = fractional_day * 86400  # seconds in a day
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    result = base_date + timedelta(days=full_days, hours=hours, minutes=minutes, seconds=seconds)
    return result


def validate_schema(df: pd.DataFrame) -> None:
    """Validate the DataFrame schema matches expected Afficionado dataset.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If schema validation fails with descriptive message.
    """
    # Check all columns present
    missing_cols = set(KNOWN_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Check for null values
    null_counts = df[KNOWN_COLUMNS].isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        raise ValueError(f"Null values found in columns: " f"{dict(null_cols)}")

    # Validate store_id values
    invalid_stores = set(df["store_id"].unique()) - VALID_STORE_IDS
    if invalid_stores:
        raise ValueError(f"Invalid store_id values: {invalid_stores}")

    # Validate product categories
    invalid_categories = set(df["product_category"].unique()) - set(KNOWN_CATEGORIES)
    if invalid_categories:
        raise ValueError(f"Unknown product categories: {invalid_categories}")

    logger.info("Schema validation passed.")


def load_xlsx(path: str) -> pd.DataFrame:
    """Load the Afficionado Coffee Roasters Excel file and enrich with derived columns.

    Args:
        path: Path to the Excel file (.xlsx).

    Returns:
        DataFrame with original columns plus derived columns:
        datetime, date, hour, day_of_week, week_of_year, revenue.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If schema validation fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    logger.info(f"Loading Excel file: {path}")
    df = pd.read_excel(path, sheet_name="Transactions", engine="openpyxl")
    logger.info(f"Loaded {len(df):,} rows from {path.name}")

    # Parse fractional time to datetime
    logger.info("Parsing fractional time values...")
    df["datetime"] = df.apply(lambda row: parse_fractional_time(row["transaction_time"], int(row["year"])), axis=1)

    # Derive additional columns
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Monday
    df["week_of_year"] = df["datetime"].dt.isocalendar().week.astype(int)
    df["revenue"] = df["transaction_qty"] * df["unit_price"]

    # Validate schema
    validate_schema(df)

    date_range = f"{df['date'].min()} to {df['date'].max()}"
    logger.info(f"Date range: {date_range}")
    logger.info(f"Total revenue: ${df['revenue'].sum():,.2f}")

    return df


def write_to_sqlite(df: pd.DataFrame, db_path: str) -> None:
    """Write DataFrame to SQLite database.

    Args:
        df: DataFrame to write.
        db_path: Path to the SQLite database file.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing {len(df):,} rows to SQLite: {db_path}")

    # Convert date objects to strings for SQLite compatibility
    df_write = df.copy()
    if "date" in df_write.columns:
        df_write["date"] = df_write["date"].astype(str)
    if "datetime" in df_write.columns:
        df_write["datetime"] = df_write["datetime"].astype(str)

    with sqlite3.connect(str(db_path)) as conn:
        df_write.to_sql("transactions", conn, if_exists="replace", index=False)

    logger.info(f"Successfully wrote transactions table to {db_path}")


def build_aggregations(df: pd.DataFrame, db_path: str) -> dict:
    """Create and persist four aggregated tables to SQLite.

    Args:
        df: Raw transaction DataFrame with derived columns.
        db_path: Path to the SQLite database.

    Returns:
        Dict mapping table name to row count.
    """
    logger.info("Building aggregation tables...")
    results = {}

    # Ensure date is string for groupby consistency
    df_agg = df.copy()
    if hasattr(df_agg["date"].iloc[0], "isoformat"):
        df_agg["date_str"] = df_agg["date"].astype(str)
    else:
        df_agg["date_str"] = df_agg["date"]

    # 1. hourly_store
    hourly_store = (
        df_agg.groupby(["date_str", "hour", "store_id"])
        .agg(
            total_qty=("transaction_qty", "sum"),
            total_revenue=("revenue", "sum"),
            transaction_count=("transaction_id", "count"),
        )
        .reset_index()
        .rename(columns={"date_str": "date"})
    )
    results["hourly_store"] = len(hourly_store)

    # 2. daily_store
    daily_store = (
        df_agg.groupby(["date_str", "store_id"])
        .agg(
            total_qty=("transaction_qty", "sum"),
            total_revenue=("revenue", "sum"),
            transaction_count=("transaction_id", "count"),
        )
        .reset_index()
        .rename(columns={"date_str": "date"})
    )
    results["daily_store"] = len(daily_store)

    # 3. daily_category
    daily_category = (
        df_agg.groupby(["date_str", "store_id", "product_category"])
        .agg(
            total_qty=("transaction_qty", "sum"),
            total_revenue=("revenue", "sum"),
            transaction_count=("transaction_id", "count"),
        )
        .reset_index()
        .rename(columns={"date_str": "date"})
    )
    results["daily_category"] = len(daily_category)

    # 4. daily_sku — top 20 SKUs by total lifetime revenue
    sku_revenue = df_agg.groupby("product_id")["revenue"].sum()
    top_20_skus = sku_revenue.nlargest(20).index.tolist()
    df_top_skus = df_agg[df_agg["product_id"].isin(top_20_skus)]

    daily_sku = (
        df_top_skus.groupby(["date_str", "store_id", "product_id"])
        .agg(
            total_qty=("transaction_qty", "sum"),
            total_revenue=("revenue", "sum"),
            transaction_count=("transaction_id", "count"),
        )
        .reset_index()
        .rename(columns={"date_str": "date"})
    )
    results["daily_sku"] = len(daily_sku)

    # Write all tables to SQLite
    with sqlite3.connect(str(db_path)) as conn:
        hourly_store.to_sql("hourly_store", conn, if_exists="replace", index=False)
        daily_store.to_sql("daily_store", conn, if_exists="replace", index=False)
        daily_category.to_sql("daily_category", conn, if_exists="replace", index=False)
        daily_sku.to_sql("daily_sku", conn, if_exists="replace", index=False)

    for table_name, count in results.items():
        logger.info(f"  {table_name}: {count:,} rows")

    return results


def run_pipeline(xlsx_path: str, db_path: str) -> None:
    """Run the full ingestion pipeline: load, aggregate, persist.

    Args:
        xlsx_path: Path to the Excel file.
        db_path: Path to the SQLite database.
    """
    try:
        logger.info("=" * 60)
        logger.info("ADIP Ingestion Pipeline Starting")
        logger.info("=" * 60)

        # Step 1: Load and validate
        df = load_xlsx(xlsx_path)

        # Step 2: Write raw transactions
        write_to_sqlite(df, db_path)

        # Step 3: Build and persist aggregations
        agg_results = build_aggregations(df, db_path)

        # Summary
        print("\n" + "=" * 60)
        print("ADIP INGESTION PIPELINE — SUMMARY")
        print("=" * 60)
        print(f"Source file: {xlsx_path}")
        print(f"Database: {db_path}")
        print(f"Total transactions: {len(df):,}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Total revenue: ${df['revenue'].sum():,.2f}")
        print("\nAggregation tables:")
        for table, count in agg_results.items():
            print(f"  {table}: {count:,} rows")
        print("=" * 60)

        logger.info("Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ADIP Ingestion Pipeline")
    parser.add_argument("--xlsx", default="data/raw/Afficionado_Coffee_Roasters.xlsx", help="Path to Excel file")
    parser.add_argument("--db", default="data/processed/adip.db", help="Path to SQLite database")
    args = parser.parse_args()
    run_pipeline(args.xlsx, args.db)
