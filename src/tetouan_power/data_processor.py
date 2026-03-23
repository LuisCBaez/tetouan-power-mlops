"""Data preprocessing for Tetouan power consumption."""

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp

from tetouan_power.config import ProjectConfig

COLUMN_RENAME = {
    "DateTime": "datetime",
    "Temperature": "temperature",
    "Humidity": "humidity",
    "Wind Speed": "wind_speed",
    "general diffuse flows": "general_diffuse_flows",
    "diffuse flows": "diffuse_flows",
    "Zone 1 Power Consumption": "zone1_consumption",
    "Zone 2  Power Consumption": "zone2_consumption",
    "Zone 3  Power Consumption": "zone3_consumption",
}


class DataProcessor:
    """Preprocess Tetouan power data and split by time."""

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        """Initialize DataProcessor with raw data, config, and Spark session."""
        self.df = pandas_df
        self.config = config
        self.spark = spark

    def preprocess(self) -> None:
        """Mutate self.df in place: rename, parse, add temporal features, select columns, generate id."""
        # Rename columns
        self.df = self.df.rename(columns=COLUMN_RENAME)
        self.df["datetime"] = pd.to_datetime(self.df["datetime"])

        # Add temporal features
        self.df["hour"] = self.df["datetime"].dt.hour
        self.df["day_of_week"] = self.df["datetime"].dt.dayofweek
        self.df["month"] = self.df["datetime"].dt.month
        self.df["is_weekend"] = (self.df["day_of_week"] >= 5).astype(int)

        # Handle missing values
        if self.df.isnull().sum().sum() > 0:
            self.df = self.df.dropna()

        # Select columns
        relevant_columns = ["datetime"] + self.config.num_features + self.config.cat_features + [self.config.target]
        self.df = self.df[relevant_columns].copy()

        # Generate id
        self.df["id"] = self.df["datetime"].astype(str)

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Time-based split. Returns (train, val, test)."""
        # Check if split is configured
        if self.config.split is None:
            raise ValueError("config.split is required for time-based split")

        # Get train and val end dates
        train_end = self.config.split.train_end
        val_end = self.config.split.val_end

        # Split data
        train_set = self.df[self.df["datetime"] < train_end]
        val_set = self.df[(self.df["datetime"] >= train_end) & (self.df["datetime"] < val_end)]
        test_set = self.df[self.df["datetime"] >= val_end]

        return train_set, val_set, test_set

    def save_to_catalog(
        self,
        train_set: pd.DataFrame,
        val_set: pd.DataFrame,
        test_set: pd.DataFrame,
    ) -> None:
        """Write train, val, test to Delta tables with update_timestamp_utc."""
        # Create base path
        base = f"{self.config.catalog_name}.{self.config.schema_name}"  # mlops_dev.tetouan_power

        # Write train, val, test to Delta tables
        for name, df in [("train_set", train_set), ("val_set", val_set), ("test_set", test_set)]:
            spark_df = self.spark.createDataFrame(df).withColumn(
                "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
            )
            spark_df.write.mode("overwrite").saveAsTable(f"{base}.{name}")

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train, val, test set tables."""
        base = f"{self.config.catalog_name}.{self.config.schema_name}"
        for name in ["train_set", "val_set", "test_set"]:
            self.spark.sql(f"ALTER TABLE {base}.{name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
