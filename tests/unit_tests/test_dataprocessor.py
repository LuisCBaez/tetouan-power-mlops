"""Unit tests for DataProcessor."""

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from tetouan_power.config import ProjectConfig
from tetouan_power.data_processor import DataProcessor


def test_data_ingestion(sample_data: pd.DataFrame) -> None:
    """Verify sample data loaded correctly."""
    assert sample_data.shape[0] > 0
    assert sample_data.shape[1] == 9  # 9 original UCI columns


def test_dataprocessor_init(
    sample_data: pd.DataFrame,
    config: ProjectConfig,
    spark_session: SparkSession,
) -> None:
    """Verify DataProcessor stores df, config, spark."""
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    assert isinstance(processor.df, pd.DataFrame)
    assert processor.df.equals(sample_data)
    assert isinstance(processor.config, ProjectConfig)
    assert isinstance(processor.spark, SparkSession)


def test_column_transformations(
    sample_data: pd.DataFrame,
    config: ProjectConfig,
    spark_session: SparkSession,
) -> None:
    """Verify preprocessing renames columns and adds temporal features."""
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()

    # ALL original names should be gone (including double-space columns)
    assert "DateTime" not in processor.df.columns
    assert "Temperature" not in processor.df.columns
    assert "Humidity" not in processor.df.columns
    assert "Wind Speed" not in processor.df.columns
    assert "general diffuse flows" not in processor.df.columns
    assert "diffuse flows" not in processor.df.columns
    assert "Zone 1 Power Consumption" not in processor.df.columns
    assert "Zone 2  Power Consumption" not in processor.df.columns  # double space
    assert "Zone 3  Power Consumption" not in processor.df.columns  # double space

    # Renamed names should exist
    assert "datetime" in processor.df.columns
    assert "temperature" in processor.df.columns
    assert "humidity" in processor.df.columns
    assert "wind_speed" in processor.df.columns
    assert "general_diffuse_flows" in processor.df.columns
    assert "diffuse_flows" in processor.df.columns
    assert "zone1_consumption" in processor.df.columns

    # Temporal features should be generated
    assert "hour" in processor.df.columns
    assert "day_of_week" in processor.df.columns
    assert "month" in processor.df.columns
    assert "is_weekend" in processor.df.columns

    # Temporal feature value ranges
    assert processor.df["hour"].between(0, 23).all()
    assert processor.df["day_of_week"].between(0, 6).all()
    assert processor.df["month"].between(1, 12).all()
    assert set(processor.df["is_weekend"].unique()).issubset({0, 1})


def test_missing_value_handling(
    sample_data: pd.DataFrame,
    config: ProjectConfig,
    spark_session: SparkSession,
) -> None:
    """Verify nulls are dropped during preprocessing."""
    n_nulls = sample_data.isnull().sum().sum()
    assert n_nulls > 0, "sample.csv should have injected nulls to test dropna()"

    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()

    assert processor.df.isnull().sum().sum() == 0


def test_column_selection(
    sample_data: pd.DataFrame,
    config: ProjectConfig,
    spark_session: SparkSession,
) -> None:
    """Verify only expected columns remain after preprocessing."""
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()

    expected_columns = set(["datetime"] + config.num_features + config.cat_features + [config.target, "id"])
    assert set(processor.df.columns) == expected_columns


def test_split_data_default_params(
    sample_data: pd.DataFrame,
    config: ProjectConfig,
    spark_session: SparkSession,
) -> None:
    """Verify time-based split produces three non-empty sets that sum to original."""
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess()
    train, val, test = processor.split_data()

    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)

    # All rows accounted for (minus nulls dropped by preprocess)
    assert len(train) + len(val) + len(test) == len(processor.df)

    # Each split should have rows (given sample data spans all periods)
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0

    # Columns should be consistent across splits
    assert set(train.columns) == set(val.columns) == set(test.columns)

    # Temporal ordering: all train dates < all val dates < all test dates
    assert train["datetime"].max() < val["datetime"].min()
    assert val["datetime"].max() < test["datetime"].min()


def test_preprocess_empty_dataframe(
    config: ProjectConfig,
    spark_session: SparkSession,
) -> None:
    """Verify graceful error on empty input."""
    processor = DataProcessor(pandas_df=pd.DataFrame(), config=config, spark=spark_session)
    with pytest.raises(KeyError):
        processor.preprocess()
