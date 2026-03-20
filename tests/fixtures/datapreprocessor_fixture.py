"""DataProcessor test fixtures."""

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from tests.unit_tests.spark_config import spark_config
from tetouan_power import PROJECT_DIR
from tetouan_power.config import ProjectConfig, Tags


@pytest.fixture(scope="session")
def spark_session() -> SparkSession:
    """Create a local SparkSession for testing."""
    spark = (
        SparkSession.builder.master(spark_config.master)
        .appName(spark_config.app_name)
        .config("spark.executor.cores", spark_config.spark_executor_cores)
        .config("spark.executor.instances", spark_config.spark_executor_instances)
        .config("spark.sql.shuffle.partitions", spark_config.spark_sql_shuffle_partitions)
        .config("spark.driver.bindAddress", spark_config.spark_driver_bindAddress)
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def config() -> ProjectConfig:
    """Load project configuration from YAML."""
    config_file_path = (PROJECT_DIR / "project_config.yaml").resolve()
    return ProjectConfig.from_yaml(config_file_path.as_posix(), env="dev")


@pytest.fixture(scope="function")
def sample_data() -> pd.DataFrame:
    """Load sample CSV with original (pre-rename) column names."""
    file_path = PROJECT_DIR / "tests" / "test_data" / "sample.csv"
    return pd.read_csv(file_path.as_posix())


@pytest.fixture(scope="session")
def tags() -> Tags:
    """Dummy Tags for test runs."""
    return Tags(git_sha="abc123", branch="test", job_run_id="1")
