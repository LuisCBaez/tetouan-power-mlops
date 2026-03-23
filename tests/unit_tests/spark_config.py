"""Spark Configuration module for local testing."""

from pydantic_settings import BaseSettings


class SparkConfig(BaseSettings):
    """Local Spark settings for unit tests."""

    master: str = "local[1]"
    app_name: str = "local_test"
    spark_executor_cores: str = "1"
    spark_executor_instances: str = "1"
    spark_sql_shuffle_partitions: str = "1"
    spark_driver_bindAddress: str = "127.0.0.1"


spark_config = SparkConfig()
