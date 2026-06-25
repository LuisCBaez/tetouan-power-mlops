"""Fixtures for CustomModel tests."""

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from loguru import logger
from pyspark.sql import SparkSession

from tests.conftest import CATALOG_DIR, MLRUNS_DIR
from tetouan_power import PROJECT_DIR
from tetouan_power.config import ProjectConfig, Tags
from tetouan_power.models.custom_model import CustomModel

whl_file_name = None


@pytest.fixture(scope="session")
def tags() -> Tags:
    """Provide a Tags instance for all tests in the session."""
    return Tags(git_sha="test123", branch="test", job_run_id="0")


@pytest.fixture(scope="session", autouse=True)
def create_mlruns_directory() -> None:
    """Clean and recreate the local MLflow tracking directory."""
    if MLRUNS_DIR.exists():
        shutil.rmtree(MLRUNS_DIR)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created {MLRUNS_DIR} for MLflow tracking")


@pytest.fixture(scope="session", autouse=True)
def build_whl_file() -> None:
    """Build a .whl file for code_paths (session-scoped, runs once)."""
    global whl_file_name
    dist_dir = PROJECT_DIR / "dist"
    original_dir = Path.cwd()

    try:
        if dist_dir.exists():
            shutil.rmtree(dist_dir)

        os.chdir(PROJECT_DIR)
        subprocess.run(["uv", "build"], check=True, text=True, capture_output=True)

        if not dist_dir.exists():
            raise FileNotFoundError(f"dist directory not found: {dist_dir}")

        whl_file = next(
            (f.name for f in dist_dir.iterdir() if f.name.endswith(".whl")),
            None,
        )
        if not whl_file:
            raise FileNotFoundError("No .whl file found in dist/")

        whl_file_name = whl_file
    finally:
        os.chdir(original_dir)


@pytest.fixture(scope="function")
def mock_custom_model(config: ProjectConfig, tags: Tags, spark_session: SparkSession) -> CustomModel:
    """Create a CustomModel with mocked Spark (reads from tests/catalog/ CSVs)."""
    instance = CustomModel(
        config=config,
        tags=tags,
        spark=spark_session,
        code_paths=[f"{PROJECT_DIR.as_posix()}/dist/{whl_file_name}"],
    )

    train_data = pd.read_csv((CATALOG_DIR / "train_set.csv").as_posix())
    train_data = train_data.where(train_data.notna(), None)

    test_data = pd.read_csv((CATALOG_DIR / "test_set.csv").as_posix())
    test_data = test_data.where(test_data.notna(), None)

    mock_spark_df_train = MagicMock()
    mock_spark_df_train.toPandas.return_value = train_data
    mock_spark_df_test = MagicMock()
    mock_spark_df_test.toPandas.return_value = test_data

    mock_spark = MagicMock()
    mock_spark.table.side_effect = [mock_spark_df_train, mock_spark_df_test]
    instance.spark = mock_spark

    return instance
