"""Tests for MLflow-related pip dependency helpers."""

import importlib.metadata
import re

from tetouan_power.mlflow_pip_deps import pyspark_pip_requirement


def test_pyspark_pip_requirement_matches_installed_distribution() -> None:
    """Logged model env should pin PySpark to the same version as the active venv."""
    expected_version = importlib.metadata.version("pyspark")
    spec = pyspark_pip_requirement()
    assert spec == f"pyspark=={expected_version}"
    assert re.fullmatch(r"pyspark==\d+\.\d+\.\d+", spec) is not None
