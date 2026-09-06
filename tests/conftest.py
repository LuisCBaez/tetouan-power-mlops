"""Conftest module."""

import mlflow
import pytest

from tetouan_power import PROJECT_DIR

CATALOG_DIR = PROJECT_DIR / "tests" / "catalog"
CATALOG_DIR.mkdir(parents=True, exist_ok=True)

pytest_plugins = [
    "tests.fixtures.datapreprocessor_fixture",
    "tests.fixtures.custom_model_fixture",
]


@pytest.fixture(scope="session", autouse=True)
def configure_mlflow_tracking(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Configure temporary SQLite tracking and registry backends."""
    tracking_db = tmp_path_factory.mktemp("mlflow") / "tracking.db"
    sqlite_uri = f"sqlite:///{tracking_db.as_posix()}"

    mlflow.set_tracking_uri(sqlite_uri)
    mlflow.set_registry_uri(sqlite_uri)
