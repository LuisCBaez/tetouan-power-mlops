"""Conftest module."""

import platform

from tetouan_power import PROJECT_DIR

MLRUNS_DIR = PROJECT_DIR / "tests" / "mlruns"
CATALOG_DIR = PROJECT_DIR / "tests" / "catalog"
CATALOG_DIR.mkdir(parents=True, exist_ok=True)

if platform.system() == "Windows":
    TRACKING_URI = f"file:///{MLRUNS_DIR.as_posix()}"
else:
    TRACKING_URI = f"file://{MLRUNS_DIR.as_posix()}"

pytest_plugins = ["tests.fixtures.datapreprocessor_fixture"]
