"""Tetouan power consumption package."""

import importlib.metadata
from pathlib import Path

THIS_DIR = Path(__file__).parent  # src/tetouan_power/
PROJECT_DIR = (THIS_DIR / "../..").resolve()  # project root (tetouan-power-mlops/)


def get_version() -> str:
    """Return the package version from pyproject.toml."""
    try:
        return importlib.metadata.version("tetouan-power-mlops")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0.dev"


__version__ = get_version()
