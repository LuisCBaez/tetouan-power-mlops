"""Pip requirement strings for MLflow model environments.

The PySpark version in ``mlflow.pyfunc.log_model(..., conda_env=...)`` must stay aligned with
the environment that trains the model: local tests use ``pyspark`` from ``--extra test``;
Databricks jobs use the cluster runtime's PySpark. Hard-coding ``pyspark==x.y.z`` in model
code drifts from ``pyproject.toml`` and from the cluster. This module derives the pin from the
*installed* ``pyspark`` distribution at log time.
"""

from __future__ import annotations

import importlib.metadata

__all__ = ["pyspark_pip_requirement"]


def pyspark_pip_requirement() -> str:
    """Return a ``pyspark==<version>`` string for the active Python environment.

    Use this when building ``additional_pip_deps`` for ``mlflow.pyfunc.log_model`` so the
    logged environment matches the interpreter that serialized the model (CI venv, notebook,
    or job cluster).

    Returns:
        Pip requirement string, e.g. ``pyspark==3.5.5``.

    Raises:
        PackageNotFoundError: If ``pyspark`` is not installed (e.g. bare ``uv sync`` without
            ``--extra test``, or a misconfigured driver). Install the test extra locally or run
            training on a cluster where PySpark is present.

    Note:
        Databricks Model Serving often provides Spark at runtime; the pin still documents the
        build-time stack and keeps optional offline installs reproducible. If your serving image
        must differ from training, override with a job-specific env or omit PySpark from serving
        envs when the pyfunc path does not need it.
    """
    version = importlib.metadata.version("pyspark")
    return f"pyspark=={version}"
