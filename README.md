# Tetouan Power Consumption -- MLOps Pipeline

End-to-end ML pipeline for forecasting power consumption in Tetouan city, built incrementally from raw data to production using MLOps best practices on AWS + Databricks.

## Dataset

| Attribute | Value |
|-----------|-------|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city) |
| Records | ~52,416 (10-minute intervals, Jan--Dec 2017) |
| Target | Zone 1 Power Consumption (kW) |
| Features | Temperature, Humidity, Wind Speed, general diffuse flows, diffuse flows |
| Temporal features | hour, day_of_week, month, is_weekend (engineered from DateTime) |
| Task | Regression (forecasting) |

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Language | Python 3.12 | Core language |
| Package manager | [uv](https://docs.astral.sh/uv/) | Fast dependency management and venv creation |
| ML framework | LightGBM + scikit-learn | Gradient boosting model + pipeline |
| Experiment tracking | MLflow | Metrics, parameters, model artifacts, dataset lineage |
| Data platform | Databricks (Unity Catalog) | Delta tables, model registry, serving |
| Cloud | AWS (S3, IAM) | Storage, authentication |
| CI/CD | GitHub Actions | Linting, testing, deployment |
| Linting | Ruff + pre-commit | Code quality and formatting |
| Testing | pytest + pyspark (local) | Unit tests with local Spark |
| Task runner | [Task](https://taskfile.dev/) | One-command workflows (optional) |

## Project Phases

| Phase | Guide | Status |
|-------|-------|--------|
| 0 | [Project Foundation](../docs/00-project-foundation.md) | Done |
| 1a | [Baseline Config & Package](../docs/01a-baseline-config.md) | Done |
| 1b | [Data Processing & Testing](../docs/01b-baseline-data-processing.md) | Done |
| 1c | [Tooling & CI](../docs/01c-baseline-tooling-ci.md) | Done |
| 1d | [Databricks Validation](../docs/01d-databricks-validation.md) | In Progress |
| 2 | [Model Experimentation](../docs/02-model-experimentation.md) | Not Started |
| 3 | Feature Engineering | Not Started |
| 4 | Model Serving | Not Started |
| 5 | CI/CD Pipeline | Not Started |
| 6 | Monitoring & App | Not Started |

## Quick Start

```powershell
# Clone and enter the project
cd tetouan-power-mlops

# Create environment and install test dependencies
# IMPORTANT: Do NOT combine --extra dev and --extra test (databricks-connect conflicts with pyspark)
uv venv -p 3.12 .venv
uv sync --extra test

# Install pre-commit hooks (one-time)
uv run pre-commit install

# Run linting
uv run pre-commit run --all-files

# Run tests with coverage
uv run pytest tests/ --cov src/tetouan_power --cov-report term
```

> **Switching to notebook/Databricks work?** Use `--extra dev` instead (in a fresh venv). See [Phase 1c](../docs/01c-baseline-tooling-ci.md) for details on why the extras conflict.

## Project Structure

```text
tetouan-power-mlops/
  .github/workflows/       # CI workflow (lint + test)
  data/                    # Local dataset (gitignored)
  notebooks/               # EDA and prototyping notebooks
  scripts/                 # Pipeline scripts (run on Databricks)
    01_process_data.py     # Preprocess raw CSV -> Delta tables (Phase 1d)
  src/
    tetouan_power/         # Python package
      models/              # Model classes (Phase 2)
        basic_model.py     # sklearn pipeline + native MLflow logging
        custom_model.py    # pyfunc wrapper + code_paths packaging
      config.py            # Pydantic config loader
      data_processor.py    # Preprocessing + time-based splits + Delta writes
      utils.py             # Post-processing utilities (Phase 2)
  tests/
    catalog/               # Train/test CSVs for model tests
    fixtures/              # Pytest fixture modules
    test_data/             # Small sample CSV for DataProcessor tests
    unit_tests/            # Unit tests
  project_config.yaml      # Feature lists, parameters, split dates, experiment names
  pyproject.toml           # Package metadata, dependencies, tool config
  Taskfile.yaml            # Task runner shortcuts
  databricks.yaml          # Databricks Asset Bundle config
```

## Acknowledgments

The MLOps patterns in this project are inspired by the [End-to-end MLOps with Databricks](https://maven.com/cauchy/mlops-with-databricks) course by Maria Vechtomova and Basak Eskili.
