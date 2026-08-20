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
| Experiment tracking | MLflow 3.x | Metrics, parameters, model artifacts, dataset lineage (Unity Catalog registry) |
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
| 1d | [Databricks Validation](../docs/01d-databricks-validation.md) | Done |
| 2 | [Model Experimentation](../docs/02-model-experimentation.md) | Done |
| 3 | [Feature Engineering](../docs/03-feature-engineering.md) | Done |
| 4 | Model Serving | Not Started |
| 5 | CI/CD Pipeline | Not Started |
| 6 | Monitoring & App | Not Started |

## Dependencies note

MLflow **3.10.x** on PyPI currently requires **pandas 2.x** and **`pyarrow<24`**; `pyproject.toml` is pinned accordingly so `uv lock` stays consistent. When MLflow relaxes those caps, bump versions there and re-lock.

Logged model environments derive **`pyspark==...`** from the installed `pyspark` package (`tetouan_power.mlflow_pip_deps`) so the pin tracks `pyproject.toml` / cluster runtime instead of a stale hard-coded string.

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

## Development Workflow (branch -> PR -> green CI -> merge)

`main` is protected by the `no-commit-to-branch` pre-commit hook. **Never commit or merge directly to `main` locally.** Every change goes through a pull request so CI runs and produces a green check before code reaches `main`.

```powershell
git checkout main
git pull origin main
git checkout -b feature/<name>     # use -b only when the branch is NEW

# ...work, then...
git add .
git commit -m "..."                # pre-commit runs; passes because you are not on main
git push -u origin feature/<name>
```

Then on GitHub: **Pull requests -> New pull request** -> base `main` <- compare `feature/<name>` -> wait for the **green CI check on the PR** -> **Merge pull request**. Finally sync local main:

```powershell
git checkout main
git pull origin main
```

> **Do not** run `git merge feature/...` on `main` locally. That produces a `push` event on `main` (CI runs as `main`, where the `no-commit-to-branch` hook fails) and you never get the green `pull_request` check. Open a PR and let CI go green there instead.

## Project Structure

```text
tetouan-power-mlops/
  .github/workflows/           # CI workflow (lint + test)
  data/                        # Local dataset (gitignored)
  notebooks/                   # EDA, prototyping, and interactive Databricks demos
    00_initial_eda.ipynb
    01_preprocessing_prototype.ipynb
    03_feature_engineering_demo.py # Serverless feature engineering demo (Phase 3)
  scripts/                     # Pipeline scripts (run on Databricks)
    01_process_data.py         # Preprocess raw CSV -> Delta tables (Phase 1d)
    02_train_register_model.py # Train + register CustomModel (Phase 2)
    03_train_register_fe_model.py # Train + register feature-aware model (Phase 3)
  src/
    tetouan_power/             # Python package
      models/                  # Model classes (Phases 2-3)
        basic_model.py         # sklearn pipeline + native MLflow logging
        custom_model.py        # pyfunc wrapper + code_paths packaging
        feature_lookup_model.py # UC feature lookups, UDF, training, and scoring (Phase 3)
      config.py                # Pydantic config loader
      data_processor.py        # Preprocessing + time-based splits + Delta writes
      mlflow_pip_deps.py       # PySpark pip pin for logged model envs (Phase 2)
      utils.py                 # Post-processing utilities (Phase 2)
  tests/
    catalog/                   # Train/test CSVs for model tests (Phase 1d / 2)
    fixtures/                  # Pytest fixture modules
      datapreprocessor_fixture.py
      custom_model_fixture.py  # CustomModel fixtures + .whl build (Phase 2)
    test_data/                 # Small sample CSV for DataProcessor tests
    unit_tests/                # Unit tests
      test_dataprocessor.py    # DataProcessor tests (Phase 1b)
      test_custom_model.py     # CustomModel lifecycle tests (Phase 2)
      test_feature_lookup_model.py # Feature lookup SQL and lazy-client tests (Phase 3)
      test_utils.py            # Post-processing tests (Phase 2)
      test_mlflow_pip_deps.py  # PySpark pin guard (Phase 2)
      spark_config.py          # Local Spark settings for tests
    conftest.py                # MLflow tracking URI + pytest plugin registration
  project_config.yaml          # Feature lists, parameters, split dates, experiment names
  pyproject.toml               # Package metadata, dependencies, tool config
  Taskfile.yaml                # Task runner shortcuts
  databricks.yaml              # Databricks Asset Bundle config
```

## Acknowledgments

The MLOps patterns in this project are inspired by the [End-to-end MLOps with Databricks](https://maven.com/cauchy/mlops-with-databricks) course by Maria Vechtomova and Basak Eskili.
