# Tetouan Power Consumption — MLOps Pipeline

End-to-end ML pipeline for forecasting power consumption in Tetouan city, built incrementally from raw data to production using MLOps best practices on AWS + Databricks.

## Dataset

| Attribute | Value |
|-----------|-------|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city) |
| Records | ~52,416 (10-minute intervals, Jan–Dec 2017) |
| Target | Zone 1 Power Consumption (kW) |
| Features | Temperature, Humidity, Wind Speed, general diffuse flows, diffuse flows |
| Task | Regression (forecasting) |

## Project Phases

| Phase | Guide | Status |
|-------|-------|--------|
| 0 | [Project Foundation](../docs/00-project-foundation.md) | Done |
| 1 | [Baseline Pipeline](../docs/01-baseline-pipeline.md) | In Progress |
| 2 | [Model Experimentation](../docs/02-model-experimentation.md) | Not Started |
| 3 | Feature Engineering | Not Started |
| 4 | Model Serving | Not Started |
| 5 | CI/CD Pipeline | Not Started |
| 6 | Monitoring & App | Not Started |

## Quick Start

```powershell
# Clone and enter the project
cd tetouan-power-mlops

# Create environment and install all dependencies
uv venv -p 3.12 .venv
uv sync --extra dev --extra test

# Run linting
uv run pre-commit run --all-files

# Run tests
uv run pytest tests/ --cov src/tetouan_power --cov-report term
```

## Project Structure

```text
tetouan-power-mlops/
  data/                    # Local dataset (gitignored)
  docs/                    # Problem statements, EDA findings
  notebooks/               # Exploration and Databricks notebooks
  scripts/                 # Pipeline scripts (run on Databricks)
  src/
    tetouan_power/         # Python package
      models/              # Model classes (Phase 2+)
      serving/             # Serving classes (Phase 4+)
  tests/
    fixtures/              # Test fixtures
    unit_tests/            # Unit tests
    integration_tests/     # Integration tests (Phase 3+)
    test_data/             # Small test CSVs
```

## Acknowledgments

The MLOps patterns in this project are inspired by the [End-to-end MLOps with Databricks](https://maven.com/cauchy/mlops-with-databricks) course by Maria Vechtomova and Basak Eskili.
