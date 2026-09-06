# Tetouan Power MLOps — Problem Statement

## Project objective

Build a reproducible end-to-end machine learning lifecycle on Databricks using Tetouan power-consumption data as the working regression use case.

The primary outcome is an MLOps reference implementation that demonstrates:

- configuration-driven development across local and Databricks environments;
- tested data preprocessing and chronological dataset splits;
- experiment tracking and dataset lineage with MLflow;
- model packaging and registration in Unity Catalog;
- Databricks Feature Engineering for reusable training and inference features; and
- incremental delivery through Git branches, pull requests, and CI checks.

Model-serving and later production phases have not started.

## Dataset and ML use case

The UCI dataset contains weather measurements and power consumption for three distribution zones in Tetouan, Morocco.

Validated facts:

- 52,416 observations from January 1 through December 30, 2017;
- a consistent ten-minute observation cadence with no internal gaps;
- no missing values or duplicate rows; and
- strong daily and seasonal demand patterns.

Zone 1 is the only modeled target in the current scope. Zones 2 and 3 remain available for later experiments.

## Implemented data contract

The preprocessing pipeline normalizes the raw columns as follows:

| Raw column | Implemented column |
|---|---|
| `DateTime` | `datetime` |
| `Temperature` | `temperature` |
| `Humidity` | `humidity` |
| `Wind Speed` | `wind_speed` |
| `general diffuse flows` | `general_diffuse_flows` |
| `diffuse flows` | `diffuse_flows` |
| `Zone 1 Power Consumption` | `zone1_consumption` |
| `Zone 2  Power Consumption` | `zone2_consumption` |
| `Zone 3  Power Consumption` | `zone3_consumption` |

The implemented temporal features are `hour`, `day_of_week`, `month`, and `is_weekend`. The `id` column is the string representation of `datetime` and is used as the feature lookup key.

## Current model contract

The current task is supervised regression:

`zone1_consumption(t) = f(weather(t), calendar_features(t))`

Each feature row and its target come from the same timestamp. The ten-minute cadence describes how often observations occur; it does not make the current target ten minutes into the future.

Accordingly, the current models estimate demand for the timestamp represented by their inputs. They do not implement a genuine one-step-ahead forecast.

The three model variants share this contract:

- `BasicModel` logs a native scikit-learn pipeline with MLflow;
- `CustomModel` packages the pipeline in an MLflow pyfunc wrapper; and
- `FeatureLookUpModel` assembles inputs through Databricks Feature Engineering before applying the regression model.

Feature lookup changes how inputs are obtained. It does not change the target horizon.

## Evaluation and data splitting

The project uses chronological splits to preserve temporal order:

- train: January through September 2017;
- validation: October through November 2017; and
- test: December 1 through December 30, 2017.

MAE is the primary model-quality metric because it is expressed in the same units as the target. RMSE, MSE, and R² are also logged to compare experiments. These metrics support the MLOps demonstration; forecasting research and accuracy optimization are not the primary project objective.

## Scope boundaries

The current implementation intentionally excludes:

- multi-zone or multi-target modeling;
- lag and rolling-window features;
- a shifted future target;
- model serving and online feature serving; and
- production monitoring.

A genuine ten-minute-ahead forecast would be a separate modeling improvement. It would require a future target created with an explicit shift or timestamp join, gap validation, split boundaries based on the forecast timestamp, regenerated Delta tables, and retraining of every model variant.

## Public implementation path

- [EDA findings](01-eda-findings.md)
- [Phase 1 Databricks validation](../notebooks/01_databricks_validation_demo.py)
- [Phase 2 model experimentation](../notebooks/02_model_experimentation_demo.py)
- [Phase 3 feature engineering](../notebooks/03_feature_engineering_demo.py)
