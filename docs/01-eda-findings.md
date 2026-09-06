# Tetouan Power Consumption — EDA Findings

## Dataset snapshot

- Source file: `data/raw/tetouan-power-consumption.csv`
- Rows: 52,416
- Raw columns: 9
- Missing values: 0
- Duplicate rows: 0
- Time range: `2017-01-01 00:00:00` through `2017-12-30 23:50:00`
- Cadence: every 10 minutes, with no internal cadence gaps

A complete 365-day year at this cadence would contain 52,560 rows. The 144-row difference is December 31, which is outside the dataset range rather than an internal missing block.

## Raw schema

- `DateTime`
- `Temperature`
- `Humidity`
- `Wind Speed`
- `general diffuse flows`
- `diffuse flows`
- `Zone 1 Power Consumption`
- `Zone 2  Power Consumption`
- `Zone 3  Power Consumption`

Zone 2 and Zone 3 contain two spaces in their raw column names.

## Implemented normalized schema

The public implementation uses the following mapping from `DataProcessor.COLUMN_RENAME`:

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

The two diffuse-flow measurements remain distinct.

## Target sanity

All three consumption columns are numeric and contain no negative values.

| Target | Minimum | Mean | Median | Maximum |
|---|---:|---:|---:|---:|
| `zone1_consumption` | 13,895.70 | 32,344.97 | 32,265.92 | 52,204.40 |
| `zone2_consumption` | 8,560.08 | 21,042.51 | 20,823.17 | 37,408.86 |
| `zone3_consumption` | 5,935.17 | 17,835.41 | 16,415.12 | 47,598.33 |

The current modeling scope uses only `zone1_consumption` as its target.

## Temporal findings

- Demand has a strong daily pattern, with lower values around 05:00–06:00 and higher values around 19:00–20:00.
- Demand also varies across the year.
- The target distribution is broad and slightly right-skewed, without pathological outliers.

The implementation derives four direct calendar features from `datetime`:

- `hour`
- `day_of_week`
- `month`
- `is_weekend`

Cyclical encodings, lag features, and rolling statistics were considered during EDA but are not part of the current implementation.

## Decisions carried into the implementation

- Preserve the native ten-minute observations.
- Use chronological train, validation, and test splits instead of a random split.
- Model Zone 1 as a single regression target.
- Keep the weather variables and implemented calendar features in the model input.
- Treat the target as same-timestamp demand; no future target shift is currently created.
- Use the dataset to demonstrate an end-to-end Databricks MLOps lifecycle rather than expand the modeling scope.

See the canonical [problem statement](00-problem-statement.md) for the current project objective and scope.
