# 01 — EDA Findings

## Dataset
- Local path: `data/raw/tetouan-power-consumption.csv`
- Raw columns (as provided):
  - `DateTime`
  - `Temperature`
  - `Humidity`
  - `Wind Speed`
  - `general diffuse flows`
  - `diffuse flows`
  - `Zone 1 Power Consumption`
  - `Zone 2  Power Consumption`  *(note double space)*
  - `Zone 3  Power Consumption`  *(note double space)*

## Snapshot
- Rows: **52,416**
- Columns: **9**
- Missing values: **0 total (all columns)**
- Duplicate rows: **0**

## Time index
- Parsed timestamp: **OK**
- Range:
  - min: **2017-01-01 00:00:00**
  - max: **2017-12-30 23:50:00**
- Cadence:
  - mode delta: **10 minutes**
  - delta distribution: **all steps are exactly 10 minutes** (no internal gaps)
- Note: a full year of 10-min data would be 52,560 rows (365*144). We have **52,416**, which is exactly **one day missing** (144 rows). The dataset ends on **Dec 30**, so **Dec 31 is missing**.

## Target sanity (all float, no negatives)
Descriptive stats:

- `zone1_consumption_kw`
  - min: **13,895.70**
  - mean: **32,344.97**
  - median: **32,265.92**
  - max: **52,204.40**

- `zone2_consumption_kw`
  - min: **8,560.08**
  - mean: **21,042.51**
  - median: **20,823.17**
  - max: **37,408.86**

- `zone3_consumption_kw`
  - min: **5,935.17**
  - mean: **17,835.41**
  - median: **16,415.12**
  - max: **47,598.33**

## Visual findings (high signal)
- Strong daily seasonality:
  - low around ~5–6 AM
  - peak around ~19–20 PM
- Clear seasonal pattern across the year (mid-year higher, end-year lower).
- Distribution: broad, slightly right-skewed, no extreme pathological outliers.

## Schema / naming normalization (recommended)
To avoid downstream bugs, normalize column names:
- Strip extra spaces in Zone 2/Zone 3 names.
- Convert to snake_case.
- IMPORTANT: keep the two diffuse features distinct (your notebook currently duplicates them):

Recommended mapping:
- `DateTime` → `datetime`
- `Temperature` → `temperature_c`
- `Humidity` → `humidity_pct`
- `Wind Speed` → `wind_speed`
- `general diffuse flows` → `general_diffuse_flows`
- `diffuse flows` → `diffuse_flows`
- `Zone 1 Power Consumption` → `zone1_consumption_kw`
- `Zone 2  Power Consumption` → `zone2_consumption_kw`
- `Zone 3  Power Consumption` → `zone3_consumption_kw`

## Decisions for Problem Statement v1
- Confirmed cadence: **10 minutes**
- Use time-based split (no random split).
- Start with single target: **Zone 1**
- Strongly include time features:
  - hour-of-day (cyclical sin/cos)
  - day-of-week
  - month/seasonality features
- Note dataset ends on **Dec 30** (missing Dec 31); document this explicitly.