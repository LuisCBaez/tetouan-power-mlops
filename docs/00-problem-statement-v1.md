# 00 — Problem Statement (Version 1)

## Status
Version: v1 (Post-EDA Validated)  
Last Updated: March 4, 2026  
Owner: LuisCa

---

## 1. Context

We received a time-series dataset of **power consumption** for three distribution zones in Tetouan city, plus weather/environment variables. We performed an initial “first contact” EDA to validate data usability before building pipeline code.

**Validated dataset facts (from EDA):**
- Time coverage: **2017-01-01 00:00:00 → 2017-12-30 23:50:00**
- Cadence: **10-minute intervals** (consistent)
- Missing values: **none**
- Duplicates: **none**
- Note: **Dec 31 is not present** (dataset ends Dec 30). There are no internal cadence gaps.

---

## 2. Business Objective

### Primary Question

Can we forecast short-term power consumption for distribution zones in Tetouan city using weather and temporal features?

### Intended Use Case
- Support operational planning for energy distribution.
- Improve short-term load forecasting accuracy.
- Provide a reproducible ML pipeline template for future forecasting projects.

---

## 3. Target Definition (v0)

The dataset contains three targets:
- `Zone 1 Power Consumption`
- `Zone 2  Power Consumption` *(note: double space)*
- `Zone 3  Power Consumption` *(note: double space)*

### Initial Scope Decision
To reduce complexity:
- Start with **Zone 1 only**
- Treat as a **single-target regression problem**
- Extend to multi-target forecasting after baseline stability

### Pipeline Name
Normalize column names to avoid downstream bugs (spaces, inconsistent casing). Recommended:
- `Zone 1 Power Consumption` → `zone1_consumption_kw`
- `Zone 2  Power Consumption` → `zone2_consumption_kw`
- `Zone 3  Power Consumption` → `zone3_consumption_kw`

Reason:
- snake_case
- consistent units in name
- stable for config-driven pipelines

---

## 4. Prediction Setup (v1)

### Data Frequency (validated)
- **10-minute cadence** (mode delta = 10 minutes)

### Baseline Forecast Task (v1)
Start with a simple, unambiguous baseline objective:
- **One-step ahead** regression at native cadence:
  - predict `zone1_consumption_kw(t+10min)` given features at time `t`

This gives a clean baseline. Later, we can add:
- multi-step horizons (e.g., 1h ahead)
- aggregation (hourly) as a config option

### Split Strategy (v1)
Use a **time-based split** (no random split). Recommended:
- Train: **2017-01 → 2017-09**
- Validation: **2017-10 → 2017-11**
- Test: **2017-12 → 2017-12-30**

Rationale:
- preserves seasonality structure
- avoids leakage
- creates a realistic “future holdout”

---

## 5. Success Metrics (KPIs)

| Metric | Role | Initial Target |
|--------|------|----------------|
| MAE | Primary business metric | < 5000 kW |
| RMSE | Secondary stability metric | Tracked |
| R² | Overall fit quality | > 0.70 |

### Why MAE?

- Same unit as target (kW)
- Interpretable by non-technical stakeholders
- Robust to extreme values compared to RMSE

Thresholds will be refined after baseline model results.

---

## 6. Constraints

| Constraint | Description | Rationale |
|------------|------------|------------|
| Reproducibility | Experiments reproducible via config + code SHA + data version | Core MLOps requirement |
| Cost | Minimal viable compute; avoid idle clusters | Control cloud spend |
| Environment Parity | Same code across dev/acc/prd; config-only differences | Prevent drift between environments |
| Latency | Batch inference acceptable initially | Reduce architectural complexity early |

---

## 7. Data Reality (v1) — What We Now Know

### Confirmed Seasonality
EDA plots show strong patterns:
- **daily** cycle (clear hour-of-day structure)
- **seasonal** variation across the year (mid-year higher)

This implies time features are mandatory:
- hour-of-day (cyclical sin/cos)
- day-of-week
- month/seasonality features (or cyclical month encoding)

### Feature Columns (raw)
Weather/environment features include:
- `Temperature`
- `Humidity`
- `Wind Speed`
- `general diffuse flows`
- `diffuse flows`

**Important naming note:**  
Keep `general diffuse flows` and `diffuse flows` distinct when normalizing names (do not map both to the same target name).


---

## 8. Risks Identified Early

- **Data leakage risk**: ensure all features are available at prediction time (especially if any “flows” are derived from future measurements).
- **Seasonality dependence**: models must capture strong hourly and seasonal effects.
- **End-of-year missing day**: dataset ends Dec 30 (Dec 31 missing). Documented and handled by time split.
- **Multi-zone coupling**: zones likely correlated; defer multi-output modeling until baseline is stable.


---

## 9. Next Step

1) Write `docs/01-eda-findings.md` (summary + confirmed schema + any quirks)  
2) Proceed: minimal data loader + schema checks + baseline model objective (t+10min)  
3) Convert naming normalization into a repeatable data contract step (bronze→silver)
