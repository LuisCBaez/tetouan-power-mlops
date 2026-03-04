# 00 — Problem Statement (Version 0)

## Status
Version: v0 (Pre-EDA Hypothesis)  
Last Updated: March 3, 2026  
Owner: LuisCa

---

## 1. Context

We have received a dataset containing historical power consumption measurements for three distribution zones in Tetouan city, along with weather and environmental variables.

This document defines the **initial business framing and technical assumptions** before inspecting the data in detail.

These definitions will be validated and potentially revised after first contact with the data.

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

The dataset includes:

- `Zone 1 Power Consumption`
- `Zone 2 Power Consumption`
- `Zone 3 Power Consumption`

### Initial Scope Decision

To reduce complexity:

- Start with **Zone 1 only**
- Treat as a **single-target regression problem**
- Extend to multi-target forecasting after baseline stability

### Pipeline Name

Rename:

`Zone 1 Power Consumption` → `zone1_consumption`

Reason:
- snake_case
- no spaces
- pipeline-friendly

---

## 4. Prediction Setup (Hypothesis)

To be validated during EDA:

- Forecast horizon: 1-step ahead (short-term)
- Data frequency: assumed sub-hourly or hourly
- Train/test split: time-based split (not random)

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
| Reproducibility | All experiments reproducible via config + code version + data version | Core MLOps requirement |
| Cost | Minimal viable compute in dev; ephemeral job clusters in production | Avoid unnecessary cloud spend |
| Environment Parity | Same code across dev/acc/prd; config-driven differences only | Prevent drift between environments |
| Latency | Batch inference acceptable initially | Reduce architectural complexity early |

---

## 7. Assumptions About Data (Pre-EDA)

These assumptions must be validated:

- Timestamp column exists and is parseable.
- No major gaps in time series.
- Weather features are available at prediction time (no leakage).
- No negative consumption values.
- Data covers full-year seasonal variation.

---

## 8. Risks Identified Early

- Data leakage via future weather features.
- Irregular timestamp intervals.
- Missing blocks of time.
- Strong seasonality requiring feature engineering.
- Multi-zone correlations complicating independent modeling.

---

## 9. Next Step

Proceed to:

`notebooks/00_initial_eda.ipynb`

Goal: Validate or revise assumptions above.

After EDA, update this document to **Version 1**.