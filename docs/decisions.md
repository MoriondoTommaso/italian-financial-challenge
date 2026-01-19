# Decisions Log — Italian Financial Challenge (Task 3)

**Project:** Italian Financial Challenge  
**Focus:** Challenge / Task 3 — Revenue Forecasting (`revenue_change`)  
**Last updated:** 2026-01-18  
**Owner (Integrator):** Tommaso Moriondo  
**Team:** Tommaso, Lorenzo, Eleonora, Carla

---

## 1) Scope
- We work **only** on **Task 3 (Revenue Forecasting)**.
- Deliverables:
  - **One final notebook** (`notebooks/final.ipynb`)
  - **Slides in Canva**, exported to `slides/final.pdf` (versioned in Git)

---

## 2) Data interpretation (confirmed)
- Unit of observation: **(company_id, fiscal_year)** = one row.
- In **train**, each company appears **4 times** with fiscal years **2018–2021**.

---

## 3) Target definition and availability (confirmed)
- Target: **`revenue_change`**, interpreted as the **YoY % change** for year `t` (growth from `t-1 → t`).
- **`revenue_change` in 2018 is 100% missing** in the training data.
  - Implication: **2018 cannot be used for supervised training** on this target.
  - 2018 can still be used as **history** to build lagged features for 2019.

---

## 4) Alignment with challenge protocol (decision)
- We stay aligned with the dataset format:
  - Predict **`revenue_change_t`** for each row/year `t` using features from year `t` plus strictly past information (lags).

---

## 5) Validation strategy (decision)
- We **avoid random splits** to prevent temporal leakage.
- Official split:
  - **Train (supervised):** fiscal_year ∈ **{2019, 2020}** and `revenue_change` not missing
  - **Validation:** fiscal_year = **2021** and `revenue_change` not missing
  - **2018:** used only to create lag features

---

## 6) Leakage prevention rules (decision)
- Any transformation (imputation/encoding/scaling) is **fit only on the training split** (2019–2020) and then applied to validation/test.
- Lag features must be created using **only past years** (e.g., `groupby(company_id).shift(1)` after sorting by `fiscal_year`).

---

## 7) Metrics (decision)
- **Primary:** RMSE
- **Secondary:** MAPE (interpretability)
- **Additional checks:** MAE (stability) and **Directional Accuracy** (growth vs decline sign)

---

## 8) Baselines (decision)
- Baseline to beat: **Sector (ATECO) mean** predictor for `revenue_change`.
- Optional sanity baseline: **predict 0%** for all observations.

---

## 9) Business framing (decision)
- Primary end-user persona: **CFO / FP&A**
- Main value proposition: **early warning + actionable budgeting support** (anticipate revenue declines and adjust planning).

---

## 10) Known risks (decision)
- **Outliers / extreme YoY changes** in `revenue_change` can dominate error metrics.
- **Temporal drift** risk is mitigated via time-based validation.

---

## 11) Open decisions (to finalize next)
- Lag strategy:
  - Option A: lag **all numeric features**
  - Option B: lag a **core subset** of financially meaningful features
- Outlier handling strategy:
  - Model-robust (e.g., Huber) vs explicit winsorization/transformations

---
