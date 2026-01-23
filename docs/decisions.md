# Decisions Log — Italian Financial Challenge (Task 3)

**Project:** Italian Financial Challenge  
**Focus:** Task 3 — Revenue Forecasting (`revenue_change`)  
**Owner (Integrator):** Tommaso Moriondo  
**Team:** Tommaso, Lorenzo, Eleonora, Carla  
**Last updated:** 2026-01-22  

---

## 1) Scope (decision)
- We work **only** on **Task 3 (Revenue Forecasting)**.
- Deliverables:
  - **Final notebook**: `notebooks/final.ipynb`
  - **Slides (Canva)** exported to: `slides/final.pdf` (versioned in Git)

---

## 2) Data interpretation (confirmed)
- Unit of observation: **(company_id, fiscal_year)** = one row.
- In train, companies span fiscal years **2018–2021** (near-balanced panel; no temporal gaps).

---

## 3) Target definition & availability (confirmed)
- Target: **`revenue_change_t`** = YoY % change (growth from `t-1 → t`).
- **`revenue_change` is 100% missing in 2018** (first-year effect).
  - 2018 cannot be a **target year** (no supervised label), but it is usable as a **predictor-year** for forecasting 2019.

---

## 4) Problem formulation (decision) — Next-year forecasting
**Decision:** We frame the task as **next fiscal year forecasting**:
- Predict **`revenue_change_t`** using only predictors available at **`t−1`**: **X(t−1) → y(t)**.

**Rationale:**
- Using same-year features X(t) makes `revenue_change_t` **directly computable** from `production_value_t` and `production_value_{t-1}` (target-derived setup).  
- The CFO/FP&A use case is to forecast **before** year `t` closes.

**Operational implication:**
- For each target-year `t`, the model inputs are the previous-year snapshot `t−1` (features suffixed or mapped as `_prev1`).

---

## 5) Split & validation strategy (decision)
- We avoid random splits to prevent temporal leakage.
- Official split is defined on **target-years** (y-year):
  - **Train target-years:** 2019–2020
  - **Validation target-year:** 2021
- Inputs are **feature-years (X-year) = target-year − 1**:
  - **X_train years:** 2018–2019  → y_train years 2019–2020
  - **X_val year:** 2020         → y_val year 2021
- Test years (separate file): 2022–2023 are used for **scoring target-years**:
  - Predict y_2022 using X_2021
  - Predict y_2023 using X_2022

---

## 6) Leakage prevention rules (decision)
- Any transformation (imputation/encoding/scaling/clipping) is **fit only on X_train (feature-years 2018–2019)** and then applied to X_val / X_test.
- **No target-derived inputs:** we do not use any same-year combination that can reconstruct `revenue_change_t` by definition (e.g., `production_value_t` for predicting `revenue_change_t`).
- Prev-year alignment is created using **only past years**:
  - sort by `fiscal_year` within `company_id` then `shift(1)` to build X(t−1).
- **Operational guardrail:** all fit/apply steps must be implemented via an **sklearn Pipeline** to enforce correctness by construction.

---

## 7) Metrics (decision)
- **Primary:** RMSE  
- **Secondary:** MAPE (interpretability)  
- **Additional checks:** MAE (stability), **Directional Accuracy** (growth vs decline sign)

---

## 8) Baselines (decision)
- Baseline to beat: **ATECO sector mean** predictor for `revenue_change` (computed on train target-years only).
- Optional sanity baseline: predict **0%** for all observations.

---

## 9) Business framing (decision)
- Primary persona: **CFO / FP&A**
- Value proposition: **early warning + budgeting support** (anticipate declines and adjust planning).

---

## 10) Known risks (decision)
- Extreme YoY outliers in `revenue_change` can dominate RMSE.
- Temporal drift is addressed via time-based validation (2021 target-year holdout).
- Forecasting with X(t−1) is inherently harder than post-hoc estimation; improvements should be judged vs baselines.

---

## 11) Prev-year features selection policy (decision)
We keep feature engineering simple while data-driven by using a lightweight screening step.

**Method**
- Rank candidate predictors using **Spearman correlation** on **training pairs**: **X(t−1) vs y(t)** for train target-years (2019–2020).

**Leakage guard**
- Spearman ranking is computed **only on train target-years** (2019–2020) using their aligned X-years (2018–2019).
- Validation (target-year 2021) and Test (target-years 2022–2023) are **never** used for feature selection.

**Redundancy guard**
- If two features are highly redundant (e.g., collinear ratios such as `current_ratio` vs `quick_ratio`), keep only one.

**Outcome**
- We use a compact Top-K predictor set (from X(t−1)) to preserve interpretability and avoid feature explosion.

---

## 12) Target transformation selection protocol (decision)
We select the best target transformation via a small controlled experiment to stabilize learning under heavy tails.

### Candidate transformations
1) **Raw:** `y = revenue_change`
2) **Signed-log:** `y = sign(y) * log1p(|y|)`  
   - invert: `y = sign(z) * expm1(|z|)`
3) **Asinh:** `y = asinh(y)`  
   - invert: `y = sinh(z)`

### Baseline models used
- **Linear regression baseline** (regularized: Ridge/ElasticNet)
- **LightGBM regressor**

### Setup (fixed across variants)
- Forecast formulation fixed: **X(t−1) → y(t)**
- Time split fixed on **target-years**:
  - Train target-years 2019–2020 (X-years 2018–2019)
  - Validation target-year 2021 (X-year 2020)
- Same preprocessing pipeline per model (hyperparameters fixed)
- Only target transformation changes

### Decision rule
- Pick the transformation with the **lowest RMSE on 2021**, computed on the **original scale** after inversion.
- If RMSE differences are small (~1–2%):
  1) lower **p95 Absolute Error**
  2) higher **Directional Accuracy**

### If models disagree
- If one transform wins for both models → select it.
- Otherwise select the winner for the **final model family** and document the trade-off.

### Freeze
- Run once, log results, and freeze to avoid validation overfitting.
