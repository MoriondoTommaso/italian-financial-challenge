# Decisions Log — Italian Financial Challenge (Task 3)

**Project:** Italian Financial Challenge  
**Focus:** Task 3 — Revenue Forecasting (`revenue_change`)  
**Owner (Integrator):** Tommaso Moriondo  
**Team:** Tommaso, Lorenzo, Eleonora, Carla  
**Last updated:** 2026-01-21  

---

## 1) Scope (decision)
- We work **only** on **Task 3 (Revenue Forecasting)**.
- Deliverables:
  - **Final notebook**: `notebooks/final.ipynb`
  - **Slides (Canva)** exported to: `slides/final.pdf` (versioned in Git)

---

## 2) Data interpretation (confirmed)
- Unit of observation: **(company_id, fiscal_year)** = one row.
- In train, each company spans fiscal years **2018–2021**.

---

## 3) Target definition & availability (confirmed)
- Target: **`revenue_change`** = YoY % change (growth from `t-1 → t`).
- **`revenue_change` is 100% missing in 2018**.
  - 2018 is **not usable for supervised training** for this target.
  - 2018 is used only as **history** to build lag features for 2019.

---

## 4) Split & validation strategy (decision)
- We avoid random splits to prevent temporal leakage.
- Official split (time-aware holdout):
  - **History (lags only):** 2018
  - **Train (supervised):** 2019–2020 where `revenue_change` is observed
  - **Validation:** 2021 where `revenue_change` is observed
- Test years (separate file): 2022–2023

---

## 5) Leakage prevention rules (decision)
- Any transformation (imputation/encoding/scaling/clipping) is **fit only on Train (2019–2020)** and then applied to Validation/Test.
- Lag features are created using **only past years**:
  - sort by `fiscal_year` within `company_id` then `shift(1)`.
- **Operational guardrail:** all fit/apply steps must be implemented via an **sklearn Pipeline** to enforce correctness by construction.

---

## 6) Metrics (decision)
- **Primary:** RMSE  
- **Secondary:** MAPE (interpretability)  
- **Additional checks:** MAE (stability), **Directional Accuracy** (growth vs decline sign)

---

## 7) Baselines (decision)
- Baseline to beat: **ATECO sector mean** predictor for `revenue_change`.
- Optional sanity baseline: predict **0%** for all observations.

---

## 8) Business framing (decision)
- Primary persona: **CFO / FP&A**
- Value proposition: **early warning + budgeting support** (anticipate declines and adjust planning).

---

## 9) Known risks (decision)
- Extreme YoY outliers in `revenue_change` can dominate RMSE.
- Temporal drift is addressed via time-based validation (2021 holdout).

---

## 10) Lagged features selection policy (decision)
We keep feature engineering simple while data-driven by using a lightweight screening step.

**Method**
- Rank candidate features by **Spearman correlation** with `revenue_change`.

**Leakage guard**
- Spearman ranking is computed **only on Train years (2019–2020)**.
- Validation (2021) and Test (2022–2023) are **never** used for feature selection.

**Redundancy guard**
- If two features are highly redundant (e.g., collinear ratios such as `current_ratio` vs `quick_ratio`), keep only one.

**Outcome**
- We create lag-1 features only for the selected **Top-K** core subset to avoid feature explosion and preserve interpretability.

---

## 11) Target transformation selection protocol (decision)
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
- Same time split (Train 2019–2020, Val 2021; 2018 history only)
- Same features + preprocessing pipeline per model (hyperparameters fixed)
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
