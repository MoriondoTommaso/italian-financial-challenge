## Feature Engineering Plan (KIS) — Task 3 Revenue Forecasting

**Goal:** Amplify predictive signal for next-year `revenue_change` while avoiding leakage and over-engineering.  
**Forecasting setup:** use only information available at year `t` to predict `revenue_change_{t+1}`.

---

### 1) Baseline feature table (mandatory)
We train on a supervised table where each row corresponds to `(company_id, fiscal_year=t)` and the label is:

- **Label:** `y_next = revenue_change_{t+1}` (created via `groupby(company_id).shift(-1)`)

**Inputs (baseline):**
- All numeric + categorical features observed at year `t` (no future data)
- Replace `inf → NaN` for ratios
- Train-only preprocessing (imputation/encoding/scaling)

**Why:** data quality checks confirm no missingness in raw financial items and correct target definition; single-feature tests show modest but real signal from scale variables (e.g., production_value, assets, debt).

---

### 2) Core principle from EDA findings
EDA indicates:
- The target is heavy-tailed with many extreme events consistent with scale changes (M&A-like behavior).
- “Scale/size” variables are the strongest standalone predictors.
- Ratios are noisy in isolation but may help via interactions.

Therefore we engineer a small set of high-ROI features capturing:
1) **size/structure**,  
2) **recent growth (momentum)**,  
3) **shock propensity / volatility proxies**,  
while keeping the pipeline simple.

---

### 3) Engineered features (max ~10 extra columns)

#### 3.1 Size factor (1 feature)
**`size_score_t`**  
Compute a continuous size score (PC1/factor) using log-scaled size variables at year `t`:

- `log1p(production_value_t)`
- `log1p(total_assets_t)`
- `log1p(total_debt_t)`
- `log1p(current_assets_t)`
(optional: `signed_log1p(shareholders_equity_t)`)

**Purpose:** provide a robust “small → large” structural signal; useful for both linear and tree models.

---

#### 3.2 Growth pack (5 features + 1 flag)
Create lag2-based growth features to capture dynamics and structural changes:

- `growth_pv = log1p(production_value_t) - log1p(production_value_{t-1})`
- `growth_assets = log1p(total_assets_t) - log1p(total_assets_{t-1})`
- `growth_debt = log1p(total_debt_t) - log1p(total_debt_{t-1})`
- `growth_curr_assets = log1p(current_assets_t) - log1p(current_assets_{t-1})`
- `growth_equity = slog1p(shareholders_equity_t) - slog1p(shareholders_equity_{t-1})`
- `has_lag2` flag (1 if t-1 exists, else 0)

**Purpose:** levels alone are noisy; growth features capture “acceleration” and scale-change signatures.

---

#### 3.3 Shock propensity / volatility proxy (1–2 features)
Keep this minimal (choose one or two):

Option A (recommended):
- `abs_growth_assets = |growth_assets|`

Option B (if useful):
- `abs_growth_pv = |growth_pv|`

**Purpose:** identify firms with unstable scale evolution, which is strongly linked to extreme revenue changes.

---

### 4) Optional: year fixed effects (only if allowed/beneficial)
Add `target_year = t+1` as categorical (one-hot) to capture macro regime shifts (e.g., COVID period).
- Use **only** year labels (no external data).
- Validate that this improves 2021 generalization.

---

### 5) Anti-leakage guardrails (non-negotiable)
- All engineered features must use only year `t` and earlier values for a given company.
- Preprocessing (imputation, scaling, PCA for size_score) is **fit on train only** and applied to val/test.
- Building lags/leads may use concatenated train+test for shifting, but **fit statistics must never use future rows**.

---

### 6) Ablation plan (to prove value, 3 runs max)
We keep experimentation tight:

1) **Base:** raw features at year `t` (plus categoricals)  
2) **Base + size_score**  
3) **Base + size_score + growth pack (+ volatility proxy)**

**Selection rule:** keep engineered blocks only if they improve validation (2021) RMSE/MAE on original scale and do not worsen stability (core vs tail slices).

---
