# Roadmap Modeling — Italian Financial Challenge (Task 3: Revenue Forecasting)

**Goal:** Predict next-year `revenue_change` (YoY % change in `production_value`) using prior-year financial snapshots.  
**Key constraints:** Time-aware validation, no leakage, heavy-tailed target with ~50% negatives → robust modeling.

---

## 0) Problem framing (what we’re actually doing)
- **Unit:** (company_id, fiscal_year)
- **Forecasting rule:** predict **year t** target using **only year t−1** information.
  - **X(t)** = features at **t−1**
  - **y(t)** = `revenue_change` at **t**

✅ This prevents “using the same-year revenues to compute the target”.

---

## 1) Data splits (time-aware holdout)
We evaluate strictly forward in time.

- **History-only:** 2018 (target missing by definition) → used only to build lags.
- **Train (supervised):** target years **2019–2020**
- **Validation:** target year **2021**
- **Test (Kaggle submission):** target years **2022–2023** (`test_features.csv`)

**Leakage guardrail:**
- All preprocessors (imputer/scaler/encoder/PCA/cluster) **fit only on train**, then applied to val/test.

---

## 2) Build the supervised feature table (core step)
### 2.1 Create lag features
For every feature available at year `t`:
- create `feature_lag1` = value at `t-1` (by sorting per `company_id`)

**Optional but valuable (minimal):**
- create `feature_lag2` = value at `t-2` (only if available)
- create `growth_lag1` = `log1p(feature_lag1) - log1p(feature_lag2)` for a small set of scale vars

### 2.2 Flags for missing history (important)
- `has_lag1` / `has_lag2`
- `is_first_observation` (company starts in 2019 → explains 2019 target missing in 38 rows)

### 2.3 Drop columns (never in X)
- Always drop other targets/labels: `bankruptcy_next_year`, `financial_health_class`, `revenue_change` itself from X.

---

## 3) Target handling (robustness to heavy tails)
### 3.1 Main choice: signed-log transform
Train models on transformed target:
- `y_slog = sign(y) * log1p(|y|)`

At prediction time:
- invert: `y = sign(z) * (expm1(|z|))`

**Metrics always computed on original scale** (per challenge).

### 3.2 Alternative (ablation)
- Yeo-Johnson target transform (optional comparison)

---

## 4) Preprocessing (train-only fit)
### 4.1 Numeric
- Replace `inf -> NaN` (ratios may explode)
- Impute numeric with **median** (robust)
- Scale numeric for linear models (StandardScaler fit on train)

### 4.2 Categorical
- Fill missing with `UNK`
- One-hot encode (KIS)

---

## 5) Structural signals (unsupervised “size” features)
**Motivation:** EDA shows strong size dependence of `revenue_change` distribution.

Choose ONE of the following (KIS):
### Option A — Continuous size score (recommended)
- Compute `size_score` = PC1 / Factor score from a small set of **log1p(scale variables)** (lag1)
- Use `size_score` as an extra feature

### Option B — Small/Medium/Large clusters
- Fit GMM or KMeans (k=3 or 4) on the same log1p(scale vars)
- Use cluster label + (better) **cluster probabilities** as features

**Rule:** Fit unsupervised model on **train only**, then transform val/test.

---

## 6) Model lineup (in order)
### 6.1 Baseline 0 — Sector benchmark (sanity check)
- Predict using `ateco_sector` mean/median of `revenue_change` (train-only)
- Fallback to global mean if sector missing/unseen

### 6.2 Baseline 1 — Linear model (Ridge / ElasticNet)
**Purpose:** stable reference + interpretability  
- Inputs: `lag1 + (few growth features) + size_score/clusters + one-hot categoricals`
- Target: signed-log
- Expectation: may underfit nonlinearities but should be stable

### 6.3 Main model — XGBoost Regressor
**Purpose:** best tabular performance under nonlinearity + interactions  
- Same feature table (lag1 + minimal growth + size + categoricals)
- Target: signed-log (recommended)
- Early stopping on validation year 2021

### 6.4 Neural network (optional, for completeness)
**Recommendation:** prefer **MLP tabular** over RNN
- RNNs need longer sequences; here per-firm history is short (mostly 4 years)
- MLP on lag features is simpler and often stronger

(If RNN is required as an experiment: seq_len 2–3 with padding/masking; treat as ablation.)

---

## 7) Evaluation protocol & reporting
### 7.1 Metrics
- **Primary:** RMSE (original scale)
- **Secondary:** MAE, MAPE (with epsilon), Directional Accuracy (sign correctness)

### 7.2 Mandatory error analysis slices (simple, high value)
- `y < 0` vs `y >= 0`
- Size buckets (Small/Medium/Large via assets quantiles or clusters)
- `ateco_macro` (Covid pattern)
- Core vs tails: p1–p99 vs top 1% |y|

**Goal:** show where each model wins/loses and why.

---

## 8) Experiment plan (KIS, 6 runs max)
1) Sector baseline
2) Ridge (lag1 only) + signed-log
3) Ridge + size_score (or cluster probs)
4) XGBoost (lag1 only) + signed-log
5) XGBoost + minimal growth features (log-diff on 5 scale vars)
6) MLP tabular (optional) + signed-log

**Stop rule:** only keep added complexity if it improves val (2021) and doesn’t degrade tail/core massively.

---

## 9) Final training & submission
- Select best model based on val performance + stability across slices
- Refit on **all supervised years available before test** (2019–2021 targets) using the same pipeline
- Generate predictions for test (2022–2023)
- Invert target transform for submission

---

## 10) Checklist (quick gates)
- [ ] Feature table uses only t−1 information for predicting year t
- [ ] No overlap between train/val targets years
- [ ] All transformers fit on train only
- [ ] `inf -> NaN` handled before imputation
- [ ] Signed-log transform applied consistently + inverted correctly
- [ ] Error analysis includes size + tail slices
