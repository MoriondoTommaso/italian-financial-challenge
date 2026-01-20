# 1) EDA Final Checklist (KIS + Business-Ready) — Task 3

## 0) Dataset sanity (Data Contract)
**Output**
- Print: train/test shape, unique fiscal years, #unique companies
- Table: missing count + % per column (sorted desc)

**Why**
- Establish what we have and what needs preprocessing.

**Takeaway (1 line)**
- “The dataset contains firm-year observations (company_id, fiscal_year) with mixed numeric and categorical features and limited missingness outside `revenue_change` and `province`.”

---

## 1) Target availability by year (split justification)
**Output**
- Table by fiscal_year: `count`, `% missing revenue_change`

**Why**
- Validates which years are usable for supervised learning.

**Takeaway**
- “`revenue_change` is fully missing in 2018, so supervised training starts from 2019; we use a time-based split.”

---

## 2) Target distribution — core view (readable)
**Output**
- Histogram of `revenue_change` clipped to p1–p99 (or xlim)
- Report robust stats: median, p95/p99, max

**Why**
- Shows the “core” of the distribution without outliers dominating.

**Takeaway**
- “The target distribution is highly skewed with extreme tails; most observations are concentrated near the center.”

---

## 3) Target distribution — heavy-tail view (transform-friendly)
**Output**
- Histogram of signed-log target: `sign(y) * log1p(|y|)`

**Why**
- Provides a clear visualization of the full distribution, motivating transformation.

**Takeaway**
- “A signed-log view confirms heavy tails and supports transforming the target for more stable learning.”

---

## 4) Temporal drift in target (2019 vs 2020 vs 2021)
**Output**
- Table by year (2019/2020/2021): mean, median, p95, p99
- Boxplot by year (optionally clipped for readability)

**Why**
- Checks whether 2021 differs from training years, affecting generalization.

**Takeaway**
- “We assess drift by comparing yearly distributions; any shift in 2021 is documented as a generalization risk.”

---

## 5) Outlier mechanism diagnosis (ratio explosion check)
**Output**
- Create `production_value_lag1` (groupby + shift)
- Table: top 20 `revenue_change` rows with `production_value_lag1` and `production_value`
- (Optional) counts: share of extreme targets with very small `production_value_lag1`

**Why**
- Explains why outliers happen (often denominator near zero), justifying robustness.

**Takeaway**
- “Extreme `revenue_change` values are consistent with ratio explosion when prior-year production value is near zero.”

---

## 6) ATECO segmentation (business volatility by sector)
**Output**
- Table: top 10 `ateco_sector` by count with median + p95 of `revenue_change`

**Why**
- Identifies sectors with higher volatility and forecasting difficulty.

**Takeaway**
- “Volatility differs across sectors; some ATECO groups show heavier tails, making forecasts intrinsically harder.”

---

## 7) Geography — coverage (descriptive, not decorative)
**Output**
- Table or bar plot: company counts by `region` (top N)

**Why**
- Documents representation bias (sample concentration).

**Takeaway**
- “Company coverage is not uniform across regions; results may be more representative where coverage is higher.”

---

## 8) Company size buckets (CFO-friendly segmentation)
**Output**
- Create size buckets using a proxy (choose ONE): `production_value` or `total_assets`
  - e.g., quartiles: Small / Medium / Large / Very Large
- Table by size bucket: count, median target, p95 target

**Why**
- CFO/FP&A interpretability: volatility and error often differ by firm size.

**Takeaway**
- “Target volatility varies by firm size; this informs both business interpretation and expected model error.”

---

# Optional (only if you have time; keep to 1 table)
## 9) Feature drift on 5 key variables (X drift)
**Output**
- Table by year for 5 features (e.g., production_value, total_assets, total_debt, operating_income, current_ratio):
  - median + p95 per year (or mean + std if preferred)

**Why**
- If feature distributions shift, it helps explain changes in performance in 2021.

**Takeaway**
- “Key financial features show (limited/material) drift across years, which can affect 2021 generalization.”

---

# EDA Exit Criteria (what the notebook should end with)
At the end of the EDA section, include **5–7 bullet takeaways** summarizing:
- split justification (2018 missing, time split)
- heavy-tail & outliers → transformation
- drift assessment (target + optionally features)
- sector + region + size insights (business context)


## 9) Correlation & Multicollinearity (Driver Analysis)
**Output**
- **Table/Barplot:** Top 10 features most correlated with `revenue_change` (use **Spearman** correlation to be robust against outliers).
- **Heatmap (masked):** Pairwise correlations between financial ratios (e.g., `roe`, `roi`, `leverage`) to spot redundancy (e.g., > 0.8)[cite: 526].

**Why**
- Identifies linear/monotonic drivers for the regression model and detects multicollinearity (e.g., `current_ratio` vs `quick_ratio`) that could destabilize linear baselines[cite: 525, 526].

**Takeaway**
- “We identified top predictive drivers (e.g., Profit Margin) and flagged highly collinear features to reduce noise and overfitting.”

# 2) Imputation Strategy Design Roadmap (No Coding Yet)

This section is **design-first**. It also makes explicit the **split boundary** to prevent leakage.

---

## 🔒 Split Boundary (Critical Rule)
Before any model evaluation, we split the training data into **train-fold** and **validation-fold**.

- ✅ **Allowed BEFORE the split (design/audit):**
  - profiling, missingness analysis, feature typing
  - defining rules and formulas (what we *will* do)
  - deciding which statistics we *will compute later* (median/mode/percentiles), **without computing them yet**

- ✅ **Required AFTER the split (fit/apply):**
  - any statistic used for imputation/clipping/encoding **must be fit on train-fold only**
  - then applied to validation-fold (and later to test)
  - put in an `sklearn` Pipeline so this happens automatically per fold/CV

- ⏳ **Special case — lag/time-derived features (t-1):**
  - prefer **time-aware split** (or walk-forward) if features use previous-year info
  - never use future values (t+1) to build features for year t

---

## Deliverables (what we should produce by the end)
- `imputation_spec.md` (final rules)
- `decision_log.md` (what we decided and why)
- `assumption_log.md` (what must be verified during the audit)
- A clear list of:
  - **Deterministically reconstructable** fields
  - **Structurally missing** fields (handled with flags / special values)
  - **Residual missing** fields (true imputation needed)

---

## Step 0 — Data Audit Plan (what we will check)  ✅ BEFORE split
**Objective:** define the checks that must be run before choosing any imputation.

**Audit checklist to define:**
1. Key integrity: confirm logical key is `(company_id, fiscal_year)`
2. Duplicates on the key + resolution rule (investigate, dedupe, keep latest, etc.)
3. Missingness overview: `% missing` per column in **train vs test**
4. Missingness by time: `% missing` per column **by fiscal_year**
5. Missingness by segments (if available): region / size buckets / sector
6. Co-missingness patterns: which columns go missing together (block missingness)
7. Derived-column feasibility: can we **recompute** missing derived fields from base columns?
8. Ratio stability: denominators that are **0 / near 0 / negative** and overlap with missingness

**Output:** a one-page audit checklist + reporting template.

---

## Step 1 — Data Contract & Feature Typing  ✅ BEFORE split
**Objective:** label each feature to select the correct family of rules.

**Feature types:**
- **Keys/IDs** (never impute; fix upstream or drop rows if invalid)
- **Categoricals** (province/region/sector…)
- **Base numerics** (assets, debt, costs…)
- **Derived numerics** (ratios, margins, changes, indicators)
- **Leakage-risk / target-adjacent** (if any)

**Output:** table: `feature → type → reconstructable? (Y/N) → notes`.

---

## Step 2 — Missingness Policy (governance rules)  ✅ BEFORE split
**Objective:** set general rules *before* handling individual features.

**Decisions to lock (policy only, no fitting):**
- Thresholds: what counts as low/medium/high missingness
- Handling differences between train vs test missingness
- When to prefer **missing flags** + special category (e.g., `"UNK"`) vs statistical fill
- When to drop a feature (too missing + not reconstructable + not stable)

**Output:** a 1-page policy with thresholds and default rules.

---

## Step 3 — Deterministic Reconstruction First (highest priority)  ✅ BEFORE split (design) / ✅ AFTER split (apply)
**Objective:** avoid “imputation” whenever the value can be computed.

**Decisions to lock (BEFORE split):**
- List all **derived** fields
- Official formulas (from data dictionary / challenge description)
- Rule: if ingredients exist → recompute; else → fallback policy
- Sanity check: acceptable tolerance between recomputed and provided values

**Application (AFTER split):**
- Apply reconstruction consistently inside the preprocessing pipeline (train-fold + val-fold + test)

**Output:** “Reconstruction Spec” table:
`feature → formula → required inputs → fallback → validation check`.

---

## Step 4 — Rule-based Imputation (residual missing only)  ✅ BEFORE split (choose rules) / ✅ AFTER split (fit parameters)
**Objective:** define conservative, explainable rules for what remains.

**Rule design (BEFORE split):**
- Categoricals:
  - `"UNK"` + `is_missing` flag (default) vs
  - group-based mode (e.g., mode within region/year) + fallback to global mode
- Base numerics:
  - global median vs group-based median (by year/size/sector) + fallback global
  - always add `is_missing` flag for any imputed numeric
- Ratios with unstable denominators:
  - never “invent” silently; use flags + defined clipping/winsorization policy

**Parameter fitting (AFTER split):**
- Compute **mode/median/group medians** on **train-fold only**, apply to validation-fold/test

**Output:** “Imputation Rulebook” grouped by feature type + standard flags.

---

## Step 5 — Outliers & Stability (post-imputation)  ✅ BEFORE split (choose method) / ✅ AFTER split (fit thresholds)
**Objective:** prevent extreme values introduced by reconstruction/imputation.

**Decisions to lock (BEFORE split):**
- Which features get clipping/winsorization
- Any transformations (e.g., `log1p`) only if justified and consistent

**Threshold fitting (AFTER split):**
- Percentiles computed **only on training fold**
- Apply same thresholds to validation fold and test

**Output:** list: `feature → stabilization rule → fit on train`.

---

## Step 6 — Train/Test Consistency & Leakage Guard  ✅ BEFORE split (contract) / ✅ AFTER split (enforced via pipeline)
**Objective:** ensure the same rules apply to train and test without leakage.

**Decisions to lock:**
- Any statistic (median/mode/percentiles) must be **fit on train**, applied to val/test
- Group-based rules must handle unseen groups in test (fallback global)
- Target must never be used in feature creation or imputation logic

**Output:** “Fit/Apply Contract” (what is learned from train vs applied to test).

---

## Step 7 — Evaluation Plan (validate choices)  ✅ AFTER split
**Objective:** test whether imputation choices help.

**Decisions to lock:**
- Baseline pipeline (minimal rules)
- A/B comparisons (small set, e.g., UNK vs mode; global vs year-based median)
- Correct split strategy (time-aware if needed) and evaluation metrics

**Output:** experiment plan with 2–4 targeted comparisons.

---

## Step 8 — Documentation & Decision Log (mandatory)  ✅ BEFORE & AFTER (continuous)
**Objective:** keep the team aligned and avoid “implicit” rules.

**What to produce:**
- `imputation_spec.md` with final rules
- `decision_log.md`: `date → decision → rationale → evidence`
- `assumption_log.md`: things to verify during Step 0 audit

**Output:** docs ready to commit to the repo.

---

## Definition of Done (Point 2 design)
Point 2 is complete when:
- All features are typed and assigned to one of:
  - deterministic reconstruction
  - structural missing handling
  - residual imputation
- Every rule has an owner, rationale, and planned validation check
- The Fit/Apply contract prevents leakage and ensures consistency
