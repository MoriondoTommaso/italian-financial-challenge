# Imputation Strategy Design Roadmap (Design-first, No Coding Yet) — Challenge 3

This roadmap defines **what rules we will implement**, and enforces the **fit/apply boundary** to prevent leakage in a time-series setting.

**Target**: `revenue_change` (continuous, %)  
**Key**: (`company_id`, `fiscal_year`) :contentReference[oaicite:13]{index=13}

---

## 🔒 Split Boundary (Critical Rule)

Before any model evaluation we split the training data into:
- **history years**: 2018 (lags only; NEVER used to fit imputers/scalers)
- **train years**: 2019–2020
- **validation years**: 2021
- test years exist in separate file (2022–2023)

**Allowed BEFORE split (design/audit only)**
- profiling, missingness analysis, feature typing
- defining deterministic formulas and governance rules
- deciding which statistics we will compute later (median/mode/percentiles), **without computing them yet**

**Required AFTER split (fit/apply)**
- any statistic used for imputation/clipping/encoding must be **fit on train-years only**
- then applied to validation-years (and later to test)
- implement via `sklearn` Pipeline/Transformer to guarantee correctness per fold

**Special case — lag/time features**
- lag features use only past values (t-1) within each `company_id`
- never use future values (t+1) to build features for year t

---

## Deliverables (end of Point 2 design)

- `imputation_spec.md` → final rules (what happens to each feature)
- `decision_log.md` → date → decision → rationale → evidence
- `assumption_log.md` → what must be verified during audit
- Feature list clearly assigned to:
  - deterministic reconstruction
  - structural missing handling
  - residual imputation (needs statistics)

---

## Step 0 — Data Audit Plan (checks to run) ✅ BEFORE split

**Objective**: define what we must measure before choosing imputation.

**Audit checklist**
1. Key integrity: confirm unique logical key is (`company_id`, `fiscal_year`)
2. Duplicates on the key + resolution rule (investigate, dedupe policy)
3. Missingness overview: `% missing` per column in **train vs test**
4. Missingness by time: `% missing` per column by `fiscal_year`
5. Missingness by segments:
   - `region`, `ateco_sector`, size bucket (e.g., by `total_assets`)
6. Co-missingness (block missingness): which columns go missing together
7. Deterministic feasibility: can we recompute *some* fields from others?
8. Denominator risks: near-0 / negative denominators driving NaN/inf in ratios

**Output**: one-page audit summary + templated tables.

---

## Step 1 — Data Contract & Feature Typing ✅ BEFORE split

### Feature inventory (from data dictionary)

**Keys/IDs (never impute)**
- `company_id`, `fiscal_year`

**Categoricals**
- `ateco_sector`, `province`, `region`, `legal_form`

**Base numerics — balance sheet**
- `total_fixed_assets`, `current_assets`, `total_assets`, `shareholders_equity`,
  `total_debt`, `short_term_debt`, `long_term_debt`

**Base numerics — income statement**
- `production_value`, `production_costs`, `operating_income`,
  `financial_income`, `financial_expenses`, `net_profit_loss`

**Derived numerics — ratios (precomputed)**
- `roe`, `roi`, `profit_margin`, `leverage`, `debt_to_assets`,
  `current_ratio`, `quick_ratio`

**Targets**
- `revenue_change` (y)
- `bankruptcy_next_year`, `financial_health_class` (MUST DROP from X always)

**Output**
- Table: `feature → type → reconstructable? → notes`

---

## Step 2 — Missingness Policy (governance) ✅ BEFORE split

**Objective**: set default rules before feature-by-feature decisions.

**Policy decisions to lock**
- Missingness thresholds (example):
  - Low: < 1%
  - Medium: 1–10%
  - High: > 10%  (tune based on audit results)
- Train vs test mismatch rule:
  - if a column is much more missing in test than train → treat as risk; prefer robust handling (flags + safe defaults)
- Always add missing indicators for imputed numerics:
  - `is_missing__<col>`
- Categoricals:
  - default: fill missing with `"UNK"` + `is_missing__<col>`
- Drop feature rule:
  - if high missingness AND not reconstructable AND unstable / noisy → drop (document rationale)

**Output**: 1-page policy + thresholds.

---

## Step 3 — Deterministic Reconstruction First ✅ BEFORE (design) / ✅ AFTER (apply)

**Objective**: prefer computed values over statistical imputation.

### Target reconstruction logic (important)
`revenue_change` definition:  
`((production_value_t - production_value_t-1) / production_value_t-1) * 100` :contentReference[oaicite:14]{index=14}

**Design decisions**
- Create `production_value_lag1` = shift(1) of `production_value` within `company_id`
- If `production_value_lag1` exists and is non-zero → `revenue_change_recomputed` is valid
- If `production_value_lag1` is missing (company first year) → target is genuinely missing (cannot supervise)
- If `production_value_lag1` is ~0 → flag potential explosion (handled in outliers policy; do not “fix” the target silently)

**Application (AFTER split)**
- Reconstruction occurs within preprocessing pipeline (train-years + val-years + test consistently)

**Output**: “Reconstruction Spec” table:
`feature → formula → required inputs → fallback → validation check`

---

## Step 4 — Rule-based Imputation (residual missing only) ✅ BEFORE (choose rules) / ✅ AFTER (fit stats)

### A) Categoricals (`ateco_sector`, `province`, `region`, `legal_form`)
**Default**
- Fill NaN with `"UNK"`
- Add `is_missing__province`, `is_missing__region`, etc.

**Optional enhancement**
- Group-mode imputation (fit on train-years only):
  - e.g., fill missing `province` by mode within (`region`) if available
  - fallback to global mode
- Must handle unseen groups in validation/test with fallback

### B) Base numerics (assets, debt, income statement items)
**Default**
- Median imputation + `is_missing__<col>`

**Optional enhancement**
- Group-based median (fit on train-years only):
  - group by (`fiscal_year`, `ateco_sector`) OR (`fiscal_year`) alone
  - fallback to global median
- Rationale: economics and sector norms differ by year/industry

### C) Ratios (`roe`, `leverage`, etc.)
Ratios can be NaN/inf due to denominators near 0 or negative. :contentReference[oaicite:15]{index=15}

**Policy**
- Convert inf → NaN
- Impute conservatively:
  - median (global or year-based) + missing flag
- Avoid recomputing ratios unless all required components exist and the audit confirms consistency (optional later)

**Output**: “Imputation Rulebook” grouped by feature type.

---

## Step 5 — Outliers & Stability ✅ BEFORE (choose method) / ✅ AFTER (fit thresholds)

**Objective**: prevent extreme values from dominating training.

**Decisions to lock (BEFORE split)**
- Winsorization (p1–p99) for heavy-tailed numerics/ratios:
  - candidates: `production_value`, `total_assets`, `total_debt`,
    `operating_income`, `net_profit_loss`, `roe`, `leverage`, `profit_margin`
- For the target:
  - modeling choice: evaluate baseline on raw target + consider signed-log target as an experiment
  - document any target transform and invert for metrics if needed

**Fit (AFTER split)**
- Compute percentile thresholds on train-years only; apply to val/test

**Output**: list `feature → stabilization rule → fit on train only`

---

## Step 6 — Train/Test Consistency & Leakage Guard ✅ BEFORE (contract) / ✅ AFTER (enforced)

**Non-negotiables**
- Fit anything (imputer, scaler, encoder, clipper) on train-years only
- Apply same rules to validation-years and test set
- Group-based rules must have robust fallbacks for unseen groups
- Never use `revenue_change` in any feature creation / imputation logic
- Always exclude `bankruptcy_next_year` and `financial_health_class` from X

**Output**: “Fit/Apply Contract” section to paste into the notebook.

---

## Step 7 — Evaluation Plan ✅ AFTER split

**Baseline pipeline**
- Simple preprocessing: UNK for categoricals + median for numerics + missing flags + light winsorization
- Model baseline: Ridge / ElasticNet, then tree/GBM baselines later

**A/B tests (2–4 maximum)**
1. Categorical: `"UNK"` vs group-mode (train-fit only)
2. Numeric: global median vs year-based median
3. Outliers: none vs winsorization p1–p99
4. Target: raw vs signed-log (if it improves stability)

**Output**: small experiment matrix + expected learnings.

---

## Step 8 — Documentation (continuous)

Keep these always updated:
- `decision_log.md` (every non-trivial choice)
- `assumption_log.md` (what to verify in audit)
- `imputation_spec.md` (final rulebook)

---

## Definition of Done (Point 2 — design)

Point 2 is complete when:
- Every feature is typed and assigned to:
  - deterministic reconstruction
  - structural missing handling
  - residual imputation
- Every rule has owner + rationale + planned validation check
- Fit/apply contract is explicit and prevents leakage
