# EDA Final Checklist (KIS + Business-Ready) — Challenge 3 (Revenue Forecasting)

**Goal**: forecast `revenue_change` (percentage change in `production_value`) with a time-aware workflow to avoid leakage.

## Dataset context (columns from data dictionary)

- **Key / time**: `company_id`, `fiscal_year`
- **Categorical**: `ateco_sector`, `province`, `region`, `legal_form`
- **Company characteristic**: `years_in_business`
- **Balance sheet**: `total_fixed_assets`, `current_assets`, `total_assets`, `shareholders_equity`, `total_debt`, `short_term_debt`, `long_term_debt`
- **Income statement**: `production_value`, `production_costs`, `operating_income`, `financial_income`, `financial_expenses`, `net_profit_loss`
- **Ratios (precomputed)**: `roe`, `roi`, `profit_margin`, `leverage`, `debt_to_assets`, `current_ratio`, `quick_ratio`
- **Targets present in train**: `revenue_change` (our target), plus `bankruptcy_next_year`, `financial_health_class` (MUST be excluded from features)
  - Drop these two targets from X always (no leakage).

> Split reference (project decision): history=2018 (lags only), train=2019–2020, validation=2021; no shuffle.

---

## 0) Dataset sanity (Data Contract)

**Output**
- Print: train/test shape, unique `fiscal_year`, #unique `company_id`
- Table: missing count + % per column (sorted desc)
- Check duplicates on logical key (`company_id`, `fiscal_year`) and report how many

**Why**
- Establish what we have and what needs preprocessing.

**Takeaway (1 line)**
- “The dataset contains firm-year observations keyed by (`company_id`, `fiscal_year`) with mixed numeric/categorical features and missingness concentrated in `revenue_change` and `province`.”

---

## 1) Target availability by year (split justification)

**Output**
- Table by `fiscal_year`: `count`, `% missing revenue_change`

**Why**
- Validates which years are usable for supervised learning.

**Takeaway**
- “`revenue_change` is missing for companies’ first observed year; in particular it is fully missing in 2018 (dataset start), so supervised learning starts from 2019 and we use a time-based split.”

---

## 2) Target distribution — core view (readable)

**Output**
- Histogram of `revenue_change` clipped to p1–p99 (or set x-limits)
- Robust stats: median, p95, p99, min/max (also report share beyond p99)

**Why**
- Shows the “core” without outliers dominating.

**Takeaway**
- “The target is heavy-tailed: most observations are near the center, with extreme growth/decline outliers.”

---

## 3) Target distribution — heavy-tail view (transform-friendly)

**Output**
- Histogram of signed-log target: `sign(y) * log1p(|y|)`

**Why**
- Visualize full distribution and motivate robust modeling / transformation.

**Takeaway**
- “Signed-log confirms heavy tails and supports using robust losses / target transforms for stability.”

---

## 4) Temporal drift in target (The "COVID V-Shape")

**Output**
- Table by `fiscal_year` (2019/2020/2021): mean, median, p95, p99
- Boxplot by year (optionally clipped for readability)
- **Specific check**: Look for contraction in 2020 vs rebound in 2021.

**Why**
- 2020 was a pandemic year (contraction), 2021 recovery. This extreme volatility justifies strict time-validation.

**Takeaway**
- “We observe a distinct 'V-shape' pattern (2020 drop, 2021 rebound); this confirms 2021 is structurally different, increasing the importance of our validation strategy.”

---

## 5) Outlier mechanism diagnosis (ratio explosion check)

**Output**
- Create `production_value_lag1` = groupby(`company_id`) + shift(1) on `production_value`
- Table: top 20 absolute `revenue_change` rows with:
  - `company_id`, `fiscal_year`, `production_value_lag1`, `production_value`, `revenue_change`
- (Optional) quantify: share of extreme targets where `production_value_lag1` is near 0 (e.g., below p5)

**Why**
- Explains extreme values (often denominator near 0), justifying winsorization / robust metrics.

**Takeaway**
- “Extreme `revenue_change` is consistent with denominator effects when prior-year `production_value` is very small.”

---

## 6) ATECO segmentation (volatility by sector)

**Output**
- Table: top 10 `ateco_sector` by count with:
  - count, median `revenue_change`, p95 `revenue_change` (and optionally p99)

**Why**
- Sectors differ in inherent volatility → forecasting difficulty differs.

**Takeaway**
- “Volatility differs across `ateco_sector`; some sectors have heavier tails, implying higher expected error.”

---

## 7) Geography — coverage (descriptive, not decorative)

**Output**
- Table/bar plot: company counts by `region` (top N)
- (Optional) missingness of `province` by `region`

**Why**
- Documents representation bias + categorical sparsity.

**Takeaway**
- “Coverage is not uniform across `region`; results may reflect areas with higher representation.”

---

## 8) Company size buckets (CFO-friendly segmentation)

**Default proxy**: `total_assets` (alternative: `production_value`)

**Output**
- Bucket firms by `total_assets` quartiles within train-years (Small/Medium/Large/Very Large)
- Table by bucket: count, median target, p95 target

**Why**
- Volatility and model error often differ by firm size.

**Takeaway**
- “Target volatility varies by size bucket; we expect different error profiles across firm sizes.”

---

## 9) Company Age Analysis (Startup Volatility)

**Feature**: `years_in_business`

**Output**
- Scatter plot: `years_in_business` vs `abs(revenue_change)` (or `revenue_change`)
- Table by Age Bucket (e.g., <5 years, 5-15 years, >15 years):
  - Median target, Standard Deviation of target

**Why**
- Hypothesis: Young companies (startups) have explosive growth/fail rates; mature companies are stable.

**Takeaway**
- “Younger companies show significantly higher volatility, suggesting `years_in_business` is a key segmentation feature for handling outliers.”

---

## 10) Feature drift on 5 key variables (X drift)

**Suggested 5 features**
- `production_value`, `total_assets`, `total_debt`, `operating_income`, `current_ratio`

**Output**
- Table by `fiscal_year` for each feature: median + p95 (or mean+std)
- Note any strong shifts in 2021 vs train years

**Why**
- Feature drift can explain performance drop in 2021.

**Takeaway**
- “Key financial drivers show (limited/material) drift across years, impacting 2021 generalization.”

---

## 11) Correlation & Multicollinearity (Driver Analysis)

**Output**
- Table/bar plot: top 10 features most correlated with `revenue_change` (Spearman)
- Correlation heatmap (masked) among **ratios** to spot redundancy:
  - `roe`, `roi`, `profit_margin`, `leverage`, `debt_to_assets`, `current_ratio`, `quick_ratio`

**Why**
- Identifies monotonic drivers and flags multicollinearity that can destabilize linear baselines.

**Takeaway**
- “We identify top drivers and flag highly collinear ratios to reduce redundancy and overfitting.”

---

## EDA Exit Criteria (end-of-notebook bullets)

End the EDA section with **5–7 bullets** covering:
- Split justification (2018 + first-year missingness; time-aware holdout)
- Heavy-tail/outliers → robust metrics + winsorization / transforms
- Drift assessment:
    - **V-Shape Target:** 2020 contraction vs 2021 rebound
    - **Feature Drift:** Stability of key predictors
- Volatility drivers:
    - **Sector** (`ateco_sector`)
    - **Size** (`total_assets`)
    - **Age** (`years_in_business` - startups vs mature)
- Any data quality anomalies found (duplicates, accounting identity violations)

---

## (Optional but recommended) Data quality checks tied to the dictionary

- Accounting identity check: `total_assets ≈ shareholders_equity + total_debt` (report violation rate; investigate extreme deviations).
- Ratio sanity: check `roe`, `leverage` for inf/NaN patterns tied to denominators (equity near 0 / negative).