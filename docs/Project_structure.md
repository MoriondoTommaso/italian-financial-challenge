# EDA Final Checklist (KIS + Business-Ready) — Task 3

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