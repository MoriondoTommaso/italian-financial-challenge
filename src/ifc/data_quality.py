"""
Data Quality & Leakage Gates (KIS)

Goal:
- Provide reusable, readable checks to validate dataset integrity before modeling.
- Return notebook-friendly tables + a PASS/WARN/FAIL summary.
- Separate "core" (hard gates) from "diagnostic" (useful warnings, not blockers).

Core checks:
1) No duplicate keys (company_id, fiscal_year)
2) Year sanity + train/test temporal separation
3) Target missingness only on first observation per company
4) Target formula consistency with production_value lag
5) No target columns leaking into test

Diagnostic checks:
- Accounting identity (assets ≈ equity + debt) using relative tolerance
- NaN/Inf in ratios (expected sometimes)
- Multi-year exact zeros in scale variables
- Simple outlier flags (informational)

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = ["run_all_checks", "render_report"]



# Constants (dataset contract)
ID_COL = "company_id"
TIME_COL = "fiscal_year"
TARGET = "revenue_change"

# Known "target-only" columns (must NOT be used as features)
AUX_TARGETS = ["bankruptcy_next_year", "financial_health_class"]

RAW_BALANCE = [
    "total_fixed_assets", "current_assets", "total_assets", "shareholders_equity",
    "total_debt", "short_term_debt", "long_term_debt",
]
RAW_INCOME = [
    "production_value", "production_costs", "operating_income",
    "financial_income", "financial_expenses", "net_profit_loss",
]
RATIO_COLS = [
    "roe", "roi", "profit_margin", "leverage",
    "debt_to_assets", "current_ratio", "quick_ratio",
]



# Helpers
def _require_cols(df: pd.DataFrame, cols: Sequence[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{where}] Missing required columns: {missing}")


def _existing_cols(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def _status(pass_cond: bool, warn_cond: bool = False) -> str:
    # Priority: FAIL > WARN > PASS
    if not pass_cond:
        return "FAIL"
    if warn_cond:
        return "WARN"
    return "PASS"


def _nan_inf_table(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    cols = _existing_cols(df, cols)
    rows: List[Dict[str, Any]] = []
    for c in cols:
        s = df[c]
        rows.append({
            "col": c,
            "n_nan": int(s.isna().sum()),
            "n_inf": int(np.isinf(s).sum()),
            "pct_nan": float(s.isna().mean() * 100),
            "pct_inf": float(np.isinf(s).mean() * 100),
        })
    if not rows:
        return pd.DataFrame(columns=["col", "n_nan", "n_inf", "pct_nan", "pct_inf"])
    out = pd.DataFrame(rows).sort_values(["n_inf", "n_nan"], ascending=False)
    return out.reset_index(drop=True)


def _missing_table(df: pd.DataFrame, cols: Sequence[str], group: str) -> pd.DataFrame:
    cols = _existing_cols(df, cols)
    if not cols:
        return pd.DataFrame([{"group": group, "col": None, "n_missing": None, "pct_missing": None}])

    miss = df[cols].isna().sum()
    out = pd.DataFrame({
        "group": group,
        "col": miss.index,
        "n_missing": miss.values,
        "pct_missing": (miss.values / len(df) * 100),
    }).sort_values("n_missing", ascending=False)
    return out.reset_index(drop=True)



# Core checks
def check_no_duplicate_keys(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    _require_cols(train_df, [ID_COL, TIME_COL], "check_no_duplicate_keys(train)")
    _require_cols(test_df, [ID_COL, TIME_COL], "check_no_duplicate_keys(test)")
    return {
        "dup_train": int(train_df.duplicated([ID_COL, TIME_COL]).sum()),
        "dup_test": int(test_df.duplicated([ID_COL, TIME_COL]).sum()),
    }


def check_years_and_separation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    expected_train_years: Optional[Sequence[int]] = None,
    expected_test_years: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    _require_cols(train_df, [TIME_COL], "check_years_and_separation(train)")
    _require_cols(test_df, [TIME_COL], "check_years_and_separation(test)")

    years_train = sorted(train_df[TIME_COL].dropna().unique().tolist())
    years_test = sorted(test_df[TIME_COL].dropna().unique().tolist())

    separation_ok = True
    if years_train and years_test:
        separation_ok = max(years_train) < min(years_test)

    train_subset_ok = True
    test_subset_ok = True
    if expected_train_years is not None:
        train_subset_ok = set(years_train).issubset(set(expected_train_years))
    if expected_test_years is not None:
        test_subset_ok = set(years_test).issubset(set(expected_test_years))

    return {
        "years_train": years_train,
        "years_test": years_test,
        "separation_ok": bool(separation_ok),
        "train_subset_ok": bool(train_subset_ok),
        "test_subset_ok": bool(test_subset_ok),
        "expected_train_years": list(expected_train_years) if expected_train_years is not None else None,
        "expected_test_years": list(expected_test_years) if expected_test_years is not None else None,
    }


def check_target_missing_only_first_obs(train_df: pd.DataFrame) -> Dict[str, Any]:
    _require_cols(train_df, [ID_COL, TIME_COL, TARGET], "check_target_missing_only_first_obs")

    df = train_df.sort_values([ID_COL, TIME_COL]).copy()
    first_year = df.groupby(ID_COL)[TIME_COL].transform("min")
    is_first_obs = df[TIME_COL].eq(first_year)

    missing_target = df[TARGET].isna()

    # In this challenge: target is expected missing on first obs (e.g., 2018 for most firms, and for firms that start in 2019 it can also be missing in 2019).
    return {
        "n_missing_target": int(missing_target.sum()),
        "n_first_obs": int(is_first_obs.sum()),
        "missing_not_first": int((missing_target & ~is_first_obs).sum()),
        "first_not_missing": int((~missing_target & is_first_obs).sum()),
        "perfect_match_missing_eq_first": bool((missing_target == is_first_obs).all()),
    }


def check_revenue_change_formula(
    train_df: pd.DataFrame,
    *,
    diff_tol_p99: float = 1e-2,  # percentage points
) -> Dict[str, Any]:
    """
    Check that revenue_change matches:
        (production_value - production_value_lag1) / production_value_lag1 * 100

    Uses p99 of absolute diff as a robust gate.
    """
    _require_cols(train_df, [ID_COL, TIME_COL, TARGET, "production_value"], "check_revenue_change_formula")

    df = train_df.sort_values([ID_COL, TIME_COL]).copy()
    df["pv_lag1"] = df.groupby(ID_COL)["production_value"].shift(1)
    df["rc_calc"] = (df["production_value"] - df["pv_lag1"]) / df["pv_lag1"] * 100

    mask = df[TARGET].notna() & df["rc_calc"].notna() & np.isfinite(df["rc_calc"])
    n_compared = int(mask.sum())
    if n_compared == 0:
        return {
            "n_compared": 0,
            "abs_diff_p99": np.nan,
            "abs_diff_max": np.nan,
            "abs_diff_median": np.nan,
            "diff_tol_p99": float(diff_tol_p99),
        }

    abs_diff = (df.loc[mask, TARGET] - df.loc[mask, "rc_calc"]).abs()
    return {
        "n_compared": n_compared,
        "abs_diff_p99": float(abs_diff.quantile(0.99)),
        "abs_diff_max": float(abs_diff.max()),
        "abs_diff_median": float(abs_diff.median()),
        "diff_tol_p99": float(diff_tol_p99),
    }


def check_no_target_leakage_in_test(test_df: pd.DataFrame) -> Dict[str, Any]:
    forbidden = [TARGET] + AUX_TARGETS
    present = [c for c in forbidden if c in test_df.columns]
    return {"forbidden_present_in_test": present, "ok": len(present) == 0}


# Diagnostic checks
def check_accounting_identity(
    df: pd.DataFrame,
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 1.0,
) -> Dict[str, Any]:
    """
    Accounting identity: total_assets ≈ shareholders_equity + total_debt.
    Use BOTH relative and absolute tolerance to avoid false positives.
    """
    _require_cols(df, ["total_assets", "shareholders_equity", "total_debt"], "check_accounting_identity")

    lhs = df["total_assets"]
    rhs = df["shareholders_equity"] + df["total_debt"]
    abs_diff = (lhs - rhs).abs()

    denom = lhs.abs().replace(0, np.nan)
    rel_diff = (abs_diff / denom).fillna(0.0)

    violations = (abs_diff > abs_tol) & (rel_diff > rel_tol)

    return {
        "rel_tol": float(rel_tol),
        "abs_tol": float(abs_tol),
        "abs_diff_p99": float(abs_diff.quantile(0.99)),
        "rel_diff_p99": float(rel_diff.quantile(0.99)),
        "violations": int(violations.sum()),
    }


def check_multi_year_zeros(df: pd.DataFrame, col: str, *, min_years: int = 2) -> Optional[int]:
    if col not in df.columns:
        return None
    _require_cols(df, [ID_COL], "check_multi_year_zeros")
    zeros = df[col].eq(0)
    counts = zeros.groupby(df[ID_COL]).sum()
    return int((counts >= min_years).sum())


def flag_outliers(train_df: pd.DataFrame) -> Dict[str, Any]:
    _require_cols(train_df, [TARGET], "flag_outliers")
    rc = train_df.loc[train_df[TARGET].notna(), TARGET]
    return {
        "revenue_change_gt_100": int((rc > 100).sum()),
        "revenue_change_lt_-80": int((rc < -80).sum()),
        "leverage_gt_50": int((train_df["leverage"] > 50).sum()) if "leverage" in train_df.columns else None,
        "production_value_negative": int((train_df["production_value"] < 0).sum()) if "production_value" in train_df.columns else None,
    }



# Main runner

def run_all_checks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    checks: Tuple[str, ...] = ("core", "diagnostic"),
    expected_train_years: Optional[Sequence[int]] = (2018, 2019, 2020, 2021),
    expected_test_years: Optional[Sequence[int]] = (2022, 2023),
    formula_diff_tol_p99: float = 1e-2,   # percentage points
    identity_rel_tol: float = 1e-6,
    identity_abs_tol: float = 1.0,
    verbose: bool = True,
    only_issues: bool = True,
) -> Dict[str, Any]:
    """
    Run checks and return a dictionary of notebook-friendly tables.
    If verbose=True, uses render_report(report).
    """
    checks_set = set(checks)
    do_core = "core" in checks_set
    do_diag = "diagnostic" in checks_set

    report: Dict[str, Any] = {}

    # Overview
    _require_cols(train_df, [ID_COL, TIME_COL], "run_all_checks(train)")
    _require_cols(test_df, [ID_COL, TIME_COL], "run_all_checks(test)")
    overview = pd.DataFrame([{
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_companies_train": int(train_df[ID_COL].nunique()),
        "n_companies_test": int(test_df[ID_COL].nunique()),
    }])
    report["overview"] = overview

    summary_rows: List[Dict[str, str]] = []

    
    # Core checks
    
    if do_core:
        dup = check_no_duplicate_keys(train_df, test_df)
        years = check_years_and_separation(
            train_df, test_df,
            expected_train_years=expected_train_years,
            expected_test_years=expected_test_years,
        )
        miss = check_target_missing_only_first_obs(train_df)
        formula = check_revenue_change_formula(train_df, diff_tol_p99=formula_diff_tol_p99)
        leak = check_no_target_leakage_in_test(test_df)

        report["keys_duplicates"] = pd.DataFrame([dup])
        report["years_and_separation"] = pd.DataFrame([years])
        report["target_missingness"] = pd.DataFrame([miss])
        report["target_formula"] = pd.DataFrame([formula])
        report["target_leakage_test"] = pd.DataFrame([{
            "forbidden_present_in_test": str(leak["forbidden_present_in_test"]),
            "ok": leak["ok"]
        }])

        # Summary statuses (core gates)
        dup_ok = (dup["dup_train"] == 0) and (dup["dup_test"] == 0)
        summary_rows.append({"check": "No duplicate keys (company_id,fiscal_year)", "status": _status(dup_ok)})

        # Year sanity: if expected years provided, require subset OK. Always require temporal separation.
        subset_required = (expected_train_years is not None) or (expected_test_years is not None)
        subset_ok = years["train_subset_ok"] and years["test_subset_ok"] if subset_required else True
        years_pass = bool(years["separation_ok"] and subset_ok)
        years_warn = not subset_required  # if we didn't enforce expected year sets
        summary_rows.append({"check": "Years sanity + train/test temporal separation", "status": _status(years_pass, warn_cond=years_warn)})

        miss_ok = (miss["missing_not_first"] == 0) and (miss["first_not_missing"] == 0)
        summary_rows.append({"check": "Target missing only on first observation per company", "status": _status(miss_ok)})

        # Formula gate on p99 (robust)
        if int(formula["n_compared"]) == 0:
            summary_rows.append({"check": "Target formula matches production_value lag (p99 abs diff)", "status": "WARN"})
        else:
            formula_ok = float(formula["abs_diff_p99"]) <= float(formula["diff_tol_p99"])
            summary_rows.append({"check": "Target formula matches production_value lag (p99 abs diff)", "status": _status(formula_ok)})

        leak_ok = leak["ok"]
        summary_rows.append({"check": "No target columns leaking into test", "status": _status(leak_ok)})

    
    # Diagnostic checks
    
    if do_diag:
        # Missingness for raw items (informational)
        missing_train = pd.concat([
            _missing_table(train_df, RAW_BALANCE, "balance_sheet"),
            _missing_table(train_df, RAW_INCOME, "income_statement"),
        ], ignore_index=True)
        missing_test = pd.concat([
            _missing_table(test_df, RAW_BALANCE, "balance_sheet"),
            _missing_table(test_df, RAW_INCOME, "income_statement"),
        ], ignore_index=True)

        ratio_train = _nan_inf_table(train_df, RATIO_COLS)
        ratio_test = _nan_inf_table(test_df, RATIO_COLS)

        identity_train = check_accounting_identity(train_df, rel_tol=identity_rel_tol, abs_tol=identity_abs_tol) \
            if set(["total_assets", "shareholders_equity", "total_debt"]).issubset(train_df.columns) else None
        identity_test = check_accounting_identity(test_df, rel_tol=identity_rel_tol, abs_tol=identity_abs_tol) \
            if set(["total_assets", "shareholders_equity", "total_debt"]).issubset(test_df.columns) else None

        outliers = flag_outliers(train_df)

        zeros = {
            "multi_year_zero_total_assets (>=2y)": check_multi_year_zeros(train_df, "total_assets", min_years=2),
            "multi_year_zero_total_debt (>=2y)": check_multi_year_zeros(train_df, "total_debt", min_years=2),
            "multi_year_zero_production_value (>=2y)": check_multi_year_zeros(train_df, "production_value", min_years=2),
        }

        report["missing_raw_train"] = missing_train
        report["missing_raw_test"] = missing_test
        report["ratio_nan_inf_train"] = ratio_train
        report["ratio_nan_inf_test"] = ratio_test

        if identity_train is not None and identity_test is not None:
            report["accounting_identity"] = pd.DataFrame([
                {"split": "train", **identity_train},
                {"split": "test", **identity_test},
            ])

            id_ok = (identity_train["violations"] == 0) and (identity_test["violations"] == 0)
            # Identity violations are usually a WARN (not a hard gate)
            summary_rows.append({"check": "Accounting identity holds (diagnostic)", "status": _status(True, warn_cond=not id_ok)})

        ratio_inf_any = (ratio_train["n_inf"].sum() + ratio_test["n_inf"].sum()) > 0
        # Ratios infs are usually WARN (division by 0, negative denominators, etc.)
        summary_rows.append({"check": "No INF values in ratios (diagnostic)", "status": _status(True, warn_cond=ratio_inf_any)})

        zeros_any = any([(zeros["multi_year_zero_total_assets (>=2y)"] or 0) > 0,
                         (zeros["multi_year_zero_total_debt (>=2y)"] or 0) > 0,
                         (zeros["multi_year_zero_production_value (>=2y)"] or 0) > 0])
        summary_rows.append({"check": "No multi-year exact zeros in scale vars (diagnostic)", "status": _status(True, warn_cond=zeros_any)})

        report["outlier_flags_train"] = pd.DataFrame([outliers])
        report["multi_year_zeros_train"] = pd.DataFrame([zeros])

    report["summary"] = pd.DataFrame(summary_rows)

    if verbose:
        render_report(report, only_issues=only_issues)

    return report



# Notebook rendering

def render_report(report: Dict[str, Any], *, only_issues: bool = True) -> None:
    """
    - Shows the PASS/WARN/FAIL summary.
    - If only_issues=True, shows only WARN/FAIL rows (or all if none).
    - If issues exist, shows a small set of detail tables.
    """
    summary = report.get("summary", pd.DataFrame())
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        print("No summary to display.")
        return

    # Filter view
    if only_issues:
        view = summary[summary["status"].isin(["WARN", "FAIL"])]
        if view.empty:
            view = summary
    else:
        view = summary

    # Notebook display if available
    try:
        from IPython.display import display
        in_notebook = True
    except Exception:
        display = None
        in_notebook = False

    print("DATA QUALITY SUMMARY (PASS/WARN/FAIL)")

    if in_notebook and display is not None:
        display(view)
        has_issues = summary["status"].isin(["WARN", "FAIL"]).any()
        if has_issues:
            for k in [
                "overview",
                "keys_duplicates",
                "years_and_separation",
                "target_missingness",
                "target_formula",
                "target_leakage_test",
                "accounting_identity",
            ]:
                df = report.get(k)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    print(f"\nDETAILS: {k}")
                    display(df)
    else:
        print(view.to_string(index=False))
        has_issues = summary["status"].isin(["WARN", "FAIL"]).any()
        if has_issues:
            for k in [
                "overview",
                "keys_duplicates",
                "years_and_separation",
                "target_missingness",
                "target_formula",
                "target_leakage_test",
                "accounting_identity",
            ]:
                df = report.get(k)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    print(f"\nDETAILS: {k}")
                    print(df.to_string(index=False))
