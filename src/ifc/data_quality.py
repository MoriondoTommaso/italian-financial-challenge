"""
Data quality & leakage gates for IFC Task 3.

Goal:
- Provide reusable, readable checks to validate dataset integrity before modeling.
- Produce notebook-friendly tables + a PASS/WARN/FAIL summary.

Public API:
- run_all_checks_readable(train_df, test_df, ...)
- print_report(report)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

__all__ = ["run_all_checks_readable", "print_report"]

ID_COL = "company_id"
TIME_COL = "fiscal_year"
TARGET = "revenue_change"

RAW_BALANCE = [
    "total_fixed_assets", "current_assets", "total_assets", "shareholders_equity",
    "total_debt", "short_term_debt", "long_term_debt"
]
RAW_INCOME = [
    "production_value", "production_costs", "operating_income",
    "financial_income", "financial_expenses", "net_profit_loss"
]
RATIO_COLS = ["roe", "roi", "profit_margin", "leverage", "debt_to_assets", "current_ratio", "quick_ratio"]


# -----------------------------
# Helpers
# -----------------------------
def _require_cols(df: pd.DataFrame, cols: List[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{where}] Missing required columns: {missing}")

def _existing_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

def _nan_inf_table(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
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

def _series_missing_table(df: pd.DataFrame, cols: List[str], group_name: str) -> pd.DataFrame:
    cols = _existing_cols(df, cols)
    if not cols:
        return pd.DataFrame([{"group": group_name, "col": None, "n_missing": None, "pct_missing": None}])

    miss = df[cols].isna().sum()
    out = pd.DataFrame({
        "group": group_name,
        "col": miss.index,
        "n_missing": miss.values,
        "pct_missing": (miss.values / len(df) * 100),
    }).sort_values("n_missing", ascending=False)
    return out.reset_index(drop=True)

def _status(pass_cond: bool, warn_cond: bool = False) -> str:
    # Priority: FAIL > WARN > PASS
    if not pass_cond:
        return "FAIL"
    if warn_cond:
        return "WARN"
    return "PASS"


# -----------------------------
# Core checks
# -----------------------------
def check_keys_and_years(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    _require_cols(train_df, [ID_COL, TIME_COL], "check_keys_and_years(train)")
    _require_cols(test_df, [ID_COL, TIME_COL], "check_keys_and_years(test)")

    dup_train = int(train_df.duplicated([ID_COL, TIME_COL]).sum())
    dup_test = int(test_df.duplicated([ID_COL, TIME_COL]).sum())
    years_train = sorted(train_df[TIME_COL].unique().tolist())
    years_test = sorted(test_df[TIME_COL].unique().tolist())

    return {"dup_train": dup_train, "dup_test": dup_test, "years_train": years_train, "years_test": years_test}

def check_target_missingness_is_first_obs(train_df: pd.DataFrame) -> Dict[str, Any]:
    _require_cols(train_df, [ID_COL, TIME_COL, TARGET], "check_target_missingness_is_first_obs")

    df = train_df.sort_values([ID_COL, TIME_COL]).copy()
    first_year = df.groupby(ID_COL)[TIME_COL].transform("min")
    is_first = df[TIME_COL].eq(first_year)
    missing_target = df[TARGET].isna()

    return {
        "n_missing_target": int(missing_target.sum()),
        "n_first_obs": int(is_first.sum()),
        "missing_not_first": int((missing_target & ~is_first).sum()),
        "first_not_missing": int((~missing_target & is_first).sum()),
        "perfect_match": bool(((missing_target == is_first).all())),
    }

def check_revenue_change_formula(train_df: pd.DataFrame) -> Dict[str, Any]:
    _require_cols(train_df, [ID_COL, TIME_COL, TARGET, "production_value"], "check_revenue_change_formula")

    df = train_df.sort_values([ID_COL, TIME_COL]).copy()
    df["pv_lag1"] = df.groupby(ID_COL)["production_value"].shift(1)
    df["rc_calc"] = (df["production_value"] - df["pv_lag1"]) / df["pv_lag1"] * 100

    mask = df[TARGET].notna() & df["rc_calc"].notna() & np.isfinite(df["rc_calc"])
    abs_diff = (df.loc[mask, TARGET] - df.loc[mask, "rc_calc"]).abs()

    # If for some reason mask is empty, avoid .max() returning NaN without explanation
    if int(mask.sum()) == 0:
        return {"n_compared": 0, "abs_diff_max": np.nan, "abs_diff_p99": np.nan, "abs_diff_median": np.nan}

    return {
        "n_compared": int(mask.sum()),
        "abs_diff_max": float(abs_diff.max()),
        "abs_diff_p99": float(abs_diff.quantile(0.99)),
        "abs_diff_median": float(abs_diff.median()),
    }

def check_accounting_identity(df: pd.DataFrame, tol_abs: float = 0.05) -> Dict[str, Any]:
    _require_cols(df, ["total_assets", "shareholders_equity", "total_debt"], "check_accounting_identity")
    diff = df["total_assets"] - (df["shareholders_equity"] + df["total_debt"])
    abs_diff = diff.abs()
    return {
        "tol_abs": float(tol_abs),
        "abs_diff_max": float(abs_diff.max()),
        "abs_diff_p99": float(abs_diff.quantile(0.99)),
        "violations": int((abs_diff > tol_abs).sum()),
    }

def flag_outliers(train_df: pd.DataFrame) -> Dict[str, Any]:
    _require_cols(train_df, [TARGET], "flag_outliers")
    # optional cols checked safely
    rc = train_df.loc[train_df[TARGET].notna(), TARGET]

    flags: Dict[str, Any] = {
        "revenue_change_gt_100": int((rc > 100).sum()),
        "revenue_change_lt_-80": int((rc < -80).sum()),
        "leverage_gt_50": int((train_df["leverage"] > 50).sum()) if "leverage" in train_df.columns else None,
        "production_value_negative": int((train_df["production_value"] < 0).sum()) if "production_value" in train_df.columns else None,
    }
    return flags

def check_multi_year_zeros(df: pd.DataFrame, col: str, min_years: int = 2) -> Optional[int]:
    if col not in df.columns:
        return None
    _require_cols(df, [ID_COL], "check_multi_year_zeros")
    zeros = df[col].eq(0)
    counts = zeros.groupby(df[ID_COL]).sum()
    return int((counts >= min_years).sum())


# -----------------------------
# Report builder
# -----------------------------
def run_all_checks_readable(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    tol_assets_identity: float = 0.05,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run data quality checks and return a dict of notebook-friendly tables.
    If verbose=True, prints a readable report via print_report().
    """

    raw: Dict[str, Any] = {}
    raw["keys_years"] = check_keys_and_years(train_df, test_df)
    raw["target_missingness"] = check_target_missingness_is_first_obs(train_df)
    raw["target_formula"] = check_revenue_change_formula(train_df)
    raw["accounting_identity_train"] = check_accounting_identity(train_df, tol_abs=tol_assets_identity)
    raw["accounting_identity_test"] = check_accounting_identity(test_df, tol_abs=tol_assets_identity)
    raw["outlier_flags_train"] = flag_outliers(train_df)

    raw["multi_year_zero_assets_train"] = check_multi_year_zeros(train_df, "total_assets", 2)
    raw["multi_year_zero_debt_train"] = check_multi_year_zeros(train_df, "total_debt", 2)
    raw["multi_year_zero_rev_train"] = check_multi_year_zeros(train_df, "production_value", 2)

    overview = pd.DataFrame([{
        "dup_train": raw["keys_years"]["dup_train"],
        "dup_test": raw["keys_years"]["dup_test"],
        "years_train": str(raw["keys_years"]["years_train"]),
        "years_test": str(raw["keys_years"]["years_test"]),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_companies_train": int(train_df[ID_COL].nunique()),
        "n_companies_test": int(test_df[ID_COL].nunique()),
    }])

    target = pd.DataFrame([raw["target_missingness"]])
    formula = pd.DataFrame([raw["target_formula"]])

    missing_train = pd.concat([
        _series_missing_table(train_df, RAW_BALANCE, "balance_sheet"),
        _series_missing_table(train_df, RAW_INCOME, "income_statement"),
    ], ignore_index=True)

    missing_test = pd.concat([
        _series_missing_table(test_df, RAW_BALANCE, "balance_sheet"),
        _series_missing_table(test_df, RAW_INCOME, "income_statement"),
    ], ignore_index=True)

    ratio_train = _nan_inf_table(train_df, RATIO_COLS)
    ratio_test = _nan_inf_table(test_df, RATIO_COLS)

    identity = pd.DataFrame([
        {"split": "train", **raw["accounting_identity_train"]},
        {"split": "test", **raw["accounting_identity_test"]},
    ])

    outliers = pd.DataFrame([raw["outlier_flags_train"]])
    zeros = pd.DataFrame([{
        "multi_year_zero_total_assets (>=2y)": raw["multi_year_zero_assets_train"],
        "multi_year_zero_total_debt (>=2y)": raw["multi_year_zero_debt_train"],
        "multi_year_zero_production_value (>=2y)": raw["multi_year_zero_rev_train"],
    }])

    # Summary PASS/WARN/FAIL
    summary_rows: List[Dict[str, str]] = []

    dup_ok = (raw["keys_years"]["dup_train"] == 0) and (raw["keys_years"]["dup_test"] == 0)
    summary_rows.append({"check": "No duplicate keys (company_id,fiscal_year)", "status": _status(dup_ok)})

    tm = raw["target_missingness"]
    miss_ok = tm["perfect_match"] and tm["missing_not_first"] == 0 and tm["first_not_missing"] == 0
    summary_rows.append({"check": "Target missing only for first-year rows", "status": _status(miss_ok)})

    fm = raw["target_formula"]
    # If formula check couldn't compare anything, WARN
    formula_pass = (fm["n_compared"] > 0) and (fm["abs_diff_max"] <= 1.0)
    formula_warn = (fm["n_compared"] == 0) or (fm["abs_diff_max"] > 0.1)
    summary_rows.append({"check": "Target formula matches production_value lag", "status": _status(formula_pass, warn_cond=formula_warn)})

    raw_missing_train_any = (missing_train["n_missing"].fillna(0) > 0).any()
    raw_missing_test_any = (missing_test["n_missing"].fillna(0) > 0).any()
    raw_miss_ok = (not raw_missing_train_any) and (not raw_missing_test_any)
    summary_rows.append({"check": "No missing in raw balance/income items", "status": _status(raw_miss_ok)})

    ratio_inf_any = (ratio_train["n_inf"].sum() + ratio_test["n_inf"].sum()) > 0
    summary_rows.append({"check": "No inf values in ratios", "status": _status(not ratio_inf_any, warn_cond=ratio_inf_any)})

    id_ok = (raw["accounting_identity_train"]["violations"] == 0) and (raw["accounting_identity_test"]["violations"] == 0)
    summary_rows.append({"check": "Accounting identity holds (assets ≈ equity + debt)", "status": _status(id_ok)})

    zeros_any = any([(raw["multi_year_zero_assets_train"] or 0) > 0,
                     (raw["multi_year_zero_debt_train"] or 0) > 0,
                     (raw["multi_year_zero_rev_train"] or 0) > 0])
    summary_rows.append({"check": "No multi-year exact zeros in key scale vars", "status": _status(not zeros_any, warn_cond=zeros_any)})

    summary = pd.DataFrame(summary_rows)

    report: Dict[str, Any] = {
        "summary": summary,
        "overview": overview,
        "target_missingness": target,
        "target_formula": formula,
        "missing_raw_train": missing_train,
        "missing_raw_test": missing_test,
        "ratio_nan_inf_train": ratio_train,
        "ratio_nan_inf_test": ratio_test,
        "accounting_identity": identity,
        "outlier_flags_train": outliers,
        "multi_year_zeros_train": zeros,
        "raw": raw,
    }

    if verbose:
        print_report(report)

    return report


def print_report(report: Dict[str, Any]) -> None:
    def _print_df(title: str, df: Any) -> None:
        print("\n" + "=" * len(title))
        print(title)
        print("=" * len(title))
        if isinstance(df, pd.DataFrame):
            print(df.to_string(index=False))
        else:
            print(str(df))

    with pd.option_context(
        "display.max_rows", 200,
        "display.max_columns", 200,
        "display.width", 140,
    ):
        _print_df("DATA QUALITY SUMMARY (PASS/WARN/FAIL)", report["summary"])
        _print_df("OVERVIEW", report["overview"])
        _print_df("TARGET MISSINGNESS (expected: only first year)", report["target_missingness"])
        _print_df("TARGET FORMULA CHECK (abs diff in percentage points)", report["target_formula"])

        # Only show missing rows >0 to keep it readable
        miss_tr = report["missing_raw_train"].copy()
        miss_te = report["missing_raw_test"].copy()
        miss_tr = miss_tr[(miss_tr["n_missing"].fillna(0) > 0)]
        miss_te = miss_te[(miss_te["n_missing"].fillna(0) > 0)]

        _print_df(
            "RAW MISSINGNESS — TRAIN (only rows with missing>0)",
            miss_tr if len(miss_tr) else pd.DataFrame([{"note": "No missing in raw balance/income items"}]),
        )
        _print_df(
            "RAW MISSINGNESS — TEST (only rows with missing>0)",
            miss_te if len(miss_te) else pd.DataFrame([{"note": "No missing in raw balance/income items"}]),
        )

        _print_df("RATIO NaN/INF — TRAIN", report["ratio_nan_inf_train"])
        _print_df("RATIO NaN/INF — TEST", report["ratio_nan_inf_test"])
        _print_df("ACCOUNTING IDENTITY (assets ≈ equity + debt)", report["accounting_identity"])
        _print_df("OUTLIER FLAGS (counts, not errors)", report["outlier_flags_train"])
        _print_df("MULTI-YEAR EXACT ZEROS (>=2 years)", report["multi_year_zeros_train"])
