from __future__ import annotations

from typing import Tuple, List, Optional
import pandas as pd

from ifc.config import SplitConfig


def apply_forecasting_holdout_split(
    df: pd.DataFrame,
    split_cfg: SplitConfig,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build forecasting dataset with X(t-1) -> y(t) and apply time-aware holdout.

    Returns:

    df_train : rows where fiscal_year in train_target_years, with prev_* features (from t-1) and y=target_col(t)
    df_val   : rows where fiscal_year in val_target_years, with prev_* features (from t-1) and y=target_col(t)

    Notes:

    - Features are renamed to prev_<col> to prevent accidental same-year leakage.
    - Any preprocessing stats must be fit only on df_train (outside this function).
    """
    # Strategy + shuffle guardrails
    if split_cfg.strategy != "forecasting_holdout":
        raise ValueError(
            f"Unsupported split strategy: {split_cfg.strategy}. Expected 'forecasting_holdout'."
        )
    if split_cfg.shuffle:
        raise ValueError("shuffle=True is not allowed for time-aware forecasting holdout splits.")

    id_col = split_cfg.id_col
    time_col = split_cfg.time_col
    target_col = split_cfg.target_col
    time_horizon = int(split_cfg.horizon_years)

    # Required columns
    required = {id_col, time_col, target_col}
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(f"Input df is missing required columns: {sorted(missing_required)}")

    # Duplicate check on logical key
    key = [id_col, time_col]
    dup_count = df.duplicated(subset=key).sum()
    if dup_count != 0:
        raise ValueError(f"Found {dup_count} duplicate rows on logical key {key}.")

    # Target-year split sets
    train_years = list(split_cfg.train_target_years or [])
    val_years = list(split_cfg.val_target_years or [])

    if not train_years or not val_years:
        raise ValueError("train_target_years and val_target_years must be non-empty.")

    set_tr, set_va = set(train_years), set(val_years)
    if set_tr & set_va:
        raise ValueError(f"Year sets must not overlap. overlap(train,val)={sorted(set_tr & set_va)}")

    if max(train_years) >= min(val_years):
        raise ValueError(
            f"Time order violation: max(train_target_years)={max(train_years)} "
            f"must be < min(val_target_years)={min(val_years)}"
        )

    # Drop leakage columns from feature space (if provided)
    drop_cols = list(drop_cols or [])
    df_feat_base = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Define feature columns (everything except id/time/target)
    excluded = {id_col, time_col, target_col}
    feature_cols = [c for c in df_feat_base.columns if c not in excluded]

    # Build prev-year feature table:
    # take rows at year (t-1), rename feature cols to prev_*, then shift year forward (+h) to align to target year t.
    prev_rename = {c: f"prev_{c}" for c in feature_cols}
    df_prev = df_feat_base[[id_col, time_col] + feature_cols].copy()
    df_prev = df_prev.rename(columns=prev_rename)
    df_prev[time_col] = df_prev[time_col] + time_horizon  # align (t-1) features to year t

    # Target table (year t)
    df_tgt = df[[id_col, time_col, target_col]].copy()

    # Merge to create forecasting frame
    df_forecast = df_tgt.merge(df_prev, on=[id_col, time_col], how="left")

    # Require supervised target for train/val
    df_forecast = df_forecast.loc[df_forecast[target_col].notna()].copy()

    # Require at least some prev_* info (otherwise no snapshot for t-1)
    prev_cols = [f"prev_{c}" for c in feature_cols]
    has_prev = df_forecast[prev_cols].notna().any(axis=1)
    df_forecast = df_forecast.loc[has_prev].copy()

    # Split by target year (year column remains target-year)
    df_train = df_forecast.loc[df_forecast[time_col].isin(train_years)].copy()
    df_val = df_forecast.loc[df_forecast[time_col].isin(val_years)].copy()

    if df_train.empty:
        raise ValueError("df_train is empty after building forecasting split. Check years and availability of t-1 features.")
    if df_val.empty:
        raise ValueError("df_val is empty after building forecasting split. Check years and availability of t-1 features.")

    # Final safety: ensure year purity
    if set(df_train[time_col].unique()) - set_tr:
        raise AssertionError("df_train contains years outside train_target_years.")
    if set(df_val[time_col].unique()) - set_va:
        raise AssertionError("df_val contains years outside val_target_years.")

    return df_train, df_val



def build_forecasting_test_frame(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_cfg: SplitConfig,
    drop_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build X_test forecasting frame for target years in split_cfg.test_target_years.

    Logic (horizon_years = h):
      - For target year t, features come from year (t - h), renamed as prev_*
      - We start from test_df keys (company_id, fiscal_year) for the requested target years,
        then left-merge the prev_* features computed from train_df/test_df as needed.

    Returns
    -------
    df_test_forecast : columns = [id_col, time_col] + prev_* (NO target column)
                       rows for fiscal_year in test_target_years
    """
    if split_cfg.strategy != "forecasting_holdout":
        raise ValueError(
            f"Unsupported split strategy: {split_cfg.strategy}. Expected 'forecasting_holdout'."
        )
    if split_cfg.shuffle:
        raise ValueError("shuffle=True is not allowed for time-aware forecasting holdout splits.")

    id_col = split_cfg.id_col
    time_col = split_cfg.time_col
    target_col = split_cfg.target_col
    h = int(split_cfg.horizon_years)

    test_target_years = list(split_cfg.test_target_years or [])
    if not test_target_years:
        raise ValueError("split_cfg.test_target_years must be non-empty to build test frame.")

    # Base keys: ONLY the rows we must predict (from test_df)
    required_keys = {id_col, time_col}
    missing_keys = required_keys - set(test_df.columns)
    if missing_keys:
        raise ValueError(f"test_df is missing required columns: {sorted(missing_keys)}")

    df_base = test_df.loc[test_df[time_col].isin(test_target_years), [id_col, time_col]].copy()

    if df_base.empty:
        raise ValueError(
            f"No rows found in test_df for test_target_years={test_target_years}. "
            f"Available years in test_df: {sorted(test_df[time_col].unique())}"
        )

    # Duplicate check on logical key in base
    dup_count = df_base.duplicated(subset=[id_col, time_col]).sum()
    if dup_count != 0:
        raise ValueError(f"Found {dup_count} duplicate rows on logical key {[id_col, time_col]} in test_df base.")

    drop_cols = list(drop_cols or [])

    # Feature columns schema should match what we used in training:
    # derive from train_df after dropping leakage cols, excluding id/time/target
    train_feat_base = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors="ignore")
    excluded = {id_col, time_col, target_col}
    feature_cols = [c for c in train_feat_base.columns if c not in excluded]

    def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        missing = [c for c in cols if c not in out.columns]
        for c in missing:
            out[c] = pd.NA
        return out

    def _make_prev_table(source_df: pd.DataFrame, feature_year: int) -> pd.DataFrame:
        # Prepare base (drop leakage cols if present)
        src = source_df.drop(columns=[c for c in drop_cols if c in source_df.columns], errors="ignore")

        # Ensure it has id/time
        missing = {id_col, time_col} - set(src.columns)
        if missing:
            raise ValueError(f"Source df missing required columns {sorted(missing)}")

        # Ensure all feature cols exist (fill missing with NA to keep schema stable)
        src = _ensure_columns(src, feature_cols)

        # Select the year we need and rename to prev_*
        df_y = src.loc[src[time_col] == feature_year, [id_col, time_col] + feature_cols].copy()
        rename_map = {c: f"prev_{c}" for c in feature_cols}
        df_y = df_y.rename(columns=rename_map)

        # Shift the time forward to align with target year
        df_y[time_col] = df_y[time_col] + h
        return df_y

    # Build prev feature tables needed for each target year
    prev_tables = []
    for t in sorted(set(test_target_years)):
        feature_year = t - h

        # Decide where to pull feature_year from
        # 2021 -> train_df, 2022 -> test_df, general rule:
        in_train = feature_year in set(train_df[time_col].unique())
        in_test = feature_year in set(test_df[time_col].unique())

        if in_train:
            prev_tables.append(_make_prev_table(train_df, feature_year))
        elif in_test:
            prev_tables.append(_make_prev_table(test_df, feature_year))
        else:
            raise ValueError(
                f"Cannot build prev features for target year {t}: "
                f"feature_year={feature_year} not found in train_df or test_df."
            )

    df_prev_all = pd.concat(prev_tables, axis=0, ignore_index=True)

    # Merge base keys with prev features
    df_test_forecast = df_base.merge(df_prev_all, on=[id_col, time_col], how="left")

    # Safety: no target col in test frame
    if target_col in df_test_forecast.columns:
        df_test_forecast = df_test_forecast.drop(columns=[target_col])

    # Safety: only prev_* besides keys
    non_prev = [
        c for c in df_test_forecast.columns
        if c not in [id_col, time_col] and not c.startswith("prev_")
    ]
    if non_prev:
        raise AssertionError(f"Unexpected non-prev feature columns in test frame: {non_prev}")

    return df_test_forecast
