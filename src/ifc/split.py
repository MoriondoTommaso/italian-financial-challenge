from __future__ import annotations
from dataclasses import asdict
from typing import Tuple
import pandas as pd
from ifc.config import SplitConfig

def apply_holdout_split(df: pd.DataFrame, split_cfg: SplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply a time-aware holdout split for Task 3.

    Returns

    df_history : rows where fiscal_year in history_years (can include target missing; used for lags only)
    df_train   : rows where fiscal_year in train_years AND target is not missing
    df_val     : rows where fiscal_year in val_years AND target is not missing
    """ 
    # safety check for split method
    if split_cfg.strategy != "holdout":
        raise ValueError(f"Unsupported split strategy: {split_cfg.strategy}. Expected 'holdout")
    if split_cfg.shuffle:
        raise ValueError("shuffle=True is not allowed for time-aware holdout splits.")
    
    id_col = split_cfg.id_col
    time_col = split_cfg.time_col
    target_col = split_cfg.target_col

    required_cols = {id_col, time_col}
    missing_required = required_cols - set(df.columns)
    if missing_required:
        raise ValueError(f"Input df is missing required columns: {sorted(missing_required)}")
    
    # safety check for years
    history_years = list(split_cfg.history_years or [])
    train_years = list(split_cfg.train_years or [])
    val_years = list(split_cfg.val_years or [])

    if not train_years or not val_years:
        raise ValueError(f"train_years and val_years must be non-empty. Got: {asdict(split_cfg)}")

    set_h, set_tr, set_va = set(history_years), set(train_years), set(val_years)
    if (set_h & set_tr) or (set_h & set_va) or (set_tr & set_va):
        raise ValueError(
            "Year sets must not overlap. "
            f"overlap(history,train)={sorted(set_h & set_tr)}, "
            f"overlap(history,val)={sorted(set_h & set_va)}, "
            f"overlap(train,val)={sorted(set_tr & set_va)}"
        )

    if max(train_years) >= min(val_years):
        raise ValueError(f"Time order violation: max(train_years)={max(train_years)} must be < min(val_years)={min(val_years)}")
    

    # Duplicate check on logical key
    key = [id_col, time_col]
    dup_count = df.duplicated(subset=key).sum()
    if dup_count != 0:
        raise ValueError(f"Found {dup_count} duplicate rows on logical key {key}.")

    # Split
    df_history = df.loc[df[time_col].isin(history_years)].copy()
    df_train = df.loc[df[time_col].isin(train_years)].copy()
    df_val = df.loc[df[time_col].isin(val_years)].copy()

    # Target checks (train/val must be supervised)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in df columns.")

    df_train = df_train.loc[df_train[target_col].notna()].copy()
    df_val = df_val.loc[df_val[target_col].notna()].copy()

    if df_train.empty:
        raise ValueError("After dropping missing targets, df_train is empty. Check target availability and split years.")
    if df_val.empty:
        raise ValueError("After dropping missing targets, df_val is empty. Check target availability and split years.")

    # Final safety: ensure year purity
    if set(df_history[time_col].unique()) - set_h:
        raise AssertionError("df_history contains years outside history_years.")
    if set(df_train[time_col].unique()) - set_tr:
        raise AssertionError("df_train contains years outside train_years.")
    if set(df_val[time_col].unique()) - set_va:
        raise AssertionError("df_val contains years outside val_years.")

    return df_history, df_train, df_val
    
    