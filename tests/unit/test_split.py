import pandas as pd
import pytest

from ifc.config import SplitConfig
from ifc.split import apply_holdout_split

def test_apply_holdout_split_happy_path():
    # Tiny synthetic dataset: 2 companies, years 2018-2021
    df = pd.DataFrame(
        {
            "company_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "fiscal_year": [2018, 2019, 2020, 2021, 2018, 2019, 2020, 2021],
            "production_value": [100, 110, 120, 130, 200, 190, 210, 220],
            # Target missing only in first year (2018), present afterwards
            "revenue_change": [None, 10.0, 9.09, 8.33, None, -5.0, 10.53, 4.76],
        }
    )

    split_cfg = SplitConfig(
        strategy="holdout",
        time_col="fiscal_year",
        id_col="company_id",
        target_col="revenue_change",
        shuffle=False,
        history_years=[2018],
        train_years=[2019, 2020],
        val_years=[2021],
    )

    df_history, df_train, df_val = apply_holdout_split(df, split_cfg)

    # Year partitions
    assert set(df_history["fiscal_year"].unique()) == {2018}
    assert set(df_train["fiscal_year"].unique()) == {2019, 2020}
    assert set(df_val["fiscal_year"].unique()) == {2021}

    # No overlap on logical key
    key = ["company_id", "fiscal_year"]
    hist_keys = set(map(tuple, df_history[key].to_numpy()))
    train_keys = set(map(tuple, df_train[key].to_numpy()))
    val_keys = set(map(tuple, df_val[key].to_numpy()))
    assert hist_keys.isdisjoint(train_keys)
    assert hist_keys.isdisjoint(val_keys)
    assert train_keys.isdisjoint(val_keys)

    # Target must be present in train/val
    assert df_train["revenue_change"].notna().all()
    assert df_val["revenue_change"].notna().all()

