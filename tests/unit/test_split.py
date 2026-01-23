import pandas as pd
from ifc.config import SplitConfig
from ifc.split import apply_forecasting_holdout_split, build_forecasting_test_frame


def test_apply_forecasting_holdout_split_happy_path():
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
        strategy="forecasting_holdout",
        time_col="fiscal_year",
        id_col="company_id",
        target_col="revenue_change",
        shuffle=False,
        horizon_years=1,
        train_target_years=[2019, 2020],
        val_target_years=[2021],
        test_target_years=[2022, 2023],
    )

    df_train, df_val = apply_forecasting_holdout_split(df, split_cfg)

    # Year partitions (target-years)
    assert set(df_train["fiscal_year"].unique()) == {2019, 2020}
    assert set(df_val["fiscal_year"].unique()) == {2021}

    # Target must be present in train/val
    assert df_train["revenue_change"].notna().all()
    assert df_val["revenue_change"].notna().all()

    # Only prev_* features besides key + target
    allowed = {split_cfg.id_col, split_cfg.time_col, split_cfg.target_col}
    non_prev = [c for c in df_train.columns if c not in allowed and not c.startswith("prev_")]
    assert non_prev == []

    # Check alignment for one feature: prev_production_value
    # Company 1:
    # y_2019 should use production_value_2018 = 100
    pv_2019 = df_train.loc[
        (df_train.company_id == 1) & (df_train.fiscal_year == 2019), "prev_production_value"
    ].iloc[0]
    assert pv_2019 == 100

    # y_2020 should use production_value_2019 = 110
    pv_2020 = df_train.loc[
        (df_train.company_id == 1) & (df_train.fiscal_year == 2020), "prev_production_value"
    ].iloc[0]
    assert pv_2020 == 110

    # y_2021 (val) should use production_value_2020 = 120
    pv_2021 = df_val.loc[
        (df_val.company_id == 1) & (df_val.fiscal_year == 2021), "prev_production_value"
    ].iloc[0]
    assert pv_2021 == 120


def test_build_forecasting_test_frame_happy_path():
    # Train data covers up to 2021; Test features cover 2022-2023
    train_df = pd.DataFrame(
        {
            "company_id": [1, 1, 1, 1],
            "fiscal_year": [2018, 2019, 2020, 2021],
            "production_value": [100, 110, 120, 130],
            "revenue_change": [None, 10.0, 9.09, 8.33],
            # Leakage label (should be droppable if present)
            "financial_health_class": [0, 1, 1, 1],
        }
    )

    test_df = pd.DataFrame(
        {
            "company_id": [1, 1],
            "fiscal_year": [2022, 2023],
            # Note: test has same-year values, but we must NOT use them directly.
            "production_value": [999, 888],
        }
    )

    split_cfg = SplitConfig(
        strategy="forecasting_holdout",
        time_col="fiscal_year",
        id_col="company_id",
        target_col="revenue_change",
        shuffle=False,
        horizon_years=1,
        train_target_years=[2019, 2020],
        val_target_years=[2021],
        test_target_years=[2022, 2023],
    )

    df_test_forecast = build_forecasting_test_frame(
        train_df=train_df,
        test_df=test_df,
        split_cfg=split_cfg,
        drop_cols=["financial_health_class"],
    )

    # Must include only target years
    assert set(df_test_forecast["fiscal_year"].unique()) == {2022, 2023}

    # Must NOT include target column
    assert "revenue_change" not in df_test_forecast.columns

    # Only prev_* besides keys
    allowed = {split_cfg.id_col, split_cfg.time_col}
    non_prev = [c for c in df_test_forecast.columns if c not in allowed and not c.startswith("prev_")]
    assert non_prev == []

    
    pv_2022 = df_test_forecast.loc[df_test_forecast.fiscal_year == 2022, "prev_production_value"].iloc[0]
    assert pv_2022 == 130


    pv_2023 = df_test_forecast.loc[df_test_forecast.fiscal_year == 2023, "prev_production_value"].iloc[0]
    assert pv_2023 == 999
