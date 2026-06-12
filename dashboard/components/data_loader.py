"""
components/data_loader.py
──────────────────────────
Load và cache tất cả dataframes dùng chung cho 3 pages.
"""

import streamlit as st
import pandas as pd

@st.cache_data
def load_forecast():
    df = pd.read_parquet("models/forecast/forecast_results.parquet")
    df["ds"]   = pd.to_datetime(df["ds"])
    df["yhat"] = df["yhat"].clip(lower=0)
    return df


@st.cache_data
def load_evaluation():
    return pd.read_csv("data/raw/sales_train_evaluation.csv")


@st.cache_data
def load_calendar():
    df = pd.read_csv("data/raw/calendar.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_features():
    df = pd.read_parquet("data/features/m5_features.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_actuals():
    """
    Melt 28 ngày cuối của evaluation về long format.
    Dùng chung cho cả 3 pages.
    """
    df_eval    = load_evaluation()
    calendar   = load_calendar()

    id_cols  = ["item_id", "store_id"]
    day_cols = [c for c in df_eval.columns if c.startswith("d_")]
    last_28  = day_cols[-28:]

    df_actual = df_eval.melt(
        id_vars=id_cols, value_vars=last_28,
        var_name="d", value_name="actual"
    ).merge(calendar[["d", "date"]], on="d", how="left")

    df_actual["date"] = pd.to_datetime(df_actual["date"])
    return df_actual


@st.cache_data
def load_all():
    """
    Load tất cả cùng lúc — dùng khi page cần đủ 4 dataframes.
    Returns: (df_forecast, df_eval, calendar, df_features)
    """
    return (
        load_forecast(),
        load_evaluation(),
        load_calendar(),
        load_features(),
    )