import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import mean_absolute_error

@st.cache_data
def compute_actuals(df_eval, calendar):
    """Tính toán dữ liệu thực tế 28 ngày cuối cùng cho tất cả items."""
    id_cols = ['item_id', 'store_id']
    day_cols = [c for c in df_eval.columns if c.startswith('d_')]
    last_28 = day_cols[-28:]
    
    df_actual = df_eval.melt(
        id_vars=id_cols, 
        value_vars=last_28, 
        var_name='d', 
        value_name='actual'
    ).merge(calendar[['d', 'date']], on='d', how='left')
    
    df_actual['date'] = pd.to_datetime(df_actual['date'])
    return df_actual

@st.cache_data
def compute_compare(df_forecast, df_actual):
    """Gom nhóm dữ liệu thực tế và dự đoán theo ngày để so sánh tổng quan."""
    df_pred_daily = df_forecast.groupby('ds')['yhat'].sum().reset_index()
    df_actual_daily = df_actual.groupby('date')['actual'].sum().reset_index()
    
    return df_actual_daily.merge(
        df_pred_daily, left_on='date', right_on='ds', how='inner'
    )

@st.cache_data
def compute_item_metrics(df_forecast, df_actual):
    """Tính toán các chỉ số sai số (MAE, RMSE, MAPE) cho từng item cụ thể."""
    df_item_pred = df_forecast.groupby(['item_id', 'store_id', 'ds'])['yhat'].sum().reset_index()
    df_item_act = df_actual.groupby(['item_id', 'store_id', 'date'])['actual'].sum().reset_index()
    
    df_item = df_item_act.merge(
        df_item_pred, left_on=['item_id', 'store_id', 'date'], 
        right_on=['item_id', 'store_id', 'ds'], how='inner'
    )
    
    metrics = df_item.groupby(['item_id', 'store_id']).apply(
        lambda g: pd.Series({
            'MAE' : mean_absolute_error(g['actual'], g['yhat']),
            'RMSE': np.sqrt(((g['actual'] - g['yhat']) ** 2).mean()),
            'MAPE': (abs(g['actual'] - g['yhat']) / (g['actual'] + 1)).mean() * 100,
            'total_actual' : g['actual'].sum(),
            'total_predicted': g['yhat'].sum(),
        })
    ).reset_index()
    return metrics

@st.cache_data
def get_item_actual(df_eval, calendar, item_id, store_id):
    """Lấy 28 ngày actual của 1 item từ evaluation file."""
    id_val = f"{item_id}_{store_id}_evaluation"
    day_cols = [c for c in df_eval.columns if c.startswith('d_')]
    last_28 = day_cols[-28:]

    row = df_eval[df_eval['id'] == id_val]
    if row.empty:
        return pd.DataFrame()

    vals = row[last_28].values.flatten()
    dates = calendar[calendar['d'].isin(last_28)].sort_values('d')['date'].values

    return pd.DataFrame({'date': pd.to_datetime(dates), 'actual': vals})

@st.cache_data
def get_item_history(df_features, item_id, store_id):
    """Lấy toàn bộ lịch sử features của 1 item."""
    return df_features[
        (df_features['item_id'] == item_id) & (df_features['store_id'] == store_id)
    ][['date', 'sales', 'sell_price', 'snap', 'event_name_1', 'event_type_1']]\
    .sort_values('date').reset_index(drop=True)

@st.cache_data
def build_store_metrics(df_forecast, df_eval, calendar):
    """Tính toán metrics tổng hợp theo từng cửa hàng (store)."""
    id_cols = ['item_id', 'store_id']
    day_cols = [c for c in df_eval.columns if c.startswith('d_')]
    last_28 = day_cols[-28:]
    
    df_actual = df_eval.melt(
        id_vars=id_cols, value_vars=last_28, var_name='d', value_name='actual'
    ).merge(calendar[['d', 'date']], on='d', how='left')
    df_actual['date'] = pd.to_datetime(df_actual['date'])
    
    df_pred = df_forecast.rename(columns={'ds': 'date'})
    df_merge = df_actual.merge(df_pred, on=['item_id', 'store_id', 'date'], how='inner')
    
    metrics = df_merge.groupby('store_id').apply(lambda g: pd.Series({
        'total_actual' : g['actual'].sum(),
        'total_predicted' : g['yhat'].sum(),
        'MAE' : mean_absolute_error(g['actual'], g['yhat']),
        'RMSE' : np.sqrt(((g['actual'] - g['yhat']) ** 2).mean()),
        'MAPE' : (abs(g['actual'] - g['yhat']) / (g['actual'] + 1)).mean() * 100,
        'bias' : (g['yhat'] - g['actual']).mean(),
        'n_items': g['item_id'].nunique(),
    })).reset_index()
    
    metrics['state'] = metrics['store_id'].str[:2]
    metrics['pct_diff'] = (metrics['total_predicted'] - metrics['total_actual']) / metrics['total_actual'] * 100
    
    return metrics, df_merge, df_actual

@st.cache_data
def build_daily_by_store(df_merge):
    """Tính tổng actual và predicted nhóm theo store và theo ngày."""
    return df_merge.groupby(['store_id', 'date']).agg(
        actual=('actual', 'sum'), predicted=('yhat', 'sum')
    ).reset_index()

@st.cache_data
def build_cat_by_store(df_merge, df_eval):
    cat_map = df_eval[['item_id', 'cat_id']].drop_duplicates()
    df = df_merge.merge(cat_map, on='item_id', how='left')
    return df.groupby(['store_id', 'cat_id']).agg(
        actual=('actual', 'sum'),
        predicted=('yhat', 'sum')
    ).reset_index()


@st.cache_data
def build_history_by_store(df_features):
    return df_features.groupby(['store_id', 'date'])['sales'].sum().reset_index()