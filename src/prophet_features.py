import pandas as pd
import numpy as np

def combine_snap_features(df: pd.DataFrame):
    df['snap'] = (
        df['snap_CA'] * (df['state_id'] == 'CA') +
        df['snap_TX'] * (df['state_id'] == 'TX') +
        df['snap_WI'] * (df['state_id'] == 'WI')
    )
    return df

def select_features(df: pd.DataFrame):
    drop_cols = ['dept_id', 'cat_id', 'state_id', 'd', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'snap_CA', 'snap_TX', 'snap_WI']
    df = df.drop(columns=drop_cols)
    
    return df