import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_num_attr(df: pd.DataFrame):
    scaler = MinMaxScaler()
    df['sell_price'] = scaler.fit_transform(df[['sell_price']])

    return df

def normalize_cat_attr(df: pd.DataFrame):
    df['event_name_1'] = (df['event_name_1'] != 'No_Event').astype(int) 
    df['event_name_2'] = (df['event_name_2'] != 'No_Event').astype(int)
    df['event_type_1'] = (df['event_type_1'] != 'No_Event').astype(int) 
    df['event_type_2'] = (df['event_type_2'] != 'No_Event').astype(int)  

    return df