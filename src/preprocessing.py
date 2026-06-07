import pandas as pd

def load_data():
    sales = pd.read_csv('../data/raw/sales_train_validation.csv')
    cal = pd.read_csv('../data/raw/calendar.csv')
    prices = pd.read_csv('../data/raw/sell_prices.csv')

    return sales, cal, prices

def clean_sales(sales_df: pd.DataFrame):
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    d_cols = list(sales_df.drop(columns=id_cols))

    sales_df = sales_df.melt(id_vars=id_cols, value_vars=d_cols, var_name='d', value_name='sales')
    
    return sales_df

def clean_cal(cal_df: pd.DataFrame):
    cal_df['date'] = pd.to_datetime(cal_df['date'])
    cal_df['event_name_1'] = cal_df['event_name_1'].fillna('No_Event')
    cal_df['event_type_1'] = cal_df['event_type_1'].fillna('No_Event')
    cal_df['event_name_2'] = cal_df['event_name_2'].fillna('No_Event')
    cal_df['event_type_2'] = cal_df['event_type_2'].fillna('No_Event')

    return cal_df

def clean_sell_prices(df: pd.DataFrame):
    df['sell_price'] = df.groupby('item_id')['sell_price'].ffill()
    df['sell_price'] = df.groupby('item_id')['sell_price'].bfill()

    return df

def merge_files(cleaned_sales_df: pd.DataFrame, cleaned_cal_df: pd.DataFrame, cleaned_prices_df: pd.DataFrame):
    df = cleaned_sales_df.merge(cleaned_cal_df, on='d', how='left')
    df = df.merge(cleaned_prices_df, on=['item_id', 'wm_yr_wk', 'store_id'], how='left')
    
    return df

def save_cleaned_data(sales_df: pd.DataFrame, cal_df: pd.DataFrame, prices_df: pd.DataFrame):
    sales_df.to_parquet('../data/processed/cleaned_sales.parquet', index=False)
    cal_df.to_parquet('../data/processed/cleaned_cal.parquet', index=False)
    prices_df.to_parquet('../data/processed/cleaned_prices.parquet', index=False)

if __name__ == '__main__':
    sales, cal, prices = load_data()
    sales = clean_sales(sales)

    print(sales.head(10))
