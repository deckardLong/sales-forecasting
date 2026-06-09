import os
import time
import joblib
import pandas as pd
from prophet import Prophet
from tqdm.notebook import tqdm

def train_model(df: pd.DataFrame, periods=28):
    models = {}
    all_forecasts = []
    groups = list(df.groupby(['item_id', 'store_id']))
    forecast_path = '../models/forecast/forecast_results.parquet'
    if os.path.exists(forecast_path):
        df_existing = pd.read_parquet(forecast_path)
        done_pairs = set(zip(df_existing['item_id'], df_existing['store_id']))
        all_forecasts.append(df_existing)
        print(f'Resume: already have {len(done_pairs)}/{len(groups)} items')
    else:
        done_pairs = set()
        print(f'New train: {len(groups)} items')

    t1 = time.time()
    for idx, ((item_id, store_id), df_group) in enumerate(tqdm(groups, desc='Training Prophet Models')):
        
        # Continue if already trained
        if (item_id, store_id) in done_pairs:
            model_path = f'../models/items/{item_id}_{store_id}.pkl'
            if os.path.exists(model_path):
                models[(item_id, store_id)] = joblib.load(model_path)
            continue

        df_prophet = df_group[['sales', 'date', 
                               'event_name_1', 'event_type_1', 
                               'event_name_2', 'event_type_2', 
                               'sell_price', 'snap']].rename(columns={'date': 'ds', 'sales': 'y'})
        model = Prophet()
        model.add_regressor('sell_price')
        model.add_regressor('snap')
        model.add_regressor('event_name_1')
        model.add_regressor('event_name_2')
        model.add_regressor('event_type_1')
        model.add_regressor('event_type_2')
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods)
        future = future.merge(
            df_prophet[['ds', 'sell_price', 'snap',
                        'event_name_1', 'event_type_1', 
                        'event_name_2', 'event_type_2']],
            on='ds', how='left'
        )
        future[['sell_price', 'snap', 
                'event_name_1', 'event_type_1', 
                'event_name_2', 'event_type_2']] = \
                future[['sell_price', 'snap', 
                        'event_name_1', 'event_type_1', 
                        'event_name_2', 'event_type_2']].ffill()
        forecast = model.predict(future)

        forecast_tail = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).copy()
        forecast_tail['item_id'] = item_id
        forecast_tail['store_id'] = store_id
        all_forecasts.append(forecast_tail)

        path = f'../models/items/{item_id}_{store_id}.pkl'
        joblib.dump(model, path)
        models[(item_id, store_id)] = model

        if (idx + 1) % 100 == 0:
            pd.concat(all_forecasts, ignore_index=True).to_parquet(forecast_path, index=False)
    t2 = time.time()
    print(f'Training time: {t2 - t1}s')

    df_forecast = pd.concat(all_forecasts, ignore_index=True)
    df_forecast.to_parquet(forecast_path, index=False)
    print('Forecast saved completely!')
    
    return models

