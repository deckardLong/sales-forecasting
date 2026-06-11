import pandas as pd
import matplotlib.pyplot as plt

def plot_actual_vs_predicted(df_forecast: pd.DataFrame, df_eval: pd.DataFrame, cal: pd.DataFrame, periods=28):
    d_cols = [c for c in df_eval.columns if c.startswith('d_')]
    last_28 = d_cols[-periods:]

    df_actual_long = df_eval.melt(
        id_vars=['item_id', 'store_id'],
        value_vars=last_28,
        var_name='d',
        value_name='actual'
    )

    # Actual
    df_actual_long = df_actual_long.merge(cal[['d', 'date']], on='d', how='left')
    df_actual_daily = df_actual_long.groupby('date')['actual'].sum().reset_index()
    df_actual_daily['date'] = pd.to_datetime(df_actual_daily['date'])

    # Predicted
    df_pred_daily = df_forecast.groupby('ds')['yhat'].sum().reset_index()
    df_pred_daily['ds'] = pd.to_datetime(df_pred_daily['ds'])

    # Merge 2 dataframes
    df_compare = df_actual_daily.merge(
        df_pred_daily,
        left_on='date', right_on='ds',
        how='inner'
    )

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Actual
    ax.plot(df_compare['date'], df_compare['actual'],
            color='steelblue', linewidth=2, 
            marker='o', markersize=4, label='Actual')

    # Predicted    
    ax.plot(df_compare['date'], df_compare['yhat'],
            color='tomato', linewidth=2, 
            marker='o', markersize=4, label='Predicted')
    
    ax.set_title('Daily Total Sales - Actual vs Predicted (28 days)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../figures/forecast/actual_predicted_sales.png')
    plt.show()

    return df_compare