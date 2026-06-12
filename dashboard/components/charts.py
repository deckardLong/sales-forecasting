import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Cấu hình layout mặc định dùng chung cho tất cả các biểu đồ Plotly
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#94a3b8', size=12),
    xaxis=dict(gridcolor='#1e2535', linecolor='#1e2535'),
    yaxis=dict(gridcolor='#1e2535', linecolor='#1e2535'),
    margin=dict(l=16, r=16, t=40, b=16),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1e2535', borderwidth=1),
    hoverlabel=dict(bgcolor='#161b27', bordercolor='#1e2535', font_color='#e2e8f0'),
)

# Mã màu cho từng cửa hàng (Store)
STORE_COLORS = {
    'CA_1': '#3b82f6', 'CA_2': '#60a5fa', 'CA_3': '#93c5fd', 'CA_4': '#bfdbfe',
    'TX_1': '#8b5cf6', 'TX_2': '#a78bfa', 'TX_3': '#c4b5fd',
    'WI_1': '#06b6d4', 'WI_2': '#22d3ee', 'WI_3': '#67e8f9',
}

# Mã màu cho từng bang (State)
STATE_COLORS = {'CA': '#3b82f6', 'TX': '#8b5cf6', 'WI': '#06b6d4'}

def plot_item_forecast(df_history, df_actual, df_pred, item_id, store_id, calendar):
    """
    Vẽ biểu đồ forecast chi tiết cho một item (được gọi từ 02_item_detail.py)
    Bao gồm: Lịch sử, Actuals, Dự đoán, Vùng Confidence Interval và các Ngày Lễ.
    """
    df_hist_tail = df_history.tail(180) # Lấy 180 ngày gần nhất
    
    # Lấy thông tin ngày lễ trong khoảng thời gian dự đoán
    holiday_dates = calendar[
        (calendar['date'].isin(df_pred['ds'])) & 
        (calendar['event_name_1'].notna()) & 
        (calendar['event_name_1'] != 'No_Event')
    ][['date', 'event_name_1']].drop_duplicates()
    
    fig = go.Figure()
    
    # Vẽ vùng confidence interval (vùng bóng mờ biểu thị upper/lower bound)
    fig.add_trace(go.Scatter(
        x=pd.concat([df_pred['ds'], df_pred['ds'][::-1]]),
        y=pd.concat([df_pred['yhat_upper'], df_pred['yhat_lower'][::-1]]),
        fill='toself', 
        fillcolor='rgba(139, 92, 246, 0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval', 
        hoverinfo='skip'
    ))
    
    # Vẽ lịch sử (Historical Sales)
    fig.add_trace(go.Scatter(
        x=df_hist_tail['date'], 
        y=df_hist_tail['sales'],
        name='Historical', 
        mode='lines',
        line=dict(color='#475569', width=1.5),
        hovertemplate='<b>Historical</b><br>%{x|%Y-%m-%d}<br>Sales: %{y}<extra></extra>'
    ))
    
    # Vẽ dữ liệu thực tế 28 ngày cuối (Actual)
    if not df_actual.empty:
        fig.add_trace(go.Scatter(
            x=df_actual['date'], 
            y=df_actual['actual'],
            name='Actual (Last 28d)', 
            mode='lines+markers',
            line=dict(color='#06b6d4', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Actual</b><br>%{x|%Y-%m-%d}<br>Sales: %{y}<extra></extra>'
        ))
    
    # Vẽ dữ liệu dự đoán (Predicted)
    fig.add_trace(go.Scatter(
        x=df_pred['ds'], 
        y=df_pred['yhat'],
        name='Predicted', 
        mode='lines+markers',
        line=dict(color='#8b5cf6', width=2, dash='dot'),
        marker=dict(size=6),
        hovertemplate='<b>Predicted</b><br>%{x|%Y-%m-%d}<br>Sales: %{y:.2f}<extra></extra>'
    ))
    
    # Đánh dấu các ngày lễ bằng ngôi sao (Holidays)
    if not holiday_dates.empty:
        fig.add_trace(go.Scatter(
            x=holiday_dates['date'],
            y=[df_pred['yhat'].max() * 1.05] * len(holiday_dates),
            name='Holidays',
            mode='markers',
            marker=dict(color='#fbbf24', size=8, symbol='star'),
            text=holiday_dates['event_name_1'],
            hovertemplate='<b>Holiday</b><br>%{x|%Y-%m-%d}<br>%{text}<extra></extra>'
        ))
        
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    return fig

def plot_actual_vs_predicted(df_compare):
    fig = go.Figure()
 
    fig.add_trace(go.Scatter(
        x=df_compare['date'], y=df_compare['actual'],
        name='Actual', mode='lines+markers',
        line=dict(color='#60a5fa', width=2),
        marker=dict(size=5, color='#60a5fa'),
        hovertemplate='<b>Actual</b><br>%{x|%b %d}<br>%{y:,.0f}<extra></extra>'
    ))
 
    fig.add_trace(go.Scatter(
        x=df_compare['date'], y=df_compare['yhat'],
        name='Predicted', mode='lines+markers',
        line=dict(color='#f472b6', width=2, dash='dot'),
        marker=dict(size=5, color='#f472b6'),
        hovertemplate='<b>Predicted</b><br>%{x|%b %d}<br>%{y:,.0f}<extra></extra>'
    ))
 
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='Total Daily Sales — Actual vs Predicted (28 Days)', font=dict(size=14, color='#e2e8f0')),
        xaxis_title='Date',
        yaxis_title='Units Sold',
        hovermode='x unified',
        height=360
    )
    return fig
 
 
def plot_historical_trend(df_features):
    df_daily = df_features.groupby('date')['sales'].sum().reset_index()
    df_daily['ma_7']  = df_daily['sales'].rolling(7).mean()
    df_daily['ma_30'] = df_daily['sales'].rolling(30).mean()
 
    fig = go.Figure()
 
    fig.add_trace(go.Scatter(
        x=df_daily['date'], y=df_daily['sales'],
        name='Daily Sales', mode='lines',
        line=dict(color='#3b82f6', width=1),
        opacity=0.4,
        hovertemplate='%{x|%b %d, %Y}<br>%{y:,.0f}<extra></extra>'
    ))
 
    fig.add_trace(go.Scatter(
        x=df_daily['date'], y=df_daily['ma_7'],
        name='7-day MA', mode='lines',
        line=dict(color='#60a5fa', width=2),
        hovertemplate='7-day MA<br>%{y:,.0f}<extra></extra>'
    ))
 
    fig.add_trace(go.Scatter(
        x=df_daily['date'], y=df_daily['ma_30'],
        name='30-day MA', mode='lines',
        line=dict(color='#f472b6', width=2),
        hovertemplate='30-day MA<br>%{y:,.0f}<extra></extra>'
    ))
 
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='Historical Sales Trend (Full Dataset)', font=dict(size=14, color='#e2e8f0')),
        xaxis_title='Date',
        yaxis_title='Units Sold',
        height=320
    )
    return fig
 
 
def plot_store_comparison(df_actual):
    df_store = df_actual.groupby('store_id')['actual'].sum().reset_index()\
                        .sort_values('actual', ascending=True)
 
    colors = ['#3b82f6', '#8b5cf6', '#06b6d4', '#f472b6',
              '#34d399', '#fbbf24', '#f87171', '#a78bfa',
              '#38bdf8', '#4ade80']
 
    fig = go.Figure(go.Bar(
        x=df_store['actual'],
        y=df_store['store_id'],
        orientation='h',
        marker=dict(color=colors[:len(df_store)], opacity=0.85),
        hovertemplate='<b>%{y}</b><br>%{x:,.0f} units<extra></extra>'
    ))
 
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='Total Actual Sales by Store (28 Days)', font=dict(size=14, color='#e2e8f0')),
        xaxis_title='Units Sold',
        height=320
    )
    return fig
 
 
def plot_category_breakdown(df_actual, df_eval):
    df_cat = df_eval[['item_id', 'cat_id']].drop_duplicates()
    df_merged = df_actual.merge(df_cat, on='item_id', how='left')
    df_cat_sum = df_merged.groupby('cat_id')['actual'].sum().reset_index()
 
    fig = go.Figure(go.Pie(
        labels=df_cat_sum['cat_id'],
        values=df_cat_sum['actual'],
        hole=0.6,
        marker=dict(colors=['#3b82f6', '#8b5cf6', '#f472b6']),
        textinfo='label+percent',
        textfont=dict(color='#e2e8f0', size=12),
        hovertemplate='<b>%{label}</b><br>%{value:,.0f} units<br>%{percent}<extra></extra>'
    ))
 
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='Sales by Category', font=dict(size=14, color='#e2e8f0')),
        showlegend=False,
        height=320
    )
    return fig
 
 
def plot_error_distribution(item_metrics):
    fig = go.Figure(go.Histogram(
        x=item_metrics['MAPE'].clip(upper=50),
        nbinsx=40,
        marker=dict(color='#3b82f6', opacity=0.8, line=dict(color='#1e2535', width=0.5)),
        hovertemplate='MAPE: %{x:.1f}%<br>Count: %{y}<extra></extra>'
    ))
 
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='MAPE Distribution Across All Items', font=dict(size=14, color='#e2e8f0')),
        xaxis_title='MAPE (%)',
        yaxis_title='Number of Items',
        height=280
    )
    return fig

def plot_components(df_history):
    """Trend + Weekly + Monthly seasonality."""
    df = df_history.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['dow']   = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # Trend — rolling 30 ngày
    df['trend'] = df['sales'].rolling(30, center=True).mean()

    # Weekly pattern
    weekly = df.groupby('dow')['sales'].mean()
    dow_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

    # Monthly pattern
    monthly = df.groupby('month')['sales'].mean()
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec']

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Long-term Trend', 'Day of Week Pattern', 'Monthly Pattern'],
        horizontal_spacing=0.08
    )

    # Trend
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['trend'],
        mode='lines', line=dict(color='#8b5cf6', width=2),
        showlegend=False,
        hovertemplate='%{x|%b %Y}<br>%{y:,.1f}<extra></extra>'
    ), row=1, col=1)

    # Weekly
    fig.add_trace(go.Bar(
        x=dow_labels, y=weekly.values,
        marker=dict(
            color=weekly.values,
            colorscale=[[0,'#1e2535'],[1,'#8b5cf6']],
            showscale=False
        ),
        showlegend=False,
        hovertemplate='%{x}<br>Avg: %{y:,.1f}<extra></extra>'
    ), row=1, col=2)

    # Monthly
    fig.add_trace(go.Bar(
        x=month_labels, y=monthly.values,
        marker=dict(
            color=monthly.values,
            colorscale=[[0,'#1e2535'],[1,'#06b6d4']],
            showscale=False
        ),
        showlegend=False,
        hovertemplate='%{x}<br>Avg: %{y:,.1f}<extra></extra>'
    ), row=1, col=3)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=280,
        title=dict(text='Seasonality Components', font=dict(size=14, color='#e2e8f0'))
    )
    fig.update_annotations(font=dict(color='#64748b', size=11))

    for i in range(1, 4):
        fig.update_xaxes(gridcolor='#1e2535', linecolor='#1e2535', row=1, col=i)
        fig.update_yaxes(gridcolor='#1e2535', linecolor='#1e2535', row=1, col=i)

    return fig

def plot_price_vs_sales(df_history):
    """Scatter: sell_price vs sales."""
    df = df_history[df_history['sell_price'] > 0].copy()

    fig = go.Figure(go.Scatter(
        x=df['sell_price'], y=df['sales'],
        mode='markers',
        marker=dict(
            color=df['sales'],
            colorscale=[[0,'#1e2535'],[0.5,'#8b5cf6'],[1,'#f472b6']],
            size=4, opacity=0.6,
            showscale=False
        ),
        hovertemplate='Price: $%{x:.2f}<br>Sales: %{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='Price vs Sales Correlation', font=dict(size=14, color='#e2e8f0')),
        xaxis_title='Sell Price ($)',
        yaxis_title='Units Sold',
        height=280
    )
    return fig


def plot_residuals(df_compare_item):
    """Residual plot: actual - predicted theo ngày."""
    df_compare_item['residual'] = df_compare_item['actual'] - df_compare_item['yhat']
    colors = ['#34d399' if r >= 0 else '#f87171' for r in df_compare_item['residual']]

    fig = go.Figure(go.Bar(
        x=df_compare_item['date'],
        y=df_compare_item['residual'],
        marker_color=colors,
        hovertemplate='%{x|%b %d}<br>Residual: %{y:+.1f}<extra></extra>'
    ))

    fig.add_hline(y=0, line_color='#475569', line_width=1)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='Forecast Residuals (Actual − Predicted)', font=dict(size=14, color='#e2e8f0')),
        xaxis_title='Date',
        yaxis_title='Residual',
        height=240
    )
    return fig

# ── Insights generator ─────────────────────────────────────────────────────────
def generate_insights(df_history, df_compare_item, metrics):
    insights = []

    # Trend insight
    first_half = df_history['sales'].iloc[:len(df_history)//2].mean()
    second_half = df_history['sales'].iloc[len(df_history)//2:].mean()
    pct_change  = (second_half - first_half) / (first_half + 1) * 100
    if pct_change > 5:
        insights.append(("📈 Upward Trend", f"Sales grew {pct_change:.1f}% in the second half of the history period."))
    elif pct_change < -5:
        insights.append(("📉 Downward Trend", f"Sales declined {abs(pct_change):.1f}% in the second half of the history period."))
    else:
        insights.append(("➡️ Stable Trend", f"Sales remained relatively stable ({pct_change:+.1f}% change over time)."))

    # Weekend insight
    df_history['dow'] = pd.to_datetime(df_history['date']).dt.dayofweek
    weekend_avg = df_history[df_history['dow'] >= 5]['sales'].mean()
    weekday_avg = df_history[df_history['dow'] < 5]['sales'].mean()
    if weekend_avg > weekday_avg * 1.1:
        insights.append(("📅 Weekend Peak", f"Weekend sales are {((weekend_avg/weekday_avg)-1)*100:.0f}% higher than weekdays."))
    elif weekday_avg > weekend_avg * 1.1:
        insights.append(("📅 Weekday Peak", f"Weekday sales are {((weekday_avg/weekend_avg)-1)*100:.0f}% higher than weekends."))

    # Accuracy insight
    mape = metrics['mape']
    if mape < 10:
        insights.append(("✅ High Accuracy", f"Model MAPE of {mape:.1f}% indicates excellent forecast accuracy."))
    elif mape < 20:
        insights.append(("⚠️ Moderate Accuracy", f"Model MAPE of {mape:.1f}% — forecast is usable but has room to improve."))
    else:
        insights.append(("❌ Low Accuracy", f"Model MAPE of {mape:.1f}% is high. Consider more data or model tuning."))

    # Zero sales insight
    zero_pct = (df_history['sales'] == 0).mean() * 100
    if zero_pct > 30:
        insights.append(("⚠️ Sparse Sales", f"{zero_pct:.0f}% of days had zero sales — intermittent demand pattern."))

    return insights

def plot_store_actual_vs_pred(store_metrics):
    stores = store_metrics.sort_values('total_actual', ascending=False)
    colors = [STORE_COLORS.get(s, '#475569') for s in stores['store_id']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Actual', x=stores['store_id'], y=stores['total_actual'],
        marker_color=colors, opacity=0.9,
        hovertemplate='<b>%{x}</b><br>Actual: %{y:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Bar(
        name='Predicted', x=stores['store_id'], y=stores['total_predicted'],
        marker_color=colors, opacity=0.45,
        hovertemplate='<b>%{x}</b><br>Predicted: %{y:,.0f}<extra></extra>'
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='Total Sales: Actual vs Predicted by Store', font=dict(size=14, color='#e2e8f0')),
        barmode='group',
        xaxis_title='Store', yaxis_title='Units Sold',
        height=340
    )
    return fig


def plot_daily_trend_by_store(daily_df, selected_stores):
    fig = go.Figure()
    for store in selected_stores:
        df_s = daily_df[daily_df['store_id'] == store].sort_values('date')
        color = STORE_COLORS.get(store, '#475569')
        fig.add_trace(go.Scatter(
            x=df_s['date'], y=df_s['actual'],
            name=store, mode='lines',
            line=dict(color=color, width=1.8),
            hovertemplate=f'<b>{store}</b><br>%{{x|%b %d}}<br>%{{y:,.0f}}<extra></extra>'
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='Daily Actual Sales by Store (28-Day Window)', font=dict(size=14, color='#e2e8f0')),
        xaxis_title='Date', yaxis_title='Units Sold',
        hovermode='x unified', height=320
    )
    return fig


def plot_mape_by_store(store_metrics):
    df = store_metrics.sort_values('MAPE')
    colors = ['#34d399' if m < 10 else '#fbbf24' if m < 20 else '#f87171'
              for m in df['MAPE']]

    fig = go.Figure(go.Bar(
        x=df['MAPE'], y=df['store_id'],
        orientation='h',
        marker_color=colors,
        text=[f"{m:.1f}%" for m in df['MAPE']],
        textposition='outside',
        textfont=dict(color='#94a3b8', size=11),
        hovertemplate='<b>%{y}</b><br>MAPE: %{x:.1f}%<extra></extra>'
    ))
    fig.add_vline(x=10, line_dash='dash', line_color='#34d399',
                  annotation_text='10%', annotation_font_color='#34d399')
    fig.add_vline(x=20, line_dash='dash', line_color='#fbbf24',
                  annotation_text='20%', annotation_font_color='#fbbf24')
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='MAPE by Store', font=dict(size=14, color='#e2e8f0')),
        xaxis_title='MAPE (%)', height=320
    )
    return fig


def plot_state_donut(store_metrics):
    state_df = store_metrics.groupby('state')['total_actual'].sum().reset_index()
    colors   = [STATE_COLORS.get(s, '#475569') for s in state_df['state']]

    fig = go.Figure(go.Pie(
        labels=state_df['state'],
        values=state_df['total_actual'],
        hole=0.65,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textfont=dict(color='#e2e8f0', size=13),
        hovertemplate='<b>%{label}</b><br>%{value:,.0f} units<br>%{percent}<extra></extra>'
    ))
    fig.add_annotation(
        text='by State', x=0.5, y=0.5,
        font=dict(size=12, color='#475569'),
        showarrow=False
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='Sales Distribution by State', font=dict(size=14, color='#e2e8f0')),
        showlegend=False, height=300
    )
    return fig


def plot_category_heatmap(cat_df):
    pivot = cat_df.pivot(index='store_id', columns='cat_id', values='actual').fillna(0)
    stores = sorted(pivot.index)
    cats   = sorted(pivot.columns)

    fig = go.Figure(go.Heatmap(
        z=[[pivot.loc[s, c] for c in cats] for s in stores],
        x=cats, y=stores,
        colorscale=[[0, '#0f1117'], [0.3, '#1e3a5f'], [0.7, '#1d4ed8'], [1, '#60a5fa']],
        hovertemplate='Store: %{y}<br>Cat: %{x}<br>Sales: %{z:,.0f}<extra></extra>',
        text=[[f"{pivot.loc[s,c]:,.0f}" for c in cats] for s in stores],
        texttemplate='%{text}',
        textfont=dict(color='#e2e8f0', size=11)
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='Sales Heatmap — Store × Category', font=dict(size=14, color='#e2e8f0')),
        xaxis_title='Category', yaxis_title='Store',
        height=340
    )
    return fig


def plot_bias_by_store(store_metrics):
    df = store_metrics.sort_values('bias')
    colors = ['#34d399' if b >= 0 else '#f87171' for b in df['bias']]

    fig = go.Figure(go.Bar(
        x=df['store_id'], y=df['bias'],
        marker_color=colors,
        text=[f"{b:+.2f}" for b in df['bias']],
        textposition='outside',
        textfont=dict(color='#94a3b8', size=11),
        hovertemplate='<b>%{x}</b><br>Bias: %{y:+.2f}<extra></extra>'
    ))
    fig.add_hline(y=0, line_color='#475569', line_width=1)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='Forecast Bias by Store (+ = Overforecast)', font=dict(size=14, color='#e2e8f0')),
        xaxis_title='Store', yaxis_title='Avg Bias (units/day)',
        height=280
    )
    return fig


def plot_historical_by_store(hist_df, selected_stores):
    fig = go.Figure()
    for store in selected_stores:
        df_s = hist_df[hist_df['store_id'] == store].sort_values('date')
        df_s['ma30'] = df_s['sales'].rolling(30).mean()
        color = STORE_COLORS.get(store, '#475569')
        fig.add_trace(go.Scatter(
            x=df_s['date'], y=df_s['ma30'],
            name=store, mode='lines',
            line=dict(color=color, width=2),
            hovertemplate=f'<b>{store}</b><br>%{{x|%b %Y}}<br>MA30: %{{y:,.0f}}<extra></extra>'
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text='Historical Sales Trend by Store (30-day MA)', font=dict(size=14, color='#e2e8f0')),
        xaxis_title='Date', yaxis_title='Units Sold (MA30)',
        hovermode='x unified', height=320
    )
    return fig


def plot_radar(store_metrics):
    """Radar chart so sánh các stores theo nhiều metrics."""
    metrics_cols = ['total_actual', 'MAE', 'RMSE', 'MAPE', 'n_items']
    labels       = ['Volume', 'MAE ↓', 'RMSE ↓', 'MAPE ↓', 'Items']

    df = store_metrics.copy()
    # Normalize 0-1 (inverted cho metrics "thấp hơn tốt hơn")
    for col in ['MAE', 'RMSE', 'MAPE']:
        df[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-9)
    for col in ['total_actual', 'n_items']:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-9)

    fig = go.Figure()
    for _, row in df.iterrows():
        store = row['store_id']
        vals  = [row[c] for c in metrics_cols]
        vals += [vals[0]]   # close radar
        lbls  = labels + [labels[0]]
        color = STORE_COLORS.get(store, '#475569')
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=lbls,
            fill='toself', name=store,
            line=dict(color=color, width=1.5),
            fillcolor=color.replace(')', ',0.08)').replace('rgb', 'rgba'),
            opacity=0.85,
            hovertemplate=f'<b>{store}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>'
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#1e2535', color='#475569'),
            angularaxis=dict(gridcolor='#1e2535', color='#64748b')
        ),
        title=dict(text='Store Performance Radar', font=dict(size=14, color='#e2e8f0')),
        height=400
    )
    return fig