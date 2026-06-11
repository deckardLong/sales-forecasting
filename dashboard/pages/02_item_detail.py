import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error
import os

# Page config 
st.set_page_config(
    page_title="Sales Forecasting — Item Detail",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header   { visibility: hidden; }

    .stApp { background-color: #0f1117; color: #e2e8f0; }

    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #1e2535;
    }
    [data-testid="stSidebar"] * { color: #94a3b8 !important; }

    .page-title {
        font-size: 28px; font-weight: 700;
        color: #f1f5f9; letter-spacing: -0.02em;
    }
    .page-subtitle { font-size: 13px; color: #475569; margin-top: 4px; }

    .item-badge {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(139, 92, 246, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 20px; padding: 5px 14px;
        font-size: 12px; font-weight: 600;
        color: #a78bfa; margin-bottom: 12px;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-card {
        background: linear-gradient(135deg, #161b27 0%, #1a2035 100%);
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 18px 22px;
        position: relative; overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #8b5cf6, #06b6d4);
    }
    .metric-label {
        font-size: 10px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.1em;
        color: #475569; margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px; font-weight: 700;
        color: #f1f5f9; line-height: 1;
    }
    .metric-sub { font-size: 11px; margin-top: 5px; color: #64748b; }
    .metric-sub.good { color: #34d399; }
    .metric-sub.warn { color: #fbbf24; }
    .metric-sub.bad  { color: #f87171; }

    .section-header {
        display: flex; align-items: center; gap: 10px;
        margin: 28px 0 14px 0;
        padding-bottom: 10px;
        border-bottom: 1px solid #1e2535;
    }
    .section-dot { width: 6px; height: 6px; background: #8b5cf6; border-radius: 50%; }
    .section-title {
        font-size: 11px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.12em; color: #475569;
    }

    .insight-card {
        background: rgba(139, 92, 246, 0.05);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 10px; padding: 14px 18px;
        margin-bottom: 8px;
    }
    .insight-title { font-size: 11px; font-weight: 600; color: #8b5cf6; margin-bottom: 4px; }
    .insight-text  { font-size: 13px; color: #94a3b8; line-height: 1.5; }

    .day-table th { font-size: 11px !important; }
    .day-table td { font-size: 12px !important; font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# Plotly base layout
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#94a3b8', size=12),
    xaxis=dict(gridcolor='#1e2535', linecolor='#1e2535'),
    yaxis=dict(gridcolor='#1e2535', linecolor='#1e2535'),
    margin=dict(l=16, r=16, t=40, b=16),
    # legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1e2535', borderwidth=1),
    hoverlabel=dict(bgcolor='#161b27', bordercolor='#1e2535', font_color='#e2e8f0'),
)


# Data loading 
@st.cache_data
def load_base_data():
    df_forecast = pd.read_parquet('models/forecast/forecast_results.parquet')
    df_eval     = pd.read_csv('data/raw/sales_train_evaluation.csv')
    calendar    = pd.read_csv('data/raw/calendar.csv')
    df_features = pd.read_parquet('data/features/m5_features.parquet')

    df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])
    df_forecast['yhat'] = df_forecast['yhat'].clip(lower=0)
    calendar['date'] = pd.to_datetime(calendar['date'])
    return df_forecast, df_eval, calendar, df_features


@st.cache_data
def get_item_actual(df_eval, calendar, item_id, store_id):
    """Lấy 28 ngày actual của 1 item từ evaluation file."""
    id_val   = f"{item_id}_{store_id}_evaluation"
    day_cols = [c for c in df_eval.columns if c.startswith('d_')]
    last_28  = day_cols[-28:]

    row = df_eval[df_eval['id'] == id_val]
    if row.empty:
        return pd.DataFrame()

    vals  = row[last_28].values.flatten()
    dates = calendar[calendar['d'].isin(last_28)].sort_values('d')['date'].values

    return pd.DataFrame({'date': pd.to_datetime(dates), 'actual': vals})


@st.cache_data
def get_item_history(df_features, item_id, store_id):
    """Lấy toàn bộ lịch sử của 1 item."""
    return df_features[
        (df_features['item_id']  == item_id) &
        (df_features['store_id'] == store_id)
    ][['date', 'sales', 'sell_price', 'snap',
       'event_name_1', 'event_type_1']]\
     .sort_values('date').reset_index(drop=True)


# Charts 
def plot_item_forecast(df_history, df_actual, df_pred, item_id, store_id, calendar):
    df_hist_tail = df_history.tail(180)

    holiday_dates = calendar[
        (calendar['date'].isin(df_pred['ds'])) &
        (calendar['event_name_1'].notna()) &
        (calendar['event_name_1'] != 'No_Event')
    ][['date', 'event_name_1']].drop_duplicates()

    fig = go.Figure()

    # Vùng confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([df_pred['ds'], df_pred['ds'][::-1]]),
        y=pd.concat([df_pred['yhat_upper'], df_pred['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(244, 114, 182, 0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Interval',
        hoverinfo='skip'
    ))

    # Lịch sử
    fig.add_trace(go.Scatter(
        x=df_hist_tail['date'], y=df_hist_tail['sales'],
        name='Historical', mode='lines',
        line=dict(color='#334155', width=1.5),
        hovertemplate='<b>Historical</b><br>%{x|%b %d, %Y}<br>%{y:,.0f} units<extra></extra>'
    ))

    # Actual 28 ngày
    fig.add_trace(go.Scatter(
        x=df_actual['date'], y=df_actual['actual'],
        name='Actual', mode='lines+markers',
        line=dict(color='#60a5fa', width=2.5),
        marker=dict(size=6, color='#60a5fa', symbol='circle'),
        hovertemplate='<b>Actual</b><br>%{x|%b %d}<br>%{y:,.0f} units<extra></extra>'
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=df_pred['ds'], y=df_pred['yhat'],
        name='Forecast', mode='lines+markers',
        line=dict(color='#f472b6', width=2.5, dash='dot'),
        marker=dict(size=6, color='#f472b6', symbol='diamond'),
        hovertemplate='<b>Forecast</b><br>%{x|%b %d}<br>%{y:,.1f} units<extra></extra>'
    ))

    # Đường phân cách history / forecast
    split_date = df_actual['date'].min()
    fig.add_vline(
        x=split_date, line_dash='dash',
        line_color='#475569', line_width=1,
        annotation_text='Forecast Start',
        annotation_font_color='#475569',
        annotation_font_size=11
    )

    # Đánh dấu ngày lễ
    for _, row in holiday_dates.iterrows():
        fig.add_vline(
            x=row['date'], line_dash='dot',
            line_color='#fbbf24', line_width=1,
            annotation_text=row['event_name_1'],
            annotation_font_color='#fbbf24',
            annotation_font_size=10,
            annotation_textangle=-45
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f'{item_id}  ·  {store_id}  —  Sales History + 28-Day Forecast',
            font=dict(size=14, color='#e2e8f0')
        ),
        xaxis_title='Date',
        yaxis_title='Units Sold',
        hovermode='x unified',
        height=400,
        legend=dict(
            orientation='h', yanchor='bottom',
            y=1.02, xanchor='right', x=1,
            bgcolor='rgba(0,0,0,0)'
        )
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


# ── Sidebar ────────────────────────────────────────────────────────────────────
try:
    df_forecast, df_eval, calendar, df_features = load_base_data()

    with st.sidebar:
        st.markdown("""
        <div style='padding: 16px 0 24px 0;'>
            <div style='font-size:18px; font-weight:700; color:#f1f5f9;'>📈 M5 Forecasting</div>
            <div style='font-size:11px; color:#475569; margin-top:4px;'>Walmart Sales · Prophet Model</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Store selector
        stores = sorted(df_forecast['store_id'].unique())
        selected_store = st.selectbox("STORE", stores, index=0)

        # Category selector
        cats = sorted(df_eval['cat_id'].unique())
        selected_cat = st.selectbox("CATEGORY", cats, index=0)

        # Item selector — filtered by category
        cat_items = sorted(
            df_eval[df_eval['cat_id'] == selected_cat]['item_id'].unique()
        )
        selected_item = st.selectbox("ITEM", cat_items, index=0)

        st.markdown("---")

        # Quick info
        item_row = df_eval[
            (df_eval['item_id']  == selected_item) &
            (df_eval['store_id'] == selected_store)
        ]
        if not item_row.empty:
            dept = item_row['dept_id'].values[0]
            st.markdown(f"""
            <div style='font-size:11px; color:#475569; line-height:2;'>
                <div>🏷️ <span style='color:#64748b;'>Dept:</span> <span style='color:#94a3b8;'>{dept}</span></div>
                <div>🏪 <span style='color:#64748b;'>Store:</span> <span style='color:#94a3b8;'>{selected_store}</span></div>
                <div>📦 <span style='color:#64748b;'>Category:</span> <span style='color:#94a3b8;'>{selected_cat}</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style='font-size:11px; color:#1e3a5f; text-align:center; padding: 8px 0;'>
            Prophet · 28-day Horizon
        </div>
        """, unsafe_allow_html=True)

    # ── Load item data ─────────────────────────────────────────────────────────
    df_actual_item = get_item_actual(df_eval, calendar, selected_item, selected_store)
    df_history     = get_item_history(df_features, selected_item, selected_store)
    df_pred_item   = df_forecast[
        (df_forecast['item_id']  == selected_item) &
        (df_forecast['store_id'] == selected_store)
    ].copy()
    
    print(f'--{df_actual_item.columns.to_list()}--')
    # Merge actual vs predicted
    df_compare_item = df_actual_item.merge(
        df_pred_item[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        left_on='date', right_on='ds', how='inner'
    )

    # ── Metrics ────────────────────────────────────────────────────────────────
    if not df_compare_item.empty:
        mae  = mean_absolute_error(df_compare_item['actual'], df_compare_item['yhat'])
        rmse = np.sqrt(((df_compare_item['actual'] - df_compare_item['yhat']) ** 2).mean())
        mape = (abs(df_compare_item['actual'] - df_compare_item['yhat']) / (df_compare_item['actual'] + 1)).mean() * 100
        bias = (df_compare_item['yhat'] - df_compare_item['actual']).mean()
        total_actual    = int(df_compare_item['actual'].sum())
        total_predicted = int(df_compare_item['yhat'].sum())
        metrics = {'mae': mae, 'rmse': rmse, 'mape': mape, 'bias': bias}
    else:
        mae = rmse = mape = bias = total_actual = total_predicted = 0
        metrics = {'mae': 0, 'rmse': 0, 'mape': 0, 'bias': 0}

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='item-badge'>🔍 {selected_item} · {selected_store}</div>
    <div class='page-title'>Item Forecast Detail</div>
    <div class='page-subtitle'>
        Category: {selected_cat} · 
        History: {len(df_history):,} days · 
        Forecast: 28 days
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI Cards ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    mape_class = "good" if mape < 10 else ("warn" if mape < 20 else "bad")
    bias_class = "good" if abs(bias) < 1 else "bad"
    bias_label = f"{'Over' if bias > 0 else 'Under'}forecasting"

    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Actual Sales</div>
            <div class='metric-value'>{total_actual:,}</div>
            <div class='metric-sub'>28-day total</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Predicted Sales</div>
            <div class='metric-value'>{total_predicted:,}</div>
            <div class='metric-sub'>28-day total</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>MAPE</div>
            <div class='metric-value'>{mape:.1f}%</div>
            <div class='metric-sub {mape_class}'>{'Excellent' if mape < 10 else 'Good' if mape < 20 else 'Needs work'}</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>MAE / RMSE</div>
            <div class='metric-value'>{mae:.2f}</div>
            <div class='metric-sub'>RMSE: {rmse:.2f}</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Forecast Bias</div>
            <div class='metric-value'>{bias:+.2f}</div>
            <div class='metric-sub {bias_class}'>{bias_label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main forecast chart ─────────────────────────────────────────────────────
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Forecast vs Actual</div>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(
        plot_item_forecast(df_history, df_actual_item, df_pred_item, selected_item, selected_store, calendar),
        use_container_width=True
    )

    # ── Components + Price ──────────────────────────────────────────────────────
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Seasonality & Price Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    col_comp, col_price = st.columns([3, 2])
    with col_comp:
        st.plotly_chart(plot_components(df_history), use_container_width=True)
    with col_price:
        st.plotly_chart(plot_price_vs_sales(df_history), use_container_width=True)

    # ── Residuals + Insights ────────────────────────────────────────────────────
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Residual Analysis & Insights</div>
    </div>
    """, unsafe_allow_html=True)

    col_resid, col_insight = st.columns([3, 2])

    with col_resid:
        if not df_compare_item.empty:
            st.plotly_chart(plot_residuals(df_compare_item.copy()), use_container_width=True)

    with col_insight:
        insights = generate_insights(df_history, df_compare_item, metrics)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        for title, text in insights:
            st.markdown(f"""
            <div class='insight-card'>
                <div class='insight-title'>{title}</div>
                <div class='insight-text'>{text}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Day-by-day table ────────────────────────────────────────────────────────
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Day-by-Day Breakdown</div>
    </div>
    """, unsafe_allow_html=True)

    if not df_compare_item.empty:
        df_table = df_compare_item[['date', 'actual', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        df_table['error']    = df_table['actual'] - df_table['yhat']
        df_table['abs_error'] = df_table['error'].abs()
        df_table['mape_day'] = df_table['abs_error'] / (df_table['actual'] + 1) * 100
        df_table['date']     = df_table['date'].dt.strftime('%b %d, %Y')
        df_table = df_table.rename(columns={
            'date'      : 'Date',
            'actual'    : 'Actual',
            'yhat'      : 'Predicted',
            'yhat_lower': 'Lower Bound',
            'yhat_upper': 'Upper Bound',
            'error'     : 'Error',
            'abs_error' : 'Abs Error',
            'mape_day'  : 'Daily MAPE %'
        }).reset_index(drop=True)
        df_table.index += 1

        st.dataframe(
            df_table.style
                .background_gradient(subset=['Daily MAPE %'], cmap='RdYlGn_r')
                .format({
                    'Actual'     : '{:.0f}',
                    'Predicted'  : '{:.1f}',
                    'Lower Bound': '{:.1f}',
                    'Upper Bound': '{:.1f}',
                    'Error'      : '{:+.1f}',
                    'Abs Error'  : '{:.1f}',
                    'Daily MAPE %': '{:.1f}%'
                }),
            use_container_width=True,
            height=400
        )

except FileNotFoundError as e:
    st.markdown("""
    <div style='text-align:center; padding:80px 40px;'>
        <div style='font-size:48px; margin-bottom:16px;'>📂</div>
        <div style='font-size:20px; font-weight:600; color:#f1f5f9; margin-bottom:8px;'>No Data Found</div>
        <div style='font-size:14px; color:#475569;'>Run the training pipeline first.</div>
    </div>
    """, unsafe_allow_html=True)
    st.error(f"Missing file: {e}")