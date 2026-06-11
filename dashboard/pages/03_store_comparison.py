import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecasting — Store Comparison",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
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

    .badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(6, 182, 212, 0.1);
        border: 1px solid rgba(6, 182, 212, 0.2);
        border-radius: 20px; padding: 4px 12px;
        font-size: 11px; font-weight: 500;
        color: #22d3ee; margin-bottom: 12px;
    }

    .metric-card {
        background: linear-gradient(135deg, #161b27 0%, #1a2035 100%);
        border: 1px solid #1e2d45; border-radius: 12px;
        padding: 18px 22px; position: relative; overflow: hidden;
    }
    .metric-card::before {
        content: ''; position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, #06b6d4, #3b82f6);
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
    .section-dot { width: 6px; height: 6px; background: #06b6d4; border-radius: 50%; }
    .section-title {
        font-size: 11px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.12em; color: #475569;
    }

    .store-rank-card {
        background: #161b27;
        border: 1px solid #1e2535;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 8px;
        display: flex; align-items: center;
        gap: 16px;
        transition: border-color 0.2s;
    }
    .store-rank-card:hover { border-color: #06b6d4; }
    .rank-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 20px; font-weight: 700;
        color: #1e3a5f; min-width: 32px;
    }
    .rank-num.top { color: #06b6d4; }
    .store-name  { font-size: 14px; font-weight: 600; color: #e2e8f0; }
    .store-meta  { font-size: 11px; color: #475569; margin-top: 2px; }
    .store-sales {
        font-family: 'JetBrains Mono', monospace;
        font-size: 15px; font-weight: 600;
        color: #22d3ee; margin-left: auto;
    }
</style>
""", unsafe_allow_html=True)

# ── Colors per store ───────────────────────────────────────────────────────────
STORE_COLORS = {
    'CA_1': '#3b82f6', 'CA_2': '#60a5fa', 'CA_3': '#93c5fd', 'CA_4': '#bfdbfe',
    'TX_1': '#8b5cf6', 'TX_2': '#a78bfa', 'TX_3': '#c4b5fd',
    'WI_1': '#06b6d4', 'WI_2': '#22d3ee', 'WI_3': '#67e8f9',
}

STATE_COLORS = {'CA': '#3b82f6', 'TX': '#8b5cf6', 'WI': '#06b6d4'}

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


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_base_data():
    df_forecast = pd.read_parquet('models/forecast/forecast_results.parquet')
    df_eval     = pd.read_csv("data/raw/sales_train_evaluation.csv")
    calendar    = pd.read_csv("data/raw/calendar.csv")
    df_features = pd.read_parquet("data/features/m5_features.parquet")

    df_forecast['ds'] = pd.to_datetime(df_forecast['ds'])
    df_forecast['yhat'] = df_forecast['yhat'].clip(lower=0)
    calendar['date'] = pd.to_datetime(calendar['date'])
    return df_forecast, df_eval, calendar, df_features


@st.cache_data
def build_store_metrics(df_forecast, df_eval, calendar):
    """Tính metrics cho từng store."""
    id_cols  = ['item_id', 'store_id']
    day_cols = [c for c in df_eval.columns if c.startswith('d_')]
    last_28  = day_cols[-28:]

    df_actual = df_eval.melt(
        id_vars=id_cols, value_vars=last_28,
        var_name='d', value_name='actual'
    ).merge(calendar[['d', 'date']], on='d', how='left')
    df_actual['date'] = pd.to_datetime(df_actual['date'])

    df_pred = df_forecast.rename(columns={'ds': 'date'})
    df_merge = df_actual.merge(df_pred, on=['item_id', 'store_id', 'date'], how='inner')

    metrics = df_merge.groupby('store_id').apply(lambda g: pd.Series({
        'total_actual'    : g['actual'].sum(),
        'total_predicted' : g['yhat'].sum(),
        'MAE'  : mean_absolute_error(g['actual'], g['yhat']),
        'RMSE' : np.sqrt(((g['actual'] - g['yhat']) ** 2).mean()),
        'MAPE' : (abs(g['actual'] - g['yhat']) / (g['actual'] + 1)).mean() * 100,
        'bias' : (g['yhat'] - g['actual']).mean(),
        'n_items': g['item_id'].nunique(),
    })).reset_index()

    metrics['state']      = metrics['store_id'].str[:2]
    metrics['pct_diff']   = (metrics['total_predicted'] - metrics['total_actual']) / metrics['total_actual'] * 100
    return metrics, df_merge, df_actual


@st.cache_data
def build_daily_by_store(df_merge):
    return df_merge.groupby(['store_id', 'date']).agg(
        actual=('actual', 'sum'),
        predicted=('yhat', 'sum')
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


# ── Charts ─────────────────────────────────────────────────────────────────────
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


# ── Sidebar ────────────────────────────────────────────────────────────────────
try:
    df_forecast, df_eval, calendar, df_features = load_base_data()
    store_metrics, df_merge, df_actual = build_store_metrics(df_forecast, df_eval, calendar)
    daily_df  = build_daily_by_store(df_merge)
    cat_df    = build_cat_by_store(df_merge, df_eval)
    hist_df   = build_history_by_store(df_features)

    all_stores = sorted(store_metrics['store_id'].unique())

    with st.sidebar:
        st.markdown("""
        <div style='padding: 16px 0 24px 0;'>
            <div style='font-size:18px; font-weight:700; color:#f1f5f9;'>📈 M5 Forecasting</div>
            <div style='font-size:11px; color:#475569; margin-top:4px;'>Walmart Sales · Prophet Model</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div style='font-size:10px; color:#475569; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;'>Store Selection</div>", unsafe_allow_html=True)

        # State quick filter
        state_filter = st.radio(
            "QUICK FILTER BY STATE",
            options=["All", "CA", "TX", "WI"],
            horizontal=True
        )

        default_stores = all_stores if state_filter == "All" \
                         else [s for s in all_stores if s.startswith(state_filter)]

        selected_stores = st.multiselect(
            "SELECT STORES",
            options=all_stores,
            default=default_stores
        )

        st.markdown("---")

        # Sort metric
        sort_by = st.selectbox(
            "RANK STORES BY",
            options=['total_actual', 'MAPE', 'MAE', 'RMSE'],
            format_func=lambda x: {
                'total_actual': '📦 Total Sales',
                'MAPE': '🎯 MAPE (accuracy)',
                'MAE' : '📏 MAE',
                'RMSE': '📐 RMSE'
            }[x]
        )

        st.markdown("---")
        st.markdown(f"""
        <div style='font-size:11px; color:#334155; text-align:center; padding:8px 0;'>
            {len(selected_stores)} stores selected<br>
            <span style='color:#1e3a5f;'>28-day forecast window</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Filter data ────────────────────────────────────────────────────────────
    if not selected_stores:
        st.warning("Please select at least one store.")
        st.stop()

    sm_filtered  = store_metrics[store_metrics['store_id'].isin(selected_stores)]
    daily_filt   = daily_df[daily_df['store_id'].isin(selected_stores)]
    cat_filt     = cat_df[cat_df['store_id'].isin(selected_stores)]
    hist_filt    = hist_df[hist_df['store_id'].isin(selected_stores)]

    # ── Header ─────────────────────────────────────────────────────────────────
    states_shown = ', '.join(sm_filtered['state'].unique())
    st.markdown(f"""
    <div class='badge'>● {len(selected_stores)} STORES SELECTED</div>
    <div class='page-title'>Store Comparison</div>
    <div class='page-subtitle'>
        States: {states_shown} · 28-Day Forecast Window · Prophet Model
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI Summary ─────────────────────────────────────────────────────────────
    total_actual    = int(sm_filtered['total_actual'].sum())
    total_predicted = int(sm_filtered['total_predicted'].sum())
    avg_mape        = sm_filtered['MAPE'].mean()
    best_store      = sm_filtered.loc[sm_filtered['MAPE'].idxmin(), 'store_id']
    worst_store     = sm_filtered.loc[sm_filtered['MAPE'].idxmax(), 'store_id']
    top_store       = sm_filtered.loc[sm_filtered['total_actual'].idxmax(), 'store_id']
    pct_diff        = (total_predicted - total_actual) / total_actual * 100

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Actual</div>
            <div class='metric-value'>{total_actual:,.0f}</div>
            <div class='metric-sub'>{len(selected_stores)} stores · 28 days</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        d_class = "good" if abs(pct_diff) < 5 else "bad"
        arrow   = "↑" if pct_diff > 0 else "↓"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Predicted</div>
            <div class='metric-value'>{total_predicted:,.0f}</div>
            <div class='metric-sub {d_class}'>{arrow} {abs(pct_diff):.1f}% vs actual</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        m_class = "good" if avg_mape < 10 else ("warn" if avg_mape < 20 else "bad")
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Avg MAPE</div>
            <div class='metric-value'>{avg_mape:.1f}%</div>
            <div class='metric-sub {m_class}'>Across {len(selected_stores)} stores</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Best Accuracy</div>
            <div class='metric-value'>{best_store}</div>
            <div class='metric-sub good'>MAPE: {sm_filtered[sm_filtered['store_id']==best_store]['MAPE'].values[0]:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Top Volume Store</div>
            <div class='metric-value'>{top_store}</div>
            <div class='metric-sub'>{sm_filtered[sm_filtered['store_id']==top_store]['total_actual'].values[0]:,.0f} units</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Store Rankings ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Store Rankings</div>
    </div>
    """, unsafe_allow_html=True)

    col_rank, col_bar = st.columns([2, 3])

    with col_rank:
        ascending = sort_by != 'total_actual'
        ranked = sm_filtered.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

        for i, row in ranked.iterrows():
            rank_class = "top" if i < 3 else ""
            medal      = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
            color = STORE_COLORS.get(row['store_id'], '#475569')
            st.markdown(f"""
            <div class='store-rank-card'>
                <div class='rank-num {rank_class}'>{medal}</div>
                <div>
                    <div class='store-name' style='color:{color};'>{row['store_id']}</div>
                    <div class='store-meta'>MAPE {row['MAPE']:.1f}% · {row['n_items']:.0f} items</div>
                </div>
                <div class='store-sales'>{row['total_actual']:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_bar:
        st.plotly_chart(plot_store_actual_vs_pred(sm_filtered), use_container_width=True)

    # ── Trend over 28 days ──────────────────────────────────────────────────────
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Daily Sales Trends</div>
    </div>
    """, unsafe_allow_html=True)

    col_daily, col_state = st.columns([3, 2])
    with col_daily:
        st.plotly_chart(plot_daily_trend_by_store(daily_filt, selected_stores), use_container_width=True)
    with col_state:
        st.plotly_chart(plot_state_donut(sm_filtered), use_container_width=True)

    # ── Accuracy Analysis ───────────────────────────────────────────────────────
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Accuracy & Bias Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    col_mape, col_bias = st.columns(2)
    with col_mape:
        st.plotly_chart(plot_mape_by_store(sm_filtered), use_container_width=True)
    with col_bias:
        st.plotly_chart(plot_bias_by_store(sm_filtered), use_container_width=True)

    # ── Heatmap + Radar ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Category Breakdown & Performance Radar</div>
    </div>
    """, unsafe_allow_html=True)

    col_heat, col_radar = st.columns([3, 2])
    with col_heat:
        st.plotly_chart(plot_category_heatmap(cat_filt), use_container_width=True)
    with col_radar:
        st.plotly_chart(plot_radar(sm_filtered), use_container_width=True)

    # ── Historical Trend ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Long-term Historical Trend</div>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(plot_historical_by_store(hist_filt, selected_stores), use_container_width=True)

    # ── Summary Table ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Full Store Metrics Table</div>
    </div>
    """, unsafe_allow_html=True)

    display_cols = ['store_id', 'state', 'n_items', 'total_actual',
                    'total_predicted', 'pct_diff', 'MAE', 'RMSE', 'MAPE', 'bias']
    df_display = sm_filtered[display_cols].sort_values(sort_by, ascending=(sort_by != 'total_actual'))\
                                          .reset_index(drop=True)
    df_display.index += 1
    df_display.columns = ['Store', 'State', 'Items', 'Actual',
                           'Predicted', 'Δ%', 'MAE', 'RMSE', 'MAPE%', 'Bias']

    st.dataframe(
        df_display.style
            .background_gradient(subset=['MAPE%'], cmap='RdYlGn_r')
            .background_gradient(subset=['Actual'], cmap='Blues')
            .format({
                'Items'    : '{:.0f}',
                'Actual'   : '{:,.0f}',
                'Predicted': '{:,.0f}',
                'Δ%'       : '{:+.1f}%',
                'MAE'      : '{:.2f}',
                'RMSE'     : '{:.2f}',
                'MAPE%'    : '{:.1f}%',
                'Bias'     : '{:+.2f}'
            }),
        use_container_width=True,
        height=420
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