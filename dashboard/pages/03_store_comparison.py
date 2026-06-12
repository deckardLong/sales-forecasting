import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error
from components.metrics import build_store_metrics, build_daily_by_store, build_cat_by_store, build_history_by_store
from components.charts import PLOTLY_LAYOUT, STORE_COLORS, STATE_COLORS, plot_store_actual_vs_pred, plot_daily_trend_by_store, \
                            plot_mape_by_store, plot_state_donut, plot_category_heatmap, plot_bias_by_store, plot_historical_by_store, plot_radar
from components.data_loader import load_all

# Page config
st.set_page_config(
    page_title="Sales Forecasting — Store Comparison",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer { visibility: hidden; }

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

# Sidebar 
try:
    df_forecast, df_eval, calendar, df_features = load_all()
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

    # Filter data 
    if not selected_stores:
        st.warning("Please select at least one store.")
        st.stop()

    sm_filtered  = store_metrics[store_metrics['store_id'].isin(selected_stores)]
    daily_filt   = daily_df[daily_df['store_id'].isin(selected_stores)]
    cat_filt     = cat_df[cat_df['store_id'].isin(selected_stores)]
    hist_filt    = hist_df[hist_df['store_id'].isin(selected_stores)]

    # Header 
    states_shown = ', '.join(sm_filtered['state'].unique())
    st.markdown(f"""
    <div class='badge'>● {len(selected_stores)} STORES SELECTED</div>
    <div class='page-title'>Store Comparison</div>
    <div class='page-subtitle'>
        States: {states_shown} · 28-Day Forecast Window · Prophet Model
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI Summary 
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

    # Store Rankings 
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
        st.plotly_chart(plot_store_actual_vs_pred(sm_filtered), width='stretch')

    # Trend over 28 days 
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Daily Sales Trends</div>
    </div>
    """, unsafe_allow_html=True)

    col_daily, col_state = st.columns([3, 2])
    with col_daily:
        st.plotly_chart(plot_daily_trend_by_store(daily_filt, selected_stores), width='stretch')
    with col_state:
        st.plotly_chart(plot_state_donut(sm_filtered), width='stretch')

    # Accuracy Analysis 
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Accuracy & Bias Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    col_mape, col_bias = st.columns(2)
    with col_mape:
        st.plotly_chart(plot_mape_by_store(sm_filtered), width='stretch')
    with col_bias:
        st.plotly_chart(plot_bias_by_store(sm_filtered), width='stretch')

    # Heatmap + Radar 
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Category Breakdown & Performance Radar</div>
    </div>
    """, unsafe_allow_html=True)

    col_heat, col_radar = st.columns([3, 2])
    with col_heat:
        st.plotly_chart(plot_category_heatmap(cat_filt), width='stretch')
    with col_radar:
        st.plotly_chart(plot_radar(sm_filtered), width='stretch')

    #  Historical Trend 
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Long-term Historical Trend</div>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(plot_historical_by_store(hist_filt, selected_stores), width='stretch')

    # Summary Table 
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
        width='stretch',
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