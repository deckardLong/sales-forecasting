import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error
from components.metrics import compute_actuals, compute_compare, compute_item_metrics
from components.charts import PLOTLY_LAYOUT, plot_actual_vs_predicted, plot_historical_trend, plot_store_comparison, plot_category_breakdown, plot_error_distribution
from components.data_loader import load_all

# Page config
st.set_page_config(
    page_title="Sales Forecasting — Overview",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme & CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
 
    /* Base */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
 
    /* Hide default streamlit header */
    #MainMenu, footer { visibility: hidden; }
 
    /* Main background */
    .stApp {
        background-color: #0f1117;
        color: #e2e8f0;
    }
 
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #1e2535;
    }
    [data-testid="stSidebar"] * { color: #94a3b8 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label { color: #64748b !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.08em; }
 
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #161b27 0%, #1a2035 100%);
        border: 1px solid #1e2d45;
        border-radius: 12px;
        padding: 20px 24px;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #475569;
        margin-bottom: 8px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 28px;
        font-weight: 700;
        color: #f1f5f9;
        line-height: 1;
    }
    .metric-delta {
        font-size: 12px;
        margin-top: 6px;
        color: #64748b;
    }
    .metric-delta.good { color: #34d399; }
    .metric-delta.bad  { color: #f87171; }
 
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 32px 0 16px 0;
        padding-bottom: 10px;
        border-bottom: 1px solid #1e2535;
    }
    .section-title {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #475569;
    }
    .section-dot {
        width: 6px; height: 6px;
        background: #3b82f6;
        border-radius: 50%;
    }
 
    /* Page title */
    .page-title {
        font-size: 32px;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .page-subtitle {
        font-size: 14px;
        color: #475569;
        margin-top: 6px;
    }
 
    /* Top badge */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 11px;
        font-weight: 500;
        color: #60a5fa;
        margin-bottom: 12px;
    }
 
    /* Data table */
    .dataframe { font-size: 12px !important; }
 
    /* Plotly chart background */
    .js-plotly-plot { border-radius: 12px; overflow: hidden; }
 
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #161b27;
        border-color: #1e2535;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)
 
# Side Bar
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 24px 0;'>
        <div style='font-size:18px; font-weight:700; color:#f1f5f9; letter-spacing:-0.02em;'>
            📈 M5 Forecasting
        </div>
        <div style='font-size:11px; color:#475569; margin-top:4px;'>
            Walmart Sales · Prophet Model
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("---")
    st.markdown("<div style='font-size:10px; color:#475569; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;'>Navigation</div>", unsafe_allow_html=True)
 
    pages = {
        "📊 Overview"         : "Current page",
        "🔍 Item Detail"      : "pages/02_item_detail.py",
        "🏪 Store Comparison" : "pages/03_store_comparison.py",
    }
    for page, _ in pages.items():
        is_active = page == "📊 Overview"
        bg = "rgba(59,130,246,0.1)" if is_active else "transparent"
        border = "rgba(59,130,246,0.4)" if is_active else "transparent"
        color = "#60a5fa" if is_active else "#64748b"
        st.markdown(f"""
        <div style='padding:8px 12px; border-radius:8px; background:{bg};
                    border:1px solid {border}; margin-bottom:4px; cursor:pointer;
                    font-size:13px; color:{color}; font-weight:{"500" if is_active else "400"};'>
            {page}
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("---")
    st.markdown("<div style='font-size:10px; color:#475569; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;'>Filters</div>", unsafe_allow_html=True)
 
    selected_stores = st.multiselect(
        "STORES",
        options=['CA_1','CA_2','CA_3','CA_4','TX_1','TX_2','TX_3','WI_1','WI_2','WI_3'],
        default=['CA_1','CA_2','CA_3','CA_4','TX_1','TX_2','TX_3','WI_1','WI_2','WI_3'],
        help="Filter by store"
    )
 
    selected_cats = st.multiselect(
        "CATEGORIES",
        options=['HOBBIES', 'FOODS', 'HOUSEHOLD'],
        default=['HOBBIES', 'FOODS', 'HOUSEHOLD'],
        help="Filter by category"
    )
 
    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:11px; color:#334155; text-align:center; padding: 8px 0;'>
        Prophet · 28-day Horizon<br>
        <span style='color:#1e3a5f;'>30,490 models trained</span>
    </div>
    """, unsafe_allow_html=True)
 
 # Main Content
try:
    df_forecast, df_eval, calendar, df_features = load_all()
    df_actual    = compute_actuals(df_eval, calendar)
    df_compare   = compute_compare(df_forecast, df_actual)
    item_metrics = compute_item_metrics(df_forecast, df_actual)
 
    # Apply filters
    if selected_stores:
        df_forecast  = df_forecast[df_forecast['store_id'].isin(selected_stores)]
        df_actual    = df_actual[df_actual['store_id'].isin(selected_stores)]
        item_metrics = item_metrics[item_metrics['store_id'].isin(selected_stores)]
 
    if selected_cats:
        cat_items = df_eval[df_eval['cat_id'].isin(selected_cats)]['item_id'].unique()
        df_forecast  = df_forecast[df_forecast['item_id'].isin(cat_items)]
        df_actual    = df_actual[df_actual['item_id'].isin(cat_items)]
        item_metrics = item_metrics[item_metrics['item_id'].isin(cat_items)]
 
    df_compare = compute_compare(df_forecast, df_actual)

    # Header
    st.markdown("""
    <div class='badge'>● LIVE DASHBOARD</div>
    <div class='page-title'>Sales Forecast Overview</div>
    <div class='page-subtitle'>M5 Competition · Walmart Retail · 10 Stores · 3,049 Products · Prophet Model</div>
    """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)

    # KPI Metrics
    total_actual    = int(df_compare['actual'].sum())
    total_predicted = int(df_compare['yhat'].sum())
    overall_mae     = mean_absolute_error(df_compare['actual'], df_compare['yhat'])
    overall_rmse    = np.sqrt(((df_compare['actual'] - df_compare['yhat']) ** 2).mean())
    overall_mape    = (abs(df_compare['actual'] - df_compare['yhat']) / (df_compare['actual'] + 1)).mean() * 100
    pct_diff        = (total_predicted - total_actual) / total_actual * 100
    n_items         = len(item_metrics)
    good_items      = (item_metrics['MAPE'] < 20).sum()
    pct_good        = good_items / n_items * 100
 
    col1, col2, col3, col4, col5 = st.columns(5)
 
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Actual Sales</div>
            <div class='metric-value'>{total_actual:,.0f}</div>
            <div class='metric-delta'>28-day window</div>
        </div>
        """, unsafe_allow_html=True)
 
    with col2:
        delta_class = "good" if abs(pct_diff) < 5 else "bad"
        arrow = "↑" if pct_diff > 0 else "↓"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Total Predicted</div>
            <div class='metric-value'>{total_predicted:,.0f}</div>
            <div class='metric-delta {delta_class}'>{arrow} {abs(pct_diff):.1f}% vs actual</div>
        </div>
        """, unsafe_allow_html=True)
 
    with col3:
        mape_class = "good" if overall_mape < 15 else "bad"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Overall MAPE</div>
            <div class='metric-value'>{overall_mape:.1f}%</div>
            <div class='metric-delta {mape_class}'>{'✓ Good' if overall_mape < 15 else '⚠ Needs improvement'}</div>
        </div>
        """, unsafe_allow_html=True)
 
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>MAE / RMSE</div>
            <div class='metric-value'>{overall_mae:.1f}</div>
            <div class='metric-delta'>RMSE: {overall_rmse:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
 
    with col5:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Items MAPE &lt; 20%</div>
            <div class='metric-value'>{pct_good:.0f}%</div>
            <div class='metric-delta good'>{good_items:,} / {n_items:,} items</div>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)

    # Actual vs Predicted
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Forecast Performance — 28-Day Window</div>
    </div>
    """, unsafe_allow_html=True)
 
    st.plotly_chart(plot_actual_vs_predicted(df_compare), width='stretch')

    # Historial Trend
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Historical Sales Trend</div>
    </div>
    """, unsafe_allow_html=True)
 
    st.plotly_chart(plot_historical_trend(df_features), width='stretch')

    # Store + Category
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Sales Breakdown</div>
    </div>
    """, unsafe_allow_html=True)
 
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.plotly_chart(plot_store_comparison(df_actual), width='stretch')
    with col_right:
        st.plotly_chart(plot_category_breakdown(df_actual, df_eval), width='stretch')

    # Error Distribution
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Model Error Analysis</div>
    </div>
    """, unsafe_allow_html=True)
 
    col_hist, col_table = st.columns([2, 3])
 
    with col_hist:
        st.plotly_chart(plot_error_distribution(item_metrics), width='stretch')
 
    with col_table:
        st.markdown("<div style='font-size:12px; color:#475569; margin-bottom:8px;'>TOP 10 WORST PERFORMING ITEMS</div>", unsafe_allow_html=True)
        worst = item_metrics.nlargest(10, 'MAPE')[['item_id', 'store_id', 'MAE', 'RMSE', 'MAPE']]\
                            .round(2).reset_index(drop=True)
        worst.index += 1
 
        st.dataframe(
            worst.style
                .background_gradient(subset=['MAPE'], cmap='RdYlGn_r')
                .format({'MAE': '{:.2f}', 'RMSE': '{:.2f}', 'MAPE': '{:.1f}%'}),
            width='stretch',
            height=280
        )

    # Top Performing Items
    st.markdown("""
    <div class='section-header'>
        <div class='section-dot'></div>
        <div class='section-title'>Top Performing Items</div>
    </div>
    """, unsafe_allow_html=True)
 
    col_best, col_highest = st.columns(2)
 
    with col_best:
        st.markdown("<div style='font-size:12px; color:#475569; margin-bottom:8px;'>✓ MOST ACCURATE (LOWEST MAPE)</div>", unsafe_allow_html=True)
        best = item_metrics.nsmallest(10, 'MAPE')[['item_id', 'store_id', 'MAPE', 'total_actual']]\
                           .round(2).reset_index(drop=True)
        best.index += 1
        st.dataframe(
            best.style
                .background_gradient(subset=['MAPE'], cmap='RdYlGn')
                .format({'MAPE': '{:.1f}%', 'total_actual': '{:,.0f}'}),
            width='stretch',
            height=280
        )
 
    with col_highest:
        st.markdown("<div style='font-size:12px; color:#475569; margin-bottom:8px;'>📦 HIGHEST VOLUME ITEMS</div>", unsafe_allow_html=True)
        highest = item_metrics.nlargest(10, 'total_actual')[['item_id', 'store_id', 'total_actual', 'total_predicted', 'MAPE']]\
                              .round(2).reset_index(drop=True)
        highest.index += 1
        st.dataframe(
            highest.style
                   .format({'total_actual': '{:,.0f}', 'total_predicted': '{:,.0f}', 'MAPE': '{:.1f}%'}),
            width='stretch',
            height=280
        )
 
except FileNotFoundError as e:
    st.markdown("""
    <div style='text-align:center; padding:80px 40px;'>
        <div style='font-size:48px; margin-bottom:16px;'>📂</div>
        <div style='font-size:20px; font-weight:600; color:#f1f5f9; margin-bottom:8px;'>No Data Found</div>
        <div style='font-size:14px; color:#475569;'>
            Run the training pipeline first to generate forecast results.<br>
            Expected: <code>models/forecast/forecast_results.parquet</code>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.error(f"Missing file: {e}")