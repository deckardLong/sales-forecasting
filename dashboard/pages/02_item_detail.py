import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error
from components.metrics import get_item_actual, get_item_history
from components.charts import PLOTLY_LAYOUT, plot_item_forecast, plot_components, plot_price_vs_sales, plot_residuals, generate_insights
from components.data_loader import load_all

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

# ── Sidebar ────────────────────────────────────────────────────────────────────
try:
    df_forecast, df_eval, calendar, df_features = load_all()

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
        width='stretch'
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
        st.plotly_chart(plot_components(df_history), width='stretch')
    with col_price:
        st.plotly_chart(plot_price_vs_sales(df_history), width='stretch')

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
            st.plotly_chart(plot_residuals(df_compare_item.copy()), width='stretch')

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
            width='stretch',
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