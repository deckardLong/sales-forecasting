"""
dashboard/app.py
─────────────────
Entry point — Landing page của Sales Forecasting Dashboard.
Chạy: streamlit run dashboard/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="M5 Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

    /* Hero */
    .hero {
        text-align: center;
        padding: 80px 40px 60px 40px;
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 8px;
        background: rgba(59,130,246,0.1);
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 20px; padding: 6px 16px;
        font-size: 12px; font-weight: 500; color: #60a5fa;
        margin-bottom: 24px; letter-spacing: 0.05em;
    }
    .hero-title {
        font-size: 52px; font-weight: 700;
        color: #f1f5f9; letter-spacing: -0.03em;
        line-height: 1.1; margin-bottom: 16px;
    }
    .hero-title span {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 18px; color: #475569;
        max-width: 600px; margin: 0 auto 48px auto;
        line-height: 1.6;
    }

    /* Stats row */
    .stats-row {
        display: flex; justify-content: center; gap: 48px;
        margin-bottom: 64px;
    }
    .stat-item { text-align: center; }
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 32px; font-weight: 700; color: #f1f5f9;
    }
    .stat-label { font-size: 12px; color: #475569; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.08em; }

    /* Page cards */
    .cards-grid {
        display: grid; grid-template-columns: repeat(3, 1fr);
        gap: 20px; max-width: 960px; margin: 0 auto 64px auto;
    }
    .page-card {
        background: #161b27;
        border: 1px solid #1e2535;
        border-radius: 16px; padding: 28px;
        cursor: pointer; transition: all 0.2s;
        text-align: left; position: relative; overflow: hidden;
    }
    .page-card::before {
        content: ''; position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
    }
    .page-card.blue::before  { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
    .page-card.purple::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
    .page-card.cyan::before  { background: linear-gradient(90deg, #06b6d4, #22d3ee); }
    .page-card:hover { border-color: #334155; transform: translateY(-2px); }

    .card-icon  { font-size: 32px; margin-bottom: 16px; }
    .card-title { font-size: 18px; font-weight: 600; color: #f1f5f9; margin-bottom: 8px; }
    .card-desc  { font-size: 13px; color: #475569; line-height: 1.6; margin-bottom: 16px; }
    .card-features { list-style: none; padding: 0; margin: 0; }
    .card-features li {
        font-size: 12px; color: #334155;
        padding: 3px 0;
        display: flex; align-items: center; gap: 8px;
    }
    .card-features li::before { content: '→'; color: #1e3a5f; }

    /* Tech stack */
    .tech-section {
        text-align: center; max-width: 800px;
        margin: 0 auto; padding-bottom: 80px;
    }
    .tech-title {
        font-size: 12px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.12em;
        color: #334155; margin-bottom: 20px;
    }
    .tech-badges { display: flex; justify-content: center; flex-wrap: wrap; gap: 10px; }
    .tech-badge {
        background: #161b27; border: 1px solid #1e2535;
        border-radius: 8px; padding: 8px 16px;
        font-size: 12px; color: #64748b;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Pipeline */
    .pipeline {
        display: flex; justify-content: center;
        align-items: center; gap: 0;
        max-width: 900px; margin: 0 auto 64px auto;
        flex-wrap: wrap;
    }
    .pipeline-step {
        background: #161b27; border: 1px solid #1e2535;
        border-radius: 10px; padding: 14px 20px;
        text-align: center; min-width: 120px;
    }
    .pipeline-step.done  { border-color: #1e3a5f; }
    .pipeline-step.active { border-color: #3b82f6; background: rgba(59,130,246,0.05); }
    .step-icon  { font-size: 20px; margin-bottom: 6px; }
    .step-label { font-size: 11px; color: #475569; }
    .step-status { font-size: 10px; margin-top: 4px; }
    .step-status.done   { color: #34d399; }
    .step-status.active { color: #60a5fa; }
    .step-status.todo   { color: #334155; }
    .pipeline-arrow {
        font-size: 18px; color: #1e2535;
        padding: 0 6px;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 24px 0;'>
        <div style='font-size:18px; font-weight:700; color:#f1f5f9;'>📈 M5 Forecasting</div>
        <div style='font-size:11px; color:#475569; margin-top:4px;'>Walmart Sales · Prophet Model</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:10px; color:#475569; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:12px;'>Navigation</div>", unsafe_allow_html=True)

    pages = [
        ("🏠 Home",             True),
        ("📊 Overview",         False),
        ("🔍 Item Detail",      False),
        ("🏪 Store Comparison", False),
    ]
    for label, is_active in pages:
        bg     = "rgba(59,130,246,0.1)" if is_active else "transparent"
        border = "rgba(59,130,246,0.3)" if is_active else "transparent"
        color  = "#60a5fa" if is_active else "#64748b"
        weight = "600" if is_active else "400"
        st.markdown(f"""
        <div style='padding:9px 12px; border-radius:8px;
                    background:{bg}; border:1px solid {border};
                    margin-bottom:4px; font-size:13px;
                    color:{color}; font-weight:{weight};'>
            {label}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#334155; line-height:1.8;'>
        <div>📅 Dataset: 2011–2016</div>
        <div>🏪 10 Walmart Stores</div>
        <div>📦 3,049 Products</div>
        <div>🤖 30,490 Prophet Models</div>
        <div>📈 28-day Horizon</div>
    </div>
    """, unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-badge'>● M5 Forecasting Competition · Walmart Retail</div>
    <div class='hero-title'>
        Sales Forecasting<br>
        <span>Dashboard</span>
    </div>
    <div class='hero-subtitle'>
        End-to-end sales forecasting pipeline for 30,490 Walmart products
        across 10 stores using Facebook Prophet with custom regressors.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Stats ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='stats-row'>
    <div class='stat-item'>
        <div class='stat-value'>30,490</div>
        <div class='stat-label'>Prophet Models</div>
    </div>
    <div class='stat-item'>
        <div class='stat-value'>58M</div>
        <div class='stat-label'>Data Points</div>
    </div>
    <div class='stat-item'>
        <div class='stat-value'>1,913</div>
        <div class='stat-label'>Days History</div>
    </div>
    <div class='stat-item'>
        <div class='stat-value'>28</div>
        <div class='stat-label'>Day Forecast</div>
    </div>
    <div class='stat-item'>
        <div class='stat-value'>3</div>
        <div class='stat-label'>US States</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Page cards ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='cards-grid'>

    <div class='page-card blue'>
        <div class='card-icon'>📊</div>
        <div class='card-title'>Overview</div>
        <div class='card-desc'>
            High-level metrics and performance summary across all stores and products.
        </div>
        <ul class='card-features'>
            <li>Actual vs Predicted (28 days)</li>
            <li>KPI metrics: MAE, RMSE, MAPE</li>
            <li>Sales by store and category</li>
            <li>MAPE distribution histogram</li>
            <li>Top & worst performing items</li>
        </ul>
    </div>

    <div class='page-card purple'>
        <div class='card-icon'>🔍</div>
        <div class='card-title'>Item Detail</div>
        <div class='card-desc'>
            Deep-dive into a single product's forecast, seasonality, and model behavior.
        </div>
        <ul class='card-features'>
            <li>180-day history + 28-day forecast</li>
            <li>Confidence intervals</li>
            <li>Seasonality components</li>
            <li>Price vs Sales correlation</li>
            <li>Residual analysis + auto insights</li>
        </ul>
    </div>

    <div class='page-card cyan'>
        <div class='card-icon'>🏪</div>
        <div class='card-title'>Store Comparison</div>
        <div class='card-desc'>
            Compare performance across stores and states with rich visual analytics.
        </div>
        <ul class='card-features'>
            <li>Store rankings with medals</li>
            <li>Category heatmap (Store × Cat)</li>
            <li>Performance radar chart</li>
            <li>Bias analysis by store</li>
            <li>Long-term trend comparison</li>
        </ul>
    </div>

</div>
""", unsafe_allow_html=True)

# ── Pipeline ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; font-size:12px; font-weight:600;
            text-transform:uppercase; letter-spacing:0.12em;
            color:#334155; margin-bottom:20px;'>
    Project Pipeline
</div>
<div class='pipeline'>
    <div class='pipeline-step done'>
        <div class='step-icon'>📥</div>
        <div class='step-label'>Ingestion</div>
        <div class='step-status done'>✓ Done</div>
    </div>
    <div class='pipeline-arrow'>→</div>
    <div class='pipeline-step done'>
        <div class='step-icon'>🧹</div>
        <div class='step-label'>Cleaning</div>
        <div class='step-status done'>✓ Done</div>
    </div>
    <div class='pipeline-arrow'>→</div>
    <div class='pipeline-step done'>
        <div class='step-icon'>🔗</div>
        <div class='step-label'>Merge</div>
        <div class='step-status done'>✓ Done</div>
    </div>
    <div class='pipeline-arrow'>→</div>
    <div class='pipeline-step done'>
        <div class='step-icon'>⚙️</div>
        <div class='step-label'>Features</div>
        <div class='step-status done'>✓ Done</div>
    </div>
    <div class='pipeline-arrow'>→</div>
    <div class='pipeline-step done'>
        <div class='step-icon'>🤖</div>
        <div class='step-label'>Training</div>
        <div class='step-status done'>✓ Done</div>
    </div>
    <div class='pipeline-arrow'>→</div>
    <div class='pipeline-step done'>
        <div class='step-icon'>📊</div>
        <div class='step-label'>Evaluation</div>
        <div class='step-status done'>✓ Done</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tech stack ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='tech-section'>
    <div class='tech-title'>Tech Stack</div>
    <div class='tech-badges'>
        <span class='tech-badge'>Python 3.11</span>
        <span class='tech-badge'>Prophet</span>
        <span class='tech-badge'>Pandas</span>
        <span class='tech-badge'>NumPy</span>
        <span class='tech-badge'>Scikit-learn</span>
        <span class='tech-badge'>Plotly</span>
        <span class='tech-badge'>Streamlit</span>
        <span class='tech-badge'>Joblib</span>
        <span class='tech-badge'>PyArrow</span>
        <span class='tech-badge'>M5 Competition</span>
    </div>
</div>
""", unsafe_allow_html=True)