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
    .hero-badge {
        display: inline-flex; align-items: center; gap: 6px;
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
        font-size: 16px; color: #475569;
        line-height: 1.6; margin-bottom: 40px;
    }

    /* Stats */
    .stat-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 30px; font-weight: 700; color: #f1f5f9;
        text-align: center;
    }
    .stat-label {
        font-size: 11px; color: #475569; margin-top: 4px;
        text-transform: uppercase; letter-spacing: 0.08em;
        text-align: center;
    }

    /* Page cards */
    .page-card {
        background: #161b27;
        border: 1px solid #1e2535;
        border-radius: 16px; padding: 28px;
        height: 100%; position: relative; overflow: hidden;
    }
    .page-card::before {
        content: ''; position: absolute;
        top: 0; left: 0; right: 0; height: 2px;
    }
    .page-card.blue::before   { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
    .page-card.purple::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
    .page-card.cyan::before   { background: linear-gradient(90deg, #06b6d4, #22d3ee); }
    .card-icon  { font-size: 32px; margin-bottom: 14px; }
    .card-title { font-size: 18px; font-weight: 600; color: #f1f5f9; margin-bottom: 8px; }
    .card-desc  { font-size: 13px; color: #475569; line-height: 1.6; margin-bottom: 16px; }
    .card-features { list-style: none; padding: 0; margin: 0; }
    .card-features li {
        font-size: 12px; color: #334155;
        padding: 3px 0;
    }
    .card-features li::before { content: '→ '; color: #1e3a5f; }

    /* Pipeline */
    .pipeline-step {
        background: #161b27; border: 1px solid #1e2535;
        border-radius: 10px; padding: 14px 12px;
        text-align: center;
    }
    .pipeline-step.done   { border-color: #1e3a5f; }
    .pipeline-step.active { border-color: #3b82f6; background: rgba(59,130,246,0.05); }
    .step-icon   { font-size: 20px; margin-bottom: 6px; }
    .step-label  { font-size: 11px; color: #475569; }
    .step-status { font-size: 10px; margin-top: 4px; }
    .step-status.done   { color: #34d399; }
    .step-status.active { color: #60a5fa; }
    .step-status.todo   { color: #334155; }
    .pipeline-arrow { text-align: center; color: #1e2535; font-size: 20px; margin-top: 20px; }

    /* Tech badges */
    .tech-badge {
        display: inline-block;
        background: #161b27; border: 1px solid #1e2535;
        border-radius: 8px; padding: 8px 16px;
        font-size: 12px; color: #64748b;
        font-family: 'JetBrains Mono', monospace;
        margin: 4px;
    }
    .tech-title {
        font-size: 12px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.12em;
        color: #334155; margin-bottom: 16px;
        text-align: center;
    }
    .divider {
        border: none; border-top: 1px solid #1e2535;
        margin: 40px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 24px 0;'>
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
        <div style='padding:9px 12px; border-radius:8px; background:{bg};
                    border:1px solid {border}; margin-bottom:4px;
                    font-size:13px; color:{color}; font-weight:{weight};'>
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
<div style='text-align:center; padding:60px 40px 40px 40px;'>
    <div class='hero-badge'>● M5 Forecasting Competition · Walmart Retail</div>
    <div class='hero-title'>
        Sales Forecasting<br><span>Dashboard</span>
    </div>
    <div class='hero-subtitle'>
        End-to-end sales forecasting pipeline for 30,490 Walmart products<br>
        across 10 stores using Facebook Prophet with custom regressors.
    </div>
</div>
""", unsafe_allow_html=True)


# ── Stats row — dùng st.columns ────────────────────────────────────────────────
s1, s2, s3, s4, s5 = st.columns(5)

with s1:
    st.markdown("<div class='stat-value'>30,490</div><div class='stat-label'>Prophet Models</div>", unsafe_allow_html=True)
with s2:
    st.markdown("<div class='stat-value'>58M</div><div class='stat-label'>Data Points</div>", unsafe_allow_html=True)
with s3:
    st.markdown("<div class='stat-value'>1,913</div><div class='stat-label'>Days History</div>", unsafe_allow_html=True)
with s4:
    st.markdown("<div class='stat-value'>28</div><div class='stat-label'>Day Forecast</div>", unsafe_allow_html=True)
with s5:
    st.markdown("<div class='stat-value'>3</div><div class='stat-label'>US States</div>", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ── Page cards — dùng st.columns ──────────────────────────────────────────────
st.markdown("<div style='text-align:center; font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:0.12em; color:#334155; margin-bottom:24px;'>Dashboard Pages</div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
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
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
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
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
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
    """, unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ── Pipeline — dùng st.columns ─────────────────────────────────────────────────
st.markdown("<div style='text-align:center; font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:0.12em; color:#334155; margin-bottom:24px;'>Project Pipeline</div>", unsafe_allow_html=True)

p1, a1, p2, a2, p3, a3, p4, a4, p5, a5, p6 = st.columns([3,1,3,1,3,1,3,1,3,1,3])

with p1:
    st.markdown("""
    <div class='pipeline-step done'>
        <div class='step-icon'>📥</div>
        <div class='step-label'>Ingestion</div>
        <div class='step-status done'>✓ Done</div>
    </div>""", unsafe_allow_html=True)
with a1:
    st.markdown("<div class='pipeline-arrow'>→</div>", unsafe_allow_html=True)
with p2:
    st.markdown("""
    <div class='pipeline-step done'>
        <div class='step-icon'>🧹</div>
        <div class='step-label'>Cleaning</div>
        <div class='step-status done'>✓ Done</div>
    </div>""", unsafe_allow_html=True)
with a2:
    st.markdown("<div class='pipeline-arrow'>→</div>", unsafe_allow_html=True)
with p3:
    st.markdown("""
    <div class='pipeline-step done'>
        <div class='step-icon'>🔗</div>
        <div class='step-label'>Merge</div>
        <div class='step-status done'>✓ Done</div>
    </div>""", unsafe_allow_html=True)
with a3:
    st.markdown("<div class='pipeline-arrow'>→</div>", unsafe_allow_html=True)
with p4:
    st.markdown("""
    <div class='pipeline-step done'>
        <div class='step-icon'>⚙️</div>
        <div class='step-label'>Features</div>
        <div class='step-status done'>✓ Done</div>
    </div>""", unsafe_allow_html=True)
with a4:
    st.markdown("<div class='pipeline-arrow'>→</div>", unsafe_allow_html=True)
with p5:
    st.markdown("""
    <div class='pipeline-step active'>
        <div class='step-icon'>🤖</div>
        <div class='step-label'>Training</div>
        <div class='step-status active'>● Active</div>
    </div>""", unsafe_allow_html=True)
with a5:
    st.markdown("<div class='pipeline-arrow'>→</div>", unsafe_allow_html=True)
with p6:
    st.markdown("""
    <div class='pipeline-step'>
        <div class='step-icon'>📊</div>
        <div class='step-label'>Evaluation</div>
        <div class='step-status todo'>○ Pending</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ── Tech stack ─────────────────────────────────────────────────────────────────
st.markdown("<div class='tech-title'>Tech Stack</div>", unsafe_allow_html=True)

tech_cols = st.columns(10)
techs = [
    "Python 3.11", "Prophet", "Pandas", "NumPy",
    "Scikit-learn", "Plotly", "Streamlit", "Joblib",
    "PyArrow", "M5 Dataset"
]
for col, tech in zip(tech_cols, techs):
    with col:
        st.markdown(f"<div class='tech-badge'>{tech}</div>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)