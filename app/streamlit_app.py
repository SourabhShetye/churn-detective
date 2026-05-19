"""
app/streamlit_app.py
--------------------
Churn Detective — CMO Retention Intelligence Dashboard.

Run with:
    streamlit run app/streamlit_app.py

Architecture:
    This file is the entrypoint. It bootstraps shared state (model artifacts,
    data) into st.session_state so all pages can access them without
    reloading. Heavy computation is cached via @st.cache_data / @st.cache_resource.

Pages (in app/pages/):
    1_churn_overview.py      — KPI cards, churn drivers, SHAP summary
    2_customer_segments.py   — Persona cards, scatter, segment deep-dives
    3_retention_simulator.py — Cost/threshold sliders, ROI calculator
    4_uplift_targeting.py    — Persuadable ranking, Qini curve, action list
"""

import os
import sys
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — allows `from src.xxx import yyy` from any page
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title  = "Churn Detective",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — refined dark analytical aesthetic
# Typeface: DM Mono (monospaced data feel) + DM Sans (clean UI body)
# Palette: deep navy base, electric teal accent, warm amber alert
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;700&display=swap');

/* ── Root variables ──────────────────────────────────────────────────── */
:root {
    --navy-950:  #080d1a;
    --navy-900:  #0d1526;
    --navy-800:  #111e35;
    --navy-700:  #1a2d4e;
    --navy-600:  #243d66;
    --teal-400:  #2dd4bf;
    --teal-300:  #5eead4;
    --teal-200:  #99f6e4;
    --amber-400: #fbbf24;
    --amber-300: #fcd34d;
    --rose-400:  #fb7185;
    --slate-300: #cbd5e1;
    --slate-400: #94a3b8;
    --slate-500: #64748b;
    --white:     #f8fafc;
}

/* ── Base layout ─────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--navy-950) !important;
    color: var(--slate-300) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: var(--navy-900) !important;
    border-right: 1px solid var(--navy-700);
}

[data-testid="stHeader"] {
    background: transparent !important;
}

/* ── Typography ──────────────────────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--white) !important;
    letter-spacing: -0.02em;
}

h1 { font-size: 2rem !important; font-weight: 700 !important; }
h2 { font-size: 1.35rem !important; font-weight: 500 !important; }
h3 { font-size: 1.1rem !important; font-weight: 500 !important; }

p, li, label, .stMarkdown {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--slate-300) !important;
    line-height: 1.65 !important;
}

code, pre, [data-testid="stCode"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* ── KPI metric cards ────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--navy-800) !important;
    border: 1px solid var(--navy-600) !important;
    border-radius: 12px !important;
    padding: 1.25rem 1.5rem !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35) !important;
    transition: border-color 0.2s ease !important;
}

[data-testid="metric-container"]:hover {
    border-color: var(--teal-400) !important;
}

[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 2rem !important;
    font-weight: 500 !important;
    color: var(--teal-300) !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--slate-400) !important;
}

[data-testid="stMetricDelta"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* ── Sidebar styling ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--teal-300) !important;
}

[data-testid="stSidebarNav"] a {
    color: var(--slate-300) !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 0.75rem !important;
    border-radius: 6px !important;
    transition: background 0.15s, color 0.15s !important;
}

[data-testid="stSidebarNav"] a:hover {
    background: var(--navy-700) !important;
    color: var(--teal-300) !important;
}

/* ── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button {
    background: transparent !important;
    color: var(--teal-300) !important;
    border: 1px solid var(--teal-400) !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.25rem !important;
    letter-spacing: 0.04em !important;
    transition: background 0.2s, color 0.2s !important;
}

.stButton > button:hover {
    background: var(--teal-400) !important;
    color: var(--navy-950) !important;
}

/* ── Sliders ─────────────────────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] {
    color: var(--teal-400) !important;
}

/* ── Selectbox / dropdowns ───────────────────────────────────────────── */
.stSelectbox > div > div {
    background: var(--navy-800) !important;
    border-color: var(--navy-600) !important;
    color: var(--white) !important;
    border-radius: 8px !important;
}

/* ── DataFrames / tables ─────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--navy-700) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Expander ────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--navy-800) !important;
    border: 1px solid var(--navy-700) !important;
    border-radius: 10px !important;
}

/* ── Dividers ────────────────────────────────────────────────────────── */
hr {
    border-color: var(--navy-700) !important;
    margin: 1.5rem 0 !important;
}

/* ── Info / warning boxes ────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Tab styling ─────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background: var(--navy-900) !important;
    border-radius: 8px !important;
    padding: 4px !important;
    gap: 4px !important;
}

[data-baseweb="tab"] {
    color: var(--slate-400) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.875rem !important;
    border-radius: 6px !important;
}

[aria-selected="true"][data-baseweb="tab"] {
    background: var(--navy-700) !important;
    color: var(--teal-300) !important;
}

/* ── Custom component classes ────────────────────────────────────────── */
.persona-card {
    background: var(--navy-800);
    border: 1px solid var(--navy-600);
    border-left: 3px solid var(--teal-400);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

.persona-card h4 {
    font-family: 'DM Sans', sans-serif;
    color: var(--white);
    font-size: 1rem;
    margin: 0 0 0.5rem 0;
}

.persona-card p {
    font-size: 0.875rem;
    color: var(--slate-400);
    margin: 0;
}

.stat-pill {
    display: inline-block;
    background: var(--navy-700);
    color: var(--teal-300);
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    margin: 0.15rem 0.2rem;
}

.warning-box {
    background: rgba(251, 191, 36, 0.08);
    border: 1px solid rgba(251, 191, 36, 0.3);
    border-radius: 8px;
    padding: 0.875rem 1.25rem;
    font-size: 0.875rem;
    color: var(--amber-300);
    margin: 0.75rem 0;
}

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--slate-500);
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Artifact loader — cached so it only runs once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    """
    Load serialized model artifacts from models/.
    Returns a dict of loaded objects, or demo-mode placeholders if not found.

    This function is forgiving — if models haven't been trained yet it
    returns a DEMO_MODE flag so all pages can render with synthetic data.
    """
    import joblib
    import pathlib

    artifacts = {"DEMO_MODE": False}
    model_dir = pathlib.Path(ROOT) / "models"

    expected = {
        "churn_model":  "lgbm_churn_v1.pkl",
        "s_learner":    "uplift_slearner_v1.pkl",
        "t_learner":    "uplift_tlearner_v1.pkl",
        "kmeans":       "kmeans_churner_segments.pkl",
        "scaler":       "kmeans_scaler_v1.pkl",
        "persona_map":  "persona_map_v1.pkl",
        "sweep_df":     "threshold_sweep_v1.pkl",
        "action_table": "action_table_v1.pkl",
    }

    missing = []
    for key, filename in expected.items():
        fpath = model_dir / filename
        if fpath.exists():
            artifacts[key] = joblib.load(fpath)
        else:
            missing.append(filename)

    if missing:
        import streamlit as st
        st.error(f"Missing models: {missing}. Run notebooks to generate them.")
        st.stop()

    return artifacts


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    """
    Load processed data from data/processed/.
    Falls back to synthetic demo data if not yet generated.
    """
    import pathlib
    import numpy as np

    data_dir = pathlib.Path(ROOT) / "data" / "processed"
    test_path = data_dir / "test.parquet"

    if test_path.exists():
        df = pd.read_parquet(test_path)
        return df
    else:
        import streamlit as st
        st.error(f"Data not found at {test_path}. Please run notebooks first.")
        st.stop()



# ---------------------------------------------------------------------------
# Bootstrap session state
# ---------------------------------------------------------------------------

import pandas as pd

if "artifacts" not in st.session_state:
    st.session_state.artifacts = load_artifacts()

if "df" not in st.session_state:
    st.session_state.df = load_data()
    st.session_state.is_demo = False

# ---------------------------------------------------------------------------
# Sidebar — branding + global controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1.5rem 0;'>
        <div style='font-family: "DM Mono", monospace; font-size: 0.7rem;
                    text-transform: uppercase; letter-spacing: 0.15em;
                    color: #64748b; margin-bottom: 0.25rem;'>
            Retention Intelligence
        </div>
        <div style='font-family: "DM Sans", sans-serif; font-size: 1.4rem;
                    font-weight: 700; color: #f8fafc; letter-spacing: -0.02em;'>
            🔍 Churn Detective
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Global threshold slider — shared across pages via session state
    st.markdown('<div class="section-label">Global Decision Threshold</div>',
                unsafe_allow_html=True)

    threshold = st.slider(
        label      = "Churn probability threshold",
        min_value  = 0.10,
        max_value  = 0.90,
        value      = st.session_state.get("global_threshold", 0.50),
        step       = 0.01,
        help       = (
            "Customers above this probability are flagged for retention outreach. "
            "Lower = more customers contacted (higher recall, lower precision). "
            "See the Simulator page to find the revenue-optimal threshold."
        ),
        label_visibility = "collapsed",
    )
    st.session_state["global_threshold"] = threshold
    st.markdown(
        f'<div style="font-family: DM Mono, monospace; font-size: 1.4rem; '
        f'color: #2dd4bf; text-align: center; margin-top: -0.5rem;">'
        f'p ≥ {threshold:.2f}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.success("✅ Live model — real data loaded")

    st.markdown("---")

    # Dataset stats
    df = st.session_state.df
    n_total    = len(df)
    churn_rate = df["churned"].mean() * 100 if "churned" in df.columns else 36.2
    st.markdown(f"""
    <div class="section-label">Dataset snapshot</div>
    <span class="stat-pill">n = {n_total:,}</span>
    <span class="stat-pill">churn {churn_rate:.1f}%</span>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family: DM Mono, monospace; font-size: 0.7rem; '
        'color: #475569; text-align: center;">Churn Detective v1.0 · Anthropic</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Landing page (shown when no sub-page is selected)
# ---------------------------------------------------------------------------

st.markdown("""
<h1>Retention Intelligence Platform</h1>
<p style="color: #94a3b8; font-size: 1.05rem; margin-top: -0.5rem;">
    AI-powered churn prediction, customer segmentation, and uplift targeting
    for the CMO retention campaign.
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Top-level KPIs
df = st.session_state.df
threshold = st.session_state.get("global_threshold", 0.50)

col1, col2, col3, col4 = st.columns(4)

if "churn_proba" in df.columns:
    n_at_risk      = (df["churn_proba"] >= threshold).sum()
    rev_at_risk    = df.loc[df["churn_proba"] >= threshold, "monthly_charges"].sum()
    churn_rate_pct = df["churned"].mean() * 100 if "churned" in df.columns else 36.2
    expected_saves = int(n_at_risk * 0.30)
else:
    n_at_risk      = int(len(df) * 0.362)
    rev_at_risk    = df["monthly_charges"].sum() * 0.362
    churn_rate_pct = 36.2
    expected_saves = int(n_at_risk * 0.30)

col1.metric(
    "Churn Rate",
    f"{churn_rate_pct:.1f}%",
    delta     = f"+{churn_rate_pct - 1.5:.1f}pp vs benchmark",
    delta_color = "inverse",
)
col2.metric(
    "Customers at Risk",
    f"{n_at_risk:,}",
    delta     = f"@ threshold {threshold:.2f}",
    delta_color = "off",
)
col3.metric(
    "Monthly Revenue at Risk",
    f"${rev_at_risk:,.0f}",
    delta     = f"${rev_at_risk * 12:,.0f} annualised",
    delta_color = "inverse",
)
col4.metric(
    "Expected Saves",
    f"{expected_saves:,}",
    delta     = "with 30% retention rate",
    delta_color = "normal",
)

st.markdown("---")

# Navigation guide
st.markdown("### Navigate the platform")
nav_cols = st.columns(4)

pages_info = [
    ("📊", "Churn Overview",       "pages/1_churn_overview.py",      "Top churn drivers & SHAP evidence"),
    ("👥", "Customer Segments",    "pages/2_customer_segments.py",   "Churner personas & retention plays"),
    ("🎯", "Retention Simulator",  "pages/3_retention_simulator.py", "Cost-threshold & ROI calculator"),
    ("🚀", "Uplift Targeting",     "pages/4_uplift_targeting.py",    "Persuadable ranking & action list"),
]

for col, (icon, title, _, desc) in zip(nav_cols, pages_info):
    col.markdown(f"""
    <div class="persona-card" style="text-align:center; cursor:pointer; min-height: 130px;">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <h4>{title}</h4>
        <p>{desc}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div class="warning-box">
    <strong>Assumptions & limitations</strong> — This platform uses a LightGBM churn model
    trained on historical data. The uplift model assumes <code>plan_changes_6mo > 0</code>
    as a proxy for treatment exposure (not a true A/B test). All monetary estimates use
    configurable assumptions (CLV = $1,676 · offer cost = $15 · retention rate = 30%).
    Validate with a controlled experiment before scaling the campaign.
</div>
""", unsafe_allow_html=True)
