"""
Afficionado Demand Intelligence Platform — Main Entrypoint
===========================================================
Run:  streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import glob
from datetime import datetime, timedelta
from pathlib import Path

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Afficionado Demand Intelligence Platform",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Project Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FORECAST_STORE = DATA_DIR / "forecast_store"
DB_PATH = DATA_DIR / "daily_store.db"

# ─── Store Mapping ──────────────────────────────────────────────────────────
STORE_MAP = {"Astoria": 3, "Hell's Kitchen": 8, "Lower Manhattan": 5}
STORE_NAMES = list(STORE_MAP.keys())
STORE_COLORS = {
    "Astoria": "#00D4AA",
    "Hell's Kitchen": "#FF6B6B",
    "Lower Manhattan": "#4ECDC4",
}
MODEL_OPTIONS = ["LightGBM", "XGBoost", "Prophet", "Ensemble"]

# ─── Custom CSS ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Global Dark Theme Overrides ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* Main container */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e6edf3;
}

/* KPI Metric Cards */
.kpi-card {
    background: linear-gradient(145deg, #1e2a3a 0%, #162032 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px 28px;
    margin: 8px 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08);
}
.kpi-card .kpi-label {
    color: #8b949e;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 6px;
}
.kpi-card .kpi-value {
    color: #f0f6fc;
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.1;
}
.kpi-card .kpi-delta {
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 8px;
}
.kpi-delta.positive { color: #3fb950; }
.kpi-delta.negative { color: #f85149; }
.kpi-delta.neutral  { color: #d29922; }

/* Section Headers */
.section-header {
    color: #f0f6fc;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 32px 0 16px 0;
    padding-bottom: 12px;
    border-bottom: 2px solid rgba(0,212,170,0.3);
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Data Tables */
.styled-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    background: #161b22;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}
.styled-table thead th {
    background: #1c2333;
    color: #8b949e;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 14px 18px;
    text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.styled-table tbody td {
    padding: 12px 18px;
    color: #e6edf3;
    font-size: 0.88rem;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.styled-table tbody tr:hover {
    background: rgba(0,212,170,0.04);
}

/* Alert badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-green  { background: rgba(63,185,80,0.15); color: #3fb950; }
.badge-yellow { background: rgba(210,153,34,0.15); color: #d29922; }
.badge-red    { background: rgba(248,81,73,0.15); color: #f85149; }
.badge-blue   { background: rgba(56,139,253,0.15); color: #388bfd; }

/* Stale data indicator */
.stale-badge {
    background: rgba(248,81,73,0.12);
    color: #f85149;
    border: 1px solid rgba(248,81,73,0.25);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    text-align: center;
    margin: 8px 0;
}
.fresh-badge {
    background: rgba(63,185,80,0.12);
    color: #3fb950;
    border: 1px solid rgba(63,185,80,0.25);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    text-align: center;
    margin: 8px 0;
}

/* Hero Section */
.hero-container {
    text-align: center;
    padding: 60px 20px 40px 20px;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00D4AA 0%, #4ECDC4 50%, #388bfd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 12px;
    line-height: 1.15;
}
.hero-subtitle {
    color: #8b949e;
    font-size: 1.15rem;
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto 40px auto;
    line-height: 1.6;
}

/* Navigation Cards */
.nav-card {
    background: linear-gradient(145deg, #1e2a3a 0%, #162032 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    transition: all 0.3s ease;
    cursor: pointer;
    min-height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
.nav-card:hover {
    border-color: rgba(0,212,170,0.3);
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(0,0,0,0.35);
}
.nav-card .nav-icon {
    font-size: 2.5rem;
    margin-bottom: 14px;
}
.nav-card .nav-title {
    color: #f0f6fc;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.nav-card .nav-desc {
    color: #8b949e;
    font-size: 0.82rem;
    line-height: 1.45;
}

/* Plotly chart container */
.plotly-chart-container {
    background: #161b22;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 20px;
    margin: 12px 0;
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

/* Streamlit overrides */
.stSelectbox label, .stMultiSelect label, .stDateInput label,
.stSlider label, .stRadio label, .stNumberInput label {
    color: #e6edf3 !important;
    font-weight: 500 !important;
}
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1e2a3a 0%, #162032 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}
div[data-testid="stMetric"] label { color: #8b949e !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #f0f6fc !important;
    font-weight: 800 !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─── Data Loading Functions ─────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_forecast_store() -> pd.DataFrame:
    """Load all Parquet files from data/forecast_store/ into a single DataFrame."""
    parquet_files = list(FORECAST_STORE.glob("*.parquet"))
    if not parquet_files:
        return _generate_demo_forecast()
    frames = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return _generate_demo_forecast()
    combined = pd.concat(frames, ignore_index=True)
    return combined


@st.cache_data(ttl=3600, show_spinner=False)
def load_actuals() -> pd.DataFrame:
    """Load historical actuals from SQLite daily_store database."""
    if not DB_PATH.exists():
        return _generate_demo_actuals()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql("SELECT * FROM daily_store", conn)
        conn.close()
        if df.empty:
            return _generate_demo_actuals()
        return df
    except Exception:
        return _generate_demo_actuals()


def _generate_demo_forecast() -> pd.DataFrame:
    """Generate realistic demo forecast data when no real data is available."""
    np.random.seed(42)
    dates = pd.date_range(
        start=datetime.now().date() - timedelta(days=30),
        end=datetime.now().date() + timedelta(days=30),
        freq="D",
    )
    rows = []
    base_rev = {"Astoria": 2800, "Hell's Kitchen": 3400, "Lower Manhattan": 4200}
    for store_name, store_id in STORE_MAP.items():
        base = base_rev[store_name]
        for i, d in enumerate(dates):
            dow = d.dayofweek
            seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * i / 7)
            weekend_boost = 1.2 if dow >= 5 else 1.0
            trend = 1 + 0.002 * i
            yhat = base * seasonal * weekend_boost * trend
            noise = np.random.normal(0, base * 0.08)
            yhat += noise
            ci80 = base * 0.12
            ci95 = base * 0.20
            transactions = int(yhat / 6.5) + np.random.randint(-20, 20)
            rows.append(
                {
                    "ds": d,
                    "store_id": store_id,
                    "store_name": store_name,
                    "yhat": round(yhat, 2),
                    "yhat_lower_80": round(yhat - ci80, 2),
                    "yhat_upper_80": round(yhat + ci80, 2),
                    "yhat_lower_95": round(yhat - ci95, 2),
                    "yhat_upper_95": round(yhat + ci95, 2),
                    "transactions": max(transactions, 50),
                    "model": np.random.choice(MODEL_OPTIONS[:3]),
                    "category_coffee": round(yhat * 0.45, 2),
                    "category_tea": round(yhat * 0.15, 2),
                    "category_pastry": round(yhat * 0.22, 2),
                    "category_sandwich": round(yhat * 0.12, 2),
                    "category_other": round(yhat * 0.06, 2),
                }
            )
    df = pd.DataFrame(rows)
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def _generate_demo_actuals() -> pd.DataFrame:
    """Generate realistic demo actuals data when no DB is available."""
    np.random.seed(123)
    dates = pd.date_range(
        start=datetime.now().date() - timedelta(days=90),
        end=datetime.now().date() - timedelta(days=1),
        freq="D",
    )
    rows = []
    base_rev = {"Astoria": 2750, "Hell's Kitchen": 3350, "Lower Manhattan": 4100}
    for store_name, store_id in STORE_MAP.items():
        base = base_rev[store_name]
        for i, d in enumerate(dates):
            dow = d.dayofweek
            seasonal = 1.0 + 0.12 * np.sin(2 * np.pi * i / 7)
            weekend_boost = 1.18 if dow >= 5 else 1.0
            trend = 1 + 0.0015 * i
            actual = base * seasonal * weekend_boost * trend
            actual += np.random.normal(0, base * 0.06)
            transactions = int(actual / 6.5) + np.random.randint(-15, 15)
            # Hourly distribution (simplified — peak hours)
            for hour in range(6, 21):
                hour_frac = _hour_weight(hour)
                hour_rev = actual * hour_frac
                hour_txn = max(int(transactions * hour_frac), 1)
                rows.append(
                    {
                        "date": d,
                        "store_id": store_id,
                        "store_name": store_name,
                        "hour": hour,
                        "revenue": round(hour_rev, 2),
                        "transactions": hour_txn,
                        "category_coffee": round(hour_rev * 0.44, 2),
                        "category_tea": round(hour_rev * 0.14, 2),
                        "category_pastry": round(hour_rev * 0.23, 2),
                        "category_sandwich": round(hour_rev * 0.13, 2),
                        "category_other": round(hour_rev * 0.06, 2),
                    }
                )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _hour_weight(hour: int) -> float:
    """Return fraction of daily revenue for a given hour (realistic coffee shop)."""
    weights = {
        6: 0.04, 7: 0.08, 8: 0.12, 9: 0.11, 10: 0.10,
        11: 0.09, 12: 0.09, 13: 0.08, 14: 0.06, 15: 0.05,
        16: 0.05, 17: 0.04, 18: 0.04, 19: 0.03, 20: 0.02,
    }
    return weights.get(hour, 0.03)


# ─── Session State Init ─────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "selected_stores": ["All"],
        "date_range": (
            datetime.now().date() - timedelta(days=30),
            datetime.now().date() + timedelta(days=30),
        ),
        "selected_model": "Ensemble",
        "forecast_df": None,
        "actuals_df": None,
        "data_loaded": False,
        "last_refresh": None,
        "store_map": STORE_MAP,
        "store_names": STORE_NAMES,
        "store_colors": STORE_COLORS,
        "model_options": MODEL_OPTIONS,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 16px 0 8px 0;">
            <span style="font-size:2.2rem;">☕</span>
            <h2 style="margin:4px 0 0 0; font-size:1.1rem; font-weight:800;
                        background: linear-gradient(135deg,#00D4AA,#4ECDC4);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text;">ADIP</h2>
            <p style="color:#8b949e; font-size:0.7rem; margin:0; letter-spacing:1px;">
                DEMAND INTELLIGENCE</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Store Selector
    selected = st.multiselect(
        "📍 Stores",
        options=["All"] + STORE_NAMES,
        default=st.session_state.selected_stores,
        help="Select stores to analyze",
    )
    if "All" in selected or not selected:
        st.session_state.selected_stores = ["All"]
    else:
        st.session_state.selected_stores = selected

    # Date Range
    st.markdown("")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input(
            "📅 From",
            value=st.session_state.date_range[0],
            key="sidebar_start_date",
        )
    with col_d2:
        end_date = st.date_input(
            "📅 To",
            value=st.session_state.date_range[1],
            key="sidebar_end_date",
        )
    st.session_state.date_range = (start_date, end_date)

    # Model Selector
    st.markdown("")
    st.session_state.selected_model = st.selectbox(
        "🤖 Model",
        options=MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(st.session_state.selected_model),
    )

    st.markdown("---")

    # Refresh Button
    if st.button("🔄  Refresh Data", use_container_width=True, type="primary"):
        st.cache_data.clear()
        st.session_state.data_loaded = False
        st.session_state.last_refresh = datetime.now()
        st.rerun()

    # Data Freshness Badge
    st.markdown("")
    if st.session_state.last_refresh:
        hours_since = (
            datetime.now() - st.session_state.last_refresh
        ).total_seconds() / 3600
        if hours_since < 1:
            st.markdown(
                '<div class="fresh-badge">✅ Data is fresh</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="stale-badge">⚠️ Data is {hours_since:.1f}h old</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="fresh-badge">🔹 Ready to load</div>',
            unsafe_allow_html=True,
        )

    # Health Indicators
    st.markdown("---")
    st.markdown(
        "<p style='color:#8b949e;font-size:0.75rem;font-weight:600;"
        "letter-spacing:1px;text-transform:uppercase;'>System Health</p>",
        unsafe_allow_html=True,
    )
    parquet_files = list(FORECAST_STORE.glob("*.parquet")) if FORECAST_STORE.exists() else []
    db_ok = DB_PATH.exists()

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        if parquet_files:
            st.success(f"Forecasts: {len(parquet_files)}", icon="✅")
        else:
            st.info("Forecasts: Demo", icon="🔹")
    with col_h2:
        if db_ok:
            st.success("DB: Online", icon="✅")
        else:
            st.info("DB: Demo", icon="🔹")


# ─── Load Data ──────────────────────────────────────────────────────────────
if not st.session_state.data_loaded:
    with st.spinner("Loading intelligence data…"):
        st.session_state.forecast_df = load_forecast_store()
        st.session_state.actuals_df = load_actuals()
        st.session_state.data_loaded = True
        if st.session_state.last_refresh is None:
            st.session_state.last_refresh = datetime.now()


# ─── Helper: Filter by Selected Stores ───────────────────────────────────────
def get_active_stores() -> list:
    """Return list of active store names based on sidebar selection."""
    if "All" in st.session_state.selected_stores:
        return STORE_NAMES
    return st.session_state.selected_stores


# ─── Main Landing Page ──────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">Afficionado Demand<br>Intelligence Platform</div>
        <div class="hero-subtitle">
            AI-powered demand forecasting, scenario planning, and operational
            intelligence for specialty coffee retail.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation Cards
nav_items = [
    ("📊", "Executive Overview", "KPIs, revenue forecasts, and peak demand alerts", "1_Executive_Overview"),
    ("🏪", "Store Forecast", "Per-store deep-dive with accuracy metrics", "2_Store_Forecast"),
    ("🔥", "Hourly Heatmap", "Demand patterns by hour and day of week", "3_Hourly_Heatmap"),
    ("🏷️", "Category Intel", "Product category trends and basket analysis", "4_Category_Intelligence"),
    ("🧪", "Model Comparison", "Model leaderboard and validation results", "5_Model_Comparison"),
    ("🎯", "Scenario Planner", "What-if analysis and P&L impact modeling", "6_Scenario_Planner"),
]

cols = st.columns(3)
for i, (icon, title, desc, page) in enumerate(nav_items):
    with cols[i % 3]:
        st.markdown(
            f"""
            <div class="nav-card">
                <div class="nav-icon">{icon}</div>
                <div class="nav-title">{title}</div>
                <div class="nav-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# Quick Stats Row
if st.session_state.data_loaded:
    fc = st.session_state.forecast_df
    ac = st.session_state.actuals_df

    today = pd.Timestamp(datetime.now().date())
    future_fc = fc[fc["ds"] >= today] if "ds" in fc.columns else fc
    total_fc_rev = future_fc["yhat"].sum() if "yhat" in future_fc.columns else 0

    st.markdown(
        '<div class="section-header">⚡ Quick Status</div>',
        unsafe_allow_html=True,
    )
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.metric("Stores Monitored", len(STORE_MAP))
    with q2:
        st.metric("Forecast Horizon", "30 days")
    with q3:
        n_models = len(fc["model"].unique()) if "model" in fc.columns else len(MODEL_OPTIONS)
        st.metric("Models Active", n_models)
    with q4:
        st.metric(
            "30-Day Forecast Revenue",
            f"${total_fc_rev:,.0f}",
        )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; padding:12px 0;">
        <span style="color:#484f58; font-size:0.75rem;">
            Afficionado Demand Intelligence Platform v2.0 &nbsp;•&nbsp;
            Powered by Streamlit + Plotly &nbsp;•&nbsp;
            Last build: June 2026
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
