"""
Scenario Planner — What-If Analysis & P&L Impact
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

st.set_page_config(
    page_title="Scenario Planner | ADIP",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

STORE_MAP = {"Astoria": 3, "Hell's Kitchen": 8, "Lower Manhattan": 5}
STORE_NAMES = list(STORE_MAP.keys())

# Cost assumptions
INVENTORY_COST_PER_UNIT = 1.20  # $/unit
STAFF_COST_PER_HOUR = 25.0  # $/hr
UNITS_PER_FTE_PER_HOUR = 150  # units/hr capacity per FTE
HOURS_PER_DAY = 15  # operating hours (6 AM - 9 PM)

PAL = {
    "bg": "rgba(0,0,0,0)",
    "grid": "rgba(255,255,255,0.04)",
    "text": "#e6edf3",
    "muted": "#8b949e",
    "best": "#3fb950",
    "base": "#388bfd",
    "worst": "#f85149",
}

st.markdown(
    """
<style>
.scenario-header {
    background: linear-gradient(135deg, #1a2332 0%, #162032 100%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
}
.scenario-header h2 {
    color: #f0f6fc; font-size: 1.3rem; font-weight: 800; margin: 0 0 8px 0;
}
.scenario-header p {
    color: #8b949e; font-size: 0.88rem; line-height: 1.6; margin: 0;
}
.pnl-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    background: #161b22;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
    margin-top: 16px;
}
.pnl-table thead th {
    background: #1c2333;
    color: #8b949e;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 14px 18px;
    text-align: right;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.pnl-table thead th:first-child { text-align: left; }
.pnl-table tbody td {
    padding: 14px 18px;
    color: #e6edf3;
    font-size: 0.88rem;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    text-align: right;
    font-weight: 500;
}
.pnl-table tbody td:first-child {
    text-align: left;
    font-weight: 700;
}
.pnl-table tbody tr:hover {
    background: rgba(56,139,253,0.04);
}
.pnl-positive { color: #3fb950 !important; }
.pnl-negative { color: #f85149 !important; }
.pnl-neutral  { color: #388bfd !important; }
.pnl-best-row { background: rgba(63,185,80,0.06); }
.pnl-worst-row { background: rgba(248,81,73,0.06); }

.controls-card {
    background: linear-gradient(145deg, #1e2a3a, #162032);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.controls-card h4 {
    color: #f0f6fc; font-weight: 700; font-size: 1rem; margin-bottom: 16px;
}
</style>
""",
    unsafe_allow_html=True,
)


def get_data():
    fc = st.session_state.get("forecast_df")
    if fc is None:
        st.warning("⚠️ No data loaded. Please visit the main page first.")
        st.stop()
    return fc.copy()


# ─── Page Header ────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="scenario-header">
        <h2>🎯 Scenario Planner</h2>
        <p>
            Model the impact of demand shocks on revenue, inventory, and staffing costs.
            Adjust the demand multiplier to simulate best-case, base-case, and worst-case scenarios.
            Use this tool for capacity planning, budget forecasting, and risk assessment.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

fc = get_data()
today = pd.Timestamp(datetime.now().date())

# ═══════════════════════════════════════════════════════════════════════════
# Layout: Controls (30%) + Chart (70%)
# ═══════════════════════════════════════════════════════════════════════════
ctrl_col, chart_col = st.columns([3, 7])

# ─── Controls ───────────────────────────────────────────────────────────────
with ctrl_col:
    st.markdown(
        '<div class="controls-card"><h4>⚙️ Scenario Parameters</h4></div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    demand_shock = st.slider(
        "📊 Demand Shock (%)",
        min_value=-30,
        max_value=30,
        value=0,
        step=1,
        format="%+d%%",
        help="Adjust base forecast by this percentage",
    )

    st.markdown("")

    scenario_store = st.selectbox(
        "🏪 Store",
        ["All Stores"] + STORE_NAMES,
        index=0,
        help="Apply scenario to specific store or all stores",
    )

    st.markdown("")

    horizon_options = {"1 Week": 7, "2 Weeks": 14, "1 Month": 30}
    horizon_label = st.selectbox(
        "📅 Planning Horizon",
        list(horizon_options.keys()),
        index=2,
    )
    horizon_days = horizon_options[horizon_label]

    st.markdown("---")

    # Cost Assumptions (display only)
    st.markdown(
        """
        <div style="margin-top:8px;">
            <p style="color:#8b949e; font-size:0.72rem; font-weight:600;
                      text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">
                Cost Assumptions
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="color:#e6edf3; font-size:0.82rem; line-height:2;">
            📦 Inventory: <b>${INVENTORY_COST_PER_UNIT:.2f}/unit</b><br>
            👤 Staff: <b>${STAFF_COST_PER_HOUR:.0f}/hr</b><br>
            ⚡ Capacity: <b>{UNITS_PER_FTE_PER_HOUR} units/FTE/hr</b><br>
            🕐 Hours/day: <b>{HOURS_PER_DAY}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Scenario Calculations ─────────────────────────────────────────────────
# Filter forecast data
future_fc = fc[
    (fc["ds"] >= today) & (fc["ds"] < today + timedelta(days=horizon_days))
].copy()

if scenario_store != "All Stores":
    future_fc = future_fc[future_fc["store_name"] == scenario_store]

if future_fc.empty:
    with chart_col:
        st.warning("No forecast data available for the selected parameters.")
    st.stop()

# Daily aggregation
daily = future_fc.groupby("ds").agg(
    yhat=("yhat", "sum"),
    transactions=("transactions", "sum"),
).reset_index().sort_values("ds")

# Scenario multipliers
shock_mult = demand_shock / 100.0
best_mult = 1 + shock_mult + 0.10  # +10% above shock
base_mult = 1 + shock_mult
worst_mult = 1 + shock_mult - 0.10  # -10% below shock

daily["best"] = daily["yhat"] * best_mult
daily["base"] = daily["yhat"] * base_mult
daily["worst"] = daily["yhat"] * worst_mult

daily["best_txn"] = daily["transactions"] * best_mult
daily["base_txn"] = daily["transactions"] * base_mult
daily["worst_txn"] = daily["transactions"] * worst_mult

# ─── Chart ──────────────────────────────────────────────────────────────────
with chart_col:
    st.markdown(
        '<div class="section-header" style="font-size:1.1rem; margin-top:0;">📈 Scenario Revenue Projection</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    # Shaded area between best and worst
    fig.add_trace(go.Scatter(
        x=pd.concat([daily["ds"], daily["ds"][::-1]]),
        y=pd.concat([daily["best"], daily["worst"][::-1]]),
        fill="toself",
        fillcolor="rgba(56,139,253,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Scenario Range",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Best case
    fig.add_trace(go.Scatter(
        x=daily["ds"], y=daily["best"], mode="lines",
        name=f"Best Case (+{shock_mult*100+10:.0f}%)",
        line=dict(color=PAL["best"], width=2, dash="dot"),
        hovertemplate="<b>%{x|%b %d}</b><br>$%{y:,.0f}<extra>Best</extra>",
    ))

    # Base case
    fig.add_trace(go.Scatter(
        x=daily["ds"], y=daily["base"], mode="lines",
        name=f"Base Case ({shock_mult*100:+.0f}%)",
        line=dict(color=PAL["base"], width=3),
        hovertemplate="<b>%{x|%b %d}</b><br>$%{y:,.0f}<extra>Base</extra>",
    ))

    # Worst case
    fig.add_trace(go.Scatter(
        x=daily["ds"], y=daily["worst"], mode="lines",
        name=f"Worst Case ({shock_mult*100-10:.0f}%)",
        line=dict(color=PAL["worst"], width=2, dash="dot"),
        hovertemplate="<b>%{x|%b %d}</b><br>$%{y:,.0f}<extra>Worst</extra>",
    ))

    # Original baseline (dashed gray)
    fig.add_trace(go.Scatter(
        x=daily["ds"], y=daily["yhat"], mode="lines",
        name="Original Forecast",
        line=dict(color="#484f58", width=1.5, dash="dash"),
        hovertemplate="<b>%{x|%b %d}</b><br>$%{y:,.0f}<extra>Original</extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PAL["bg"], plot_bgcolor=PAL["bg"],
        height=420, margin=dict(l=10, r=10, t=20, b=30),
        font=dict(family="Inter", color=PAL["text"]),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center", font=dict(size=10)),
        xaxis=dict(showgrid=True, gridcolor=PAL["grid"], title=""),
        yaxis=dict(showgrid=True, gridcolor=PAL["grid"], title="", tickprefix="$"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════
# P&L Impact Table
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div class="section-header">💰 P&L Impact Analysis</div>',
    unsafe_allow_html=True,
)

def compute_pnl(total_rev: float, total_txn: float, label: str) -> dict:
    """Compute P&L for a scenario."""
    units = total_txn  # 1 transaction ≈ 1 unit
    inventory_cost = units * INVENTORY_COST_PER_UNIT
    # Staff: FTE needed = peak units per hour / capacity
    avg_daily_units = units / max(horizon_days, 1)
    avg_hourly_units = avg_daily_units / HOURS_PER_DAY
    ftes_needed = max(np.ceil(avg_hourly_units / UNITS_PER_FTE_PER_HOUR), 1)
    staff_cost = ftes_needed * STAFF_COST_PER_HOUR * HOURS_PER_DAY * horizon_days
    net_margin = total_rev - inventory_cost - staff_cost
    return {
        "Scenario": label,
        "Total Revenue": total_rev,
        "Inventory Cost": inventory_cost,
        "Staff Cost": staff_cost,
        "Net Margin": net_margin,
        "FTEs": ftes_needed,
    }

original_rev = daily["yhat"].sum()
original_txn = daily["transactions"].sum()

pnl_rows = [
    compute_pnl(daily["best"].sum(), daily["best_txn"].sum(), "🟢 Best Case"),
    compute_pnl(daily["base"].sum(), daily["base_txn"].sum(), "🔵 Base Case"),
    compute_pnl(daily["worst"].sum(), daily["worst_txn"].sum(), "🔴 Worst Case"),
]

# Add vs base column
base_rev = pnl_rows[1]["Total Revenue"]
for row in pnl_rows:
    diff = row["Total Revenue"] - base_rev
    row["Vs Base"] = diff

# Build HTML table
html = '<table class="pnl-table"><thead><tr>'
html += "<th>Scenario</th><th>Total Revenue</th><th>Vs Base</th>"
html += "<th>Inventory Cost</th><th>Staff Cost</th><th>FTEs</th><th>Net Margin</th>"
html += "</tr></thead><tbody>"

for i, row in enumerate(pnl_rows):
    row_class = ""
    if i == 0:
        row_class = "pnl-best-row"
    elif i == 2:
        row_class = "pnl-worst-row"

    # Color code values
    vs_base_cls = "pnl-positive" if row["Vs Base"] > 0 else ("pnl-negative" if row["Vs Base"] < 0 else "pnl-neutral")
    margin_cls = "pnl-positive" if row["Net Margin"] > 0 else "pnl-negative"

    html += f'<tr class="{row_class}">'
    html += f'<td>{row["Scenario"]}</td>'
    html += f'<td style="font-weight:700;">${row["Total Revenue"]:,.0f}</td>'
    html += f'<td class="{vs_base_cls}">${row["Vs Base"]:+,.0f}</td>'
    html += f'<td style="color:#f85149;">${row["Inventory Cost"]:,.0f}</td>'
    html += f'<td style="color:#d29922;">${row["Staff Cost"]:,.0f}</td>'
    html += f'<td>{row["FTEs"]:.0f}</td>'
    html += f'<td class="{margin_cls}" style="font-weight:700;">${row["Net Margin"]:,.0f}</td>'
    html += "</tr>"

html += "</tbody></table>"
st.markdown(html, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Export CSV ─────────────────────────────────────────────────────────────
export_rows = []
for scenario_name, col_rev, col_txn in [
    ("Best Case", "best", "best_txn"),
    ("Base Case", "base", "base_txn"),
    ("Worst Case", "worst", "worst_txn"),
]:
    for _, row in daily.iterrows():
        units = row[col_txn]
        rev = row[col_rev]
        inv_cost = units * INVENTORY_COST_PER_UNIT
        hourly_units = units / HOURS_PER_DAY
        ftes = max(np.ceil(hourly_units / UNITS_PER_FTE_PER_HOUR), 1)
        staff = ftes * STAFF_COST_PER_HOUR * HOURS_PER_DAY
        export_rows.append({
            "Date": row["ds"].strftime("%Y-%m-%d"),
            "Scenario": scenario_name,
            "Store": scenario_store,
            "Revenue": round(rev, 2),
            "Transactions": round(units, 0),
            "Inventory Cost": round(inv_cost, 2),
            "Staff Cost": round(staff, 2),
            "Net Margin": round(rev - inv_cost - staff, 2),
            "FTEs": ftes,
            "Demand Shock (%)": demand_shock,
            "Horizon": horizon_label,
        })

export_df = pd.DataFrame(export_rows)
csv_buffer = io.StringIO()
export_df.to_csv(csv_buffer, index=False)

c_dl1, c_dl2, c_dl3 = st.columns([2, 2, 6])
with c_dl1:
    st.download_button(
        label="📥 Export Scenario CSV",
        data=csv_buffer.getvalue(),
        file_name=f"scenario_plan_{horizon_label.replace(' ', '_').lower()}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("<br>", unsafe_allow_html=True)
