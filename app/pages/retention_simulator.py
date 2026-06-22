"""
app/pages/3_retention_simulator.py
------------------------------------
Page 3: Retention Simulator — cost/threshold sliders, ROI calculator,
expected net value curve.
"""

import os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="Simulator · Churn Detective", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;700&display=swap');
:root { --navy-950:#080d1a; --navy-800:#111e35; --navy-700:#1a2d4e; --navy-600:#243d66;
        --teal-400:#2dd4bf; --teal-300:#5eead4; --amber-300:#fcd34d;
        --rose-400:#fb7185; --slate-300:#cbd5e1; --slate-400:#94a3b8;
        --slate-500:#64748b; --white:#f8fafc; }
html, body, [data-testid="stAppViewContainer"] {
    background-color:var(--navy-950) !important; color:var(--slate-300) !important;
    font-family:'DM Sans',sans-serif !important; }
h1,h2,h3 { font-family:'DM Sans',sans-serif !important; color:var(--white) !important; }
p,li     { font-family:'DM Sans',sans-serif !important; color:var(--slate-300) !important; }
[data-testid="metric-container"] { background:var(--navy-800) !important;
    border:1px solid var(--navy-600) !important; border-radius:12px !important;
    padding:1.25rem 1.5rem !important; }
[data-testid="stMetricValue"] { font-family:'DM Mono',monospace !important;
    font-size:1.8rem !important; color:var(--teal-300) !important; }
[data-testid="stMetricLabel"] { font-size:0.78rem !important; text-transform:uppercase !important;
    letter-spacing:0.08em !important; color:var(--slate-400) !important; }
.section-label { font-family:'DM Mono',monospace; font-size:0.7rem; text-transform:uppercase;
    letter-spacing:0.12em; color:var(--slate-500); margin-bottom:0.5rem; }
</style>
""", unsafe_allow_html=True)

PLOT_TEMPLATE = dict(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111e35",
    font=dict(family="DM Sans", color="#cbd5e1", size=12),
    xaxis=dict(gridcolor="#1a2d4e", linecolor="#1a2d4e"),
    yaxis=dict(gridcolor="#1a2d4e", linecolor="#1a2d4e"),
))
TEAL  = "#2dd4bf"
AMBER = "#fbbf24"
ROSE  = "#fb7185"

df        = st.session_state.get("df", pd.DataFrame())
is_demo   = st.session_state.get("is_demo", True)

st.markdown("## 🎯 Retention Campaign Simulator")
st.markdown('<p style="color:#94a3b8; margin-top:-0.5rem;">Tune business parameters and find the revenue-optimal campaign design.</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar-style parameter panel ────────────────────────────────────────

param_col, result_col = st.columns([1, 2])

with param_col:
    st.markdown("### Campaign Parameters")
    st.markdown('<div class="section-label">Model threshold</div>', unsafe_allow_html=True)
    threshold = st.slider("Churn prob threshold", 0.10, 0.90,
                          st.session_state.get("global_threshold", 0.50), 0.01,
                          label_visibility="collapsed")
    st.session_state["global_threshold"] = threshold

    st.markdown('<div class="section-label" style="margin-top:1rem;">Offer cost per customer ($)</div>', unsafe_allow_html=True)
    offer_cost = st.slider("Offer cost", 5.0, 60.0, 15.0, 1.0, label_visibility="collapsed")

    st.markdown('<div class="section-label" style="margin-top:1rem;">Retention success rate (%)</div>', unsafe_allow_html=True)
    success_rate_pct = st.slider("Success rate", 5, 60, 30, 1, label_visibility="collapsed", format="%d%%")
    success_rate = success_rate_pct / 100

    st.markdown('<div class="section-label" style="margin-top:1rem;">Avg customer lifetime (months)</div>', unsafe_allow_html=True)
    lifetime_mo = st.slider("Lifetime months", 6, 48, 24, 1, label_visibility="collapsed")

    st.markdown('<div class="section-label" style="margin-top:1rem;">Avg monthly revenue ($)</div>', unsafe_allow_html=True)
    avg_rev = st.slider("Avg revenue", 40.0, 120.0, 69.84, 0.50, label_visibility="collapsed")

    clv = avg_rev * lifetime_mo

    st.markdown(f"""
    <div style="background:#111e35; border:1px solid #1a2d4e; border-radius:8px;
                padding:1rem; margin-top:1rem; font-family:DM Mono,monospace; font-size:0.8rem;">
        <div style="color:#64748b; font-size:0.7rem; text-transform:uppercase;
                    letter-spacing:0.1em; margin-bottom:0.5rem;">Derived CLV</div>
        <div style="color:#5eead4; font-size:1.4rem;">${clv:,.0f}</div>
        <div style="color:#64748b; font-size:0.75rem;">{avg_rev:.2f} × {lifetime_mo} months</div>
    </div>
    """, unsafe_allow_html=True)

with result_col:
    st.markdown("### Campaign Results")

    if "churn_proba" not in df.columns or df.empty:
        import streamlit as st
        st.error("Run notebook 02_baseline_model.ipynb to generate churn probabilities.")
        st.stop()
        
    proba  = df["churn_proba"].values
    y_true = df["churned"].values if "churned" in df.columns else (proba > 0.5).astype(int)
    thresholds = np.linspace(0.10, 0.90, 161)

    # Compute sweep
    sweep_rows = []
    for t in thresholds:
        flagged    = proba >= t
        tp         = ((flagged) & (y_true == 1)).sum()
        fp         = ((flagged) & (y_true == 0)).sum()
        n_contact  = int(tp + fp)
        rev_saved  = tp * success_rate * clv
        camp_cost  = n_contact * offer_cost
        net_val    = rev_saved - camp_cost
        roi        = (net_val / camp_cost * 100) if camp_cost > 0 else 0
        sweep_rows.append({
            "threshold":   round(t, 3),
            "n_contacted": n_contact,
            "net_value":   round(net_val, 0),
            "roi_pct":     round(roi, 1),
            "rev_saved":   round(rev_saved, 0),
            "camp_cost":   round(camp_cost, 0),
        })
    sweep_df = pd.DataFrame(sweep_rows)

    # Current threshold results
    cur = sweep_df[sweep_df["threshold"] == min(sweep_rows, key=lambda r: abs(r["threshold"] - threshold))["threshold"]].iloc[0]
    opt_row = sweep_df.loc[sweep_df["net_value"].idxmax()]

    m1, m2, m3 = st.columns(3)
    m1.metric("Customers Contacted", f"{int(cur['n_contacted']):,}", delta=f"@ p≥{threshold:.2f}")
    m2.metric("Expected Net Value",  f"${cur['net_value']:,.0f}")
    m3.metric("ROI",                 f"{cur['roi_pct']:.0f}%")

    m4, m5, m6 = st.columns(3)
    m4.metric("Revenue Saved",       f"${cur['rev_saved']:,.0f}")
    m5.metric("Campaign Cost",       f"${cur['camp_cost']:,.0f}", delta_color="inverse")
    m6.metric("Optimal Threshold",   f"{opt_row['threshold']:.2f}",
              delta=f"${opt_row['net_value']:,.0f} max net value", delta_color="normal")

    # Net value curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep_df["threshold"], y=sweep_df["net_value"],
        mode="lines", name="Net Value",
        line=dict(color=TEAL, width=2.5),
        fill="tozeroy", fillcolor="rgba(45,212,191,0.08)",
        hovertemplate="Threshold: %{x:.2f}<br>Net value: $%{y:,.0f}<extra></extra>",
    ))
    # Current threshold marker
    fig.add_vline(x=threshold, line_dash="dot", line_color=AMBER,
                  annotation_text=f"  Current: {threshold:.2f}",
                  annotation_font=dict(color=AMBER, size=11))
    # Optimal threshold marker
    fig.add_vline(x=opt_row["threshold"], line_dash="dash", line_color=ROSE,
                  annotation_text=f"  Optimal: {opt_row['threshold']:.2f}",
                  annotation_font=dict(color=ROSE, size=11))
    fig.add_hline(y=0, line_color="#475569", line_width=1)
    fig.update_layout(
        **PLOT_TEMPLATE["layout"].to_plotly_json(),
        height=320,
        xaxis_title="Decision Threshold",
        yaxis_title="Expected Net Value ($)",
        margin=dict(t=20, b=40, l=60, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Peak net value ${opt_row['net_value']:,.0f} at threshold {opt_row['threshold']:.2f} "
               f"— contacting {int(opt_row['n_contacted']):,} customers.")

st.markdown("---")
st.markdown("""
<div style="background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.3);
            border-radius:8px; padding:0.875rem 1.25rem; font-size:0.85rem; color:#fcd34d;">
    <strong>Assumptions:</strong> CLV is simplified (monthly revenue × lifetime months, undiscounted).
    Retention success rate is a flat probability — the uplift model refines this per customer.
    Offer cost covers agent time + offer value combined. All figures are estimates pending A/B validation.
</div>
""", unsafe_allow_html=True)
