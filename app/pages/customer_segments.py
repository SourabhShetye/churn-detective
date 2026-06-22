"""
app/pages/2_customer_segments.py
----------------------------------
Page 2: Customer Segments — churner personas, scatter, deep-dives.
"""

import os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="Segments · Churn Detective", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;700&display=swap');
:root { --navy-950:#080d1a; --navy-900:#0d1526; --navy-800:#111e35; --navy-700:#1a2d4e;
        --navy-600:#243d66; --teal-400:#2dd4bf; --teal-300:#5eead4; --amber-300:#fcd34d;
        --rose-400:#fb7185; --slate-300:#cbd5e1; --slate-400:#94a3b8; --slate-500:#64748b;
        --white:#f8fafc; }
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--navy-950) !important; color: var(--slate-300) !important;
    font-family: 'DM Sans', sans-serif !important; }
h1,h2,h3 { font-family:'DM Sans',sans-serif !important; color:var(--white) !important; }
p,li { font-family:'DM Sans',sans-serif !important; color:var(--slate-300) !important; }
.persona-card { background:var(--navy-800); border:1px solid var(--navy-600);
    border-radius:12px; padding:1.25rem 1.5rem; height:100%; }
.persona-card h4 { font-size:1rem; color:var(--white); margin:0.5rem 0 0.4rem 0; }
.persona-card p  { font-size:0.85rem; color:var(--slate-400); margin:0.25rem 0; }
.stat-pill { display:inline-block; background:var(--navy-700); color:var(--teal-300);
    font-family:'DM Mono',monospace; font-size:0.75rem; padding:0.2rem 0.65rem;
    border-radius:999px; margin:0.15rem 0.2rem; }
.section-label { font-family:'DM Mono',monospace; font-size:0.7rem; text-transform:uppercase;
    letter-spacing:0.12em; color:var(--slate-500); margin-bottom:0.5rem; }
[data-testid="metric-container"] { background:var(--navy-800) !important;
    border:1px solid var(--navy-600) !important; border-radius:12px !important;
    padding:1rem 1.25rem !important; }
[data-testid="stMetricValue"] { font-family:'DM Mono',monospace !important;
    font-size:1.6rem !important; color:var(--teal-300) !important; }
[data-testid="stMetricLabel"] { font-size:0.75rem !important; text-transform:uppercase !important;
    letter-spacing:0.08em !important; color:var(--slate-400) !important; }
</style>
""", unsafe_allow_html=True)

PLOT_TEMPLATE = dict(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111e35",
    font=dict(family="DM Sans", color="#cbd5e1", size=12),
    xaxis=dict(gridcolor="#1a2d4e", linecolor="#1a2d4e"),
    yaxis=dict(gridcolor="#1a2d4e", linecolor="#1a2d4e"),
))
SEGMENT_COLORS = {
    "💸 Price-Sensitive Shoppers":       "#2dd4bf",
    "😤 Frustrated Early Adopters":      "#fb7185",
    "🌙 Quietly Disengaging Veterans":   "#fbbf24",
    "⚠️ At-Risk Budget Customers":       "#818cf8",
}
DEFAULT_COLOR = "#94a3b8"

df        = st.session_state.get("df", pd.DataFrame())
threshold = st.session_state.get("global_threshold", 0.50)
is_demo   = st.session_state.get("is_demo", True)

st.markdown("## 👥 Customer Segments")
st.markdown('<p style="color:#94a3b8; margin-top:-0.5rem;">Not all churners are alike — know who you\'re targeting before you spend.</p>', unsafe_allow_html=True)
st.markdown("---")

if df.empty or "segment_label" not in df.columns:
    st.warning("Segment labels not yet available. Run notebook 03_segmentation.ipynb first.")
    st.stop()

# Show ALL historical churners for persona cards (full picture of the problem)
all_churners = df[df["churned"] == 1].copy() if "churned" in df.columns else df.copy()

# Flagged churners = above current threshold (for threshold-responsive stats)
if "churned" in df.columns and "churn_proba" in df.columns:
    churners = df[(df["churned"] == 1) & (df["churn_proba"] >= threshold)].copy()
else:
    churners = all_churners.copy()

# ── Persona summary cards ─────────────────────────────────────────────────

st.markdown("### Churner Personas")
st.markdown(
    f'<p style="color:#94a3b8; font-size:0.875rem;">Showing <strong style="color:#5eead4;">'
    f'{len(churners):,} churners</strong> flagged at threshold p ≥ {threshold:.2f} '
    f'(out of {len(all_churners):,} total churners in dataset).</p>',
    unsafe_allow_html=True
)

PERSONA_META = {
    "💸 Price-Sensitive Shoppers": {
        "icon": "💸",
        "play": "15–20% discount for 1-year contract upgrade",
        "color": "#2dd4bf",
    },
    "😤 Frustrated Early Adopters": {
        "icon": "😤",
        "play": "Proactive support outreach + free security tier",
        "color": "#fb7185",
    },
    "🌙 Quietly Disengaging Veterans": {
        "icon": "🌙",
        "play": "Personal account call + exclusive loyalty plan",
        "color": "#fbbf24",
    },
    "⚠️ At-Risk Budget Customers": {
        "icon": "⚠️",
        "play": "Flexible payment plan or bill credit",
        "color": "#818cf8",
    },
}

# Get segment counts from flagged churners
segs = churners["segment_label"].value_counts() if not churners.empty else pd.Series(dtype=int)

# Only show personas that exist in the data
existing_personas = {
    name: meta for name, meta in PERSONA_META.items()
    if name in all_churners["segment_label"].values
}

# If no personas found, fall back to all
if not existing_personas:
    existing_personas = PERSONA_META

cols = st.columns(max(len(existing_personas), 1))

for col, (seg_name, meta) in zip(cols, existing_personas.items()):
    n   = int(segs.get(seg_name, 0))
    total_in_seg = int(all_churners["segment_label"].eq(seg_name).sum())
    pct = n / len(churners) * 100 if len(churners) > 0 else 0
    avg_rev = churners.loc[churners["segment_label"] == seg_name, "monthly_charges"].mean() \
              if n > 0 else 0

    col.markdown(f"""
    <div class="persona-card" style="border-top: 3px solid {meta['color']};">
        <div style="font-size:1.8rem;">{meta['icon']}</div>
        <h4>{seg_name}</h4>
        <span class="stat-pill">{n:,} flagged</span>
        <span class="stat-pill">{total_in_seg:,} total</span>
        <span class="stat-pill">{pct:.0f}% of flagged</span>
        <p style="margin-top:0.6rem; color:#94a3b8; font-size:0.82rem;">
            Avg revenue (flagged): <strong style="color:{meta['color']};">${avg_rev:.0f}/mo</strong>
        </p>
        <p style="margin-top:0.4rem; font-size:0.8rem; color:#64748b; font-style:italic;">
            ▶ {meta['play']}
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Segment scatter ───────────────────────────────────────────────────────

st.markdown("### Segment Map — Tenure vs Monthly Charges")
st.markdown('<p style="font-size:0.875rem; color:#94a3b8;">Each dot is a churned customer. '
            'Clusters reveal the distinct cost-and-commitment profiles of each persona.</p>',
            unsafe_allow_html=True)

scatter_df = churners.copy()
scatter_df["color"] = scatter_df["segment_label"].map(SEGMENT_COLORS).fillna(DEFAULT_COLOR)
scatter_df["size"]  = scatter_df.get("churn_proba", pd.Series(0.5, index=scatter_df.index)) * 14 + 4

fig = go.Figure()
for seg, color in SEGMENT_COLORS.items():
    sub = scatter_df[scatter_df["segment_label"] == seg]
    if sub.empty:
        continue
    fig.add_trace(go.Scatter(
        x=sub["tenure_months"],
        y=sub["monthly_charges"],
        mode="markers",
        name=seg,
        marker=dict(
            color=color,
            size=sub["size"] if "size" in sub.columns else 7,
            opacity=0.7,
            line=dict(width=0.5, color="#080d1a"),
        ),
        hovertemplate=(
            "<b>" + seg + "</b><br>"
            "Tenure: %{x} mo<br>"
            "Monthly charges: $%{y:.2f}<br>"
            "<extra></extra>"
        ),
    ))

fig.update_layout(
    **PLOT_TEMPLATE["layout"].to_plotly_json(),
    height=440,
    xaxis_title="Tenure (months)",
    yaxis_title="Monthly Charges ($)",
    legend=dict(
        orientation="h", y=-0.18, x=0,
        font=dict(family="DM Sans", size=11),
    ),
    margin=dict(t=20, b=80, l=60, r=20),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Segment deep-dive ─────────────────────────────────────────────────────

st.markdown("### Segment Deep-Dive")
available_segs = [s for s in PERSONA_META.keys() if s in all_churners["segment_label"].values]
if not available_segs:
    available_segs = list(PERSONA_META.keys())
selected = st.selectbox("Select a persona", available_segs)
sub = churners[churners["segment_label"] == selected]

if sub.empty:
    st.info("No customers in this segment.")
else:
    meta = PERSONA_META[selected]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers",          f"{len(sub):,}")
    m2.metric("Avg Monthly Rev",    f"${sub['monthly_charges'].mean():.0f}")
    m3.metric("Avg Tenure",         f"{sub['tenure_months'].mean():.0f} mo")
    m4.metric("Avg Support Calls",  f"{sub['support_calls_3mo'].mean():.1f}")

    feat_col, dist_col = st.columns([1, 1])

    with feat_col:
        st.markdown(f"""
        <div style="background:#111e35; border:1px solid #1a2d4e; border-left:3px solid {meta['color']};
                    border-radius:10px; padding:1.25rem 1.5rem; margin-top:1rem;">
            <div style="font-family:DM Mono,monospace; font-size:0.7rem; text-transform:uppercase;
                        letter-spacing:0.1em; color:#64748b; margin-bottom:0.5rem;">Retention Play</div>
            <p style="color:#f8fafc; font-size:0.95rem; margin:0; line-height:1.6;">
                {meta['play']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with dist_col:
        if "contract_type" in sub.columns:
            ct_dist = sub["contract_type"].value_counts().reset_index()
            ct_dist.columns = ["Contract", "Count"]
            fig2 = go.Figure(go.Bar(
                x=ct_dist["Contract"], y=ct_dist["Count"],
                marker_color=meta["color"],
                hovertemplate="%{x}: %{y}<extra></extra>",
            ))
            fig2.update_layout(
                **PLOT_TEMPLATE["layout"].to_plotly_json(),
                height=220, showlegend=False,
                title=dict(text="Contract mix", font=dict(size=12, color="#94a3b8")),
                margin=dict(t=40, b=20, l=20, r=20),
            )
            st.plotly_chart(fig2, use_container_width=True)
