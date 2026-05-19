"""
app/pages/4_uplift_targeting.py
---------------------------------
Page 4: Uplift Targeting — Persuadable ranking, Qini curve, action list export.
This is the operational output: the ranked list the retention team actually dials from.
"""

import os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

st.set_page_config(page_title="Uplift Targeting · Churn Detective", layout="wide")

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
.section-label { font-family:'DM Mono',monospace; font-size:0.7rem; text-transform:uppercase;
    letter-spacing:0.12em; color:var(--slate-500); margin-bottom:0.5rem; }
.quadrant-card { background:var(--navy-800); border:1px solid var(--navy-600);
    border-radius:10px; padding:1.1rem 1.25rem; text-align:center; }
.quadrant-card h4 { font-size:0.95rem; color:var(--white); margin:0.4rem 0 0.2rem 0; }
.quadrant-card p  { font-size:0.8rem; color:var(--slate-400); margin:0; }
[data-testid="metric-container"] { background:var(--navy-800) !important;
    border:1px solid var(--navy-600) !important; border-radius:12px !important;
    padding:1rem 1.25rem !important; }
[data-testid="stMetricValue"] { font-family:'DM Mono',monospace !important;
    font-size:1.7rem !important; color:var(--teal-300) !important; }
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
TEAL  = "#2dd4bf"
AMBER = "#fbbf24"
ROSE  = "#fb7185"
INDIGO = "#818cf8"

df        = st.session_state.get("df", pd.DataFrame())
threshold = st.session_state.get("global_threshold", 0.50)
is_demo   = st.session_state.get("is_demo", True)

st.markdown("## 🚀 Uplift Targeting")
st.markdown('<p style="color:#94a3b8; margin-top:-0.5rem;">Who will actually be <em>saved</em> by the offer — not just who is likely to leave.</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Four-quadrant explainer ───────────────────────────────────────────────

st.markdown("### The Four Uplift Quadrants")
st.markdown('<p style="font-size:0.875rem; color:#94a3b8; margin-top:-0.3rem;">Standard churn models only split on the vertical axis. Uplift modeling adds the horizontal dimension — who actually responds to the offer.</p>', unsafe_allow_html=True)

q1, q2, q3, q4 = st.columns(4)
quadrants = [
    ("🎯 Persuadables",   TEAL,   "High churn risk + responds to offer",   "← PRIMARY TARGET", True),
    ("💀 Lost Causes",    ROSE,   "High churn risk + offer won't help",    "Save budget — let go", False),
    ("😴 Sure Savers",    AMBER,  "Low churn risk + positive ITE",         "Will stay anyway — don't waste $", False),
    ("🐕 Sleeping Dogs",  INDIGO, "Low churn risk + negative ITE",         "Don't contact — may trigger churn", False),
]
for col, (name, color, desc, note, primary) in zip([q1, q2, q3, q4], quadrants):
    border = f"border-top: 3px solid {color};" if primary else f"border-top: 2px solid {color}40;"
    col.markdown(f"""
    <div class="quadrant-card" style="{border}">
        <h4>{name}</h4>
        <p>{desc}</p>
        <p style="margin-top:0.5rem; color:{color}; font-size:0.78rem; font-style:italic;">{note}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Quadrant breakdown ────────────────────────────────────────────────────

if "churn_proba" in df.columns and "uplift_score" in df.columns:
    ite_threshold = df["uplift_score"].median()
    df["uplift_quadrant"] = np.select(
        [
            (df["churn_proba"] >= threshold) & (df["uplift_score"] > ite_threshold),
            (df["churn_proba"] >= threshold) & (df["uplift_score"] <= ite_threshold),
            (df["churn_proba"] <  threshold) & (df["uplift_score"] > ite_threshold),
            (df["churn_proba"] <  threshold) & (df["uplift_score"] <= ite_threshold),
        ],
        ["Persuadable", "Lost Cause", "Sure Saver", "Sleeping Dog"],
        default="Unknown",
    )

    counts = df["uplift_quadrant"].value_counts()
    st.markdown("### Quadrant Distribution")
    m1, m2, m3, m4 = st.columns(4)
    color_map = {"Persuadable": TEAL, "Lost Cause": ROSE, "Sure Saver": AMBER, "Sleeping Dog": INDIGO}
    metrics = [m1, m2, m3, m4]
    for col, (quad, color) in zip(metrics, color_map.items()):
        n   = counts.get(quad, 0)
        pct = n / len(df) * 100
        col.metric(quad, f"{n:,}", delta=f"{pct:.1f}% of base")

    st.markdown("---")

    # ── Uplift decile chart ───────────────────────────────────────────────

    st.markdown("### Uplift Gain by Decile")
    st.markdown('<p style="font-size:0.875rem; color:#94a3b8;">Targeting the top 20% by uplift score should capture 40%+ of recoverable churn. If it doesn\'t, review the model calibration.</p>', unsafe_allow_html=True)

    df_sorted = df.sort_values("uplift_score", ascending=False).reset_index(drop=True)
    df_sorted["decile"] = pd.qcut(df_sorted.index, 10, labels=False) + 1
    decile_grp = df_sorted.groupby("decile").agg(
        n=("uplift_score", "count"),
        avg_uplift=("uplift_score", "mean"),
        avg_churn_proba=("churn_proba", "mean"),
    ).reset_index()
    decile_grp["pct_of_total_uplift"] = (
        decile_grp["avg_uplift"] / decile_grp["avg_uplift"].sum() * 100
    ).round(1)
    decile_grp["cumulative_pct"] = decile_grp["pct_of_total_uplift"].cumsum().round(1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=decile_grp["decile"],
        y=decile_grp["pct_of_total_uplift"],
        name="Uplift share per decile",
        marker_color=[TEAL if d <= 3 else AMBER if d <= 6 else ROSE for d in decile_grp["decile"]],
        hovertemplate="Decile %{x}<br>Uplift share: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=decile_grp["decile"],
        y=decile_grp["cumulative_pct"],
        name="Cumulative %",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color="#f8fafc", width=2, dash="dot"),
        marker=dict(size=6, color="#f8fafc"),
        hovertemplate="Decile %{x}<br>Cumulative: %{y:.1f}%<extra></extra>",
    ))
    base_layout = PLOT_TEMPLATE["layout"].to_plotly_json()
    base_layout.pop("yaxis", None)
    fig.update_layout(
        **base_layout,
        height=360,
        xaxis_title="Score Decile (1 = highest uplift)",
        yaxis=dict(title="% of Total Uplift", gridcolor="#1a2d4e"),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                    range=[0, 110], showgrid=False),
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=20, b=40, l=60, r=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Ranked action table ───────────────────────────────────────────────

    st.markdown("### Ranked Action List — Persuadables Only")

    persuadables = df[df["uplift_quadrant"] == "Persuadable"].copy()
    persuadables = persuadables.sort_values("uplift_score", ascending=False).reset_index(drop=True)
    persuadables["priority_rank"] = persuadables.index + 1

    # Segment → action mapping
    def _action(seg):
        if "Price" in str(seg):    return "Offer: Contract upgrade + 15% discount"
        if "Frustrated" in str(seg): return "Offer: Free tech support tier upgrade"
        if "Veteran" in str(seg):  return "Offer: Loyalty plan + personal call"
        return "Offer: General retention bundle"

    if "segment_label" in persuadables.columns:
        persuadables["recommended_action"] = persuadables["segment_label"].apply(_action)
    else:
        persuadables["recommended_action"] = "Offer: General retention bundle"

    display_cols = ["priority_rank", "customer_id", "monthly_charges",
                    "churn_proba", "uplift_score",
                    "segment_label", "recommended_action"]
    display_cols = [c for c in display_cols if c in persuadables.columns]

    # Filter controls
    filter_col, dl_col = st.columns([3, 1])
    with filter_col:
        seg_options = ["All"] + list(persuadables["segment_label"].unique()) \
                      if "segment_label" in persuadables.columns else ["All"]
        selected_seg = st.selectbox("Filter by segment", seg_options)
    with dl_col:
        st.markdown("<br>", unsafe_allow_html=True)
        csv = persuadables[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download target list",
            data=csv,
            file_name="persuadables_action_list.csv",
            mime="text/csv",
        )

    if selected_seg != "All" and "segment_label" in persuadables.columns:
        persuadables = persuadables[persuadables["segment_label"] == selected_seg]

    st.dataframe(
        persuadables[display_cols].head(200).style
            .format({
                "monthly_charges": "${:.2f}",
                "churn_proba":     "{:.3f}",
                "uplift_score":    "{:.3f}",
            })
            .background_gradient(subset=["uplift_score"], cmap="YlGn"),
        use_container_width=True,
        height=380,
    )
    st.caption(f"Showing top {min(200, len(persuadables))} of {len(persuadables):,} persuadable customers. "
               f"Download for full list.")

else:
    st.info("""
    Uplift scores not yet available.

    Run **notebook 04_uplift_modeling.ipynb** to generate `churn_proba` and `uplift_score` columns,
    then save the enriched dataset to `data/processed/test.parquet`.

    """)

st.markdown("---")
st.markdown("""
<div style="background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.3);
            border-radius:8px; padding:0.875rem 1.25rem; font-size:0.85rem; color:#fcd34d;">
    <strong>Uplift model assumptions:</strong>
    Treatment indicator = <code>plan_changes_6mo > 0</code> (proxy — not a true A/B test).
    S-Learner ITE used as primary ranking signal; T-Learner used for validation.
    Positive ITE = offer reduces churn probability. Validate Qini curve AUUC > 0.3
    before trusting the ranking. Run an A/B test in the first campaign wave to calibrate.
</div>
""", unsafe_allow_html=True)
