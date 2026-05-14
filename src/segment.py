"""
src/segment.py
--------------
KMeans-based churner segmentation and persona labeling.

Design principles:
  - Segmentation runs ONLY on churned==1 customers (we're segmenting the
    problem population, not the whole base).
  - Persona labels are derived deterministically from cluster centroid
    signatures — no magic strings hard-coded to cluster indices (which
    would break if k or data changes).
  - All plotting helpers return matplotlib Figure objects so they can be
    saved to outputs/figures/ from notebooks OR rendered in Streamlit.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# 1.  Feature set for clustering (must match features.py KMEANS_FEATURES)
# ---------------------------------------------------------------------------

KMEANS_FEATURES = [
    "monthly_charges",
    "tenure_months",
    "support_calls_3mo",
    "avg_data_gb_3mo",
    "late_payments_6mo",
    "service_frustration_index",
    "charges_per_tenure_month",
]

# Human-readable axis labels for plots
FEATURE_LABELS = {
    "monthly_charges":           "Monthly Charges ($)",
    "tenure_months":             "Tenure (months)",
    "support_calls_3mo":         "Support Calls (3 mo)",
    "avg_data_gb_3mo":           "Avg Data Usage (GB)",
    "late_payments_6mo":         "Late Payments (6 mo)",
    "service_frustration_index": "Frustration Index",
    "charges_per_tenure_month":  "Charges / Tenure Month",
}


# ---------------------------------------------------------------------------
# 2.  K selection — elbow + silhouette
# ---------------------------------------------------------------------------

def select_k(
    X_scaled: np.ndarray,
    k_range: range = range(2, 8),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit KMeans for each k in k_range and return inertia + silhouette scores.

    Use the returned DataFrame to plot an elbow curve and choose k:
      - Elbow: where marginal inertia drop flattens.
      - Silhouette: higher = better separated clusters. Aim for > 0.3.

    Returns
    -------
    DataFrame: k | inertia | silhouette_score
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if k > 1 else np.nan
        rows.append({
            "k":                 k,
            "inertia":           round(km.inertia_, 2),
            "silhouette_score":  round(sil, 4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3.  Fit final KMeans model
# ---------------------------------------------------------------------------

def fit_kmeans(
    df_churners: pd.DataFrame,
    k: int,
    features: list[str] = KMEANS_FEATURES,
    random_state: int = 42,
) -> tuple[KMeans, StandardScaler, np.ndarray]:
    """
    Fit KMeans on the churner subset.

    Parameters
    ----------
    df_churners : DataFrame containing only churned==1 rows,
                  with engineered features already added.
    k           : Number of clusters chosen from select_k() analysis.
    features    : Feature list (default = KMEANS_FEATURES).

    Returns
    -------
    km      : Fitted KMeans instance.
    scaler  : Fitted StandardScaler (needed to transform new data).
    labels  : Cluster label array aligned with df_churners index.
    """
    missing = [f for f in features if f not in df_churners.columns]
    if missing:
        raise ValueError(f"Clustering features missing from DataFrame: {missing}")

    X = df_churners[features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=random_state, n_init=15, max_iter=500)
    labels = km.fit_predict(X_scaled)

    return km, scaler, labels


# ---------------------------------------------------------------------------
# 4.  Centroid profile table
# ---------------------------------------------------------------------------
def centroid_profile(
    df_churners: pd.DataFrame,
    labels: np.ndarray,
    features: list[str] = KMEANS_FEATURES,
) -> pd.DataFrame:
    df = df_churners[features + ["monthly_charges"]].copy()
    df = df.loc[:, ~df.columns.duplicated()]
    df["cluster_id"] = labels

    rows = []
    for cid in sorted(df["cluster_id"].unique()):
        sub = df[df["cluster_id"] == cid]
        row = {"cluster_id": int(cid),
               "n": len(sub),
               "revenue_at_risk": round(sub["monthly_charges"].sum(), 2),
               "pct_of_churners": 0.0}
        for f in features:
            if not isinstance(sub[f], pd.Series):
                print(f"WARNING: {f} is {type(sub[f])}, shape={sub[f].shape}")
        for f in features:
            col = sub[f] if isinstance(sub[f], pd.Series) else sub[f].iloc[:, 0]
            row[f] = round(float(pd.to_numeric(col, errors='coerce').mean()), 2)
        rows.append(row)

    agg = pd.DataFrame(rows)
    agg["pct_of_churners"] = (agg["n"] / agg["n"].sum() * 100).round(1)
    return agg


# ---------------------------------------------------------------------------
# 5.  Persona labeling — deterministic from centroid signatures
# ---------------------------------------------------------------------------

# Persona definitions: each is a dict of feature → (direction, weight)
# direction: 'high' or 'low' relative to the median centroid value for that feature.
# The persona whose signature best matches a centroid wins.

_PERSONA_SIGNATURES: list[dict] = [
    {
        "name":        "💸 Price-Sensitive Shoppers",
        "description": (
            "Short-tenure, month-to-month customers on moderate plans. "
            "They haven't committed and are actively comparing alternatives. "
            "A targeted discount or contract-upgrade incentive is likely to retain them."
        ),
        "retention_play": "Offer 15–20% discount for switching to a 1-year contract.",
        "signals": {
            "monthly_charges":    "high",
            "tenure_months":      "low",
            "support_calls_3mo":  "low",
            "late_payments_6mo":  "low",
        },
    },
    {
        "name":        "😤 Frustrated Early Adopters",
        "description": (
            "Newer customers with high support call volumes and service friction. "
            "They signed up with high expectations and experienced repeated issues. "
            "Price is NOT the primary driver — fixing service perception is."
        ),
        "retention_play": "Proactive tech support outreach + free security/support tier upgrade.",
        "signals": {
            "support_calls_3mo":         "high",
            "service_frustration_index": "high",
            "tenure_months":             "low",
            "monthly_charges":           "mid",
        },
    },
    {
        "name":        "🌙 Quietly Disengaging Veterans",
        "description": (
            "Long-tenure, high-value customers whose engagement has quietly dropped. "
            "Low support calls (they've stopped trying), potential plan changes. "
            "High CLV — losing one of these hurts most."
        ),
        "retention_play": "Personal outreach from senior account team + exclusive loyalty plan.",
        "signals": {
            "tenure_months":     "high",
            "monthly_charges":   "high",
            "support_calls_3mo": "low",
            "avg_data_gb_3mo":   "low",
        },
    },
    {
        "name":        "⚠️ At-Risk Budget Customers",
        "description": (
            "Lower-spend customers with late payment history, suggesting financial stress. "
            "Likely to churn without any campaign — but also low CLV recovery if retained. "
            "Prioritise only if uplift model confirms persuadability."
        ),
        "retention_play": "Flexible payment plan or temporary bill credit.",
        "signals": {
            "late_payments_6mo": "high",
            "monthly_charges":   "low",
            "avg_data_gb_3mo":   "low",
        },
    },
]


def _score_signature(
    centroid_row: pd.Series,
    signature: dict,
    median_vals: pd.Series,
) -> float:
    """
    Score how well a centroid matches a persona signature.
    Returns a score in [0, 1] — higher = better match.
    """
    signals = signature["signals"]
    hits = 0
    total = len(signals)

    for feature, direction in signals.items():
        if feature not in centroid_row.index:
            total -= 1
            continue
        val    = centroid_row[feature]
        median = median_vals.get(feature, 0)

        if direction == "high" and val > median:
            hits += 1
        elif direction == "low" and val < median:
            hits += 1
        elif direction == "mid":
            # 'mid' = within 25% of median
            if median * 0.75 <= val <= median * 1.25:
                hits += 1

    return hits / total if total > 0 else 0.0


def assign_personas(
    profile_df: pd.DataFrame,
    features: list[str] = KMEANS_FEATURES,
) -> pd.DataFrame:
    """
    Assign a persona label to each cluster based on centroid signature matching.

    Parameters
    ----------
    profile_df : Output of centroid_profile(). Must have cluster_id as a column.

    Returns
    -------
    profile_df with added columns:
        persona_name, persona_description, retention_play, persona_match_score
    """
    feature_cols = [f for f in features if f in profile_df.columns]
    median_vals  = profile_df[feature_cols].median()

    assigned_personas: dict[int, dict] = {}  # cluster_id → persona
    used_personas: set[str] = set()

    # Greedy assignment: sort clusters by highest match score for any persona
    cluster_ids = profile_df["cluster_id"].tolist()

    # Build score matrix
    score_matrix = {}
    for _, row in profile_df.iterrows():
        cid = row["cluster_id"]
        score_matrix[cid] = {}
        for persona in _PERSONA_SIGNATURES:
            score_matrix[cid][persona["name"]] = _score_signature(
                row[feature_cols], persona, median_vals
            )

    # Assign: each cluster gets its best-matching unassigned persona
    for cid in cluster_ids:
        scores = score_matrix[cid]
        # Filter out already-used personas
        available = {k: v for k, v in scores.items() if k not in used_personas}
        if not available:
            # All personas used — assign a generic label
            assigned_personas[cid] = {
                "persona_name":        f"Segment {cid}",
                "persona_description": "Mixed churn profile. Review centroid manually.",
                "retention_play":      "General retention bundle offer.",
                "persona_match_score": 0.0,
            }
        else:
            best_name  = max(available, key=available.get)
            best_score = available[best_name]
            best_def   = next(p for p in _PERSONA_SIGNATURES if p["name"] == best_name)
            assigned_personas[cid] = {
                "persona_name":        best_name,
                "persona_description": best_def["description"],
                "retention_play":      best_def["retention_play"],
                "persona_match_score": round(best_score, 2),
            }
            used_personas.add(best_name)

    # Merge back into profile_df
    persona_df = pd.DataFrame.from_dict(assigned_personas, orient="index")
    persona_df.index.name = "cluster_id"
    persona_df = persona_df.reset_index()

    return profile_df.merge(persona_df, on="cluster_id")


# ---------------------------------------------------------------------------
# 6.  Predict cluster for new / test data
# ---------------------------------------------------------------------------

def predict_segment(
    df: pd.DataFrame,
    km: KMeans,
    scaler: StandardScaler,
    persona_map: pd.DataFrame,
    features: list[str] = KMEANS_FEATURES,
) -> pd.Series:
    """
    Assign cluster labels (and persona names) to new rows.

    Parameters
    ----------
    df          : DataFrame with engineered features.
    km          : Fitted KMeans from fit_kmeans().
    scaler      : Fitted StandardScaler from fit_kmeans().
    persona_map : Output of assign_personas() — maps cluster_id → persona_name.

    Returns
    -------
    Series of persona_name strings aligned with df.index.
    """
    X = df[features].fillna(0).values
    X_scaled = scaler.transform(X)
    cluster_ids = km.predict(X_scaled)

    id_to_persona = dict(
        zip(persona_map["cluster_id"], persona_map["persona_name"])
    )
    return pd.Series(
        [id_to_persona.get(c, f"Segment {c}") for c in cluster_ids],
        index=df.index,
        name="segment_label",
    )


# ---------------------------------------------------------------------------
# 7.  Segment summary for Streamlit persona cards
# ---------------------------------------------------------------------------

def segment_summary(
    df_full: pd.DataFrame,
    segment_col: str = "segment_label",
    monthly_charges_col: str = "monthly_charges",
) -> pd.DataFrame:
    """
    Return a per-segment summary table suitable for Streamlit persona cards.

    Output columns
    --------------
    segment_label, n_customers, pct_of_total,
    avg_monthly_charges, total_monthly_revenue_at_risk,
    avg_tenure_months, avg_support_calls, avg_frustration_index,
    persona_description, retention_play  (joined from _PERSONA_SIGNATURES)
    """
    grp = df_full.groupby(segment_col).agg(
        n_customers=(monthly_charges_col, "count"),
        avg_monthly_charges=(monthly_charges_col, "mean"),
        total_revenue_at_risk=(monthly_charges_col, "sum"),
        avg_tenure_months=("tenure_months", "mean"),
        avg_support_calls=("support_calls_3mo", "mean"),
        avg_frustration_index=("service_frustration_index", "mean"),
    ).round(2).reset_index()

    grp["pct_of_total"] = (
        grp["n_customers"] / grp["n_customers"].sum() * 100
    ).round(1)

    # Join persona descriptions
    persona_lookup = {
        p["name"]: {
            "persona_description": p["description"],
            "retention_play":      p["retention_play"],
        }
        for p in _PERSONA_SIGNATURES
    }
    grp["persona_description"] = grp[segment_col].map(
        lambda x: persona_lookup.get(x, {}).get("persona_description", "—")
    )
    grp["retention_play"] = grp[segment_col].map(
        lambda x: persona_lookup.get(x, {}).get("retention_play", "—")
    )

    return grp.sort_values("total_revenue_at_risk", ascending=False).reset_index(drop=True)
