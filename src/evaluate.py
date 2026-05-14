"""
src/evaluate.py
---------------
Model evaluation, cost-aware threshold tuning, and business value functions.

Design principles:
  - All functions are pure (no I/O side effects).
  - Business parameters are isolated in a single config dataclass so the
    Streamlit dashboard can override them with slider values.
  - Threshold tuning operates over the full [0.05, 0.95] range and returns
    the full curve — the caller decides how to display/use it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# 1.  Business parameters — override in Streamlit via dataclass replace()
# ---------------------------------------------------------------------------

@dataclass
class BusinessParams:
    """
    All monetary/rate assumptions in one place.
    Documented so the CMO can challenge any individual number.

    Assumptions & sources
    ---------------------
    avg_monthly_revenue      : Mean monthly_charges from EDA (~$69.84).
                               Replace with actual ARPU from finance.

    avg_customer_lifetime_mo : Industry median for postpaid telecom ~24 mo.
                               Adjust once you have cohort survival data.

    retention_offer_cost     : Estimated cost per outreach (agent time +
                               offer value). Conservative at $15; true cost
                               depends on offer type (discount vs. upgrade).

    retention_success_rate   : Probability a Persuadable customer stays given
                               the offer. 0.30 is a common industry baseline.
                               Will be refined once A/B results arrive.

    false_alarm_cost         : Cost of contacting a customer who wasn't going
                               to churn (offer_cost only — no CLV risk).
                               Defaults to offer_cost.

    discount_rate_annual     : For NPV of CLV; 10% annual = ~0.83% monthly.
    """

    avg_monthly_revenue:      float = 69.84
    avg_customer_lifetime_mo: int   = 24
    retention_offer_cost:     float = 15.00
    retention_success_rate:   float = 0.30
    false_alarm_cost:         float = 15.00   # same as offer cost by default
    discount_rate_annual:     float = 0.10

    @property
    def monthly_discount_rate(self) -> float:
        return self.discount_rate_annual / 12

    @property
    def customer_lifetime_value(self) -> float:
        """
        CLV using discounted cash flow over avg_customer_lifetime_mo.
        CLV = R × [1 - (1+d)^-n] / d   where d = monthly discount rate, n = months.
        """
        d = self.monthly_discount_rate
        n = self.avg_customer_lifetime_mo
        if d == 0:
            return self.avg_monthly_revenue * n
        return self.avg_monthly_revenue * (1 - (1 + d) ** -n) / d


# ---------------------------------------------------------------------------
# 2.  Full scorecard — the honest multi-metric report
# ---------------------------------------------------------------------------

def full_scorecard(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "model",
) -> pd.DataFrame:
    """
    Return a single-row DataFrame with all metrics the CMO brief requires.

    Metrics
    -------
    auc_roc        : Standard discriminatory power.
    auc_pr         : Area under Precision-Recall curve. Better for imbalanced.
    brier_score    : Probabilistic calibration (lower = better).
    f1_default     : F1 at threshold=0.5.
    f1_at_thresh   : F1 at the provided threshold.
    precision_t    : Precision at threshold.
    recall_t       : Recall at threshold (= sensitivity).
    specificity_t  : True negative rate at threshold.
    tp / fp / tn / fn : Raw confusion matrix cells.
    """
    y_true = np.array(y_true)
    y_pred_default = (y_proba >= 0.5).astype(int)
    y_pred_t       = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t).ravel()

    precision_t  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_t     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    scorecard = {
        "model":          model_name,
        "threshold":      threshold,
        "auc_roc":        round(roc_auc_score(y_true, y_proba), 4),
        "auc_pr":         round(average_precision_score(y_true, y_proba), 4),
        "brier_score":    round(brier_score_loss(y_true, y_proba), 4),
        "f1_at_0.5":      round(f1_score(y_true, y_pred_default), 4),
        "f1_at_thresh":   round(f1_score(y_true, y_pred_t), 4),
        "precision":      round(precision_t, 4),
        "recall":         round(recall_t, 4),
        "specificity":    round(specificity, 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "n_flagged":      int(tp + fp),
        "flag_rate_pct":  round((tp + fp) / len(y_true) * 100, 1),
    }
    return pd.DataFrame([scorecard])


# ---------------------------------------------------------------------------
# 3.  Threshold sweep — expected net value curve
# ---------------------------------------------------------------------------

def threshold_sweep(
    y_true:  np.ndarray | pd.Series,
    y_proba: np.ndarray,
    params:  Optional[BusinessParams] = None,
    n_steps: int = 180,
) -> pd.DataFrame:
    """
    Sweep thresholds from 0.05 → 0.95 and compute expected net business value
    at each threshold.

    Business value formula (per threshold t)
    -----------------------------------------
    For each customer flagged as churn:
        True positive  (actual churner):
            Expected value = CLV × retention_success_rate − offer_cost
        False positive (non-churner flagged):
            Expected value = −false_alarm_cost  (wasted offer)

    Net value = Σ(TP × EV_tp) + Σ(FP × EV_fp)

    Returns
    -------
    DataFrame with columns:
        threshold, tp, fp, tn, fn, precision, recall, f1,
        net_value, revenue_saved, campaign_cost, n_contacted, roi_pct
    """
    if params is None:
        params = BusinessParams()

    clv     = params.customer_lifetime_value
    ev_tp   = clv * params.retention_success_rate - params.retention_offer_cost
    ev_fp   = -params.false_alarm_cost

    y_true  = np.array(y_true)
    thresholds = np.linspace(0.05, 0.95, n_steps)
    rows = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        revenue_saved  = tp * params.retention_success_rate * clv
        campaign_cost  = (tp + fp) * params.retention_offer_cost
        net_value      = tp * ev_tp + fp * ev_fp
        n_contacted    = int(tp + fp)
        roi            = (net_value / campaign_cost * 100) if campaign_cost > 0 else 0.0

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        rows.append({
            "threshold":     round(t, 4),
            "tp":            int(tp),
            "fp":            int(fp),
            "tn":            int(tn),
            "fn":            int(fn),
            "precision":     round(prec, 4),
            "recall":        round(rec, 4),
            "f1":            round(f1, 4),
            "net_value":     round(net_value, 2),
            "revenue_saved": round(revenue_saved, 2),
            "campaign_cost": round(campaign_cost, 2),
            "n_contacted":   n_contacted,
            "roi_pct":       round(roi, 1),
        })

    df = pd.DataFrame(rows)
    return df


def optimal_threshold(sweep_df: pd.DataFrame, metric: str = "net_value") -> float:
    """
    Return the threshold that maximises the chosen metric.
    metric options: 'net_value', 'f1', 'roi_pct'
    """
    idx = sweep_df[metric].idxmax()
    return float(sweep_df.loc[idx, "threshold"])


# ---------------------------------------------------------------------------
# 4.  ROC and PR curve data (for plotting in notebooks / Streamlit)
# ---------------------------------------------------------------------------

def roc_curve_data(y_true, y_proba) -> pd.DataFrame:
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def pr_curve_data(y_true, y_proba) -> pd.DataFrame:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns n+1 points; align lengths
    thresholds = np.append(thresholds, np.nan)
    return pd.DataFrame({
        "precision": precision,
        "recall":    recall,
        "threshold": thresholds,
    })


# ---------------------------------------------------------------------------
# 5.  Cross-validation scorecard aggregator
# ---------------------------------------------------------------------------

def aggregate_cv_scores(cv_results: list[dict]) -> pd.DataFrame:
    """
    Aggregate per-fold scorecards from cross-validation into mean ± std.

    Parameters
    ----------
    cv_results : list of dicts, each dict = output of full_scorecard() .iloc[0].to_dict()

    Returns
    -------
    DataFrame with mean and std for each numeric metric.
    """
    df = pd.DataFrame(cv_results)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    agg = pd.DataFrame({
        "mean": df[numeric_cols].mean().round(4),
        "std":  df[numeric_cols].std().round(4),
    })
    return agg


# ---------------------------------------------------------------------------
# 6.  Revenue-at-risk summary (for Streamlit KPI cards)
# ---------------------------------------------------------------------------

def revenue_at_risk_summary(
    df: pd.DataFrame,
    churn_proba_col: str = "churn_proba",
    monthly_charges_col: str = "monthly_charges",
    threshold: float = 0.5,
    params: Optional[BusinessParams] = None,
) -> dict:
    """
    Compute headline KPIs for the Streamlit dashboard top cards.

    Returns
    -------
    dict with keys:
        n_total, n_at_risk, pct_at_risk,
        monthly_revenue_at_risk, annual_revenue_at_risk,
        clv_at_risk, expected_saves, net_value_of_campaign
    """
    if params is None:
        params = BusinessParams()

    flagged = df[df[churn_proba_col] >= threshold]
    n_at_risk = len(flagged)
    monthly_rev = flagged[monthly_charges_col].sum()

    clv_at_risk = monthly_rev * params.avg_customer_lifetime_mo  # simplified

    expected_saves   = n_at_risk * params.retention_success_rate
    revenue_saved    = expected_saves * params.avg_monthly_revenue * params.avg_customer_lifetime_mo
    campaign_cost    = n_at_risk * params.retention_offer_cost
    net_value        = revenue_saved - campaign_cost

    return {
        "n_total":               len(df),
        "n_at_risk":             n_at_risk,
        "pct_at_risk":           round(n_at_risk / len(df) * 100, 1),
        "monthly_revenue_at_risk": round(monthly_rev, 0),
        "annual_revenue_at_risk":  round(monthly_rev * 12, 0),
        "clv_at_risk":             round(clv_at_risk, 0),
        "expected_saves":          round(expected_saves, 0),
        "net_value_of_campaign":   round(net_value, 0),
    }


# ---------------------------------------------------------------------------
# 7.  Per-customer action table (ranked output list)
# ---------------------------------------------------------------------------

def build_action_table(
    df_meta: pd.DataFrame,        # customer_id, monthly_charges, segment_label
    y_proba: np.ndarray,
    uplift_scores: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    params: Optional[BusinessParams] = None,
) -> pd.DataFrame:
    """
    Build the ranked customer action list for the CMO / ops team.

    Output columns
    --------------
    customer_id, monthly_charges, churn_probability, uplift_score,
    segment_label, recommended_action, expected_net_value, priority_rank
    """
    if params is None:
        params = BusinessParams()

    df = df_meta.copy().reset_index(drop=True)
    df["churn_probability"] = np.round(y_proba, 4)

    if uplift_scores is not None:
        df["uplift_score"] = np.round(uplift_scores, 4)
    else:
        df["uplift_score"] = np.nan

    clv   = params.customer_lifetime_value
    ev_tp = clv * params.retention_success_rate - params.retention_offer_cost

    df["is_flagged"] = (df["churn_probability"] >= threshold).astype(int)

    # Expected net value if contacted = churn_proba × EV_tp (for unflagged: 0)
    df["expected_net_value"] = np.where(
        df["is_flagged"] == 1,
        df["churn_probability"] * ev_tp,
        0.0,
    ).round(2)

    # Action logic
    def _action(row):
        if row["is_flagged"] == 0:
            return "Monitor — no action"
        seg = str(row.get("segment_label", ""))
        if "Price" in seg or "Budget" in seg:
            return "Offer: Contract upgrade + discount"
        elif "Frustrated" in seg or "Support" in seg:
            return "Offer: Free tech support tier upgrade"
        elif "Veteran" in seg or "Disengag" in seg:
            return "Offer: Loyalty plan + personal outreach"
        else:
            return "Offer: General retention bundle"

    df["recommended_action"] = df.apply(_action, axis=1)

    # Priority rank: uplift_score if available, else churn_probability
    sort_col = "uplift_score" if uplift_scores is not None else "churn_probability"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    df["priority_rank"] = df.index + 1

    output_cols = [
        "priority_rank", "customer_id", "monthly_charges",
        "churn_probability", "uplift_score", "segment_label",
        "recommended_action", "expected_net_value",
    ]
    return df[[c for c in output_cols if c in df.columns]]
