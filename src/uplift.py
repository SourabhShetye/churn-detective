"""
src/uplift.py
-------------
Meta-learner uplift modeling for the Churn Detective project.

The core question we answer here is NOT "who will churn?" but
"who will STAY if we make them an offer?" — the Persuadable population.

Architecture
------------
We implement two meta-learners and compare them:

  S-Learner (Single model):
    Train ONE model on (X, T) where T is the treatment indicator.
    ITE = f(X, T=1) - f(X, T=0)
    Pro: Simple, low variance. Con: Can under-detect treatment effect
    if model regularises the T feature away.

  T-Learner (Two models):
    Train μ₀(X) on control group, μ₁(X) on treatment group.
    ITE = μ₁(X) - μ₀(X)
    Pro: Each model fully specialised. Con: High variance if groups are small.

Treatment variable construction
--------------------------------
We do NOT have a true randomised control trial. We construct a synthetic
treatment indicator and document all assumptions transparently.

Strategy A (preferred): Use plan_changes_6mo > 0 as a proxy for
  "customer was contacted / engaged with a retention-like event".
  Assumption: customers who changed plans did so in response to some
  form of outreach. This is weak but directionally valid.

Strategy B (fallback): Random synthetic assignment T ~ Bernoulli(0.5).
  Treats the problem as a simulation. ATE will be ~0 by construction
  but ITE variance still reveals relative persuadability.

Both strategies are flagged clearly in the Limitations section.

Validation
----------
Without ground-truth uplift labels we validate using:
  - Qini curve & AUUC (Area Under Uplift Curve)
  - Uplift by decile table (top 20% should capture > 40% of total uplift)
  - Sensitivity analysis: does ITE ranking agree between S and T learner?
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# 1.  Treatment variable construction
# ---------------------------------------------------------------------------

def build_treatment_variable(
    df: pd.DataFrame,
    strategy: str = "plan_changes",
    random_state: int = 42,
) -> pd.Series:
    """
    Construct binary treatment indicator T ∈ {0, 1}.

    Parameters
    ----------
    strategy : 'plan_changes' | 'random'
        'plan_changes' : T=1 if plan_changes_6mo > 0  (preferred)
        'random'       : T ~ Bernoulli(0.5)  (simulation fallback)

    Returns
    -------
    pd.Series of int (0/1) aligned with df.index, named 'treatment'.
    """
    if strategy == "plan_changes":
        if "plan_changes_6mo" not in df.columns:
            raise ValueError("Column 'plan_changes_6mo' not found. Use strategy='random'.")
        T = (df["plan_changes_6mo"] > 0).astype(int)
        treatment_rate = T.mean()
        print(
            f"[uplift] Treatment strategy: plan_changes_6mo > 0  "
            f"→ {T.sum():,} treated ({treatment_rate:.1%} of population)\n"
            f"  ⚠  ASSUMPTION: plan changes proxy engagement with retention offer.\n"
            f"     Validate with A/B test before production deployment."
        )
    elif strategy == "random":
        rng = np.random.default_rng(random_state)
        T = pd.Series(
            rng.integers(0, 2, size=len(df)).astype(int),
            index=df.index,
            name="treatment",
        )
        print(
            "[uplift] Treatment strategy: RANDOM (simulation mode)\n"
            "  ⚠  ITE estimates reflect relative persuadability, not causal effects.\n"
            "     ATE will be ~0 by construction. Use for ranking only."
        )
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose 'plan_changes' or 'random'.")

    return T.rename("treatment")


# ---------------------------------------------------------------------------
# 2.  S-Learner
# ---------------------------------------------------------------------------

class SLearner:
    """
    S-Learner meta-learner for Individual Treatment Effect estimation.

    A single base model is trained on (X ∪ {T}, Y).
    ITE(x) = P(Y=1 | X=x, T=1) − P(Y=1 | X=x, T=0)

    Parameters
    ----------
    base_model : sklearn-compatible classifier with predict_proba().
                 Default: LGBMClassifier with sensible churn-task settings.
    """

    def __init__(self, base_model=None):
        if base_model is None:
            try:
                from lightgbm import LGBMClassifier
                base_model = LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    is_unbalance=True,
                    random_state=42,
                    verbose=-1,
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                base_model = GradientBoostingClassifier(
                    n_estimators=200, random_state=42
                )
        self.base_model = base_model
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        T: pd.Series,
        y: pd.Series,
    ) -> "SLearner":
        """
        Fit on the combined feature matrix [X | T].

        Parameters
        ----------
        X : Feature DataFrame (without treatment column).
        T : Binary treatment Series (0/1).
        y : Binary outcome Series — churned (1) or not (0).
        """
        X_train = X.copy()
        X_train["_treatment"] = T.values
        self.base_model.fit(X_train, y)
        self._feature_names = list(X_train.columns)
        self._fitted = True
        return self

    def predict_ite(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict Individual Treatment Effect for each row in X.

        Returns
        -------
        np.ndarray of shape (n,) — positive values indicate the offer
        is expected to REDUCE churn probability (the Persuadable signal).
        """
        self._check_fitted()
        X0 = X.copy()
        X0["_treatment"] = 0
        X1 = X.copy()
        X1["_treatment"] = 1

        p0 = self.base_model.predict_proba(X0)[:, 1]
        p1 = self.base_model.predict_proba(X1)[:, 1]

        # ITE = P(churn|T=0) - P(churn|T=1)
        # Positive ITE → offer REDUCES churn → Persuadable
        return p0 - p1

    def predict_churn_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Base churn probability (T=0 counterfactual — no offer given)."""
        self._check_fitted()
        X0 = X.copy()
        X0["_treatment"] = 0
        return self.base_model.predict_proba(X0)[:, 1]

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call .fit() before predicting.")


# ---------------------------------------------------------------------------
# 3.  T-Learner
# ---------------------------------------------------------------------------

class TLearner:
    """
    T-Learner meta-learner for Individual Treatment Effect estimation.

    Two separate base models:
      μ₀(X) trained on control group (T=0)
      μ₁(X) trained on treatment group (T=1)
    ITE(x) = μ₀(x) − μ₁(x)  [reduction in churn probability]

    Parameters
    ----------
    control_model, treatment_model : sklearn-compatible classifiers.
                                     If None, defaults to LGBMClassifier.
    """

    def __init__(self, control_model=None, treatment_model=None):
        def _default_lgbm():
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    is_unbalance=True,
                    random_state=42,
                    verbose=-1,
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(n_estimators=200, random_state=42)

        self.control_model   = control_model   or _default_lgbm()
        self.treatment_model = treatment_model or _default_lgbm()
        self._fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        T: pd.Series,
        y: pd.Series,
    ) -> "TLearner":
        """
        Fit control model on T=0 rows, treatment model on T=1 rows.
        Warns if either group is small (< 200 rows → high variance).
        """
        mask_control   = (T == 0)
        mask_treatment = (T == 1)

        n_ctrl = mask_control.sum()
        n_trt  = mask_treatment.sum()

        if n_ctrl < 200 or n_trt < 200:
            warnings.warn(
                f"T-Learner: control n={n_ctrl}, treatment n={n_trt}. "
                "Small groups → high ITE variance. Consider S-Learner instead.",
                UserWarning,
                stacklevel=2,
            )

        self.control_model.fit(X[mask_control], y[mask_control])
        self.treatment_model.fit(X[mask_treatment], y[mask_treatment])
        self._fitted = True
        return self

    def predict_ite(self, X: pd.DataFrame) -> np.ndarray:
        """
        ITE = P(churn | no offer) − P(churn | with offer)
        Positive → offer reduces churn → Persuadable.
        """
        self._check_fitted()
        p0 = self.control_model.predict_proba(X)[:, 1]
        p1 = self.treatment_model.predict_proba(X)[:, 1]
        return p0 - p1

    def predict_churn_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Base churn probability from control model (no offer counterfactual)."""
        self._check_fitted()
        return self.control_model.predict_proba(X)[:, 1]

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call .fit() before predicting.")


# ---------------------------------------------------------------------------
# 4.  Qini curve & AUUC
# ---------------------------------------------------------------------------

def qini_curve(
    y_true: np.ndarray | pd.Series,
    ite_scores: np.ndarray,
    treatment: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """
    Compute the Qini curve for uplift model evaluation.

    The Qini curve plots cumulative incremental conversions (customers
    retained IN ADDITION to the baseline) as we target an increasing
    fraction of the population, ordered by descending ITE score.

    Without ground-truth uplift labels we use the Adjusted Qini approach:
      Incremental gains = (TP_treated / N_treated) − (TP_control / N_control)
    scaled by population fraction.

    Parameters
    ----------
    y_true    : Binary outcome (1 = churned without offer, proxy for response).
    ite_scores: Predicted ITE from S/T-Learner (higher = more persuadable).
    treatment : Binary treatment indicator (1 = received offer).

    Returns
    -------
    DataFrame: fraction_targeted | qini_gain | random_gain
    """
    y_true    = np.array(y_true)
    ite_scores = np.array(ite_scores)
    treatment = np.array(treatment)

    # Sort by descending ITE
    order = np.argsort(-ite_scores)
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    n = len(y_sorted)
    n_treated  = treatment.sum()
    n_control  = n - n_treated

    rows = []
    cum_treated_positive  = 0
    cum_control_positive  = 0
    cum_treated_n         = 0
    cum_control_n         = 0

    for i in range(n):
        if t_sorted[i] == 1:
            cum_treated_n        += 1
            cum_treated_positive += y_sorted[i]
        else:
            cum_control_n        += 1
            cum_control_positive += y_sorted[i]

        rate_treated = cum_treated_positive / cum_treated_n if cum_treated_n > 0 else 0
        rate_control = cum_control_positive / cum_control_n if cum_control_n > 0 else 0

        # Qini gain = lift above control baseline, scaled by treated size
        qini_gain = (
            cum_treated_positive
            - cum_treated_n * (cum_control_positive / cum_control_n)
            if cum_control_n > 0 else 0
        )

        rows.append({
            "fraction_targeted": round((i + 1) / n, 4),
            "qini_gain":         round(qini_gain, 4),
            "random_gain":       round((i + 1) / n * (
                n_treated * y_true[treatment == 1].mean()
                - n_control * y_true[treatment == 0].mean()
            ) if n_treated > 0 and n_control > 0 else 0, 4),
        })

    return pd.DataFrame(rows)


def auuc_score(qini_df: pd.DataFrame) -> float:
    """
    Area Under the Uplift Curve (normalised to [0, 1]).
    Uses trapezoidal integration on the qini_gain column.
    Higher = better uplift model.
    """
    x = qini_df["fraction_targeted"].values
    y = qini_df["qini_gain"].values
    auc = np.trapz(y, x)
    # Normalise by the area of the 'perfect' model (theoretical max)
    y_random = qini_df["random_gain"].values
    auc_random = np.trapz(y_random, x)
    if auc_random == 0:
        return 0.0
    return round(auc / abs(auc_random), 4)


# ---------------------------------------------------------------------------
# 5.  Uplift by decile table (key CMO output)
# ---------------------------------------------------------------------------

def uplift_decile_table(
    y_true: np.ndarray | pd.Series,
    ite_scores: np.ndarray,
    treatment: np.ndarray | pd.Series,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Compute uplift metrics by score decile.

    This table is the primary output for the CMO:
      "Targeting the top 20% of customers by uplift score captures X% of
       all recoverable churn — at a cost of $Y per saved customer."

    Returns
    -------
    DataFrame with columns:
        decile, n_total, n_treated, n_control,
        outcome_rate_treated, outcome_rate_control,
        uplift_rate, cumulative_uplift_rate, pct_of_total_uplift
    """
    y_true    = np.array(y_true)
    ite_scores = np.array(ite_scores)
    treatment = np.array(treatment)

    # Assign decile (1 = top ITE, 10 = bottom ITE)
    df = pd.DataFrame({
        "y":         y_true,
        "ite":       ite_scores,
        "treatment": treatment,
    })
    df["decile"] = pd.qcut(
        -df["ite"], q=n_deciles, labels=False, duplicates="drop"
    ) + 1  # 1-indexed, 1 = highest ITE

    rows = []
    for d in sorted(df["decile"].unique()):
        sub = df[df["decile"] == d]
        treated = sub[sub["treatment"] == 1]
        control = sub[sub["treatment"] == 0]

        n_trt = len(treated)
        n_ctl = len(control)

        rate_trt = treated["y"].mean() if n_trt > 0 else np.nan
        rate_ctl = control["y"].mean() if n_ctl > 0 else np.nan
        uplift   = (rate_trt - rate_ctl) if (not np.isnan(rate_trt) and not np.isnan(rate_ctl)) else np.nan

        rows.append({
            "decile":                 int(d),
            "n_total":                len(sub),
            "n_treated":              n_trt,
            "n_control":              n_ctl,
            "outcome_rate_treated":   round(rate_trt, 4) if not np.isnan(rate_trt) else None,
            "outcome_rate_control":   round(rate_ctl, 4) if not np.isnan(rate_ctl) else None,
            "uplift_rate":            round(uplift, 4) if not np.isnan(uplift) else None,
        })

    result = pd.DataFrame(rows)

    # Cumulative uplift & share of total
    total_uplift = result["uplift_rate"].sum()
    result["cumulative_uplift_rate"] = result["uplift_rate"].cumsum().round(4)
    result["pct_of_total_uplift"] = (
        result["uplift_rate"] / total_uplift * 100
    ).round(1) if total_uplift != 0 else 0.0

    return result


# ---------------------------------------------------------------------------
# 6.  Persuadable flag + four-quadrant classification
# ---------------------------------------------------------------------------

def classify_four_quadrants(
    churn_proba: np.ndarray,
    ite_scores: np.ndarray,
    churn_threshold: float = 0.5,
    ite_threshold: float = 0.0,
) -> pd.Series:
    """
    Assign each customer to one of the four uplift quadrants.

    Quadrant definitions
    --------------------
    Persuadable   : high churn risk + positive ITE (offer helps)  ← PRIMARY TARGET
    Lost Cause    : high churn risk + negative/zero ITE (offer won't help)
    Sure Saver    : low churn risk + positive ITE (would stay anyway, wasted $)
    Sleeping Dog  : low churn risk + negative ITE (don't contact — might irritate)

    Returns
    -------
    pd.Series of string labels.
    """
    high_churn = churn_proba >= churn_threshold
    positive_ite = ite_scores > ite_threshold

    conditions = [
        ( high_churn &  positive_ite),
        ( high_churn & ~positive_ite),
        (~high_churn &  positive_ite),
        (~high_churn & ~positive_ite),
    ]
    choices = ["Persuadable", "Lost Cause", "Sure Saver", "Sleeping Dog"]

    return pd.Series(
        np.select(conditions, choices, default="Unknown"),
        name="uplift_quadrant",
    )


# ---------------------------------------------------------------------------
# 7.  Convenience wrapper — fit both learners, return comparison dict
# ---------------------------------------------------------------------------

def fit_and_compare_learners(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    T_train: pd.Series,
    T_test:  pd.Series,
    y_train: pd.Series,
    y_test:  pd.Series,
) -> dict:
    """
    Fit S-Learner and T-Learner, return ITE arrays + comparison metrics.

    Returns
    -------
    dict with keys:
        s_learner, t_learner       : Fitted model objects
        ite_s_train, ite_s_test    : S-Learner ITE arrays
        ite_t_train, ite_t_test    : T-Learner ITE arrays
        churn_proba_test           : Base churn probability (S-Learner, T=0)
        qini_s, qini_t             : Qini curve DataFrames
        auuc_s, auuc_t             : AUUC scores
        decile_s, decile_t         : Decile table DataFrames
        ite_rank_correlation       : Spearman correlation between S and T ITE rankings
                                     (high = learners agree on who is persuadable)
    """
    from scipy.stats import spearmanr

    s = SLearner()
    t = TLearner()

    print("[uplift] Fitting S-Learner…")
    s.fit(X_train, T_train, y_train)
    print("[uplift] Fitting T-Learner…")
    t.fit(X_train, T_train, y_train)

    ite_s_test = s.predict_ite(X_test)
    ite_t_test = t.predict_ite(X_test)
    churn_proba = s.predict_churn_proba(X_test)

    qini_s = qini_curve(y_test, ite_s_test, T_test)
    qini_t = qini_curve(y_test, ite_t_test, T_test)

    auuc_s = auuc_score(qini_s)
    auuc_t = auuc_score(qini_t)

    decile_s = uplift_decile_table(y_test, ite_s_test, T_test)
    decile_t = uplift_decile_table(y_test, ite_t_test, T_test)

    rank_corr, _ = spearmanr(ite_s_test, ite_t_test)

    print(f"\n[uplift] AUUC — S-Learner: {auuc_s:.4f} | T-Learner: {auuc_t:.4f}")
    print(f"[uplift] ITE rank correlation (Spearman): {rank_corr:.3f}")
    if rank_corr > 0.6:
        print("         ✅ Learners agree on persuadable ranking — use S-Learner ITE.")
    else:
        print("         ⚠  Low agreement. Investigate; consider X-Learner or DR-Learner.")

    return {
        "s_learner":            s,
        "t_learner":            t,
        "ite_s_test":           ite_s_test,
        "ite_t_test":           ite_t_test,
        "churn_proba_test":     churn_proba,
        "qini_s":               qini_s,
        "qini_t":               qini_t,
        "auuc_s":               auuc_s,
        "auuc_t":               auuc_t,
        "decile_s":             decile_s,
        "decile_t":             decile_t,
        "ite_rank_correlation": round(rank_corr, 4),
    }
