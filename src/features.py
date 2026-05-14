"""
src/features.py
---------------
Feature engineering pipeline for the Churn Detective project.

Design principles:
  - All transformations are pure functions or sklearn-compatible transformers.
  - No side effects — nothing is written to disk here.
  - Every engineered feature is documented with business rationale.
  - Categorical handling is dual-path:
      * LightGBM path  → assign pandas Categorical dtype (no encoding needed)
      * sklearn path   → OrdinalEncoder via ColumnTransformer (for RF, KMeans)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


# ---------------------------------------------------------------------------
# 1.  Schema constants — single source of truth for column names
# ---------------------------------------------------------------------------

RAW_SCHEMA = {
    "id":            "customer_id",
    "target":        "churned",
    "numeric": [
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "support_calls_3mo",
        "avg_data_gb_3mo",
        "late_payments_6mo",
        "plan_changes_6mo",
        "senior_citizen",          # 0/1 int — treated as numeric
    ],
    "categorical": [
        "contract_type",
        "internet_service",
        "online_security",
        "tech_support",
        "streaming_tv",
        "payment_method",
        "paperless_billing",
        "partner",
        "dependents",
        "phone_service",
        "multiple_lines",
    ],
}

# Ordered categories for OrdinalEncoder (where order is meaningful)
ORDINAL_CATEGORIES = {
    "contract_type":   ["Month-to-month", "One year", "Two year"],
    "internet_service":["No", "DSL", "Fiber optic"],
}

# Binary Yes/No columns (mapped 1/0)
BINARY_YES_NO = [
    "online_security", "tech_support", "streaming_tv",
    "paperless_billing", "partner", "dependents",
    "phone_service", "multiple_lines",
]

# Payment method nominal (will be one-hot in sklearn path)
PAYMENT_NOMINAL = ["payment_method"]


# ---------------------------------------------------------------------------
# 2.  Raw loading & basic cleaning
# ---------------------------------------------------------------------------

def load_raw(path: str) -> pd.DataFrame:
    """Load CSV, coerce dtypes, and perform sanity checks."""
    df = pd.read_csv(path)

    # Enforce expected columns
    expected = (
        [RAW_SCHEMA["id"], RAW_SCHEMA["target"]]
        + RAW_SCHEMA["numeric"]
        + RAW_SCHEMA["categorical"]
    )
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # total_charges can be whitespace-string in some Telco variants
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    # senior_citizen arrives as 0/1 int — keep, but document
    df["senior_citizen"] = df["senior_citizen"].astype(int)

    return df


def drop_unusable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that cannot be used:
      - Null total_charges (typically tenure=0 rows — customer never billed)
      - Duplicate customer_id (keep first)
    """
    n_before = len(df)
    df = df.dropna(subset=["total_charges"])
    df = df.drop_duplicates(subset=["customer_id"], keep="first")
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[features] Dropped {n_dropped} unusable rows.")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3.  Engineered features — each with business rationale
# ---------------------------------------------------------------------------

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns that carry signal not present in raw features.

    New columns
    -----------
    charges_per_tenure_month : float
        Average billed per month of actual tenure.
        Catches billing anomalies — customers billed far above their stated
        monthly_charges are likely experiencing unexpected fees → churn risk.
        Formula: total_charges / (tenure_months + 1)   [+1 avoids div/0]

    service_frustration_index : int
        Composite of support_calls_3mo + late_payments_6mo.
        Both signals independently predict churn; their sum creates a single
        'how annoyed is this customer' axis useful for segmentation.

    is_new_customer : int  (0/1)
        tenure_months ≤ 6  →  1
        New customers have a qualitatively different churn profile
        (onboarding failure) vs. churning veterans (disengagement).

    is_long_tenure : int  (0/1)
        tenure_months ≥ 48  →  1
        Long-tenured customers who churn are a different crisis — they are
        'quietly disengaging veterans' who represent higher CLV loss.

    monthly_charges_bin : str (category)
        $20-$45 / $45-$70 / $70-$95 / $95-$120
        Discretized price tier — useful for segmentation and EDA grouping.
        Not used in tree models (they handle continuous fine) but kept for
        the Streamlit segment view.

    has_no_support_services : int  (0/1)
        Both online_security AND tech_support == 'No'  →  1
        Customers with zero support services are more exposed to churn
        from the first service incident — compound vulnerability.

    payment_is_manual : int  (0/1)
        payment_method in {'Electronic check', 'Mailed check'}  →  1
        Manual payment methods correlate with disengagement and higher
        late_payments — a friction signal.
    """
    df = df.copy()

    # --- charges_per_tenure_month ---
    df["charges_per_tenure_month"] = (
        df["total_charges"] / (df["tenure_months"] + 1)
    ).round(4)

    # --- service_frustration_index ---
    df["service_frustration_index"] = (
        df["support_calls_3mo"] + df["late_payments_6mo"]
    )

    # --- lifecycle flags ---
    df["is_new_customer"] = (df["tenure_months"] <= 6).astype(int)
    df["is_long_tenure"]  = (df["tenure_months"] >= 48).astype(int)

    # --- price tier ---
    df["monthly_charges_bin"] = pd.cut(
        df["monthly_charges"],
        bins=[20, 45, 70, 95, 120],
        labels=["$20–45", "$45–70", "$70–95", "$95–120"],
        right=True,
        include_lowest=True,
    ).astype(str)

    # --- support services gap ---
    no_security = df["online_security"].isin(["No", "No internet service"])
    no_support  = df["tech_support"].isin(["No", "No internet service"])
    df["has_no_support_services"] = (no_security & no_support).astype(int)

    # --- payment friction ---
    manual_methods = {"Electronic check", "Mailed check"}
    df["payment_is_manual"] = df["payment_method"].isin(manual_methods).astype(int)

    return df


ENGINEERED_NUMERIC = [
    "charges_per_tenure_month",
    "service_frustration_index",
    "is_new_customer",
    "is_long_tenure",
    "has_no_support_services",
    "payment_is_manual",
]


# ---------------------------------------------------------------------------
# 4.  LightGBM path — categorical dtype assignment (no encoding)
# ---------------------------------------------------------------------------

def prepare_for_lgbm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame ready for LightGBM:
      - Categorical columns assigned pandas Categorical dtype.
      - Binary Yes/No mapped to 1/0 int.
      - 'No internet service' / 'No phone service' collapsed to 'No'
        (they carry the same meaning and reduce cardinality).
      - customer_id and monthly_charges_bin dropped (non-feature columns).
    """
    df = df.copy()

    # Collapse redundant category values
    redundant_map = {
        "No internet service": "No",
        "No phone service":    "No",
    }
    for col in RAW_SCHEMA["categorical"]:
        if col in df.columns:
            df[col] = df[col].replace(redundant_map)

    # Binary Yes/No → int
    for col in BINARY_YES_NO:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # Ordinal categoricals → Categorical dtype (LightGBM reads these natively)
    for col, order in ORDINAL_CATEGORIES.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=order, ordered=True)

    # Nominal categoricals → unordered Categorical
    for col in PAYMENT_NOMINAL:
        if col in df.columns:
            df[col] = pd.Categorical(df[col])

    # Drop non-feature columns
    drop_cols = ["customer_id", "monthly_charges_bin", "churned"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


# ---------------------------------------------------------------------------
# 5.  sklearn path — full ColumnTransformer for RF / KMeans
# ---------------------------------------------------------------------------

def _get_all_numeric_cols(df: pd.DataFrame) -> list[str]:
    base = RAW_SCHEMA["numeric"].copy()
    engineered = [c for c in ENGINEERED_NUMERIC if c in df.columns]
    return base + engineered


class BinaryYesNoEncoder(BaseEstimator, TransformerMixin):
    """Map Yes→1, No→0, unknown→0 for a list of columns."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        redundant_map = {
            "No internet service": "No",
            "No phone service":    "No",
        }
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].replace(redundant_map)
                X[col] = X[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        return X


def build_sklearn_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer for Random Forest / KMeans:
      - Numeric cols  → StandardScaler
      - Binary Yes/No → BinaryYesNoEncoder → passthrough (already 0/1)
      - Ordinal cats  → OrdinalEncoder with known category order
      - Payment       → OrdinalEncoder (treat as nominal ordinal proxy)

    Returns the unfitted transformer; caller fits on train split.
    """
    numeric_cols  = _get_all_numeric_cols(df)
    ordinal_cols  = list(ORDINAL_CATEGORIES.keys())
    ordinal_cats  = list(ORDINAL_CATEGORIES.values())

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                [c for c in numeric_cols if c in df.columns],
            ),
            (
                "ord",
                OrdinalEncoder(
                    categories=ordinal_cats,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                [c for c in ordinal_cols if c in df.columns],
            ),
            (
                "pay",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                [c for c in PAYMENT_NOMINAL if c in df.columns],
            ),
        ],
        remainder="drop",   # binary cols handled separately via BinaryYesNoEncoder
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_names_after_transform(
    preprocessor: ColumnTransformer, df: pd.DataFrame
) -> list[str]:
    """Return feature names in the same order as the transformed array."""
    numeric_cols = _get_all_numeric_cols(df)
    ordinal_cols = list(ORDINAL_CATEGORIES.keys())
    names = (
        [c for c in numeric_cols if c in df.columns]
        + [c for c in ordinal_cols if c in df.columns]
        + [c for c in PAYMENT_NOMINAL if c in df.columns]
    )
    return names


# ---------------------------------------------------------------------------
# 6.  KMeans feature subset  (churners-only segmentation)
# ---------------------------------------------------------------------------

KMEANS_FEATURES = [
    "monthly_charges",
    "tenure_months",
    "support_calls_3mo",
    "avg_data_gb_3mo",
    "late_payments_6mo",
    "service_frustration_index",   # engineered
    "charges_per_tenure_month",    # engineered
]


def prepare_for_kmeans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return scaled numeric feature matrix for KMeans clustering.
    Input df should already have engineered features added.
    Only churned==1 rows should be passed (caller's responsibility).
    """
    missing = [c for c in KMEANS_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"KMeans features missing from df: {missing}")

    X = df[KMEANS_FEATURES].copy().fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=KMEANS_FEATURES, index=df.index), scaler


# ---------------------------------------------------------------------------
# 7.  Full preprocessing convenience function
# ---------------------------------------------------------------------------

def build_model_ready_data(
    raw_path: str,
    target_col: str = "churned",
    random_state: int = 42,
) -> dict:
    """
    End-to-end convenience function used by notebooks.

    Returns
    -------
    dict with keys:
        df_clean     : cleaned + engineered full DataFrame
        X_lgbm       : feature DataFrame ready for LightGBM (train+test rows)
        y            : target Series
        customer_ids : Series (for post-hoc joins)
    """
    from sklearn.model_selection import train_test_split

    df = load_raw(raw_path)
    df = drop_unusable(df)
    df = add_engineered_features(df)

    customer_ids = df["customer_id"].copy()
    y = df[target_col].copy()

    X_lgbm = prepare_for_lgbm(df)

    return {
        "df_clean":     df,
        "X_lgbm":       X_lgbm,
        "y":            y,
        "customer_ids": customer_ids,
    }
