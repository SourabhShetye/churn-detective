import sys
sys.path.insert(0, '.')
import pandas as pd
import lightgbm as lgb
from src.features import prepare_for_lgbm

# Load data
test_df = pd.read_parquet('data/processed/test.parquet')
X_test_lgbm = prepare_for_lgbm(test_df)

print(f'X_test_lgbm shape: {X_test_lgbm.shape}')
print(f'X_test_lgbm columns: {len(X_test_lgbm.columns)}')

# Load model
import joblib
try:
    lgbm_model = joblib.load('models/lgbm_churn_v1.pkl')
    print(f'Model loaded, n_features_: {lgbm_model.n_features_}')
    
    # Test SHAP
    import shap
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer(X_test_lgbm)
    print(f'SHAP values shape: {shap_values.values.shape}')
    print(f'Has feature_names: {hasattr(shap_values, "feature_names")}')
    if hasattr(shap_values, 'feature_names'):
        print(f'Feature names length: {len(shap_values.feature_names)}')
except Exception as e:
    print(f'Error: {e}')