import json

# Read the notebook
with open('notebooks/baseline_model.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the specific syntax error
content = content.replace(
    'print(f"SHAP values shape: {shap_values.values.shape}")print(f"X_test_lgbm columns: {len(X_test_lgbm.columns)}")',
    'print(f"SHAP values shape: {shap_values.values.shape}")\nprint(f"X_test_lgbm columns: {len(X_test_lgbm.columns)}")'
)

# Write back
with open('notebooks/baseline_model.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed syntax error')