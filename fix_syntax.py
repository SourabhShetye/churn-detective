import json

with open('notebooks/baseline_model.ipynb', 'r') as f:
    nb = json.load(f)

# Fix the syntax error in the SHAP cell
for i, cell in enumerate(nb['cells']):
    source = cell.get('source', [])
    if 'shap_values = explainer(X_test_lgbm)' in ''.join(source):
        print(f'Found SHAP cell: {i+1}')
        # Fix the concatenated print statements
        new_source = []
        for line in source:
            if 'print(f"SHAP values shape: {shap_values.values.shape}")print(f"X_test_lgbm columns: {len(X_test_lgbm.columns)}")' in line:
                new_source.append('print(f"SHAP values shape: {shap_values.values.shape}")')
                new_source.append('print(f"X_test_lgbm columns: {len(X_test_lgbm.columns)}")')
            else:
                new_source.append(line)
        cell['source'] = new_source
        break

with open('notebooks/baseline_model.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Fixed syntax error in SHAP cell')