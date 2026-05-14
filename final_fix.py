import json

with open('notebooks/baseline_model.ipynb', 'r') as f:
    nb = json.load(f)

# Fix the SHAP cell
for i, cell in enumerate(nb['cells']):
    source = cell.get('source', [])
    source_text = ''.join(source)
    if 'explainer = shap.TreeExplainer(lgbm_model)' in source_text and 'shap_values = explainer(X_test_lgbm)' in source_text:
        print(f'Found SHAP cell: {i+1}')
        # Fix the explainer call and the concatenated print
        new_source = []
        for line in source:
            if 'explainer = shap.TreeExplainer(lgbm_model)' in line:
                new_source.append('explainer = shap.TreeExplainer(lgbm_model, feature_names=list(X_test_lgbm.columns))')
            elif 'print(f"SHAP values shape: {shap_values.values.shape}")print(f"X_test_lgbm columns: {len(X_test_lgbm.columns)}")' in line:
                new_source.append('print(f"SHAP values shape: {shap_values.values.shape}")')
                new_source.append('print(f"X_test_lgbm columns: {len(X_test_lgbm.columns)}")')
            else:
                new_source.append(line)
        cell['source'] = new_source
        print('Fixed SHAP cell')
        break

with open('notebooks/baseline_model.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Notebook fixed')