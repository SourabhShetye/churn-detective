import json

with open('notebooks/baseline_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the SHAP cell and modify it
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        source_str = ''.join(source)
        if 'shap.TreeExplainer(lgbm_model)' in source_str:
            # Replace the explainer line
            new_source = []
            for line in source:
                if 'explainer   = shap.TreeExplainer(lgbm_model)' in line:
                    new_source.append('explainer   = shap.TreeExplainer(lgbm_model, feature_names=X_test_lgbm.columns.tolist())\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source
            break

with open('notebooks/baseline_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Updated SHAP explainer with feature_names')