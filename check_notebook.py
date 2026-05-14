import json

with open('notebooks/baseline_model.ipynb', 'r') as f:
    nb = json.load(f)

# Find the SHAP cell and check its content
for i, cell in enumerate(nb['cells']):
    source = cell.get('source', [])
    if 'explainer = shap.TreeExplainer(lgbm_model, feature_names=list(X_test_lgbm.columns))' in ''.join(source):
        print(f'Found SHAP cell: {i+1}')
        print('Current source:')
        for j, line in enumerate(source):
            print(f'{j+1}: {repr(line)}')
        break