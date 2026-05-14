import json

with open('notebooks/baseline_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Remove debug prints and fix syntax
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        new_source = []
        for line in source:
            # Skip debug prints
            if 'print(f"SHAP values shape:' in line or 'print(f"X_test_lgbm columns:' in line:
                continue
            # Fix concatenated prints
            if '")print(' in line:
                parts = line.split('")print(')
                if len(parts) == 2:
                    new_source.append(parts[0] + '")')
                    new_source.append('print(' + parts[1])
                else:
                    new_source.append(line)
            else:
                new_source.append(line)
        cell['source'] = new_source

with open('notebooks/baseline_model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Cleaned up notebook')