import sys
sys.path.insert(0, '.')
import pandas as pd
from src.features import prepare_for_lgbm

# Load data
test_df = pd.read_parquet('data/processed/test.parquet')
X_test_lgbm = prepare_for_lgbm(test_df)

print(f'X_test_lgbm shape: {X_test_lgbm.shape}')
print('X_test_lgbm dtypes:')
print(X_test_lgbm.dtypes.value_counts())
print(f'Categorical columns: {X_test_lgbm.select_dtypes("category").shape[1]}')
print(f'Numeric columns: {X_test_lgbm.select_dtypes(["int64", "float64"]).shape[1]}')