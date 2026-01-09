""" 
LightGBM Regression Analysis Program
- Competitive benchmark against Deep Learning
- Optimized for robustness against outliers using MAE objective
- Includes Feature Importance and Clean R2 evaluation
"""
import sys
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# Argument parser configuration
parser = argparse.ArgumentParser(description='LightGBM Regression from Excel Data')
parser.add_argument('--file', type=str, default='sample_data.xlsx', help='Path to the excel file')
parser.add_argument('--sheet', type=str, default='Sheet1', help='Sheet name')
parser.add_argument('--save_model', type=str, default='lgbm_model.pkl', help='Path to save the model')
args = parser.parse_args()

# Load and preprocessing
try:
    with pd.ExcelFile(args.file) as xls:
        sheet_name = args.sheet if args.sheet in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet_name).dropna()
        print(f"Loaded: {args.file} [{sheet_name}] | Shape: {df.shape}")
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

# Feature Engineering (Optional but recommended for benchmarking)
# Note: LightGBM handles Group_ID better if it's treated as a category
if 'Group_ID' in df.columns:
    df['Group_ID'] = df['Group_ID'].astype('category')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split data (No scaling needed for LightGBM)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Configuration
# Using 'regression_l1' (MAE) to be robust against outliers, similar to Huber Loss delta=0.1
params = {
    'objective': 'regression_l1', 
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'verbosity': -1,
    'seed': 42,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'n_jobs': -1
}

# Training with Early Stopping
print("Starting LightGBM training...")
model = lgb.LGBMRegressor(**params, n_estimators=1000)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# Save the trained model
joblib.dump(model, args.save_model)
print(f"\nModel saved to: {args.save_model}")

# Final Evaluation
y_pred = model.predict(X_test)

# Evaluate using the same logic as NN (Raw vs Clean R2)
z_test = np.abs((y_test - y_test.mean()) / y_test.std())
clean_idx = np.where(z_test < 3)[0]

r2_clean = r2_score(y_test.iloc[clean_idx], y_pred[clean_idx])
r2_raw = r2_score(y_test, y_pred)

print(f"\nFinal Evaluation (LightGBM):")
print(f"Raw R2 Score (with outliers): {r2_raw:.4f}")
print(f"Clean R2 Score (outliers removed): {r2_clean:.4f}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Feature Importance
lgb.plot_importance(model, ax=ax1, max_num_features=10, importance_type='gain', title='Feature Importance (Gain)')

# Actual vs Predicted
ax2.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', color='coral')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_title(f'LightGBM: Actual vs Predicted\n(Clean R² = {r2_clean:.4f})', fontsize=12)
ax2.set_xlabel('Actual Values')
ax2.set_ylabel('Predicted Values')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
