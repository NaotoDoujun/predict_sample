""" 
Random Forest Regression Pipeline
- Predictive modeling for Excel datasets
- Feature importance analysis
- Model persistence (Export functionality)
"""
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Argument parser configuration
parser = argparse.ArgumentParser(description='Random Forest Regression')
parser.add_argument('--file', type=str, default='sample_data.xlsx', help='Path to Excel file')
parser.add_argument('--sheet', type=str, default='Sheet1', help='Sheet name')
parser.add_argument('--trees', type=int, default=100, help='Number of trees (n_estimators)')
parser.add_argument('--depth', type=int, default=None, help='Maximum depth of trees')
parser.add_argument('--save_model', type=str, default='rf_model.pkl', help='Output model filename')
args = parser.parse_args()

# Load and Preprocess data
try:
    with pd.ExcelFile(args.file) as xls:
        sheet_name = args.sheet if args.sheet in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet_name).dropna()
        print(f"Loaded: {args.file} [{sheet_name}] | Shape: {df.shape}")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# Prepare Features and Target
feature_names = df.columns[:-1].tolist()
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
print(f"Training Random Forest (Trees: {args.trees}, Depth: {args.depth})...")
model = RandomForestRegressor(
    n_estimators=args.trees,
    max_depth=args.depth,
    random_state=42,
    n_jobs=-1,
    bootstrap=True
)
model.fit(X_train, y_train)

# Persist Model
joblib.dump(model, args.save_model)
print(f"Model saved to {args.save_model}")

# Prediction and Metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Performance Metrics ---")
print(f"R2 Score: {r2:.4f}")
print(f"MSE     : {mse:.4f}")
print(f"MAE     : {mae:.4f}")

# Visualization - 1: Actual vs Predicted
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5, color='teal', edgecolors='white')
line_coords = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(line_coords, line_coords, 'r--', lw=2, label='Ideal (y=x)')
plt.title(f'RF Regression: Actual vs Predicted (R²={r2:.3f})')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.grid(True, alpha=0.3)

# Visualization - 2: Feature Importance (Top 10)
plt.subplot(1, 2, 2)
importances = model.feature_importances_
indices = np.argsort(importances)[-10:] # Top 10 features
plt.barh(range(len(indices)), importances[indices], color='darkblue', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.title('Top 10 Feature Importances')
plt.xlabel('Relative Importance')

plt.tight_layout()
plt.show()
