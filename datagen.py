""" 
Synthetic Data Generator for AI Benchmarking
- Generates complex numerical and categorical relationships
- Supports custom feature sizing and sample counts
- Includes noise and outliers for robust model testing
"""
import argparse
import pandas as pd
import numpy as np

def generate_data():
    # Argument parser for flexible configuration
    parser = argparse.ArgumentParser(description='AI Sample Data Generator')
    parser.add_argument('--samples', type=int, default=2000, help='Number of rows to generate')
    parser.add_argument('--features', type=int, default=48, help='Number of numerical features')
    parser.add_argument('--output', type=str, default='sample_data.xlsx', help='Output filename')
    args = parser.parse_args()

    np.random.seed(42)
    n = args.samples
    p = args.features

    # 1. Generate numerical features (X)
    X = np.random.uniform(0, 100, size=(n, p))
    columns = [f'Feature_{i}' for i in range(p)]
    df = pd.DataFrame(X, columns=columns)

    # 2. Add some "Categorical" flavor (encoded as integers for simplicity)
    # This simulates different groups/regions that might affect the target
    df['Group_ID'] = np.random.randint(1, 5, size=n)

    # 3. Create complex target variable (y)
    # Linear part
    weights = np.random.uniform(0.5, 2.0, size=p)
    y = np.dot(X, weights)

    # Non-linear part: Feature_0 squared and Interaction between Feature_1 & Feature_2
    y += (df['Feature_0'] ** 1.5) * 0.1
    y += (df['Feature_1'] * df['Feature_2']) * 0.01
    
    # Categorical effect: Different groups have different offsets
    group_effects = {1: 50, 2: -30, 3: 10, 4: 100}
    y += df['Group_ID'].map(group_effects)

    # 4. Add Gaussian Noise
    noise = np.random.normal(0, df.target.mean() * 0.05 if hasattr(df, 'target') else 15, size=n)
    y += noise

    # 5. Inject Outliers (0.5% of data)
    outlier_idx = np.random.choice(n, size=int(n * 0.005), replace=False)
    y[outlier_idx] += np.random.normal(500, 1000, size=len(outlier_idx))

    df['Target'] = y

    # Save to Excel
    try:
        df.to_excel(args.output, index=False)
        print(f"Successfully created: {args.output}")
        print(f"Configuration: {n} samples, {p} features + 1 categorical group")
        print(f"Sample data (first 5 rows):\n{df.head()}")
    except Exception as e:
        print(f"Error saving to Excel: {e}")

if __name__ == "__main__":
    generate_data()
