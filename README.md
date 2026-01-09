# Project Overview: Excel-Based Regression Analysis Pipeline
This project provides a complete workflow for numerical prediction using Excel datasets, ranging from synthetic data generation to advanced machine learning modeling.

## Requirement
* torch 2.9.1
* scikit-learn 1.8.0
* lightgbm 4.6.0

### Install
```bash
pip install -r requirements.txt
```

## 1. Synthetic Data Generator
A program to generate optimal sample data for AI model testing and save it in Excel format.
* Description: Creates a structured dataset with 48 features and a target variable based on linear relationships with Gaussian noise.
* Purpose: To provide reliable benchmarking data for testing regression models.

### Run
```bash
# Generate Sample data
python ./datagen.py
```

## 2. Deep Learning Regression (PyTorch)
A regression analysis program that loads data from Excel and predicts values using PyTorch deep learning.
* Description: Implements a Multi-Layer Perceptron (MLP) with fully connected layers and ReLU activation.
* Key Features: Includes data standardization, mini-batch training using DataLoader, and performance visualization.

### Run
```bash
# Run NN regression
python ./neuralnetworks.py
```

## 3. Random Forest Regression (scikit-learn)
A regression analysis program that predicts numerical values from Excel data using Random Forest (ensemble learning).
* Description: Utilizes an ensemble-based approach for high-performance prediction without the need for feature scaling.
* Key Features: Provides fast, multi-core processing (n_jobs=-1) and evaluates model accuracy using R² score and MSE.

### Run
```bash
# Run RF regression
python ./randomforest.py
```

## 4. Gradient Boosting Regression (LightGBM)
A high-performance gradient boosting framework used as a competitive benchmark against the Deep Learning model.
* Description: Implements a Gradient Boosting Decision Tree (GBDT) optimized for tabular data.
* Key Features: Built-in handling of categorical features, L1 (MAE) objective for outlier robustness, and early stopping to prevent overfitting.

### Run
```bash
# Run LGBM regression
python ./lgbm.py
```
