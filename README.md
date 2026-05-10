# Customer Churn Prediction

A binary classification project that predicts which bank customers are likely to churn, using a comparison of six machine learning models with threshold tuning.

## Overview

Customer churn is costly to acquire vs. retain. This project trains multiple classifiers on historical customer data, handles class imbalance with SMOTE, and selects the best model by optimising the F1 score across the precision-recall curve — rather than defaulting to a 0.5 probability threshold.

## Dataset

Sourced from a SQLite database (`churn.db`, table: `customers`). Key features include credit score, geography, gender, age, tenure, balance, number of products, and activity status. The target variable is `Exited` (1 = churned, 0 = retained).

## Models Compared

| Model | Handles Imbalance | Scaled |
|---|---|---|
| Logistic Regression | SMOTE | ✓ |
| Random Forest | SMOTE | — |
| K-Nearest Neighbors | SMOTE | ✓ |
| Support Vector Machine | SMOTE | ✓ |
| XGBoost | SMOTE | — |
| Gradient Boosting | SMOTE | — |

## Setup

```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib
```

Place `churn.db` in the project root, then run:

```bash
python train.py
```

## Output

- Per-model F1 scores and optimal thresholds printed to console
- Confusion matrix and classification report for the best model
- Precision-Recall vs Threshold plot for visual threshold selection
