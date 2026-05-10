# =============================================================================
# Customer Churn Prediction
# Binary classification: predict which customers are likely to churn
# =============================================================================

import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb


# =============================================================================
# 1. Load Data
# =============================================================================

conn = sqlite3.connect("churn.db")
df = pd.read_sql("SELECT * FROM customers", conn)
conn.close()

# Target and features
y = df["Exited"]
X = df.drop(columns=["Exited", "RowNumber", "CustomerId", "Surname"])

# One-hot encode categorical columns (e.g. Geography, Gender)
X = pd.get_dummies(X, drop_first=True)

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape:  {y.shape}")


# =============================================================================
# 2. Train / Test Split
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =============================================================================
# 3. Define Models
# Each pipeline applies SMOTE to address class imbalance before training.
# Scaling is included where the algorithm is distance- or gradient-sensitive.
# =============================================================================

models = {
    "Logistic Regression": make_pipeline_imb(
        SMOTE(random_state=42),
        StandardScaler(),
        LogisticRegression(random_state=42),
    ),
    "Random Forest": make_pipeline_imb(
        SMOTE(random_state=42),
        RandomForestClassifier(random_state=42),
    ),
    "K-Nearest Neighbors": make_pipeline_imb(
        SMOTE(random_state=42),
        StandardScaler(),
        KNeighborsClassifier(),
    ),
    "Support Vector Machine": make_pipeline_imb(
        SMOTE(random_state=42),
        StandardScaler(),
        SVC(probability=True, random_state=42),
    ),
    "XGBoost": make_pipeline_imb(
        SMOTE(random_state=42),
        XGBClassifier(eval_metric="logloss", random_state=42),
    ),
    "Gradient Boosting": make_pipeline_imb(
        SMOTE(random_state=42),
        GradientBoostingClassifier(random_state=42),
    ),
}


# =============================================================================
# 4. Train & Evaluate — Threshold Tuning via Precision-Recall Curve
# Rather than defaulting to 0.5, we select the threshold that maximises F1.
# =============================================================================

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

    # Drop the last element added by sklearn (no corresponding threshold)
    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)

    results[name] = {
        "model": model,
        "best_f1": f1_scores[best_idx],
        "best_threshold": thresholds[best_idx],
    }

    print(f"{name:30s} | F1 = {f1_scores[best_idx]:.4f} | Threshold = {thresholds[best_idx]:.4f}")


# =============================================================================
# 5. Best Model — Final Evaluation
# =============================================================================

best_name = max(results, key=lambda x: results[x]["best_f1"])
best_model = results[best_name]["model"]
best_threshold = results[best_name]["best_threshold"]

y_probs = best_model.predict_proba(X_test)[:, 1]
y_pred_tuned = (y_probs >= best_threshold)
y_pred_default = (y_probs >= 0.5)

print(f"\n{'='*60}")
print(f"Best model:     {best_name}")
print(f"Best threshold: {best_threshold:.4f}")
print(f"{'='*60}")

print("\nConfusion Matrix (tuned threshold):")
print(confusion_matrix(y_test, y_pred_tuned))

print("\nClassification Report (tuned threshold):")
print(classification_report(y_test, y_pred_tuned))

print("Classification Report (default threshold = 0.5):")
print(classification_report(y_test, y_pred_default))


# =============================================================================
# 6. Precision-Recall vs Threshold Plot
# =============================================================================

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision, label="Precision")
plt.plot(thresholds, recall, label="Recall")
plt.axvline(best_threshold, color="gray", linestyle="--", label=f"Best threshold ({best_threshold:.2f})")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title(f"Precision & Recall vs Threshold — {best_name}")
plt.legend()
plt.tight_layout()
plt.show()
