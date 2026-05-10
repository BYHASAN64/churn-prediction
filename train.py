#Binary classification:~Predict which customers are going to churn~

import numpy as np
import sqlite3
import pandas as pd

conn = sqlite3.connect("churn.db")
df = pd.read_sql("SELECT * FROM customers", conn)

y = df['Exited']
X = df.drop(columns=["Exited", "RowNumber", "CustomerId", "Surname"])
X = pd.get_dummies(X, drop_first=True)#This removes the Exited column and other unnecesary variables from X
print(X.shape)
print(y.shape)

# X_train, X_test, y_train, y_test =X[:5000], X[5000:], y[:5000],y[5000:]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
#
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb# make_pipeline_imb creates a "wrapper" that dictates how the data is processed before it reaches the model
models = {
    'Logistic Regression': make_pipeline_imb(
        SMOTE(random_state=42),
        StandardScaler(),
        LogisticRegression(random_state=42)
    ),

    'Random Forest': make_pipeline_imb(
        SMOTE(random_state=42),
        RandomForestClassifier(random_state=42)
    ),

    'K-Nearest Neighbors': make_pipeline_imb(
        SMOTE(random_state=42),
        StandardScaler(),
        KNeighborsClassifier()
    ),

    'Support Vector Machine': make_pipeline_imb(
        SMOTE(random_state=42),
        StandardScaler(),
        SVC(probability=True, random_state=42)
    ),

    'XGBoost': make_pipeline_imb(
        SMOTE(random_state=42),
        XGBClassifier(eval_metric='logloss', random_state=42)
    ),

    'Gradient Boosting': make_pipeline_imb(
        SMOTE(random_state=42),
        GradientBoostingClassifier(random_state=42)
    )
}
from sklearn.metrics import precision_recall_curve
#model = LogisticRegression(max_iter=5000)
results={}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]

    
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

    precision = precision[:-1]
    recall = recall[:-1]
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    
    results[name] = {
        "model": model,
        "best_f1": f1_scores[best_idx],
        "best_threshold": thresholds[best_idx]
    }

    print(f"{name}: F1 = {f1_scores[best_idx]:.4f}")
    print(results[name]["best_f1"], results[name]["best_threshold"])

best_name = max(results, key=lambda x: results[x]["best_f1"])
best_model = results[best_name]["model"]
best_threshold = results[best_name]["best_threshold"]
y_probs = best_model.predict_proba(X_test)[:, 1]

print(f"\nBest model: {best_name}")
print(f"Best threshold: {best_threshold:.4f}")
print("Final y_pred:")
y_pred_final = (y_probs >= best_threshold)


print("Confusion matrix")


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_final)

print(cm)
from sklearn.metrics import classification_report ##
print("classification report:")

print(classification_report(y_test, y_pred_final))
default_pred = (y_probs >= 0.5)

print(classification_report(y_test, default_pred))


import matplotlib.pyplot as plt
# The standard fix for plotting P-R curves against thresholds
plt.plot(thresholds, precision[:], label="Precision")
plt.plot(thresholds, recall[:], label="Recall")

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.show()





