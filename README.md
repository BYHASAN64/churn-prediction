# 📊 Customer Churn Prediction (Machine Learning Project)

## 📌 Project Overview

This project predicts whether a customer will churn (leave the service) using machine learning models. The goal is to identify high-risk customers and improve retention strategies.

---

## 📂 Dataset

The dataset is stored in a local SQLite database (`churn.db`) and originally contains customer demographic and banking information.

Target variable:

* `Exited` → 1 if customer churned, 0 otherwise

---

## ⚙️ Data Processing

* Loaded data from SQLite database
* Removed irrelevant features:

  * RowNumber
  * CustomerId
  * Surname
* Applied one-hot encoding to categorical variables
* Split dataset into training and testing sets (80/20)

---

## 🧠 Feature Engineering

* Converted categorical variables using `pd.get_dummies`
* Balanced dataset using SMOTE (Synthetic Minority Oversampling Technique)

---

## 🤖 Models Used

Multiple machine learning models were trained and compared:

* Logistic Regression
* Random Forest
* K-Nearest Neighbors
* Support Vector Machine
* XGBoost
* Gradient Boosting

All models were evaluated using a pipeline including:

* SMOTE
* StandardScaler (where needed)

---

## 📊 Evaluation Strategy

* Precision-Recall Curve analysis
* F1-score optimization
* Threshold tuning (not fixed at 0.5)
* Confusion matrix analysis
* Classification report comparison

---

## 🏆 Model Selection

The best model was selected based on highest F1-score after threshold optimization.

---

## 📈 Key Insight

* Threshold tuning significantly improves churn detection performance
* Class imbalance handling (SMOTE) is critical for this dataset
* Ensemble models outperform basic linear models in most cases

---

## 🚀 How to Run

```bash
python3 src/train.py
```

---

## 🧠 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn
* Matplotlib
* SQLite

---

## 📌 Future Improvements

* Hyperparameter tuning (GridSearchCV / Optuna)
* Model deployment (Flask / FastAPI)
* Feature importance analysis
* Real-time prediction API

---

## 👨‍💻 Author

Hasan Bay

Machine Learning Enthusiast | Focused on AI & Data Science
