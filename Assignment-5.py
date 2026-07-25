"""
AI-ML Assignment 5
Employee Attrition Prediction using Decision Tree and Random Forest Classification

Name            : AADISH ADLAK
Registration No.: 23BCE10681
Application No. : IN26010985
Batch           : 9A
Email           : adlakaadish@gmail.com

Dataset: IBM HR Analytics Employee Attrition & Performance
Kaggle : https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Run:
    python Assignment-5.py
Outputs (saved in ./outputs/):
    confusion_matrix_decision_tree.png
    confusion_matrix_random_forest.png
    feature_importance_random_forest.png
    model_comparison.csv
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

sns.set_style("whitegrid")
RANDOM_STATE = 42

DATA_PATH = os.path.join("data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------------
# TASK 1: DATA UNDERSTANDING
# ---------------------------------------------------------------------------
section("TASK 1: DATA UNDERSTANDING")

df = pd.read_csv(DATA_PATH)

print("\n--- First 5 records ---")
print(df.head())

# Identify target and feature types
target_variable = "Attrition"

numerical_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = df.select_dtypes(include=["object"]).columns.tolist()
categorical_features.remove(target_variable)

print(f"\nTarget variable: '{target_variable}'")
print(f"\nNumerical features ({len(numerical_features)}):\n{numerical_features}")
print(f"\nCategorical features ({len(categorical_features)}):\n{categorical_features}")

print("\n--- Dataset Info ---")
df.info()

print("\n--- Summary Statistics (numerical) ---")
print(df.describe())

print("\n--- Summary Statistics (categorical) ---")
print(df.describe(include="object"))

print(f"\nShape of dataset: {df.shape}")
print("\nTarget class distribution:")
print(df[target_variable].value_counts())
print(df[target_variable].value_counts(normalize=True).round(3) * 100, "%")


# ---------------------------------------------------------------------------
# TASK 2: DATA PREPROCESSING
# ---------------------------------------------------------------------------
section("TASK 2: DATA PREPROCESSING")

# 2.1 Check for missing values
print("\n--- Missing values per column ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

# 2.2 Remove unnecessary columns
# EmployeeCount, StandardHours -> constant value for every row (no predictive power)
# EmployeeNumber -> unique identifier, not a real feature
# Over18 -> constant value ('Y' for every row)
useless_cols = ["EmployeeCount", "StandardHours", "EmployeeNumber", "Over18"]
print(f"\nDropping constant / identifier columns: {useless_cols}")
df_clean = df.drop(columns=useless_cols)

# refresh feature lists after dropping columns
numerical_features = [c for c in numerical_features if c not in useless_cols]
categorical_features = [c for c in categorical_features if c not in useless_cols]

# 2.3 Encode categorical variables
# Target variable -> Label Encoding (Yes/No -> 1/0)
le_target = LabelEncoder()
df_clean[target_variable] = le_target.fit_transform(df_clean[target_variable])  # Yes=1, No=0
print(f"\nTarget encoding map: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# Remaining categorical predictors -> One-Hot Encoding
df_encoded = pd.get_dummies(df_clean, columns=categorical_features, drop_first=True)
print(f"\nShape after one-hot encoding: {df_encoded.shape}")

# 2.4 Train / test split (80/20), stratified because classes are imbalanced
X = df_encoded.drop(columns=[target_variable])
y = df_encoded[target_variable]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTraining set size: {X_train.shape}")
print(f"Testing set size : {X_test.shape}")


# ---------------------------------------------------------------------------
# TASK 3: MODEL DEVELOPMENT
# ---------------------------------------------------------------------------
section("TASK 3: MODEL DEVELOPMENT")

# Model 1: Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree model trained.")

# Model 2: Random Forest Classifier (100 estimators)
rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest model trained (n_estimators=100).")


# ---------------------------------------------------------------------------
# TASK 4: MODEL EVALUATION AND COMPARISON
# ---------------------------------------------------------------------------
section("TASK 4: MODEL EVALUATION AND COMPARISON")


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["No", "Yes"]))
    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}


results_dt = evaluate("Decision Tree", y_test, y_pred_dt)
results_rf = evaluate("Random Forest", y_test, y_pred_rf)

comparison_df = pd.DataFrame([results_dt, results_rf])
comparison_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
print("\n--- Model Comparison Table ---")
print(comparison_df.to_string(index=False))


# Confusion matrices
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")


plot_confusion_matrix(y_test, y_pred_dt, "Confusion Matrix - Decision Tree",
                       "confusion_matrix_decision_tree.png")
plot_confusion_matrix(y_test, y_pred_rf, "Confusion Matrix - Random Forest",
                       "confusion_matrix_random_forest.png")

# Feature importance plot (Random Forest)
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title("Top 15 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_importance_random_forest.png"), dpi=150)
plt.close()
print("Saved: feature_importance_random_forest.png")

print("\n--- Top 10 Most Important Features (Random Forest) ---")
print(top_features.head(10))


# ---------------------------------------------------------------------------
# Observations (printed for the report / README)
# ---------------------------------------------------------------------------
section("OBSERVATIONS")

acc_winner = "Random Forest" if results_rf["Accuracy"] >= results_dt["Accuracy"] else "Decision Tree"
f1_winner = "Random Forest" if results_rf["F1-Score"] >= results_dt["F1-Score"] else "Decision Tree"
recall_winner = "Random Forest" if results_rf["Recall"] >= results_dt["Recall"] else "Decision Tree"
top5_features = ", ".join(top_features.head(5).index.tolist())

observations = f"""
1. Overall accuracy: {acc_winner} achieved the higher overall accuracy
   (Decision Tree = {results_dt['Accuracy']:.4f}, Random Forest = {results_rf['Accuracy']:.4f}).
   Random Forest's ensemble of {rf_model.n_estimators} trees averages out the errors
   of individual trees, which usually reduces variance compared to a single
   Decision Tree, but a single well-fit tree can occasionally match it on a
   dataset of this size.

2. Class imbalance effect: the dataset is imbalanced (~84% "No" vs ~16% "Yes"
   attrition). This pulls overall accuracy up for both models while making the
   minority class ("Yes") harder to recall. {recall_winner} achieved the higher
   recall on the "Yes" class in this run (Decision Tree recall =
   {results_dt['Recall']:.4f}, Random Forest recall = {results_rf['Recall']:.4f}),
   which matters more than accuracy for an HR team that wants to catch as many
   at-risk employees as possible.

3. F1-Score (balance of precision & recall): {f1_winner} produced the better
   F1-score for the "Yes" class (Decision Tree = {results_dt['F1-Score']:.4f},
   Random Forest = {results_rf['F1-Score']:.4f}). A single Decision Tree, despite
   overfitting the training data, can sometimes achieve higher recall/F1 on the
   minority class than Random Forest, because bagging + majority voting tends to
   bias Random Forest predictions toward the majority class on imbalanced data
   unless class weighting or resampling is used.

4. Feature importance: {top5_features} are the strongest predictors of attrition
   according to the Random Forest model, which aligns with HR domain intuition
   (compensation, overtime workload, distance from home and tenure strongly
   influence an employee's decision to leave).
"""
print(observations)

with open(os.path.join(OUT_DIR, "observations.txt"), "w") as f:
    f.write(observations)

# ---------------------------------------------------------------------------
# TASK 5: CONCLUSION
# ---------------------------------------------------------------------------
section("TASK 5: CONCLUSION")

conclusion = f"""
On the IBM HR Analytics dataset, {acc_winner} produced the better overall
accuracy ({max(results_rf['Accuracy'], results_dt['Accuracy']):.2%} vs
{min(results_rf['Accuracy'], results_dt['Accuracy']):.2%}), while the
{f1_winner} produced the better F1-score for identifying employees who
actually leave. In general, Random Forest tends to outperform a single
Decision Tree because it is an ensemble of many trees, each trained on a
random bootstrap sample of the data with a random subset of features at
every split. Averaging (or majority-voting) their predictions cancels out
the idiosyncratic errors of any one tree, which lowers variance, reduces
overfitting, and produces a decision boundary that generalizes better to
unseen employees.

A key limitation of a single Decision Tree is that it overfits easily: if
allowed to grow deep, it memorizes noise in the training data and its
predictions become unstable, as a small change in the training set can
produce a completely different tree. A key limitation of Random Forest is
interpretability and cost: because it combines the votes of many trees, it
is far harder to explain a single prediction to a non-technical HR manager
than the simple "if-then" rules of one Decision Tree, and it is more
computationally expensive to train and to run at prediction time,
especially as the number of estimators grows. On an imbalanced dataset such
as this one, both models also need extra care (e.g. class weighting,
resampling, or threshold tuning) to reliably catch the minority "Attrition
= Yes" class, since raw accuracy is dominated by the majority "No" class.
"""
print(conclusion)

with open(os.path.join(OUT_DIR, "conclusion.txt"), "w") as f:
    f.write(conclusion)

print("\nAll outputs saved in the 'outputs/' folder.")
print("Script finished successfully.")
