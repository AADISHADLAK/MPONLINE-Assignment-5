# Employee Attrition Prediction using Decision Tree and Random Forest Classification

**AI-ML Assignment 5**

| Field | Detail |
|---|---|
| **Name** | AADISH ADLAK |
| **Registration Number** | 23BCE10681 |
| **Application Number** | IN26010985 |
| **Batch Number** | 9A |
| **Assignment Number** | Assignment - 5 |
| **Email Address** | adlakaadish@gmail.com |
| **GitHub Repository** | https://github.com/AADISHADLAK/MPONLINE-Assignment-5 |

---

## 🎯 Objective

A company wants to identify employees who are likely to leave the organization based on their
demographic, professional, and work-related attributes. This project builds and compares two
classification models — a **Decision Tree Classifier** and a **Random Forest Classifier** — to
predict employee attrition (whether an employee will leave, `Attrition = Yes/No`) using the IBM HR
Analytics dataset.

## 📊 Dataset

**IBM HR Analytics Employee Attrition & Performance Dataset**

- **Source (Kaggle):** https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
- **Records:** 1,470 employees
- **Features:** 35 columns (demographic, job-related, and satisfaction attributes)
- **Target variable:** `Attrition` (`Yes` = employee left, `No` = employee stayed)

> ⚠️ **The dataset CSV is not pushed to GitHub** (it is listed in `.gitignore`), in line with the
> assignment's instruction not to redistribute the dataset without an explicit license. Download it
> yourself from the Kaggle link above and place it at `data/WA_Fn-UseC_-HR-Employee-Attrition.csv`
> before running the notebook/script.

## 🧰 Libraries Used

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data loading, cleaning, manipulation |
| `matplotlib`, `seaborn` | Data visualization (plots, confusion matrices, feature importance) |
| `scikit-learn` | Preprocessing (`LabelEncoder`, `train_test_split`), models (`DecisionTreeClassifier`, `RandomForestClassifier`), metrics (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report`) |
| `jupyter` | Notebook environment |

Install everything with:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
MPONLINE-Assignment-5/
├── Assignment-5.ipynb                 # Main Jupyter notebook (all 5 tasks, with explanations)
├── Assignment-5.py                    # Equivalent standalone Python script
├── bonus_hyperparameter_tuning.py     # Optional bonus challenge script
├── requirements.txt                   # Python dependencies
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Dataset (place here before running)
├── outputs/
│   ├── confusion_matrix_decision_tree.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrices_comparison.png
│   ├── feature_importance_random_forest.png
│   ├── model_comparison.csv
│   ├── observations.txt
│   └── conclusion.txt
└── README.md
```

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/AADISHADLAK/MPONLINE-Assignment-5.git
   cd MPONLINE-Assignment-5
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from Kaggle (link above) and place the CSV at:
   ```
   data/WA_Fn-UseC_-HR-Employee-Attrition.csv
   ```
4. Run either the notebook or the script:
   ```bash
   jupyter notebook Assignment-5.ipynb
   # OR
   python Assignment-5.py
   ```
   All plots and result tables are saved automatically to the `outputs/` folder.

## 🧪 Methodology

### Task 1 — Data Understanding
- Loaded the dataset with Pandas and inspected the first five records.
- Identified **26 numerical features** (e.g. `Age`, `MonthlyIncome`, `DistanceFromHome`,
  `TotalWorkingYears`) and **8 categorical features** (e.g. `BusinessTravel`, `Department`,
  `JobRole`, `MaritalStatus`, `OverTime`), with `Attrition` as the **target variable**.
- Reviewed `df.info()` and `df.describe()` to check data types and summary statistics.
- Confirmed the target is **imbalanced**: ~84% `No` vs ~16% `Yes`.

### Task 2 — Data Preprocessing
- **Missing values:** none found in the dataset (`df.isnull().sum()` = 0 for all columns).
- **Unnecessary columns removed:** `EmployeeCount`, `StandardHours`, `Over18` (constant value for
  every row — zero predictive power) and `EmployeeNumber` (a unique row identifier, not a real
  feature).
- **Encoding:**
  - Target variable `Attrition` → **Label Encoding** (`No` → 0, `Yes` → 1).
  - Remaining categorical predictors → **One-Hot Encoding** (`pd.get_dummies`, `drop_first=True`)
    to avoid the dummy-variable trap.
- **Train/test split:** 80% training / 20% testing using `train_test_split`, **stratified** on the
  target so both splits preserve the same class proportions (important given the class imbalance).

### Task 3 — Model Development
- **Model 1 — Decision Tree Classifier** (`sklearn.tree.DecisionTreeClassifier`, default
  parameters, `random_state=42`).
- **Model 2 — Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`,
  `n_estimators=100`, `random_state=42`).
- Both models were trained on the **same** training split and used to predict on the same test
  split for a fair comparison.

### Task 4 — Model Evaluation and Comparison
Both models were evaluated on Accuracy, Precision, Recall and F1-Score for the `Yes` (attrition)
class, plus confusion matrices and a Random Forest feature-importance plot.

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree | 0.7653 | 0.3103 | 0.3830 | 0.3429 |
| Random Forest | 0.8333 | 0.4167 | 0.1064 | 0.1695 |

*(Exact values, `random_state=42`, 80/20 stratified split — see `outputs/model_comparison.csv` for
the machine-readable version. Values may vary very slightly across scikit-learn versions.)*

**Confusion Matrices:**

`outputs/confusion_matrix_decision_tree.png`, `outputs/confusion_matrix_random_forest.png`, and a
side-by-side comparison in `outputs/confusion_matrices_comparison.png`.

**Feature Importance (Random Forest):** `outputs/feature_importance_random_forest.png` — the top
predictors of attrition are `MonthlyIncome`, `Age`, `TotalWorkingYears`, `DailyRate`, `HourlyRate`,
`DistanceFromHome`, `MonthlyRate`, `YearsAtCompany`, `OverTime`, and `YearsWithCurrManager`.

### Observations

1. **Overall accuracy:** Random Forest achieved a higher overall accuracy (**83.33%**) than the
   Decision Tree (**76.53%**), since averaging 100 trees smooths out the errors any single tree
   would make on this data.
2. **Class imbalance effect:** because the dataset is ~84% "No" / ~16% "Yes", accuracy alone is a
   misleading metric — the Decision Tree actually achieved a **higher recall (38.3% vs 10.6%)** on
   the minority "Yes" class, meaning it caught more employees who genuinely left, even though its
   overall accuracy was lower.
3. **F1-Score:** the Decision Tree also produced a **higher F1-score (0.343 vs 0.169)** for the
   "Yes" class in this run. This illustrates that Random Forest's majority-voting mechanism can bias
   predictions toward the majority class on imbalanced data unless techniques like class weighting,
   SMOTE/resampling, or probability-threshold tuning are applied.
4. **Feature importance:** `MonthlyIncome`, `Age`, `TotalWorkingYears`, `OverTime`, and
   `DistanceFromHome` are consistently the strongest predictors — this matches HR domain intuition
   that lower-paid, younger, less-tenured, and overworked employees (long commutes, frequent
   overtime) are more likely to leave.

## ⚖️ Model Comparison

| Aspect | Decision Tree | Random Forest |
|---|---|---|
| Overall Accuracy | Lower (76.53%) | Higher (83.33%) |
| Recall on "Yes" (attrition) | Higher (38.3%) | Lower (10.6%) |
| F1-Score on "Yes" | Higher (0.343) | Lower (0.169) |
| Overfitting risk | High (single deep tree) | Lower (ensemble averaging) |
| Interpretability | High (simple if-then rules, easy to visualize) | Low (aggregate of 100 trees) |
| Training/prediction cost | Low | Higher (100 trees) |
| Stability to data changes | Low (small changes → very different tree) | High (bagging reduces variance) |

## ✅ Conclusion

On this dataset, **Random Forest achieved the higher overall accuracy (83.33% vs 76.53%)**, while
the **Decision Tree achieved a better F1-score and recall for the minority "Attrition = Yes" class**
— which is arguably more important for an HR use case where the goal is to *catch* at-risk
employees rather than maximize raw accuracy on an imbalanced dataset. Random Forest generally
outperforms a single Decision Tree because it is an **ensemble** of many trees, each trained on a
random bootstrap sample of the data with a random subset of features considered at every split.
Combining (majority-voting) their predictions cancels out the idiosyncratic errors any individual
tree would make, which reduces variance, curbs overfitting, and typically produces a decision
boundary that generalizes better to unseen employees.

A key **limitation of Decision Trees** is that they overfit easily: an unpruned tree keeps splitting
until it memorizes noise in the training data, and a small change in the training set (or a
slightly different split) can produce a structurally very different tree, making predictions
unstable. A key **limitation of Random Forest** is reduced interpretability and higher computational
cost: because a prediction is the aggregated vote of 100 trees, it is far harder to explain a single
prediction to a non-technical HR manager than the simple if-then rules of one Decision Tree, and
training/inference is more expensive as the number of estimators grows. On imbalanced datasets like
this one, both models need extra care — class weighting, resampling, or threshold tuning — to
reliably identify the employees who are actually at risk of leaving.

## 🎁 Bonus Challenge (Optional — Hyperparameter Tuning)

`bonus_hyperparameter_tuning.py` varies the Decision Tree's `max_depth` parameter and reports the
effect on performance:

| max_depth | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| None (fully grown) | 0.7653 | 0.3103 | 0.3830 | 0.3429 |
| 3 | 0.8333 | 0.4545 | 0.2128 | 0.2899 |
| 5 | 0.8333 | 0.4444 | 0.1702 | 0.2462 |
| 8 | 0.8197 | 0.4167 | 0.3191 | 0.3614 |
| 10 | 0.8027 | 0.3778 | 0.3617 | 0.3696 |

**Effect of `max_depth`:** a fully-grown tree (`max_depth=None`) has the lowest bias but the
highest variance — it splits until nearly every leaf is pure, so it fits noise in the training set,
which can *lower* test accuracy compared to a shallower tree even though it may capture more true
minority-class patterns (higher recall/F1 here). Restricting `max_depth` to a small value (3–5)
prunes the tree and raises test accuracy, but pushes predictions toward the majority "No" class,
lowering recall/F1 for attrition cases. A moderate depth (8–10) gives the best trade-off between
overall accuracy and the ability to still catch true attrition cases in this experiment.

## 📤 Submission Notes

- Dataset is **not** committed to this repository (only the Kaggle link is provided), per the
  assignment's redistribution policy.
- This repository is kept **public** until evaluation is completed.
- Submitted via the Google Form with the details listed in the table at the top of this README.

---
*Assignment submitted for the AI-ML course (MPOnline). Deadline: 27 July 2026, 11:59 PM IST.*
