# Credit-Default-Prediction
Credit Default Prediction – A Quantitative Risk Modeling Approach
# 🧠 Credit Default Prediction using Logistic Regression

This project applies a full **credit risk modeling pipeline** to the Kaggle "Give Me Some Credit" training dataset, simulating challenges faced in **quantitative risk management**. It combines data preprocessing, model development, interpretability, and performance improvement techniques relevant to real-world risk modeling tasks.

## 🗂️ Dataset
- **Source**: [Give Me Some Credit – Kaggle](https://www.kaggle.com/datasets/c/GiveMeSomeCredit)
- **Objective**: Develop a predictive model to estimate the probability of serious delinquency (credit default) and gain insights into the relative contribution of key features driving default risk.

---

## ⚙️ Pipeline Overview

### 1. Data Preprocessing
- Handled missing values via **median imputation**
- Removed non-informative features (e.g., ID columns)
- Performed **feature scaling** to improve model convergence

### 2. Baseline Logistic Regression
- Initial model failed to converge on raw data
- After scaling: convergence achieved, but poor **F1 score** and **recall**

### 3. Model Interpretability & Diagnosis
- Used **SHAP (SHapley Additive exPlanations)** to identify misleading feature attributions
- Diagnosed multicollinearity using **VIF (Variance Inflation Factor)**
- Dropped redundant or misleading variables based on VIF and SHAP analysis

### 4. Addressing Class Imbalance
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)**
- Improved **recall** and **F1 score** significantly after resampling

---

## 📊 Model Evaluation

| Model Stage                    | ROC-AUC | F1 Score | Precision | Recall  |
|-------------------------------|---------|----------|-----------|---------|
| Scaled Data (No Resampling)   | 0.693   | 0.072    | 0.52      | 0.04   |
| VIF-Filtered Features          | 0.665   | 0.028    | 0.52      | 0.014   |
| After SMOTE Resampling         | **0.755** | **0.672**  | **0.696**   | **0.650** |

---

## 💡 Key Learnings

- **Model explainability** (SHAP) helps detect counterintuitive feature effects
- **Multicollinearity** can distort model coefficients and SHAP values
- **SMOTE** is highly effective for improving classification performance in imbalanced credit data
- A thorough **QRM pipeline** combines modeling, diagnostics, and domain reasoning

---

## 🛠️ Tech Stack

- Python (Pandas, NumPy, Scikit-learn)
- SHAP
- Statsmodels (VIF)
- Imbalanced-learn (SMOTE)
- Matplotlib / Seaborn

---

## 📁 Project Structure
