# 💰 Loan Repayment Prediction Using Ensemble Learning Methods

<div align="center">

# 🏦 Loan Repayment Prediction — Ensemble Machine Learning

![Python](https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge\&logo=pandas\&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge\&logo=numpy\&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)
![Ensemble Learning](https://img.shields.io/badge/Ensemble-Learning-success?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-85%25-brightgreen?style=for-the-badge)

> Predict whether a borrower is likely to fully repay a loan using multiple Ensemble Learning techniques including Decision Trees, Bagging, AdaBoost, Random Forest, and Gradient Boosting.

</div>

---

## ⚠️ Business Context

Financial institutions face significant risk from loan defaults. Machine Learning models can help banks identify high-risk applicants and make informed lending decisions, reducing financial losses and improving credit assessment processes.

---

## 📌 Table of Contents

* About the Project
* Workflow
* Dataset
* Exploratory Data Analysis
* Data Preprocessing
* Model Development
* Ensemble Learning Techniques
* Results & Performance Analysis
* Project Structure
* Getting Started
* Tech Stack
* Future Improvements
* References

---

# 🔬 About the Project

This project focuses on predicting whether a loan applicant will fully repay a loan using supervised machine learning and ensemble learning methods.

The dataset contains information about applicants including:

* Credit Policy Status
* Loan Purpose
* Interest Rate
* Installment Amount
* Annual Income
* Debt-to-Income Ratio
* FICO Credit Score
* Revolving Balance
* Credit History Information

Several machine learning algorithms were trained and evaluated to determine the most effective approach for loan repayment prediction.

### Key Objectives

* Analyze loan applicant data
* Perform data preprocessing and feature engineering
* Visualize important financial indicators
* Train multiple ensemble learning models
* Compare performance of different algorithms
* Identify the best-performing model

---

# ⚙️ Workflow

```text
Loan Applicant Data
          │
          ▼
   Data Loading
          │
          ▼
 Data Cleaning & Validation
          │
          ▼
 Label Encoding
 (Purpose Feature)
          │
          ▼
 Exploratory Data Analysis
          │
          ▼
 Feature Correlation Analysis
          │
          ▼
 Train-Test Split
 (70% Train / 30% Test)
          │
          ▼
 Model Training
 ┌───────────────────────┐
 │ Decision Tree         │
 │ Bagging Classifier    │
 │ AdaBoost              │
 │ Random Forest         │
 │ Gradient Boosting     │
 └───────────────────────┘
          │
          ▼
 Performance Evaluation
          │
          ▼
 Best Model Selection
```

---

# 📊 Dataset

| Property         | Details               |
| ---------------- | --------------------- |
| Dataset Size     | 9,578 Records         |
| Features         | 14                    |
| Target Variable  | not.fully.paid        |
| Missing Values   | None                  |
| Problem Type     | Binary Classification |
| Train-Test Split | 70%-30%               |

### Target Classes

| Value | Meaning        |
| ----- | -------------- |
| 0     | Fully Paid     |
| 1     | Not Fully Paid |

---

# 📈 Features Used

| Feature           | Description                                 |
| ----------------- | ------------------------------------------- |
| credit.policy     | Customer meets credit underwriting criteria |
| purpose           | Purpose of the loan                         |
| int.rate          | Interest rate                               |
| installment       | Monthly installment amount                  |
| log.annual.inc    | Log annual income                           |
| dti               | Debt-to-income ratio                        |
| fico              | FICO credit score                           |
| days.with.cr.line | Days with credit line                       |
| revol.bal         | Revolving balance                           |
| revol.util        | Revolving utilization rate                  |
| inq.last.6mths    | Credit inquiries in last 6 months           |
| delinq.2yrs       | Delinquencies in last 2 years               |
| pub.rec           | Public records                              |
| not.fully.paid    | Target Variable                             |

---

# 🧹 Data Preprocessing

### Handling Categorical Data

The **purpose** feature was categorical and encoded using:

```python
from sklearn.preprocessing import LabelEncoder

df['purpose'] = LabelEncoder().fit_transform(df['purpose'])
```

### Missing Values

```python
df.isnull().sum().sum()
```

Result:

```text
0
```

No missing values were present in the dataset.

---

# 📊 Exploratory Data Analysis

The following analyses were performed:

### FICO Score Distribution

* Compared credit policy approved and non-approved customers
* Analyzed repayment behavior against FICO scores

### Loan Purpose Analysis

Count plots were generated to understand:

* Loan categories
* Default distribution across loan purposes

### Interest Rate vs FICO

Joint plots helped visualize:

* Relationship between credit score and interest rate
* Risk assessment trends

### Correlation Heatmap

A correlation matrix was generated to identify influential features.

Key influential variables:

* Interest Rate
* Credit Policy
* FICO Score
* Credit Inquiries in Last 6 Months

---

# 🌲 Model Development

## Decision Tree Classifier

Hyperparameter tuning was performed using GridSearchCV.

```python
param_grid = {
    'max_depth':[2,3,4,5,6,7,8,9,10,11,13,15,20]
}
```

Best Parameter:

```python
max_depth = 2
```

### Performance

| Metric         | Value  |
| -------------- | ------ |
| Train Accuracy | 83.74% |
| Test Accuracy  | 84.58% |

---

# 🎯 Ensemble Learning Techniques

## 1️⃣ Bagging Classifier

Base Estimator:

```python
DecisionTreeClassifier(max_depth=2)
```

Parameters:

```python
n_estimators = 100
bootstrap = True
```

### Result

| Metric     | Value  |
| ---------- | ------ |
| Mean Score | 73.10% |

Observation:

Bagging did not improve performance significantly.

---

## 2️⃣ AdaBoost Classifier

Base Estimator:

```python
DecisionTreeClassifier(max_depth=2)
```

Learning Rate:

```python
0.5
```

### Result

| Metric         | Value |
| -------------- | ----- |
| Train Accuracy | 85%   |
| Test Accuracy  | 84%   |

Observation:

Performance remained similar to the Decision Tree model.

---

## 3️⃣ Random Forest Classifier ⭐

Parameters:

```python
RandomForestClassifier(
    n_estimators=600
)
```

### Performance

| Metric        | Value |
| ------------- | ----- |
| Test Accuracy | 84.7% |

### Advantages

* Handles feature interactions effectively
* Reduces overfitting
* Provides stable predictions

---

## 4️⃣ AdaBoost with Random Forest

### Performance

| Metric        | Value  |
| ------------- | ------ |
| Test Accuracy | 84.76% |

Observation:

Only marginal improvement over Random Forest.

---

## 5️⃣ Gradient Boosting

Learning Rate:

```python
0.05
```

### Performance

| Metric        | Value  |
| ------------- | ------ |
| Test Accuracy | 84.45% |

Observation:

Comparable performance but not superior to Random Forest.

---

# 📉 Results Comparison

| Model             | Accuracy |
| ----------------- | -------- |
| Decision Tree     | 84.58%   |
| Bagging           | 73.10%   |
| AdaBoost          | 84.00%   |
| Random Forest     | 84.70%   |
| AdaBoost + RF     | 84.76%   |
| Gradient Boosting | 84.45%   |

---

# 🏆 Best Model

### Random Forest Classifier

```text
Accuracy: ~85%
```

Reasons:

* Highest overall performance
* Robust against overfitting
* Stable predictions across folds
* Effective handling of complex feature interactions

---

# 📁 Project Structure

```text
Loan Repayment Prediction/
│
├── loan_data.csv
├── Loan_Repayment_Prediction.ipynb
├── Loan_Repayment_Prediction.pdf
├── requirements.txt
└── README.md
```

---

# 🚀 Getting Started

## 1. Clone Repository

```bash
git clone <repository-link>
cd Loan-Repayment-Prediction
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run Notebook

```bash
jupyter notebook Loan_Repayment_Prediction.ipynb
```

---

# 🛠️ Tech Stack

| Category         | Technology                                                         |
| ---------------- | ------------------------------------------------------------------ |
| Language         | Python                                                             |
| Data Processing  | Pandas, NumPy                                                      |
| Visualization    | Matplotlib, Seaborn                                                |
| Machine Learning | Scikit-Learn                                                       |
| Models           | Decision Tree, Random Forest, AdaBoost, Bagging, Gradient Boosting |
| Notebook         | Jupyter Notebook                                                   |

---

# 🔮 Future Improvements

* Handle class imbalance using SMOTE
* Perform feature selection
* Tune ensemble hyperparameters further
* Apply XGBoost and LightGBM
* Deploy model using Flask or Streamlit
* Create a loan approval prediction dashboard

---

# 📚 References

* Scikit-Learn Documentation
* Ensemble Learning Methods
* Credit Risk Prediction Research Papers
* Python Data Science Handbook

---

<div align="center">

### 💡 Machine Learning Project

Predicting Loan Repayment using Ensemble Learning Techniques

⭐ If you found this project useful, consider starring the repository.

</div>
