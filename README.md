Credit Risk Prediction Model (Loan Default Classification)
A machine learning project to predict whether a personal loan applicant will fully repay or default (charged-off). This project uses Logistic Regression, feature engineering, and class imbalance handling to build a real-world credit risk model.

🚀 Project Overview
Banks and lending institutions face substantial risk when approving personal loans. This project builds a predictive system that uses customer and loan attributes to estimate the likelihood of loan default, enabling smarter lending decisions.

📊 Objectives
* Predict whether a loan will be fully paid or charged off
* Identify the most influential financial and behavioral factors
* Improve model performance on the minority class (defaulters)
* Provide a pipeline suitable for deployment

🧹 Data Preprocessing & Cleaning
Key steps:
* Handled missing values across numerical and categorical features
* Removed true outliers only from highly continuous variables
* Processed skewed columns using log transforms where needed
* Dropped uninformative, leakage-prone, or overly sparse columns

🔧 Feature Engineering
Created 41+ high-quality features, including:
* Ordinal Encoding for grade and sub_grade
* Target encoding for high-cardinality features like purpose
* Employment length transformation (converted ordinal text to numeric)
* Date feature extraction
    * issue_date → year, month
    * earliest_cr_line → credit history length (in months)
* Ratio features (loan amount vs income)
* Standardization of all numerical features

⚖️ Handling Class Imbalance
Loan defaults formed a small portion of the dataset. To fix this:
* Used SMOTE oversampling
* Enabled class_weight='balanced' in Logistic Regression
* Tuned probability threshold using Precision-Recall curve
This significantly improved minority class detection.

🤖 Modeling Approach
Built a machine learning pipeline using:
* StandardScaler
* Logistic Regression (balanced)
* GridSearchCV for hyperparameter tuning (5-fold CV)
Best hyperparameter: C = 0.1

📈 Model Performance
Test Metrics:
Metric	Score
ROC-AUC	0.7072
Recall (Defaulters)	0.64
Accuracy	0.66
Precision (Defaulters)	0.32

🔍 Interpretation
* The model successfully identifies 64% of loan defaulters, which is critical for lending decisions.
* ROC-AUC of 0.71 indicates strong ranking ability in a highly imbalanced setting.

🧠 Business Insights
Key drivers of loan default:
* Higher interest rates significantly increase default probability
* Shorter credit history → higher risk
* Higher DTI (debt-to-income ratio) and loan amounts increase default likelihood
* Employment stability (emp_length) has moderate influence
These insights help financial institutions refine lending policies and reduce losses.

🛠️ Tech Stack
* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn
* Matplotlib, Seaborn
