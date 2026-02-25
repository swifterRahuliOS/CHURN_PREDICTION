# Customer Churn Prediction with Temporal Drift Handling

## Project Overview

This project builds a customer churn prediction system for a
subscription-based business. The dataset is time-indexed (24 months),
and customer behavior changes over time. The objective is to:

-   Predict churn accurately
-   Avoid temporal data leakage
-   Handle data drift
-   Optimize business cost

------------------------------------------------------------------------

## Dataset Summary

-   Total observations: **5,000**
-   Total periods: **24 months**
-   Overall churn rate: **36.38%**
-   Target variable: `Churn` (1 = churned, 0 = retained)

### Features

-   `Period` -- Month index
-   `Tenure` -- Customer tenure
-   `Monthly_Amount` -- Billing amount
-   `Support_Calls` -- Number of support interactions
-   `Contract` -- Contract type

------------------------------------------------------------------------

## Methodology

### Drift-Oriented EDA

-   Analyzed churn rate per period
-   Checked feature distribution shifts
-   Studied missing values behavior
-   Confirmed presence of temporal drift

### Leakage-Free Pipeline

Used **scikit-learn Pipeline + ColumnTransformer**:

-   Numeric → Median Imputation + StandardScaler
-   Categorical → Constant Imputation ("MISSING") + OneHotEncoder
-   Preprocessing fitted only on training data

No random split was used. Train on past → Test on future.

------------------------------------------------------------------------

## Retraining Strategy

### Compared:

-   Fixed training
-   Rolling retraining

Rolling retraining performed better and was selected.

------------------------------------------------------------------------

## Model Evaluation

### Logistic Regression

Average AUC ≈ **0.87**

### Random Forest

Average AUC ≈ **0.86**

Logistic Regression selected due to better stability and performance.

------------------------------------------------------------------------

## Optimal Window Selection

Backtesting tested window sizes:

  Window   Avg AUC
  -------- ---------
  3        0.8707
  6        0.8667
  9        0.8639
  12       0.8590

Best window = **3 months**

------------------------------------------------------------------------

## Business Cost Optimization

Assumed: - False Negative cost = 10 - False Positive cost = 2

Optimal threshold = **0.228** Minimum business cost = **118**

Final Confusion Matrix (Last Period):

    [[64 39]
     [ 4 127]]

------------------------------------------------------------------------

## Final Production Setup

-   Model: Logistic Regression
-   Retraining: Rolling (3-month window)
-   Decision Threshold: 0.228
-   Average AUC ≈ 0.87

------------------------------------------------------------------------

## Key Takeaways

-   Temporal leakage must be avoided in time-based datasets
-   Rolling retraining adapts better to drift
-   Shorter history windows can improve adaptability
-   ML models must align with business cost, not just accuracy

------------------------------------------------------------------------