# Telecom Churn Prediction --- Imbalance and Threshold Optimization

A machine learning project for predicting customer churn using the
OpenML Telecom Churn dataset.

The main objective is not to maximize overall accuracy, but to identify
as many customers at risk of churn as possible while keeping the number
of unnecessary retention offers under control.

## Project Overview

Customer churn is an imbalanced binary classification problem: most
customers stay, while a smaller proportion leave.

For a retention campaign, a **false negative** can be particularly
costly: failing to identify a customer who is about to churn means
losing the opportunity to intervene.

The project therefore focuses on the performance of **class 1 (churn)**,
especially:

-   **Recall** --- how many actual churners are identified;
-   **Precision** --- how many customers flagged as churners actually
    churn;
-   **F1-score** --- a balance between precision and recall.

Overall accuracy is reported as a secondary metric rather than the
primary optimization target.

## Dataset

The project uses the `churn` dataset from OpenML:

``` python
from sklearn.datasets import fetch_openml

churn_data = fetch_openml(
    name="churn",
    version=1,
    as_frame=True
)

df_telecom = churn_data.frame
```

The dataset contains **5,000 customers** and **21 columns**.

### Class Balance

    Class Meaning     Proportion
  ------- --------- ------------
        0 Stayed          85.86%
        1 Churned         14.14%

The substantial class imbalance makes accuracy alone potentially
misleading.

## Data Cleaning and Feature Preparation

Several columns were removed before modeling.

### Removed Identifiers and Redundant Features

The following columns were dropped:

``` python
drop_cols = [
    "phone_number",
    "area_code",
    "total_day_charge",
    "total_eve_charge",
    "total_night_charge",
    "total_intl_charge"
]
```

The charge features were removed because they are deterministic
transformations of the corresponding usage-minute features and therefore
provide redundant information.

`phone_number` was treated as a non-informative identifier, while
`area_code` was also excluded from the modeling features.

### Target Encoding

The original categorical target was converted to a binary variable:

``` text
0 = stayed
1 = churned
```

The binary plan features were also converted to integer values.

## Train/Test Split

The data was split into training and test sets using an 80/20 split.

Stratification was used to preserve the original class proportions:

``` python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The test set contains **1,000 observations**, including 141 churned
customers.

# Model Exploration

Several classifiers were evaluated using the same held-out test set.

## 1. Logistic Regression --- Baseline

A standardized Logistic Regression model was used as the baseline:

``` python
make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42)
)
```

For the churn class:

    Precision   Recall     F1
  ----------- -------- ------
         0.62     0.20   0.30

Overall accuracy was **0.87**.

The baseline model has good overall accuracy but identifies only 20% of
actual churners.

This illustrates why accuracy alone is not sufficient for this problem.

## 2. Balanced Logistic Regression

Class weighting was then introduced:

``` python
LogisticRegression(
    random_state=42,
    class_weight="balanced"
)
```

Churn-class performance:

    Precision   Recall     F1
  ----------- -------- ------
         0.35     0.71   0.47

The recall increased substantially from **0.20 to 0.71**, but precision
dropped from **0.62 to 0.35**.

This demonstrates the fundamental precision-recall trade-off created by
class imbalance handling.

## 3. Random Forest

A class-balanced Random Forest was evaluated next:

``` python
RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)
```

Churn-class performance:

    Precision   Recall     F1
  ----------- -------- ------
         0.92     0.63   0.75

The Random Forest achieved a strong balance between precision and
recall, with particularly high precision.

## 4. HistGradientBoosting

A `HistGradientBoostingClassifier` was then tested:

``` python
HistGradientBoostingClassifier(
    random_state=42
)
```

Churn-class performance:

    Precision   Recall     F1
  ----------- -------- ------
         0.92     0.77   0.83

This produced the strongest overall F1-score among the initial models.

# Feature Selection with Permutation Importance

Permutation importance was calculated on the held-out test data to
identify which features contributed most to the model's predictive
performance.

The strongest features were:

  Feature                             Mean Importance
  --------------------------------- -----------------
  `total_day_minutes`                           0.094
  `number_customer_service_calls`               0.050
  `international_plan`                          0.050
  `total_eve_minutes`                           0.036
  `voice_mail_plan`                             0.029
  `total_intl_calls`                            0.026
  `total_intl_minutes`                          0.018
  `total_night_minutes`                         0.007

The model was subsequently retrained using these eight features.

# Slim HistGradientBoosting Model

The selected feature set was:

``` python
top_features = [
    "total_day_minutes",
    "number_customer_service_calls",
    "international_plan",
    "total_eve_minutes",
    "total_intl_calls",
    "voice_mail_plan",
    "total_intl_minutes",
    "total_night_minutes"
]
```

The reduced model produced:

    Precision   Recall     F1
  ----------- -------- ------
         0.91     0.79   0.84

Compared with the full-feature HGB model, the reduced model maintained
--- and slightly improved --- churn-class F1-score while using
substantially fewer features.

# Class-Imbalance Handling

The final stage used balanced sample weights:

``` python
from sklearn.utils.class_weight import compute_sample_weight

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

model_slim_balanced.fit(
    X_train_slim,
    y_train,
    sample_weight=sample_weights
)
```

The model was trained with sample weights that compensate for the
unequal class frequencies.

# Decision Threshold Optimization

A standard binary classifier uses a probability threshold of 0.50:

``` python
y_pred = (probability > 0.50).astype(int)
```

However, the threshold can be changed depending on the business
objective.

The trained model produced churn probabilities:

``` python
probs_balanced = model_slim_balanced.predict_proba(
    X_test_slim
)[:, 1]
```

Several thresholds were evaluated.

### Higher Thresholds

    Threshold   Recall   Precision
  ----------- -------- -----------
         0.30     0.83        0.72
         0.35     0.83        0.75
         0.40     0.82        0.77
         0.45     0.82        0.79
         0.50     0.82        0.81
         0.55     0.81        0.82

### Lower Thresholds

    Threshold   Recall   Precision
  ----------- -------- -----------
         0.10     0.89        0.48
         0.15     0.88        0.61
         0.20     0.87        0.67
         0.25     0.84        0.71

A threshold of **0.15** was selected for the final experiment because it
substantially increased recall while retaining a moderate level of
precision.

# Final Model

The final configuration combines:

1.  eight selected features;
2.  `HistGradientBoostingClassifier`;
3.  balanced training sample weights;
4.  a custom probability threshold of `0.15`.

The resulting churn-class performance was:

  Metric               Score
  --------------- ----------
  **Precision**     **0.61**
  **Recall**        **0.88**
  **F1-score**      **0.72**

Overall accuracy was **0.90**.

The model therefore identifies **88% of the churned customers in the
held-out dataset**, while 61% of customers flagged as churners actually
belong to the churn class.

# Model Comparison

  ---------------------------------------------------------------------------------------
  Model / Approach          Threshold       Recall    Precision           F1     Accuracy
  ---------------------- ------------ ------------ ------------ ------------ ------------
  Logistic Regression            0.50         0.20         0.62         0.30         0.87

  Balanced Logistic              0.50         0.71         0.35         0.47         0.78
  Regression                                                                 

  Random Forest                  0.50         0.63         0.92         0.75          ---

  HistGradientBoosting           0.50         0.77         0.92         0.83         0.95

  **Slim HGB + balanced      **0.15**     **0.88**     **0.61**     **0.72**     **0.90**
  weights + custom                                                           
  threshold**                                                                
  ---------------------------------------------------------------------------------------

The main optimization target was recall for the churn class rather than
global accuracy.

# Important Evaluation Note

The threshold was selected by evaluating several candidate thresholds on
the held-out test set.

Therefore, the reported final metrics should be interpreted as the
performance of the selected configuration on this dataset, **not as an
unbiased estimate of future generalization performance**.

A stricter evaluation would use a separate validation set (or
cross-validation) for threshold selection and reserve the test set for
one final evaluation.

# Business Interpretation

The final threshold changes the operating point of the classifier.

At the default threshold of 0.50, the balanced model identifies about
82% of churners with 81% precision.

At the selected threshold of 0.15:

-   recall increases to **88%**;
-   precision decreases to **61%**.

This is a deliberate trade-off.

The model becomes more willing to flag customers as potential churners,
reducing the number of missed churn cases at the cost of generating more
false positives.

# Technical Stack

-   **Python 3**
-   **Pandas** --- data preparation and analysis
-   **NumPy** --- numerical operations
-   **scikit-learn**
    -   `LogisticRegression`
    -   `RandomForestClassifier`
    -   `HistGradientBoostingClassifier`
    -   `StandardScaler`
    -   `train_test_split`
    -   `permutation_importance`
    -   `compute_sample_weight`
    -   classification metrics
-   **OpenML API** --- dataset retrieval

# Repository Structure

``` text
telecom-churn-prediction/
│
├── README.md
└── telecom-churn-prediction.ipynb
```

# What This Project Demonstrates

-   Handling an imbalanced binary classification problem
-   Choosing evaluation metrics according to a business objective
-   Comparing linear and tree-based classifiers
-   Class balancing with sample weights
-   Feature selection using permutation importance
-   Training a reduced-feature gradient boosting model
-   Working with predicted probabilities rather than only hard class
    labels
-   Optimizing a decision threshold according to a precision-recall
    trade-off
-   Interpreting model performance from a business perspective
-   Recognizing the distinction between test-set evaluation and unbiased
    generalization estimates
