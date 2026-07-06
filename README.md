# telecom-churn-prediction
Telecom customer churn prediction using HistGradientBoostingClassifier with class imbalance handling and business-driven threshold optimization.
# Telecom Churn Prediction with Class Imbalance Optimization

This project implements a machine learning workflow to predict customer churn in the telecommunications industry using the OpenML Churn dataset. The primary objective is to build a robust classifier that maximizes the identification of churning customers (**Recall**) while balancing cost-effectiveness for the business (**Precision**).

## Project Overview & Business Logic
In customer retention, missing a customer who is about to leave (False Negative) is much more expensive than offering a promotional discount to a loyal customer (False Positive). Therefore, the model's threshold was explicitly tuned to prioritize **Recall**, optimizing the trade-off so the company doesn't waste marketing budget on false alarms.

## Key Pipeline Steps
1. **Exploratory Data Analysis:** Identified a significant class imbalance (86% loyal, 14% churned).
2. **Feature Engineering & Cleaning:** Removed redundant data (e.g., total charges, which perfectly correlated with minutes call duration) and dropped non-informative identifiers.
3. **Data Splitting:** Applied stratified splitting (80/20) to maintain class proportions.
4. **Model Exploration:** Compared Logistic Regression, Random Forest, and HistGradientBoosting.
5. **Feature Selection:** Used Permutation Importance to isolate the top 8 features, reducing model noise and maintaining performance.
6. **Imbalance Mitigation:** Applied `compute_sample_weight` during the gradient boosting training phase and optimized the classification probability threshold.

## Results & Precision-Recall Optimization

By switching from a default baseline boosting model to a class-weighted model with a custom decision threshold ($0.15$), the pipeline achieved a significant business-driven optimization:

| Model Approach | Threshold | Recall (Class 1) | Precision (Class 1) | Business Impact |
| :--- | :---: | :---: | :---: | :--- |
| Default Unbalanced | 0.02 | **90%** | 40% | 60% false alarms; high budget waste |
| **Weighted + Tuned (Final)** | **0.15** | **88%** | **61%** | **Captures 88% of churn while reducing false alarms by 21%** |

The final model relies on the top 8 features, led by `total_day_minutes`, `number_customer_service_calls`, and `international_plan`.

## Repository Structure
* `notebook.ipynb` - Complete Jupyter Notebook with data processing, modeling, and evaluation.
* `README.md` - Project documentation.

## Technical Stack
* Python
* Pandas, NumPy
* Scikit-Learn (HistGradientBoostingClassifier, Permutation Importance)
