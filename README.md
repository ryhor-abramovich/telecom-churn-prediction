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

By shifting from a default baseline model to a class-weighted Gradient Boosting architecture with a custom decision threshold ($0.15$), the pipeline achieved a massive business-driven optimization:

| Model / Approach | Decision Threshold | Recall (Class 1) | Precision (Class 1) | F1-Score | Global Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Vanilla Logistic Regression (Baseline) | 0.50 | 20% | 62% | 0.30 | 0.87 |
| 2. Balanced Logistic Regression | 0.50 | 71% | 35% | 0.47 | 0.78 |
| 3. HistGradientBoosting (All features) | 0.50 | 77% | 92% | 0.83 | 0.95 |
| 4. **Slim HGB + Custom Threshold (Final)** | **0.15** | **88%** | **61%** | **0.72** | **0.90** |

**Business Impact:** The final optimized pipeline captures **88% of all at-risk customers** (retaining critical subscriber revenue), while maintaining an optimal Precision of 61% to prevent the marketing department from wasting retention budget on false alarms.


## Repository Structure
* `notebook.ipynb` - Complete Jupyter Notebook with data processing, modeling, and evaluation.
* `README.md` - Project documentation.

## 🛠️ Technical Stack
* **Language:** Python 3
* **Machine Learning:** Scikit-Learn (`HistGradientBoostingClassifier`, `RandomForestClassifier`, `LogisticRegression`, `permutation_importance`, `compute_sample_weight`)
* **Data Pipelines:** Pandas, NumPy, OpenML API
