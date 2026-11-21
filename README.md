# Financial Fraud Detection and Investigation Intelligence

An end-to-end data science project combining machine learning and business intelligence to investigate fraudulent financial transactions, identify risk patterns, and support evidence-based investigation prioritisation.

## 1. Business Problem

Financial institutions process millions of transactions, but fraud investigation teams have limited time and resources. Manually reviewing every transaction is impractical.

The challenge is therefore twofold:

- **Detection:** How can machine learning help distinguish potentially fraudulent transactions from legitimate ones when fraud is extremely rare?
- **Investigation prioritisation:** Where is fraudulent activity concentrated, what transaction characteristics are associated with it, and which areas deserve closer investigation?

This project addresses both challenges through predictive modelling and a supplementary fraud investigation analysis.

## 2. Project Objectives

- Analyse patterns in fraudulent and legitimate financial transactions.
- Investigate fraud concentration across transaction types, transaction values and time periods.
- Examine account balance movements associated with fraudulent transactions.
- Develop and compare machine learning classification models.
- Address severe class imbalance and evaluate models using appropriate metrics.
- Optimise the classification threshold and interpret model predictions using SHAP.
- Translate analytical findings into preliminary investigation priorities.

## 3. Dataset Overview

The project uses a synthetic financial transaction dataset.

| Dataset characteristic | Value |
|---|---:|
| Total transactions | 6,362,620 |
| Total original features | 11 |
| Legitimate transactions | 6,354,407 |
| Fraudulent transactions | 8,213 |
| Fraud prevalence | 0.1291% |
| Time steps | 743 |

The dataset contains transaction details such as transaction type, amount, sender and recipient account balances, time step, and fraud labels.

**Class imbalance is a central challenge:** only around 0.13% of transactions are labelled fraudulent.

## 4. Analytical Approach

### A. Exploratory Data Analysis & Feature Engineering

- Examined transaction distributions and fraud prevalence.
- Investigated fraud patterns across transaction types.
- Analysed transaction amounts and account balance movements.
- Created a log-transformed transaction amount feature using `log1p`.
- Derived destination account type from the recipient account identifier.
- Selected relevant categorical and numerical features for modelling.

### B. Data Preparation

- Separated independent variables and the fraud target.
- Used a stratified 80/20 training and testing split.
- Applied one-hot encoding to categorical variables.
- Preserved numerical features in the modelling pipeline.
- Used sparse matrices to handle the large encoded dataset efficiently.

### C. Machine Learning Models

Three classification algorithms were developed and evaluated:

- Logistic Regression
- Random Forest
- XGBoost

A majority-class baseline was also evaluated to demonstrate why accuracy alone is misleading for highly imbalanced classification.

### D. Model Evaluation & Optimisation

The evaluation included:

- Precision
- Recall
- F1 Score
- ROC-AUC
- Precision-Recall AUC (PR-AUC)
- Confusion Matrix

The XGBoost model was further evaluated using a validation dataset. The classification threshold was selected using validation predictions and then evaluated on the test dataset.

An eight-feature version of the XGBoost model was also compared with the original nine-feature version.

### E. Model Explainability

SHAP (SHapley Additive exPlanations) was used to investigate model behaviour through:

- SHAP summary analysis
- Individual prediction explanation
- SHAP waterfall visualisation

## 5. Model Performance

### Initial model comparison

| Model | Accuracy | Precision | Recall | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 73.26% | 0.43% | 89.59% | 0.9047 | 0.0162 |
| Random Forest | 88.48% | 0.82% | 73.40% | 0.9061 | 0.1836 |
| XGBoost | 85.30% | 0.71% | 80.77% | 0.9177 | 0.1894 |

### Threshold-optimised XGBoost

The selected classification threshold was **0.71438694**.

| Metric | Test Performance |
|---|---:|
| Accuracy | 90.92% |
| Precision | 1.03% |
| Recall | 72.67% |
| F1 Score | 0.0202 |
| ROC-AUC | 0.9169 |
| PR-AUC | 0.1882 |

**Interpretation:** The model identifies approximately 72.67% of labelled fraudulent transactions at the selected threshold. However, precision is only 1.03%, meaning that many transactions flagged by the model are not labelled as fraud. This indicates a substantial false-positive investigation workload.

The model should therefore be viewed as an experimental risk-ranking approach, not a production-ready fraud detection system.

### Why accuracy alone is misleading

The majority-class baseline achieved **99.87% accuracy**, but its precision and recall for fraud were both zero.

This demonstrates why fraud detection requires evaluation metrics that account for class imbalance, particularly precision, recall and PR-AUC.

## 6. Fraud Investigation & Business Intelligence Findings

The supplementary investigation analysis examined the full transaction dataset and identified several important patterns.

### Finding 1: Fraud is rare but financially significant

- 8,213 transactions were labelled fraudulent.
- Fraud prevalence was 0.1291%.
- The total value of transactions labelled fraudulent was approximately **12.06 billion**.

This is fraud-associated transaction value, not verified financial loss. It does not account for recoveries or actual economic impact.

### Finding 2: Fraud was concentrated in two transaction types

| Transaction type | Fraud cases | Observed fraud rate |
|---|---:|---:|
| TRANSFER | 4,097 | 0.7688% |
| CASH_OUT | 4,116 | 0.1840% |

These two transaction types accounted for all labelled fraud in this dataset.

TRANSFER had the higher observed fraud rate, while CASH_OUT had a substantially larger transaction volume. Both contributed approximately half of the fraud-associated transaction value.

### Finding 3: Existing fraud flagging had limited coverage

Only 16 out of 8,213 labelled fraudulent transactions were identified by the existing `isFlaggedFraud` indicator (approximately 0.195%).

This highlights limited coverage by that particular flag in the dataset.

### Finding 4: Fraud patterns varied across time-step volume categories

Low-volume time steps accounted for:

- 50.53% of fraudulent transactions.
- Approximately 52.52% of fraud-associated transaction value.

This suggests that investigation strategies focused only on high-volume periods could overlook suspicious activity.

However, these are simulated time steps, not actual calendar dates or business hours. The pattern requires further validation and should not be interpreted as a causal relationship.

### Finding 5: Fraudulent transactions showed different transaction and balance profiles

- Average fraudulent transaction amount was approximately 1.47 million, compared with 178,197 for legitimate transactions.
- Fraudulent transactions showed a substantially higher average reduction in sender balance.
- Recipient balance movements differed between fraudulent CASH_OUT and TRANSFER transactions.

These are aggregate statistical differences and do not establish that any individual transaction is fraudulent.

## 7. Investigation Priorities

Based on the observed patterns, the analysis suggests the following areas for further investigation:

1. Examine TRANSFER and CASH_OUT transactions more closely.
2. Consider both fraud frequency and transaction value when prioritising cases.
3. Investigate unusual sender balance reductions and recipient balance movements.
4. Avoid relying exclusively on high-volume periods when monitoring transaction activity.
5. Review the coverage of existing fraud flags and evaluate opportunities for improvement.

These are preliminary analytical recommendations, not validated operational rules.

## 8. Limitations

- **Synthetic data:** The dataset is simulated and may not represent real-world banking behaviour or fraud strategies.
- **Fraud-associated value:** The total value of labelled fraudulent transactions is not equivalent to confirmed financial loss.
- **Class imbalance:** Fraud represents only 0.1291% of observations, making model evaluation challenging.
- **False positives:** The threshold-optimised model has low precision, which could create a substantial investigation workload.
- **Time-step patterns:** Unusual concentrations may reflect the dataset's simulation process.
- **No causal conclusions:** Observed relationships are descriptive associations, not proof of causation.
- **Operational readiness:** The models and investigation priorities have not been validated in a live financial institution.

## 9. Tools & Technologies

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualisation:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Model Explainability:** SHAP
- **Development Environment:** Jupyter Notebook

## 10. Project Files

| File | Description |
|---|---|
| `Financial_Fraud_Detection.ipynb` | Data preparation, feature engineering, model development, evaluation, threshold optimisation and SHAP explainability |
| `Fraud_Loss_Investigation_Analysis.ipynb` | Fraud distribution, transaction analysis, time-step investigation, balance movements and business recommendations |

## 11. How to Run the Project

1. Clone or download this repository.
2. Install the required Python libraries.
3. Obtain the synthetic transaction dataset used in the notebooks.
4. Place `Synthetic_Financial_datasets_log.csv` in the same working directory as the notebooks.
5. Open the notebooks in Jupyter Notebook or JupyterLab.
6. Run the cells in sequence.

Install the main dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap jupyter
```

**Note:** The original CSV dataset is not included in this repository because of its large file size.

## 12. Conclusion

This project combines predictive modelling with investigative data analysis to explore financial fraud from both a machine learning and business perspective.

The machine learning work demonstrates the challenges of detecting rare fraudulent transactions, comparing classification algorithms, selecting a decision threshold and interpreting model predictions.

The investigation analysis identifies patterns across transaction types, transaction values, time periods and account balance movements, providing a preliminary framework for prioritising further investigation.

The key takeaway is that effective fraud analytics requires more than a high accuracy score. It requires balancing fraud detection, false-positive workload, financial exposure and the practical needs of investigation teams.
