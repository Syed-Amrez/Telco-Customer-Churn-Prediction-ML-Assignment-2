# Telco Customer Churn Prediction — Machine Learning Assignment 2
**Student:** Syed Amrez  
**Student ID:** 2026AJ05890

A Machine Learning project that predicts Telco customer churn using five classification algorithms. Includes data preprocessing, model comparison, performance evaluation, and a Streamlit dashboard web application for interactive evaluation of saved models on labelled test data, without retraining.


## Problem Statement

Customer churn occurs when a customer stops using a company's services. For telecommunications providers, identifying customers likely to churn can help guide retention efforts.

The objective of this project is to predict whether a customer will churn (`Yes`) or remain (`No`) using customer demographics, account details, and service information. Five classification algorithms are trained on the same dataset and evaluated using **Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC)**.

## Dataset Description

**Dataset:** IBM Telco Customer Churn  
**Public source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Description |
|---|---|
| Task | Binary classification |
| Total records | 7,043 customers |
| Original columns | 21 |
| Input features | 19 after excluding `customerID` and target `Churn` |
| Target | `Churn` (`Yes` or `No`) |
| Train/test split | 80% training / 20% testing |
| Test-set size | 1,409 records |
| Split strategy | Stratified, `random_state=42` |

The dataset includes information about customer tenure, contract type, internet service, payment method, monthly charges, total charges, and other services.

### Preprocessing and data leakage prevention

1. Excluded `customerID` because it is an identifier, not a predictive feature.
2. Separated the target column `Churn` from the input features.
3. Converted `TotalCharges` to numeric; invalid or blank values become missing values.
4. Split the data into training and testing sets **before fitting preprocessing**.
5. Imputed missing numerical values using the training-set median and scaled numerical features using `StandardScaler`.
6. Imputed missing categorical values using the training-set most frequent value and encoded categories using `OneHotEncoder(handle_unknown="ignore")`.
7. Kept preprocessing inside each saved scikit-learn model pipeline to avoid leakage from the test data.

The deployed application loads the saved pipelines and evaluates them on the supplied labelled CSV; **it does not retrain or tune models during prediction**.

## GitHub Repository Link

**GitHub repository:** [Telco-Customer-Churn-Prediction-ML-Assignment-2](https://github.com/Syed-Amrez/Telco-Customer-Churn-Prediction-ML-Assignment-2)

The repository contains the Streamlit source code, dependency list, labelled test CSV, trained model pipelines, and project documentation.

## Models Used and Evaluation Metrics

All five algorithms were evaluated on the **same held-out test set (1,409 records)**.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0.8055** | **0.8419** | **0.6572** | 0.5588 | **0.6040** | **0.4790** |
| Decision Tree | 0.7289 | 0.6573 | 0.4896 | 0.5053 | 0.4974 | 0.3119 |
| K-Nearest Neighbors (KNN) | 0.7637 | 0.7892 | 0.5527 | 0.5749 | 0.5636 | 0.4018 |
| Naive Bayes | 0.6948 | 0.8074 | 0.4589 | **0.8369** | 0.5928 | 0.4245 |
| Support Vector Machine (SVM) | 0.7913 | 0.7905 | 0.6418 | 0.4840 | 0.5518 | 0.4264 |

*Scores are rounded to four decimal places. Precision, Recall, and F1 are calculated for the positive class (`Churn = Yes`). AUC is based on model scores or probabilities.*

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Best overall balance: highest accuracy, AUC, precision, F1, and MCC among the five models. A strong choice for this dataset. |
| **Decision Tree** | Lowest AUC, F1, and MCC. Its individual predictions are interpretable, but this model generalizes less effectively than the alternatives on the held-out test set. |
| **KNN** | Moderate results across metrics, with recall slightly higher than Logistic Regression, but lower overall F1 and MCC. |
| **Naive Bayes** | Highest recall (0.8369), detecting the largest share of actual churners, but with low precision and accuracy; it produces more false-positive churn predictions. |
| **Support Vector Machine** | Second-highest accuracy and precision, but lower recall than Logistic Regression, resulting in a lower F1 score. |
| **Overall Winner** | **Logistic Regression**, based on its leading accuracy, AUC, F1, and MCC. **Naive Bayes** may be preferred if identifying as many churners as possible is more important than avoiding false alarms. |

### Best Model: Logistic Regression

For Logistic Regression, the confusion matrix on the held-out test set is:

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| **Actual No Churn** | 926 | 109 |
| **Actual Churn** | 165 | 209 |

The model correctly classified 926 non-churning customers and 209 churning customers. It missed 165 actual churners and incorrectly flagged 109 non-churners.

## Streamlit Web Application

The interactive dashboard provides:

- **Project Overview:** problem statement, dataset, preprocessing, and model information.
- **Dataset:** upload a labelled CSV or use the included `telco_test.csv`; preview records and dataset dimensions.
- **Model Analysis & Predictions:** compare all five models on the selected test data, choose one model from a dropdown, view the six evaluation metrics, inspect customer predictions, display a confusion matrix and classification report, and download prediction results.

**Live Streamlit application:** [Telco-Customer-Churn-Prediction-ML-Assignment-2](https://telco-customer-churn-prediction-ml-assignment-2-znuukmjtw7rpq9.streamlit.app/)

> The uploaded CSV must include the original model input features and the actual `Churn` column (`Yes`/`No`). The app evaluates previously trained pipelines and does not fit them on uploaded test data.

## Repository Structure

```text
Telco-Customer-Churn-Prediction-ML-Assignment-2/
├── app.py
├── requirements.txt
├── README.md
├── telco_test.csv
├── .streamlit/
│   └── config.toml
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── k-nearest_neighbors.pkl
    ├── naive_bayes.pkl
    └── support_vector_machine.pkl
```

## Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/Syed-Amrez/Telco-Customer-Churn-Prediction-ML-Assignment-2.git
   cd Telco-Customer-Churn-Prediction-ML-Assignment-2
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application:

   ```bash
   streamlit run app.py
   ```

4. Open the local URL shown by Streamlit, usually `http://localhost:8501`.

The project uses **Streamlit 1.58.0** to keep the local and deployed tab styling consistent. The trained models were saved as `.pkl` pipelines and loaded with `joblib`.

## Deployment

The application is deployed using **Streamlit Community Cloud** from the GitHub repository. The deployment uses the `main` branch and `app.py` as the entry point.

The deployment environment installs dependencies from `requirements.txt` and loads the saved pipelines in the `model/` folder.

