# Telco Customer Churn Prediction — Machine Learning Assignment 2
**Student:** Syed Amrez  
**Student ID:** 2026AJ05890

A Machine Learning project that predicts Telco customer churn using five classification algorithms. Includes data preprocessing, model comparison, performance evaluation, and a Streamlit dashboard web application for interactive evaluation of saved models on labelled test data, without retraining.


## a. Problem Statement

Customer churn occurs when a customer stops using a company's services. For telecommunications providers, identifying customers likely to churn can help guide retention efforts.

The objective of this project is to predict whether a customer will churn (`Yes`) or remain (`No`) using customer demographics, account details, and service information. Five classification algorithms are trained on the same dataset and evaluated using **Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC)**.

## b. Dataset Description

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

## c. GitHub Repository Link

**GitHub repository:** [Telco-Customer-Churn-Prediction-ML-Assignment-2](https://github.com/Syed-Amrez/Telco-Customer-Churn-Prediction-ML-Assignment-2)

The repository contains the Streamlit source code, dependency list, labelled test CSV, trained model pipelines, and project documentation.

## d. Models Used and Evaluation Metrics

The five classification models implemented on the same dataset are:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN) Classifier
4. Naive Bayes Classifier (Gaussian)
5. Support Vector Machine (SVM)

All five algorithms were evaluated on the **same held-out test set (1,409 records)**.

The evaluation metrics are:
- **Accuracy:** Overall proportion of correct predictions.
- **AUC:** Area under the ROC curve.
- **Precision:** Proportion of predicted churn cases that were actual churn cases.
- **Recall:** Proportion of actual churn cases correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.
- **MCC:** Matthews Correlation Coefficient, a balanced measure based on all four confusion-matrix outcomes.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0.8055** | **0.8419** | **0.6572** | 0.5588 | **0.6040** | **0.4790** |
| Decision Tree | 0.7289 | 0.6573 | 0.4896 | 0.5053 | 0.4974 | 0.3119 |
| K-Nearest Neighbors (KNN) | 0.7637 | 0.7892 | 0.5527 | 0.5749 | 0.5636 | 0.4018 |
| Naive Bayes | 0.6948 | 0.8074 | 0.4589 | **0.8369** | 0.5928 | 0.4245 |
| Support Vector Machine (SVM) | 0.7913 | 0.7905 | 0.6418 | 0.4840 | 0.5518 | 0.4264 |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Best overall balance: highest accuracy, AUC, precision, F1, and MCC among the five models. A strong choice for this dataset. |
| **Decision Tree** | Lowest AUC, F1, and MCC among the five models. Its performance on this test set was comparatively weaker. |
| **KNN** | Moderate results across metrics, with recall slightly higher than Logistic Regression, but lower overall F1 and MCC. |
| **Naive Bayes** | Highest recall, detecting the largest share of actual churners, but with low precision and accuracy; it produces more false-positive churn predictions. |
| **Support Vector Machine** | Second-highest accuracy and precision, but lower recall than Logistic Regression, resulting in a lower F1 score. |
| **Overall Winner** | **Logistic Regression**, based on its leading accuracy, AUC, F1, and MCC. **Naive Bayes** may be preferred if identifying as many churners as possible is more important than avoiding false alarms. |

### Overall Winner for the Dataset

**Logistic Regression is the overall winning model** based on these evaluation results. It achieved the best accuracy, AUC, F1 score, and MCC among the five models, indicating the strongest overall balance on the held-out test data.

Naive Bayes may be useful when identifying as many actual churn cases as possible is the primary objective, because it achieved the highest recall. However, this comes with lower precision and more false-positive predictions

### Best Model: Logistic Regression

For Logistic Regression, the confusion matrix on the held-out test set is:

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| **Actual No Churn** | 926 | 109 |
| **Actual Churn** | 165 | 209 |

The model correctly classified 926 non-churning customers and 209 churning customers. It missed 165 actual churners and incorrectly flagged 109 non-churners.

## Streamlit Application and Deployment

**Live Streamlit application:** [Telco-Customer-Churn-Prediction-ML-Assignment-2](https://telco-customer-churn-prediction-ml-assignment-2-znuukmjtw7rpq9.streamlit.app/)

The interactive application is built using Streamlit and deployed on Streamlit Community Cloud.

### Application Features

- **Project Overview:** problem statement, dataset, preprocessing, and model information.
- **Upload Dataset:** upload a labelled CSV or use the included `telco_test.csv`; preview records and dataset dimensions.
- **Model Analysis & Predictions:** compare all five models on the selected test data, choose one model from a dropdown, view the six evaluation metrics, inspect customer predictions, display a confusion matrix and classification report, and download prediction results.

> The uploaded CSV must include the original model input features and the actual `Churn` column (`Yes`/`No`).The application loads the saved model pipelines and evaluates them on the supplied test dataset. It does **not** retrain the models on uploaded test data.

## Repository Structure

```text
Telco-Customer-Churn-Prediction-ML-Assignment-2/
│
├── .streamlit/
│   └── config.toml
│
├── Dataset csv files/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── telco_test.csv
│
├── model/
│   ├── decision_tree.pkl
│   ├── k-nearest_neighbors.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   └── support_vector_machine.pkl
│
├── Train models source code.py
├── app.py
├── requirements.txt
├── telco_test.csv
└── README.md
```
## Conclusion

This project demonstrates an end-to-end machine learning classification workflow for predicting customer churn. Five classification algorithms were implemented and evaluated using the same held-out test dataset and six evaluation metrics.

Logistic Regression delivered the strongest overall results, while Naive Bayes achieved the highest recall. The saved preprocessing-and-model pipelines were integrated into a Streamlit dashboard, allowing users to upload test data, compare models, inspect evaluation results, and view customer-level predictions.

The project covers dataset preparation, leakage-aware model development, evaluation, interactive application development, and cloud deployment.

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

