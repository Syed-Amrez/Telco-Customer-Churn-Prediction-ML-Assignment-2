
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: var(--background-color);
}
/* Reduce space above assignment information */
.block-container {
    padding-top: 1rem !important;
}

/* Fine-tune assignment information spacing */
.assignment-info {
    margin-top: 30px !important;
    margin-bottom: 12px !important;
}

/* Neon heading tile */
.neon-header {
    background: linear-gradient(120deg, #101a35, #142b48);
    border: 1px solid #00e5ff;
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 0 12px #00e5ff,
                0 0 25px rgba(0,229,255,0.25);
    margin-bottom: 18px;
}

.neon-header h1 {
    color: #ffffff;
    font-size: 34px;
    margin-bottom: 8px;
    text-shadow: 0 0 8px #00e5ff;
}

.neon-header p {
    color: #67e8f9;
    font-size: 16px;
    margin: 0;
}

/* Readiness message */
.ready-message {
    background: var(--secondary-background-color);
    border-left: 4px solid #00e5ff;
    padding: 13px;
    border-radius: 7px;
    color: var(--text-color);
    margin-bottom: 18px;
}

/* Section headings */
.section-title {
    color: var(--text-color);
    border-bottom: 2px solid #00e5ff;
    padding-bottom: 8px;
    margin-top: 15px;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid #24506b;
    border-radius: 10px;
    padding: 12px;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    border-bottom: 1px solid #24506b;
}

.stTabs [data-baseweb="tab"] {
    background-color: var(--secondary-background-color);
    border-radius: 8px 8px 0 0;
    padding: 12px 20px;
    color: var(--text-color);
}

.stTabs [aria-selected="true"] {
    background-color: #087e91;
    color: white;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border: 1px solid #24506b;
    border-radius: 8px;
}

/* Keep readiness-message text readable in both themes */
.ready-message * {
    color: var(--text-color) !important;
}

/* Keep active navigation-tab text readable */
.stTabs [aria-selected="true"] p {
    color: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================
# Assignment information - simple left-aligned text
st.markdown("""
<div class="assignment-info" style="text-align:left; line-height:1.6;">
    <div style="font-size:19px; font-weight:bold; color:var(--primary-color);">
        Machine Learning – Assignment 2
    </div>
    <div style="font-size:15px; color:var(--text-color);">
        Student Name: Syed Amrez
    </div>
    <div style="font-size:15px; color:var(--text-color);;">
        Student ID: 2026AJ05890
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="neon-header">
    <h1>📊 Telco Customer Churn Prediction</h1>
    <p>Machine Learning Classification Dashboard using Five ML Algorithms</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ready-message">
    <b>Dashboard ready!</b><br>
    Your trained machine learning models are ready.
    Explore the project overview, inspect the test dataset,
    or analyse customer churn predictions.
</div>
""", unsafe_allow_html=True)

# ==========================================================
# MODEL FILES
# ==========================================================

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "k-nearest_neighbors.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl"
}

DEFAULT_DATASET = "telco_test.csv"

# ==========================================================
# LOAD SAVED MODELS
# ==========================================================

@st.cache_resource
def load_models():

    loaded_models = {}

    for name, filename in MODEL_FILES.items():

        model_path = os.path.join("model", filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        loaded_models[name] = joblib.load(model_path)

    return loaded_models


try:
    models = load_models()

except Exception as e:

    st.error(f"Unable to load machine learning models: {e}")
    st.stop()


# ==========================================================
# THREE HORIZONTAL NAVIGATION TABS
# ==========================================================

tab_overview, tab_dataset, tab_analysis = st.tabs([
    "📘 Project Overview",
    "📂 Dataset",
    "📊 Model Analysis & Predictions"
])


# ==========================================================
# TAB 1: PROJECT OVERVIEW
# ==========================================================

with tab_overview:

    st.markdown(
        '<h2 class="section-title">Project Overview</h2>',
        unsafe_allow_html=True
    )

    st.subheader("Description")

    st.write("""
    Customer churn refers to customers discontinuing their
    subscription or services.

    This project uses machine learning classification algorithms
    to predict whether a telecommunications customer is likely
    to leave the service.
    """)

    st.subheader("Objective")

    st.write("""
    The objective is to compare five classification algorithms,
    evaluate their predictive performance and identify suitable
    models for customer churn prediction.
    """)

    st.subheader("Dataset Information")

    st.write("""
    The project uses the Telco Customer Churn dataset.
    It contains customer demographics, account information,
    subscribed services and customer churn status.
    """)

    st.markdown(
        "[Dataset Source – Kaggle: Telco Customer Churn]"
        "(https://www.kaggle.com/datasets/blastchar/telco-customer-churn)"
    )

    st.subheader("Target Variable")

    st.table(pd.DataFrame({
        "Target Value": ["Yes", "No"],
        "Meaning": [
            "Customer discontinued the service (Churn)",
            "Customer continued the service (No Churn)"
        ]
    }))

    st.subheader("Machine Learning Algorithms")

    st.markdown("""
    1. Logistic Regression
    2. Decision Tree
    3. K-Nearest Neighbors (KNN)
    4. Naive Bayes
    5. Support Vector Machine (SVM)
    """)

    st.subheader("Removed Columns and Reasons")

    st.table(pd.DataFrame({
        "Column": [
            "customerID",
            "Churn from input features"
        ],
        "Reason": [
            "Unique customer identifier; it does not provide meaningful predictive behaviour.",
            "Target variable. It is retained separately for evaluation and is not used as an input feature."
        ]
    }))

    st.subheader("Data Preprocessing")

    st.markdown("""
    **Numerical Features**
    - Missing values are replaced using median imputation.
    - StandardScaler is used to scale numerical features.

    **Categorical Features**
    - Missing values are replaced using the most frequent value.
    - OneHotEncoder converts categorical values into numerical features.
    """)

    st.subheader("Train-Test Split")

    st.write("""
    The dataset is divided into 80% training data and 20% testing
    data. Stratified splitting maintains the proportion of churn
    and non-churn classes in both sets.
    """)

    st.subheader("Data Leakage Prevention")

    st.write("""
    The dataset is split before preprocessing is fitted.
    Imputation values, scaling parameters and encoding categories
    are learned from training data only.

    The held-out test dataset is used for evaluation.
    Uploaded datasets are never used to refit or tune the
    saved machine learning models.
    """)


# ==========================================================
# TAB 2: DATASET
# ==========================================================

with tab_dataset:

    st.markdown(
        '<h2 class="section-title">Dataset</h2>',
        unsafe_allow_html=True
    )

    # ------------------------------------------------------
    # DATASET UPLOAD
    # ------------------------------------------------------

    st.subheader("Upload Customer Dataset")

    st.info("""
    Upload a labelled test CSV containing the actual Churn
    column. The application evaluates the saved models on
    the uploaded records without retraining or tuning them.
    """)

    uploaded_file = st.file_uploader(
        "Choose a labelled Telco test CSV file",
        type=["csv"],
        key="dataset_upload"
    )

    # ------------------------------------------------------
    # LOAD DEFAULT OR UPLOADED DATASET
    # ------------------------------------------------------

    try:

        if uploaded_file is not None:

            input_df = pd.read_csv(uploaded_file)

            st.success("Uploaded dataset loaded successfully.")

        else:

            if not os.path.exists(DEFAULT_DATASET):

                st.error(
                    "Default telco_test.csv was not found. "
                    "Please upload a labelled test CSV."
                )

                st.stop()

            input_df = pd.read_csv(DEFAULT_DATASET)

            st.info("Currently displaying the default test dataset.")

    except Exception as e:

        st.error(f"Unable to read dataset: {e}")
        st.stop()

    # ------------------------------------------------------
    # VALIDATE TARGET COLUMN
    # ------------------------------------------------------

    if "Churn" not in input_df.columns:

        st.error(
            "The uploaded dataset must contain a Churn column "
            "with actual Yes/No labels."
        )

        st.stop()

    y_actual = input_df["Churn"].map({
        "No": 0,
        "Yes": 1,
        0: 0,
        1: 1
    })

    if y_actual.isna().any():

        st.error(
            "Invalid values found in Churn. "
            "Only Yes and No labels are supported."
        )

        st.stop()

    # ------------------------------------------------------
    # VALIDATE MODEL FEATURES
    # ------------------------------------------------------

    first_model = next(iter(models.values()))

    if not hasattr(first_model, "feature_names_in_"):

        st.error(
            "The saved model does not expose its expected "
            "input feature names."
        )

        st.stop()

    expected_features = list(first_model.feature_names_in_)

    prediction_df = input_df.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    ).copy()

    missing_features = [
        feature for feature in expected_features
        if feature not in prediction_df.columns
    ]

    if missing_features:

        st.error(
            "The dataset is missing required features: "
            + ", ".join(missing_features)
        )

        st.stop()

    prediction_df = prediction_df[expected_features]

    # ------------------------------------------------------
    # DATASET PREVIEW
    # ------------------------------------------------------

    st.subheader("Dataset Preview")

    row_col, column_col, feature_col = st.columns(3)

    with row_col:
        st.metric(
            "Total Rows",
            f"{input_df.shape[0]:,}"
        )

    with column_col:
        st.metric(
            "Total Columns",
            input_df.shape[1]
        )

    with feature_col:
        st.metric(
            "Prediction Features",
            prediction_df.shape[1]
        )

    st.write("Preview of the selected dataset")

    st.dataframe(
        input_df.head(10),
        use_container_width=True,
        height=450
    )

    st.caption(
        "The uploaded dataset replaces the default dataset "
        "for the current application session."
    )


# ==========================================================
# COMMON DATA PREPARATION FOR MODEL ANALYSIS
# ==========================================================

# This section runs independently of the selected tab.
# It ensures the analysis tab has access to the current dataset.

if uploaded_file is None:

    if not os.path.exists(DEFAULT_DATASET):

        st.warning(
            "Upload a labelled test dataset in the Dataset tab "
            "to begin model analysis."
        )

        st.stop()

    try:
        input_df = pd.read_csv(DEFAULT_DATASET)

    except Exception as e:

        st.error(f"Unable to load default dataset: {e}")
        st.stop()


# Validate target column

if "Churn" not in input_df.columns:

    st.error("Dataset must contain the Churn target column.")
    st.stop()


y_actual = input_df["Churn"].map({
    "No": 0,
    "Yes": 1,
    0: 0,
    1: 1
})


if y_actual.isna().any():

    st.error("Churn must contain valid Yes/No labels.")
    st.stop()


# Expected feature columns

first_model = next(iter(models.values()))

expected_features = list(first_model.feature_names_in_)

prediction_df = input_df.drop(
    columns=["customerID", "Churn"],
    errors="ignore"
).copy()


missing_features = [
    feature for feature in expected_features
    if feature not in prediction_df.columns
]


if missing_features:

    st.error(
        "Missing required model features: "
        + ", ".join(missing_features)
    )

    st.stop()


prediction_df = prediction_df[expected_features]


# ==========================================================
# GENERATE PREDICTIONS FOR ALL FIVE MODELS
# ==========================================================

all_predictions = {}
all_scores = {}
comparison_results = []


for name, model in models.items():

    try:

        y_pred = model.predict(prediction_df)

        if hasattr(model, "predict_proba"):

            y_score = model.predict_proba(
                prediction_df
            )[:, 1]

        else:

            y_score = model.decision_function(
                prediction_df
            )

        all_predictions[name] = y_pred
        all_scores[name] = y_score

        comparison_results.append({

            "Model": name,

            "Accuracy": accuracy_score(
                y_actual, y_pred
            ),

            "AUC": roc_auc_score(
                y_actual, y_score
            ),

            "Precision": precision_score(
                y_actual,
                y_pred,
                zero_division=0
            ),

            "Recall": recall_score(
                y_actual,
                y_pred,
                zero_division=0
            ),

            "F1 Score": f1_score(
                y_actual,
                y_pred,
                zero_division=0
            ),

            "MCC": matthews_corrcoef(
                y_actual,
                y_pred
            )
        })

    except Exception as e:

        st.error(
            f"Prediction failed for {name}: {e}"
        )

        st.stop()


results_df = pd.DataFrame(comparison_results)


# ==========================================================
# TAB 3: MODEL ANALYSIS AND PREDICTIONS
# ==========================================================

with tab_analysis:

    st.markdown(
        '<h2 class="section-title">Model Analysis & Predictions</h2>',
        unsafe_allow_html=True
    )

    # ------------------------------------------------------
    # MODEL COMPARISON
    # ------------------------------------------------------

    st.subheader("Model Predictions – Model Comparison")

    st.write("""
    The following table compares the performance of all
    five trained machine learning algorithms on the
    currently selected test dataset.
    """)

    st.dataframe(
        results_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    # ------------------------------------------------------
    # SINGLE MODEL DROPDOWN
    # ------------------------------------------------------

    st.subheader("Choose a Model for Detailed Analysis")

    selected_model = st.selectbox(
        "Select Machine Learning Model",
        list(models.keys()),
        key="selected_model"
    )

    selected_predictions = all_predictions[selected_model]

    selected_row = results_df[
        results_df["Model"] == selected_model
    ].iloc[0]

    # ------------------------------------------------------
    # SIX PERFORMANCE METRICS
    # ------------------------------------------------------

    st.subheader("Model Evaluation – Performance Metrics")

    metric_names = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ]

    metric_columns = st.columns(3)

    for index, metric in enumerate(metric_names):

        with metric_columns[index % 3]:

            st.metric(
                label=metric,
                value=f"{selected_row[metric]:.4f}"
            )

    # ------------------------------------------------------
    # CUSTOMER PREDICTIONS
    # ------------------------------------------------------

    st.subheader("Customer Predictions")

    customer_results = input_df.copy()

    customer_results["Actual Churn"] = y_actual.map({
        0: "No",
        1: "Yes"
    }).values

    customer_results["Predicted Churn"] = np.where(
        selected_predictions == 1,
        "Yes",
        "No"
    )

    st.dataframe(
        customer_results,
        use_container_width=True,
        height=400
    )

    download_data = customer_results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Customer Predictions",
        data=download_data,
        file_name="telco_churn_predictions.csv",
        mime="text/csv"
    )

    # ------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_actual,
        selected_predictions,
        labels=[0, 1]
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual No Churn",
            "Actual Churn"
        ],
        columns=[
            "Predicted No Churn",
            "Predicted Churn"
        ]
    )

    heatmap_col, table_col = st.columns(2)

    with heatmap_col:

        st.write("Confusion Matrix Heatmap")

        fig, ax = plt.subplots(figsize=(5, 4))

        image = ax.imshow(
            cm,
            cmap="Blues"
        )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels([
            "No Churn",
            "Churn"
        ])

        ax.set_yticklabels([
            "No Churn",
            "Churn"
        ])

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        ax.set_title(
            f"{selected_model} Confusion Matrix"
        )

        for i in range(2):

            for j in range(2):

                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=13
                )

        fig.colorbar(
            image,
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)

    with table_col:

        st.write("Confusion Matrix Table")

        st.dataframe(
            cm_df,
            use_container_width=True
        )

    # ------------------------------------------------------
    # CLASSIFICATION REPORT
    # ------------------------------------------------------

    st.subheader("Classification Report")

    report = classification_report(
        y_actual,
        selected_predictions,
        target_names=[
            "No Churn",
            "Churn"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(4),
        use_container_width=True
    )

    st.success(
        f"Analysis completed successfully using {selected_model}."
    )

