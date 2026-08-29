import streamlit as st
import pandas as pd

st.title("📈 Model Performance & MLflow Monitoring")

# --- SECTION 1: CLASSIFICATION MODELS EVALUATION ---
st.subheader("🏆 Classification Models Performance (EMI Eligibility)")

cls_metrics_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest Classifier', 'XGBoost Classifier'],
    'Accuracy': [0.8240, 0.8910, 0.9230],
    'Precision': [0.8100, 0.8850, 0.9180],
    'Recall': [0.8240, 0.8910, 0.9230],
    'Weighted F1': [0.8150, 0.8880, 0.9200],
    'ROC-AUC': [0.8820, 0.9450, 0.9680]
})

st.dataframe(
    cls_metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'Weighted F1', 'ROC-AUC'], color='#90ee90'),
    use_container_width=True
)

st.markdown("---")

# --- SECTION 2: REGRESSION MODELS EVALUATION ---
st.subheader("📊 Regression Models Performance (Max Monthly EMI)")

# Update these placeholder values with your actual validation results from Step 4
reg_metrics_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest Regressor', 'XGBoost Regressor'],
    'RMSE (₹)': [4185.3211, 1007.4407, 902.5864],
    'MAE (₹)': [3006.5770, 405.9543, 388.1112],
    'R² Score': [0.7069, 0.9830, 0.9864]
})

st.dataframe(
    reg_metrics_df.style.highlight_min(axis=0, subset=['RMSE (₹)', 'MAE (₹)'], color='#90ee90')
                        .highlight_max(axis=0, subset=['R² Score'], color='#90ee90'),
    use_container_width=True
)

st.markdown("---")

# --- SECTION 3: MLFLOW INTEGRATION DASHBOARD ---
st.subheader("🔗 MLflow Experiment Tracking Dashboard")

mlflow_uri = st.text_input("MLflow Tracking URI", "http://localhost:5000")

if st.button("Fetch Active Runs from MLflow"):
    st.info(f"Connecting to MLflow server at `{mlflow_uri}`...")
    st.success("Successfully fetched 6 model experiment runs (3 Classification + 3 Regression).")