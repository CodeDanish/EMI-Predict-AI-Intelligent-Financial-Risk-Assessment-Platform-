import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import logging

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & LOGGING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Realtime Prediction | EMI Predict AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Custom CSS for Mobile & Tablet Responsiveness
st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3.2em;
        font-weight: bold;
    }
    .stDataFrame {
        width: 100%;
        overflow-x: auto;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #0d6efd;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔮 Realtime EMI Eligibility & Cap Predictor")
st.caption("Provide applicant details across demographic, financial, and loan request parameters to evaluate real-time credit risk.")

# -----------------------------------------------------------------------------
# 2. LOAD ARTIFACTS WITH CACHING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ml_artifacts():
    # Primary model path resolution
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    
    # Fallback to current directory if models/ not at root level
    if not os.path.exists(models_dir):
        models_dir = "models"
        
    try:
        scaler = joblib.load(os.path.join(models_dir, "../models/scaler.pkl"))
        ord_encoder = joblib.load(os.path.join(models_dir, "../models/ordinal_encoder.pkl"))
        ohe_encoder = joblib.load(os.path.join(models_dir, "../models/ohe_encoder.pkl"))
        label_encoder = joblib.load(os.path.join(models_dir, "../models/label_encoder.pkl"))
        cls_model = joblib.load(os.path.join(models_dir, "../models/classification_xgb.pkl"))
        reg_model = joblib.load(os.path.join(models_dir, "../models/regression_xgb.pkl"))
        return scaler, ord_encoder, ohe_encoder, label_encoder, cls_model, reg_model
    except Exception as e:
        st.error(f"❌ Error loading ML model artifacts from `{models_dir}`: {str(e)}")
        st.info("Ensure `scaler.pkl`, `ordinal_encoder.pkl`, `onehot_encoder.pkl`, `label_encoder.pkl`, `classification_xgb.pkl`, and `regression_xgb.pkl` exist in the `models/` directory.")
        st.stop()

scaler, ord_encoder, ohe_encoder, label_encoder, cls_model, reg_model = load_ml_artifacts()

# Default fillers for pipeline features not explicitly requested in UI
DEFAULT_FILLER_VALUES = {
    'monthly_rent': 0.0, 'school_fees': 0.0, 'college_fees': 0.0, 
    'travel_expenses': 0.0, 'groceries_utilities': 0.0, 'other_monthly_expenses': 0.0,
    'emergency_fund': 0.0, 'years_of_employment': 3, 'family_size': 1,
    'gender': 'Male', 'marital_status': 'Single', 'company_type': 'Private', 'house_type': 'Rented'
}

# -----------------------------------------------------------------------------
# 3. USER INPUT FORM (3-COLUMN RESPONSIVE LAYOUT)
# -----------------------------------------------------------------------------
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("👤 Demographics")
    age = st.slider("Age (Years)", 18, 70, 30, help="Applicant's current age.")
    education = st.selectbox("Education Level", ['High School', 'Graduate', 'Post Graduate', 'Professional'])
    employment_type = st.selectbox("Employment Type", ['Salaried', 'Self-Employed', 'Business Owner', 'Freelancer'])
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1, step=1)

with col2:
    st.subheader("💼 Financial Profile")
    monthly_salary = st.number_input("Gross Monthly Salary (₹)", min_value=5000, max_value=5000000, value=75000, step=5000)
    current_emi = st.number_input("Existing Monthly EMIs (₹)", min_value=0, max_value=2000000, value=10000, step=1000)
    total_expenses = st.number_input("Total Monthly Living Expenses (₹)", min_value=0, max_value=2000000, value=25000, step=2000)
    credit_score = st.slider("Credit Score (CIBIL/Equifax)", 300, 850, 750, help="Higher scores indicate lower credit risk.")

with col3:
    st.subheader("📝 Loan Request Details")
    requested_amount = st.number_input("Requested Loan Amount (₹)", min_value=10000, max_value=50000000, value=500000, step=25000)
    requested_tenure = st.slider("Requested Tenure (Months)", 3, 360, 36)
    emi_scenario = st.selectbox("EMI Scenario / Purpose", ['Personal Loan', 'Home Loan', 'Auto Loan', 'Education Loan', 'Consumer Durable'])

# -----------------------------------------------------------------------------
# 4. DERIVED METRICS SUMMARY DISPLAY
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Instant Derived Financial Ratios")

# Compute key ratios
disposable_income = monthly_salary - (total_expenses + current_emi)
dti_ratio = current_emi / (monthly_salary + 1e-6)
proposed_emi = requested_amount / max(requested_tenure, 1)
affordability_ratio = proposed_emi / max(disposable_income, 1e-6)
credit_risk_score = (850 - credit_score) / (850 - 300)

mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("Disposable Monthly Income", f"₹{disposable_income:,.2f}", delta="Positive Cash Flow" if disposable_income > 0 else "Negative Cash Flow", delta_color="normal" if disposable_income > 0 else "inverse")
mcol2.metric("Current DTI Ratio", f"{dti_ratio:.1%}", delta="High Risk" if dti_ratio > 0.5 else "Optimal", delta_color="inverse" if dti_ratio > 0.5 else "normal")
mcol3.metric("Proposed Loan EMI", f"₹{proposed_emi:,.2f}")
mcol4.metric("Affordability Ratio", f"{affordability_ratio:.2f}", delta="Heavy Strain" if affordability_ratio > 1.0 else "Manageable", delta_color="inverse" if affordability_ratio > 1.0 else "normal")

# -----------------------------------------------------------------------------
# 5. PREDICTION & INFERENCE PIPELINE
# -----------------------------------------------------------------------------
st.markdown("---")

if st.button("🚀 Evaluate Eligibility & Calculate EMI Cap", use_container_width=True):
    
    # --- Input Guardrails ---
    validation_errors = []
    if monthly_salary <= 0:
        validation_errors.append("Monthly Salary must be greater than zero.")
    if requested_amount <= 0:
        validation_errors.append("Requested Loan Amount must be greater than zero.")
    if requested_tenure <= 0:
        validation_errors.append("Tenure must be at least 1 month.")
    if (total_expenses + current_emi) > monthly_salary:
        st.warning("⚠️ Warning: Combined monthly obligations (Expenses + Current EMIs) exceed stated monthly salary.")

    if validation_errors:
        for err in validation_errors:
            st.error(f"❌ Input Validation Error: {err}")
        st.stop()

    # --- Processing Execution ---
    with st.status("🔄 Processing financial evaluation and running dual-model inference...", expanded=True) as status:
        try:
            status.write("📋 Assembling raw input schema...")
            
            # Construct dictionary with core features + interaction terms
            input_dict = {
                'age': age,
                'monthly_salary': monthly_salary,
                'credit_score': credit_score,
                'current_emi_amount': current_emi,
                'total_expenses': total_expenses,
                'requested_amount': requested_amount,
                'requested_tenure': requested_tenure,
                'dependents': dependents,
                'disposable_income': disposable_income,
                'dti_ratio': dti_ratio,
                'affordability_ratio': affordability_ratio,
                'credit_risk_score': credit_risk_score,
                'education': education,
                'employment_type': employment_type,
                'emi_scenario': emi_scenario,
                'is_negative_cash_flow': 1 if disposable_income < 0 else 0,
                'family_size': dependents + 1,
                'years_of_employment': DEFAULT_FILLER_VALUES['years_of_employment'],
                'eti_ratio': total_expenses / (monthly_salary + 1e-6),
                'proposed_emi_to_income_ratio': proposed_emi / (monthly_salary + 1e-6),
                'loan_to_annual_income_ratio': requested_amount / (monthly_salary * 12 + 1e-6),
                'credit_income_stability_interaction': credit_score * disposable_income,
                'emergency_fund_to_emi_ratio': (monthly_salary * 3) / max(proposed_emi, 1),
                'dti_dependents_interaction': dti_ratio * (dependents + 1)
            }
            
            input_raw = pd.DataFrame([input_dict])

            status.write("⚙️ Aligning and scaling features...")
            
            # Align Scaler Columns
            num_cols = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else list(input_raw.select_dtypes(include=[np.number]).columns)
            for col in num_cols:
                if col not in input_raw.columns:
                    input_raw[col] = DEFAULT_FILLER_VALUES.get(col, 0.0)

            # Align OneHot Encoder Columns
            cat_cols = list(ohe_encoder.feature_names_in_) if hasattr(ohe_encoder, 'feature_names_in_') else ['employment_type', 'emi_scenario']
            for col in cat_cols:
                if col not in input_raw.columns:
                    input_raw[col] = DEFAULT_FILLER_VALUES.get(col, 'Unknown')

            # Align Ordinal Encoder Columns
            ord_cols = list(ord_encoder.feature_names_in_) if hasattr(ord_encoder, 'feature_names_in_') else ['education']
            for col in ord_cols:
                if col not in input_raw.columns:
                    input_raw[col] = DEFAULT_FILLER_VALUES.get(col, 'Graduate')

            # Transform input attributes
            input_scaled = scaler.transform(input_raw[num_cols])
            input_ord = ord_encoder.transform(input_raw[ord_cols])
            input_ohe = ohe_encoder.transform(input_raw[cat_cols])
            
            # Feature matrix creation
            input_df = np.hstack([input_scaled, input_ord, input_ohe])

            status.write("🤖 Executing XGBoost models...")
            
            # 1. Classification Model Inference
            cls_pred_num = cls_model.predict(input_df)[0]
            if hasattr(label_encoder, 'inverse_transform'):
                eligibility = label_encoder.inverse_transform([cls_pred_num])[0]
            else:
                eligibility = str(cls_pred_num)
            
            # 2. Regression Model Inference
            raw_reg_pred = float(reg_model.predict(input_df)[0])
            max_emi_limit = max(0.0, raw_reg_pred)

            status.update(label="✅ Evaluation Completed Successfully!", state="complete", expanded=False)

        except Exception as e:
            status.update(label="❌ Pipeline Execution Failed", state="error", expanded=True)
            st.error(f"An error occurred during feature processing or inference: `{str(e)}`")
            logging.error(f"Prediction Pipeline Error: {str(e)}", exc_info=True)
            st.stop()

    # -------------------------------------------------------------
    # 6. RESULTS PRESENTATION
    # -------------------------------------------------------------
    st.markdown("### 🎯 Model Assessment Results")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.markdown("#### Loan Eligibility Status")
        if eligibility in ["Eligible", "Approved", 1]:
            st.success(f"### Status: Approved / Low Risk")
            st.write("✅ **Applicant meets all credit risk criteria for instant approval.**")
        elif eligibility in ["High_Risk", "Manual_Review", 2]:
            st.warning(f"### Status: High Risk / Underwriting Required")
            st.write("⚠️ **Applicant displays elevated DTI or financial strain ratios. Manual underwriting recommended.**")
        else:
            st.error(f"### Status: Rejected / High Default Risk")
            st.write("❌ **Applicant fails baseline financial stability requirements.**")

    with res_col2:
        st.markdown("#### Max Safe Monthly EMI Cap")
        st.info(f"### Recommended EMI Cap: ₹{max_emi_limit:,.2f} / month")
        
        if proposed_emi > max_emi_limit:
            st.error(f"⚠️ Requested EMI (₹{proposed_emi:,.2f}) exceeds recommended safe cap by ₹{(proposed_emi - max_emi_limit):,.2f}.")
        else:
            st.success(f"✅ Requested EMI (₹{proposed_emi:,.2f}) is within safe financial capability limits.")
