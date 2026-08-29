import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🔮 Real-Time EMI Risk & Capacity Prediction")

# Load artifacts (Cache to avoid reloading on user interactions)
@st.cache_resource
def load_artifacts():
    # Replace with path to your fitted models
    scaler = joblib.load('models/scaler.pkl')
    ord_encoder = joblib.load('models/ordinal_encoder.pkl')
    ohe_encoder = joblib.load('models/ohe_encoder.pkl')
    le = joblib.load('models/label_encoder.pkl')
    cls_model = joblib.load('models/classification_xgb.pkl')
    reg_model = joblib.load('models/regression_xgb.pkl')
    return scaler, ord_encoder, ohe_encoder, le, cls_model, reg_model

# Load artifacts
scaler, ord_encoder, ohe_encoder, label_encoder, cls_model, reg_model = load_artifacts()

# Helper for layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Personal & Demographics")
    age = st.slider("Age", 21, 65, 30)
    education = st.selectbox("Education Level", ['High School', 'Graduate', 'Post Graduate', 'Professional'])
    employment_type = st.selectbox("Employment Type", ['Salaried', 'Self-Employed', 'Business'])
    dependents = st.number_input("Dependents", 0, 10, 1)

with col2:
    st.subheader("💼 Financial Profile")
    monthly_salary = st.number_input("Monthly Salary (₹)", 10000, 1000000, 65000, step=5000)
    credit_score = st.slider("Credit Score", 300, 850, 750)
    current_emi = st.number_input("Current Monthly EMIs (₹)", 0, 500000, 10000)
    total_expenses = st.number_input("Monthly Living Expenses (₹)", 5000, 500000, 25000)

with col3:
    st.subheader("📝 Loan Details")
    emi_scenario = st.selectbox("Lending Scenario", ['Personal Loan', 'Vehicle Loan', 'Home Loan', 'E-commerce', 'Appliance'])
    requested_amount = st.number_input("Requested Loan Amount (₹)", 10000, 10000000, 300000, step=10000)
    requested_tenure = st.slider("Requested Tenure (Months)", 3, 360, 36)

st.markdown("---")

if st.button("🚀 Evaluate Eligibility & Max EMI", use_container_width=True):
    # 1. Feature Engineering On-The-Fly
    disposable_income = monthly_salary - (total_expenses + current_emi)
    dti_ratio = current_emi / (monthly_salary + 1e-6)
    proposed_emi = requested_amount / max(requested_tenure, 1)
    affordability_ratio = proposed_emi / max(disposable_income, 1e-6)
    credit_risk_score = (850 - credit_score) / (850 - 300)
    
    # 2. Define Feature Lists matching the training set
    # 1. Recreate complete numerical feature list matching training fit
    numerical_features = list(scaler.feature_names_in_)

    # 2. Build input dictionary containing both raw and calculated metrics
    input_dict = {
        # Expenses & Core Metrics
        'total_expenses': total_expenses,
        'current_emi_amount': current_emi,
        'monthly_salary': monthly_salary,
        'disposable_income': disposable_income,
        'requested_amount': requested_amount,
        'requested_tenure': requested_tenure,
        
        # Raw Demographics & Income
        'age': age,
        'monthly_salary': monthly_salary,
        'credit_score': credit_score,
        'current_emi_amount': current_emi,
        'requested_amount': requested_amount,
        'requested_tenure': requested_tenure,
        'dependents': dependents,
        'family_size': dependents + 1,
        'years_of_employment': 3,  # default or UI input
        
        # Raw Expenses (fill defaults if not individual UI inputs)
        'monthly_rent': total_expenses * 0.4,
        'school_fees': 0,
        'college_fees': 0,
        'travel_expenses': total_expenses * 0.1,
        'groceries_utilities': total_expenses * 0.3,
        'other_monthly_expenses': total_expenses * 0.2,
        'emergency_fund': monthly_salary * 3,
        'is_negative_cash_flow': 1 if disposable_income < 0 else 0,
        
        # Derived Ratios & Scores
        'disposable_income': disposable_income,
        'dti_ratio': dti_ratio,
        'affordability_ratio': affordability_ratio,
        'credit_risk_score': credit_risk_score,
        'eti_ratio': total_expenses / (monthly_salary + 1e-6),
        'proposed_emi_to_income_ratio': proposed_emi / (monthly_salary + 1e-6),
        'employment_stability_score': 0.7,
        'financial_stability_index': 0.7,
        'loan_to_annual_income_ratio': requested_amount / (monthly_salary * 12 + 1e-6),
        'credit_income_stability_interaction': credit_score * disposable_income,
        'emergency_fund_to_emi_ratio': (monthly_salary * 3) / max(proposed_emi, 1),
        'dti_dependents_interaction': dti_ratio * (dependents + 1)
    }

    # Add categorical entries
    input_dict['education'] = education
    input_dict['employment_type'] = employment_type
    input_dict['emi_scenario'] = emi_scenario
    input_dict['gender'] = 'Male'
    input_dict['marital_status'] = 'Single'
    input_dict['company_type'] = 'Private'
    input_dict['house_type'] = 'Rented'

    # Convert to DataFrame
    input_raw = pd.DataFrame([input_dict])

    # Get expected categorical features directly from encoders
    categorical_nominal = list(ohe_encoder.feature_names_in_) if hasattr(ohe_encoder, 'feature_names_in_') else ['employment_type', 'emi_scenario']
    categorical_ordinal = list(ord_encoder.feature_names_in_) if hasattr(ord_encoder, 'feature_names_in_') else ['education']

    # 3. Scale using the exact feature names fitted on scaler
    input_scaled = scaler.transform(input_raw[scaler.feature_names_in_])
    input_ord = ord_encoder.transform(input_raw[['education']])
    input_ohe = ohe_encoder.transform(input_raw[categorical_nominal])

    # Combine into a single feature array for model inference
    input_df = np.hstack([input_scaled, input_ord, input_ohe])

    # 5. Generate actual model predictions
    cls_pred_num = cls_model.predict(input_df)[0]
    eligibility = label_encoder.inverse_transform([cls_pred_num])[0]
    max_emi_limit = float(reg_model.predict(input_df)[0])

    # 6. Display Results in UI
    st.markdown("### 🎯 Model Assessment Results")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if eligibility == "Eligible":
            st.success(f"**Eligibility Status:** {eligibility}")
        elif eligibility == "High_Risk":
            st.warning(f"**Eligibility Status:** {eligibility}")
        else:
            st.error(f"**Eligibility Status:** {eligibility}")
            
    with res_col2:
        st.info(f"**Max Safe Monthly EMI Cap:** ₹{max(0.0, max_emi_limit):,.2f}")