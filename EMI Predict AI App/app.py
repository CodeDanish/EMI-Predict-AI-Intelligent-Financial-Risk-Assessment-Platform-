import streamlit as st

st.set_page_config(
    page_title="EMIPredict AI Platform",
    page_icon="🏦",
    layout="wide",  # Enables fluid full-width responsiveness
    initial_sidebar_state="auto"  # Collapses sidebar automatically on mobile screens
)

st.title("🏦 EMIPredict AI Platform")
st.subheader("Financial Risk Assessment & Safe EMI Cap Estimation")

# Custom CSS injection for cross-device mobile touch optimization
st.markdown("""
    <style>
    /* Make buttons and selectboxes easier to tap on mobile devices */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
    }
    /* Ensure tables scroll horizontally on small screens */
    .stDataFrame {
        width: 100%;
        overflow-x: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Grid Layout: 3 columns on Desktop, automatically wraps on smaller mobile screens
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("👤 Applicant Demographics")
    age = st.slider("Age", 21, 65, 30)
    education = st.selectbox("Education Level", ['High School', 'Graduate', 'Post Graduate', 'Professional'])

with col2:
    st.subheader("💼 Financial Profile")
    monthly_salary = st.number_input("Monthly Salary (₹)", 10000, 1000000, 65000, step=5000)
    credit_score = st.slider("Credit Score", 300, 850, 750)

with col3:
    st.subheader("📝 Loan Details")
    requested_amount = st.number_input("Requested Loan Amount (₹)", 10000, 10000000, 300000, step=10000)
    requested_tenure = st.slider("Requested Tenure (Months)", 3, 360, 36)

# Responsive Action Button
evaluate_btn = st.button("🚀 Evaluate Risk & Max EMI", use_container_width=True)

# Sidebar status banner
st.sidebar.success("System Status: Online")
st.sidebar.info("Model Version: v1.4.0 (XGBoost)")