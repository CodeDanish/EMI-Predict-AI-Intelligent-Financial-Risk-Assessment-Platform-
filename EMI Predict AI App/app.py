import streamlit as st
import os
import joblib

# -----------------------------------------------------------------------------
# 1. GLOBAL PAGE CONFIGURATION & RESPONSIVE STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EMI Predict AI | Financial Risk Assessment Platform",
    page_icon="🏦",
    layout="wide",  # Full-width responsive layout for mobile, tablet & desktop
    initial_sidebar_state="expanded"
)

# Global Custom CSS Injection
st.markdown("""
    <style>
    /* Responsive button and tap target styling */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3.2em;
        font-weight: bold;
    }
    /* Horizontal overflow containment for dataframes and tables */
    .stDataFrame {
        width: 100%;
        overflow-x: auto;
    }
    /* Hero section formatting */
    .hero-container {
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #0d6efd 0%, #0a58ca 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    .hero-container h1 {
        color: white;
        margin-bottom: 0.5rem;
    }
    .card {
        padding: 1.5rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. GLOBAL ARTIFACT HEALTHCHECK & PRE-LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def check_and_preload_artifacts():
    """Verify that all necessary serialized ML artifacts exist in the models directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    if not os.path.exists(models_dir):
        models_dir = "models"
        
    required_artifacts = [
        "scaler.pkl",
        "ordinal_encoder.pkl",
        "ohe_encoder.pkl",
        "label_encoder.pkl",
        "classification_xgb.pkl",
        "regression_xgb.pkl"
    ]
    
    missing_files = []
    if os.path.exists(models_dir):
        for artifact in required_artifacts:
            if not os.path.exists(os.path.join(models_dir, artifact)):
                missing_files.append(artifact)
    else:
        missing_files = required_artifacts

    return models_dir, missing_files

models_path, missing_artifacts = check_and_preload_artifacts()

# -----------------------------------------------------------------------------
# 3. HERO & PLATFORM DASHBOARD
# -----------------------------------------------------------------------------
st.markdown("""
    <div class="hero-container">
        <h1>🏦 EMI Predict AI</h1>
        <p style="font-size: 1.2rem;">Intelligent Financial Risk Assessment & Dual-Model Loan Allocation Platform</p>
    </div>
""", unsafe_allow_html=True)

# Display status warning if artifacts are missing
if missing_artifacts:
    st.error(f"⚠️ Missing {len(missing_artifacts)} required ML artifact(s) in `{models_path}/`:")
    for item in missing_artifacts:
        st.write(f"- `{item}`")
    st.info("Please run your model training notebook to generate and save these `.pkl` files before running predictions.")
else:
    st.success("✅ Machine Learning Models & Transformation Pipelines Loaded Successfully!")

st.markdown("---")

# -----------------------------------------------------------------------------
# 4. PLATFORM ARCHITECTURE & CAPABILITIES
# -----------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="card">
            <h3>🔮 Dual-Model Inference</h3>
            <p>Runs dual <b>XGBoost Classifier</b> and <b>XGBoost Regressor</b> pipelines concurrently to evaluate eligibility risk and calculate safe monthly EMI limits.</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="card">
            <h3>📊 Interactive Analytics</h3>
            <p>Explore financial metric correlations, Debt-to-Income (DTI) distribution curves, and risk segmentation using interactive Plotly visual charts.</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="card">
            <h3>⚙️ Robust Pipeline Handling</h3>
            <p>Equipped with dynamic schema alignment, auto-calculation of derived ratios, missing feature imputation, and error handling.</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. NAVIGATION PROMPT
# -----------------------------------------------------------------------------
st.info("👈 **Get Started:** Select **`1_🔮_Realtime_Prediction`** from the sidebar to perform real-time loan eligibility evaluations.")
