import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("📊 Dataset Exploration & Visual Analytics")

@st.cache_data
def load_data():
    # Generate sample dataframe representation for visualization demo
    data = pd.DataFrame({
        'emi_scenario': np.random.choice(['Personal Loan', 'Vehicle', 'E-commerce'], 1000),
        'credit_score': np.random.randint(300, 850, 1000),
        'monthly_salary': np.random.normal(60000, 15000, 1000),
        'emi_eligibility': np.random.choice(['Eligible', 'High_Risk', 'Not_Eligible'], 1000, p=[0.6, 0.25, 0.15])
    })
    return data

df = load_data()

st.sidebar.header("Filter Options")
selected_scenario = st.sidebar.multiselect("Lending Scenario", df['emi_scenario'].unique(), default=df['emi_scenario'].unique())
filtered_df = df[df['emi_scenario'].isin(selected_scenario)]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Eligibility Distribution by Scenario")
    # Render responsive Plotly charts
    fig1 = px.histogram(filtered_df, x="emi_scenario", color="emi_eligibility", barmode="group", title="Eligibility Breakdown by Scenario")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Credit Score vs Salary Dispersion")
    fig2 = px.scatter(filtered_df, x="monthly_salary", y="credit_score", color="emi_eligibility", opacity=0.7)
    fig2.update_layout(autosize=True, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig2, use_container_width=True)