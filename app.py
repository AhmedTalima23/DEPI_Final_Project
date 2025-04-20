import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and preprocessors
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')  # if you used one

st.title("🔮 Employee Attrition Prediction")

# Manual input of features
st.header("📋 Enter Employee Details")

age = st.slider("Age", 18, 65, 30)
monthly_income = st.number_input("Monthly Income", min_value=1000, step=100)
job_satisfaction = st.selectbox("Job Satisfaction", ["Very High", "High", "Medium", "Low"])
performance_rating = st.selectbox("Performance Rating", ["High", "Average", "Low", "Below Average"])
company_reputation = st.selectbox("Company Reputation", ["Excellent", "Good", "Fair", "Poor"])
gender = st.selectbox("Gender", ["Male", "Female"])
overtime = st.selectbox("Overtime", ["Yes", "No"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
remote_work = st.selectbox("Remote Work", ["Yes", "No"])
company_size = st.selectbox("Company Size", ["Large", "Medium", "Small"])
education_level = st.selectbox("Education Level", ["High School", "Associate Degree", "Bachelor's Degree", "Masters Degree", "PhD"])

# Add all features in a DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "Monthly Income": monthly_income,
    "Job Satisfaction": job_satisfaction,
    "Performance Rating": performance_rating,
    "Company Reputation": company_reputation,
    "Gender": gender,
    "Overtime": overtime,
    "Marital Status": marital_status,
    "Remote Work": remote_work,
    "Company Size": company_size,
    "Education Level": education_level
    # Add more fields as needed based on your model
}])

st.subheader("🔍 Preview of Input Data")
st.write(input_data)

# Prediction button
if st.button("Predict"):
    try:
        # Encode and scale
        encoded_data = encoder.transform(input_data)
        scaled_data = scaler.transform(encoded_data)

        prediction = model.predict(scaled_data)[0]
        result = "🟢 Stayed" if prediction == 0 else "🔴 Left"
        st.success(f"Prediction: {result}")

    except Exception as e:
        st.error(f"Something went wrong during prediction: {e}")
