import streamlit as st
import pandas as pd
import numpy as np
import joblib
from collections import Counter

# Load feature column structure
columns = joblib.load("models/columns.pkl")

# Define model names (must match saved filenames)
model_names = [
    'Random Forest_model.pkl',
    'XGBoost_model.pkl',
    'Gradient Boosting_model.pkl',
    'Logistic Regression_model.pkl',
    'Decision Tree_model.pkl',
    'K-Nearest Neighbors_model.pkl',
    'voting_model.pkl'
]

# Load models from the models folder
models = {name: joblib.load(f"models/{name}_model.pkl") for name in model_names}

st.title("🔍 Employee Attrition Prediction")
st.write("Please enter employee information below:")

# User input fields
satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.6)
number_project = st.number_input("Number of Projects", min_value=1, max_value=10, value=3)
average_monthly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=310, value=160)
time_spend_company = st.number_input("Years at Company", min_value=1, max_value=10, value=3)
Work_accident = st.selectbox("Had Work Accident?", [0, 1])
promotion_last_5years = st.selectbox("Promoted in Last 5 Years?", [0, 1])
department = st.selectbox("Department", ['sales', 'technical', 'support', 'IT', 'HR', 'product_mng', 'marketing', 'RandD', 'accounting', 'management'])
salary = st.selectbox("Salary Level", ['low', 'medium', 'high'])

# Build input feature dictionary
input_data = {
    'satisfaction_level': satisfaction_level,
    'last_evaluation': last_evaluation,
    'number_project': number_project,
    'average_montly_hours': average_monthly_hours,
    'time_spend_company': time_spend_company,
    'Work_accident': Work_accident,
    'promotion_last_5years': promotion_last_5years,
    f'Departments_{department}': 1,
    f'salary_{salary}': 1
}

# Convert to DataFrame and align with expected model input
input_df = pd.DataFrame([input_data])
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[columns]

# Run prediction when user clicks the button
if st.button("Predict Attrition"):
    st.subheader("🧠 Individual Model Predictions:")
    votes = []

    for name, model in models.items():
        try:
            prediction = model.predict(input_df)[0]
            result = "Left" if prediction == 1 else "Stayed"
            st.write(f"**{name.upper()}**: {result}")
            votes.append(result)
        except Exception as e:
            st.error(f"{name} model error: {e}")

    # Majority voting
    final_decision = Counter(votes).most_common(1)[0][0]
    st.markdown("---")
    st.success(f"🗳️ Final Decision (Majority Vote): **{final_decision}**")
    st.write("🔢 Vote Counts:", dict(Counter(votes)))
