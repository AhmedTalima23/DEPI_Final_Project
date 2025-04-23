import streamlit as st
import pandas as pd
import numpy as np
import joblib
from collections import Counter

# Load column structure and models
columns = joblib.load("models/columns.pkl")
model_names = ['rf', 'xgb', 'gb', 'svm', 'lr', 'dt', 'knn', 'voting']
models = {name: joblib.load(f'models/{name}_model.pkl') for name in model_names}

st.title("🔍 Employee Attrition Prediction")
st.write("Fill out the employee details:")

# User inputs
satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.6)
number_project = st.number_input("Number of Projects", 1, 10, 3)
average_montly_hours = st.number_input("Average Monthly Hours", 50, 310, 160)
time_spend_company = st.number_input("Years at Company", 1, 10, 3)
Work_accident = st.selectbox("Work Accident", [0, 1])
promotion_last_5years = st.selectbox("Promotion in Last 5 Years", [0, 1])
Departments  = st.selectbox("Departments ", ['sales', 'technical', 'support', 'IT', 'HR', 'product_mng', 'marketing', 'RandD', 'accounting', 'management'])
salary = st.selectbox("Salary Level", ['low', 'medium', 'high'])

# Input dictionary
input_data = {
    'satisfaction_level': satisfaction_level,
    'last_evaluation': last_evaluation,
    'number_project': number_project,
    'average_montly_hours': average_montly_hours,
    'time_spend_company': time_spend_company,
    'Work_accident': Work_accident,
    'promotion_last_5years': promotion_last_5years,
    'Departments ' + Departments: 1,
    'salary_' + salary: 1
}

# Create full feature row
input_df = pd.DataFrame([input_data])
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[columns]

if st.button("Predict Attrition"):
    st.subheader("🧠 Individual Model Predictions:")
    votes = []
    for name, model in models.items():
        try:
            pred = model.predict(input_df)[0]
            result = "Left" if pred == 1 else "Stayed"
            st.write(f"**{name.upper()}**: {result}")
            votes.append(result)
        except Exception as e:
            st.error(f"{name} error: {e}")

    # Final majority vote
    final = Counter(votes).most_common(1)[0][0]
    st.markdown("---")
    st.success(f"🗳️ Final Decision (Majority Vote): **{final}**")
