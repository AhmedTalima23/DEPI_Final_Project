import streamlit as st
import pandas as pd
import numpy as np
import joblib
from collections import Counter

# Load feature column structure
try:
    columns = joblib.load("models/columns.pkl")
except FileNotFoundError:
    st.error("Error: Could not load columns.pkl. Please check the file path.")
    st.stop()

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

# Load models with error handling
models = {}
for name in model_names:
    try:
        models[name] = joblib.load(f"models/{name}")
    except FileNotFoundError:
        st.warning(f"Warning: Could not load {name}. Skipping this model.")

if not models:
    st.error("Error: No models were loaded. Please check your model files.")
    st.stop()

st.title("🔍 Employee Attrition Prediction")
st.write("Please enter employee information below:")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
    last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.6)
    number_project = st.number_input("Number of Projects", min_value=1, max_value=10, value=3)
    average_monthly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=310, value=160)

with col2:
    time_spend_company = st.number_input("Years at Company", min_value=1, max_value=10, value=3)
    Work_accident = st.selectbox("Had Work Accident?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    promotion_last_5years = st.selectbox("Promoted in Last 5 Years?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
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
    st.markdown("---")
    st.subheader("🧠 Prediction Results")
    
    votes = []
    predictions = {}
    confidence_scores = {}
    
    # Create a container for model predictions
    results_container = st.container()
    
    with results_container:
        st.write("### Individual Model Predictions:")
        
        for name, model in models.items():
            try:
                # Get prediction
                prediction = model.predict(input_df)[0]
                result = "Left" if prediction == 1 else "Stayed"
                predictions[name] = result
                votes.append(result)
                
                # Get confidence score if available
                confidence = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_df)[0]
                    confidence = proba[1] if prediction == 1 else proba[0]
                    confidence_scores[name] = confidence
                
                # Display prediction with colored box
                if prediction == 1:
                    st.error(f"**{name.replace('_model.pkl', '').upper()}**: 🚨 Left (Confidence: {confidence:.2%})" if confidence else f"**{name.replace('_model.pkl', '').upper()}**: 🚨 Left")
                else:
                    st.success(f"**{name.replace('_model.pkl', '').upper()}**: ✅ Stayed (Confidence: {confidence:.2%})" if confidence else f"**{name.replace('_model.pkl', '').upper()}**: ✅ Stayed")
                
            except Exception as e:
                st.warning(f"⚠️ {name.replace('_model.pkl', '')} model error: {str(e)}")
    
    # Calculate majority vote
    vote_counts = Counter(votes)
    total_votes = len(votes)
    final_decision = vote_counts.most_common(1)[0][0]
    left_percentage = vote_counts.get("Left", 0) / total_votes
    
    # Display voting results as text
    st.markdown("---")
    st.subheader("🗳️ Voting Results")
    st.write(f"**Final Decision**: {final_decision} ({vote_counts.get('Left', 0)} Left vs {vote_counts.get('Stayed', 0)} Stayed)")
    st.write(f"**Left Percentage**: {left_percentage:.2%}")
    
    # Display detailed vote breakdown
    st.write("### Detailed Vote Count:")
    vote_df = pd.DataFrame.from_dict(vote_counts, orient='index', columns=['Count'])
    st.dataframe(vote_df.style.background_gradient(cmap='Blues'))
    
    # Display risk factors
    st.markdown("---")
    st.subheader("⚠️ Key Risk Factors")
    
    risk_factors = []
    if satisfaction_level < 0.3:
        risk_factors.append(f"Very low satisfaction level ({satisfaction_level:.1%})")
    if average_monthly_hours > 250:
        risk_factors.append(f"High monthly hours ({average_monthly_hours} hrs/month)")
    if time_spend_company > 5 and promotion_last_5years == 0:
        risk_factors.append(f"Long tenure ({time_spend_company} years) without promotion")
    if number_project > 6:
        risk_factors.append(f"High project load ({number_project} projects)")
    
    if risk_factors:
        for factor in risk_factors:
            st.warning(f"• {factor}")
    else:
        st.info("No significant risk factors identified")
