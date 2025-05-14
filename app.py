import streamlit as st

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="👩‍💼 Employee Churn Prediction",
    page_icon=":office_worker:",
    layout="wide",
    initial_sidebar_state="expanded"
)
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import requests
# from dotenv import load_dotenv
# # Load environment variables
# load_dotenv()




# Configuration
GEMINI_API_KEY = "AIzaSyAmSUHmgkGVQMIFhqe7EQaPg2RQ7lDy8w4"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# Load models and data
@st.cache_resource
def load_model_and_columns():
    models = [joblib.load('XGBoost_model.pkl')]  # Replace with your actual model path
    dummy_columns = joblib.load('dummy_columns.pkl')  # Replace with your actual columns path
    return models, dummy_columns

models, dummy_columns = load_model_and_columns()

# Custom CSS with softer colors
st.markdown("""
    <style>
    :root {
        --primary: #5a7faa;
        --secondary: #2a7f9d;
        --accent: #6ec1e8;
        --background: #f8f9fa;
        --card: #ffffff;
        --text: #333333;
        --positive: #5cb85c;
        --negative: #d9534f;
    }
    
    .main {
        background-color: var(--background);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(90, 127, 170, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--card) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        padding: 2rem;
        border-radius: 10px;
        color: white;
    }
    
    .header-text {
        padding-left: 2rem;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .prediction-card {
        border-radius: 12px;
        padding: 22px;
        margin: 15px 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        background-color: var(--card);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .positive {
        border-left: 5px solid var(--negative);
        background-color: rgba(217, 83, 79, 0.03);
    }
    
    .negative {
        border-left: 5px solid var(--positive);
        background-color: rgba(92, 184, 92, 0.03);
    }
    
    .sidebar .sidebar-content {
        background-color: var(--card);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stSlider>div>div>div>div {
        background: var(--accent) !important;
    }
    
    .stButton>button {
        background-color: var(--secondary);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: var(--primary);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        color: var(--text);
        font-size: 0.9rem;
    }
    
    .team-credits {
        background-color: rgba(74, 111, 165, 0.1);
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    .team-members {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 8px 15px;
        margin: 10px 0;
    }
    
    .block-container {
        padding-top: 2rem;
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to call Gemini API
def call_gemini_api(prompt):
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error with Gemini API request: {str(e)}")
        return None

# Generate report using Gemini
def generate_employee_report(employee_data, prediction_result, probability):
    prompt = f"""
    Generate a comprehensive employee retention report with the following details:
    
    Employee Information:
    - Age: {employee_data.get('age', 'N/A')}
    - Department: {employee_data.get('department', 'N/A')}
    - Salary Level: {employee_data.get('salary', 'N/A')}
    - Monthly Income: {employee_data.get('monthly_income', 'N/A')}
    - Years at Company: {employee_data.get('years_at_company', 'N/A')}
    - Satisfaction Level: {employee_data.get('satisfaction_level', 0)*100:.0f}%
    
    Prediction Results:
    - Retention Risk: {'High' if prediction_result == 1 else 'Low'}
    - Probability: {probability*100:.1f}%
    
    Please provide:
    1. Risk analysis summary
    2. Key contributing factors
    3. Recommended retention strategies
    4. Development opportunities
    5. Management suggestions
    
    Format the response in professional business language with clear sections.
    """
    
    response = call_gemini_api(prompt)
    if response and 'candidates' in response and len(response['candidates']) > 0:
        return response['candidates'][0]['content']['parts'][0]['text']
    return None

# App header
def render_header():
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(Image.open("churn.png"), width=400)  # Replace with your image path
    with col2:
        st.markdown("""
        <div class="header-text">
            <h1 class="header-title">👩‍💼 Employee Churn Prediction</h1>
            <p class="header-subtitle">
                Predict churn risks • Improve retention • Strengthen your workforce
            </p>
        </div>
        """, unsafe_allow_html=True)

# Prediction page
def prediction_page():
    render_header()
    
    with st.sidebar:
        st.markdown("## Employee Details")
        st.markdown("Complete the form to assess retention risk")
        
        with st.expander("Personal Factors", expanded=True):
            satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5, 0.01,
                                        help="Employee's overall job satisfaction")
            st.caption(f"Current value: {satisfaction_level:.0%}")
            
            department = st.selectbox("Department", 
                                  ['sales', 'technical', 'support', 'hr', 
                                   'accounting', 'marketing', 'product_mng', 
                                   'management', 'RandD'],
                                  help="Employee's department")
            
            salary = st.selectbox("Salary Level", ['low', 'medium', 'high'],
                                help="Employee's salary tier")
        
        with st.expander("Performance Metrics", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                last_evaluation = st.slider("Evaluation Score", 0.0, 1.0, 0.7, 0.05)
            with col2:
                number_project = st.number_input("Project Count", 1, 10, 3)
            
            average_montly_hours = st.slider("Monthly Hours", 50, 400, 160, 10,
                                          help="Average working hours per month")
        
        with st.expander("Employment History", expanded=False):
            time_spend_company = st.select_slider("Company Tenure", 
                                               options=list(range(1, 21)), 
                                               value=3)
            
            work_accident = st.radio("Work Accident", ["No", "Yes"], index=0,
                                  horizontal=True)
            
            promotion_last_5years = st.radio("Recent Promotion", ["No", "Yes"], 
                                          index=0, horizontal=True)
        
        predict_clicked = st.button("Assess Retention Risk", type="primary")

    if predict_clicked:
        # Convert inputs
        work_accident = 1 if work_accident == "Yes" else 0
        promotion_last_5years = 1 if promotion_last_5years == "Yes" else 0

        # Create input DataFrame
        input_df = pd.DataFrame([{
            'satisfaction_level': satisfaction_level,
            'last_evaluation': last_evaluation,
            'number_project': number_project,
            'average_montly_hours': average_montly_hours,
            'time_spend_company': time_spend_company,
            'Work_accident': work_accident,
            'promotion_last_5years': promotion_last_5years,
            'Departments': department,
            'salary': salary,
        }])

        # Encode categorical variables
        input_df = pd.get_dummies(input_df, columns=['Departments', 'salary'], drop_first=True)

        # Ensure same columns as training set
        for col in dummy_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[dummy_columns]  # Reorder columns

        # Predict with all models
        predictions = [model.predict(input_df)[0] for model in models]
        prediction_probs = [model.predict_proba(input_df)[0][1] for model in models]
        final_prediction = max(set(predictions), key=predictions.count)
        avg_prob = np.mean(prediction_probs)

        # Show result
        st.subheader("Retention Risk Assessment")
        
        if final_prediction == 1:
            risk_level = "High Risk"
            risk_color = "var(--negative)"
            card_class = "positive"
            message = "⚠️ Higher probability of employee leaving"
            icon = "⚠️"
        else:
            risk_level = "Low Risk"
            risk_color = "var(--positive)"
            card_class = "negative"
            message = "✅ Employee likely to stay"
            icon = "✅"
        
        # Risk card
        st.markdown(f"""
        <div class="prediction-card {card_class}">
            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 1.8rem; margin-right: 12px; opacity: 0.9;">{icon}</span>
                <div>
                    <h2 style="color:{risk_color}; margin:0; font-weight:600; opacity: 0.9;">{risk_level}</h2>
                    <p style="margin:0; font-size: 1rem; opacity: 0.8;">{message}</p>
                </div>
            </div>
            <div style="background: rgba(0,0,0,0.03); padding: 12px; border-radius: 8px;">
                <h4 style="margin-top:0; font-weight:500; opacity: 0.9;">Risk Probability: <strong>{avg_prob*100:.1f}%</strong></h4>
                <div style="height: 6px; background: #f0f0f0; border-radius: 3px; margin: 8px 0;">
                    <div style="height: 100%; width: {avg_prob*100}%; background: {risk_color}; border-radius: 3px; opacity: 0.8;"></div>
                </div>
                <p style="font-size: 0.85rem; margin-bottom:0; opacity: 0.7;">Scale: 0% (No risk) → 100% (Certain to leave)</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# Analytics page
def analytics_page():
    render_header()
    st.header("Retention Overview")
    st.write("## Key Retention Metrics")
    
    # Metrics cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Retention Rate", "82%", "+2% from last quarter")
    with col2:
        st.metric("Average Satisfaction", "6.8/10", "-0.3 from last quarter")
    with col3:
        st.metric("Avg Company Tenure", "3.2 years", "stable")
    
    st.markdown("---")
    
    # Charts
    st.subheader("Top Retention Factors")
    factors = pd.DataFrame({
        'Factor': ['Satisfaction', 'Workload', 'Salary', 'Career Growth', 'Work Environment'],
        'Impact': [85, 72, 68, 63, 58]
    })
    st.bar_chart(factors.set_index('Factor'))
    
    st.markdown("---")
    st.subheader("Quick Recommendations")
    st.write("""
    - **Focus on satisfaction**: Employees with satisfaction below 5/10 are 3x more likely to leave
    - **Monitor workload**: Those working >200 hours/month show higher turnover
    - **Review compensation**: Low salary tier employees have 25% higher churn
    - **Career development**: Employees without promotion in 3+ years are at risk
    """)
    
    # Department comparison
    st.markdown("---")
    st.subheader("By Department")
    dept_data = pd.DataFrame({
        'Department': ['Sales', 'Engineering', 'Technical', 'HR', 'Management'],
        'Retention Rate': [75, 88, 82, 91, 89],
        'Avg Satisfaction': [6.2, 7.1, 6.8, 7.4, 7.3]
    })
    st.dataframe(dept_data.style.highlight_max(axis=0, color='#5cb85c'))

# Home page
def home_page():
    render_header()
    st.write("""
    ## Welcome to the Employee Retention Analytics Platform
    
    This comprehensive tool helps HR professionals and managers:
    - Predict employee churn risk
    - Identify key retention factors
    - Generate actionable insights
    - Create detailed retention reports
    
    ### How to Use This App:
    1. **Prediction Dashboard**: Assess individual employee retention risk
    2. **Analytics Insights**: View organizational trends and patterns
    3. **Report Generator**: Create detailed retention analysis reports
    
    Get started by selecting a section from the sidebar.
    """)
    
    st.image(Image.open("hr_analytics.jpg"), width=700)  # Replace with your image path



# Report Generator page
def report_generator_page():
    render_header()
    st.title("📝 Employee Retention Report Generator")
    
    with st.form("employee_details_form"):
        st.subheader("Employee Information")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=5000)
            department = st.selectbox("Department", 
                                   ['Sales', 'Technical', 'Support', 'HR', 
                                    'Accounting', 'Marketing', 'Product Management', 
                                    'Management', 'R&D'])
        with col2:
            years_at_company = st.number_input("Years at Company", min_value=0, max_value=50, value=5)
            distance_from_home = st.number_input("Distance from Home (miles)", min_value=0, max_value=100, value=10)
            salary = st.selectbox("Salary Level", ['Low', 'Medium', 'High'])
        
        st.subheader("Job Satisfaction Metrics")
        satisfaction_level = st.slider("Satisfaction Level (0-100%)", 0, 100, 50) / 100
        last_evaluation = st.slider("Last Evaluation Score (0-100%)", 0, 100, 70) / 100
        
        submitted = st.form_submit_button("Generate Comprehensive Report")
    
    if submitted and GEMINI_API_KEY:
        with st.spinner("Generating detailed report..."):
            employee_data = {
                'age': age,
                'monthly_income': monthly_income,
                'years_at_company': years_at_company,
                'distance_from_home': distance_from_home,
                'department': department,
                'salary': salary,
                'satisfaction_level': satisfaction_level,
                'last_evaluation': last_evaluation
            }
            
            # For demo, we'll use a mock prediction
            mock_prediction = 0 if satisfaction_level > 0.6 else 1
            mock_probability = 0.85 if satisfaction_level < 0.5 else 0.25
            
            report = generate_employee_report(employee_data, mock_prediction, mock_probability)
            
            if report:
                st.success("Report generated successfully!")
                st.markdown("---")
                st.subheader("Employee Retention Analysis Report")
                st.markdown(report)
                
                st.download_button(
                    label="Download Full Report",
                    data=report,
                    file_name=f"retention_report_{department}_{age}.md",
                    mime="text/markdown"
                )
            else:
                st.error("Failed to generate report. Please try again.")
    elif submitted and not GEMINI_API_KEY:
        st.error("Gemini API key not configured. Report generation disabled.")

# Footer
def render_footer():
    st.markdown("""
    <div class="footer">
        <div class="team-credits">
            <p style="font-weight: 600; text-align: center; margin-bottom: 10px;">Project Development Team</p>
            <div class="team-members">
                <span>• Ahmed Mohamed</span>
                <span>• Theodore Naguib</span>
                <span>• Malak Torky</span>
                <span>• Shrouk Emam</span>
                <span>• Salah Eldin Mohamed</span>
                <span>• Seif Ahmed</span>
            </div>
            <p style="text-align: center; margin-top: 10px;">
                Supervised by: <span style="font-weight: 600;">Eng. Mahmoud Talaat</span>
            </p>
        </div>
        <p style="margin-top: 20px;">Employee Retention Pro • Powered by HR Analytics • v2.1</p>
    </div>
    """, unsafe_allow_html=True)

# Main app
def main():
    # Navigation
    pages = {
        "🏠 Home": home_page,
        "🔮 Prediction Dashboard": prediction_page,
        "📈 Analytics Insights": analytics_page,
        "📝 Report Generator": report_generator_page
    }
    
    with st.sidebar:
        st.title("Navigation")
        selected = st.radio("Go to", list(pages.keys()))
    
    # Display the selected page
    pages[selected]()
    render_footer()

if __name__ == "__main__":
    main()
