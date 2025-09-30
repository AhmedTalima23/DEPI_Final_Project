# 👩‍💼 Employee Churn Prediction & Analytics Platform

A **Streamlit-based HR Analytics platform** designed to predict employee churn, analyze retention factors, generate professional reports, and provide an AI-powered HR chatbot using **Google Gemini API**.

---

## 🚀 Features

### 🔮 Prediction Dashboard
- Input employee details (satisfaction, workload, tenure, promotions, salary, etc.)
- Predict **employee churn risk** (High / Low)
- Display **probability scores** with interactive visual risk cards
- Uses trained machine learning model (`XGB.pkl`)

### 📈 Analytics Insights
- View **retention metrics** (overall retention rate, satisfaction, average tenure)
- Explore **department-wise analysis**
- Identify **top retention factors**
- Get **recommendations** to improve retention

### 📝 Report Generator
- Generates **comprehensive employee retention reports**
- Powered by **Gemini 1.5 Flash**
- Includes:
  - Risk analysis summary
  - Key contributing factors
  - Retention strategies
  - Development opportunities
  - Management suggestions
- Download reports as **Markdown files**

### 💬 Employee Service Chatbot
- AI-powered assistant using **Gemini 2.0 Flash**
- Provides instant responses to HR-related queries
- Keeps conversation history for a natural chat experience

### 🎨 Modern UI
- Custom CSS for a clean, corporate look
- Gradient headers, styled metrics, interactive tabs
- Dark/soft accent colors for professional readability

---

## 📂 Project Structure
📁 employee-churn-app
│── app.py # Main Streamlit application
│── XGB.pkl # Trained XGBoost model
│── dummy_columns.pkl # One-hot encoded training columns
│── churn.png # Dashboard logo
│── hr_analytics.jpg # Landing page banner
│── requirements.txt # Dependencies
│── README.md # Documentation

