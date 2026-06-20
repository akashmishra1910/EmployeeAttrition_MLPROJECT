

📊 # Employee Attrition Intelligence Platform

An end-to-end Machine Learning application that predicts employee attrition risk and provides actionable workforce insights. The system helps HR teams identify employees who are likely to leave the organization and enables proactive retention strategies.

⸻

🚀 Project Overview

Employee attrition is a major challenge for organizations, leading to increased recruitment costs, productivity loss, and knowledge gaps.

This project leverages Machine Learning techniques to analyze employee data and predict attrition probability with high accuracy. The application is deployed through an interactive Streamlit dashboard for real-time predictions and HR analytics.

⸻

✨ Features

🔍 Attrition Risk Prediction

* Predicts whether an employee is likely to leave the organization.
* Displays attrition probability score.
* Provides risk classification:
    * 🟢 Low Risk
    * 🟡 Medium Risk
    * 🔴 High Risk

📈 HR Analytics Dashboard

* Department-wise attrition trends
* Salary distribution analysis
* Job satisfaction insights
* Work-life balance analysis
* Overtime impact visualization

🤖 Machine Learning Pipeline

* Data preprocessing
* Feature encoding
* Feature scaling
* SMOTE oversampling for class balancing
* Model training and evaluation

📊 Explainable Predictions

* Probability-based predictions
* Confidence scores
* Employee retention insights

🎨 Modern Streamlit UI

* Responsive dark-themed dashboard
* Interactive widgets
* Real-time prediction engine
* Visual analytics

⸻

🛠️ Tech Stack

Programming Language

* Python

Machine Learning

* Scikit-Learn
* Random Forest Classifier
* Logistic Regression
* XGBoost (optional)
* SMOTE

Data Analysis

* Pandas
* NumPy

Visualization

* Matplotlib
* Plotly
* Seaborn

Deployment

* Streamlit

⸻

📂 Project Structure

Employee-Attrition-Intelligence-Platform/
│
├── app.py
├── Employee_Attrition_Prediction_System.ipynb
│
├── Model/
│   ├── attrition_model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── feature_names.pkl
│
├── Dataset/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── requirements.txt
├── README.md
└── assets/

⸻

📋 Dataset

Dataset Used:

IBM HR Analytics Employee Attrition & Performance Dataset

Features include:

* Age
* Department
* Job Role
* Monthly Income
* Years at Company
* Overtime
* Job Satisfaction
* Environment Satisfaction
* Work-Life Balance
* Distance From Home
* Education Level
* Business Travel

Target Variable:

Attrition
├── Yes (Employee likely to leave)
└── No (Employee likely to stay)

⸻

⚙️ Installation

Clone Repository

git clone https://github.com/akashmishra1910/EmployeeAttrition_MLPROJECT.git
cd EmployeeAttrition_MLPROJECT
Create Virtual Environment

python -m venv venv

Activate environment:

source venv/bin/activate

or

venv\Scripts\activate

Install Dependencies

pip install -r requirements.txt

⸻

▶️ Run Application

streamlit run app.py

Application will start at:

http://localhost:8501

⸻

🧠 Machine Learning Workflow

Raw Data
    ↓
Data Cleaning
    ↓
Label Encoding
    ↓
Feature Scaling
    ↓
SMOTE Oversampling
    ↓
Model Training
    ↓
Evaluation
    ↓
Model Saving
    ↓
Streamlit Deployment

⸻

📊 Model Performance

Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	76.5%	38.5%	78.7%	51.7%
Random Forest	Best Performing	High	High	High

Selected Model: Random Forest Classifier

Reason:

* Better generalization
* Handles non-linearity
* Robust against overfitting
* Strong performance on imbalanced data

⸻

🔮 Future Enhancements

* SHAP Explainability
* Employee Retention Recommendations
* Department-Level Risk Forecasting
* Real-Time Database Integration
* Cloud Deployment (AWS/Azure)
* Automated Model Retraining
* HR Chatbot Assistant


🎯 Business Impact

This solution can help organizations:

✅ Reduce employee turnover

✅ Improve workforce planning

✅ Lower recruitment costs

✅ Increase employee retention

✅ Enable data-driven HR decisions

⸻

👨‍💻 Author

Akash Mishra

Data Science | Machine Learning | AI Enthusiast

GitHub: https://github.com/yourusername

LinkedIn: https://linkedin.com/in/akashmishra1910

⸻

⭐ Support

If you found this project useful:

⭐ Star the repository

🍴 Fork the repository

📢 Share with others

⸻

License

This project is licensed under the MIT License.
