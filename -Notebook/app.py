
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Employee Attrition Intelligence Platform", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load("attrition_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_assets()

st.title("📊 Employee Attrition Intelligence Platform")
st.markdown("Predict employee attrition risk and generate retention recommendations.")

st.sidebar.header("Employee Inputs")

data = {}
for feature in feature_names:
    data[feature] = st.sidebar.number_input(feature, value=0.0)

if st.button("Predict Attrition Risk"):
    X = pd.DataFrame([data])[feature_names]
    X_scaled = scaler.transform(X)

    prob = float(model.predict_proba(X_scaled)[0][1])
    pred = int(model.predict(X_scaled)[0])

    if prob < 0.30:
        risk = "Low Risk 🟢"
    elif prob < 0.60:
        risk = "Medium Risk 🟡"
    else:
        risk = "High Risk 🔴"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Risk Score", f"{prob:.1%}")

    with col2:
        st.metric("Prediction", "Leave ❌" if pred else "Stay ✅")

    with col3:
        st.metric("Risk Category", risk)

    st.subheader("Retention Recommendations")

    recs = []

    if "OverTime" in data and data["OverTime"] > 0:
        recs.append("Improve work-life balance and reduce overtime.")

    if "MonthlyIncome" in data and data["MonthlyIncome"] < 5000:
        recs.append("Review compensation package.")

    if "DistanceFromHome" in data and data["DistanceFromHome"] > 15:
        recs.append("Consider hybrid or remote work options.")

    if "JobSatisfaction" in data and data["JobSatisfaction"] <= 2:
        recs.append("Conduct employee engagement discussions.")

    if not recs:
        recs.append("No major retention concerns identified.")

    for r in recs:
        st.write("•", r)

    st.subheader("Top Input Values")
    st.dataframe(X.T.rename(columns={0: "Value"}))

st.markdown("---")
st.caption("Developed by Akash Mishra")
