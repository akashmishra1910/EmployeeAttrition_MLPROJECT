"""
app.py - Employee Attrition Intelligence Platform (Modern Light-Mode SaaS)
--------------------------------------------------------------------------
A commercial-grade People Analytics & Explainable AI Dashboard built with Streamlit.

Design System:
- Theme: Stripe / Linear-inspired Light Mode (#F7F8FC background, #FFFFFF cards)
- Primary Palette: Deep Indigo (#4338CA), Royal Violet (#6D5AE6), Soft Lavender (#EEEAFF)
- Semantic Colors: Emerald (#16A673), Amber (#E9A23B), Coral Red (#E5484D)
- Sections: Overview, Workforce Analytics, Predict Attrition, Explain Predictions,
            What-If Simulation, Model Performance, and How It Works.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys

# ── Ensure module files are importable regardless of how Streamlit Cloud mounts the repo ──
# APP_DIR = the folder containing app.py (i.e. Streamlit_App/)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# REPO_ROOT = one level up from Streamlit_App/ (i.e. the repository root)
REPO_ROOT = os.path.dirname(APP_DIR)

for _p in (APP_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Modular application imports — always resolved via sys.path above
from explainability import (
    create_shap_explainer,
    explain_prediction,
    create_shap_bar_chart,
    create_global_feature_importance_chart,
    FEATURE_DISPLAY_NAMES
)
from recommendations import generate_retention_recommendations
from risk_scoring import compute_attrition_risk, categorize_risk

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global SaaS Light Theme CSS ───────────────────────────────────────────────
st.markdown("""
<style>
/* Font & Root App Canvas */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

[data-testid="stAppViewContainer"] {
    background-color: #F7F8FC;
    color: #1E2235;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E9F2 !important;
    padding-top: 1rem;
}
[data-testid="stSidebar"] * {
    color: #1E2235 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 14px;
    font-weight: 500;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 10px 14px;
    border-radius: 10px;
    transition: all 0.15s ease-in-out;
    margin-bottom: 4px;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background-color: #F1F3F9;
}

/* Header & Banner */
.header-container {
    background: linear-gradient(135deg, #FFFFFF 0%, #FAF5FF 100%);
    border: 1px solid #E5E9F2;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
.eyebrow-label {
    color: #6D5AE6;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.header-title {
    color: #1E2235;
    font-size: 26px;
    font-weight: 700;
    margin: 0;
}
.header-subtitle {
    color: #697386;
    font-size: 14px;
    margin: 6px 0 0 0;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    background-color: #ECFDF5;
    border: 1px solid #A7F3D0;
    color: #065F46 !important;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    margin-top: 8px;
}
.status-dot {
    height: 8px;
    width: 8px;
    background-color: #10B981;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}

/* SaaS KPI Cards */
.kpi-card {
    background-color: #FFFFFF;
    border: 1px solid #E5E9F2;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(67, 56, 202, 0.06);
}
.kpi-title {
    color: #697386;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0 0 6px 0;
}
.kpi-value {
    color: #1E2235;
    font-size: 28px;
    font-weight: 700;
    margin: 0 0 4px 0;
}
.kpi-subtext {
    color: #929AAF;
    font-size: 12px;
    margin: 0;
}

/* White Content Surfaces */
.saas-card {
    background-color: #FFFFFF;
    border: 1px solid #E5E9F2;
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
}

/* Recommendation Badges & Cards */
.rec-card-warn {
    background-color: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    color: #92400E;
    font-size: 13.5px;
}
.rec-card-danger {
    background-color: #FEF2F2;
    border: 1px solid #FECACA;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    color: #991B1B;
    font-size: 13.5px;
}
.rec-card-info {
    background-color: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    color: #1E40AF;
    font-size: 13.5px;
}
.rec-card-success {
    background-color: #ECFDF5;
    border: 1px solid #A7F3D0;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    color: #065F46;
    font-size: 13.5px;
}

/* Callouts & Model Badges */
.model-pill {
    background-color: #EEEAFF;
    border: 1px solid #D8D2FF;
    color: #4338CA;
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 14px;
}
.info-callout {
    background-color: #F8FAFC;
    border-left: 4px solid #6D5AE6;
    padding: 14px 18px;
    border-radius: 0 10px 10px 0;
    font-size: 13px;
    color: #334155;
    margin-bottom: 16px;
}

/* Form Action Buttons */
.stButton > button {
    width: 100%;
    height: 48px;
    border-radius: 10px;
    border: none;
    background: linear-gradient(90deg, #4338CA 0%, #6D5AE6 100%);
    color: #FFFFFF !important;
    font-weight: 600;
    font-size: 15px;
    box-shadow: 0 2px 4px rgba(67, 56, 202, 0.2);
    transition: opacity 0.15s ease, transform 0.1s ease;
}
.stButton > button:hover {
    opacity: 0.92;
    transform: translateY(-1px);
}

/* Form Inputs Styling */
div[data-baseweb="select"] {
    border-radius: 8px !important;
    border: 1px solid #D1D5DB !important;
}
div[data-baseweb="input"] input {
    color: #1E2235 !important;
}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load Model & Artifacts ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Analytics Engine...")
def load_artifacts():
    search_dirs = [
        "Model",
        "../Model",
        os.path.join(APP_DIR, "Model"),
        os.path.join(APP_DIR, "../Model"),
    ]
    model = None
    for d in search_dirs:
        p = os.path.join(d, "attrition_model.pkl")
        if os.path.exists(p):
            model         = joblib.load(p)
            scaler        = joblib.load(os.path.join(d, "scaler.pkl"))
            encoders      = joblib.load(os.path.join(d, "encoders.pkl"))
            feature_names = joblib.load(os.path.join(d, "feature_names.pkl"))
            break
            
    if model is None:
        # Final fallback attempt
        try:
            model         = joblib.load("Model/attrition_model.pkl")
            scaler        = joblib.load("Model/scaler.pkl")
            encoders      = joblib.load("Model/encoders.pkl")
            feature_names = joblib.load("Model/feature_names.pkl")
        except Exception:
            model         = joblib.load("../Model/attrition_model.pkl")
            scaler        = joblib.load("../Model/scaler.pkl")
            encoders      = joblib.load("../Model/encoders.pkl")
            feature_names = joblib.load("../Model/feature_names.pkl")
        
    explainer = create_shap_explainer(model, feature_names)
    return model, scaler, encoders, feature_names, explainer

@st.cache_data
def load_dataset():
    search_paths = [
        "Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "../Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv",
        os.path.join(APP_DIR, "Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv"),
        os.path.join(APP_DIR, "../Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv"),
    ]
    for p in search_paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return pd.read_csv("../Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

try:
    model, scaler, encoders, feature_names, explainer = load_artifacts()
    df_raw = load_dataset()
    engine_ready = True
except Exception as e:
    engine_ready = False
    engine_error = str(e)

# ── Feature Builder Helper ────────────────────────────────────────────────────
def build_features_from_dict(raw_data) -> pd.DataFrame:
    """
    Constructs a DataFrame matching the exact feature order of feature_names.pkl.
    """
    row = {
        "Age":                      raw_data["Age"],
        "BusinessTravel":           encoders["BusinessTravel"].transform([raw_data["BusinessTravel"]])[0],
        "DailyRate":                800,           # median baseline
        "Department":               encoders["Department"].transform([raw_data["Department"]])[0],
        "DistanceFromHome":         raw_data["DistanceFromHome"],
        "Education":                3,             # median baseline
        "EducationField":           encoders["EducationField"].transform([raw_data["EducationField"]])[0],
        "EnvironmentSatisfaction":  raw_data["EnvironmentSatisfaction"],
        "Gender":                   encoders["Gender"].transform([raw_data["Gender"]])[0],
        "HourlyRate":               65,            # median baseline
        "JobInvolvement":           3,             # median baseline
        "JobLevel":                 raw_data["JobLevel"],
        "JobRole":                  encoders["JobRole"].transform([raw_data["JobRole"]])[0],
        "JobSatisfaction":          raw_data["JobSatisfaction"],
        "MaritalStatus":            encoders["MaritalStatus"].transform([raw_data["MaritalStatus"]])[0],
        "MonthlyIncome":            raw_data["MonthlyIncome"],
        "MonthlyRate":              14000,         # median baseline
        "NumCompaniesWorked":       2,             # median baseline
        "OverTime":                 encoders["OverTime"].transform([raw_data["OverTime"]])[0],
        "PercentSalaryHike":        15,            # median baseline
        "PerformanceRating":        3,             # median baseline
        "RelationshipSatisfaction": 3,             # median baseline
        "StockOptionLevel":         raw_data["StockOptionLevel"],
        "TotalWorkingYears":        raw_data["TotalWorkingYears"],
        "TrainingTimesLastYear":    2,             # median baseline
        "WorkLifeBalance":          raw_data.get("WorkLifeBalance", 3),
        "YearsAtCompany":           raw_data["YearsAtCompany"],
        "YearsInCurrentRole":       3,             # median baseline
        "YearsSinceLastPromotion":  raw_data.get("YearsSinceLastPromotion", 1),
        "YearsWithCurrManager":     3,             # median baseline
        # Consistent Engineered Features
        "IncomePerYear":            raw_data["MonthlyIncome"] * 12,
        "ExperienceRatio":          raw_data["YearsAtCompany"] / (raw_data["TotalWorkingYears"] + 1),
    }
    df = pd.DataFrame([row])
    return df[feature_names]

# ── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px;">
        <div style="background:#4338CA;color:white;font-weight:700;font-size:18px;width:38px;height:38px;border-radius:10px;display:flex;align-items:center;justify-content:center;">
            PA
        </div>
        <div>
            <div style="font-weight:700;font-size:15px;color:#1E2235;">PeopleAnalytics</div>
            <div style="font-size:11px;color:#697386;">Retention Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    navigation = st.radio(
        "Navigation Menu",
        options=[
            "📊 Overview",
            "📈 Workforce Analytics",
            "🎯 Predict Attrition",
            "🔍 Explain Predictions",
            "🧪 What-If Simulation",
            "🏆 Model Performance",
            "ℹ️ How It Works"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:12px;font-size:12px;color:#64748B;">
        <b style="color:#1E2235;">Platform Info</b><br>
        • Model: Logistic Regression<br>
        • Strategy: SMOTE Balanced<br>
        • Primary Metric: Recall (78.7%)<br>
        • Status: Production Ready
    </div>
    """, unsafe_allow_html=True)

# ── Top SaaS Header Banner ────────────────────────────────────────────────────
st.markdown("""
<div class="header-container">
    <div class="eyebrow-label">People Analytics & Retention Intelligence</div>
    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
        <div>
            <h1 class="header-title">Employee Attrition Intelligence</h1>
            <p class="header-subtitle">
                Understand workforce dynamics, identify flight risk early, and simulate proactive retention strategies.
            </p>
        </div>
        <div style="text-align:right;">
            <div class="status-pill"><span class="status-dot"></span>Analytics Engine Ready</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not engine_ready:
    st.error(f"⚠️ Engine Error: Could not load trained models: {engine_error}")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW PAGE
# ══════════════════════════════════════════════════════════════════════════════
if navigation == "📊 Overview":
    # Calculate real dataset KPIs
    total_emp = len(df_raw)
    left_emp = int((df_raw["Attrition"] == "Yes").sum())
    attrition_rate = (left_emp / total_emp) * 100
    avg_tenure = df_raw["YearsAtCompany"].mean()
    
    # KPI Grid
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-title">Total Headcount</p>
            <p class="kpi-value">{total_emp:,}</p>
            <p class="kpi-subtext">Active & past employee records</p>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-title">Departed Employees</p>
            <p class="kpi-value" style="color:#E5484D;">{left_emp:,}</p>
            <p class="kpi-subtext">Recorded voluntary attrition</p>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-title">Overall Attrition Rate</p>
            <p class="kpi-value">{attrition_rate:.1f}%</p>
            <p class="kpi-subtext">Baseline organizational rate</p>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-title">Average Company Tenure</p>
            <p class="kpi-value">{avg_tenure:.1f} yrs</p>
            <p class="kpi-subtext">Organizational average tenure</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # Core Overview Visuals
    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Overall Retention vs Attrition")
        st.caption("Distribution of employees who stayed vs those who left.")
        stay_count = total_emp - left_emp
        fig_donut = go.Figure(go.Pie(
            labels=["Retained (Stay)", "Turnover (Leave)"],
            values=[stay_count, left_emp],
            hole=0.62,
            marker=dict(colors=["#16A673", "#E5484D"]),
            textinfo="label+percent",
            textfont=dict(color="#FFFFFF", size=13),
        ))
        fig_donut.update_layout(
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            margin=dict(t=10, b=10, l=10, r=10), height=260,
            showlegend=True, legend=dict(orientation="h", y=-0.1)
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("### 🏢 Attrition by Department")
        st.caption("Turnover headcounts and rate comparisons across divisions.")
        dept_summary = df_raw.groupby("Department")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        dept_summary.columns = ["Department", "AttritionRate"]
        fig_dept = go.Figure(go.Bar(
            x=dept_summary["Department"],
            y=dept_summary["AttritionRate"],
            marker=dict(color=["#6D5AE6", "#4338CA", "#E9A23B"], cornerradius=6),
            text=[f"{r:.1f}%" for r in dept_summary["AttritionRate"]],
            textposition="outside",
            textfont=dict(color="#1E2235", size=11)
        ))
        fig_dept.update_layout(
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            yaxis=dict(title="Turnover Rate (%)", range=[0, 30], gridcolor="#F1F3F9", tickfont=dict(color="#697386")),
            xaxis=dict(tickfont=dict(color="#1E2235")),
            margin=dict(t=10, b=10, l=10, r=10), height=260
        )
        st.plotly_chart(fig_dept, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Key Insights Callout
    st.markdown("""
    <div class="saas-card">
        <h4 style="margin-top:0;color:#1E2235;">💡 Executive Takeaways</h4>
        <div style="display:grid;grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px;">
            <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:14px;">
                <b style="color:#E5484D;">Sales Department Risk:</b> Sales displays the highest turnover rate (~20.6%), driven primarily by high commute distances and frequent client travel demands.
            </div>
            <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:14px;">
                <b style="color:#6D5AE6;">Overtime Impact:</b> Employees working regular overtime leave at a rate exceeding 30%, more than triple their non-overtime peers.
            </div>
            <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:14px;">
                <b style="color:#16A673;">Tenure Stabilizer:</b> Employees with over 5 years tenure at the firm show significantly lower flight risk, highlighting the value of early-tenure retention programs.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2. WORKFORCE ANALYTICS PAGE (WITH FUNCTIONAL FILTERS)
# ══════════════════════════════════════════════════════════════════════════════
elif navigation == "📈 Workforce Analytics":
    st.markdown("### 📈 Interactive Workforce Exploration")
    st.caption("Slice and examine empirical turnover factors across departments and roles.")

    # Filter Bar
    with st.expander("🔍 Filter Workforce Records", expanded=True):
        f_c1, f_c2, f_c3 = st.columns(3)
        with f_c1:
            sel_dept = st.multiselect("Department:", options=list(df_raw["Department"].unique()), default=[])
        with f_c2:
            sel_ot = st.multiselect("Overtime Status:", options=list(df_raw["OverTime"].unique()), default=[])
        with f_c3:
            sel_travel = st.multiselect("Business Travel:", options=list(df_raw["BusinessTravel"].unique()), default=[])

    filtered_df = df_raw.copy()
    if sel_dept:
        filtered_df = filtered_df[filtered_df["Department"].isin(sel_dept)]
    if sel_ot:
        filtered_df = filtered_df[filtered_df["OverTime"].isin(sel_ot)]
    if sel_travel:
        filtered_df = filtered_df[filtered_df["BusinessTravel"].isin(sel_travel)]

    st.markdown(f"**Showing {len(filtered_df):,} of {len(df_raw):,} employees** matching current criteria.")

    w1, w2 = st.columns([1, 1], gap="medium")
    with w1:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("#### Overtime Turnover Comparison")
        ot_stat = filtered_df.groupby("OverTime")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        ot_stat.columns = ["Overtime", "Rate"]
        fig_ot = go.Figure(go.Bar(
            x=ot_stat["Overtime"],
            y=ot_stat["Rate"],
            marker=dict(color=["#16A673", "#E5484D"], cornerradius=6),
            text=[f"{r:.1f}%" for r in ot_stat["Rate"]],
            textposition="outside"
        ))
        fig_ot.update_layout(
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            yaxis=dict(title="Turnover Rate (%)", gridcolor="#F1F3F9", tickfont=dict(color="#697386")),
            xaxis=dict(title="Overtime Status", tickfont=dict(color="#1E2235")),
            height=280, margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_ot, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with w2:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("#### Attrition Rate by Job Satisfaction")
        sat_stat = filtered_df.groupby("JobSatisfaction")["Attrition"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        sat_stat.columns = ["Score", "Rate"]
        fig_sat = go.Figure(go.Bar(
            x=[f"Level {s}" for s in sat_stat["Score"]],
            y=sat_stat["Rate"],
            marker=dict(color=["#E5484D", "#E9A23B", "#6D5AE6", "#16A673"], cornerradius=6),
            text=[f"{r:.1f}%" for r in sat_stat["Rate"]],
            textposition="outside"
        ))
        fig_sat.update_layout(
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            yaxis=dict(title="Turnover Rate (%)", gridcolor="#F1F3F9", tickfont=dict(color="#697386")),
            xaxis=dict(title="Job Satisfaction Score (1-4)", tickfont=dict(color="#1E2235")),
            height=280, margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_sat, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # Monthly Income Distribution by Attrition
    st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
    st.markdown("#### Monthly Income Distribution (Stay vs Leave)")
    fig_hist = px.histogram(
        filtered_df, x="MonthlyIncome", color="Attrition",
        barmode="overlay", color_discrete_map={"No": "#16A673", "Yes": "#E5484D"},
        labels={"MonthlyIncome": "Monthly Income (₹)", "Attrition": "Turnover Status"}
    )
    fig_hist.update_layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        xaxis=dict(gridcolor="#F1F3F9", tickfont=dict(color="#697386")),
        yaxis=dict(gridcolor="#F1F3F9", tickfont=dict(color="#697386")),
        height=300, margin=dict(t=10, b=10, l=10, r=10)
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. PREDICT ATTRITION PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif navigation == "🎯 Predict Attrition":
    st.markdown("### 🎯 Assess Individual Employee Flight Risk")
    st.caption("Provide employee details below to generate a model-estimated flight risk score and tailored recommendations.")

    # Organized Form Layout
    with st.form("employee_assessment_form"):
        st.markdown("#### 1. Demographic & Personal Attributes")
        f1, f2, f3 = st.columns(3)
        with f1:
            age = st.number_input("Age (Years)", 18, 60, 30)
        with f2:
            gender = st.selectbox("Gender", encoders["Gender"].classes_)
        with f3:
            marital_status = st.selectbox("Marital Status", encoders["MaritalStatus"].classes_)

        st.markdown("#### 2. Job Role & Work Environment")
        j1, j2, j3 = st.columns(3)
        with j1:
            department = st.selectbox("Department", encoders["Department"].classes_)
        with j2:
            job_role = st.selectbox("Job Role", encoders["JobRole"].classes_)
        with j3:
            education_field = st.selectbox("Education Field", encoders["EducationField"].classes_)

        j4, j5, j6 = st.columns(3)
        with j4:
            business_travel = st.selectbox("Business Travel Frequency", encoders["BusinessTravel"].classes_)
        with j5:
            overtime = st.selectbox("Regular Overtime?", encoders["OverTime"].classes_)
        with j6:
            distance_from_home = st.slider("Distance from Home (km)", 1, 30, 5)

        st.markdown("#### 3. Compensation & Career Tenure")
        c1, c2, c3 = st.columns(3)
        with c1:
            monthly_income = st.slider("Monthly Income (₹)", 1000, 50000, 5000, step=500)
        with c2:
            job_level = st.slider("Job Seniority Level (1-5)", 1, 5, 2)
        with c3:
            stock_option_level = st.slider("Stock Option Level (0-3)", 0, 3, 1)

        c4, c5, c6 = st.columns(3)
        with c4:
            total_working_years = st.slider("Total Working Experience (Years)", 0, 40, 8)
        with c5:
            years_at_company = st.slider("Tenure at Current Company (Years)", 0, 40, 5)
        with c6:
            years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)

        st.markdown("#### 4. Workplace Satisfaction Ratings (Scale 1–4)")
        s1, s2, s3 = st.columns(3)
        with s1:
            job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        with s2:
            environment_satisfaction = st.slider("Work Environment Satisfaction", 1, 4, 3)
        with s3:
            work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)

        submit_btn = st.form_submit_button("🚀 Analyze Flight Risk & Influential Factors")

    if submit_btn:
        employee_data = {
            "Age": age, "Gender": gender, "MaritalStatus": marital_status,
            "Department": department, "JobRole": job_role, "EducationField": education_field,
            "BusinessTravel": business_travel, "OverTime": overtime,
            "MonthlyIncome": monthly_income, "JobSatisfaction": job_satisfaction,
            "EnvironmentSatisfaction": environment_satisfaction, "WorkLifeBalance": work_life_balance,
            "StockOptionLevel": stock_option_level, "JobLevel": job_level,
            "TotalWorkingYears": total_working_years, "YearsAtCompany": years_at_company,
            "YearsSinceLastPromotion": years_since_last_promotion,
            "DistanceFromHome": distance_from_home,
        }
        
        # Build features and scale
        X_df = build_features_from_dict(employee_data)
        X_scaled = scaler.transform(X_df)
        
        # Prediction
        proba = model.predict_proba(X_scaled)[0]
        risk_score, stay_pct = compute_attrition_risk(proba, model.classes_)
        risk_tier = categorize_risk(risk_score)
        
        # SHAP calculation
        shap_res = explain_prediction(explainer, X_scaled, feature_names, top_k=5)
        
        # Recommendations
        recs = generate_retention_recommendations(employee_data)
        
        # Save in session
        st.session_state.current_prediction = {
            "employee": employee_data,
            "risk_score": risk_score,
            "stay_pct": stay_pct,
            "tier": risk_tier,
            "shap_res": shap_res,
            "recs": recs,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }

    # Display results if available
    if "current_prediction" in st.session_state:
        res = st.session_state.current_prediction
        tier = res["tier"]
        
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        st.markdown("### 📋 Risk Assessment Results")
        
        # Top Result Card
        st.markdown(f"""
        <div style="background:{tier['bg_color']};border:1.5px solid {tier['border_color']};border-radius:16px;padding:22px 26px;margin-bottom:20px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <span style="background:{tier['color']};color:white;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:700;">
                        {tier['tier'].upper()} RISK
                    </span>
                    <h2 style="color:#1E2235;margin:10px 0 4px 0;font-size:32px;font-weight:700;">
                        {res['risk_score']}% <span style="font-size:16px;color:#697386;font-weight:400;">Attrition Probability</span>
                    </h2>
                    <p style="color:#334155;margin:0;font-size:14px;">{tier['verdict']}</p>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:12px;color:#697386;">Assessment Time: {res['timestamp']}</div>
                    <div style="font-size:13px;font-weight:600;color:#1E2235;margin-top:4px;">Model: Logistic Regression + SMOTE</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1, 1], gap="medium")
        with res_col1:
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            st.markdown("#### Top Driving Factors (SHAP Attribution)")
            st.caption("Factors pushing the model's estimate toward attrition (Red) or retention (Green).")
            shap_fig = create_shap_bar_chart(res["shap_res"]["top_features"])
            st.plotly_chart(shap_fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with res_col2:
            st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
            st.markdown("#### 💡 Tailored Retention Recommendations")
            st.caption("Constructive, transparent suggestions based on identified friction points.")
            style_map = {"danger": "rec-card-danger", "warning": "rec-card-warn", "info": "rec-card-info", "success": "rec-card-success"}
            for rec in res["recs"]:
                cls = style_map.get(rec["level"], "rec-card-info")
                st.markdown(f"""
                <div class="{cls}">
                    <b>{rec['icon']} {rec['title']}</b> <i>({rec['category']})</i><br>
                    {rec['action']}
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4. EXPLAIN PREDICTIONS (SHAP DEEP DIVE)
# ══════════════════════════════════════════════════════════════════════════════
elif navigation == "🔍 Explain Predictions":
    st.markdown("### 🔍 Model Explainability & SHAP Decomposition")
    st.caption("Inspect the mathematical reasoning behind the Logistic Regression predictions.")

    st.markdown("""
    <div class="info-callout">
        <b>Why Explainability Matters in HR:</b> Black-box predictions create skepticism. By using 
        <b>SHAP (SHapley Additive exPlanations)</b>, we decompose the log-odds prediction into exact feature 
        contributions without altering the trained model.
    </div>
    """, unsafe_allow_html=True)

    e1, e2 = st.columns([1, 1], gap="medium")
    with e1:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("#### Global Model Weights (Top Features)")
        st.caption("Overall coefficient magnitude in the Logistic Regression decision boundary.")
        global_fig = create_global_feature_importance_chart(model, feature_names, top_k=10)
        st.plotly_chart(global_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with e2:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("#### Individual Assessment Explanation")
        if "current_prediction" in st.session_state:
            curr = st.session_state.current_prediction
            st.caption(f"Decomposition for current assessment (Risk: {curr['risk_score']}%)")
            shap_fig2 = create_shap_bar_chart(curr["shap_res"]["top_features"])
            st.plotly_chart(shap_fig2, use_container_width=True, config={"displayModeBar": False})
            
            with st.expander("🔎 View All Feature Attributions"):
                st.dataframe(
                    curr["shap_res"]["all_shap_df"][["Feature", "SHAPValue", "PushesToward"]],
                    use_container_width=True, hide_index=True
                )
        else:
            st.info("No individual employee assessment has been generated yet. Please visit **Predict Attrition** to assess a profile.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="saas-card">
        <h4 style="margin-top:0;color:#1E2235;">📖 Understanding the Mathematical Output</h4>
        <p style="font-size:13.5px;color:#64748B;line-height:1.6;">
            The model computes the linear score $z = \\beta_0 + \\sum \\beta_i X_i$. 
            The SHAP value $\\phi_i$ expresses how much feature $i$ shifted $z$ away from the baseline expectation.
            <br>• <b>Positive SHAP (Red):</b> Pushes $z$ upwards toward the Attrition threshold.
            <br>• <b>Negative SHAP (Green):</b> Pulls $z$ downwards toward the Retention baseline.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 5. WHAT-IF SIMULATION PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif navigation == "🧪 What-If Simulation":
    st.markdown("### 🧪 What-If Risk Sensitivity Simulation")
    st.caption("Modify key workplace levers to observe how the model's estimated flight risk changes in real-time.")

    # Base record from session or standard baseline
    if "current_prediction" in st.session_state:
        base_emp = st.session_state.current_prediction["employee"].copy()
    else:
        base_emp = {
            "Age": 30, "Gender": "Male", "MaritalStatus": "Single",
            "Department": "Sales", "JobRole": "Sales Executive", "EducationField": "Life Sciences",
            "BusinessTravel": "Travel_Frequently", "OverTime": "Yes",
            "MonthlyIncome": 3500, "JobSatisfaction": 2, "EnvironmentSatisfaction": 2,
            "WorkLifeBalance": 2, "StockOptionLevel": 0, "JobLevel": 2,
            "TotalWorkingYears": 7, "YearsAtCompany": 4, "YearsSinceLastPromotion": 3,
            "DistanceFromHome": 18,
        }

    # Calculate baseline
    base_df = build_features_from_dict(base_emp)
    base_scaled = scaler.transform(base_df)
    base_proba = model.predict_proba(base_scaled)[0]
    base_risk, _ = compute_attrition_risk(base_proba, model.classes_)
    base_tier = categorize_risk(base_risk)

    sim_c1, sim_c2 = st.columns([1, 1], gap="large")

    with sim_c1:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("#### 🛠️ Adjust Workplace Levers")
        
        sim_ot = st.selectbox("Overtime Status:", ["No", "Yes"], index=0 if base_emp["OverTime"] == "No" else 1)
        sim_js = st.slider("Job Satisfaction (1-4):", 1, 4, int(base_emp["JobSatisfaction"]))
        sim_es = st.slider("Work Environment Satisfaction (1-4):", 1, 4, int(base_emp["EnvironmentSatisfaction"]))
        sim_wlb = st.slider("Work-Life Balance (1-4):", 1, 4, int(base_emp.get("WorkLifeBalance", 3)))
        sim_inc = st.slider("Monthly Income (₹):", 1000, 50000, int(base_emp["MonthlyIncome"]), step=500)
        sim_promo = st.slider("Years Since Last Promotion:", 0, 15, int(base_emp.get("YearsSinceLastPromotion", 1)))
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Compute simulation
    sim_emp = base_emp.copy()
    sim_emp["OverTime"] = sim_ot
    sim_emp["JobSatisfaction"] = sim_js
    sim_emp["EnvironmentSatisfaction"] = sim_es
    sim_emp["WorkLifeBalance"] = sim_wlb
    sim_emp["MonthlyIncome"] = sim_inc
    sim_emp["YearsSinceLastPromotion"] = sim_promo

    sim_df = build_features_from_dict(sim_emp)
    sim_scaled = scaler.transform(sim_df)
    sim_proba = model.predict_proba(sim_scaled)[0]
    sim_risk, _ = compute_attrition_risk(sim_proba, model.classes_)
    sim_tier = categorize_risk(sim_risk)
    
    delta = round(sim_risk - base_risk, 1)

    with sim_c2:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Simulation Outcome")
        
        m_c1, m_c2, m_c3 = st.columns(3)
        m_c1.metric("Original Risk", f"{base_risk}%")
        m_c2.metric("Simulated Risk", f"{sim_risk}%")
        m_c3.metric("Change", f"{delta:+0.1f}% pts", delta=f"{delta}%", delta_color="inverse")

        st.markdown(f"""
        <div style="background:{sim_tier['bg_color']};border:1px solid {sim_tier['border_color']};border-radius:12px;padding:16px;margin:16px 0;">
            <p style="margin:0 0 4px;font-size:13px;color:#697386;">Simulated Tier: <b style="color:{sim_tier['color']};">{sim_tier['label']}</b></p>
            <p style="margin:0;font-size:14px;color:#1E2235;">
                {'✅ The proposed adjustments achieved a <b>' + str(abs(delta)) + '% reduction</b> in estimated attrition probability.' if delta < 0 else '⚠️ The proposed changes resulted in an increased flight risk estimate.' if delta > 0 else 'No change in estimated risk.'}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Comparison Bar Chart
        comp_fig = go.Figure(data=[
            go.Bar(name="Original Scenario", x=["Attrition Risk"], y=[base_risk],
                   marker_color=base_tier["color"], text=[f"{base_risk}%"], textposition="auto"),
            go.Bar(name="Simulated Scenario", x=["Attrition Risk"], y=[sim_risk],
                   marker_color=sim_tier["color"], text=[f"{sim_risk}%"], textposition="auto")
        ])
        comp_fig.update_layout(
            barmode="group",
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            yaxis=dict(range=[0, 100], title="Predicted Risk (%)", gridcolor="#F1F3F9", tickfont=dict(color="#697386")),
            xaxis=dict(tickfont=dict(color="#1E2235")),
            height=240, margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(comp_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 6. MODEL PERFORMANCE PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif navigation == "🏆 Model Performance":
    st.markdown("### 🏆 Rigorous Model Evaluation (Held-Out Test Set)")
    st.caption("Measured performance on a 20% stratified test split (N = 294, 47 true attrition cases).")

    # True test metrics
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Recall (Attrition = 1)", "78.7%", help="Catches nearly 8 out of 10 actual departures.")
    p2.metric("ROC-AUC Score", "0.813", help="High discriminative power across all thresholds.")
    p3.metric("Overall Accuracy", "76.5%")
    p4.metric("F1-Score", "0.518")

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    c_cm, c_bench = st.columns([1, 1], gap="medium")
    with c_cm:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("#### Test Confusion Matrix (N = 294)")
        # Actual values from evaluation: TN=188, FP=59, FN=10, TP=37
        cm_data = [[188, 59], [10, 37]]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=["Predicted Stay", "Predicted Leave"],
            y=["Actual Stay", "Actual Leave"],
            colorscale=[[0, "#EEEAFF"], [1, "#4338CA"]],
            text=[[f"TN: 188<br>(True Negative)", f"FP: 59<br>(False Alarm)"],
                  [f"FN: 10<br>(Missed Risk)", f"TP: 37<br>(Correct Catch)"]],
            texttemplate="%{text}",
            textfont={"size": 13, "color": "#1E2235"}
        ))
        fig_cm.update_layout(
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            height=280, margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
        st.caption("⚠️ Notice only **10 False Negatives** out of 47 departures, preserving workforce security.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_bench:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("#### Model Benchmark Comparison")
        bench_df = pd.DataFrame([
            {"Model": "Logistic Regression + SMOTE", "Accuracy": "76.5%", "Recall": "78.7%", "ROC-AUC": "0.813", "Status": "Selected 🏆"},
            {"Model": "Class-Weight Balanced LR", "Accuracy": "75.5%", "Recall": "76.6%", "ROC-AUC": "0.813", "Status": "Alternative"},
            {"Model": "Random Forest + SMOTE", "Accuracy": "84.4%", "Recall": "25.5%", "ROC-AUC": "0.809", "Status": "Poor Recall ⚠️"},
            {"Model": "Decision Tree + SMOTE", "Accuracy": "78.9%", "Recall": "38.3%", "ROC-AUC": "0.625", "Status": "Suboptimal"},
        ])
        st.dataframe(bench_df, hide_index=True, use_container_width=True)
        st.markdown("""
        <p style="font-size:12.5px;color:#64748B;line-height:1.5;margin-top:10px;">
            <b>Why Recall Matters:</b> In HR analytics, a False Negative (missing an employee who resigns) costs up to 1.5–2× their annual salary in replacement costs. Random Forest's 25.5% recall is unacceptable for retention triage, making Logistic Regression with SMOTE the proven winner.
        </p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 7. HOW IT WORKS PAGE (EDUCATIONAL ARCHITECTURE)
# ══════════════════════════════════════════════════════════════════════════════
elif navigation == "ℹ️ How It Works":
    st.markdown("### ℹ️ System Architecture & Presentation Guide")
    st.caption("Complete, transparent documentation for technical presentations and viva evaluations.")

    st.markdown("""
    <div class="saas-card">
        <h4 style="margin-top:0;color:#1E2235;">🔄 End-to-End Analytics Pipeline</h4>
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin:16px 0;">
            <div style="background:#F1F3F9;padding:12px 16px;border-radius:10px;font-size:13px;font-weight:600;">
                1. Data Cleaning & Drop Non-Informative Columns
            </div>
            →
            <div style="background:#F1F3F9;padding:12px 16px;border-radius:10px;font-size:13px;font-weight:600;">
                2. Feature Engineering (Income & Tenure Ratios)
            </div>
            →
            <div style="background:#F1F3F9;padding:12px 16px;border-radius:10px;font-size:13px;font-weight:600;">
                3. LabelEncoding & StandardScaler
            </div>
            →
            <div style="background:#F1F3F9;padding:12px 16px;border-radius:10px;font-size:13px;font-weight:600;">
                4. SMOTE Training Resampling
            </div>
            →
            <div style="background:#F1F3F9;padding:12px 16px;border-radius:10px;font-size:13px;font-weight:600;">
                5. Logistic Regression + SHAP Attribution
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    hw1, hw2 = st.columns([1, 1], gap="medium")
    with hw1:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("#### 🎯 Mathematical Foundations")
        st.markdown("""
        - **Logistic Hypothesis:**
          $$P(Y=1 \\mid X) = \\frac{1}{1 + e^{-z}} \\quad \\text{where} \\quad z = \\beta_0 + \\sum_{i=1}^p \\beta_i X_i$$
        - **Risk Score Formulation:**
          $$\\text{Risk Score (\\%)} = P(\\text{Attrition}=1 \\mid X) \\times 100$$
        - **SHAP Additive Decomposition:**
          $$\\ln\\left(\\frac{P}{1-P}\\right) = \\phi_0 + \\sum_{i=1}^M \\phi_i$$
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with hw2:
        st.markdown("<div class='saas-card'>", unsafe_allow_html=True)
        st.markdown("#### ⚖️ Ethical Guidelines & Responsible AI")
        st.markdown("""
        1. **Observational, Not Causal:** SHAP attributions reflect statistical relationships in historical data, not absolute proof of why an individual resigns.
        2. **Supportive Actions Only:** Recommendations focus on supportive retention check-ins and workload mitigation. The system must never be used for automated or punitive actions.
        3. **Confidence & Calibration:** Model scores serve as ranking and triage aids; final workforce decisions always require human empathy and dialogue.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:28px 0 10px;color:#929AAF;font-size:13px;">
    Developed by <strong style="color:#4338CA">Akash Mishra</strong>
    · PeopleAnalytics Retention Intelligence Platform · Clean Light-Mode SaaS Edition
</div>
""", unsafe_allow_html=True)