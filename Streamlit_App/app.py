import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# Import modular upgrades
from explainability import create_shap_explainer, explain_prediction, create_shap_bar_chart
from recommendations import generate_retention_recommendations
from risk_scoring import compute_attrition_risk, categorize_risk

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Retention Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (Preserves dark styling & clean readable input cards) ──────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0B1120 0%, #0F2444 60%, #1a3a6b 100%);
}
[data-testid="stSidebar"] {
    background: rgba(10, 18, 38, 0.97);
    border-right: 1px solid rgba(56, 138, 220, 0.18);
}
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    color: #94A3B8 !important; font-size: 13px;
}

/* Sidebar inputs — white background, black text */
[data-testid="stSidebar"] div[data-baseweb="select"] {
    background: #ffffff !important; border-radius: 8px !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] * { color: #0f172a !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] svg { fill: #0f172a !important; }
[data-testid="stSidebar"] div[data-baseweb="select"] > div { background: #ffffff !important; }
[data-testid="stSidebar"] div[data-baseweb="input"],
[data-testid="stSidebar"] div[data-baseweb="base-input"] {
    background: #ffffff !important; border-radius: 8px !important;
}
[data-testid="stSidebar"] div[data-baseweb="input"] input,
[data-testid="stSidebar"] div[data-baseweb="base-input"] input {
    color: #0f172a !important; background: #ffffff !important; font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stTickBarMax"],
[data-testid="stSidebar"] .stSlider p { color: #CBD5E1 !important; }
ul[data-testid="stSelectboxVirtualDropdown"] li,
ul[data-testid="stSelectboxVirtualDropdown"] li span { color: #0f172a !important; }

/* Metric cards — white background, dark text */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(56,138,220,0.3);
    border-radius: 14px;
    padding: 16px 18px;
}
div[data-testid="metric-container"] * { color: #000000 !important; }
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] [data-testid="stMetricLabel"],
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p,
div[data-testid="metric-container"] [data-testid="stMetricLabel"] span {
    color: #1e293b !important; font-size: 13px !important; font-weight: 500 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"],
div[data-testid="metric-container"] [data-testid="stMetricValue"] > div,
div[data-testid="metric-container"] [data-testid="stMetricValue"] p {
    color: #000000 !important; font-size: 22px !important; font-weight: 700 !important;
}

/* Primary buttons */
.stButton > button {
    width: 100%; height: 50px; border-radius: 12px; border: none;
    background: linear-gradient(90deg, #1E6FD9, #0EA5E9);
    color: white; font-weight: 600; font-size: 16px; transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* Section headers */
.section-header {
    font-size: 11px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #64748B;
    margin: 18px 0 10px; padding-bottom: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

/* Recommendation cards */
.rec-card   { background:rgba(30,111,217,0.12); border:1px solid rgba(30,111,217,0.25);  border-radius:10px; padding:12px 16px; margin-bottom:8px; color:#BAD4F7; font-size:14px; }
.warn-card  { background:rgba(239,159,39,0.12); border:1px solid rgba(239,159,39,0.25);  border-radius:10px; padding:12px 16px; margin-bottom:8px; color:#FCD580; font-size:14px; }
.danger-card{ background:rgba(226,75,74,0.12);  border:1px solid rgba(226,75,74,0.30);   border-radius:10px; padding:12px 16px; margin-bottom:8px; color:#FAAAA9; font-size:14px; }
.success-card{background:rgba(99,153,34,0.12); border:1px solid rgba(99,153,34,0.30);   border-radius:10px; padding:12px 16px; margin-bottom:8px; color:#B4DD80; font-size:14px; }
.model-badge{background:rgba(99,153,34,0.15);  border:1px solid rgba(99,153,34,0.35);   border-radius:8px;  padding:6px 14px; color:#B4DD80; font-size:13px; display:inline-block; margin-bottom:12px;}
.info-card  {background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:10px; padding:12px 16px; margin-bottom:12px; font-size:13px; color:#CBD5E1;}

footer{visibility:hidden;} #MainMenu{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model & explainer...")
def load_artifacts():
    model         = joblib.load("../Model/attrition_model.pkl") if False else joblib.load("Model/attrition_model.pkl")
    scaler        = joblib.load("Model/scaler.pkl")
    encoders      = joblib.load("Model/encoders.pkl")
    feature_names = joblib.load("Model/feature_names.pkl")
    # Initialize SHAP explainer once and cache it
    explainer     = create_shap_explainer(model, feature_names)
    return model, scaler, encoders, feature_names, explainer

try:
    model, scaler, encoders, feature_names, explainer = load_artifacts()
    model_loaded = True
except Exception as e:
    # Try relative fallback
    try:
        model         = joblib.load("../Model/attrition_model.pkl")
        scaler        = joblib.load("../Model/scaler.pkl")
        encoders      = joblib.load("../Model/encoders.pkl")
        feature_names = joblib.load("../Model/feature_names.pkl")
        explainer     = create_shap_explainer(model, feature_names)
        model_loaded  = True
    except Exception as e2:
        model_loaded = False
        load_error = str(e2)

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);
            border-radius:18px;padding:22px 28px;margin-bottom:20px;">
    <p style="color:#64748B;font-size:12px;letter-spacing:.1em;text-transform:uppercase;margin:0 0 4px">
        Developed by Akash Mishra
    </p>
    <h1 style="color:white;margin:0;font-size:26px;font-weight:700;">
        📊 Employee Retention Intelligence Platform
    </h1>
    <p style="color:#94A3B8;margin:6px 0 0;font-size:14px;">
        Predict attrition flight risk, explain predictions with SHAP attribution, and simulate proactive retention actions.
    </p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Could not load model files: {load_error}")
    st.info("Ensure `attrition_model.pkl`, `scaler.pkl`, `encoders.pkl`, and `feature_names.pkl` exist in `Model/`.")
    st.stop()

# ── Sidebar Inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👤 Employee Profile")

    st.markdown('<p class="section-header">Demographics</p>', unsafe_allow_html=True)
    age            = st.number_input("Age", 18, 60, 30)
    gender         = st.selectbox("Gender",         encoders["Gender"].classes_)
    marital_status = st.selectbox("Marital Status", encoders["MaritalStatus"].classes_)

    st.markdown('<p class="section-header">Work Details</p>', unsafe_allow_html=True)
    department      = st.selectbox("Department",      encoders["Department"].classes_)
    job_role        = st.selectbox("Job Role",         encoders["JobRole"].classes_)
    education_field = st.selectbox("Education Field", encoders["EducationField"].classes_)
    business_travel = st.selectbox("Business Travel", encoders["BusinessTravel"].classes_)
    overtime        = st.selectbox("Overtime",         encoders["OverTime"].classes_)

    st.markdown('<p class="section-header">Compensation & Satisfaction</p>', unsafe_allow_html=True)
    monthly_income           = st.slider("Monthly Income (₹)",           1000, 50000, 5000, step=500)
    job_satisfaction         = st.slider("Job Satisfaction (1–4)",        1, 4, 3)
    environment_satisfaction = st.slider("Environment Satisfaction (1–4)", 1, 4, 3)
    work_life_balance        = st.slider("Work-Life Balance (1–4)",       1, 4, 3)
    stock_option_level       = st.slider("Stock Option Level (0–3)",      0, 3, 1)

    st.markdown('<p class="section-header">Career & Tenure</p>', unsafe_allow_html=True)
    job_level                  = st.slider("Job Level (1–5)",           1, 5, 2)
    total_working_years        = st.slider("Total Working Years",       0, 40, 8)
    years_at_company           = st.slider("Years at Company",          0, 40, 5)
    years_since_last_promotion = st.slider("Years Since Last Promotion",0, 15, 1)
    distance_from_home         = st.slider("Distance from Home (km)",   1, 30, 5)

    predict_clicked = st.button("🚀 Predict & Explain Risk")

# ── Feature Builder ───────────────────────────────────────────────────────────
def build_features_from_dict(raw_data) -> pd.DataFrame:
    """
    Constructs a DataFrame matching the exact feature order of feature_names.pkl.
    """
    row = {
        "Age":                      raw_data["Age"],
        "BusinessTravel":           encoders["BusinessTravel"].transform([raw_data["BusinessTravel"]])[0],
        "DailyRate":                800,           # median default
        "Department":               encoders["Department"].transform([raw_data["Department"]])[0],
        "DistanceFromHome":         raw_data["DistanceFromHome"],
        "Education":                3,             # median default
        "EducationField":           encoders["EducationField"].transform([raw_data["EducationField"]])[0],
        "EnvironmentSatisfaction":  raw_data["EnvironmentSatisfaction"],
        "Gender":                   encoders["Gender"].transform([raw_data["Gender"]])[0],
        "HourlyRate":               65,            # median default
        "JobInvolvement":           3,             # median default
        "JobLevel":                 raw_data["JobLevel"],
        "JobRole":                  encoders["JobRole"].transform([raw_data["JobRole"]])[0],
        "JobSatisfaction":          raw_data["JobSatisfaction"],
        "MaritalStatus":            encoders["MaritalStatus"].transform([raw_data["MaritalStatus"]])[0],
        "MonthlyIncome":            raw_data["MonthlyIncome"],
        "MonthlyRate":              14000,         # median default
        "NumCompaniesWorked":       2,             # median default
        "OverTime":                 encoders["OverTime"].transform([raw_data["OverTime"]])[0],
        "PercentSalaryHike":        15,            # median default
        "PerformanceRating":        3,             # median default
        "RelationshipSatisfaction": 3,             # median default
        "StockOptionLevel":         raw_data["StockOptionLevel"],
        "TotalWorkingYears":        raw_data["TotalWorkingYears"],
        "TrainingTimesLastYear":    2,             # median default
        "WorkLifeBalance":          raw_data.get("WorkLifeBalance", 3),
        "YearsAtCompany":           raw_data["YearsAtCompany"],
        "YearsInCurrentRole":       3,             # median default
        "YearsSinceLastPromotion":  raw_data.get("YearsSinceLastPromotion", 1),
        "YearsWithCurrManager":     3,             # median default
        # Engineered features
        "IncomePerYear":            raw_data["MonthlyIncome"] * 12,
        "ExperienceRatio":          raw_data["YearsAtCompany"] / (raw_data["TotalWorkingYears"] + 1),
    }
    df = pd.DataFrame([row])
    return df[feature_names]

# Raw inputs dictionary for current employee
current_employee = {
    "Age": age,
    "Gender": gender,
    "MaritalStatus": marital_status,
    "Department": department,
    "JobRole": job_role,
    "EducationField": education_field,
    "BusinessTravel": business_travel,
    "OverTime": overtime,
    "MonthlyIncome": monthly_income,
    "JobSatisfaction": job_satisfaction,
    "EnvironmentSatisfaction": environment_satisfaction,
    "WorkLifeBalance": work_life_balance,
    "StockOptionLevel": stock_option_level,
    "JobLevel": job_level,
    "TotalWorkingYears": total_working_years,
    "YearsAtCompany": years_at_company,
    "YearsSinceLastPromotion": years_since_last_promotion,
    "DistanceFromHome": distance_from_home,
}

# ── Visualization Helpers ─────────────────────────────────────────────────────
def gauge_chart(score, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 44, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748B", "tickfont": {"color": "#64748B", "size": 11}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.04)",
            "bordercolor": "rgba(255,255,255,0.08)",
            "steps": [
                {"range": [0,  30],  "color": "rgba(99,153,34,0.12)"},
                {"range": [30, 60],  "color": "rgba(239,159,39,0.12)"},
                {"range": [60, 100], "color": "rgba(226,75,74,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.85, "value": score},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=10, l=20, r=20), height=230, font={"color": "white"},
    )
    return fig

def donut_chart(stay_pct, leave_pct):
    fig = go.Figure(go.Pie(
        labels=["Stay", "Leave"], values=[stay_pct, leave_pct], hole=0.58,
        marker=dict(colors=["#639922", "#E24B4A"]),
        textinfo="label+percent", textfont=dict(color="white", size=13),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10), height=220,
    )
    return fig

# ── Tabbed Dashboard Layout ───────────────────────────────────────────────────
tab_predict, tab_whatif, tab_batch = st.tabs([
    "🎯 Individual Prediction & SHAP",
    "🧪 What-If Risk Simulation",
    "👥 Ranked Employee Overview"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: INDIVIDUAL PREDICTION & SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        # Top Quick Profile Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Monthly Income",  f"₹{monthly_income:,}")
        m2.metric("Department",      department[:14])
        m3.metric("Distance",        f"{distance_from_home} km")
        m4.metric("Overtime",        overtime)

        st.markdown("---")

        # Execute prediction when clicked or show stored results
        if predict_clicked or "last_prediction" in st.session_state:
            if predict_clicked:
                X_df = build_features_from_dict(current_employee)
                X_scaled = scaler.transform(X_df)
                
                # Safe probability extraction using classes_
                proba = model.predict_proba(X_scaled)[0]
                risk_score, stay_pct = compute_attrition_risk(proba, model.classes_)
                risk_info = categorize_risk(risk_score)
                pred_label = "Leave" if risk_score >= 50.0 else "Stay"
                
                # SHAP explanation
                shap_result = explain_prediction(explainer, X_scaled, feature_names, top_k=5)
                
                # Recommendations
                recs = generate_retention_recommendations(current_employee)
                
                # Store in session
                st.session_state.last_prediction = {
                    "risk_score": risk_score,
                    "stay_pct": stay_pct,
                    "risk_info": risk_info,
                    "pred_label": pred_label,
                    "shap_result": shap_result,
                    "recs": recs,
                    "employee": current_employee.copy(),
                }
                
                # Save into history
                st.session_state.history.insert(0, {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Role": job_role,
                    "Department": department,
                    "Risk Score": int(risk_score),
                    "Level": risk_info["label"],
                    "Prediction": pred_label,
                })

            res = st.session_state.last_prediction
            r_info = res["risk_info"]

            # 1. Prediction Overview Cards
            ga, gb = st.columns(2, gap="medium")
            with ga:
                st.markdown(f"### {r_info['label']}")
                st.caption(r_info["verdict"])
                st.plotly_chart(gauge_chart(res["risk_score"], r_info["color"]),
                                use_container_width=True, config={"displayModeBar": False})
            with gb:
                st.caption("Probability breakdown — Stay vs Leave")
                st.plotly_chart(donut_chart(res["stay_pct"], res["risk_score"]),
                                use_container_width=True, config={"displayModeBar": False})
                st.markdown(f"""
                <div style="display:flex;gap:12px;justify-content:center;margin-top:2px">
                    <span style="color:#B4DD80;font-size:13px">● Stay {res['stay_pct']}%</span>
                    <span style="color:#FAAAA9;font-size:13px">● Leave {res['risk_score']}%</span>
                </div>
                """, unsafe_allow_html=True)

            dm1, dm2, dm3, dm4 = st.columns(4)
            dm1.metric("Attrition Risk", f"{res['risk_score']}%")
            dm2.metric("Prediction", f"{res['pred_label']} {'⚠️' if res['pred_label'] == 'Leave' else '✅'}")
            dm3.metric("Retention Prob.", f"{res['stay_pct']}%")
            dm4.metric("Risk Tier", r_info["tier"])

            st.markdown("---")

            # 2. FEATURE 1: SHAP EXPLAINABILITY SECTION
            st.markdown("### 🔍 Model Explainability (SHAP Attribution)")
            st.markdown("""
            <div class="info-card">
                <b>What does this show?</b> SHAP (SHapley Additive exPlanations) explains <i>why</i> the Logistic Regression model made this specific prediction.
                <br>• <span style="color:#FAAAA9;"><b>Positive values (Red)</b></span> push the model output toward <b>Attrition</b>.
                <br>• <span style="color:#B4DD80;"><b>Negative values (Green)</b></span> push the model output toward <b>Retention (Staying)</b>.
                <br><i>Note: SHAP values represent statistical attribution in log-odds space, not guaranteed causal causes.</i>
            </div>
            """, unsafe_allow_html=True)

            # SHAP Bar Chart
            shap_fig = create_shap_bar_chart(res["shap_result"]["top_features"])
            st.plotly_chart(shap_fig, use_container_width=True, config={"displayModeBar": False})

            # Top Factors Bullet Points
            st.markdown("#### 📌 Key Influential Factors for this Employee:")
            for item in res["shap_result"]["top_features"]:
                icon = "🔺" if item["PushesToward"] == "Leave" else "🔻"
                color = "#FAAAA9" if item["PushesToward"] == "Leave" else "#B4DD80"
                st.markdown(
                    f"- **{item['Feature']}**: {icon} <span style='color:{color};'>{item['Direction']}</span> "
                    f"`(Impact: {item['SHAPValue']:+.3f} log-odds)`",
                    unsafe_allow_html=True
                )

            # Detailed SHAP Table Expander
            with st.expander("🔎 View Complete Feature Attribution Table"):
                st.dataframe(
                    res["shap_result"]["all_shap_df"][["Feature", "SHAPValue", "PushesToward"]],
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown("---")

            # 3. FEATURE 3: RETENTION RECOMMENDATIONS SECTION
            st.markdown("### 💡 Tailored Retention Recommendations")
            st.caption("Transparent, rule-based workplace suggestions to support employee engagement.")
            card_styles = {"danger": "danger-card", "warning": "warn-card", "info": "rec-card", "success": "success-card"}
            for rec in res["recs"]:
                cls = card_styles.get(rec["level"], "rec-card")
                st.markdown(f"""
                <div class="{cls}">
                    <b>{rec['icon']} {rec['title']}</b> <i>({rec['category']})</i><br>
                    {rec['action']}
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center;padding:56px 0;color:#475569;">
                <p style="font-size:48px;margin-bottom:12px">📊</p>
                <p style="font-size:17px;font-weight:600;color:#94A3B8;">Ready for Employee Assessment</p>
                <p style="font-size:14px;color:#64748B;max-width:360px;margin:8px auto 0">
                    Adjust employee attributes on the left and click
                    <strong style="color:#60A5FA">Predict & Explain Risk</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        # Model info badge
        st.markdown("""
        <div class="model-badge">
            ✅ Model: Logistic Regression · SMOTE Resampled · Recall Optimized
        </div>
        """, unsafe_allow_html=True)

        # Employee snapshot table
        st.markdown("### 👤 Employee Snapshot")
        snap = pd.DataFrame({
            "Field": ["Age", "Gender", "Marital Status", "Department", "Job Role",
                      "Education Field", "Business Travel", "Overtime", "Monthly Income",
                      "Job Satisfaction", "Work Environment", "Work-Life Balance",
                      "Job Level", "Stock Options", "Total Experience", "Tenure"],
            "Value": [age, gender, marital_status, department, job_role,
                      education_field, business_travel, overtime, f"₹{monthly_income:,}",
                      f"{job_satisfaction}/4", f"{environment_satisfaction}/4", f"{work_life_balance}/4",
                      job_level, stock_option_level,
                      f"{total_working_years} yrs", f"{years_at_company} yrs"],
        })
        st.dataframe(snap, hide_index=True, use_container_width=True)

        st.markdown("---")

        # Prediction History Table
        st.markdown("### 🕐 Recent Assessment History")
        if not st.session_state.history:
            st.caption("No assessments run in this session.")
        else:
            df_hist = pd.DataFrame(st.session_state.history)
            df_hist.index = range(1, len(df_hist) + 1)
            st.dataframe(
                df_hist,
                use_container_width=True,
                column_config={
                    "Risk Score": st.column_config.ProgressColumn(
                        "Risk Score", min_value=0, max_value=100, format="%d%%"
                    )
                },
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: FEATURE 4 - WHAT-IF RISK SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_whatif:
    st.markdown("### 🧪 What-If Risk Simulation (Hypothetical Scenarios)")
    st.markdown("""
    <div class="info-card">
        Simulate how modifying workplace levers (such as eliminating overtime, improving job satisfaction, or adjusting compensation)
        impacts the model's estimated attrition probability.
        <br><i>Important: This is a statistical model-based simulation to explore sensitivity, not a guarantee of behavioral change.</i>
    </div>
    """, unsafe_allow_html=True)

    # Base profile prediction
    base_df = build_features_from_dict(current_employee)
    base_scaled = scaler.transform(base_df)
    base_prob = model.predict_proba(base_scaled)[0]
    base_risk, _ = compute_attrition_risk(base_prob, model.classes_)
    base_tier = categorize_risk(base_risk)

    sim_col1, sim_col2 = st.columns([1, 1], gap="large")

    with sim_col1:
        st.markdown("#### 🛠️ Adjust Hypothetical Workplace Levers")
        
        sim_overtime = st.selectbox(
            "Hypothetical Overtime:",
            options=["No", "Yes"],
            index=0 if overtime == "No" else 1,
            key="sim_ot"
        )
        sim_job_sat = st.slider(
            "Hypothetical Job Satisfaction (1-4):",
            min_value=1, max_value=4, value=job_satisfaction,
            key="sim_js"
        )
        sim_env_sat = st.slider(
            "Hypothetical Environment Satisfaction (1-4):",
            min_value=1, max_value=4, value=environment_satisfaction,
            key="sim_es"
        )
        sim_income = st.slider(
            "Hypothetical Monthly Income (₹):",
            min_value=1000, max_value=50000, value=monthly_income, step=500,
            key="sim_inc"
        )
        sim_wlb = st.slider(
            "Hypothetical Work-Life Balance (1-4):",
            min_value=1, max_value=4, value=work_life_balance,
            key="sim_wlb"
        )
        sim_promo = st.slider(
            "Hypothetical Years Since Last Promotion:",
            min_value=0, max_value=15, value=years_since_last_promotion,
            key="sim_pr"
        )

    # Construct hypothetical copy
    hypothetical_employee = current_employee.copy()
    hypothetical_employee["OverTime"] = sim_overtime
    hypothetical_employee["JobSatisfaction"] = sim_job_sat
    hypothetical_employee["EnvironmentSatisfaction"] = sim_env_sat
    hypothetical_employee["MonthlyIncome"] = sim_income
    hypothetical_employee["WorkLifeBalance"] = sim_wlb
    hypothetical_employee["YearsSinceLastPromotion"] = sim_promo

    # Compute simulation output
    sim_df = build_features_from_dict(hypothetical_employee)
    sim_scaled = scaler.transform(sim_df)
    sim_prob = model.predict_proba(sim_scaled)[0]
    sim_risk, _ = compute_attrition_risk(sim_prob, model.classes_)
    sim_tier = categorize_risk(sim_risk)
    
    diff = round(sim_risk - base_risk, 1)

    with sim_col2:
        st.markdown("#### 📊 Simulation Impact Comparison")
        
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Original Risk", f"{base_risk}%", delta=None)
        sc2.metric("Simulated Risk", f"{sim_risk}%", delta=f"{diff}%", delta_color="inverse")
        sc3.metric("Change", f"{diff:+0.1f}% pts")

        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:16px;border:1px solid rgba(255,255,255,0.1);margin-top:10px;">
            <p style="margin:0 0 6px;color:#94A3B8;font-size:13px;">Original Risk Tier: <b>{base_tier['label']}</b></p>
            <p style="margin:0 0 10px;color:#94A3B8;font-size:13px;">Simulated Risk Tier: <b>{sim_tier['label']}</b></p>
            <p style="margin:0;font-size:14px;color: {'#B4DD80' if diff < 0 else '#FAAAA9' if diff > 0 else '#CBD5E1'};">
                <b>Analysis:</b> The hypothetical changes resulted in a <b>{abs(diff)}% {'reduction' if diff < 0 else 'increase' if diff > 0 else 'no change'}</b> in predicted attrition risk.
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
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 100], title="Predicted Risk (%)", gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94A3B8")),
            xaxis=dict(tickfont=dict(color="white")),
            legend=dict(font=dict(color="white")),
            height=260,
            margin=dict(t=20, b=20, l=10, r=10)
        )
        st.plotly_chart(comp_fig, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: RANKED EMPLOYEE OVERVIEW (BATCH TRIAGE)
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### 👥 Workforce Flight Risk Overview & Ranking")
    st.markdown("""
    <div class="info-card">
        Demonstrates cohort-level risk triage using the dataset test records. Allows HR managers to prioritize retention conversations for high-risk cohorts without exposing sensitive identities.
    </div>
    """, unsafe_allow_html=True)

    @st.cache_data
    def load_cohort_sample():
        try:
            df_raw = pd.read_csv("Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv").sample(n=60, random_state=42)
        except Exception:
            df_raw = pd.read_csv("../Dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv").sample(n=60, random_state=42)
            
        records = []
        for i, r in df_raw.iterrows():
            item = {
                "Age": r["Age"], "Gender": r["Gender"], "MaritalStatus": r["MaritalStatus"],
                "Department": r["Department"], "JobRole": r["JobRole"], "EducationField": r["EducationField"],
                "BusinessTravel": r["BusinessTravel"], "OverTime": r["OverTime"], "MonthlyIncome": r["MonthlyIncome"],
                "JobSatisfaction": r["JobSatisfaction"], "EnvironmentSatisfaction": r["EnvironmentSatisfaction"],
                "WorkLifeBalance": r.get("WorkLifeBalance", 3), "StockOptionLevel": r["StockOptionLevel"],
                "JobLevel": r["JobLevel"], "TotalWorkingYears": r["TotalWorkingYears"],
                "YearsAtCompany": r["YearsAtCompany"], "YearsSinceLastPromotion": r.get("YearsSinceLastPromotion", 1),
                "DistanceFromHome": r["DistanceFromHome"]
            }
            f_df = build_features_from_dict(item)
            s_scaled = scaler.transform(f_df)
            prob = model.predict_proba(s_scaled)[0]
            risk, _ = compute_attrition_risk(prob, model.classes_)
            tier = categorize_risk(risk)
            records.append({
                "Employee ID": f"EMP-{r['EmployeeNumber']}",
                "Department": r["Department"],
                "Job Role": r["JobRole"],
                "Overtime": r["OverTime"],
                "Tenure (Yrs)": r["YearsAtCompany"],
                "Monthly Income": f"₹{r['MonthlyIncome']:,}",
                "Risk Score": risk,
                "Risk Category": tier["label"],
            })
        return pd.DataFrame(records).sort_values(by="Risk Score", ascending=False).reset_index(drop=True)

    df_cohort = load_cohort_sample()
    
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        dept_filter = st.multiselect("Filter by Department:", options=list(df_cohort["Department"].unique()), default=[])
    
    filtered_df = df_cohort[df_cohort["Department"].isin(dept_filter)] if dept_filter else df_cohort

    st.dataframe(
        filtered_df,
        use_container_width=True,
        column_config={
            "Risk Score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=100, format="%.1f%%"
            )
        },
        hide_index=True
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 8px;color:#475569;font-size:13px;">
    Developed by <strong style="color:#60A5FA">Akash Mishra</strong>
    · Employee Retention Intelligence Platform
    · Model: Logistic Regression + SMOTE · Explainability: SHAP
</div>
""", unsafe_allow_html=True)