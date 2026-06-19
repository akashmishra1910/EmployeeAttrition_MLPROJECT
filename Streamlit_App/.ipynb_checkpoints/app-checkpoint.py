import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
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

/* Metric cards — white background, black text */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(56,138,220,0.3);
    border-radius: 14px;
    padding: 18px 20px;
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

/* Buttons */
.stButton > button {
    width: 100%; height: 52px; border-radius: 12px; border: none;
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
.rec-card  { background:rgba(30,111,217,0.12); border:1px solid rgba(30,111,217,0.25);  border-radius:10px; padding:12px 16px; margin-bottom:8px; color:#BAD4F7; font-size:14px; }
.warn-card { background:rgba(239,159,39,0.12); border:1px solid rgba(239,159,39,0.25);  border-radius:10px; padding:12px 16px; margin-bottom:8px; color:#FCD580; font-size:14px; }
.danger-card{background:rgba(226,75,74,0.12);  border:1px solid rgba(226,75,74,0.30);   border-radius:10px; padding:12px 16px; margin-bottom:8px; color:#FAAAA9; font-size:14px; }
.success-card{background:rgba(99,153,34,0.12); border:1px solid rgba(99,153,34,0.30);   border-radius:10px; padding:12px 16px; margin-bottom:8px; color:#B4DD80; font-size:14px; }
.model-badge{background:rgba(99,153,34,0.15);  border:1px solid rgba(99,153,34,0.35);   border-radius:8px;  padding:6px 14px; color:#B4DD80; font-size:13px; display:inline-block; margin-bottom:12px;}

footer{visibility:hidden;} #MainMenu{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model         = joblib.load("../-Model/attrition_model.pkl")
    scaler        = joblib.load("../-Model/scaler.pkl")
    encoders      = joblib.load("../-Model/encoders.pkl")
    feature_names = joblib.load("../-Model/feature_names.pkl")
    return model, scaler, encoders, feature_names

try:
    model, scaler, encoders, feature_names = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error   = str(e)

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);
            border-radius:18px;padding:22px 28px;margin-bottom:24px;">
    <p style="color:#64748B;font-size:12px;letter-spacing:.1em;text-transform:uppercase;margin:0 0 4px">
        Developed by Akash Mishra
    </p>
    <h1 style="color:white;margin:0;font-size:26px;font-weight:700;">
        📊 Employee Attrition Intelligence Platform
    </h1>
    <p style="color:#94A3B8;margin:6px 0 0;font-size:15px;">
        Predict employee turnover risk using Logistic Regression — optimized for high recall and balanced with SMOTE.
    </p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Could not load model files: {load_error}")
    st.info("Ensure `attrition_model.pkl`, `scaler.pkl`, `encoders.pkl`, and `feature_names.pkl` exist in `../-Model/`.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
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
    monthly_income           = st.slider("Monthly Income (₹)",          1000, 50000, 5000, step=500)
    job_satisfaction         = st.slider("Job Satisfaction (1–4)",       1, 4, 3)
    environment_satisfaction = st.slider("Environment Satisfaction (1–4)",1, 4, 3)
    stock_option_level       = st.slider("Stock Option Level (0–3)",     0, 3, 1)

    st.markdown('<p class="section-header">Career & Tenure</p>', unsafe_allow_html=True)
    job_level           = st.slider("Job Level (1–5)",          1, 5, 2)
    total_working_years = st.slider("Total Working Years",      0, 40, 8)
    years_at_company    = st.slider("Years at Company",         0, 40, 5)
    distance_from_home  = st.slider("Distance from Home (km)", 1, 30, 5)

    predict_clicked = st.button("🚀 Predict Attrition Risk")

# ── Feature builder — must match notebook column order exactly ────────────────
def build_features() -> pd.DataFrame:
    """
    Constructs a single-row DataFrame with all features in the exact
    order the scaler was fitted on during training.
    """
    row = {
        "Age":                      age,
        "BusinessTravel":           encoders["BusinessTravel"].transform([business_travel])[0],
        "DailyRate":                800,           # median default
        "Department":               encoders["Department"].transform([department])[0],
        "DistanceFromHome":         distance_from_home,
        "Education":                3,             # median default
        "EducationField":           encoders["EducationField"].transform([education_field])[0],
        "EnvironmentSatisfaction":  environment_satisfaction,
        "Gender":                   encoders["Gender"].transform([gender])[0],
        "HourlyRate":               65,            # median default
        "JobInvolvement":           3,             # median default
        "JobLevel":                 job_level,
        "JobRole":                  encoders["JobRole"].transform([job_role])[0],
        "JobSatisfaction":          job_satisfaction,
        "MaritalStatus":            encoders["MaritalStatus"].transform([marital_status])[0],
        "MonthlyIncome":            monthly_income,
        "MonthlyRate":              14000,         # median default
        "NumCompaniesWorked":       2,             # median default
        "OverTime":                 encoders["OverTime"].transform([overtime])[0],
        "PercentSalaryHike":        15,            # median default
        "PerformanceRating":        3,             # median default
        "RelationshipSatisfaction": 3,             # median default
        "StockOptionLevel":         stock_option_level,
        "TotalWorkingYears":        total_working_years,
        "TrainingTimesLastYear":    2,             # median default
        "WorkLifeBalance":          3,             # median default
        "YearsAtCompany":           years_at_company,
        "YearsInCurrentRole":       3,             # median default
        "YearsSinceLastPromotion":  1,             # median default
        "YearsWithCurrManager":     3,             # median default
        # Engineered features
        "IncomePerYear":            monthly_income * 12,
        "ExperienceRatio":          years_at_company / (total_working_years + 1),
    }
    # Reorder to match training column order
    df = pd.DataFrame([row])
    df = df[feature_names]
    return df

# ── Per-input risk contributions (heuristic) ──────────────────────────────────
def compute_input_factors():
    factors = {}
    factors["Monthly income"]         = round(max(0, 1 - monthly_income / 25000) * 100)
    factors["Overtime"]               = 90 if overtime == "Yes" else 10
    factors["Job satisfaction"]       = round((4 - job_satisfaction) / 3 * 100)
    factors["Env. satisfaction"]      = round((4 - environment_satisfaction) / 3 * 100)
    factors["Distance from home"]     = round(min(1, distance_from_home / 30) * 100)
    factors["Stock options"]          = round((3 - stock_option_level) / 3 * 100)
    travel_map = {"Travel_Frequently": 100, "Travel_Rarely": 40, "Non-Travel": 0}
    factors["Business travel"]        = travel_map.get(business_travel, 0)
    marital_map = {"Single": 100, "Divorced": 40, "Married": 5}
    factors["Marital status risk"]    = marital_map.get(marital_status, 5)
    factors["Short tenure"]           = round(max(0, 1 - years_at_company / 15) * 100)
    factors["Early career"]           = round(max(0, 1 - total_working_years / 20) * 100)
    return dict(sorted(factors.items(), key=lambda x: -x[1]))

# ── Recommendations ───────────────────────────────────────────────────────────
def get_recommendations(risk_score):
    recs = []
    if overtime == "Yes":
        recs.append(("⚠️", "warn",    "Reduce overtime — it is a primary predictor of attrition."))
    if monthly_income < 6000:
        recs.append(("💰", "danger",  "Review compensation — income is significantly below the retention threshold."))
    if distance_from_home > 15:
        recs.append(("🏠", "warn",    "Offer hybrid or remote work to ease commute burden."))
    if job_satisfaction < 2:
        recs.append(("😟", "danger",  "Low job satisfaction — consider role redesign or growth pathways."))
    if environment_satisfaction < 2:
        recs.append(("🏢", "warn",    "Poor environment satisfaction — run a team culture audit."))
    if stock_option_level == 0:
        recs.append(("📈", "rec",     "Introduce stock or profit-sharing to build long-term commitment."))
    if business_travel == "Travel_Frequently":
        recs.append(("✈️", "warn",    "Frequent travel is a strong attrition driver — review the travel policy."))
    if marital_status == "Single" and risk_score > 40:
        recs.append(("👤", "rec",     "Single employees at elevated risk — consider mentorship or team-building programs."))
    if years_at_company < 2:
        recs.append(("📅", "warn",    "Low tenure — strengthen onboarding and 90-day check-ins."))
    if not recs:
        recs.append(("✅", "success", "No major retention concerns detected. Keep up current engagement practices."))
    return recs

# ── Plotly charts ─────────────────────────────────────────────────────────────
def gauge_chart(score):
    color = "#639922" if score < 30 else "#EF9F27" if score < 50 else "#E24B4A"
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
                {"range": [0,  30],  "color": "rgba(99,153,34,0.10)"},
                {"range": [30, 50],  "color": "rgba(239,159,39,0.10)"},
                {"range": [50, 100], "color": "rgba(226,75,74,0.10)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.85, "value": score},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=10, l=20, r=20), height=240, font={"color": "white"},
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
        margin=dict(t=10, b=10, l=10, r=10), height=230,
    )
    return fig

def factor_bar_chart(factors):
    labels = list(factors.keys())
    values = list(factors.values())
    colors = ["#E24B4A" if v >= 70 else "#EF9F27" if v >= 40 else "#378ADD" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=[f"{v}%" for v in values], textposition="outside",
        textfont=dict(color="white", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 120], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(color="#94A3B8", size=11)),
        margin=dict(t=4, b=4, l=10, r=50), height=300, bargap=0.32,
    )
    return fig

def benchmark_chart(current_income):
    bands  = ["<₹3k", "₹3–6k", "₹6–10k", "₹10–20k", ">₹20k"]
    avg    = [72, 55, 38, 22, 12]
    colors = ["#E24B4A", "#EF9F27", "#BA7517", "#378ADD", "#639922"]
    hi = 0 if current_income < 3000 else 1 if current_income < 6000 else \
         2 if current_income < 10000 else 3 if current_income < 20000 else 4
    border = ["rgba(255,255,255,0.8)" if i == hi else "rgba(0,0,0,0)" for i in range(5)]
    fig = go.Figure(go.Bar(
        x=bands, y=avg,
        marker=dict(color=colors, line=dict(color=border, width=2), cornerradius=5),
        text=[f"{v}%" for v in avg], textposition="outside",
        textfont=dict(color="white", size=12),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="#94A3B8", size=12), showgrid=False),
        yaxis=dict(range=[0, 90], tickfont=dict(color="#64748B", size=11),
                   gridcolor="rgba(255,255,255,0.06)", ticksuffix="%"),
        margin=dict(t=8, b=4, l=10, r=10), height=220,
    )
    return fig

# ── Main layout ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

# ─── LEFT COLUMN ─────────────────────────────────────────────────────────────
with col_left:

    # Top metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Monthly income",  f"₹{monthly_income:,}")
    m2.metric("Department",      department[:14])
    m3.metric("Distance",        f"{distance_from_home} km")
    m4.metric("Overtime",        overtime)

    st.markdown("---")

    # ── Prediction block ──────────────────────────────────────────────────────
    if predict_clicked:
        with st.spinner("Running Logistic Regression prediction..."):
            X_input   = build_features()
            X_scaled  = scaler.transform(X_input)
            pred      = model.predict(X_scaled)[0]
            proba     = model.predict_proba(X_scaled)[0]
            risk_score = round(float(proba[1]) * 100, 1)
            stay_pct   = round(float(proba[0]) * 100, 1)

        # Risk level - Calibrated for Logistic Regression
        if risk_score < 30:
            level   = "🟢 LOW RISK"
            verdict = "Employee shows strong retention signals."
        elif risk_score < 50:
            level   = "🟡 MEDIUM RISK"
            verdict = "Moderate flight risk — address the highlighted factors."
        else:
            level   = "🔴 HIGH RISK"
            verdict = "High flight risk — immediate intervention recommended."

        # Save to history
        st.session_state.history.insert(0, {
            "Time":       datetime.now().strftime("%H:%M:%S"),
            "Role":       job_role,
            "Department": department,
            "Risk Score": int(risk_score),
            "Level":      level,
            "Prediction": "Leave" if pred == 1 else "Stay",
        })

        # Gauge + donut side by side
        ga, gb = st.columns(2, gap="medium")
        with ga:
            st.markdown(f"### {level}")
            st.caption(verdict)
            st.plotly_chart(gauge_chart(risk_score), use_container_width=True,
                            config={"displayModeBar": False})
        with gb:
            st.caption("Probability breakdown — Stay vs Leave")
            st.plotly_chart(donut_chart(stay_pct, risk_score),
                            use_container_width=True, config={"displayModeBar": False})
            st.markdown(f"""
            <div style="display:flex;gap:10px;justify-content:center;margin-top:4px">
                <span style="color:#B4DD80;font-size:13px">● Stay {stay_pct}%</span>
                <span style="color:#FAAAA9;font-size:13px">● Leave {risk_score}%</span>
            </div>
            """, unsafe_allow_html=True)

        # Detailed metrics row
        st.markdown("---")
        dm1, dm2, dm3, dm4 = st.columns(4)
        dm1.metric("Risk Score",   f"{risk_score}%")
        dm2.metric("Prediction",   "Leave ⚠️" if pred == 1 else "Stay ✅")
        dm3.metric("Stay Prob.",   f"{stay_pct}%")
        dm4.metric("Predictions",  len(st.session_state.history))

        # Recommendations
        st.markdown("### 🎯 Retention recommendations")
        card_map = {"warn": "warn-card", "danger": "danger-card",
                    "rec": "rec-card", "success": "success-card"}
        for icon, kind, text in get_recommendations(risk_score):
            st.markdown(f'<div class="{card_map[kind]}">{icon} {text}</div>',
                        unsafe_allow_html=True)

    else:
        # Placeholder before first prediction
        st.markdown("""
        <div style="text-align:center;padding:56px 0;color:#475569;">
            <p style="font-size:48px;margin-bottom:12px">📊</p>
            <p style="font-size:17px;font-weight:600;color:#94A3B8;">Ready to predict</p>
            <p style="font-size:14px;color:#64748B;max-width:320px;margin:8px auto 0">
                Fill in the employee details in the sidebar and click
                <strong style="color:#60A5FA">Predict Attrition Risk</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Benchmark chart — always visible
    st.markdown("### 📊 Attrition risk by income band — benchmark")
    st.caption("Highlighted bar = current employee's income band.")
    st.plotly_chart(benchmark_chart(monthly_income), use_container_width=True,
                    config={"displayModeBar": False})

    # History table — always visible
    st.markdown("### 🕐 Prediction history")
    if not st.session_state.history:
        st.caption("No predictions yet.")
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

# ─── RIGHT COLUMN ─────────────────────────────────────────────────────────────
with col_right:

    # Model info badge
    st.markdown("""
    <div class="model-badge">
        ✅ Model: Logistic Regression · SMOTE balanced · Optimized for High Recall
    </div>
    """, unsafe_allow_html=True)

    # Input-driven risk factors
    st.markdown("### 🎯 Input risk factor breakdown")
    st.caption("Heuristic contribution of each input to overall attrition risk.")
    factors = compute_input_factors()
    st.plotly_chart(factor_bar_chart(factors), use_container_width=True,
                    config={"displayModeBar": False})

    st.markdown("---")

    # Employee snapshot
    st.markdown("### 👤 Employee snapshot")
    snap = pd.DataFrame({
        "Field": ["Age", "Gender", "Marital status", "Department", "Job role",
                  "Education field", "Business travel", "Overtime",
                  "Job level", "Stock options", "Total experience", "Tenure"],
        "Value": [age, gender, marital_status, department, job_role,
                  education_field, business_travel, overtime,
                  job_level, stock_option_level,
                  f"{total_working_years} yrs", f"{years_at_company} yrs"],
    })
    st.dataframe(snap, hide_index=True, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 8px;color:#334155;font-size:13px;">
    Developed by <strong style="color:#60A5FA">Akash Mishra</strong>
    · Employee Attrition Intelligence Platform
    · Model: Logistic Regression + SMOTE
</div>
""", unsafe_allow_html=True)