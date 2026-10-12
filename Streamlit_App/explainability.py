"""
explainability.py - SHAP Model Explanation Module (Light-Mode SaaS Edition)
---------------------------------------------------------------------------
Explains individual employee attrition predictions using SHAP (SHapley Additive exPlanations).

Key Design Principles:
1. Reuses the trained Logistic Regression model without altering it.
2. Uses shap.LinearExplainer for mathematically exact, fast log-odds attribution.
3. Maps raw feature names to human-readable business labels.
4. Distinguishes model attribution from causal claims (explains model reasoning, not proof).
5. Provides beautifully styled Plotly charts suited for a clean light-mode SaaS theme.
"""

import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Dictionary mapping raw feature column names to clear HR business labels
FEATURE_DISPLAY_NAMES = {
    "Age": "Age",
    "BusinessTravel": "Business Travel Frequency",
    "DailyRate": "Daily Rate",
    "Department": "Department",
    "DistanceFromHome": "Distance from Home (km)",
    "Education": "Education Level",
    "EducationField": "Education Field",
    "EnvironmentSatisfaction": "Work Environment Satisfaction",
    "Gender": "Gender",
    "HourlyRate": "Hourly Rate",
    "JobInvolvement": "Job Involvement Level",
    "JobLevel": "Job Seniority Level",
    "JobRole": "Job Role",
    "JobSatisfaction": "Job Satisfaction Score",
    "MaritalStatus": "Marital Status",
    "MonthlyIncome": "Monthly Income (₹)",
    "MonthlyRate": "Monthly Billing Rate",
    "NumCompaniesWorked": "Past Companies Worked",
    "OverTime": "Overtime Status",
    "PercentSalaryHike": "Last Salary Hike (%)",
    "PerformanceRating": "Performance Rating",
    "RelationshipSatisfaction": "Coworker Relationship Satisfaction",
    "StockOptionLevel": "Stock Option Level",
    "TotalWorkingYears": "Total Career Experience (Yrs)",
    "TrainingTimesLastYear": "Training Sessions Last Year",
    "WorkLifeBalance": "Work-Life Balance Score",
    "YearsAtCompany": "Tenure at Company (Yrs)",
    "YearsInCurrentRole": "Years in Current Role",
    "YearsSinceLastPromotion": "Years Since Last Promotion",
    "YearsWithCurrManager": "Years with Current Manager",
    "IncomePerYear": "Annualized Income (₹)",
    "ExperienceRatio": "Tenure vs Experience Ratio",
}


def create_shap_explainer(model, feature_names):
    """
    Creates a SHAP LinearExplainer for the Logistic Regression model.
    Uses Independent masker with zero baseline (since features are standard-scaled, mean=0).
    """
    baseline = np.zeros((1, len(feature_names)))
    masker = shap.maskers.Independent(baseline)
    explainer = shap.LinearExplainer(model, masker=masker)
    return explainer


def explain_prediction(explainer, scaled_features, feature_names, top_k=5):
    """
    Generates SHAP feature attributions for a single employee record.
    
    Parameters:
    -----------
    explainer : shap.LinearExplainer
    scaled_features : numpy.ndarray (1, num_features)
    feature_names : list of str
    top_k : int, number of top features to highlight (default 5)
    
    Returns:
    --------
    dict containing:
        - top_features: list of dicts with name, raw_name, shap_value, direction, impact
        - all_shap_df: full DataFrame sorted by absolute SHAP value
        - base_value: expected log-odds baseline
    """
    shap_explanation = explainer(scaled_features)
    shap_vals = shap_explanation.values[0]
    base_val = float(shap_explanation.base_values[0])
    
    records = []
    for raw_name, val in zip(feature_names, shap_vals):
        readable = FEATURE_DISPLAY_NAMES.get(raw_name, raw_name)
        direction = "Pushes Toward Leaving (Higher Risk)" if val > 0 else "Pushes Toward Staying (Lower Risk)"
        records.append({
            "Feature": readable,
            "RawFeature": raw_name,
            "SHAPValue": float(val),
            "AbsSHAP": abs(float(val)),
            "Direction": direction,
            "PushesToward": "Leave" if val > 0 else "Stay",
        })
        
    df_all = pd.DataFrame(records).sort_values(by="AbsSHAP", ascending=False).reset_index(drop=True)
    top_features = df_all.head(top_k).to_dict(orient="records")
    
    return {
        "top_features": top_features,
        "all_shap_df": df_all,
        "base_value": base_val,
    }


def create_shap_bar_chart(top_features):
    """
    Builds a clean, styled horizontal bar chart showing top SHAP feature contributions
    optimized for the modern Light-Mode SaaS interface.
    Positive values (#E5484D - Coral Red) push toward attrition.
    Negative values (#16A673 - Jade Green) push toward retention.
    """
    # Reverse so largest impact appears on top
    items = list(reversed(top_features))
    labels = [item["Feature"] for item in items]
    vals = [item["SHAPValue"] for item in items]
    colors = ["#E5484D" if v > 0 else "#16A673" for v in vals]
    texts = [f"{v:+.2f} log-odds" for v in vals]
    
    fig = go.Figure(go.Bar(
        x=vals,
        y=labels,
        orientation="h",
        marker=dict(color=colors, cornerradius=5),
        text=texts,
        textposition="outside",
        textfont=dict(color="#1E2235", size=11, family="Inter, sans-serif"),
    ))
    
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(
            title=dict(text="SHAP Feature Contribution (Log-Odds Impact on Risk Output)", font=dict(color="#697386", size=11)),
            gridcolor="#F1F3F9",
            tickfont=dict(color="#697386", size=10),
            zerolinecolor="#CBD5E1",
            zerolinewidth=1.5,
        ),
        yaxis=dict(tickfont=dict(color="#1E2235", size=11, family="Inter, sans-serif")),
        margin=dict(t=10, b=30, l=10, r=90),
        height=260,
    )
    return fig


def create_global_feature_importance_chart(model, feature_names, top_k=10):
    """
    Computes global feature weights from Logistic Regression coefficients
    and renders an elegant light-mode horizontal ranking chart.
    """
    coefs = model.coef_[0]
    records = []
    for raw_name, coef in zip(feature_names, coefs):
        records.append({
            "Feature": FEATURE_DISPLAY_NAMES.get(raw_name, raw_name),
            "Coefficient": float(coef),
            "AbsImpact": abs(float(coef)),
            "Type": "Increases Risk" if coef > 0 else "Decreases Risk"
        })
    df_global = pd.DataFrame(records).sort_values(by="AbsImpact", ascending=True).tail(top_k)
    
    colors = ["#E5484D" if c > 0 else "#16A673" for c in df_global["Coefficient"]]
    
    fig = go.Figure(go.Bar(
        x=df_global["AbsImpact"],
        y=df_global["Feature"],
        orientation="h",
        marker=dict(color=colors, cornerradius=5),
        text=[f"{val:.2f}" for val in df_global["AbsImpact"]],
        textposition="outside",
        textfont=dict(color="#1E2235", size=11),
    ))
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(
            title=dict(text="Overall Model Importance (|Log-Odds Weight|)", font=dict(color="#697386", size=11)),
            gridcolor="#F1F3F9",
            tickfont=dict(color="#697386", size=10),
        ),
        yaxis=dict(tickfont=dict(color="#1E2235", size=11)),
        margin=dict(t=10, b=30, l=10, r=50),
        height=320,
    )
    return fig
