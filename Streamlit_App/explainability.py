"""
explainability.py - SHAP Model Explanation Module
--------------------------------------------------
Explains individual employee attrition predictions using SHAP (SHapley Additive exPlanations).

Key Design Principles:
1. Reuses the trained Logistic Regression model without altering it.
2. Uses shap.LinearExplainer for mathematically exact, fast log-odds attribution.
3. Maps raw feature names to human-readable business labels.
4. Distinguishes model attribution from causal claims (explains model reasoning, not proof).
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
    Builds a clean, styled horizontal bar chart showing top SHAP feature contributions.
    Positive values (Red) push toward attrition.
    Negative values (Green) push toward retention.
    """
    # Reverse so largest impact appears on top
    items = list(reversed(top_features))
    labels = [item["Feature"] for item in items]
    vals = [item["SHAPValue"] for item in items]
    colors = ["#E24B4A" if v > 0 else "#639922" for v in vals]
    texts = [f"{v:+.3f} (Log-Odds)" for v in vals]
    
    fig = go.Figure(go.Bar(
        x=vals,
        y=labels,
        orientation="h",
        marker=dict(color=colors, cornerradius=4),
        text=texts,
        textposition="outside",
        textfont=dict(color="white", size=12),
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text="SHAP Contribution (Log-Odds Impact on Model Output)", font=dict(color="#94A3B8", size=11)),
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="#94A3B8", size=11),
            zerolinecolor="rgba(255,255,255,0.2)",
        ),
        yaxis=dict(tickfont=dict(color="#CBD5E1", size=12)),
        margin=dict(t=10, b=30, l=10, r=80),
        height=280,
    )
    return fig
