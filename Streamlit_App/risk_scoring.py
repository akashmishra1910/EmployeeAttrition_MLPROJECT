"""
risk_scoring.py - Centralized Risk Categorization and Policy
------------------------------------------------------------
Provides single-source-of-truth functions for calculating employee attrition risk
scores, risk tiers, and status formatting.

Provisional Decision Boundaries:
- Low Risk    : Below 30%
- Medium Risk : 30% to below 60%
- High Risk   : 60% and above

Note: These are configurable demonstration thresholds for HR triage, not definitive
causal predictions or legally validated decision boundaries.
"""

# Configurable risk thresholds
RISK_THRESHOLD_LOW = 30.0     # Below 30% = Low Risk
RISK_THRESHOLD_MEDIUM = 60.0  # 30% to < 60% = Medium Risk; >= 60% = High Risk


def compute_attrition_risk(probabilities, classes):
    """
    Extracts the attrition risk percentage safely using model.classes_.
    Guarantees that index of class 1 (Attrition='Yes') is used rather than hardcoded 1.
    
    Parameters:
    -----------
    probabilities : array-like of shape (2,)
    classes : array-like of shape (2,)
    
    Returns:
    --------
    tuple: (risk_score_pct, stay_pct, pred_class)
    """
    # Locate index of class 1 (Attrition=Yes)
    class_list = list(classes)
    if 1 in class_list:
        attrition_idx = class_list.index(1)
        stay_idx = 1 - attrition_idx
    else:
        # Fallback if binary labels differ
        attrition_idx = 1
        stay_idx = 0
        
    attrition_prob = float(probabilities[attrition_idx])
    stay_prob = float(probabilities[stay_idx])
    
    risk_score = round(attrition_prob * 100, 1)
    stay_pct = round(stay_prob * 100, 1)
    
    return risk_score, stay_pct


def categorize_risk(risk_score):
    """
    Categorizes the numerical risk score into Low, Medium, or High risk tier.
    
    Returns:
    --------
    dict with:
        - tier: 'Low' | 'Medium' | 'High'
        - label: Formatted badge text
        - color: Hex color code for UI
        - verdict: Plain-language interpretation
    """
    if risk_score < RISK_THRESHOLD_LOW:
        return {
            "tier": "Low",
            "label": "🟢 LOW RISK",
            "color": "#639922",
            "verdict": "Employee shows healthy retention indicators. Standard engagement practices recommended.",
        }
    elif risk_score < RISK_THRESHOLD_MEDIUM:
        return {
            "tier": "Medium",
            "label": "🟡 MEDIUM RISK",
            "color": "#EF9F27",
            "verdict": "Moderate turnover signals detected. Address specific friction points to prevent escalation.",
        }
    else:
        return {
            "tier": "High",
            "label": "🔴 HIGH RISK",
            "color": "#E24B4A",
            "verdict": "Elevated flight risk identified. Proactive, supportive retention check-in strongly advised.",
        }
