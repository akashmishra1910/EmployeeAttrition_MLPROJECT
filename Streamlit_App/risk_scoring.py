"""
risk_scoring.py - Centralized Risk Categorization and Light-Mode Badges
-----------------------------------------------------------------------
Provides single-source-of-truth functions for calculating employee attrition risk
scores, risk tiers, and status formatting.

Provisional Decision Boundaries:
- Low Risk    : Below 30%
- Medium Risk : 30% to below 60%
- High Risk   : 60% and above
"""

# Configurable risk thresholds
RISK_THRESHOLD_LOW = 30.0     # Below 30% = Low Risk
RISK_THRESHOLD_MEDIUM = 60.0  # 30% to < 60% = Medium Risk; >= 60% = High Risk


def compute_attrition_risk(probabilities, classes):
    """
    Extracts the attrition risk percentage safely using model.classes_.
    Guarantees that index of class 1 (Attrition='Yes') is used rather than hardcoded 1.
    """
    class_list = list(classes)
    if 1 in class_list:
        attrition_idx = class_list.index(1)
        stay_idx = 1 - attrition_idx
    else:
        attrition_idx = 1
        stay_idx = 0
        
    attrition_prob = float(probabilities[attrition_idx])
    stay_prob = float(probabilities[stay_idx])
    
    risk_score = round(attrition_prob * 100, 1)
    stay_pct = round(stay_prob * 100, 1)
    
    return risk_score, stay_pct


def categorize_risk(risk_score):
    """
    Categorizes the numerical risk score into Low, Medium, or High risk tier
    with modern SaaS semantic colors.
    """
    if risk_score < RISK_THRESHOLD_LOW:
        return {
            "tier": "Low",
            "label": "Low Flight Risk",
            "badge_html": "<span class='badge-low'>🟢 Low Risk</span>",
            "color": "#16A673",
            "bg_color": "#ECFDF5",
            "border_color": "#A7F3D0",
            "verdict": "Employee demonstrates strong retention indicators. Standard engagement practices recommended.",
        }
    elif risk_score < RISK_THRESHOLD_MEDIUM:
        return {
            "tier": "Medium",
            "label": "Medium Flight Risk",
            "badge_html": "<span class='badge-medium'>🟡 Medium Risk</span>",
            "color": "#E9A23B",
            "bg_color": "#FFFBEB",
            "border_color": "#FDE68A",
            "verdict": "Moderate turnover signals detected. Proactively address highlighted workplace friction points.",
        }
    else:
        return {
            "tier": "High",
            "label": "High Flight Risk",
            "badge_html": "<span class='badge-high'>🔴 High Risk</span>",
            "color": "#E5484D",
            "bg_color": "#FEF2F2",
            "border_color": "#FECACA",
            "verdict": "Elevated flight risk identified. Proactive, confidential retention check-in strongly advised.",
        }
