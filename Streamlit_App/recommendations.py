"""
recommendations.py - Rule-Based Employee Retention Recommendations
-------------------------------------------------------------------
Turns employee work and satisfaction signals into supportive, constructive
workplace actions.

Guiding Principles:
1. Transparent, simple Python rules (no black-box or LLM needed).
2. Uses raw employee inputs (e.g. Overtime == 'Yes'), not encoded integers.
3. Suggests supportive HR actions, never punitive or terminating actions.
4. Explains that these are recommendations for discussion, not guaranteed solutions.
"""

def generate_retention_recommendations(employee_input):
    """
    Evaluates key workplace factors and returns actionable retention recommendations.
    
    Parameters:
    -----------
    employee_input : dict
        Raw employee details collected from the form.
        Expected keys: 'OverTime', 'MonthlyIncome', 'JobSatisfaction',
        'EnvironmentSatisfaction', 'WorkLifeBalance', 'YearsSinceLastPromotion',
        'DistanceFromHome', 'BusinessTravel', 'StockOptionLevel', 'YearsAtCompany'.
        
    Returns:
    --------
    list of dicts, each with keys:
        - 'icon': emoji representation
        - 'category': 'Workload' | 'Compensation' | 'Culture & Wellbeing' | 'Career Growth' | 'Commute'
        - 'level': 'danger' | 'warning' | 'info' | 'success'
        - 'title': Short descriptive title
        - 'action': Recommended practical workplace follow-up
    """
    recs = []
    
    # 1. Overtime pattern
    if employee_input.get("OverTime") == "Yes":
        recs.append({
            "icon": "⏳",
            "category": "Workload & Wellbeing",
            "level": "warning",
            "title": "Review Workload & Overtime Pattern",
            "action": "The employee regularly logs overtime. Consider conducting a workload audit, exploring task redistribution, or offering comp-off hours to prevent chronic burnout."
        })
        
    # 2. Low Job Satisfaction
    job_sat = employee_input.get("JobSatisfaction", 3)
    if job_sat <= 2:
        recs.append({
            "icon": "💬",
            "category": "Role Engagement",
            "level": "danger",
            "title": "Confidential Role Alignment Discussion",
            "action": f"Job satisfaction is currently low ({job_sat}/4). Arrange a structured 1-on-1 check-in to discuss daily responsibilities, role friction points, and opportunities for meaningful tasks."
        })
        
    # 3. Low Environment Satisfaction
    env_sat = employee_input.get("EnvironmentSatisfaction", 3)
    if env_sat <= 2:
        recs.append({
            "icon": "🏢",
            "category": "Workplace Climate",
            "level": "warning",
            "title": "Assess Team Climate & Workplace Environment",
            "action": f"Environment satisfaction is rated {env_sat}/4. Evaluate team dynamics, peer collaboration, and physical/remote tooling to ensure the employee feels well-supported."
        })
        
    # 4. Work-Life Balance
    wlb = employee_input.get("WorkLifeBalance", 3)
    if wlb <= 2:
        recs.append({
            "icon": "⚖️",
            "category": "Flexibility & Balance",
            "level": "warning",
            "title": "Discuss Schedule & Flexibility Options",
            "action": f"Work-life balance is rated {wlb}/4. Discuss flexible working hours, hybrid options, or clear boundaries on after-hours communication."
        })
        
    # 5. Promotion Stagnation (4+ years without promotion)
    promo_years = employee_input.get("YearsSinceLastPromotion", 0)
    if promo_years >= 4:
        recs.append({
            "icon": "🚀",
            "category": "Career Growth",
            "level": "warning",
            "title": "Initiate Career Progression Review",
            "action": f"The employee has been in their current grade for {promo_years} years without promotion. Formulate a transparent career development roadmap with clear milestones for advancement."
        })
        
    # 6. Low Compensation
    income = employee_input.get("MonthlyIncome", 5000)
    if income < 4000:
        recs.append({
            "icon": "💰",
            "category": "Compensation",
            "level": "danger",
            "title": "Conduct Market Benchmark Compensation Review",
            "action": f"Monthly income (₹{income:,}) is below industry retention thresholds for similar roles. Review compensation bands against peer standards to remain competitive."
        })
        
    # 7. Commute Stress
    distance = employee_input.get("DistanceFromHome", 5)
    if distance >= 20:
        recs.append({
            "icon": "🚗",
            "category": "Commute & Flexibility",
            "level": "info",
            "title": "Mitigate Commute Fatigue",
            "action": f"Employee commutes {distance} km each way. Explore remote work arrangements, staggered commute shifts, or transit pass allowances to ease commute burden."
        })
        
    # 8. Frequent Travel
    travel = employee_input.get("BusinessTravel")
    if travel == "Travel_Frequently":
        recs.append({
            "icon": "✈️",
            "category": "Travel Policy",
            "level": "info",
            "title": "Review Frequent Travel Requirements",
            "action": "Frequent business travel is a known turnover catalyst. Evaluate whether virtual meeting channels can substitute for non-critical travel."
        })
        
    # 9. Stock Options / Long-term Incentive
    stock = employee_input.get("StockOptionLevel", 1)
    if stock == 0:
        recs.append({
            "icon": "📈",
            "category": "Long-Term Retention",
            "level": "info",
            "title": "Explore Long-Term Incentive / ESOPs",
            "action": "The employee holds no stock options (Level 0). Consider introducing performance-linked incentive bonuses or equity vesting to foster long-term commitment."
        })
        
    # Fallback if no specific trigger fired
    if not recs:
        recs.append({
            "icon": "✅",
            "category": "General Retention",
            "level": "success",
            "title": "Maintain Positive Engagement",
            "action": "No acute risk triggers detected. Continue regular check-ins, recognition programs, and ongoing skills development to keep engagement strong."
        })
        
    return recs
