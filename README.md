# 📊 Employee Retention Intelligence Platform

An end-to-end Machine Learning and Explainable AI (XAI) application that predicts employee attrition risk, explains predictions using SHAP (SHapley Additive exPlanations), and provides actionable workforce retention insights.

Developed by **Akash Mishra** | B.Tech Computer Science & Engineering

---

## 🚀 Project Overview

Employee attrition (turnover) is a costly challenge for modern organizations, resulting in steep replacement costs, loss of institutional knowledge, and team disruption.

This platform bridges machine learning with actionable HR strategy:
1. **Predictive Risk Assessment:** Estimates the probability that an employee will leave.
2. **Transparent Explainability (SHAP):** Unpacks the model's "black box" to reveal the specific factors driving the individual prediction.
3. **Proactive Retention Recommendations:** Translates model attributions into supportive workplace interventions.
4. **What-If Risk Simulation:** Allows HR leaders to test how modifying workplace levers (e.g., eliminating overtime, improving role satisfaction) changes estimated flight risk.
5. **Ranked Cohort Triage:** Prioritizes at-risk employee cohorts across departments.

---

## 🧠 Machine Learning & Methodology

### 1. Business Problem & Dataset
- **Dataset:** IBM HR Analytics Employee Attrition dataset (1,470 records, 35 features).
- **Target:** `Attrition` (`Yes` = 1, `No` = 0).
- **Class Imbalance:** 84% retained (`No`), 16% resigned (`Yes`).

### 2. Why Class Imbalance Matters
In employee turnover prediction, missing an employee who leaves (**False Negative**) is vastly more costly to an organization than checking in on an employee who is content (**False Positive**). Therefore, **Recall** on the minority class (`Attrition = 1`) is our primary evaluation metric.

### 3. Pipeline & Techniques
- **Preprocessing:**
  - Removed constant/non-informative columns (`EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours`).
  - Feature Engineering:
    - $\text{IncomePerYear} = \text{MonthlyIncome} \times 12$
    - $\text{ExperienceRatio} = \frac{\text{YearsAtCompany}}{\text{TotalWorkingYears} + 1}$
  - Categorical Encoding: `LabelEncoder` for nominal attributes.
  - Scaling: `StandardScaler` fitted on training split only.
- **Resampling:** Synthetic Minority Over-sampling Technique (**SMOTE**) applied exclusively to the training split (`X_train_scaled`, `y_train`) to balance classes without test-set leakage.
- **Model:** **Logistic Regression** ($L_2$ regularized, `max_iter=1000`).

### 4. Mathematical Foundations

#### Logistic Regression Hypothesis:
$$P(Y=1 \mid X) = \frac{1}{1 + e^{-z}} \quad \text{where} \quad z = \beta_0 + \sum_{i=1}^p \beta_i X_i$$

#### Risk Score Calculation:
$$\text{Risk Score (\%)} = P(\text{Attrition}=1 \mid X) \times 100$$

#### Provisional Risk Tiers:
- **🟢 Low Risk:** $< 30\%$
- **🟡 Medium Risk:** $30\% \le \text{Score} < 60\%$
- **🔴 High Risk:** $\ge 60\%$
*(Note: These are configurable triage thresholds for demonstration, not hard causal boundaries.)*

---

## 📊 Measured Model Performance (Held-Out Test Set)

Evaluated on 20% stratified test split ($N = 294$, containing 47 true attrition cases):

| Model & Approach | Accuracy | Precision | **Recall (Attrition=1)** | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression + SMOTE** (Selected) | **76.5%** | **38.5%** | **78.7%** 🎯 | **0.518** | **0.813** 🏆 | **0.617** |
| Logistic Regression (Class-Weight Balanced) | 75.5% | 37.1% | 76.6% | 0.500 | 0.813 | 0.616 |
| Random Forest + SMOTE | 84.4% | 52.2% | 25.5% ⚠️ | 0.343 | 0.809 | 0.584 |
| Decision Tree + SMOTE | 78.9% | 35.3% | 38.3% | 0.367 | 0.625 | 0.320 |

> **Key Evaluation Insight:** While Random Forest achieves higher superficial accuracy (84.4%), it fails to detect nearly 75% of actual resignations (Recall: 25.5%). Logistic Regression with SMOTE identifies almost 8 out of 10 at-risk employees (**78.7% Recall**), making it the optimal model for HR retention triage.

---

## 🔍 Explainable AI (XAI) with SHAP

The platform uses `shap.LinearExplainer` to decompose each employee's prediction:

$$\ln\left(\frac{P}{1 - P}\right) = \phi_0 + \sum_{i=1}^M \phi_i$$

- **$\phi_0$ (Base Value):** The expected background log-odds of attrition.
- **$\phi_i$ (SHAP Attribution):**
  - **Positive ($\phi_i > 0$, Red):** Pushes the model toward predicting attrition (higher flight risk).
  - **Negative ($\phi_i < 0$, Green):** Pushes the model toward predicting retention (stability).

> ⚠️ **Important Interpretation Principle:** SHAP explains the *mathematical reasoning of the model*, not physical real-world causation. For example, working overtime increases the model's flight risk estimate, but does not guarantee an employee will quit.

---

## 🛠️ Tech Stack & Architecture

- **Language:** Python 3.10+
- **Machine Learning:** `scikit-learn`, `imbalanced-learn` (SMOTE), `joblib`
- **Explainability:** `shap`
- **Interactive UI:** `streamlit`, `plotly`
- **Data Manipulation:** `pandas`, `numpy`

### Project Structure:
```text
Employee-Retention-Intelligence/
├── Dataset/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── Model/
│   ├── attrition_model.pkl    # Trained Logistic Regression model
│   ├── scaler.pkl             # StandardScaler instance
│   ├── encoders.pkl           # Categorical LabelEncoders dict
│   └── feature_names.pkl      # Feature column names & order
├── Notebook/
│   └── Employee_Attrition_Project_Notebook.ipynb
└── Streamlit_App/
    ├── app.py                 # Streamlit application
    ├── explainability.py      # SHAP LinearExplainer & Plotly visualization
    ├── recommendations.py     # Transparent rule-based retention logic
    ├── risk_scoring.py        # Centralized risk scoring & tier categorization
    └── requirements.txt       # Project dependencies
```

---

## 💻 Installation & Execution

### 1. Clone the repository
```bash
git clone https://github.com/akashmishra1910/EmployeeAttrition_MLPROJECT.git
cd EmployeeAttrition_MLPROJECT/Streamlit_App
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ⚠️ Limitations & Ethical Considerations

1. **Probability Calibration:** Logistic Regression probabilities with SMOTE training are effective ranking heuristics, but are slightly shifted by synthetic sampling and may benefit from Platt scaling / isotonic calibration before absolute probability interpretation.
2. **Observational vs. Causal:** Predictions and SHAP attributions represent observed statistical correlations, not experimental causal proof.
3. **Supportive Actions Only:** Retention suggestions are supportive workplace guidelines (workload checks, 1-on-1 conversations). The system must never be used for punitive, adverse, or automated termination decisions.

---

## 👨‍💻 Author

**Akash Mishra**  
B.Tech Computer Science & Engineering  
- GitHub: [akashmishra1910](https://github.com/akashmishra1910)  
- LinkedIn: [akashmishra1910](https://linkedin.com/in/akashmishra1910)
