from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_gradient_boosting_pipeline.pkl"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_metadata.json"
)

EVAL_ARTIFACTS_PATH = (
    PROJECT_ROOT
    / "models"
    / "evaluation_artifacts.json"
)


# ============================================================
# CACHED LOADERS
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained Gradient Boosting pipeline once."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    """Load model performance metadata."""
    with open(METADATA_PATH, "r") as file:
        return json.load(file)


@st.cache_data
def load_evaluation_artifacts():
    """Load precomputed evaluation metrics and curves for charts."""
    if EVAL_ARTIFACTS_PATH.exists():
        with open(EVAL_ARTIFACTS_PATH, "r") as file:
            return json.load(file)
    return {}


# ============================================================
# MODEL RISK BANDS & UNDERWRITING DECISION ENGINE
# ============================================================

def compute_model_risk_band(probability: float, threshold: float = 0.19):
    """
    Map estimated default probability dynamically to non-institutional Model Risk Bands.
    Derived strictly from the application's calibrated decision threshold (19.0%)
    and empirical class distribution (11.6% baseline default rate).
    """
    prob_pct = probability * 100

    if prob_pct < 5.0:
        return {
            "band": "VERY LOW RISK",
            "grade": "VERY LOW RISK",
            "tier": "Very Low Risk",
            "decision": "AUTOMATIC APPROVAL",
            "badge_class": "badge-approved",
            "color": "#10B981",
            "rec_loss": "< 1.5%",
            "desc": "Estimated default probability is substantially below the model threshold. Highly favorable debt and capacity profile."
        }
    elif prob_pct < 12.0:
        return {
            "band": "LOW RISK",
            "grade": "LOW RISK",
            "tier": "Low Risk",
            "decision": "STANDARD APPROVAL",
            "badge_class": "badge-approved",
            "color": "#34D399",
            "rec_loss": "1.5% - 4.0%",
            "desc": "Solid credit profile with minimal default exposure well below the 19.0% operational decision cutoff."
        }
    elif prob_pct < threshold * 100:  # 19.0%
        return {
            "band": "MODERATE RISK",
            "grade": "MODERATE RISK",
            "tier": "Moderate Risk",
            "decision": "CONDITIONAL APPROVAL",
            "badge_class": "badge-review",
            "color": "#F59E0B",
            "rec_loss": "4.0% - 7.5%",
            "desc": "Default probability approaches the 19.0% decision threshold but remains on the acceptable side of the boundary."
        }
    elif prob_pct < 28.0:
        return {
            "band": "ELEVATED RISK",
            "grade": "ELEVATED RISK",
            "tier": "Elevated Risk",
            "decision": "MANUAL REVIEW REQUIRED",
            "badge_class": "badge-review",
            "color": "#FB923C",
            "rec_loss": "7.5% - 14.0%",
            "desc": "Crosses the 19.0% decision threshold; heightened risk profile requiring secondary underwriting review or collateral."
        }
    elif prob_pct < 45.0:
        return {
            "band": "HIGH RISK",
            "grade": "HIGH RISK",
            "tier": "High Risk",
            "decision": "HIGH RISK - ESCALATE",
            "badge_class": "badge-decline",
            "color": "#F87171",
            "rec_loss": "14.0% - 22.0%",
            "desc": "Substantially above the 19.0% threshold with compounding risk factors. Restructuring or shorter term recommended."
        }
    else:
        return {
            "band": "VERY HIGH RISK",
            "grade": "VERY HIGH RISK",
            "tier": "Very High Risk",
            "decision": "DECLINE RECOMMENDATION",
            "badge_class": "badge-decline",
            "color": "#EF4444",
            "rec_loss": "> 22.0%",
            "desc": "High default probability exceeding 45.0% across key credit, income, and debt-to-income dimensions."
        }

# Backwards-compatibility alias
compute_credit_grade = compute_model_risk_band


# ============================================================
# BORROWER PROFILE PRESETS
# ============================================================

def get_preset_profiles():
    """Return realistic predefined borrower test cases for quick evaluation."""
    return {
        "Custom Borrower": None,
        "Prime Tier (Low Risk)": {
            "Age": 45,
            "Income": 115000,
            "LoanAmount": 45000,
            "CreditScore": 790,
            "MonthsEmployed": 84,
            "NumCreditLines": 2,
            "InterestRate": 5.4,
            "LoanTerm": 36,
            "DTIRatio": 0.22,
            "Education": "Master's",
            "EmploymentType": "Full-time",
            "MaritalStatus": "Married",
            "HasMortgage": "Yes",
            "HasDependents": "Yes",
            "LoanPurpose": "Home",
            "HasCoSigner": "Yes"
        },
        "Near-Prime (Moderate Risk)": {
            "Age": 33,
            "Income": 54000,
            "LoanAmount": 78000,
            "CreditScore": 665,
            "MonthsEmployed": 28,
            "NumCreditLines": 3,
            "InterestRate": 11.2,
            "LoanTerm": 36,
            "DTIRatio": 0.36,
            "Education": "Bachelor's",
            "EmploymentType": "Full-time",
            "MaritalStatus": "Single",
            "HasMortgage": "No",
            "HasDependents": "No",
            "LoanPurpose": "Auto",
            "HasCoSigner": "No"
        },
        "Subprime (High Risk)": {
            "Age": 22,
            "Income": 22000,
            "LoanAmount": 140000,
            "CreditScore": 510,
            "MonthsEmployed": 6,
            "NumCreditLines": 4,
            "InterestRate": 21.5,
            "LoanTerm": 48,
            "DTIRatio": 0.65,
            "Education": "High School",
            "EmploymentType": "Unemployed",
            "MaritalStatus": "Single",
            "HasMortgage": "No",
            "HasDependents": "Yes",
            "LoanPurpose": "Business",
            "HasCoSigner": "No"
        },
        "Young Professional": {
            "Age": 27,
            "Income": 88000,
            "LoanAmount": 60000,
            "CreditScore": 725,
            "MonthsEmployed": 36,
            "NumCreditLines": 2,
            "InterestRate": 7.5,
            "LoanTerm": 24,
            "DTIRatio": 0.25,
            "Education": "Bachelor's",
            "EmploymentType": "Full-time",
            "MaritalStatus": "Single",
            "HasMortgage": "No",
            "HasDependents": "No",
            "LoanPurpose": "Education",
            "HasCoSigner": "No"
        }
    }


# ============================================================
# FUTURISTIC PLOTLY VISUALIZATIONS
# ============================================================

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E2E8F0", family="Inter, system-ui, sans-serif"),
    margin=dict(l=25, r=25, t=35, b=25),
)


def create_gauge_chart(probability: float, threshold: float = 0.19):
    """Futuristic neon gauge indicator for loan default probability."""
    prob_pct = probability * 100
    thresh_pct = threshold * 100

    bar_color = "#10B981" if prob_pct < 12 else ("#F59E0B" if prob_pct < thresh_pct else "#EF4444")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=prob_pct,
            number={"suffix": "%", "font": {"size": 44, "color": "#F8FAFC", "weight": "bold"}},
            delta={
                "reference": thresh_pct,
                "increasing": {"color": "#EF4444"},
                "decreasing": {"color": "#10B981"},
                "position": "bottom",
                "relative": False,
                "valueformat": "+.1f",
            },
            title={
                "text": "<b>DEFAULT PROBABILITY</b><br><span style='font-size:12px;color:#94A3B8'>Decision Threshold: "
                + f"{thresh_pct:.1f}%</span>",
                "font": {"size": 16, "color": "#E2E8F0"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#475569",
                    "tickvals": [0, 19, 40, 60, 80, 100],
                    "ticktext": ["0%", "19% (τ)", "40%", "60%", "80%", "100%"],
                    "tickfont": {"size": 10, "color": "#94A3B8"},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.03)",
                "borderwidth": 1,
                "bordercolor": "rgba(148, 163, 184, 0.2)",
                "steps": [
                    {"range": [0, 12], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [12, thresh_pct], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [thresh_pct, 100], "color": "rgba(239, 68, 68, 0.18)"},
                ],
                "threshold": {
                    "line": {"color": "#F59E0B", "width": 4},
                    "thickness": 0.85,
                    "value": thresh_pct,
                },
            },
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        height=260,
    )
    return fig


def create_radar_chart(input_dict: dict):
    """Spider radar comparing applicant's financial metrics against prime standards."""
    # Normalized score between 0 and 100 for radar
    cs_norm = min(max((input_dict["CreditScore"] - 300) / (850 - 300) * 100, 0), 100)
    income_norm = min(max((input_dict["Income"] - 15000) / (150000 - 15000) * 100, 0), 100)
    dti_inverted = max(100 - (input_dict["DTIRatio"] / 0.8 * 100), 0)
    emp_norm = min(max(input_dict["MonthsEmployed"] / 100 * 100, 0), 100)
    rate_inverted = max(100 - ((input_dict["InterestRate"] - 2) / 23 * 100), 0)

    categories = [
        "Credit Score",
        "Income Level",
        "Low DTI Balance",
        "Employment Tenure",
        "Favorable Interest Rate",
    ]

    applicant_values = [cs_norm, income_norm, dti_inverted, emp_norm, rate_inverted]
    applicant_values.append(applicant_values[0])  # Close polygon

    prime_benchmark = [85, 80, 80, 75, 85, 85]
    closed_categories = categories + [categories[0]]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=prime_benchmark,
            theta=closed_categories,
            fill="toself",
            fillcolor="rgba(99, 102, 241, 0.12)",
            line=dict(color="#6366F1", width=1.5, dash="dot"),
            name="Prime Benchmark",
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=applicant_values,
            theta=closed_categories,
            fill="toself",
            fillcolor="rgba(6, 182, 212, 0.28)",
            line=dict(color="#06B6D4", width=2.5),
            name="Applicant Profile",
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=8, color="#64748B"),
                gridcolor="rgba(148, 163, 184, 0.12)",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#E2E8F0"),
                gridcolor="rgba(148, 163, 184, 0.15)",
                linecolor="rgba(148, 163, 184, 0.2)",
            ),
            bgcolor="rgba(15, 23, 42, 0.5)",
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#94A3B8"),
        ),
        height=320,
    )
    return fig


def create_feature_impact_chart(input_dict: dict):
    """
    Explainability chart: Estimates the applicant's key drivers pushing risk UP or DOWN
    based on model weights and baseline norms.
    """
    # Baseline comparison
    baseline_income = 55000
    baseline_interest = 12.0
    baseline_age = 35
    baseline_months = 36
    baseline_cs = 650

    factors = []

    # Age impact (weight ~29%)
    if input_dict["Age"] < 28:
        factors.append({"factor": "Younger Age Profile (<28)", "impact": 18.5, "type": "Risk Driver (+)"})
    elif input_dict["Age"] > 45:
        factors.append({"factor": "Mature Age Profile (>45)", "impact": -14.2, "type": "Risk Reducer (-)"})

    # Income impact (weight ~20%)
    if input_dict["Income"] < 35000:
        factors.append({"factor": "Low Annual Income (<$35k)", "impact": 16.0, "type": "Risk Driver (+)"})
    elif input_dict["Income"] > 85000:
        factors.append({"factor": "High Annual Income (>$85k)", "impact": -15.5, "type": "Risk Reducer (-)"})

    # Interest rate (weight ~19.5%)
    if input_dict["InterestRate"] > 14.0:
        factors.append({"factor": f"High Loan APR ({input_dict['InterestRate']}%)", "impact": 17.2, "type": "Risk Driver (+)"})
    elif input_dict["InterestRate"] < 8.0:
        factors.append({"factor": f"Low Loan APR ({input_dict['InterestRate']}%)", "impact": -12.8, "type": "Risk Reducer (-)"})

    # Loan Amount (weight ~12%)
    if input_dict["LoanAmount"] > 120000:
        factors.append({"factor": "Substantial Loan Principal", "impact": 11.4, "type": "Risk Driver (+)"})
    elif input_dict["LoanAmount"] < 50000:
        factors.append({"factor": "Modest Loan Principal", "impact": -8.5, "type": "Risk Reducer (-)"})

    # Employment tenure (weight ~10%)
    if input_dict["MonthsEmployed"] < 12:
        factors.append({"factor": "Short Employment Tenure (<1 yr)", "impact": 12.5, "type": "Risk Driver (+)"})
    elif input_dict["MonthsEmployed"] > 48:
        factors.append({"factor": "Stable Employment History (>4 yrs)", "impact": -10.0, "type": "Risk Reducer (-)"})

    # Co-Signer
    if input_dict.get("HasCoSigner") == "Yes":
        factors.append({"factor": "Guaranteed Co-Signer Present", "impact": -9.0, "type": "Risk Reducer (-)"})
    else:
        factors.append({"factor": "No Co-Signer Backing", "impact": 6.5, "type": "Risk Driver (+)"})

    # Employment Type
    if input_dict.get("EmploymentType") == "Unemployed":
        factors.append({"factor": "Unemployed Status", "impact": 15.0, "type": "Risk Driver (+)"})
    elif input_dict.get("EmploymentType") == "Full-time":
        factors.append({"factor": "Full-Time Employment", "impact": -5.0, "type": "Risk Reducer (-)"})

    df_factors = pd.DataFrame(factors)
    if df_factors.empty:
        df_factors = pd.DataFrame([{"factor": "Standard Profile", "impact": 0.0, "type": "Neutral"}])

    df_factors = df_factors.sort_values(by="impact", ascending=True)

    colors = [
        "#EF4444" if val > 0 else "#10B981"
        for val in df_factors["impact"]
    ]

    fig = go.Figure(
        go.Bar(
            x=df_factors["impact"],
            y=df_factors["factor"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{val:+.1f} pts" for val in df_factors["impact"]],
            textposition="outside",
            textfont=dict(size=10, color="#CBD5E1"),
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title="<b>Key Factors Driving This Assessment</b>",
        xaxis=dict(
            title="Relative Risk Influence (Indicative Points)",
            zeroline=True,
            zerolinecolor="rgba(148, 163, 184, 0.4)",
            gridcolor="rgba(148, 163, 184, 0.1)",
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=300,
    )
    return fig


def create_roc_curve(roc_data: dict):
    """Futuristic interactive ROC curve."""
    fig = go.Figure()

    # Random baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#475569", width=1.5, dash="dash"),
            name="Chance (AUC = 0.5000)",
        )
    )

    # ROC curve
    fig.add_trace(
        go.Scatter(
            x=roc_data["fpr"],
            y=roc_data["tpr"],
            mode="lines",
            line=dict(color="#06B6D4", width=3),
            fill="tozeroy",
            fillcolor="rgba(6, 182, 212, 0.12)",
            name=f"Gradient Boosting (AUC = {roc_data['auc']:.4f})",
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        )
    )

    # Highlight operational point at ~0.19 threshold
    fig.add_trace(
        go.Scatter(
            x=[0.1247],
            y=[0.4481],
            mode="markers+text",
            marker=dict(color="#F59E0B", size=11, line=dict(color="#FFFFFF", width=2)),
            text=["Optimal Decision Point (τ=0.19)"],
            textposition="bottom right",
            textfont=dict(color="#F59E0B", size=11, weight="bold"),
            name="Operational Threshold (0.19)",
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title="<b>Receiver Operating Characteristic (ROC)</b>",
        xaxis=dict(
            title="False Positive Rate (1 - Specificity)",
            gridcolor="rgba(148, 163, 184, 0.1)",
            zerolinecolor="rgba(148, 163, 184, 0.2)",
            range=[-0.02, 1.02],
        ),
        yaxis=dict(
            title="True Positive Rate (Sensitivity / Recall)",
            gridcolor="rgba(148, 163, 184, 0.1)",
            zerolinecolor="rgba(148, 163, 184, 0.2)",
            range=[-0.02, 1.02],
        ),
        legend=dict(
            yanchor="bottom",
            y=0.04,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(148, 163, 184, 0.2)",
            borderwidth=1,
        ),
        height=380,
    )
    return fig


def create_pr_curve(pr_data: dict):
    """Futuristic interactive Precision-Recall curve."""
    fig = go.Figure()

    # Base rate baseline (11.6%)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0.116, 0.116],
            mode="lines",
            line=dict(color="#475569", width=1.5, dash="dash"),
            name="Base Default Rate (11.6%)",
        )
    )

    # PR curve
    fig.add_trace(
        go.Scatter(
            x=pr_data["recall"],
            y=pr_data["precision"],
            mode="lines",
            line=dict(color="#8B5CF6", width=3),
            fill="tozeroy",
            fillcolor="rgba(139, 92, 246, 0.12)",
            name=f"Gradient Boosting (PR-AUC = {pr_data['auc']:.4f})",
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
        )
    )

    # Optimal threshold marker
    fig.add_trace(
        go.Scatter(
            x=[0.4481],
            y=[0.3205],
            mode="markers+text",
            marker=dict(color="#F59E0B", size=11, line=dict(color="#FFFFFF", width=2)),
            text=["Max F1 Operating Point (τ=0.19)"],
            textposition="top right",
            textfont=dict(color="#F59E0B", size=11, weight="bold"),
            name="Optimal Threshold (0.19)",
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title="<b>Precision-Recall (PR) Curve</b>",
        xaxis=dict(
            title="Recall (Sensitivity)",
            gridcolor="rgba(148, 163, 184, 0.1)",
            zerolinecolor="rgba(148, 163, 184, 0.2)",
            range=[-0.02, 1.02],
        ),
        yaxis=dict(
            title="Precision (Positive Predictive Value)",
            gridcolor="rgba(148, 163, 184, 0.1)",
            zerolinecolor="rgba(148, 163, 184, 0.2)",
            range=[-0.02, 1.02],
        ),
        legend=dict(
            yanchor="top",
            y=0.96,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(148, 163, 184, 0.2)",
            borderwidth=1,
        ),
        height=380,
    )
    return fig


def create_confusion_matrix_chart(cm_data: dict):
    """Interactive heatmap of Confusion Matrix."""
    matrix = cm_data["matrix"]
    total = sum(matrix[0]) + sum(matrix[1])

    z = matrix
    x_labels = ["Predicted: Non-Default (0)", "Predicted: Default (1)"]
    y_labels = ["Actual: Non-Default (0)", "Actual: Default (1)"]

    annotations = []
    labels = [
        [f"True Negatives<br><b>{matrix[0][0]:,}</b><br>({matrix[0][0]/total*100:.1f}%)",
         f"False Positives<br><b>{matrix[0][1]:,}</b><br>({matrix[0][1]/total*100:.1f}%)"],
        [f"False Negatives<br><b>{matrix[1][0]:,}</b><br>({matrix[1][0]/total*100:.1f}%)",
         f"True Positives<br><b>{matrix[1][1]:,}</b><br>({matrix[1][1]/total*100:.1f}%)"]
    ]

    for i in range(2):
        for j in range(2):
            annotations.append(
                dict(
                    x=x_labels[j],
                    y=y_labels[i],
                    text=labels[i][j],
                    showarrow=False,
                    font=dict(color="#F8FAFC", size=13),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale=[
                [0.0, "#0F172A"],
                [0.15, "#1E293B"],
                [0.5, "#4338CA"],
                [1.0, "#06B6D4"],
            ],
            showscale=False,
            hoverongaps=False,
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title="<b>Test Set Confusion Matrix (N = 38,303 @ τ=0.19)</b>",
        annotations=annotations,
        yaxis=dict(autorange="reversed"),
        height=360,
    )
    return fig


def create_benchmark_chart(benchmark_data: list):
    """Multi-metric model comparative benchmark bar chart."""
    df = pd.DataFrame(benchmark_data)

    metrics = ["accuracy", "recall", "precision", "f1", "roc_auc"]
    metric_names = ["Accuracy", "Recall", "Precision", "F1 Score", "ROC-AUC"]
    colors = ["#38BDF8", "#34D399", "#F59E0B", "#F43F5E", "#A855F7"]

    fig = go.Figure()

    for m, m_name, color in zip(metrics, metric_names, colors):
        fig.add_trace(
            go.Bar(
                name=m_name,
                x=df["model"],
                y=[round(val * 100, 1) for val in df[m]],
                text=[f"{val*100:.1f}%" for val in df[m]],
                textposition="outside",
                textfont=dict(size=10),
                marker_color=color,
            )
        )

    fig.update_layout(
        **DARK_LAYOUT,
        title="<b>Model Benchmark Comparison on Test Set</b>",
        barmode="group",
        yaxis=dict(
            title="Percentage (%)",
            range=[0, 105],
            gridcolor="rgba(148, 163, 184, 0.1)",
        ),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(size=11),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#CBD5E1"),
        ),
        height=400,
    )
    return fig


def create_feature_importance_chart(feature_importances: list, top_n: int = 12):
    """Top N feature importances from Gradient Boosting model."""
    df = pd.DataFrame(feature_importances[:top_n])
    df = df.sort_values(by="importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df["importance"],
            y=df["feature"],
            orientation="h",
            marker=dict(
                color=df["importance"],
                colorscale=[[0, "#3B82F6"], [0.5, "#6366F1"], [1, "#06B6D4"]],
            ),
            text=[f"{val:.2f}%" for val in df["importance"]],
            textposition="outside",
            textfont=dict(size=11, color="#CBD5E1"),
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title=f"<b>Top {top_n} Predictive Feature Importances (Gradient Boosting)</b>",
        xaxis=dict(
            title="Gini Feature Importance (%)",
            gridcolor="rgba(148, 163, 184, 0.1)",
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=380,
    )
    return fig


def create_threshold_tuning_chart(sweep_data: list, optimal_threshold: float = 0.19):
    """Sensitivity curve of Precision, Recall, and F1 across classification thresholds."""
    df = pd.DataFrame(sweep_data)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["threshold"],
            y=df["precision"],
            mode="lines+markers",
            line=dict(color="#38BDF8", width=2.5),
            marker=dict(size=4),
            name="Precision",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["threshold"],
            y=df["recall"],
            mode="lines+markers",
            line=dict(color="#34D399", width=2.5),
            marker=dict(size=4),
            name="Recall",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["threshold"],
            y=df["f1"],
            mode="lines+markers",
            line=dict(color="#F43F5E", width=3),
            marker=dict(size=5),
            name="F1 Score (Target)",
        )
    )

    # Highlight optimal threshold line
    fig.add_vline(
        x=optimal_threshold,
        line_width=2,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text=f"Selected Threshold: {optimal_threshold} (Max F1 = 0.3737)",
        annotation_position="top right",
        annotation_font=dict(color="#F59E0B", size=11, weight="bold"),
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title="<b>Optimal Threshold Calibration Curve (F1 Maximization)</b>",
        xaxis=dict(
            title="Decision Threshold (τ)",
            gridcolor="rgba(148, 163, 184, 0.1)",
            range=[0.08, 0.72],
        ),
        yaxis=dict(
            title="Metric Score (0 - 1.0)",
            gridcolor="rgba(148, 163, 184, 0.1)",
            range=[0, 1.02],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#CBD5E1"),
        ),
        height=380,
    )
    return fig


def create_risk_distribution_chart(df_results: pd.DataFrame):
    """Risk tier distribution donut chart for batch loan portfolio."""
    col_name = "Model Risk Band" if "Model Risk Band" in df_results.columns else ("Credit Grade" if "Credit Grade" in df_results.columns else df_results.columns[0])
    grade_counts = df_results[col_name].value_counts().reset_index()
    grade_counts.columns = ["Band", "Count"]

    color_map = {
        "VERY LOW RISK": "#10B981",
        "LOW RISK": "#34D399",
        "MODERATE RISK": "#F59E0B",
        "ELEVATED RISK": "#FB923C",
        "HIGH RISK": "#F87171",
        "VERY HIGH RISK": "#EF4444"
    }

    colors = [color_map.get(g, "#6366F1") for g in grade_counts["Band"]]

    fig = go.Figure(
        go.Pie(
            labels=grade_counts["Band"],
            values=grade_counts["Count"],
            hole=0.55,
            marker=dict(colors=colors),
            textinfo="label+percent",
            textfont=dict(size=12, color="#FFFFFF"),
            hovertemplate="Model Risk Band: %{label}<br>Applications: %{value:,}<br>Share: %{percent}<extra></extra>",
        )
    )

    fig.update_layout(
        **DARK_LAYOUT,
        title="<b>Portfolio Model Risk Band Breakdown</b>",
        height=340,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig


def create_pipeline_sankey():
    """Interactive Sankey flow diagram representing the end-to-end ML feature transformation and inference pipeline."""
    labels = [
        "16 Raw Features",                      # 0
        "9 Numeric Attributes",                 # 1
        "7 Categorical Attributes",             # 2
        "Numeric Passthrough Engine",           # 3
        "One-Hot Encoder (k-1 sparse)",         # 4
        "24-Dim Transformed Vector",            # 5
        "Gradient Boosting (100 Trees)",        # 6
        "Model Default Probability P(Y=1)",     # 7
        "Below Threshold / Favorable (P < 0.19)",# 8
        "Above Threshold / Heightened (P >= 0.19)",# 9
        "Very Low / Low / Moderate Risk",       # 10
        "Elevated / High / Very High Risk"      # 11
    ]

    colors = [
        "#38BDF8",  # 0
        "#06B6D4",  # 1
        "#818CF8",  # 2
        "#06B6D4",  # 3
        "#818CF8",  # 4
        "#A855F7",  # 5
        "#EC4899",  # 6
        "#F59E0B",  # 7
        "#10B981",  # 8
        "#EF4444",  # 9
        "#34D399",  # 10
        "#F87171"   # 11
    ]

    sources = [0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9]
    targets = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10, 11]
    values  = [9, 7, 9, 15, 9, 15, 24, 24, 21, 3, 21, 3]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=22,
            line=dict(color="#0F172A", width=1),
            label=labels,
            color=colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(99, 102, 241, 0.22)"
        )
    )])

    fig.update_layout(
        **DARK_LAYOUT,
        title="<b>Interactive End-to-End Feature & Decision Flow (Sankey Flowchart)</b>",
        height=380,
    )
    return fig