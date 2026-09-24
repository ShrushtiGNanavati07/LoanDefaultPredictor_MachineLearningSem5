import streamlit as st
import pandas as pd
import numpy as np
import io
import importlib
import utils
importlib.reload(utils)

from utils import (
    load_model,
    load_metadata,
    load_evaluation_artifacts,
    compute_model_risk_band,
    create_gauge_chart,
    create_radar_chart,
    create_feature_impact_chart,
    create_roc_curve,
    create_pr_curve,
    create_confusion_matrix_chart,
    create_benchmark_chart,
    create_feature_importance_chart,
    create_threshold_tuning_chart,
    create_pipeline_sankey
)

# ============================================================
# PAGE CONFIGURATION & SYSTEM THEME
# ============================================================

st.set_page_config(
    page_title="Loan Sentry | AI Credit Risk Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODEL & EVALUATION DATA
# ============================================================

model = load_model()
metadata = load_metadata()
eval_artifacts = load_evaluation_artifacts()

THRESHOLD = float(metadata.get("threshold", 0.19))

# ============================================================
# FUTURISTIC ULTRA-MODERN CSS
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ---------- GLOBAL STYLES ---------- */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.12), transparent 35%),
                    radial-gradient(circle at 90% 85%, rgba(6, 182, 212, 0.10), transparent 35%),
                    radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 0.5), transparent 100%),
                    #070B14;
        color: #E2E8F0;
    }

    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3.5rem;
        max-width: 1440px;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background: #090F1D;
        border-right: 1px solid rgba(99, 102, 241, 0.18);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    /* ---------- BRAND HEADER ---------- */
    .brand-box {
        padding: 0.5rem 0.5rem 1.2rem 0.5rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        margin-bottom: 1.2rem;
    }

    .brand-title {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .brand-badge {
        display: inline-block;
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        margin-top: 0.4rem;
    }

    /* ---------- HERO SECTION ---------- */
    .hero-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(30, 41, 59, 0.5) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #06B6D4, #6366F1, #A855F7);
    }

    .hero-eyebrow {
        color: #06B6D4;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }

    .hero-desc {
        color: #94A3B8;
        font-size: 0.95rem;
        line-height: 1.6;
        max-width: 820px;
    }

    /* ---------- GLASS CARDS ---------- */
    .glass-card {
        background: rgba(13, 21, 37, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.12);
        box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
    }

    .card-heading {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .card-subheading {
        font-size: 0.78rem;
        color: #64748B;
        margin-bottom: 1rem;
    }

    /* ---------- METRIC CARDS ---------- */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .kpi-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        text-align: left;
        position: relative;
    }

    .kpi-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 15%;
        right: 15%;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.4), transparent);
    }

    .kpi-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #94A3B8;
        margin-bottom: 0.4rem;
    }

    .kpi-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
    }

    .kpi-sub {
        font-size: 0.72rem;
        color: #10B981;
        margin-top: 0.3rem;
        font-weight: 600;
    }

    /* ---------- DECISION RESULT CARD ---------- */
    .decision-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(20, 30, 55, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 1.5rem;
        box-shadow: 0 12px 35px -8px rgba(0, 0, 0, 0.6);
        position: relative;
    }

    .decision-badge {
        display: inline-block;
        padding: 0.45rem 1.2rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .badge-approved {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.15);
    }

    .badge-review {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.15);
    }

    .badge-decline {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.15);
    }

    .grade-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.2rem;
    }

    /* ---------- SUMMARY ATTRIBUTE GRID ---------- */
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 0.8rem;
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 0.9rem 1.2rem;
        margin-top: 1.2rem;
    }

    .summary-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94A3B8;
    }

    .summary-value {
        font-size: 1rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 0.2rem;
    }

    /* ---------- INPUT CONTROLS ---------- */
    .stNumberInput input, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background: #0B1120 !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
    }

    .stNumberInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 1px #38BDF8 !important;
    }

    /* ---------- PRIMARY BUTTON ---------- */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #06B6D4 0%, #6366F1 50%, #8B5CF6 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.08em !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
        transition: all 0.25s ease !important;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6) !important;
    }

    /* ---------- FOOTER ---------- */
    .footer-box {
        text-align: center;
        color: #475569;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(148, 163, 184, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR NAVIGATION & SYSTEM HEALTH
# ============================================================

with st.sidebar:
    st.markdown("""
    <div class="brand-box">
        <div class="brand-title">◈ LOAN SENTRY</div>
        <div class="brand-badge">ENTERPRISE RISK INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🎯 Risk Assessment",
            "📈 Model Analytics",
            "📑 Project Dossier"
        ],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 12px; padding: 1rem; margin-top: 1rem;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
            <span style="font-size:0.7rem; font-weight:700; color:#94A3B8; letter-spacing:0.1em;">SYSTEM STATUS</span>
            <span style="font-size:0.7rem; color:#10B981; font-weight:700;">● ONLINE</span>
        </div>
        <div style="font-size:0.8rem; color:#E2E8F0; font-weight:600; margin-bottom:0.2rem;">Gradient Boosting Pipeline</div>
        <div style="font-size:0.72rem; color:#64748B;">Classification τ: <b style="color:#F59E0B">0.190</b> (Max F1)</div>
        <div style="font-size:0.72rem; color:#64748B;">ROC-AUC: <b style="color:#38BDF8">0.7595</b> | PR-AUC: <b style="color:#A855F7">0.3278</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # st.caption("Machine Learning Academic Project — Sem 5")


# ============================================================
# PAGE 1: RISK ASSESSMENT & DECISION CENTER
# ============================================================

if page == "🎯 Risk Assessment":

    # --- HERO ---
    st.markdown("""
    <div class="hero-container">
        <div class="hero-eyebrow">AI CREDIT UNDERWRITING PLATFORM</div>
        <div class="hero-title">Individual Loan Risk Assessment</div>
        <div class="hero-desc">
            Evaluate prospective borrower profiles through the production Gradient Boosting classifier.
            Calibrated on 255,000+ historical loans with dynamic threshold optimization to deliver reliable default probabilities and actionable underwriting guidance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- INPUT FORM ---
    with st.container():
        st.markdown("""
        <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;">
            <span>◈ Borrower Financial & Loan Parameters</span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        # COLUMN 1: Personal & Demographics
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-heading">👤 Demographics & Stability</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-subheading">Personal background and tenure</div>', unsafe_allow_html=True)

            age = st.number_input(
                "Age (Years)",
                min_value=18,
                max_value=69,
                value=33,
                key="input_age"
            )

            education = st.selectbox(
                "Highest Education Level",
                ["High School", "Bachelor's", "Master's", "PhD"],
                index=1,
                key="input_edu"
            )

            marital_status = st.selectbox(
                "Marital Status",
                ["Single", "Married", "Divorced"],
                index=0,
                key="input_marital"
            )

            has_dependents = st.selectbox(
                "Has Dependents",
                ["No", "Yes"],
                index=0,
                key="input_dep"
            )

            st.markdown('</div>', unsafe_allow_html=True)

        # COLUMN 2: Employment & Income
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-heading">💼 Financial Capacity</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-subheading">Earning power and debt obligations</div>', unsafe_allow_html=True)

            income = st.number_input(
                "Annual Income ($)",
                min_value=15000,
                max_value=150000,
                value=54000,
                step=1000,
                key="input_income"
            )

            employment_type = st.selectbox(
                "Employment Type",
                ["Full-time", "Part-time", "Self-employed", "Unemployed"],
                index=0,
                key="input_emp_type"
            )

            months_employed = st.number_input(
                "Employment Tenure (Months)",
                min_value=0,
                max_value=119,
                value=28,
                key="input_months_emp"
            )

            credit_score = st.number_input(
                "Credit Score (FICO)",
                min_value=300,
                max_value=850,
                value=665,
                key="input_cs"
            )

            st.markdown('</div>', unsafe_allow_html=True)

        # COLUMN 3: Loan Terms & Credit Structure
        with col3:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-heading">📑 Loan Specifications</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-subheading">Facility details and covenants</div>', unsafe_allow_html=True)

            loan_amount = st.number_input(
                "Requested Loan Amount ($)",
                min_value=5000,
                max_value=250000,
                value=78000,
                step=1000,
                key="input_loan_amt"
            )

            interest_rate = st.number_input(
                "Interest Rate (% APR)",
                min_value=2.0,
                max_value=25.0,
                value=11.2,
                step=0.1,
                format="%.1f",
                key="input_interest"
            )

            loan_term = st.number_input(
                "Loan Term (Months)",
                min_value=12,
                max_value=60,
                value=36,
                step=12,
                key="input_term"
            )

            dti_ratio = st.number_input(
                "Debt-to-Income (DTI Ratio)",
                min_value=0.10,
                max_value=0.90,
                value=0.36,
                step=0.01,
                format="%.2f",
                key="input_dti"
            )

            st.markdown('</div>', unsafe_allow_html=True)

        # Secondary Parameters Row
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            num_credit_lines = st.number_input(
                "Active Credit Lines",
                min_value=1,
                max_value=4,
                value=3,
                key="input_lines"
            )
        with col_c2:
            loan_purpose = st.selectbox(
                "Loan Purpose",
                ["Auto", "Business", "Education", "Home", "Other"],
                index=0,
                key="input_purpose"
            )
        with col_c3:
            has_mortgage = st.selectbox(
                "Existing Mortgage",
                ["No", "Yes"],
                index=0,
                key="input_mortgage"
            )
        with col_c4:
            has_cosigner = st.selectbox(
                "Co-Signer Backing",
                ["No", "Yes"],
                index=0,
                key="input_cosigner"
            )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_btn = st.button("RUN RISK INFERENCE ◈", use_container_width=True)

    # Prepare DataFrame matching pipeline input schema
    current_input = {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner
    }
    input_df = pd.DataFrame([current_input])

    # Run inference
    try:
        raw_prob = float(model.predict_proba(input_df)[0][1])
        prob_percent = raw_prob * 100
        risk_info = compute_model_risk_band(raw_prob, THRESHOLD)

        # Store in session state for simulator
        st.session_state["last_prob"] = raw_prob
        st.session_state["last_input"] = current_input

    except Exception as e:
        st.error(f"Inference Error: {str(e)}")
        raw_prob = 0.1418
        prob_percent = 14.18
        risk_info = compute_model_risk_band(raw_prob, THRESHOLD)

    # --- RESULT BANNER ---
    st.markdown(f"""<div class="decision-banner">
<div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1.5rem;">
<div>
<div style="font-size:0.75rem; font-weight:700; letter-spacing:0.18em; color:#94A3B8; text-transform:uppercase;">DEFAULT PROBABILITY</div>
<div style="font-size:3.2rem; font-weight:800; color:#F8FAFC; font-family:'JetBrains Mono', monospace; line-height:1.1; margin:0.2rem 0 0.6rem 0;" class="decision-badge {risk_info['badge_class']}">● {risk_info['band']}</div>
<div style="margin-bottom:0.75rem;"><span class="decision-badge ">{prob_percent:.1f}%</span></div>
<div style="color:#94A3B8; font-size:0.88rem; margin-top:0.4rem; max-width:680px; line-height:1.6;">{risk_info['desc']}</div>
</div>
<div style="text-align:right; min-width:220px; background:rgba(15,23,42,0.6); padding:1.2rem 1.4rem; border-radius:14px; border:1px solid rgba(148,163,184,0.15);">
<div style="font-size:0.72rem; font-weight:700; letter-spacing:0.15em; color:#94A3B8; text-transform:uppercase;">MODEL RISK BAND</div>
<div style="font-size:1.35rem; font-weight:800; color:{risk_info['color']}; margin-top:0.25rem; font-family:'Plus Jakarta Sans', sans-serif;">{risk_info['band']}</div>
<div style="margin-top:0.85rem; padding-top:0.6rem; border-top:1px solid rgba(148,163,184,0.12);">
<div style="font-size:0.72rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em;">Default Probability</div>
<div style="font-size:1.1rem; font-weight:700; color:#F8FAFC; font-family:'JetBrains Mono', monospace;">{prob_percent:.1f}%</div>
</div>
<div style="margin-top:0.4rem;">
<div style="font-size:0.72rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em;">Model Threshold</div>
<div style="font-size:0.95rem; font-weight:700; color:#F59E0B; font-family:'JetBrains Mono', monospace;">{THRESHOLD*100:.1f}%</div>
</div>
<div style="font-size:0.68rem; color:#64748B; margin-top:0.5rem;">Internal Model Classification</div>
</div>
</div>
<div class="summary-grid">
<div><div class="summary-label">Credit Score</div><div class="summary-value">{credit_score}</div></div>
<div><div class="summary-label">DTI Ratio</div><div class="summary-value">{dti_ratio:.2f}</div></div>
<div><div class="summary-label">Loan Amount</div><div class="summary-value">${loan_amount:,.0f}</div></div>
<div><div class="summary-label">Interest Rate</div><div class="summary-value">{interest_rate:.1f}%</div></div>
<div><div class="summary-label">Co-Signer</div><div class="summary-value">{has_cosigner}</div></div>
</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- INTERACTIVE VISUALIZATIONS ROW ---
    col_vis1, col_vis2 = st.columns([1, 1])

    with col_vis1:
        st.markdown('<div class="glass-card"><div class="card-heading">🎯 Default Probability Gauge</div><div class="card-subheading">Calibrated needle against safe, cautionary, and default breach zones (τ = ' + f'{THRESHOLD*100:.1f}%)</div>', unsafe_allow_html=True)
        fig_gauge = create_gauge_chart(raw_prob, THRESHOLD)
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_vis2:
        st.markdown('<div class="glass-card"><div class="card-heading">🕸 Financial Strength Radar</div><div class="card-subheading">Applicant dimensions benchmarked against Prime standard portfolio</div>', unsafe_allow_html=True)
        fig_radar = create_radar_chart(current_input)
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- FEATURE IMPACT EXPLAINABILITY ---
    st.markdown('<div class="glass-card"><div class="card-heading">🔍 Feature Impact & Explainability Breakdown</div><div class="card-subheading">Estimated contribution of applicant attributes toward increasing or reducing default risk</div>', unsafe_allow_html=True)
    fig_impact = create_feature_impact_chart(current_input)
    st.plotly_chart(fig_impact, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- WHAT-IF SENSITIVITY SIMULATOR ---
    with st.expander("⚡ Interactive 'What-If' Scenario Playground & Risk Mitigation", expanded=False):
        st.markdown("""
        Test how restructuring the application or adding mitigations changes the default probability in real-time.
        """)
        sim_col1, sim_col2, sim_col3 = st.columns(3)

        with sim_col1:
            sim_loan = st.slider(
                "Adjust Loan Amount ($)",
                min_value=5000,
                max_value=200000,
                value=int(loan_amount),
                step=5000
            )
        with sim_col2:
            sim_rate = st.slider(
                "Adjust APR (%)",
                min_value=2.0,
                max_value=24.0,
                value=float(interest_rate),
                step=0.5
            )
        with sim_col3:
            sim_cosigner = st.selectbox(
                "Toggle Co-Signer",
                ["No", "Yes"],
                index=0 if has_cosigner == "No" else 1
            )

        # Run simulated inference
        sim_input = current_input.copy()
        sim_input["LoanAmount"] = sim_loan
        sim_input["InterestRate"] = sim_rate
        sim_input["HasCoSigner"] = sim_cosigner

        sim_df = pd.DataFrame([sim_input])
        sim_prob = float(model.predict_proba(sim_df)[0][1])
        prob_delta = (sim_prob - raw_prob) * 100

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric(
                "Simulated Probability",
                f"{sim_prob*100:.1f}%",
                delta=f"{prob_delta:+.1f}%",
                delta_color="inverse"
            )
        with res_col2:
            sim_risk = compute_model_risk_band(sim_prob, THRESHOLD)
            st.markdown(f"""<div style="background:rgba(15,23,42,0.7); border:1px solid rgba(148,163,184,0.15); border-radius:10px; padding:0.8rem 1.2rem; margin-top:0.2rem;">
<b>Revised Model Risk Band:</b> <span style="color:{sim_risk['color']}; font-weight:700;">{sim_risk['band']}</span> ({sim_risk['decision']})<br>
<span style="font-size:0.8rem; color:#94A3B8;">{sim_risk['desc']}</span>
</div>""", unsafe_allow_html=True)

    # --- Decision Support Disclaimer ---
    st.markdown("""<div style="margin-top:1.2rem; padding:0.75rem 1rem; background:rgba(15,23,42,0.6); border-left:3px solid #64748B; border-radius:0 8px 8px 0; font-size:0.8rem; color:#94A3B8;">
<b>Decision Support Notice:</b> Model output is intended as decision support and should be reviewed alongside applicable lending policies, institutional risk appetite, and comprehensive applicant information.
</div>""", unsafe_allow_html=True)


# ============================================================
# PAGE 2: MODEL ANALYTICS & EMPIRICAL VALIDATION
# ============================================================

elif page == "📈 Model Analytics":

    st.markdown("""
    <div class="hero-container">
        <div class="hero-eyebrow">EMPIRICAL BENCHMARKING & VALIDATION</div>
        <div class="hero-title">Model Performance Analytics</div>
        <div class="hero-desc">
            Rigorous evaluation on the held-out test partition (N = 38,303 samples).
            Demonstrating why Gradient Boosting with calibrated decision threshold (τ = 0.19) delivers optimal recall and discrimination under severe target class imbalance.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- KPI METRICS GRID ---
    st.markdown("""
    <div class="metric-grid">
        <div class="kpi-card">
            <div class="kpi-label">Test Accuracy</div>
            <div class="kpi-value">82.56%</div>
            <div class="kpi-sub">Overall Correct</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">ROC-AUC</div>
            <div class="kpi-value" style="color:#38BDF8;">0.7595</div>
            <div class="kpi-sub">High Discrimination</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">PR-AUC</div>
            <div class="kpi-value" style="color:#A855F7;">0.3278</div>
            <div class="kpi-sub">Baseline: 0.1161</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Recall (Sensitivity)</div>
            <div class="kpi-value" style="color:#34D399;">44.81%</div>
            <div class="kpi-sub">Defaults Captured</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Precision</div>
            <div class="kpi-value" style="color:#F59E0B;">32.05%</div>
            <div class="kpi-sub">Positive Accuracy</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">F1-Score</div>
            <div class="kpi-value" style="color:#F43F5E;">37.37%</div>
            <div class="kpi-sub">Maximized @ 0.19</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_eval1, tab_eval2, tab_eval3, tab_eval4 = st.tabs([
        "📉 Discriminative Curves (ROC & PR)",
        "🎯 Confusion Matrix & Threshold Tuning",
        "⚖ Model Benchmark Comparison",
        "🧬 Feature Importance & Drivers"
    ])

    with tab_eval1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(create_roc_curve(eval_artifacts["roc_data"]), use_container_width=True)
            st.caption("The ROC curve achieves 0.7595 AUC, indicating strong rank-ordering capability across default risks.")
        with c2:
            st.plotly_chart(create_pr_curve(eval_artifacts["pr_data"]), use_container_width=True)
            st.caption("Under 11.6% class imbalance, PR-AUC of 0.3278 almost triples the baseline random expectation.")

    with tab_eval2:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(create_confusion_matrix_chart(eval_artifacts["confusion_matrix"]), use_container_width=True)
            st.caption("Classification matrix on 38,303 test instances at the calibrated decision threshold τ = 0.19.")
        with c2:
            st.plotly_chart(create_threshold_tuning_chart(eval_artifacts["threshold_sweep"], THRESHOLD), use_container_width=True)
            st.caption("Precision-Recall trade-off showing clear peak in F1-score at τ = 0.19 on validation split.")

    with tab_eval3:
        st.plotly_chart(create_benchmark_chart(eval_artifacts["model_benchmark"]), use_container_width=True)
        st.markdown("""
        <div style="background:rgba(15,23,42,0.8); border:1px solid rgba(148,163,184,0.15); border-radius:12px; padding:1.2rem; margin-top:0.8rem;">
            <b style="color:#38BDF8;">Benchmark Analysis Summary:</b>
            <ul style="color:#94A3B8; font-size:0.88rem; margin-top:0.4rem; padding-left:1.2rem;">
                <li><b>Gradient Boosting (Selected Champion):</b> Yields the highest ROC-AUC (0.7595) and PR-AUC (0.3278). With threshold tuning (τ = 0.19), it achieves balanced 44.81% recall and 37.37% F1.</li>
                <li><b>Logistic Regression:</b> Strong linear baseline (ROC-AUC 0.7573), but lower precision when recall is boosted.</li>
                <li><b>Random Forest (Default):</b> Suffers severe recall deficit (only 1.26% recall at default 0.50 threshold due to heavy class imbalance).</li>
                <li><b>Decision Tree:</b> Showed severe overfitting during cross-validation, performing poorly on generalized holdout test data.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab_eval4:
        st.plotly_chart(create_feature_importance_chart(eval_artifacts["feature_importances"], top_n=15), use_container_width=True)
        st.caption("Top 15 features by Gini importance in the final Gradient Boosting ensemble.")


# ============================================================
# PAGE 3: PROJECT DOSSIER & SUBMISSION OVERVIEW
# ============================================================

elif page == "📑 Project Dossier":

    st.markdown("""
    <div class="hero-container">
        <div class="hero-eyebrow">ACADEMIC & TECHNICAL SUBMISSION DOSSIER</div>
        <div class="hero-title">Loan Default Prediction System</div>
        <div class="hero-desc">
            Comprehensive documentation for the Semester 5 Machine Learning Capstone. Detailing the problem formulation, dataset characteristics, pipeline engineering, threshold calibration, and production integration.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="card-heading">🎯 Project Overview & Objective</div>
            <div style="font-size:0.88rem; color:#94A3B8; line-height:1.7;">
                Loan default estimation is a critical operational problem in retail banking.
                This project builds an automated, robust machine learning pipeline capable of predicting default probabilities for prospective loan applicants while maintaining high interpretability and precision.
            </div>
            <br>
            <div class="card-heading">📊 Dataset Anatomy</div>
            <div style="font-size:0.88rem; color:#94A3B8; line-height:1.7;">
                • <b>Instances:</b> 255,347 cleaned loan records<br>
                • <b>Features:</b> 16 input attributes (9 numeric financial metrics, 7 categorical indicators)<br>
                • <b>Target Variable:</b> Binary <code>Default</code> indicator (0 = Repaid, 1 = Defaulted)<br>
                • <b>Class Imbalance:</b> 88.39% non-default vs <b>11.61% default</b> (extreme class imbalance)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <div class="card-heading">⚙ Model Engineering & Split Protocol</div>
            <div style="font-size:0.88rem; color:#94A3B8; line-height:1.7;">
                • <b>Validation Split:</b> Stratified 70% Training (178,742), 15% Validation (38,302), and 15% Test (38,303).<br>
                • <b>Preprocessing Pipeline:</b> Encapsulated in <code>ColumnTransformer</code> with numeric passthrough and categorical <code>OneHotEncoder</code>.<br>
                • <b>Champion Algorithm:</b> <code>GradientBoostingClassifier</code> (100 estimators, deviance loss, random_state=42).<br>
                • <b>Threshold Calibration:</b> Default 0.50 threshold yields only ~5% recall due to imbalance. Optimized to <b>τ = 0.190</b> via validation F1-score maximization, boosting recall to 44.81%.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- ARCHITECTURE FLOW DIAGRAM ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">◈ End-to-End Pipeline Architecture Flow (Sankey Flowchart)</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-subheading">Feature ingestion, encoding transforms, ensemble tree processing, and calibrated risk band classification</div>', unsafe_allow_html=True)
    fig_sankey = create_pipeline_sankey()
    st.plotly_chart(fig_sankey, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- MODEL RISK BANDS CALIBRATION MATRIX ---
    st.markdown("### ◈ Model Risk Bands Calibration Matrix")
    st.markdown("""
    <div style="font-size:0.84rem; color:#94A3B8; margin-bottom:0.8rem;">
        Risk bands are internal statistical boundaries calibrated around the model's decision threshold (19.0%) and empirical default distribution.
        These are <b>Model Risk Bands</b> intended for decision support and do not represent official credit rating agency evaluations.
    </div>
    """, unsafe_allow_html=True)

    bands_data = [
        {"Model Risk Band": "VERY LOW RISK", "Probability Range": "0.0% – 5.0%", "Relation to Cutoff": "Well Below Threshold (<5%)", "Underwriting Guidance": "Substantially low estimated default probability. Eligible for prime streamlined processing."},
        {"Model Risk Band": "LOW RISK", "Probability Range": "5.0% – 12.0%", "Relation to Cutoff": "Below Threshold (5–12%)", "Underwriting Guidance": "Solid financial standing with low default likelihood well beneath threshold."},
        {"Model Risk Band": "MODERATE RISK", "Probability Range": "12.0% – 19.0%", "Relation to Cutoff": "Approaching Threshold (12–19%)", "Underwriting Guidance": "Within acceptable risk variance, below the 19.0% decision threshold."},
        {"Model Risk Band": "ELEVATED RISK", "Probability Range": "19.0% – 28.0%", "Relation to Cutoff": "Crosses Threshold (19–28%)", "Underwriting Guidance": "Crosses the 19.0% model threshold; heightened default probability requiring secondary review."},
        {"Model Risk Band": "HIGH RISK", "Probability Range": "28.0% – 45.0%", "Relation to Cutoff": "Substantially Above Threshold (28–45%)", "Underwriting Guidance": "Elevated risk profile with multiple compounding adverse debt factors."},
        {"Model Risk Band": "VERY HIGH RISK", "Probability Range": "> 45.0%", "Relation to Cutoff": "Severe Default Likelihood (>45%)", "Underwriting Guidance": "High default probability across key financial, leverage, and credit dimensions."}
    ]
    st.dataframe(pd.DataFrame(bands_data), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- CHECKLIST ---
    st.markdown("### ◈ Project Verification & Submission Checklist")
    checklist_data = [
        {"Requirement": "Automated ML Pipeline", "Implementation": "End-to-end scikit-learn Pipeline with ColumnTransformer & GradientBoosting", "Status": "✅ Verified"},
        {"Requirement": "Class Imbalance Mitigation", "Implementation": "Calibrated decision threshold (τ = 0.190) maximizing F1-score on validation set", "Status": "✅ Verified"},
        {"Requirement": "Interactive Frontend", "Implementation": "Streamlit application with futuristic glassmorphism theme and reactive inputs", "Status": "✅ Verified"},
        {"Requirement": "Graphs & Visualizations", "Implementation": "Speedometer Gauge, Risk Radar, Feature Impact Bar Chart, ROC Curve, PR Curve, Confusion Matrix", "Status": "✅ Verified"},
        {"Requirement": "Architecture Flow Diagram", "Implementation": "Interactive Sankey flow diagram mapping 16 features to 24-dim encoding and risk bands", "Status": "✅ Verified"},
        {"Requirement": "Model Risk Banding", "Implementation": "Descriptive non-institutional 6-tier Model Risk Bands derived from empirical threshold logic", "Status": "✅ Verified"},
        {"Requirement": "Explainability & What-If", "Implementation": "Feature impact breakdown chart and real-time sensitivity scenario simulator", "Status": "✅ Verified"},
        {"Requirement": "Clean Code & Dependencies", "Implementation": "Proper requirements.txt, cached memory management, and modular architecture", "Status": "✅ Verified"}
    ]
    st.dataframe(pd.DataFrame(checklist_data), use_container_width=True, hide_index=True)


# ============================================================
# GLOBAL FOOTER
# ============================================================

st.markdown("""
<div class="footer-box">
    <b>LOAN SENTRY</b> &nbsp;•&nbsp; Machine Learning Credit Risk Analytics System &nbsp;•&nbsp; Semester 5 Capstone Submission
    <br>
    Model-estimated probabilities are generated for analytical purposes and decision support.
</div>
""", unsafe_allow_html=True)
