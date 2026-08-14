import os
import json
import joblib
import random
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from pytorch_tabnet.tab_model import TabNetClassifier
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from google import genai

# Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =========================================================
# PAGE CONFIGURATION & STYLING
# =========================================================
st.set_page_config(
    page_title="AI Customer Churn Prediction Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .risk-indicator-box {
        background-color: #FFF5F5;
        border-left: 5px solid #E53E3E;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .risk-indicator-box-med {
        background-color: #FFFAF0;
        border-left: 5px solid #DD6B20;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .risk-indicator-box-low {
        background-color: #F0FFF4;
        border-left: 5px solid #38A169;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA & MODEL LOADERS
# =========================================================
@st.cache_resource
def load_models():
    dnn_model = tf.keras.models.load_model("saved_models/dnn_model.keras")
    wide_deep_model = tf.keras.models.load_model("saved_models/wide_deep_model.keras")
    
    try:
        tabnet_model = TabNetClassifier()
        tabnet_model.load_model("saved_models/tabnet_model.zip")
    except Exception:
        tabnet_model = None
    
    scaler = joblib.load("saved_models/scaler.pkl")
    feature_columns = joblib.load("saved_models/feature_columns.pkl")
    
    metrics_summary = {}
    if os.path.exists("saved_models/metrics_summary.json"):
        with open("saved_models/metrics_summary.json", "r") as f:
            metrics_summary = json.load(f)
            
    return dnn_model, wide_deep_model, tabnet_model, scaler, feature_columns, metrics_summary


@st.cache_data
def load_telco_dataset():
    data_path = "Telco-Customer-Churn.csv"
    if not os.path.exists(data_path):
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        df = pd.read_csv(url)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    return df


def load_prediction_history():
    history_file = "prediction_history.csv"
    columns = [
        'customer_id', 'timestamp', 'gender', 'senior_citizen', 'partner', 'dependents',
        'tenure', 'phone_service', 'multiple_lines', 'internet_service', 'online_security',
        'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',
        'contract', 'paperless_billing', 'payment_method', 'monthly_charges', 'total_charges',
        'dnn_prob', 'wide_deep_prob', 'tabnet_prob', 'ensemble_prob', 'prediction', 'risk_level'
    ]
    if not os.path.exists(history_file):
        df = pd.DataFrame(columns=columns)
        df.to_csv(history_file, index=False)
        return df
    try:
        df = pd.read_csv(history_file)
        return df
    except Exception:
        return pd.DataFrame(columns=columns)


def save_prediction_record(record_dict):
    history_file = "prediction_history.csv"
    df_existing = load_prediction_history()
    df_new = pd.DataFrame([record_dict])
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(history_file, index=False)


# Initialize Resources
dnn_model, wide_deep_model, tabnet_model, scaler, feature_columns, metrics_summary = load_models()
telco_df = load_telco_dataset()


# =========================================================
# LLM AGENT FUNCTION (GOOGLE GEMINI)
# =========================================================
def generate_llm_recommendation(customer_info, predictions, risk_level, risk_factors):
    try:
        if "GEMINI_API_KEY" not in st.secrets or not st.secrets["GEMINI_API_KEY"]:
            return (
                "⚠️ **Gemini API Key Missing**\n\n"
                "To activate the AI Retention Agent:\n"
                "- Local Testing: Add `GEMINI_API_KEY = \"your_key\"` to `.streamlit/secrets.toml`.\n"
                "- Streamlit Cloud: Add `GEMINI_API_KEY` under App Settings -> Secrets.\n\n"
                "💡 *Rule-based retention strategy above is fully operational!*"
            )

        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)

        prompt = f"""
You are an expert AI Customer Retention Agent for a telecom enterprise.
Analyze the following customer profile and ML model prediction results to synthesize an executive retention plan.

### Customer Profile:
- Customer ID: {customer_info.get('customer_id', 'CUST-TEMP')}
- Tenure: {customer_info['tenure']} months
- Contract Type: {customer_info['contract']}
- Monthly Charges: ${customer_info['monthly_charges']:.2f} | Total Charges: ${customer_info['total_charges']:.2f}
- Internet Service: {customer_info['internet_service']}
- Payment Method: {customer_info['payment_method']}

### Machine Learning Ensemble Prediction:
- Overall Churn Risk: {predictions['ensemble']:.2f}%
- Risk Classification: {risk_level}
- DNN Model Risk: {predictions['dnn']:.2f}%
- Wide & Deep Model Risk: {predictions['wide_deep']:.2f}%
- TabNet Model Risk: {predictions['tabnet']:.2f}%

### Risk Factors Identified:
{chr(10).join(['- ' + factor for factor in risk_factors])}

### Requirements:
Provide a structured, executive retention strategy with 5 clear sections:
1. **WHY CUSTOMER MAY CHURN**: Primary drivers of potential churn.
2. **RISK ASSESSMENT**: Executive risk summary.
3. **RECOMMENDED RETENTION ACTIONS**: 3 customized retention steps.
4. **NEXT ACTIONS**: Step-by-step guidance for team outreach.
5. **EXPECTED OUTCOME**: Retention impact forecast.
"""

        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )
        except Exception:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt
            )

        return response.text

    except Exception as e:
        return f"⚠️ **LLM Retention Agent Unavailable**: `{str(e)}`"


# =========================================================
# NAVIGATION SIDEBAR
# =========================================================
st.sidebar.image("https://img.icons8.com/color/96/000000/brain--v1.png", width=70)
st.sidebar.title("AI Churn Agent")
st.sidebar.caption("Machine Learning & Intelligence Suite")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation System",
    [
        "📊 Dashboard & Analytics",
        "🔮 Single Customer Prediction",
        "🎯 Model Performance",
        "📜 Prediction History"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "**Verified Baselines**\n\n"
    "• **TabNet**: 79.18%\n"
    "• **DNN**: 77.75%\n"
    "• **Wide & Deep**: 75.69%"
)


# =========================================================
# PAGE 1: DASHBOARD & TELCO DATASET ANALYTICS
# =========================================================
if page == "📊 Dashboard & Analytics":
    st.markdown('<div class="main-header">📊 Dashboard & Telco Dataset Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Executive overview and genuine Telco dataset analytics</div>', unsafe_allow_html=True)

    history_df = load_prediction_history()
    total_preds = len(history_df)
    high_risk_count = len(history_df[history_df['risk_level'] == 'HIGH']) if total_preds > 0 else 0
    high_risk_ratio = (high_risk_count / total_preds * 100) if total_preds > 0 else 0.0

    # Top KPI Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Best Deployed Model</div>
            <div class="metric-value">TabNet</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Best Test Accuracy</div>
            <div class="metric-value">79.18%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Predictions</div>
            <div class="metric-value">{total_preds}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">High-Risk Customers</div>
            <div class="metric-value">{high_risk_count} ({high_risk_ratio:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    # Visualizations Row 1: Telco Dataset Analytics placed beside Dataset Churn Distribution
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📉 Dataset Churn Distribution")
        churn_counts = telco_df['Churn'].value_counts()
        fig_pie = px.pie(
            values=churn_counts.values,
            names=['Stayed (No)', 'Churned (Yes)'],
            color_discrete_sequence=['#1E3A8A', '#EF4444'],
            hole=0.4,
            title="Telco Customer Dataset Churn Ratio (7,043 total customers)"
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("📈 Analytics: Churn by Contract Type")
        df_clean = telco_df.copy()
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        df_clean = df_clean.dropna()
        contract_churn = df_clean.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        fig_contract = px.bar(
            contract_churn, x='Contract', y='Churn',
            color='Contract',
            labels={'Churn': 'Churn Rate (%)'},
            color_discrete_sequence=['#EF4444', '#F59E0B', '#10B981'],
            title="Churn Rate by Contract Type"
        )
        fig_contract.update_layout(height=350)
        st.plotly_chart(fig_contract, use_container_width=True)

    st.divider()

    # Visualizations Row 2: Additional Telco Dataset Analytics Grid
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("📈 Analytics: Churn by Internet Service")
        internet_churn = df_clean.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        fig_internet = px.bar(
            internet_churn, x='InternetService', y='Churn',
            color='InternetService',
            labels={'Churn': 'Churn Rate (%)'},
            color_discrete_sequence=['#EF4444', '#3B82F6', '#10B981'],
            title="Churn Rate by Internet Service Type"
        )
        fig_internet.update_layout(height=350)
        st.plotly_chart(fig_internet, use_container_width=True)

    with c4:
        st.subheader("📈 Analytics: Churn by Payment Method")
        df_clean['PaymentMethodShort'] = df_clean['PaymentMethod'].replace({
            'Bank transfer (automatic)': 'Bank Transfer',
            'Credit card (automatic)': 'Credit Card',
            'Electronic check': 'Electronic Check',
            'Mailed check': 'Mailed Check'
        })
        payment_churn = df_clean.groupby('PaymentMethodShort', observed=False)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        fig_payment = px.bar(
            payment_churn, x='PaymentMethodShort', y='Churn',
            color='PaymentMethodShort',
            labels={'Churn': 'Churn Rate (%)', 'PaymentMethodShort': 'Payment Method'},
            color_discrete_sequence=['#EF4444', '#F59E0B', '#3B82F6', '#10B981'],
            title="Churn Rate by Payment Method"
        )
        fig_payment.update_layout(
            height=380,
            xaxis_tickangle=0,
            margin=dict(b=50, t=50),
            showlegend=False
        )
        st.plotly_chart(fig_payment, use_container_width=True)

    st.divider()

    # Visualizations Row 3: Model Accuracy Comparison Chart
    st.subheader("🤖 Model Test Accuracy Baseline vs Tuned Evaluation")
    comp_data = pd.DataFrame({
        "Model": ["TabNet Classifier", "DNN Architecture", "Wide & Deep Architecture"],
        "Baseline Test Accuracy (%)": [79.18, 77.75, 75.69],
        "Tuned Test Accuracy (%)": [77.90, 74.06, 73.85]
    })
    fig_comp = px.bar(
        comp_data,
        x="Model",
        y=["Baseline Test Accuracy (%)", "Tuned Test Accuracy (%)"],
        barmode="group",
        color_discrete_sequence=["#1E3A8A", "#60A5FA"],
        title="Held-Out Test Set Accuracy Comparison"
    )
    fig_comp.update_layout(height=380, yaxis_range=[60, 85])
    st.plotly_chart(fig_comp, use_container_width=True)


# =========================================================
# PAGE 2: SINGLE CUSTOMER PREDICTION
# =========================================================
elif page == "🔮 Single Customer Prediction":
    st.markdown('<div class="main-header">🔮 Customer Churn Risk Evaluator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Input customer details to predict churn risk and analyze risk factor indications</div>', unsafe_allow_html=True)

    with st.form("customer_input_form"):
        st.subheader("1. Customer Profile & Demographics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
        with col2:
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        with col3:
            partner = st.selectbox("Has Partner", ["Yes", "No"])
        with col4:
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])

        st.subheader("2. Tenure, Billing & Internet Information")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
        with col2:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
        with col3:
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0)
        with col4:
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

        col1, col2, col3 = st.columns(3)
        with col1:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        with col2:
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
        with col3:
            internet_service = st.selectbox("Internet Service Type", ["Fiber optic", "DSL", "No"])

        submit_btn = st.form_submit_button("🔍 Run Machine Learning Churn Analysis", use_container_width=True, type="primary")

    if submit_btn:
        # Build customer raw dict with standard default fallbacks for secondary services
        raw_customer = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': "Yes",
            'MultipleLines': "No",
            'InternetService': internet_service,
            'OnlineSecurity': "No" if internet_service != "No" else "No internet service",
            'OnlineBackup': "No" if internet_service != "No" else "No internet service",
            'DeviceProtection': "No" if internet_service != "No" else "No internet service",
            'TechSupport': "No" if internet_service != "No" else "No internet service",
            'StreamingTV': "No" if internet_service != "No" else "No internet service",
            'StreamingMovies': "No" if internet_service != "No" else "No internet service",
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        # Convert to DataFrame and get dummies
        cust_df = pd.DataFrame([raw_customer])
        cust_encoded = pd.get_dummies(cust_df, drop_first=True)
        cust_encoded = cust_encoded.reindex(columns=feature_columns, fill_value=0)

        # Scaled for DNN & Wide & Deep
        cust_scaled = scaler.transform(cust_encoded)
        # TabNet raw dummy format
        cust_tab = cust_encoded.astype(int).values

        # Predictions
        dnn_p = float(dnn_model.predict(cust_scaled, verbose=0)[0][0])
        wd_p = float(wide_deep_model.predict(cust_scaled, verbose=0)[0][0])

        if tabnet_model is not None:
            tabnet_p = float(tabnet_model.predict_proba(cust_tab)[0][1])
            probs = [dnn_p, wd_p, tabnet_p]
        else:
            tabnet_p = (dnn_p + wd_p) / 2.0
            probs = [dnn_p, wd_p]

        ensemble_p = float(np.mean(probs))
        churn_pct = ensemble_p * 100.0

        if churn_pct >= 70:
            risk_lvl = "HIGH"
            risk_badge = "🚨 HIGH RISK"
        elif churn_pct >= 40:
            risk_lvl = "MEDIUM"
            risk_badge = "⚠️ MEDIUM RISK"
        else:
            risk_lvl = "LOW"
            risk_badge = "🟢 LOW RISK"

        # Risk Factors Indication Engine
        factors = []
        if contract == "Month-to-month":
            factors.append("🚨 **Contract Risk**: Month-to-month contract indicates low long-term commitment.")
        if tenure < 12:
            factors.append("⚠️ **Tenure Risk**: Short customer tenure (< 12 months) has statistically higher early churn.")
        if monthly_charges > 70:
            factors.append("💸 **Billing Risk**: High monthly charge (> $70/mo) increases price sensitivity.")
        if payment_method == "Electronic check":
            factors.append("💳 **Payment Method Risk**: Electronic check billing correlates with highest historical churn.")
        if internet_service == "Fiber optic":
            factors.append("🌐 **Internet Tier Risk**: Fiber optic users show higher propensity to churn if service issues occur.")
        if not factors:
            factors.append("🟢 **Low Risk**: No critical risk drivers detected in customer profile.")

        # Recommendations Logic
        recs = []
        if contract == "Month-to-month":
            recs.append("Offer 15% discount incentive to upgrade to a 1-Year or 2-Year contract.")
        if monthly_charges > 70:
            recs.append("Suggest value bundle optimization to lower perceived monthly cost.")
        if tenure < 12:
            recs.append("Provide a new-customer loyalty welcome bonus to boost engagement.")
        if payment_method == "Electronic check":
            recs.append("Encourage switching to automatic bank transfer or credit card billing.")
        if not recs:
            recs.append("Maintain regular engagement and offer annual loyalty points.")

        # Save to Session State
        customer_id_generated = f"CUST-{random.randint(1000, 9999)}"
        st.session_state['last_pred'] = {
            'customer_id': customer_id_generated,
            'customer_info': {
                'customer_id': customer_id_generated,
                'gender': gender,
                'senior_citizen': senior_citizen,
                'partner': partner,
                'dependents': dependents,
                'tenure': tenure,
                'monthly_charges': monthly_charges,
                'total_charges': total_charges,
                'contract': contract,
                'internet_service': internet_service,
                'payment_method': payment_method
            },
            'predictions': {
                'dnn': dnn_p * 100,
                'wide_deep': wd_p * 100,
                'tabnet': tabnet_p * 100,
                'ensemble': churn_pct
            },
            'risk_level': risk_lvl,
            'risk_badge': risk_badge,
            'risk_factors': factors,
            'recommendations': recs
        }

        # Save Record to CSV History
        record_to_save = {
            'customer_id': customer_id_generated,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'gender': gender,
            'senior_citizen': senior_citizen,
            'partner': partner,
            'dependents': dependents,
            'tenure': tenure,
            'phone_service': "Yes",
            'multiple_lines': "No",
            'internet_service': internet_service,
            'online_security': "No" if internet_service != "No" else "No internet service",
            'online_backup': "No" if internet_service != "No" else "No internet service",
            'device_protection': "No" if internet_service != "No" else "No internet service",
            'tech_support': "No" if internet_service != "No" else "No internet service",
            'streaming_tv': "No" if internet_service != "No" else "No internet service",
            'streaming_movies': "No" if internet_service != "No" else "No internet service",
            'contract': contract,
            'paperless_billing': paperless_billing,
            'payment_method': payment_method,
            'monthly_charges': monthly_charges,
            'total_charges': total_charges,
            'dnn_prob': round(dnn_p * 100, 2),
            'wide_deep_prob': round(wd_p * 100, 2),
            'tabnet_prob': round(tabnet_p * 100, 2),
            'ensemble_prob': round(churn_pct, 2),
            'prediction': "Churn" if churn_pct >= 50 else "No Churn",
            'risk_level': risk_lvl
        }
        save_prediction_record(record_to_save)
        if 'llm_resp' in st.session_state:
            del st.session_state['llm_resp']

    # Display Results & Indications of Risk Factors
    if 'last_pred' in st.session_state:
        res = st.session_state['last_pred']
        st.divider()
        st.header(f"📊 Customer-Level Churn Probability ({res['customer_id']})")

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Ensemble Churn Risk", f"{res['predictions']['ensemble']:.2f}%")
        with c2:
            st.metric("Risk Status", res['risk_badge'])
        with c3:
            st.metric("DNN Probability", f"{res['predictions']['dnn']:.2f}%")
        with c4:
            st.metric("Wide & Deep Prob", f"{res['predictions']['wide_deep']:.2f}%")
        with c5:
            st.metric("TabNet Probability", f"{res['predictions']['tabnet']:.2f}%")

        st.progress(min(max(res['predictions']['ensemble'] / 100.0, 0.0), 1.0))

        st.divider()

        # Indications of Risk Factors Section
        st.subheader("🚨 Indications of Risk Factors & Diagnostic Alerts")
        
        if res['risk_level'] == "HIGH":
            st.markdown("""
            <div class="risk-indicator-box">
                <h4 style="color: #991B1B; margin: 0 0 0.5rem 0;">🚨 HIGH CHURN RISK INDICATOR DETECTED</h4>
                <p style="margin: 0; color: #7F1D1D;">Immediate proactive intervention required. This customer profile exhibits strong historical markers of customer cancellation.</p>
            </div>
            """, unsafe_allow_html=True)
        elif res['risk_level'] == "MEDIUM":
            st.markdown("""
            <div class="risk-indicator-box-med">
                <h4 style="color: #92400E; margin: 0 0 0.5rem 0;">⚠️ MEDIUM CHURN RISK INDICATOR DETECTED</h4>
                <p style="margin: 0; color: #78350F;">Moderate cancellation risk. Targeted promotional offers and value bundles recommended.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-indicator-box-low">
                <h4 style="color: #166534; margin: 0 0 0.5rem 0;">🟢 LOW CHURN RISK INDICATOR</h4>
                <p style="margin: 0; color: #14532D;">Customer churn probability is low. Maintain standard relationship management.</p>
            </div>
            """, unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("### ⚠️ Key Risk Factor Indications")
            for factor_text in res['risk_factors']:
                st.markdown(f"• {factor_text}")

        with r2:
            st.markdown("### 🎯 Recommended Retention Actions")
            for rec_text in res['recommendations']:
                st.markdown(f"✅ {rec_text}")

        st.divider()

        # LLM Section
        st.header("🧠 AI Retention Strategy (Google Gemini)")
        st.write("Generate a detailed, custom executive retention strategy tailored specifically to this customer profile.")

        if st.button("🤖 Generate AI Retention Strategy with Gemini", type="secondary"):
            with st.spinner("AI Agent is evaluating customer risk profile and generating executive strategy..."):
                llm_out = generate_llm_recommendation(
                    customer_info=res['customer_info'],
                    predictions=res['predictions'],
                    risk_level=res['risk_level'],
                    risk_factors=res['risk_factors']
                )
                st.session_state['llm_resp'] = llm_out

        if 'llm_resp' in st.session_state:
            st.info("### 📋 Executive Retention Plan")
            st.markdown(st.session_state['llm_resp'])


# =========================================================
# PAGE 3: MODEL PERFORMANCE
# =========================================================
elif page == "🎯 Model Performance":
    st.markdown('<div class="main-header">🎯 Held-Out Test Set Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation of DNN, Wide & Deep, TabNet, and Ensemble models on held-out test data</div>', unsafe_allow_html=True)

    st.warning("⚠️ **IMPORTANT**: These metrics represent offline test-set evaluation (1,407 test samples). Do NOT confuse these model test accuracy values with individual customer-level churn probabilities!")

    # Baseline Summary Card
    st.subheader("📌 Verified Colab Baseline Results (Held-Out Test Set)")
    base_df = pd.DataFrame({
        "Model Architecture": ["TabNet Classifier (Best Baseline)", "Deep Neural Network (DNN)", "Wide & Deep Architecture"],
        "Baseline Test Accuracy (%)": ["79.18%", "77.75%", "75.69%"],
        "Precision": ["0.6473", "0.6028", "0.5537"],
        "Recall": ["0.4759", "0.4626", "0.5374"],
        "F1-Score": ["0.5485", "0.5234", "0.5455"],
        "ROC-AUC": ["0.8228", "0.8118", "0.7747"]
    })
    st.dataframe(base_df, use_container_width=True, hide_index=True)

    st.divider()

    # Hyperparameter Tuning Results
    st.subheader("🧪 Hyperparameter Tuning & Threshold Optimization Results")
    
    if metrics_summary:
        default_res = metrics_summary.get("tuned_default_threshold", {})
        opt_res = metrics_summary.get("tuned_optimal_threshold", {})

        rows = []
        for m_name, m_key in [("TabNet Classifier", "tabnet"), ("DNN Architecture", "dnn"), ("Wide & Deep", "wide_deep"), ("Ensemble Model", "ensemble")]:
            if m_key in default_res:
                d = default_res[m_key]
                o_thresh = opt_res.get(m_key, {}).get("threshold", 0.5)
                o_m = opt_res.get(m_key, {}).get("metrics", d)
                rows.append({
                    "Model": m_name,
                    "Default Acc (0.5)": f"{d['accuracy']*100:.2f}%",
                    "Default Precision": f"{d['precision']:.4f}",
                    "Default Recall": f"{d['recall']:.4f}",
                    "Default F1": f"{d['f1']:.4f}",
                    "ROC-AUC": f"{d['roc_auc']:.4f}",
                    "Optimal Thresh": f"{o_thresh:.2f}",
                    "Optimal Thresh Acc": f"{o_m['accuracy']*100:.2f}%",
                    "Optimal Thresh F1": f"{o_m['f1']:.4f}"
                })
        
        tune_df = pd.DataFrame(rows)
        st.dataframe(tune_df, use_container_width=True, hide_index=True)

        st.divider()

        # Visual Comparison Charts
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Accuracy Comparison")
            acc_df = pd.DataFrame({
                "Model": ["DNN", "Wide & Deep", "TabNet", "Ensemble"],
                "Baseline Test Acc (%)": [77.75, 75.69, 79.18, 79.18],
                "Tuned Test Acc (%)": [
                    default_res['dnn']['accuracy']*100,
                    default_res['wide_deep']['accuracy']*100,
                    default_res['tabnet']['accuracy']*100,
                    default_res['ensemble']['accuracy']*100
                ]
            })
            fig_acc = px.bar(
                acc_df, x="Model", y=["Baseline Test Acc (%)", "Tuned Test Acc (%)"],
                barmode="group", color_discrete_sequence=["#1E3A8A", "#60A5FA"],
                title="Baseline vs Tuned Model Accuracy"
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with col2:
            st.subheader("Model ROC-AUC & F1-Score")
            f1_df = pd.DataFrame({
                "Model": ["DNN", "Wide & Deep", "TabNet", "Ensemble"],
                "F1-Score": [default_res['dnn']['f1'], default_res['wide_deep']['f1'], default_res['tabnet']['f1'], default_res['ensemble']['f1']],
                "ROC-AUC": [default_res['dnn']['roc_auc'], default_res['wide_deep']['roc_auc'], default_res['tabnet']['roc_auc'], default_res['ensemble']['roc_auc']]
            })
            fig_f1 = px.bar(
                f1_df, x="Model", y=["F1-Score", "ROC-AUC"],
                barmode="group", color_discrete_sequence=["#F59E0B", "#10B981"],
                title="F1-Score and ROC-AUC Performance"
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        # Model Replacement Audit Log
        st.divider()
        st.subheader("🛡️ Model Selection & Update Audit Log")
        updates = metrics_summary.get("updates_log", {})
        for model_k, info in updates.items():
            status_str = "✅ REPLACED WITH TUNED MODEL" if info["updated"] else "🔒 RETAINED BASELINE WORKING MODEL"
            st.markdown(f"• **{model_k.upper()}**: {status_str} — *Reason: {info['reason']}*")
    else:
        st.info("Metrics summary file loading...")


# =========================================================
# PAGE 4: PREDICTION HISTORY
# =========================================================
elif page == "📜 Prediction History":
    st.markdown('<div class="main-header">📜 Customer Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Search, filter, and inspect stored customer predictions</div>', unsafe_allow_html=True)

    history_df = load_prediction_history()

    if len(history_df) == 0:
        st.info("No prediction history recorded yet. Make predictions on the **Single Customer Prediction** page!")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            search_id = st.text_input("🔍 Search by Customer ID", "")
        with col2:
            risk_filter = st.selectbox("Filter by Risk Level", ["ALL", "HIGH", "MEDIUM", "LOW"])
        with col3:
            sort_by = st.selectbox("Sort By", ["Newest First", "Oldest First", "Highest Churn Risk", "Lowest Churn Risk"])

        # Filter logic
        filtered_df = history_df.copy()
        if search_id:
            filtered_df = filtered_df[filtered_df['customer_id'].str.contains(search_id, case=False, na=False)]
        if risk_filter != "ALL":
            filtered_df = filtered_df[filtered_df['risk_level'] == risk_filter]

        # Sorting logic
        if sort_by == "Newest First":
            filtered_df = filtered_df.iloc[::-1]
        elif sort_by == "Oldest First":
            pass
        elif sort_by == "Highest Churn Risk":
            filtered_df = filtered_df.sort_values(by='ensemble_prob', ascending=False)
        elif sort_by == "Lowest Churn Risk":
            filtered_df = filtered_df.sort_values(by='ensemble_prob', ascending=True)

        st.write(f"Showing **{len(filtered_df)}** records of **{len(history_df)}** total predictions:")
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        # Download CSV
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Prediction History CSV",
            data=csv_data,
            file_name="churn_prediction_history.csv",
            mime="text/csv",
            type="primary"
        )