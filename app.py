import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from pytorch_tabnet.tab_model import TabNetClassifier
from google import genai


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Customer Churn & Retention Agent",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():
    dnn_model = tf.keras.models.load_model("saved_models/dnn_model.keras")
    wide_deep_model = tf.keras.models.load_model("saved_models/wide_deep_model.keras")
    
    tabnet_model = TabNetClassifier()
    tabnet_model.load_model("saved_models/tabnet_model.zip")
    
    scaler = joblib.load("saved_models/scaler.pkl")
    feature_columns = joblib.load("saved_models/feature_columns.pkl")
    
    return dnn_model, wide_deep_model, tabnet_model, scaler, feature_columns


dnn_model, wide_deep_model, tabnet_model, scaler, feature_columns = load_models()


# =========================================================
# LLM AGENT FUNCTION
# =========================================================

def generate_llm_recommendation(customer_info, predictions, risk_level, risk_factors):
    """
    Generates personalized retention actions using Google Gemini LLM via google-genai SDK.
    Handles missing API keys and errors gracefully.
    """
    try:
        if "GEMINI_API_KEY" not in st.secrets or not st.secrets["GEMINI_API_KEY"]:
            return (
                "⚠️ **Gemini API Key missing.**\n\n"
                "To activate the AI Retention Agent:\n"
                "- **Local Testing**: Add `GEMINI_API_KEY = \"your_api_key_here\"` in `.streamlit/secrets.toml`.\n"
                "- **Streamlit Cloud**: Add `GEMINI_API_KEY` under **App Settings -> Secrets**.\n\n"
                "💡 *Rule-based retention strategy above is fully operational!*"
            )

        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)

        prompt = f"""
You are an expert AI Customer Retention Agent for a telecom company.
Analyze the following customer profile and model prediction results to create a tailored retention strategy.

### Customer Profile:
- Tenure: {customer_info['tenure']} months
- Monthly Charges: ${customer_info['monthly_charges']:.2f}
- Total Charges: ${customer_info['total_charges']:.2f}
- Contract Type: {customer_info['contract']}
- Internet Service: {customer_info['internet_service']}
- Payment Method: {customer_info['payment_method']}

### Machine Learning Ensemble Prediction:
- Overall Churn Probability: {predictions['ensemble']:.2f}%
- Risk Level: {risk_level}
- DNN Model Prediction: {predictions['dnn']:.2f}%
- Wide & Deep Model Prediction: {predictions['wide_deep']:.2f}%
- TabNet Model Prediction: {predictions['tabnet']:.2f}%

### Key Risk Factors Identified:
{chr(10).join(['- ' + factor for factor in risk_factors])}

### Request:
Provide a structured, executive retention strategy with these 5 concise sections:
1. **Churn Risk Explanation**: Why this customer might leave.
2. **Main Risk Factors**: The critical drivers of potential churn.
3. **3 Personalized Retention Actions**: Specific steps the customer success team should take.
4. **Recommended Customer Offer**: A specific discount, plan upgrade, or contract term.
5. **Short Action Plan for Retention Team**: Step-by-step next steps for outreach.

Keep the response practical, professional, concise, and business-focused.
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
        return (
            "⚠️ **LLM Retention Agent currently unavailable.**\n\n"
            "The rule-based retention recommendations above are still available.\n\n"
            f"**Error details**: `{str(e)}`"
        )


# =========================================================
# HEADER & TITLE
# =========================================================

st.title("🤖 AI Customer Churn Prediction & Retention Agent")
st.markdown(
    "Predict customer churn risk using Deep Learning models (DNN, Wide & Deep, TabNet) "
    "and generate personalized retention strategies powered by **Google Gemini AI Agent**."
)
st.divider()


# =========================================================
# CUSTOMER INPUT FORM
# =========================================================

st.header("👤 Customer Profile Input")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12,
        help="Number of months customer has stayed with the company."
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        value=70.0,
        help="Current monthly charge amount."
    )

with col3:
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=840.0,
        help="Total amount charged to customer to date."
    )

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    internet_service = st.selectbox(
        "Internet Service Type",
        ["DSL", "Fiber optic", "No"]
    )

with col3:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

st.divider()


# =========================================================
# DATA PREPARATION HELPER
# =========================================================

def prepare_customer_data():
    customer = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Contract": [contract],
        "InternetService": [internet_service],
        "PaymentMethod": [payment_method]
    })
    
    customer = pd.get_dummies(customer)
    customer = customer.reindex(columns=feature_columns, fill_value=0)
    customer_scaled = scaler.transform(customer)
    
    return customer_scaled


# =========================================================
# PREDICTION EXECUTION
# =========================================================

if st.button("🔍 Predict Customer Churn Risk", use_container_width=True, type="primary"):
    try:
        customer_scaled = prepare_customer_data()

        # Model Inferences
        dnn_prob = float(dnn_model.predict(customer_scaled, verbose=0)[0][0])
        wide_deep_prob = float(wide_deep_model.predict(customer_scaled, verbose=0)[0][0])
        tabnet_prob = float(tabnet_model.predict_proba(customer_scaled)[0][1])

        # Ensemble Average Probability
        ensemble_prob = float(np.mean([dnn_prob, wide_deep_prob, tabnet_prob]))
        churn_percentage = ensemble_prob * 100.0

        # Risk Classification
        if churn_percentage >= 70:
            risk_level = "HIGH"
            risk_icon = "🔴"
        elif churn_percentage >= 40:
            risk_level = "MEDIUM"
            risk_icon = "🟡"
        else:
            risk_level = "LOW"
            risk_icon = "🟢"

        # Risk Factors Rule Engine
        risk_factors = []
        if contract == "Month-to-month":
            risk_factors.append("Month-to-month contract indicates lower customer commitment.")
        if tenure < 12:
            risk_factors.append("Customer has relatively short tenure (< 12 months).")
        if monthly_charges > 70:
            risk_factors.append("Monthly charges are high (> $70).")
        if payment_method == "Electronic check":
            risk_factors.append("Payment method is Electronic check (higher historical churn rate).")
        if internet_service == "Fiber optic":
            risk_factors.append("Fiber optic service user (requires monitoring customer satisfaction).")
        if not risk_factors:
            risk_factors.append("No major predefined risk factors detected.")

        # Predefined Recommendations Rule Engine
        recommendations = []
        if contract == "Month-to-month":
            recommendations.append("Offer an incentive discount to switch to a 1-year or 2-year contract.")
        if monthly_charges > 70:
            recommendations.append("Suggest a value bundle or optimized service tier to lower cost perception.")
        if tenure < 12:
            recommendations.append("Provide a new-customer loyalty bonus or free add-on feature.")
        if payment_method == "Electronic check":
            recommendations.append("Encourage switching to automatic bank transfer or credit card billing.")
        if internet_service == "Fiber optic":
            recommendations.append("Conduct a technical health check and offer service quality check-in.")
        if not recommendations:
            recommendations.append("Continue regular engagement and offer standard loyalty perks.")

        # Persist prediction state in Streamlit Session State
        st.session_state['prediction_results'] = {
            'customer_info': {
                'tenure': tenure,
                'monthly_charges': monthly_charges,
                'total_charges': total_charges,
                'contract': contract,
                'internet_service': internet_service,
                'payment_method': payment_method
            },
            'predictions': {
                'dnn': dnn_prob * 100.0,
                'wide_deep': wide_deep_prob * 100.0,
                'tabnet': tabnet_prob * 100.0,
                'ensemble': churn_percentage
            },
            'risk_level': risk_level,
            'risk_icon': risk_icon,
            'risk_factors': risk_factors,
            'recommendations': recommendations
        }

        # Clear any old LLM response when a new prediction is made
        if 'llm_response' in st.session_state:
            del st.session_state['llm_response']

    except Exception as e:
        st.error("Error computing prediction results.")
        st.exception(e)


# =========================================================
# DISPLAY RESULTS SECTION
# =========================================================

if 'prediction_results' in st.session_state:
    res = st.session_state['prediction_results']
    
    st.header("📊 Churn Risk Analysis & Predictions")

    # Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Ensemble Churn Risk", f"{res['predictions']['ensemble']:.2f}%")
    with col2:
        st.metric("Risk Level", f"{res['risk_icon']} {res['risk_level']}")
    with col3:
        st.metric("DNN Model", f"{res['predictions']['dnn']:.2f}%")
    with col4:
        st.metric("Wide & Deep Model", f"{res['predictions']['wide_deep']:.2f}%")
    with col5:
        st.metric("TabNet Model", f"{res['predictions']['tabnet']:.2f}%")

    st.progress(min(max(res['predictions']['ensemble'] / 100.0, 0.0), 1.0))

    # Model Breakdown Table
    st.subheader("🤖 Model Prediction Comparison")
    model_df = pd.DataFrame({
        "Model Name": ["DNN (Deep Neural Network)", "Wide & Deep Architecture", "TabNet Classifier"],
        "Predicted Churn Probability (%)": [
            f"{res['predictions']['dnn']:.2f}%",
            f"{res['predictions']['wide_deep']:.2f}%",
            f"{res['predictions']['tabnet']:.2f}%"
        ]
    })
    st.dataframe(model_df, use_container_width=True, hide_index=True)

    # Risk Factors & Rule-based Recommendations
    col_risk, col_rec = st.columns(2)

    with col_risk:
        st.subheader("⚠️ Detected Risk Factors")
        for factor in res['risk_factors']:
            st.write(f"• {factor}")

    with col_rec:
        st.subheader("🎯 Rule-Based Recommendations")
        for rec in res['recommendations']:
            st.write(f"✅ {rec}")

    # AI Decision Banner
    st.subheader("🛡️ Strategic Retention Priority")
    if res['risk_level'] == "HIGH":
        st.error("🚨 **URGENT**: High churn probability detected. Immediate proactive retention intervention required!")
    elif res['risk_level'] == "MEDIUM":
        st.warning("⚠️ **MEDIUM PRIORITY**: Moderate churn risk. Targeted promotional offers recommended.")
    else:
        st.success("🟢 **LOW PRIORITY**: Standard engagement. Customer churn probability is low.")

    st.divider()

    # =========================================================
    # LLM RETENTION AGENT SECTION
    # =========================================================

    st.header("🧠 LLM Customer Retention Agent (Gemini AI)")
    st.write("Generate a detailed, personalized executive retention plan tailored specifically to this customer.")

    if st.button("🤖 Generate AI Retention Strategy with Gemini", type="secondary"):
        with st.spinner("AI Retention Agent is evaluating customer data and generating strategy..."):
            llm_text = generate_llm_recommendation(
                customer_info=res['customer_info'],
                predictions=res['predictions'],
                risk_level=res['risk_level'],
                risk_factors=res['risk_factors']
            )
            st.session_state['llm_response'] = llm_text

    # Display LLM Response if generated
    if 'llm_response' in st.session_state:
        st.info("### 📋 Executive Retention Plan")
        st.markdown(st.session_state['llm_response'])