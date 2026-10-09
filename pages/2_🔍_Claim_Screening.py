import os
import streamlit as st
import pandas as pd

from utils.prediction import predict_claim, load_model_bundle
from utils.explainability import get_claim_explanation

st.set_page_config(
    page_title="Claim Screening | FraudShield AI",
    page_icon="🔍",
    layout="wide"
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .result-legitimate {
        background: rgba(16, 185, 129, 0.12);
        border: 2px solid #10b981;
        color: #6ee7b7;
        padding: 1.8rem;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .result-moderate {
        background: rgba(245, 158, 11, 0.15);
        border: 2px solid #f59e0b;
        color: #fcd34d;
        padding: 1.8rem;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .result-fraudulent {
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid #ef4444;
        color: #fca5a5;
        padding: 1.8rem;
        border-radius: 18px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .status-title {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .prob-value {
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0.4rem 0;
    }
    .disclaimer-text {
        color: #94a3b8;
        font-size: 0.85rem;
        font-style: italic;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Load Model Metadata
# ---------------------------------------------------------
try:
    _, bundle = load_model_bundle()
    best_model_name = bundle.get("best_model_name", "Trained ML Model")
except Exception as e:
    st.error(f"Model loading error: {e}. Please run 'python train_model.py' first.")
    st.stop()

# ---------------------------------------------------------
# Header & Scenario Presets
# ---------------------------------------------------------
st.title("🔍 Transaction & Claim Screening")
st.caption(f"Automatic fraud classification powered by **{best_model_name}** with calibrated risk tiers.")

# Presets for quick evaluation
st.markdown("##### 🚀 Quick Scenario Presets")
preset_cols = st.columns(3)

if "preset_data" not in st.session_state:
    st.session_state.preset_data = {
        "amount": 85.0,
        "type": "POS Payment",
        "time": 14.0,
        "device": "Mobile",
        "loc": "New York",
        "pm": "Credit Card",
        "prev_fraud": 0,
        "age": 250,
        "txns_24h": 2
    }

with preset_cols[0]:
    if st.button("🟢 Typical Low-Risk Claim", use_container_width=True):
        st.session_state.preset_data = {
            "amount": 85.0,
            "type": "POS Payment",
            "time": 14.0,
            "device": "Mobile",
            "loc": "New York",
            "pm": "Credit Card",
            "prev_fraud": 0,
            "age": 250,
            "txns_24h": 2
        }
        st.rerun()

with preset_cols[1]:
    if st.button("🟠 Moderate Anomaly Claim", use_container_width=True):
        st.session_state.preset_data = {
            "amount": 1950.0,
            "type": "POS Payment",
            "time": 4.0,
            "device": "Mobile",
            "loc": "Chicago",
            "pm": "Credit Card",
            "prev_fraud": 2,
            "age": 150,
            "txns_24h": 7
        }
        st.rerun()

with preset_cols[2]:
    if st.button("🔴 Critical High-Risk Claim", use_container_width=True):
        st.session_state.preset_data = {
            "amount": 4800.0,
            "type": "Online Purchase",
            "time": 3.0,
            "device": "Tablet",
            "loc": "Miami",
            "pm": "UPI",
            "prev_fraud": 3,
            "age": 8,
            "txns_24h": 16
        }
        st.rerun()

# ---------------------------------------------------------
# Transaction Entry Form
# ---------------------------------------------------------
with st.form("claim_form"):
    st.subheader("Transaction Characteristics")

    col1, col2, col3 = st.columns(3)

    with col1:
        transaction_amount = st.number_input(
            "Transaction Amount ($)",
            min_value=1.0,
            max_value=100000.0,
            value=float(st.session_state.preset_data["amount"]),
            step=50.0
        )
        transaction_type = st.selectbox(
            "Transaction Type",
            ["POS Payment", "Online Purchase", "ATM Withdrawal", "Bill Payment", "Bank Transfer"],
            index=["POS Payment", "Online Purchase", "ATM Withdrawal", "Bill Payment", "Bank Transfer"].index(st.session_state.preset_data["type"])
        )
        transaction_time = st.number_input(
            "Hour of Transaction (0 - 23)",
            min_value=0.0,
            max_value=23.0,
            value=float(st.session_state.preset_data["time"]),
            step=1.0,
            help="Execution hour (0 to 23 hours)"
        )

    with col2:
        device_used = st.selectbox(
            "Device Used",
            ["Mobile", "Desktop", "Tablet"],
            index=["Mobile", "Desktop", "Tablet"].index(st.session_state.preset_data["device"])
        )
        location = st.selectbox(
            "Location Metro",
            ["New York", "San Francisco", "Chicago", "Boston", "Houston", "Miami"],
            index=["New York", "San Francisco", "Chicago", "Boston", "Houston", "Miami"].index(st.session_state.preset_data["loc"])
        )
        payment_method = st.selectbox(
            "Payment Method",
            ["Credit Card", "Debit Card", "UPI", "Net Banking"],
            index=["Credit Card", "Debit Card", "UPI", "Net Banking"].index(st.session_state.preset_data["pm"])
        )

    with col3:
        previous_fraud = st.number_input(
            "Previous Fraud Incidents",
            min_value=0,
            max_value=20,
            value=int(st.session_state.preset_data["prev_fraud"]),
            step=1,
            help="Number of historically recorded fraudulent incidents"
        )
        account_age = st.number_input(
            "Account Age (Days)",
            min_value=0,
            max_value=3650,
            value=int(st.session_state.preset_data["age"]),
            step=1,
            help="Days elapsed since customer account opening"
        )
        transactions_24h = st.number_input(
            "Transactions in Last 24 Hours",
            min_value=0,
            max_value=100,
            value=int(st.session_state.preset_data["txns_24h"]),
            step=1,
            help="Number of transactions initiated in the past 24 hours"
        )

    submitted = st.form_submit_button("🛡️ Execute Fraud Risk Assessment", use_container_width=True)

# ---------------------------------------------------------
# Assessment Results Execution
# ---------------------------------------------------------
if submitted:
    claim_dict = {
        "Transaction_Amount": transaction_amount,
        "Transaction_Type": transaction_type,
        "Time_of_Transaction": transaction_time,
        "Device_Used": device_used,
        "Location": location,
        "Payment_Method": payment_method,
        "Previous_Fraudulent_Transactions": previous_fraud,
        "Account_Age": account_age,
        "Number_of_Transactions_Last_24H": transactions_24h
    }

    result = predict_claim(claim_dict)
    prediction = result["prediction"]
    status_label = result["status"]
    risk_tier = result.get("risk_tier", "Low")
    is_fraud = result["is_fraud"]
    prob = result["fraud_probability"]

    st.markdown("---")
    st.subheader("Model Assessment Findings")

    res_col1, res_col2 = st.columns([1, 1.4])

    prob_display = f"{prob * 100:.1f}%" if prob is not None else "N/A"

    with res_col1:
        if risk_tier == "High":
            st.markdown(
                f"""
                <div class="result-fraudulent">
                    <div class="status-title">🔴 {status_label}</div>
                    <div class="prob-value">{prob_display}</div>
                    <p style="margin:0; font-size:1rem;">High Fraud Risk Tier</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif risk_tier == "Moderate":
            st.markdown(
                f"""
                <div class="result-moderate">
                    <div class="status-title">🟠 {status_label}</div>
                    <div class="prob-value">{prob_display}</div>
                    <p style="margin:0; font-size:1rem;">Moderate Fraud Risk Tier</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-legitimate">
                    <div class="status-title">🟢 {status_label}</div>
                    <div class="prob-value">{prob_display}</div>
                    <p style="margin:0; font-size:1rem;">Low Fraud Risk Tier</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    with res_col2:
        if risk_tier == "High":
            st.error(
                "**Assessment Message:** The model flagged this transaction as high risk and potentially fraudulent. "
                "Immediate transaction hold or investigative review is required."
            )
            st.markdown(
                """
                - **Risk Category:** High Risk ($\ge 60\%$)
                - **Action:** Flag for immediate suspension or manual investigation by risk officers.
                """
            )
        elif risk_tier == "Moderate":
            st.warning(
                "**Assessment Message:** The model detected moderate risk anomalies. "
                "Secondary step-up verification (such as OTP / 2FA SMS confirmation) is recommended."
            )
            st.markdown(
                """
                - **Risk Category:** Moderate Risk ($25\% \le \text{Risk} < 60\%$)
                - **Action:** Prompt the cardholder for two-factor authentication before releasing funds.
                """
            )
        else:
            st.success(
                "**Assessment Message:** The model classified this transaction as legitimate with low fraud risk based on the provided information."
            )
            st.markdown(
                """
                - **Risk Category:** Low Risk ($< 25\%$)
                - **Action:** Cleared for frictionless automated approval.
                """
            )

        if prob is not None:
            st.progress(float(min(1.0, max(0.0, prob))))

        st.markdown(
            """
            <div class="disclaimer-text">
                ⚠️ <b>Disclaimer:</b> This model prediction reflects statistical patterns in historical training data 
                and does not constitute an absolute guarantee that a transaction is safe or fraudulent.
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------------------------------------------------
    # Transaction Details & Indicators
    # ---------------------------------------------------------
    st.markdown("### 📋 Transaction Details & Indicators")
    indicators = get_claim_explanation(claim_dict, is_fraud=is_fraud)

    ind_data = []
    for title, cat, desc in indicators:
        ind_data.append({
            "Feature / Indicator": title,
            "Category": cat,
            "Observed Transaction Value": desc
        })

    ind_df = pd.DataFrame(ind_data)
    st.dataframe(ind_df, use_container_width=True)
