import os
import streamlit as st
import joblib

st.set_page_config(
    page_title="FraudShield AI | Project Intelligence & Risk Advisor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0b1120;
    }
    .hero {
        padding: 2.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #064e3b 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.1rem;
        max-width: 900px;
        line-height: 1.6;
    }
    .card {
        background: #111827;
        padding: 1.4rem;
        border-radius: 16px;
        border: 1px solid #1f2937;
        height: 100%;
    }
    .card-title {
        color: #38bdf8;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    .card-value {
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Check Artifact Health
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(CURRENT_DIR, "model", "fraud_model.pkl")
model_loaded = os.path.exists(model_path)
model_name = "XGBoost Classifier (Tuned)" if model_loaded else "Awaiting Training"

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ FraudShield AI")
    st.caption("End-to-End Enterprise Fraud Detection System")
    st.divider()

    st.markdown(
        """
        ### 📌 Navigation
        Use the sidebar navigation tabs to access:
        - **📊 Dashboard**: Dataset metrics, EDA & distribution charts
        - **🔍 Claim Screening**: Real-time transaction scoring & explainability
        - **📈 Analytics**: Multi-model comparison, threshold tuning & confusion matrix
        """
    )
    st.divider()
    st.caption(f"Status: **{'🟢 Production Model Active' if model_loaded else '🟡 Training Required'}**")

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">🛡️ FraudShield AI</div>
        <div class="hero-subtitle">
            An academic-grade, end-to-end Machine Learning intelligence system for fraud detection and risk analysis. 
            Engineered to handle severe class imbalance, multi-algorithm evaluation, dynamic decision threshold optimization, 
            and transparent risk attribution.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Core Capabilities Cards
# ---------------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">01 • EXPLORATORY ANALYSIS</div>
            <div class="card-value">Dataset Intelligence</div>
            <p style="color:#94a3b8; font-size:0.95rem;">
                Audit 50,000+ real transactions, analyze duplicate records, inspect missing values, and evaluate class imbalance distributions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">02 • REAL-TIME INFERENCE</div>
            <div class="card-value">Claim Screening</div>
            <p style="color:#94a3b8; font-size:0.95rem;">
                Score individual claims in real-time. Calculate fraud probabilities, confidence bounds, and actionable policy recommendations.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">03 • ML RIGOR & OPTIMIZATION</div>
            <div class="card-value">Analytics & Tuning</div>
            <p style="color:#94a3b8; font-size:0.95rem;">
                Benchmark 7 ML algorithms, analyze PR-AUC / ROC curves, and interactively adjust the decision threshold to minimize False Negatives.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ---------------------------------------------------------
# Architectural Pipeline Flow
# ---------------------------------------------------------
st.subheader("Architectural Workflow")

cols = st.columns(4)
steps = [
    ("Stage 1", "Data Ingestion & Integrity", "Median & Mode imputation, 881 duplicate rows filtered, ID columns decoupled."),
    ("Stage 2", "Feature Engineering", "Velocity ratio, off-hours night indicator, account age ratios, log amount scaling."),
    ("Stage 3", "Imbalance & Model Suite", "Balanced weighting, SMOTE, and benchmarking across Random Forest, XGBoost, GB, KNN, etc."),
    ("Stage 4", "Threshold Optimization", "Cost-sensitive decision boundary tuning to maximize fraud recall and minimize financial write-offs.")
]

for col, (stage, title, desc) in zip(cols, steps):
    with col:
        st.markdown(
            f"""
            <div class="card">
                <div style="color:#38bdf8; font-size:0.8rem; font-weight:700;">{stage}</div>
                <h4 style="color:white; margin:0.4rem 0;">{title}</h4>
                <p style="color:#94a3b8; font-size:0.9rem;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.info("💡 **Getting Started:** Select **Claim Screening** in the sidebar to evaluate a transaction, or open **Analytics** to view the full model comparison matrix.")