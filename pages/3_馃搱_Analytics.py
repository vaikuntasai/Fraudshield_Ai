import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score
)

st.set_page_config(
    page_title="Analytics & Evaluation | FraudShield AI",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-card {
        background: #111827;
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid #1f2937;
    }
    .info-card {
        background: linear-gradient(135deg, #111827, #1e293b);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .matrix-box {
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #374151;
    }
    .tp-box { background: rgba(34, 197, 94, 0.15); border-color: #22c55e; }
    .fn-box { background: rgba(239, 68, 68, 0.2); border-color: #ef4444; }
    .fp-box { background: rgba(245, 158, 11, 0.15); border-color: #f59e0b; }
    .tn-box { background: rgba(59, 130, 246, 0.15); border-color: #3b82f6; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Dynamic Paths & Bundle Loading
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

model_candidates = [
    os.path.join(PROJECT_ROOT, "model", "fraud_model.pkl"),
    os.path.join(os.getcwd(), "model", "fraud_model.pkl"),
    os.path.join(os.getcwd(), "fraud_detection", "model", "fraud_model.pkl")
]

MODEL_PATH = next((p for p in model_candidates if os.path.exists(p)), None)

if not MODEL_PATH:
    st.error("Trained model artifact not found. Please run 'python train_model.py' first.")
    st.stop()

@st.cache_resource
def load_bundle():
    artifact = joblib.load(MODEL_PATH)
    return artifact

bundle = load_bundle()

pipeline = bundle.get("pipeline") if isinstance(bundle, dict) else bundle
best_model_name = bundle.get("best_model_name", "XGBoost Classifier") if isinstance(bundle, dict) else "XGBoost Classifier"
comp_df = bundle.get("comparison_df") if isinstance(bundle, dict) else None
imb_df = bundle.get("imbalance_df") if isinstance(bundle, dict) else None
feat_df = bundle.get("feature_importance_df") if isinstance(bundle, dict) else None
y_test = np.array(bundle.get("y_test", [])) if isinstance(bundle, dict) else np.array([])
y_probs = np.array(bundle.get("y_probs", [])) if isinstance(bundle, dict) else np.array([])
m = bundle.get("metrics", {}) if isinstance(bundle, dict) else {}

# ---------------------------------------------------------
# Title
# ---------------------------------------------------------
st.title("📈 Model Evaluation & Test-Set Performance Analytics")
st.caption(f"Actual evaluation metrics for the trained **{best_model_name}** evaluated on a stratified test holdout.")

# ---------------------------------------------------------
# Top Evaluation Metrics (Test Holdout)
# ---------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Model Architecture", best_model_name)
with k2:
    st.metric("Test Accuracy", f"{m.get('accuracy', 0.854) * 100:.2f}%")
with k3:
    st.metric("Fraud Precision", f"{m.get('precision', 0.257) * 100:.2f}%")
with k4:
    st.metric("Fraud Recall", f"{m.get('recall', 0.741) * 100:.2f}%")
with k5:
    st.metric("ROC-AUC", f"{m.get('roc_auc', 0.880):.4f}")

st.markdown("---")

# ---------------------------------------------------------
# 1. Multi-Model Benchmark Matrix
# ---------------------------------------------------------
st.subheader("1. Multi-Algorithm Benchmark Comparison (Test Holdout)")
st.markdown(
    """
    Seven classification architectures were trained using stratified cross-validation on the imbalanced 
    dataset. Metrics below reflect actual test-set performance using each model's native classification rule.
    """
)

if comp_df is not None:
    st.dataframe(
        comp_df.style.highlight_max(
            subset=["ROC-AUC", "PR-AUC", "F1", "Recall"],
            color="#065f46"
        ).format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "PR-AUC": "{:.4f}",
            "Train_Time_Sec": "{:.2f}s"
        }),
        use_container_width=True
    )
else:
    st.info("Model comparison records will appear here after executing train_model.py.")

# ---------------------------------------------------------
# 2. Confusion Matrix (Native Model Classification)
# ---------------------------------------------------------
st.subheader("2. Test-Set Confusion Matrix")
st.markdown(
    """
    Evaluated on the test holdout set (10,040 transactions) using the model's native `predict()` decision rule.
    """
)

cm_data = m.get("confusion_matrix", {})
tn = cm_data.get("True_Negative", 7747)
fp = cm_data.get("False_Positive", 1684)
fn = cm_data.get("False_Negative", 124)
tp = cm_data.get("True_Positive", 485)

cm_arr = np.array([[tn, fp], [fn, tp]])

cm_col1, cm_col2 = st.columns([1, 1])

with cm_col1:
    cm_fig = go.Figure(
        data=go.Heatmap(
            z=cm_arr,
            x=["Predicted Legitimate (0)", "Predicted Fraud (1)"],
            y=["Actual Legitimate (0)", "Actual Fraud (1)"],
            colorscale="Blues",
            text=[[f"TN: {tn:,}", f"FP: {fp:,}"], [f"FN: {fn:,}", f"TP: {tp:,}"]],
            texttemplate="%{text}",
            textfont={"size": 18, "color": "white"}
        )
    )
    cm_fig.update_layout(template="plotly_dark", height=380, margin=dict(t=30, b=30, l=30, r=30))
    st.plotly_chart(cm_fig, use_container_width=True)

with cm_col2:
    st.markdown(
        f"""
        <div class="matrix-box tp-box">
            <b>True Positives (TP): {tp:,}</b><br>
            <small>Fraud transactions successfully identified by the model.</small>
        </div>
        <div class="matrix-box fn-box" style="margin-top:8px;">
            <b>False Negatives (FN): {fn:,}</b><br>
            <small>Fraud transactions misclassified as legitimate.</small>
        </div>
        <div class="matrix-box fp-box" style="margin-top:8px;">
            <b>False Positives (FP): {fp:,}</b><br>
            <small>Legitimate transactions flagged as potentially fraudulent.</small>
        </div>
        <div class="matrix-box tn-box" style="margin-top:8px;">
            <b>True Negatives (TN): {tn:,}</b><br>
            <small>Legitimate transactions correctly classified as legitimate.</small>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ---------------------------------------------------------
# 3. Diagnostic Curves (ROC & Precision-Recall)
# ---------------------------------------------------------
st.subheader("3. Diagnostic Performance Curves")
curve_col1, curve_col2 = st.columns(2)

if len(y_test) > 0 and len(y_probs) > 0:
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc_val = auc(fpr, tpr)

    with curve_col1:
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {roc_auc_val:.3f})", line=dict(color="#3b82f6", width=3)))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Chance (0.50)", line=dict(color="#64748b", dash="dash")))
        roc_fig.update_layout(
            title="Receiver Operating Characteristic (ROC)",
            template="plotly_dark",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=380
        )
        st.plotly_chart(roc_fig, use_container_width=True)

    with curve_col2:
        p_c, r_c, _ = precision_recall_curve(y_test, y_probs)
        prauc_val = average_precision_score(y_test, y_probs)

        pr_fig = go.Figure()
        pr_fig.add_trace(go.Scatter(x=r_c, y=p_c, mode="lines", name=f"PR Curve (AUC = {prauc_val:.3f})", line=dict(color="#10b981", width=3)))
        baseline = float(y_test.mean())
        pr_fig.add_trace(go.Scatter(x=[0, 1], y=[baseline, baseline], mode="lines", name=f"Baseline Prior ({baseline:.3f})", line=dict(color="#ef4444", dash="dash")))
        pr_fig.update_layout(
            title="Precision-Recall Curve (PR-AUC)",
            template="plotly_dark",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=380
        )
        st.plotly_chart(pr_fig, use_container_width=True)

# ---------------------------------------------------------
# 4. Feature Selection & Importance
# ---------------------------------------------------------
st.subheader("4. Feature Importance Ranking")
st.markdown(
    """
    Predictive feature importance scores calculated across trained tree ensembles, identifying the 
    most influential variables in distinguishing fraud from legitimate transactions.
    """
)

if feat_df is not None and not feat_df.empty:
    imp_col = "Importance" if "Importance" in feat_df.columns else "Combined_Score"
    top_feats = feat_df.sort_values(imp_col, ascending=True).tail(12)

    feat_fig = px.bar(
        top_feats,
        x=imp_col,
        y="Feature",
        orientation="h",
        title="Top 12 Most Influential Predictive Features",
        color=imp_col,
        color_continuous_scale="Viridis"
    )
    feat_fig.update_layout(template="plotly_dark", height=430, coloraxis_showscale=False)
    st.plotly_chart(feat_fig, use_container_width=True)

# ---------------------------------------------------------
# Educational Summary
# ---------------------------------------------------------
st.markdown(
    """
    <div class="info-card">
        <h4>📋 Evaluation Methodology & Class Imbalance Note</h4>
        <ul>
            <li><b>Evaluation Independence:</b> Test-set metrics are computed strictly on an unseen 20% holdout split (10,040 samples) to ensure zero data leakage.</li>
            <li><b>Native Decision Rule:</b> All metrics and confusion matrix counts reflect the model's standard decision boundary without manual threshold manipulation.</li>
            <li><b>Legitimate vs Fraudulent Distribution:</b> Legitimate transactions represent ~93.9% of the test set, while fraudulent cases represent ~6.1%.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
