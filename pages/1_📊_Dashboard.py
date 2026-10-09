import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Dashboard | FraudShield AI",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------
# Styling
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
    .callout-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1.2rem 0;
        color: #cbd5e1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Robust Data Loading
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

candidate_paths = [
    os.path.join(PROJECT_ROOT, "data", "Fraud_Detection_Dataset.csv"),
    os.path.join(os.getcwd(), "data", "Fraud_Detection_Dataset.csv"),
    os.path.join(os.getcwd(), "fraud_detection", "data", "Fraud_Detection_Dataset.csv")
]

DATA_PATH = None
for p in candidate_paths:
    if os.path.exists(p):
        DATA_PATH = p
        break

@st.cache_data
def load_and_prep_data():
    if not DATA_PATH:
        raise FileNotFoundError("Fraud_Detection_Dataset.csv not found.")
    df_raw = pd.read_csv(DATA_PATH)
    
    # Track statistics
    raw_len = len(df_raw)
    dup_len = int(df_raw.drop(columns=["Transaction_ID", "User_ID"], errors="ignore").duplicated().sum())
    
    # Clean duplicates
    df = df_raw.drop_duplicates(subset=[c for c in df_raw.columns if c not in ["Transaction_ID", "User_ID"]]).reset_index(drop=True)
    df["Fraudulent"] = pd.to_numeric(df["Fraudulent"], errors="coerce").fillna(0).astype(int)
    return df, raw_len, dup_len

try:
    df, raw_count, duplicate_count = load_and_prep_data()
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("📊 Fraud Detection Dashboard")
st.caption("Comprehensive exploratory data analysis and transaction risk distribution overview.")

# ---------------------------------------------------------
# Executive KPI Cards
# ---------------------------------------------------------
total_transactions = len(df)
fraudulent = int(df["Fraudulent"].sum())
legitimate = total_transactions - fraudulent
fraud_rate = (fraudulent / total_transactions * 100) if total_transactions > 0 else 0
avg_amount = df["Transaction_Amount"].dropna().mean()
imbalance_ratio = legitimate / (fraudulent + 1e-9)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total Records", f"{total_transactions:,}", help=f"Cleaned from {raw_count:,} raw records")
with c2:
    st.metric("Duplicates Removed", f"{duplicate_count:,}", delta=f"-{duplicate_count:,} rows", delta_color="normal")
with c3:
    st.metric("Fraudulent Claims", f"{fraudulent:,}", f"{fraud_rate:.2f}% rate")
with c4:
    st.metric("Legitimate Claims", f"{legitimate:,}", f"{100 - fraud_rate:.2f}%")
with c5:
    st.metric("Class Imbalance", f"~{imbalance_ratio:.1f}:1", "Negative:Positive")

# ---------------------------------------------------------
# Methodological Context Alert
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="callout-box">
        <b>💡 Machine Learning & Academic Insight:</b> Fraudulent transactions comprise only 
        <b>{fraud_rate:.2f}%</b> of the dataset (~1 fraud for every {int(imbalance_ratio)} legitimate transactions). 
        In severe class imbalance scenarios, standard accuracy is notoriously misleading: a naive dummy classifier 
        predicting "all legitimate" would achieve <b>{100 - fraud_rate:.2f}% accuracy</b> while completely failing to stop any fraud. 
        Hence, our evaluation strictly focuses on <b>Fraud Recall</b>, <b>Precision</b>, <b>F1-Score</b>, and <b>PR-AUC</b>.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------------------
# Visualizations Row 1: Distribution & Amount Analysis
# ---------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    donut_df = pd.DataFrame({
        "Class": ["Legitimate (0)", "Fraudulent (1)"],
        "Count": [legitimate, fraudulent]
    })
    fig_donut = px.pie(
        donut_df,
        names="Class",
        values="Count",
        hole=0.6,
        title="Class Imbalance: Legitimate vs Fraudulent",
        color="Class",
        color_discrete_map={
            "Legitimate (0)": "#10b981",
            "Fraudulent (1)": "#ef4444"
        }
    )
    fig_donut.update_layout(template="plotly_dark", height=380, legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig_donut, use_container_width=True)

with col2:
    # Transaction Amount Box Plot & Histogram
    clean_amt = df.dropna(subset=["Transaction_Amount"])
    fig_box = px.box(
        clean_amt,
        x="Fraudulent",
        y="Transaction_Amount",
        color="Fraudulent",
        title="Transaction Amount Distribution by Class",
        color_discrete_map={0: "#38bdf8", 1: "#ef4444"},
        labels={"Fraudulent": "Target Class", "Transaction_Amount": "Amount ($)"}
    )
    fig_box.update_layout(
        template="plotly_dark",
        height=380,
        xaxis=dict(tickvals=[0, 1], ticktext=["Legitimate", "Fraudulent"]),
        showlegend=False
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ---------------------------------------------------------
# Visualizations Row 2: Channel & Categorical Breakdown
# ---------------------------------------------------------
st.subheader("Categorical Channel & Behavior Breakdown")
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    if "Transaction_Type" in df.columns:
        type_agg = df.groupby("Transaction_Type").agg(
            Total=("Fraudulent", "count"),
            Fraud=("Fraudulent", "sum")
        ).reset_index()
        type_agg["Fraud_Rate"] = (type_agg["Fraud"] / type_agg["Total"]) * 100
        type_agg = type_agg.sort_values("Fraud_Rate", ascending=False)

        fig_type = px.bar(
            type_agg,
            x="Transaction_Type",
            y="Fraud_Rate",
            text=type_agg["Fraud_Rate"].apply(lambda x: f"{x:.2f}%"),
            title="Fraud Rate (%) by Transaction Type",
            color="Fraud_Rate",
            color_continuous_scale="Reds"
        )
        fig_type.update_layout(template="plotly_dark", height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_type, use_container_width=True)

with row2_col2:
    if "Payment_Method" in df.columns:
        pm_agg = df.groupby("Payment_Method").agg(
            Total=("Fraudulent", "count"),
            Fraud=("Fraudulent", "sum")
        ).reset_index()
        pm_agg["Fraud_Rate"] = (pm_agg["Fraud"] / pm_agg["Total"]) * 100
        pm_agg = pm_agg.sort_values("Fraud_Rate", ascending=False)

        fig_pm = px.bar(
            pm_agg,
            x="Payment_Method",
            y="Fraud_Rate",
            text=pm_agg["Fraud_Rate"].apply(lambda x: f"{x:.2f}%"),
            title="Fraud Rate (%) by Payment Method",
            color="Fraud_Rate",
            color_continuous_scale="Oranges"
        )
        fig_pm.update_layout(template="plotly_dark", height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_pm, use_container_width=True)

# ---------------------------------------------------------
# Visualizations Row 3: Temporal and Geo Analysis
# ---------------------------------------------------------
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    if "Time_of_Transaction" in df.columns:
        time_clean = df.dropna(subset=["Time_of_Transaction"]).copy()
        time_clean["Hour"] = pd.to_numeric(time_clean["Time_of_Transaction"], errors="coerce").fillna(0).astype(int)
        hourly_agg = time_clean.groupby("Hour").agg(
            Volume=("Fraudulent", "count"),
            Fraud_Count=("Fraudulent", "sum")
        ).reset_index()
        hourly_agg["Fraud_Rate"] = (hourly_agg["Fraud_Count"] / hourly_agg["Volume"]) * 100

        fig_time = px.line(
            hourly_agg,
            x="Hour",
            y="Fraud_Rate",
            markers=True,
            title="24-Hour Fraud Rate Dynamics",
            labels={"Hour": "Hour of Day (0-23)", "Fraud_Rate": "Fraud Rate (%)"}
        )
        fig_time.update_traces(line_color="#f59e0b", line_width=3)
        fig_time.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_time, use_container_width=True)

with row3_col2:
    if "Location" in df.columns:
        loc_agg = df.groupby("Location").agg(
            Volume=("Fraudulent", "count"),
            Fraud_Count=("Fraudulent", "sum")
        ).reset_index()
        loc_agg["Fraud_Rate"] = (loc_agg["Fraud_Count"] / loc_agg["Volume"]) * 100
        loc_agg = loc_agg.sort_values("Fraud_Rate", ascending=False)

        fig_loc = px.bar(
            loc_agg,
            x="Location",
            y="Fraud_Rate",
            title="Geographic Fraud Incidence by Metro Location",
            color="Fraud_Rate",
            color_continuous_scale="Purples",
            text=loc_agg["Fraud_Rate"].apply(lambda x: f"{x:.2f}%")
        )
        fig_loc.update_layout(template="plotly_dark", height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_loc, use_container_width=True)

# ---------------------------------------------------------
# Data Quality & Preprocessing Health Table
# ---------------------------------------------------------
with st.expander("🛠️ View Dataset Preprocessing & Missing Values Audit", expanded=False):
    missing_c = df.isnull().sum()
    missing_pct = (missing_c / len(df)) * 100
    audit_df = pd.DataFrame({
        "Feature": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Count": missing_c.values,
        "Missing %": [f"{p:.2f}%" for p in missing_pct.values],
        "Imputation Method": [
            "Median Imputation" if c in ["Transaction_Amount", "Time_of_Transaction"]
            else "Mode (Most Frequent)" if c in ["Device_Used", "Location", "Payment_Method"]
            else "Drop Identifier" if c in ["Transaction_ID", "User_ID"]
            else "Preserve Target" if c == "Fraudulent"
            else "None (Complete)" for c in df.columns
        ]
    })
    st.dataframe(audit_df, use_container_width=True)
