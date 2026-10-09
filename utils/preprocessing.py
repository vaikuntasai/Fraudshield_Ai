import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Fraudulent"

DROP_COLUMNS = [
    "Transaction_ID",
    "User_ID"
]

NUMERIC_FEATURES = [
    "Transaction_Amount",
    "Log_Transaction_Amount",
    "Time_of_Transaction",
    "Previous_Fraudulent_Transactions",
    "Account_Age",
    "Number_of_Transactions_Last_24H",
    "Transaction_Hour",
    "Is_Night_Transaction",
    "Amount_Per_Account_Day",
    "Fraud_History_Flag",
    "High_Transaction_Frequency",
    "Txn_Velocity_Ratio"
]

CATEGORICAL_FEATURES = [
    "Transaction_Type",
    "Device_Used",
    "Location",
    "Payment_Method"
]


def load_data(file_path):
    """Load the fraud detection dataset safely."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    df = pd.read_csv(file_path)
    return df


def handle_duplicates(df, drop=True):
    """
    Detect and handle duplicate records.
    Returns: cleaned_df, duplicate_count, before_count, after_count
    """
    before_count = len(df)
    # Check duplicates ignoring identifier if present
    cols_to_check = [c for c in df.columns if c not in DROP_COLUMNS]
    duplicate_count = int(df.duplicated(subset=cols_to_check).sum())
    
    if drop and duplicate_count > 0:
        cleaned_df = df.drop_duplicates(subset=cols_to_check).reset_index(drop=True)
    else:
        cleaned_df = df.copy()
        
    after_count = len(cleaned_df)
    return cleaned_df, duplicate_count, before_count, after_count


def analyze_missing_values(df):
    """
    Analyze missing values across all features.
    Returns DataFrame with Count, Percentage, and Imputation Strategy.
    """
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    
    strategies = []
    for col in df.columns:
        if col in NUMERIC_FEATURES or col == "Transaction_Amount" or col == "Time_of_Transaction":
            strategies.append("Median Imputation")
        elif col in CATEGORICAL_FEATURES:
            strategies.append("Mode (Most Frequent) Imputation")
        elif col in DROP_COLUMNS:
            strategies.append("Dropped Identifier")
        else:
            strategies.append("Standard Treatment")
            
    summary_df = pd.DataFrame({
        "Feature": df.columns,
        "Missing_Count": missing_counts.values,
        "Missing_Percentage": missing_pct.values,
        "Imputation_Strategy": strategies
    })
    return summary_df


def create_features(df):
    """
    Perform expert feature engineering:
    - Time parsing and hour extraction
    - Night transaction flag (high-risk window: 22:00 to 06:00)
    - Amount per account day ratio (velocity of expenditure)
    - Skewness correction: Log transformation of monetary amount
    - Fraud history boolean flag
    - High transaction velocity flags
    """
    df = df.copy()

    # Convert transaction time safely
    time_numeric = pd.to_numeric(df.get("Time_of_Transaction"), errors="coerce")
    df["Time_of_Transaction"] = time_numeric
    df["Transaction_Hour"] = time_numeric.fillna(12.0)

    # Night transaction indicator (typically higher anomaly rate)
    df["Is_Night_Transaction"] = (
        (df["Transaction_Hour"] < 6) | (df["Transaction_Hour"] >= 22)
    ).astype(int)

    # Skewness correction for transaction amount
    amt = pd.to_numeric(df.get("Transaction_Amount"), errors="coerce").fillna(0.0)
    amt_clean = np.maximum(0, amt)
    df["Transaction_Amount"] = amt
    df["Log_Transaction_Amount"] = np.log1p(amt_clean)

    # Account age safety
    acc_age = pd.to_numeric(df.get("Account_Age"), errors="coerce").fillna(0.0)
    df["Account_Age"] = acc_age

    # Amount per account age (new accounts spending large amounts are high risk)
    df["Amount_Per_Account_Day"] = amt_clean / (acc_age + 1.0)

    # Previous fraud history indicator
    prev_fraud = pd.to_numeric(df.get("Previous_Fraudulent_Transactions"), errors="coerce").fillna(0)
    df["Previous_Fraudulent_Transactions"] = prev_fraud
    df["Fraud_History_Flag"] = (prev_fraud > 0).astype(int)

    # High transaction frequency in 24 hours
    txns_24h = pd.to_numeric(df.get("Number_of_Transactions_Last_24H"), errors="coerce").fillna(0)
    df["Number_of_Transactions_Last_24H"] = txns_24h
    df["High_Transaction_Frequency"] = (txns_24h >= 10).astype(int)

    # Velocity ratio: transactions last 24h normalized by account age in months
    account_months = (acc_age / 30.0) + 1.0
    df["Txn_Velocity_Ratio"] = txns_24h / account_months

    return df


def get_feature_data(df, drop_duplicates=True):
    """
    Prepare feature matrix X and target vector y.
    Handles duplicate records and creates engineered features.
    """
    if drop_duplicates:
        df, _, _, _ = handle_duplicates(df, drop=True)

    df = create_features(df)

    # Drop identifiers and target
    X = df.drop(
        columns=DROP_COLUMNS + [TARGET_COLUMN],
        errors="ignore"
    )

    y = None
    if TARGET_COLUMN in df.columns:
        y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)

    return X, y


def build_preprocessor():
    """
    Build scikit-learn ColumnTransformer:
    - Numerical: Median Imputer + StandardScaler
    - Categorical: Most Frequent Imputer + OneHotEncoder
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


def prepare_input_dataframe(data):
    """
    Convert a single claim dictionary or dataframe
    into the format expected by the model pipeline.
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    df = create_features(df)
    return df
