import os
import joblib
import pandas as pd
import numpy as np

from utils.preprocessing import prepare_input_dataframe

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "model",
    "fraud_model.pkl"
)


def load_model_bundle():
    """Load model pipeline and bundle metadata safely."""
    candidate_paths = [
        MODEL_PATH,
        os.path.join(os.getcwd(), "model", "fraud_model.pkl"),
        os.path.join(os.getcwd(), "fraud_detection", "model", "fraud_model.pkl")
    ]

    actual_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            actual_path = p
            break

    if not actual_path:
        raise FileNotFoundError(
            "Trained model not found at candidate paths. Run: python train_model.py"
        )

    artifact = joblib.load(actual_path)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        pipeline = artifact["pipeline"]
        return pipeline, artifact
    else:
        return artifact, {}


def load_model():
    """Returns the scikit-learn pipeline for prediction."""
    pipeline, _ = load_model_bundle()
    return pipeline


def predict_claim(claim_data):
    """
    Predict classification for a transaction using the trained model:
    - Low Risk (< 25% probability): 'LEGITIMATE TRANSACTION'
    - Moderate Risk (25% to 60% probability): 'MODERATE RISK TRANSACTION'
    - High Risk (>= 60% probability): 'FRAUDULENT TRANSACTION'
    """
    pipeline, _ = load_model_bundle()

    df = prepare_input_dataframe(claim_data)

    raw_prediction = pipeline.predict(df)[0]
    prediction = int(raw_prediction)

    fraud_probability = None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(df)[0]
        if len(probs) > 1:
            fraud_probability = float(probs[1])
        else:
            fraud_probability = float(probs[0])

    # Assign risk tier based on continuous model probability
    if fraud_probability is not None:
        if fraud_probability >= 0.60:
            risk_tier = "High"
            status = "FRAUDULENT TRANSACTION"
            is_fraud = True
        elif fraud_probability >= 0.25:
            risk_tier = "Moderate"
            status = "MODERATE RISK TRANSACTION"
            is_fraud = bool(prediction == 1)
        else:
            risk_tier = "Low"
            status = "LEGITIMATE TRANSACTION"
            is_fraud = False
    else:
        risk_tier = "High" if prediction == 1 else "Low"
        status = "FRAUDULENT TRANSACTION" if prediction == 1 else "LEGITIMATE TRANSACTION"
        is_fraud = bool(prediction == 1)

    return {
        "prediction": prediction,
        "status": status,
        "risk_tier": risk_tier,
        "is_fraud": is_fraud,
        "fraud_probability": fraud_probability
    }
