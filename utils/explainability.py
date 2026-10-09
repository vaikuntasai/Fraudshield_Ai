import numpy as np
import pandas as pd


def get_feature_importance(model_or_bundle):
    """
    Extract feature importances from trained model or model bundle.
    """
    try:
        # If model bundle dictionary is passed
        if isinstance(model_or_bundle, dict):
            if "feature_importance_df" in model_or_bundle:
                return model_or_bundle["feature_importance_df"]
            elif "pipeline" in model_or_bundle:
                model_or_bundle = model_or_bundle["pipeline"]

        preprocessor = model_or_bundle.named_steps.get("preprocessor")
        classifier = model_or_bundle.named_steps.get("classifier")

        if preprocessor is not None and classifier is not None:
            feature_names = preprocessor.get_feature_names_out()
            clean_names = [
                f.replace("numeric__", "").replace("categorical__", "").replace("remainder__", "")
                for f in feature_names
            ]

            if hasattr(classifier, "feature_importances_"):
                importances = classifier.feature_importances_
            elif hasattr(classifier, "coef_"):
                importances = np.abs(classifier.coef_[0])
            else:
                return pd.DataFrame(columns=["Feature", "Importance"])

            result = pd.DataFrame({
                "Feature": clean_names,
                "Importance": importances
            })
            result = result.sort_values("Importance", ascending=False).reset_index(drop=True)
            return result
        return pd.DataFrame(columns=["Feature", "Importance"])
    except Exception:
        return pd.DataFrame(columns=["Feature", "Importance"])


def get_claim_explanation(claim_data, is_fraud=False):
    """
    Provides clear, interpretable context for the transaction characteristics.
    """
    indicators = []

    amount = float(claim_data.get("Transaction_Amount", 0.0) or 0.0)
    previous_fraud = int(claim_data.get("Previous_Fraudulent_Transactions", 0) or 0)
    transactions_24h = int(claim_data.get("Number_of_Transactions_Last_24H", 0) or 0)
    account_age = float(claim_data.get("Account_Age", 0) or 0)
    transaction_time = float(claim_data.get("Time_of_Transaction", 12.0) or 12.0)
    txn_type = str(claim_data.get("Transaction_Type", "Online Purchase"))
    payment_method = str(claim_data.get("Payment_Method", "Credit Card"))

    # Fraud history
    if previous_fraud > 0:
        indicators.append((
            "Previous Fraud History",
            "Elevated Risk",
            f"Account has {previous_fraud} recorded fraudulent incident(s) on file."
        ))
    else:
        indicators.append((
            "Fraud History",
            "Standard",
            "Clean account record with zero prior fraudulent incidents."
        ))

    # Frequency velocity
    if transactions_24h >= 10:
        indicators.append((
            "Transaction Frequency (24H)",
            "Elevated Risk",
            f"{transactions_24h} transactions executed in the past 24 hours (high frequency)."
        ))
    else:
        indicators.append((
            "Transaction Frequency (24H)",
            "Normal",
            f"{transactions_24h} transaction(s) in the past 24 hours (within expected usage)."
        ))

    # Execution time
    if transaction_time < 6 or transaction_time >= 22:
        indicators.append((
            "Execution Window",
            "Off-Hours",
            f"Transaction initiated at {int(transaction_time):02d}:00 hours (night window)."
        ))
    else:
        indicators.append((
            "Execution Window",
            "Standard Hours",
            f"Transaction initiated at {int(transaction_time):02d}:00 hours (standard daytime window)."
        ))

    # Account tenure
    if account_age < 30:
        indicators.append((
            "Account Age",
            "New Account",
            f"Account is {int(account_age)} days old (recently opened)."
        ))
    else:
        indicators.append((
            "Account Age",
            "Established Account",
            f"Account is {int(account_age)} days old (tenured account)."
        ))

    # Amount
    if amount > 3000:
        indicators.append((
            "Transaction Amount",
            "Above Average",
            f"${amount:,.2f} transaction volume."
        ))
    else:
        indicators.append((
            "Transaction Amount",
            "Normal Range",
            f"${amount:,.2f} transaction volume."
        ))

    return indicators
