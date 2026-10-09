import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier


def run_feature_selection_analysis(X_preprocessed, y, feature_names, top_k=15):
    """
    Perform multi-method feature selection analysis:
    1. Tree-based Gini importance (Extra Trees)
    2. Mutual Information classification score
    3. Combined relative ranking
    """
    # 1. Tree-based importance
    et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    et.fit(X_preprocessed, y)
    tree_importances = et.feature_importances_

    # 2. Mutual Information score (subsample if large for speed)
    sample_size = min(10000, len(y))
    np.random.seed(42)
    sample_idx = np.random.choice(len(y), size=sample_size, replace=False)
    X_sub = X_preprocessed[sample_idx]
    y_sub = y.iloc[sample_idx] if hasattr(y, 'iloc') else y[sample_idx]

    mi_scores = mutual_info_classif(X_sub, y_sub, random_state=42)

    # 3. Create DataFrame
    results_df = pd.DataFrame({
        "Feature": feature_names,
        "Tree_Importance": tree_importances,
        "Mutual_Info": mi_scores
    })

    # Normalized scores
    results_df["Norm_Tree"] = results_df["Tree_Importance"] / (results_df["Tree_Importance"].max() + 1e-9)
    results_df["Norm_MI"] = results_df["Mutual_Info"] / (results_df["Mutual_Info"].max() + 1e-9)
    results_df["Combined_Score"] = (0.6 * results_df["Norm_Tree"]) + (0.4 * results_df["Norm_MI"])

    results_df = results_df.sort_values("Combined_Score", ascending=False).reset_index(drop=True)
    results_df["Rank"] = results_df.index + 1

    top_features = results_df.head(top_k)["Feature"].tolist()

    return results_df, top_features
