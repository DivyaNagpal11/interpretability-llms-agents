"""
Model training module.
Trains an XGBoost classifier on the credit default dataset.
"""

import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import os


def train_model(X_train, y_train, X_test, y_test):
    """
    Train XGBoost model for credit default prediction.

    Returns:
        trained model
    """

    print("\nTraining XGBoost model...")

    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nModel Performance:")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    return model
 

def save_model(model, path="model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def load_model(path="model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)