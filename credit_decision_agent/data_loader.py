"""
Data loading and preprocessing module.
Downloads the UCI Credit Card Default dataset and prepares train/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import os
import pickle


FEATURE_NAMES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERICAL_FEATURES = [f for f in FEATURE_NAMES if f not in CATEGORICAL_FEATURES]


def load_and_prepare_data(test_size=0.2, random_state=42):
    """
    Fetch the UCI Credit Card Default dataset, preprocess, and split.

    Returns:
        dict with X_train, X_test, y_train, y_test, scaler, feature_names, raw_test_df
    """

    print("Fetching UCI Credit Card Default dataset...")
    dataset = fetch_ucirepo(id=350)

    X = dataset.data.features
    y = dataset.data.targets.values.ravel()

    # Rename columns to standard names if needed
    X.columns = FEATURE_NAMES

    # Store raw data before scaling for interpretability
    raw_df = X.copy()
    raw_df["default"] = y

    # Split first to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[NUMERICAL_FEATURES] = scaler.fit_transform(X_train[NUMERICAL_FEATURES])
    X_test_scaled[NUMERICAL_FEATURES] = scaler.transform(X_test[NUMERICAL_FEATURES])

    # Keep raw test data for explanations
    raw_test_df = X_test.copy()
    raw_test_df["default"] = y_test

    print(f"Dataset loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    print(f"Default rate: {y.mean():.2%}")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "raw_test_df": raw_test_df,
        "raw_df": raw_df
    }


def get_customer_profile(raw_record: pd.Series) -> dict:
    """
    Convert a raw record into a human-readable customer profile.
    """
    sex_map = {1: "Male", 2: "Female"}
    education_map = {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}
    marriage_map = {1: "Married", 2: "Single", 3: "Others"}

    # Payment status interpretation
    pay_status_map = {
        -2: "No consumption", -1: "Paid in full", 0: "Revolving credit used",
        1: "1 month delay", 2: "2 months delay", 3: "3 months delay",
        4: "4 months delay", 5: "5 months delay", 6: "6 months delay",
        7: "7 months delay", 8: "8 months delay", 9: "9+ months delay"
    }

    profile = {
        "credit_limit": float(raw_record["LIMIT_BAL"]),
        "sex": sex_map.get(int(raw_record["SEX"]), "Unknown"),
        "education": education_map.get(int(raw_record["EDUCATION"]), "Others"),
        "marriage": marriage_map.get(int(raw_record["MARRIAGE"]), "Others"),
        "age": int(raw_record["AGE"]),
        "recent_payment_status": pay_status_map.get(int(raw_record["PAY_0"]), "Unknown"),
        "payment_history": {
            "month_1": pay_status_map.get(int(raw_record["PAY_0"]), "Unknown"),
            "month_2": pay_status_map.get(int(raw_record["PAY_2"]), "Unknown"),
            "month_3": pay_status_map.get(int(raw_record["PAY_3"]), "Unknown"),
        },

        "latest_bill_amount": float(raw_record["BILL_AMT1"]),
        "latest_payment_amount": float(raw_record["PAY_AMT1"]),
        "credit_utilization": float(raw_record["BILL_AMT1"]) / max(float(raw_record["LIMIT_BAL"]), 1),
        "total_bill_amounts": sum(float(raw_record[f"BILL_AMT{i}"]) for i in range(1, 7)),
        "total_payment_amounts": sum(float(raw_record[f"PAY_AMT{i}"]) for i in range(1, 7)),
        "missed_payments_count": sum(1 for i in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
                                      if int(raw_record[i]) > 0),
    }

    return profile