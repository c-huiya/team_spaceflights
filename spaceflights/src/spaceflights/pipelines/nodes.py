# src/spaceflights/pipelines/nodes.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def preprocess_data(data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    1) Compute a new 'repeat_buyer' flag.
    2) Drop any rows where it’s NaN.
    3) Return the DataFrame (still contains non-numeric columns).
    """
    customer_col = "customer_unique_id"
    order_col = "order_id"
    target_col = parameters["target_column"]

    # Build the flag: 1 if a customer has >1 distinct order, else 0
    data[target_col] = (
        data
        .groupby(customer_col)[order_col]
        .transform("nunique")
        .gt(1)
        .astype(int)
    )

    # Drop any rows where repeat_buyer is missing (unlikely, but safe)
    data = data.dropna(subset=[target_col])
    return data


def prepare_features(data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    1) Drop the target column itself so X doesn't include it.
    2) Keep only numeric columns (int/float).
    """
    target_col = parameters["target_column"]
    X_all = data.drop(columns=[target_col])
    X_numeric = X_all.select_dtypes(include=[np.number])
    return X_numeric


def split_data(data: pd.DataFrame, parameters: dict):
    """
    1) data already has 'repeat_buyer'. Extract y.
    2) Build X via prepare_features (drops non-numeric columns).
    3) Split into train/test.
    Returns: X_train, X_test, y_train, y_test
    """
    target_col = parameters["target_column"]
    train_frac = parameters["train_fraction"]
    rnd_state = parameters["random_state"]

    y = data[target_col]
    X = prepare_features(data, parameters)

    return train_test_split(
        X,
        y,
        test_size=1.0 - train_frac,
        random_state=rnd_state
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict):
    """
    1) Calculate scale_pos_weight = (#neg) / (#pos).
    2) Build XGBClassifier with that weight + your hyperparameters.
    3) Fit on (X_train, y_train).
    4) Return the fitted model.
    """
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    model_params = parameters["model_params"].copy()
    model_params["scale_pos_weight"] = scale_pos_weight

    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, parameters: dict) -> dict:
    """
    1) Get predicted probabilities for class "1".
    2) Use probability_threshold to create binary preds.
    3) Compute accuracy, precision, recall, ROC AUC.
    4) Print and return them.
    """
    probs = model.predict_proba(X_test)[:, 1]

    thresh = parameters.get("probability_threshold", 0.30)
    preds = (probs >= thresh).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    roc = roc_auc_score(y_test, probs)

    print("\n=== Evaluation metrics (threshold = "
          f"{thresh:.2f}) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"ROC AUC  : {roc:.4f}")
    print("==========================\n")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc,
    }
