import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def preprocess_data(data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    1) Compute the 'repeat_buyer' flag in memory:
         - Group by customer_unique_id, count unique order_id per customer
         - If that count > 1 → 1 (repeat), else 0
       (Your CSV does NOT already contain a 'repeat_buyer' column, so we must add it.)

    2) Drop any rows where repeat_buyer is NaN (shouldn't happen, but just in case).
       Returns the DataFrame with a new column named parameters['target_column'].
    """
    customer_col = "customer_unique_id"
    order_col = "order_id"
    target_col = parameters["target_column"]  # e.g. "repeat_buyer"

    # (a) For each row, count how many unique orders that customer has. Then compare to >1.
    data[target_col] = (
        data
        .groupby(customer_col)[order_col]
        .transform("nunique")
        .gt(1)
        .astype(int)
    )

    # (b) Drop any rows where repeat_buyer is missing (unlikely, but safe).
    data = data.dropna(subset=[target_col])

    return data


def prepare_features(data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    We have a DataFrame that now includes the newly created 'repeat_buyer' column.
    Before we hand anything to XGBoost, we must drop all non-numeric columns (ID strings, timestamps, city/state strings, etc.).
    - First drop the target column (so X will not accidentally contain the label).
    - Then select only numeric dtypes (int, float).
    """
    target_col = parameters["target_column"]  # "repeat_buyer"
    # 1) Drop the target itself from X:
    X_all = data.drop(columns=[target_col])

    # 2) Keep only numeric-typed columns (int, float, bool). Everything else (object, datetime) is dropped.
    X_numeric = X_all.select_dtypes(include=[np.number])

    return X_numeric


def split_data(data: pd.DataFrame, parameters: dict):
    """
    1) Accept the fully-cleaned DataFrame (with 'repeat_buyer' already computed).
    2) Extract y = data[target_column].
    3) Extract X = only numeric features via prepare_features().
    4) Perform train_test_split using train_fraction & random_state from parameters.
       Returns: X_train, X_test, y_train, y_test
    """
    target_col = parameters["target_column"]
    train_frac = parameters["train_fraction"]
    rnd_state = parameters["random_state"]

    # (1) y is the newly created flag:
    y = data[target_col]

    # (2) X is only numeric features (all object columns removed):
    X = prepare_features(data, parameters)

    # (3) Rely on sklearn for reproducible splitting:
    return train_test_split(
        X,
        y,
        test_size=1.0 - train_frac,
        random_state=rnd_state
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict):
    """
    1) Build an XGBoostClassifier with the optimized hyperparameters from parameters["model_params"].
    2) Fit on (X_train, y_train). Since X_train is purely numeric, XGBoost will not complain.
    3) Return the fitted model.
    """
    model = xgb.XGBClassifier(**parameters["model_params"])
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    1) model.predict(X_test) → predicted labels
    2) model.predict_proba(X_test)[:,1] → predicted probability for the "1" (repeat buyer) class
    3) Compute accuracy, precision, recall, ROC AUC based on those outputs
    4) Return a dict: {"accuracy":…, "precision":…, "recall":…, "roc_auc":…}
    """
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }
