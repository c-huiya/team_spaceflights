import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score



def preprocessing(data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Prepares customer-level features with repeat_buyer label.
    """

    # Dropping customers without delivered orders for better comparison
    data["order_purchase_timestamp"] = pd.to_datetime(data["order_purchase_timestamp"])
    data["order_delivered_customer_date"] = pd.to_datetime(data["order_delivered_customer_date"])
    data = data[data["order_delivered_customer_date"].notna()]
    data["delivery_time_days"] = (
        data["order_delivered_customer_date"] - data["order_purchase_timestamp"]
    ).dt.days

    # Count unique orders per customer is 1 if >1 order
    customer_order_counts = data.groupby("customer_unique_id")["order_id"].nunique()
    repeat_buyer_map = (customer_order_counts > 1).astype(int)
    data["repeat_buyer"] = data["customer_unique_id"].map(repeat_buyer_map)

    # Aggregate one row per customer
    features_df = data.groupby("customer_unique_id").agg({
        "payment_value": "sum",
        "freight_value": "sum",
        "review_score": "mean",
        "product_category_name": "nunique",
        "order_item_id": "count",
        "delivery_time_days": "mean", 
        "customer_state": "first",
        "repeat_buyer": "first",
    }).rename(columns={
        "payment_value": "total_spent",
        "freight_value": "shipping_fee",
        "review_score": "avg_review",
        "product_category_name": "unique_categories",
        "order_item_id": "total_items"
    })


    # Label encode state
    le = LabelEncoder()
    features_df["customer_state"] = le.fit_transform(features_df["customer_state"])

    # Drop rows with missing labels
    features_df = features_df.dropna(subset=[parameters["target_column"]])
    return features_df


def split_data(data: pd.DataFrame, parameters: dict):
    """
    Splits data into stratified train/test sets.
    """
    target_col = parameters["target_column"]
    train_frac = parameters["train_fraction"]
    rnd_state = parameters["random_state"]

    y = data[target_col]
    X = data.drop(columns=[target_col])

    return train_test_split(
        X,
        y,
        test_size=1.0 - train_frac,
        random_state=rnd_state,
        stratify=y
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict):
    """
    Trains model using RandomizedSearchCV on XGBClassifier and saves best params.
    """
    param_dist = parameters["param_dist"]
    randsearch = RandomizedSearchCV(
        estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=parameters["random_state"]),
        param_distributions=param_dist,
        n_iter=parameters["n_iter"],
        scoring=parameters["scoring"],
        cv=parameters["cv"],
        n_jobs=1,
        verbose=1,
        random_state=parameters["random_state"]
    )   
    randsearch.fit(X_train, y_train)

    print("Best Parameters:", randsearch.best_params_)
    return randsearch.best_estimator_, randsearch.best_params_

def save_best_params(best_params: dict) -> None:
    import os, json
    os.makedirs("conf/base/model_params", exist_ok=True)
    with open("conf/base/model_params/best_xgb_params.json", "w") as f:
        json.dump(best_params, f)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, parameters: dict) -> dict:
    threshold = parameters["threshold"]
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print(f"\nClassification Report on test set (threshold = {threshold:.2f}):")
    print(classification_report(y_test, y_pred, zero_division=0))

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}\n")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

