import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import json


def preprocessing(
    olist_orders: pd.DataFrame,
    olist_order_items: pd.DataFrame,
    olist_order_payments: pd.DataFrame,
    olist_order_reviews: pd.DataFrame,
    olist_products: pd.DataFrame,
    olist_sellers: pd.DataFrame,
    olist_customers: pd.DataFrame,
    olist_geolocation: pd.DataFrame,
    parameters: dict
) -> pd.DataFrame:
    """
    Preprocesses and aggregates multiple Olist datasets into a single customer-level dataset
    with engineered features and a 'repeat_buyer' target label.
    """

    # Alias inputs for clarity
    orders = olist_orders
    order_items = olist_order_items
    order_payments = olist_order_payments
    order_reviews = olist_order_reviews
    product_df = olist_products
    seller_df = olist_sellers
    customers = olist_customers
    geo = olist_geolocation

    # Drop rows with missing values from products
    product_df = product_df.dropna()

    # Parse timestamp columns in orders
    ts_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in ts_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    # Drop orders missing purchase timestamp
    orders = orders.dropna(subset=["order_purchase_timestamp"])

    # Parse review timestamps
    for col in ["review_creation_date", "review_answer_timestamp"]:
        order_reviews[col] = pd.to_datetime(order_reviews[col], errors="coerce")

    # Ensure review score is integer
    order_reviews["review_score"] = order_reviews["review_score"].astype(int)

    # Drop review text columns (high nulls) and sort reviews
    order_rev_clean = order_reviews.drop(columns=["review_comment_title", "review_comment_message"])
    order_rev_clean = order_rev_clean.sort_values(by='review_score', ascending=False)

    # Merge all relevant tables step by step
    merged_df = pd.merge(orders, order_items, on='order_id', how='inner')
    merged_df = pd.merge(merged_df, order_payments, on='order_id', how='inner')
    merged_df = pd.merge(merged_df, order_rev_clean, on='order_id', how='inner')
    merged_df = pd.merge(merged_df, product_df, on='product_id', how='inner')
    merged_df = pd.merge(merged_df, seller_df, on='seller_id', how='inner')
    merged_df = pd.merge(merged_df, customers, on='customer_id', how='inner')

    # Create geolocation summary per ZIP prefix
    geo_summary = geo.groupby("geolocation_zip_code_prefix", as_index=False).agg({
        "geolocation_lat": "mean",
        "geolocation_lng": "mean"
    }).rename(columns={
        "geolocation_zip_code_prefix": "zip_code_prefix",
        "geolocation_lat": "avg_lat",
        "geolocation_lng": "avg_lng"
    })

    # Merge geolocation info for customer and seller ZIPs
    merged_df = pd.merge(
        merged_df,
        geo_summary.rename(columns={
            "zip_code_prefix": "customer_zip_code_prefix",
            "avg_lat": "customer_lat",
            "avg_lng": "customer_lng"
        }),    
        on="customer_zip_code_prefix",
        how="left"
    )
    merged_df = pd.merge(
        merged_df,
        geo_summary.rename(columns={
            "zip_code_prefix": "seller_zip_code_prefix",
            "avg_lat": "seller_lat",
            "avg_lng": "seller_lng"
        }),
        on="seller_zip_code_prefix",
        how="left"
    )

    # Ensure key timestamps are in datetime format
    merged_df["order_purchase_timestamp"] = pd.to_datetime(merged_df["order_purchase_timestamp"])
    merged_df["order_delivered_customer_date"] = pd.to_datetime(merged_df["order_delivered_customer_date"])

    # Remove orders with missing delivery dates
    merged_df = merged_df[merged_df["order_delivered_customer_date"].notna()]

    # Calculate delivery time in days
    merged_df["delivery_time_days"] = (
        merged_df["order_delivered_customer_date"] - merged_df["order_purchase_timestamp"]
    ).dt.days

    # Generate binary target: 1 if customer placed more than one unique order
    customer_order_counts = merged_df.groupby("customer_unique_id")["order_id"].nunique()
    repeat_buyer_map = (customer_order_counts > 1).astype(int)
    merged_df["repeat_buyer"] = merged_df["customer_unique_id"].map(repeat_buyer_map)

    # Aggregate into one row per customer with selected features
    merged_df = merged_df.groupby("customer_unique_id").agg({
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

    # Print class counts for debugging
    print("After preprocessing, value counts:", merged_df["repeat_buyer"].value_counts().to_dict())

    return merged_df


def encode_state(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the 'customer_state' categorical column into numerical labels using LabelEncoder.
    """
    le = LabelEncoder()
    data["customer_state"] = le.fit_transform(data["customer_state"])
    return data


def split_data(data: pd.DataFrame, parameters: dict):
    """
    Splits the customer-level DataFrame into train and test sets using stratified sampling
    based on the target column to maintain class distribution.
    """
    target_col = parameters["target_column"]
    train_frac = parameters["train_fraction"]
    rnd_state = parameters["random_state"]

    y = data[target_col]
    X = data.drop(columns=[target_col])

    # Perform stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_frac,
        random_state=rnd_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict):
    """
    Trains an XGBoost classifier using previously optimized hyperparameters.
    Removes object columns to ensure numeric input for XGBoost.
    """
    X_train = X_train.select_dtypes(exclude='object')

    # Load best parameters from JSON
    with open(parameters["best_params_path"], "r") as f:
        best_params = json.load(f)

    # Initialize and train the model
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=parameters["random_state"],
        **best_params
    )
    model.fit(X_train, y_train)

    # Log the loaded parameters
    print("Loaded Best Parameters:", best_params)
    return model, best_params

def save_best_params(best_params: dict) -> None:
    """
    Saves the best XGBoost parameters to JSON file, handling NumPy data types.
    """
    import os
    import numpy as np

    def convert(o):
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        raise TypeError

    os.makedirs("conf/base/model_params", exist_ok=True)
    with open("conf/base/model_params/best_xgb_params.json", "w") as f:
        json.dump(best_params, f, default=convert)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, parameters: dict) -> dict:
    """
    Evaluates model on the test set using specified probability threshold.
    Calculates and prints classification metrics (excluding classification report).
    """
    # Remove non-numeric features
    X_test = X_test.select_dtypes(exclude='object')

    # Predict probabilities for positive class
    threshold = parameters["threshold"]
    y_proba = model.predict_proba(X_test)[:, 1]

    # Convert probabilities to binary predictions using threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Compute and print performance metrics
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
