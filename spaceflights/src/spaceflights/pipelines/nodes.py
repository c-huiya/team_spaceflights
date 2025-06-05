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
    1. Load all raw CSVs via the provided DataFrames.
    2. Clean and dedupe each table (drop missing/duplicate rows).
    3. Merge tables step by step into a single DataFrame.
    4. Compute 'repeat_buyer' flag and add geolocation features.
    5. Aggregate into one row per customer with features + target.
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

    # Clean product and seller tables:
    # Drop any rows with missing values or exact duplicates
    product_df = product_df.dropna()

    # Parse all important timestamp columns in orders with error handling
    ts_cols = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    ]
    for col in ts_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    orders = orders.dropna(subset=["order_purchase_timestamp"])

        # Parse review timestamp columns safely
    for col in ["review_creation_date", "review_answer_timestamp"]:
        order_reviews[col] = pd.to_datetime(order_reviews[col], errors="coerce")

    order_reviews["review_score"] = order_reviews["review_score"].astype(int) #make sure review score is int

    #dropping both columns with high null value
    order_rev_clean = order_reviews.drop(columns=["review_comment_title", "review_comment_message"])

    # 1. Merge orders with order_items
    merged_df = pd.merge(orders, order_items, on='order_id', how='inner')

    # 2. Merge the result with order_payments
    merged_df = pd.merge(merged_df, order_payments, on='order_id', how='inner')

    # 3. Merge the result with order_reviews
    merged_df = pd.merge(merged_df, order_rev_clean, on='order_id', how='inner')

    # Step 4: Merge with products (via product_id)
    merged_df = pd.merge(merged_df, product_df, on='product_id', how='inner')

    # Step 5: Merge with sellers (via seller_id)
    merged_df = pd.merge(merged_df, seller_df, on='seller_id', how='inner')

    # Step 6: Merge with customers (via customer_id)
    merged_df = pd.merge(merged_df, customers, on='customer_id', how='inner')

    # Grouping geolocation:
    # Compute average latitude/longitude for each ZIP prefix
    geo_summary = geo.groupby("geolocation_zip_code_prefix", as_index=False).agg({
        "geolocation_lat": "mean",
        "geolocation_lng": "mean"
    }).rename(columns={
        "geolocation_zip_code_prefix": "zip_code_prefix",
        "geolocation_lat": "avg_lat",
        "geolocation_lng": "avg_lng"
    })

    # Merge geolocation for customer and seller:
    # Add average customer latitude/longitude by ZIP prefix
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
    # Add average seller latitude/longitude by ZIP prefix
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

    # Convert timestamps
    merged_df["order_purchase_timestamp"] = pd.to_datetime(merged_df["order_purchase_timestamp"])
    merged_df["order_delivered_customer_date"] = pd.to_datetime(merged_df["order_delivered_customer_date"])

    # Keep only orders that have a delivery date
    merged_df = merged_df[merged_df["order_delivered_customer_date"].notna()]

    # Compute delivery time in days per order
    merged_df["delivery_time_days"] = (
        merged_df["order_delivered_customer_date"] - merged_df["order_purchase_timestamp"]
    ).dt.days

    # Creating of target column called 'repeat_buyer':
    # If a customer has more than one unique order_id then repeat buyer = 1, else 0
    customer_order_counts = merged_df.groupby("customer_unique_id")["order_id"].nunique()

    repeat_buyer_map = (customer_order_counts > 1).astype(int)

    merged_df["repeat_buyer"] = merged_df["customer_unique_id"].map(repeat_buyer_map)


    # Aggregate to one row per customer:
    # Compute per-customer aggregated features and keep the repeat_buyer flag
    features_df = merged_df.groupby("customer_unique_id").agg({
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
    print("After preprocessing, total counts:", features_df["repeat_buyer"].value_counts().to_dict())

    return features_df

def encode_state(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the 'customer_state' column into integer labels.
    """
    le = LabelEncoder()
    data["customer_state"] = le.fit_transform(data["customer_state"])
    return data


def split_data(data: pd.DataFrame, parameters: dict):
    """
    Splits the customer-level DataFrame into train/test sets.
    Uses stratified sampling on the target column to preserve class balance.
    """
    target_col = parameters["target_column"]
    train_frac = parameters["train_fraction"]
    rnd_state = parameters["random_state"]

    y = data[target_col]
    X = data.drop(columns=[target_col])

    print("Full‐data counts:",   y.value_counts().to_dict())

    # Stratified split: train_frac for training, rest for testing
    return train_test_split(
        X,
        y,
        test_size=1.0 - train_frac,
        random_state=rnd_state,
        stratify=y
    )



def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict):
    """
    Loads best hyperparameters from JSON and trains an XGBoost classifier.
    Drops object columns to match the optimized notebook training process.
    """
    import json

    # Drop object (non-numeric) columns before training
    X_train = X_train.select_dtypes(exclude='object')

    # Load best parameters from JSON file
    with open(parameters["best_params_path"], "r") as f:
        best_params = json.load(f)

    # Create and train model using loaded parameters
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=parameters["random_state"],
        **best_params
    )
    model.fit(X_train, y_train)

    print("Loaded Best Parameters:", best_params)
    print("Train‐set counts:",  y_train.value_counts().to_dict())

    return model, best_params


def save_best_params(best_params: dict) -> None:
    """
    Helper to write best_params dictionary to JSON in conf/base/model_params.
    Converts NumPy scalars to native Python types if needed.
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
    Evaluates the trained model using the provided test set and probability threshold.
    Drops object columns to ensure alignment with the training process.
    """
    # Drop object columns (as in training)
    X_test = X_test.select_dtypes(exclude='object')

    # Get predicted probabilities
    threshold = parameters["threshold"]
    y_proba = model.predict_proba(X_test)[:, 1]

    # Apply threshold to get binary predictions
    y_pred = (y_proba >= threshold).astype(int)

    # Print classification report
    print(f"\nClassification Report on test set (threshold = {threshold:.2f}):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Compute and return evaluation metrics
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
    print("Test set class counts:", y_test.value_counts())

    print("Test‐set counts:",   y_test.value_counts().to_dict())   



    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }