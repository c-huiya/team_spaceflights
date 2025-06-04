import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
    1. Load all raw CSVs.
    2. Clean and dedupe each table.
    3. Merge tables step by step.
    4. Compute 'repeat_buyer' label and geolocation features.
    5. Aggregate into one row per customer with features + target.

    """
    orders = olist_orders
    order_items = olist_order_items
    order_payments = olist_order_payments
    order_reviews = olist_order_reviews
    product_df = olist_products
    seller_df = olist_sellers
    customers = olist_customers
    geo = olist_geolocation

    # Clean products data: drop any missing or duplicate rows.
    product_df = product_df.dropna().drop_duplicates()

    # Clean seller data: drop any missing or duplicate rows
    seller_df = seller_df.dropna().drop_duplicates()

    # Clean reviews: drop comment columns and duplicate review IDs
    order_rev_clean = (
        order_reviews
        .drop(columns=["review_comment_title", "review_comment_message"])
        .drop_duplicates(subset="review_id")
    )

    # Clean orders: drop duplicate rows and any missing purchase timestamps
    orders = orders.drop_duplicates().dropna(subset=["order_purchase_timestamp"])

    # Build geolocation summary (one row per ZIP prefix)
    geo_summary = (
        geo
        .groupby("geolocation_zip_code_prefix", as_index=False)
        .agg({"geolocation_lat": "mean", "geolocation_lng": "mean"})
        .rename(columns={
            "geolocation_zip_code_prefix": "zip_code_prefix",
            "geolocation_lat": "avg_lat",
            "geolocation_lng": "avg_lng"
        })
    )

    # Merge the tables step-by-step
    merged_df = pd.merge(orders, order_items, on="order_id", how="inner")
    merged_df = pd.merge(merged_df, order_payments, on="order_id", how="inner")
    merged_df = pd.merge(merged_df, order_rev_clean, on="order_id", how="inner")
    merged_df = pd.merge(merged_df, product_df, on="product_id", how="inner")
    merged_df = pd.merge(merged_df, seller_df, on="seller_id", how="inner")
    merged_df = pd.merge(merged_df, customers, on="customer_id", how="inner")

    # Convert timestamps and keep only delivered orders
    merged_df["order_purchase_timestamp"] = pd.to_datetime(merged_df["order_purchase_timestamp"])
    merged_df["order_delivered_customer_date"] = pd.to_datetime(merged_df["order_delivered_customer_date"])
    merged_df = merged_df[merged_df["order_delivered_customer_date"].notna()]
    merged_df["delivery_time_days"] = (
        merged_df["order_delivered_customer_date"] - merged_df["order_purchase_timestamp"]
    ).dt.days

    # Create repeat_buyer label (1 if customer has >1 order)
    customer_order_counts = merged_df.groupby("customer_unique_id")["order_id"].nunique()
    repeat_buyer_map = (customer_order_counts > 1).astype(int)
    merged_df["repeat_buyer"] = merged_df["customer_unique_id"].map(repeat_buyer_map)

    # Merge geolocation for customer (avg latitude/longitude by ZIP)
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

    # Merge geolocation for seller
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

    # Aggregate into one row per customer and rename columns
    features_df = (
        merged_df
        .groupby("customer_unique_id")
        .agg({
            "payment_value": "sum",            # total_spent
            "freight_value": "sum",            # shipping_fee
            "review_score": "mean",            # avg_review
            "product_category_name": "nunique",# unique_categories
            "order_item_id": "count",          # total_items
            "delivery_time_days": "mean",      # avg_delivery_time
            "customer_state": "first",         # to be encoded
            "repeat_buyer": "first"            # target
        })
        .rename(columns={
            "payment_value": "total_spent",
            "freight_value": "shipping_fee",
            "review_score": "avg_review",
            "product_category_name": "unique_categories",
            "order_item_id": "total_items"
        })
    )

    features_df = features_df.dropna(subset=[parameters["target_column"]])

    return features_df

def encode_state(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the 'customer_state' column using LabelEncoder.
    """
    le = LabelEncoder()
    data["customer_state"] = le.fit_transform(data["customer_state"])
    return data


def split_data(data: pd.DataFrame, parameters: dict):
    """
    Splits data into stratified train/test sets based on the target column.
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
    with open(parameters["best_params_path"], "r") as f:
        best_params = json.load(f)

    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=parameters["random_state"],
        **best_params
    )
    model.fit(X_train, y_train)

    print("Loaded Best Parameters:", best_params)
    return model, best_params


def save_best_params(best_params: dict) -> None:
    import os

    def convert(o):
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        raise TypeError

    os.makedirs("conf/base/model_params", exist_ok=True)
    with open("conf/base/model_params/best_xgb_params.json", "w") as f:
        json.dump(best_params, f, default=convert)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, parameters: dict) -> dict:
    """
    Evaluates the trained model at a given probability threshold.
    Prints classification report + metrics (accuracy, precision, recall, F1, AUC).
    Returns a dictionary of metrics.
    """
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

