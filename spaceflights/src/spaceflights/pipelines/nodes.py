import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def preprocessing(data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Load raw CSVs, clean tables, merge everything, then build one row per customer
    with features and the `repeat_buyer` label.
    """
    # Load raw CSVs
    orders = pd.read_csv("data/raw/olist_orders_dataset.csv")
    order_items = pd.read_csv("data/raw/olist_order_items_dataset.csv")
    order_payments = pd.read_csv("data/raw/olist_order_payments_dataset.csv")
    order_reviews = pd.read_csv("data/raw/olist_order_reviews_dataset.csv")
    product_df = pd.read_csv("data/raw/olist_products_dataset.csv")
    seller_df = pd.read_csv("data/raw/olist_sellers_dataset.csv")
    customers = pd.read_csv("data/raw/olist_customers_dataset.csv")
    geo = pd.read_csv("data/raw/olist_geolocation_dataset.csv")

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

    # Encode the customer_state as an integer
    le = LabelEncoder()
    features_df["customer_state"] = le.fit_transform(features_df["customer_state"])

    # Drop any customers missing the repeat_buyer label
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

