import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb


def preprocessing(data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Prepares customer-level features with repeat_buyer label.
    """
    # Count unique orders per customer is 1 if >1 order
    customer_order_counts = data.groupby("customer_unique_id")["order_id"].nunique()
    repeat_buyer_map = (customer_order_counts > 1).astype(int)
    data["repeat_buyer"] = data["customer_unique_id"].map(repeat_buyer_map)

    # Aggregate one row per customer
    features_df = data.groupby("customer_unique_id").agg({
        "order_id": "nunique",
        "payment_value": "sum",
        "freight_value": "sum",
        "review_score": "mean",
        "product_category_name": "nunique",
        "order_item_id": "count",
        "customer_state": "first",
        "repeat_buyer": "first",
    }).rename(columns={
        "order_id": "num_orders",
        "payment_value": "total_spent",
        "freight_value": "shipping_fee",
        "review_score": "avg_review",
        "product_category_name": "unique_categories",
        "order_item_id": "total_items",
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
    X = data.drop(columns=[target_col, "num_orders"])

    return train_test_split(
        X,
        y,
        test_size=1.0 - train_frac,
        random_state=rnd_state,
        stratify=y
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict):
    """
    Trains model using GridSearchCV on XGBClassifier.
    """
    base_model = xgb.XGBClassifier(random_state=parameters["random_state"])
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=parameters["param_grid"],
        scoring=parameters["scoring"],
        cv=parameters["cv"],
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    print("Best Parameters:", grid.best_params_)
    return grid.best_estimator_


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, parameters: dict) -> dict:
    """
    Evaluates the model and prints metrics.
    """
    y_pred = model.predict(X_test)
    print("\nClassification Report on test set:")
    print(classification_report(y_test, y_pred, zero_division=0))

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"\nAccuracy: {acc:.4f}")
    print(f"\nPrecision: {precision:.4f}")
    print(f"\nRecall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
