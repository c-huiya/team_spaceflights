# src/spaceflights/pipelines/nodes.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb


def notebook_preprocessing(data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    1) Build repeat_buyer at the order level:
         - For each customer_unique_id, count distinct order_id.
         - If count > 1 → 1, else 0. Attach to every row.
    2) Aggregate to exactly one row per customer, with columns:
         num_orders, total_spent, shipping_fee, avg_review,
         unique_categories, total_items, customer_state, repeat_buyer
    3) Label-encode customer_state.
    4) Return a DataFrame with one row per customer.
    """
    # (1) Compute repeat_buyer mapping per customer:
    customer_order_counts = data.groupby("customer_unique_id")["order_id"].nunique()
    repeat_buyer_map = (customer_order_counts > 1).astype(int)
    data["repeat_buyer"] = data["customer_unique_id"].map(repeat_buyer_map)

    # (2) Aggregate to customer level:
    features_df = data.groupby("customer_unique_id").agg({
        "order_id": "nunique",             # → num_orders
        "payment_value": "sum",            # → total_spent
        "freight_value": "sum",            # → shipping_fee
        "review_score": "mean",            # → avg_review
        "product_category_name": "nunique",# → unique_categories
        "order_item_id": "count",          # → total_items
        "customer_state": "first",         # → (encode later)
        "repeat_buyer": "first",           # → label
    }).rename(columns={
        "order_id": "num_orders",
        "payment_value": "total_spent",
        "freight_value": "shipping_fee",
        "review_score": "avg_review",
        "product_category_name": "unique_categories",
        "order_item_id": "total_items",
    })

    # (3) Label-encode customer_state exactly as notebook did:
    le = LabelEncoder()
    features_df["customer_state"] = le.fit_transform(features_df["customer_state"])

    # (4) Drop any rows with missing repeat_buyer (unlikely but safe)
    features_df = features_df.dropna(subset=[parameters["target_column"]])

    return features_df


def split_data(data: pd.DataFrame, parameters: dict):
    """
    1) Given the customer-level DataFrame from notebook_preprocessing,
       extract X and y exactly as the notebook did:
         - y = data['repeat_buyer']
         - X = data.drop(columns=['repeat_buyer', 'num_orders'])
           (keep only numeric columns for X; but at this point all columns
            except 'customer_state' are numeric, and we already encoded it.)
    2) Perform train_test_split(...) with:
         test_size = 1 - parameters["train_fraction"],
         random_state = parameters["random_state"],
         stratify = y
       This replicates notebook’s `train_test_split(..., stratify=y, test_size=0.3, random_state=42)`.
    3) Return X_train, X_test, y_train, y_test.
    """
    target_col = parameters["target_column"]
    train_frac = parameters["train_fraction"]
    rnd_state = parameters["random_state"]

    y = data[target_col]
    # Drop both 'repeat_buyer' and 'num_orders' from features:
    X = data.drop(columns=[target_col, "num_orders"])

    # Now do a stratified 70/30 split:
    return train_test_split(
        X,
        y,
        test_size=1.0 - train_frac,
        random_state=rnd_state,
        stratify=y
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict):
    """
    1) Instantiate an XGBClassifier(random_state=42). EXACTLY as in notebook.
    2) Use GridSearchCV with:
         - param_grid = parameters["param_grid"]
         - cv = parameters["cv"]
         - scoring = parameters["scoring"]
       (these were ['n_estimators': [50,100,150], 'learning_rate':[0.002,0.02,0.2]], cv=5, scoring='f1')
    3) Fit the grid search on (X_train, y_train).
    4) Print out grid_search.best_params_ to match the notebook's “Best Parameters: …” line.
    5) Return grid_search.best_estimator_.
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
    1) Use model.predict(X_test) to get final labels (threshold = 0.5, same as notebook).
    2) Print classification_report exactly as notebook did.
    3) Compute accuracy_score and f1_score (zero_division=0) and print them.
    4) Return a dict containing 'accuracy' and 'f1' so Kedro can store metrics if desired.
    """
    y_pred = model.predict(X_test)

    # Print classification report (matching notebook)
    print("\nClassification Report on TEST set:")
    print(classification_report(y_test, y_pred, zero_division=0))

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"\nAccuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}\n")

    return {"accuracy": acc, "f1": f1}
