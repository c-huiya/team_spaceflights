from kedro.pipeline import Pipeline, node
from .nodes import (
    preprocessing,
    encode_state,
    split_data,
    train_model,
    evaluate_model,
    save_best_params
)

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preprocessing,
                inputs=[
                    "olist_orders",
                    "olist_order_items",
                    "olist_order_payments",
                    "olist_order_reviews",
                    "olist_products",
                    "olist_sellers",
                    "olist_customers",
                    "olist_geolocation",
                    "parameters"
                ],
                outputs="preprocessed_data",
                name="1_preprocessing",
            ),
            node(
                func=encode_state,
                inputs="preprocessed_data",
                outputs="encoded_data",
                name="encode_customer_state"
            ),
            node(
                func=split_data,
                inputs=["encoded_data", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="2_split_data",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "parameters"],
                outputs=["model", "best_params"],
                name="3_train_model",
            ),
            node(
                func=save_best_params,
                inputs="best_params",
                outputs=None,
                name="4_save_best_params",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test", "parameters"],
                outputs="metrics",
                name="5_evaluate_model",
            ),
            node(
                func=lambda m: m,
                inputs="model",
                outputs="saved_model_xgboost",
                name="6_save_model",
            ),
        ]
    )
