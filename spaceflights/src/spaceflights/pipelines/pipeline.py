from kedro.pipeline import Pipeline, node
from .nodes import (
    preprocessing,
    split_data,
    train_model,
    evaluate_model
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Pipeline:
    1 = preprocess raw data to features
    2 = splits into train/test
    3 = train XGBoost with GridSearch
    4 = evaluate and print metrics
    5 = saves trained model
    """
    return Pipeline(
        [
            node(
                func=preprocessing,
                inputs=["final_merged_olist_with_geolocation", "parameters"],
                outputs="preprocessed_data",
                name="1_preprocessing",
            ),
            node(
                func=split_data,
                inputs=["preprocessed_data", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="2_split_data",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "parameters"],
                outputs="model",
                name="3_train_model",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test", "parameters"],
                outputs="metrics",
                name="4_evaluate_model",
            ),
            node(
                func=lambda m: m,
                inputs="model",
                outputs="saved_model_xgboost",
                name="5_save_model",
            ),
        ]
    )
