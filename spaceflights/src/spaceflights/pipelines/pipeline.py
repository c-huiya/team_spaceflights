# src/spaceflights/pipelines/pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import preprocess_data, split_data, train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    """
    1_preprocess_data = compute 'repeat_buyer'
    2_split_data      = drop non-numeric, split train/test
    3_train_model     = train XGBClassifier on X_train, y_train
    4_evaluate_model  = apply threshold, print metrics
    5_save_model      = write trained model back to saved_model_xgboost
    """
    return Pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["final_merged_olist_with_geolocation", "parameters"],
                outputs="preprocessed_data",
                name="1_preprocess_data",
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
                # This just tells Kedro to pickle “model” into saved_model_xgboost.
                func=lambda m: m,
                inputs="model",
                outputs="saved_model_xgboost",
                name="5_save_model",
            ),
        ]
    )
