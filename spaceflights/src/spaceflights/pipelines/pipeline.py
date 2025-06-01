# src/spaceflights/pipelines/pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import (
    notebook_preprocessing,
    split_data,
    train_model,
    evaluate_model
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    01_notebook_preprocessing → raw order-level CSV → customer-level features + repeat_buyer
    02_split_data            → stratified train/test split (70/30, random_state=42)
    03_train_model           → GridSearchCV over XGBClassifier (cv=5, scoring='f1')
    04_evaluate_model        → print classification report + accuracy/F1
    05_save_model            → write the final best_model to `saved_model_xgboost` (Pickle)
    """
    return Pipeline(
        [
            node(
                func=notebook_preprocessing,
                inputs=["final_merged_olist_with_geolocation", "parameters"],
                outputs="preprocessed_data",
                name="01_notebook_preprocessing",
            ),
            node(
                func=split_data,
                inputs=["preprocessed_data", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="02_split_data",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "parameters"],
                outputs="model",
                name="03_train_model",
            ),
            node(
                func=evaluate_model,
                inputs=["model", "X_test", "y_test", "parameters"],
                outputs="metrics",
                name="04_evaluate_model",
            ),
            node(
                # Pass-through: pickle “model” into saved_model_xgboost per catalog.yml
                func=lambda m: m,
                inputs="model",
                outputs="saved_model_xgboost",
                name="05_save_model",
            ),
        ]
    )
