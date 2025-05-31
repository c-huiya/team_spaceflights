from kedro.pipeline import Pipeline, node
from .nodes import preprocess_data, split_data, train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    """
    Five-node pipeline:
      01_preprocess_data  = compute 'repeat_buyer'
      02_split_data       = filter numeric features & split
      03_train_model      = fit the XGBClassifier
      04_evaluate_model   = compute metrics on the test split
      05_save_model       = persist the model into the catalog
    """
    return Pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["final_merged_olist_with_geolocation", "parameters"],
                outputs="preprocessed_data",
                name="01_preprocess_data",
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
                inputs=["model", "X_test", "y_test"],
                outputs="metrics",
                name="04_evaluate_model",
            ),
            node(
                # lambda simply returns the model object so Kedro can write it via the catalog:
                func=lambda model: model,
                inputs="model",
                outputs="saved_model_xgboost",
                name="05_save_model",
            ),
        ]
    )
