from kedro.pipeline import Pipeline
from spaceflights.pipelines.pipeline import create_pipeline

def register_pipelines() -> dict[str, Pipeline]:

    """Register the project’s pipelines so Kedro can find them."""
    
    return {"__default__": create_pipeline()}
