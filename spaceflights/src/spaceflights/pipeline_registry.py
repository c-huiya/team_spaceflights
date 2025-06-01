from kedro.pipeline import Pipeline
from spaceflights.pipelines.pipeline import create_pipeline

def register_pipelines() -> dict[str, Pipeline]:

    """Register project’s pipelines so Kedro can find."""
    
    return {"__default__": create_pipeline()}
