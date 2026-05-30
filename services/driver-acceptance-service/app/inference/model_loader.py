import json
import joblib

from src.utils.paths import (
    SERVICE_MODELS_DIR,
    SERVICE_DATA_ARTIFACTS_DIR
)

MODEL_FILE = (
    SERVICE_MODELS_DIR /
    "lightgbm_model.pkl"
)

ARTIFACT_FILE = (
    SERVICE_DATA_ARTIFACTS_DIR /
    "feature_artifacts.json"
)


def load_model():

    return joblib.load(
        MODEL_FILE
    )


def load_artifacts():

    with open(
        ARTIFACT_FILE,
        "r"
    ) as f:

        return json.load(f)
    