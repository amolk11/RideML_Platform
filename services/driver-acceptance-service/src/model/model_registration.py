import json
import yaml
import mlflow

from src.utils.logger import get_logger
from src.utils.paths import (
    SERVICE_MODELS_DIR
)

logger = get_logger(
    logger_name=__name__,
    log_file="model_registration.log"
)

PARAMS_FILE = "params.yaml"

RUN_INFO_FILE = (
    SERVICE_MODELS_DIR /
    "run_info.json"
)

MODEL_METADATA_FILE = (
    SERVICE_MODELS_DIR /
    "model_metadata.json"
)


def load_params() -> dict:

    with open(
        PARAMS_FILE,
        "r"
    ) as f:

        return yaml.safe_load(f)


def load_run_id() -> str:

    with open(
        RUN_INFO_FILE,
        "r"
    ) as f:

        return json.load(f)["run_id"]


def register_model(
    model_uri: str,
    model_name: str
) -> int:

    registered_model = (
        mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
    )

    return int(
        registered_model.version
    )


def save_model_metadata(
    run_id: str,
    model_name: str,
    model_version: int,
    model_uri: str
) -> None:

    metadata = {
        "run_id": run_id,
        "model_name": model_name,
        "model_version": model_version,
        "model_uri": model_uri
    }

    SERVICE_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        MODEL_METADATA_FILE,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )


def main():

    try:

        logger.info(
            "Model registration started."
        )

        # -------------------------
        # Configuration
        # -------------------------

        params = load_params()

        mlflow.set_tracking_uri(
            params["mlflow"][
                "tracking_uri"
            ]
        )

        mlflow.set_experiment(
            params["mlflow"][
                "experiment_name"
            ]
        )

        # -------------------------
        # Registration Details
        # -------------------------

        model_name = (
            params[
                "model_registration"
            ]["model_name"]
        )

        run_id = load_run_id()

        model_uri = (
            f"runs:/{run_id}/model"
        )

        logger.info(
            f"Run ID: {run_id}"
        )

        logger.info(
            f"Model URI: {model_uri}"
        )

        # -------------------------
        # Register Model
        # -------------------------

        version = register_model(
            model_uri=model_uri,
            model_name=model_name
        )

        logger.info(
            f"Model registered successfully. "
            f"Version: {version}"
        )

        # -------------------------
        # Save Metadata
        # -------------------------

        save_model_metadata(
            run_id=run_id,
            model_name=model_name,
            model_version=version,
            model_uri=model_uri
        )

        logger.info(
            f"Model metadata saved to "
            f"{MODEL_METADATA_FILE}"
        )

        logger.info(
            "Model registration completed."
        )

    except Exception:

        logger.exception(
            "Model registration failed."
        )

        raise


if __name__ == "__main__":
    main()
