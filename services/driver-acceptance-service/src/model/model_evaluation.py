import json
import pickle
import mlflow
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.model.classification_evaluator import (
    ClassificationEvaluator
)

from src.utils.logger import get_logger
from src.utils.paths import (
    SERVICE_DATA_FEATURES_DIR,
    SERVICE_MODELS_DIR,
    REPORTS_DIR
)

logger = get_logger(
    logger_name=__name__,
    log_file="model_evaluation.log"
)

# MLflow Configuration
mlflow.set_tracking_uri("http://ec2-16-171-45-136.eu-north-1.compute.amazonaws.com:5000")

mlflow.set_experiment("driver-acceptance-service")

MODEL_FILE = (
    SERVICE_MODELS_DIR /
    "lightgbm_model.pkl"
)

RUN_INFO_FILE = (
    SERVICE_MODELS_DIR /
    "run_info.json"
)

TEST_FILE = (
    SERVICE_DATA_FEATURES_DIR /
    "test.csv"
)

REPORT_FILE = (
    REPORTS_DIR /
    "evaluation_report.json"
)


def load_run_id() -> str:

    try:

        logger.info(
            "Loading MLflow run ID..."
        )

        with open(
            RUN_INFO_FILE,
            "r"
        ) as f:

            run_info = json.load(f)

        run_id = run_info["run_id"]

        logger.info(
            "Run ID loaded successfully."
        )

        return run_id

    except Exception:

        logger.exception(
            "Failed to load run ID."
        )

        raise


def load_model():

    try:

        logger.info(
            "Loading model..."
        )

        with open(
            MODEL_FILE,
            "rb"
        ) as f:

            model = pickle.load(f)

        logger.info(
            "Model loaded successfully."
        )

        return model

    except Exception:

        logger.exception(
            "Failed to load model."
        )

        raise


def load_test_data():

    try:

        logger.info(
            f"Loading test data from "
            f"{TEST_FILE}"
        )

        test_df = pd.read_csv(
            TEST_FILE
        )

        logger.info(
            f"Test data loaded successfully. "
            f"Shape: {test_df.shape}"
        )

        return test_df

    except Exception:

        logger.exception(
            "Failed to load test data."
        )

        raise


def prepare_test_data(
    test_df: pd.DataFrame
):

    try:

        logger.info(
            "Preparing test data..."
        )

        X_test = test_df.drop(
            columns=["target"]
        )

        y_test = test_df["target"]

        logger.info(
            "Test data preparation completed."
        )

        return X_test, y_test

    except Exception:

        logger.exception(
            "Failed to prepare test data."
        )

        raise


def save_evaluation_report(
    evaluator,
    y_test
):

    try:

        logger.info(
            "Saving evaluation report..."
        )

        report = {

            "accuracy": accuracy_score(
                y_test,
                evaluator.y_pred
            ),

            "precision": precision_score(
                y_test,
                evaluator.y_pred
            ),

            "recall": recall_score(
                y_test,
                evaluator.y_pred
            ),

            "f1_score": f1_score(
                y_test,
                evaluator.y_pred
            ),

            "roc_auc": roc_auc_score(
                y_test,
                evaluator.y_prob
            )
        }

        REPORTS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            REPORT_FILE,
            "w"
        ) as f:

            json.dump(
                report,
                f,
                indent=4
            )

        logger.info(
            f"Evaluation report saved at "
            f"{REPORT_FILE}"
        )

        return report

    except Exception:

        logger.exception(
            "Failed to save evaluation report."
        )

        raise


def main():

    try:

        logger.info(
            "Model evaluation started."
        )

        run_id = load_run_id()

        model = load_model()

        test_df = load_test_data()

        X_test, y_test = (
            prepare_test_data(
                test_df
            )
        )

        with mlflow.start_run(
            run_id=run_id
        ):

            logger.info(
                "Initializing evaluator..."
            )

            evaluator = (
                ClassificationEvaluator(
                    model,
                    X_test,
                    y_test
                )
            )

            evaluator.basic_metrics()

            evaluator.classification_report()

            evaluator.confusion_matrix_plot()

            evaluator.threshold_analysis()

            report = (
                save_evaluation_report(
                    evaluator,
                    y_test
                )
            )

            mlflow.log_artifact(
                str(REPORT_FILE)
            )

            logger.info(
                f"Accuracy  : {report['accuracy']:.4f}"
            )

            logger.info(
                f"Precision : {report['precision']:.4f}"
            )

            logger.info(
                f"Recall    : {report['recall']:.4f}"
            )

            logger.info(
                f"F1 Score  : {report['f1_score']:.4f}"
            )

            logger.info(
                f"ROC AUC   : {report['roc_auc']:.4f}"
            )

            logger.info(
                "Model evaluation completed."
            )

    except Exception:

        logger.exception(
            "Evaluation pipeline failed."
        )

        raise


if __name__ == "__main__":
    main()
