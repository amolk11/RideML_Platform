import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import (RAW_DATA_DIR, SERVICE_DATA_RAW_DIR)


logger = get_logger(
    logger_name=__name__,
    log_file="data_ingestion.log"
)

SOURCE_FILE = RAW_DATA_DIR / "rides.csv"

DESTINATION_FILE = SERVICE_DATA_RAW_DIR / "rides.csv"


def load_data(file_path):
    """
    Load dataset from global data directory.
    """

    try:
        logger.info(f"Loading dataset from {file_path}")

        df = pd.read_csv(file_path)

        logger.info(
            f"Dataset loaded successfully. Shape: {df.shape}"
        )

        return df

    except Exception:
        logger.exception("Failed to load dataset.")
        raise


def save_data(df, file_path):
    """
    Save dataset to service data directory.
    """

    try:
        df.to_csv(file_path, index=False)

        logger.info(
            f"Dataset saved successfully at {file_path.resolve()}"
        )

    except Exception:
        logger.exception("Failed to save dataset.")
        raise


def main():

    logger.info("Data ingestion started.")

    if not SOURCE_FILE.exists():
        logger.error(
            f"Source dataset not found at {SOURCE_FILE.resolve()}"
        )
        return

    df = load_data(SOURCE_FILE)

    save_data(df, DESTINATION_FILE)

    logger.info("Data ingestion completed.")


if __name__ == "__main__":
    main()
    