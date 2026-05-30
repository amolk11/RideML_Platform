from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]

# Global data
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Service directory
SERVICE_DIR = ROOT_DIR / "services" / "driver-acceptance-service"

# Service data
SERVICE_DATA_DIR = SERVICE_DIR / "data"
SERVICE_DATA_RAW_DIR = SERVICE_DATA_DIR / "raw"
SERVICE_DATA_PROCESSED_DIR = SERVICE_DATA_DIR / "processed"

# Directory for features
SERVICE_DATA_FEATURES_DIR = (SERVICE_DATA_DIR / "features")
SERVICE_DATA_ARTIFACTS_DIR = (SERVICE_DATA_DIR /"artifacts")
SERVICE_DATA_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
SERVICE_DATA_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Logs and models
LOG_DIR = SERVICE_DIR / "logs"
SERVICE_MODELS_DIR = (SERVICE_DIR / "models")

REPORTS_DIR = (SERVICE_DIR / "reports")

# Create directories
SERVICE_DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
SERVICE_DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
SERVICE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
