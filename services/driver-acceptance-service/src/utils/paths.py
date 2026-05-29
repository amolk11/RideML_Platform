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

# Logs and models
LOG_DIR = SERVICE_DIR / "logs"
MODEL_DIR = SERVICE_DIR / "models"

# Create directories
SERVICE_DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
SERVICE_DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)