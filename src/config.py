from pathlib import Path

# Project root (directory where Churn_Prediction_Dashboard.py is located)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data folder
DATA_DIR = ROOT_DIR / "data"

def data_path(filename: str) -> Path:
    """Return absolute path to a file inside /data."""
    return DATA_DIR / filename