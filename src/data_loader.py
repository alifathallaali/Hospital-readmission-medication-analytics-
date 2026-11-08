"""Data loader utilities for PharmaLens analytics pipeline."""
import json
from pathlib import Path
import pandas as pd

# Define base paths relative to project root
ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DATA_DIR = ROOT_DIR / "app_data"


def load_kpis(filename: str = "kpis.json") -> dict:
    """Loads KPI summary dictionary from app_data."""
    file_path = APP_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"KPI file not found at: {file_path}")
    return json.loads(file_path.read_text(encoding="utf-8"))


def load_export_table(filename: str) -> pd.DataFrame:
    """Reads a CSV export file safely from app_data."""
    file_path = APP_DATA_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Export CSV not found at: {file_path}")
    return pd.read_csv(file_path)