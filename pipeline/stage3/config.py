from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DAILY_DIR = PROCESSED_DIR / "daily"
HOURLY_DIR = PROCESSED_DIR / "hourly"
FEATURES_DIR = PROCESSED_DIR / "features"
EVENTS_DIR = PROCESSED_DIR / "events"

# Ensure output directories exist
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

# Input Matrices
CLOSE_MATRIX_PATH = DAILY_DIR / "close_matrix.parquet"

# Output Files
LOG_RETURNS_PATH = FEATURES_DIR / "log_returns.parquet"
DAILY_VOLATILITY_PATH = FEATURES_DIR / "daily_volatility.parquet"
VOLATILITY_SUMMARY_PATH = FEATURES_DIR / "volatility_summary.csv"

CUSUM_EVENTS_PATH = EVENTS_DIR / "cusum_events.parquet"
EVENT_STATISTICS_PATH = EVENTS_DIR / "event_statistics.csv"
EVENT_METADATA_PATH = EVENTS_DIR / "event_metadata.json"

# Core Parameters
VOLATILITY_SPAN = 100
CUSUM_THRESHOLD_MULTIPLIER = 1.0
MIN_OBSERVATIONS = 100

# Toggle Flags
USE_LOG_RETURNS = True
USE_NUMBA = True
SAVE_RETURNS = True
