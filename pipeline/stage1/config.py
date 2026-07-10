import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
# Since the script is in the `pipeline` directory, we set BASE_DIR to the parent
BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DAILY_DIR = PROCESSED_DIR / "daily"
HOURLY_DIR = PROCESSED_DIR / "hourly"
DIAGNOSTICS_DIR = PROCESSED_DIR / "diagnostics"

LOG_FILE = DIAGNOSTICS_DIR / "validation_log.txt"
MANIFEST_FILE = PROCESSED_DIR / "manifest.json"

# =============================================================================
# CONSTANTS & METADATA
# =============================================================================
MASTER_TIMEZONE = "America/New_York"
PIPELINE_VERSION = "2.0.0"
DATA_VERSION = "1.0.0"

# Standard OHLCV Columns expected in price datasets
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Validation thresholds
MIN_PRICE = 1e-6
MIN_VOLUME = 0

# Diagnostics files
DIAG_COVERAGE = DIAGNOSTICS_DIR / "coverage.csv"
DIAG_MISSING = DIAGNOSTICS_DIR / "missing.csv"
DIAG_SUMMARY = DIAGNOSTICS_DIR / "summary.csv"
DIAG_ASSET_LIFETIMES = DIAGNOSTICS_DIR / "asset_lifetimes.csv"
DIAG_FEATURE_STATS = DIAGNOSTICS_DIR / "feature_statistics.csv"
