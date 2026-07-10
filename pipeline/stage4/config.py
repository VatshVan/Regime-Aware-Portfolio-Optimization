from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DAILY_DIR = PROCESSED_DIR / "daily"
FEATURES_DIR = PROCESSED_DIR / "features"
EVENTS_DIR = PROCESSED_DIR / "events"
LABELS_DIR = PROCESSED_DIR / "labels"

# Ensure output directory exists
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Input Matrices
CLOSE_MATRIX_PATH = DAILY_DIR / "close_matrix.parquet"
HIGH_MATRIX_PATH = DAILY_DIR / "high_matrix.parquet"
LOW_MATRIX_PATH = DAILY_DIR / "low_matrix.parquet"

DAILY_VOLATILITY_PATH = FEATURES_DIR / "daily_volatility.parquet"
CUSUM_EVENTS_PATH = EVENTS_DIR / "cusum_events.parquet"

# Output Files
TRIPLE_BARRIER_LABELS_PATH = LABELS_DIR / "triple_barrier_labels.parquet"
BARRIER_PATHS_PATH = LABELS_DIR / "barrier_paths.parquet"
LABEL_STATISTICS_PATH = LABELS_DIR / "label_statistics.csv"
LABEL_METADATA_PATH = LABELS_DIR / "label_metadata.json"

# Core Parameters
PT_MULTIPLIER = 1.0
SL_MULTIPLIER = 1.0
VERTICAL_BARRIER_DAYS = 5 # Trading days
MIN_OBSERVATIONS = 100

# Supported collision policies: "conservative" (SL first), "optimistic" (PT first), "ignore" (Vertical)
COLLISION_POLICY = "conservative"

# Toggle Flags
USE_NUMBA = True
