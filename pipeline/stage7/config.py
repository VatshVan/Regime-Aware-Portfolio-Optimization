from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Inputs
LABELS_DIR = PROCESSED_DIR / "labels"
TRIPLE_BARRIER_PATH = LABELS_DIR / "triple_barrier_labels.parquet"

# Outputs
CV_DIR = PROCESSED_DIR / "cross_validation"
CV_DIR.mkdir(parents=True, exist_ok=True)

FOLDS_PATH = CV_DIR / "folds.parquet"
EMBARGO_REGIONS_PATH = CV_DIR / "embargo_regions.parquet"
CV_METADATA_PATH = CV_DIR / "cv_metadata.json"

# Parameters
N_FOLDS = 5
# Purging enforces strict elimination of overlapping event spans between train/test.
# Embargo is the extra wait time (fraction of dataset or fixed time) applied *after* a test set
# to prevent serial correlation leaking to the subsequent training set.
EMBARGO_PCT = 0.01  # 1% of the total chronological dataset length
