from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Inputs
LABELS_DIR = PROCESSED_DIR / "labels"
TRIPLE_BARRIER_PATH = LABELS_DIR / "triple_barrier_labels.parquet"
CLOSE_MATRIX_PATH = PROCESSED_DIR / "daily" / "close_matrix.parquet"

# Outputs
WEIGHTS_DIR = PROCESSED_DIR / "sample_weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_WEIGHTS_PATH = WEIGHTS_DIR / "sample_weights.parquet"
BOOTSTRAPPED_INDICES_PATH = WEIGHTS_DIR / "bootstrapped_indices.parquet"
UNIQUENESS_STATS_PATH = WEIGHTS_DIR / "uniqueness_statistics.csv"
WEIGHTS_METADATA_PATH = WEIGHTS_DIR / "weighting_metadata.json"

# Parameters
BOOTSTRAP_SAMPLES = 500000  # Size of bootstrap sample. Or None to match input size.
RANDOM_SEED = 42
