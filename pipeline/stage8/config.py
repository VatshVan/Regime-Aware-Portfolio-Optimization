from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Inputs
MACRO_MATRIX_PATH = PROCESSED_DIR / "daily" / "macro_matrix.parquet"

# Outputs
REGIMES_DIR = PROCESSED_DIR / "regimes"
REGIMES_DIR.mkdir(parents=True, exist_ok=True)

REGIME_LABELS_PATH = REGIMES_DIR / "regime_labels.parquet"
REGIME_PROBS_PATH = REGIMES_DIR / "regime_probabilities.parquet"
TRANSITION_MATRIX_PATH = REGIMES_DIR / "transition_matrix.csv"
STATE_STATS_PATH = REGIMES_DIR / "state_statistics.csv"
REGIME_METADATA_PATH = REGIMES_DIR / "regime_metadata.json"

# Parameters
N_STATES = 3
HMM_FEATURES = [
    "sp500_Close",    # Broad market representation
    "vix_Close",      # Market volatility
    "us10y"           # Interest rates / Macro context
]

# Walk-forward parameters
EXPANDING_WINDOW_MIN = 252 * 2  # Wait 2 years before predicting regimes
RETRAIN_FREQUENCY = 60          # Retrain HMM every 60 days to speed up execution
RANDOM_SEED = 42
