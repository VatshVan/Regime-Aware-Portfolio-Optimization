from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Inputs
FEATURE_MATRIX_PATH = PROCESSED_DIR / "features" / "feature_matrix.parquet"
PREDICTIONS_PATH = PROCESSED_DIR / "models" / "predictions.parquet"
TRIPLE_BARRIER_PATH = PROCESSED_DIR / "labels" / "triple_barrier_labels.parquet"

# Outputs
META_MODELS_DIR = PROCESSED_DIR / "meta_models"
META_MODELS_DIR.mkdir(parents=True, exist_ok=True)

META_PREDICTIONS_PATH = META_MODELS_DIR / "meta_predictions.parquet"
META_METADATA_PATH = META_MODELS_DIR / "meta_metadata.json"

# Parameters
RANDOM_SEED = 42
PRIMARY_MODEL = "xgboost"  # We'll use XGBoost as the primary model for meta-labeling
