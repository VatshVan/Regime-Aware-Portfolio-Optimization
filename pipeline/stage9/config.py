from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Inputs
FEATURE_MATRIX_PATH = PROCESSED_DIR / "features" / "feature_matrix.parquet"
TRIPLE_BARRIER_PATH = PROCESSED_DIR / "labels" / "triple_barrier_labels.parquet"
SAMPLE_WEIGHTS_PATH = PROCESSED_DIR / "sample_weights" / "sample_weights.parquet"
BOOTSTRAPPED_INDICES_PATH = PROCESSED_DIR / "sample_weights" / "bootstrapped_indices.parquet"
FOLDS_PATH = PROCESSED_DIR / "cross_validation" / "folds.parquet"
REGIME_LABELS_PATH = PROCESSED_DIR / "regimes" / "regime_labels.parquet"

# Outputs
MODELS_DIR = PROCESSED_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = MODELS_DIR / "predictions.parquet"
PROBABILITIES_PATH = MODELS_DIR / "probabilities.parquet"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"
TRAINING_METADATA_PATH = MODELS_DIR / "training_metadata.json"

# Parameters
RANDOM_SEED = 42
TARGET_COL = "Label"
# True if evaluating a 3-class classifier (-1, 0, 1), False for binary (-1, 1)
MULTI_CLASS = True 

# Supported models: "logistic", "random_forest", "xgboost", "lightgbm", "catboost"
MODELS_TO_TRAIN = ["xgboost", "lightgbm"] 
# We'll just stick to the 2 fastest and most standard GBDTs to save time. 
