import time
import json
import pandas as pd

from . import config
from . import utils
from . import hmm
from . import diagnostics

def run_stage8():
    logger = utils.setup_logging("Stage8")
    logger.info("Starting Stage 8: Market Regime Detection")
    
    overall_start = time.perf_counter()
    
    logger.info("Loading Macro Matrix...")
    macro_df = pd.read_parquet(config.MACRO_MATRIX_PATH)
    
    # Check if we have the required features
    missing_features = [f for f in config.HMM_FEATURES if f not in macro_df.columns]
    if missing_features:
        raise ValueError(f"Missing required macro features: {missing_features}")
        
    logger.info("Running Expanding Window Gaussian HMM...")
    start_t = time.perf_counter()
    
    # 1. Fit HMM and get expanding probabilities
    regime_probs_df, final_model, final_scaler = hmm.generate_expanding_regimes(macro_df, config)
    
    logger.info(f"HMM execution completed in {time.perf_counter() - start_t:.2f} seconds.")
    
    # 2. Save Probabilities and Discrete Labels
    logger.info("Saving regime outputs...")
    regime_probs_df.to_parquet(config.REGIME_PROBS_PATH)
    
    discrete_labels_df = regime_probs_df[['Regime_Label']]
    discrete_labels_df.to_parquet(config.REGIME_LABELS_PATH)
    
    # 3. Generate Diagnostics on the final model state
    logger.info("Generating state diagnostics and transition matrix...")
    trans_df, stats_df = diagnostics.generate_diagnostics(final_model, final_scaler, config)
    
    # 4. Metadata
    metadata = {
        "pipeline_version": "8.0",
        "n_states": config.N_STATES,
        "features": config.HMM_FEATURES,
        "expanding_window_min": config.EXPANDING_WINDOW_MIN,
        "retrain_frequency": config.RETRAIN_FREQUENCY,
        "regime_distribution": discrete_labels_df['Regime_Label'].value_counts().to_dict(),
        "random_seed": config.RANDOM_SEED
    }
    
    # Ensure json serializable
    metadata["regime_distribution"] = {str(k): int(v) for k, v in metadata["regime_distribution"].items()}
    
    with open(config.REGIME_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
        
    runtime = time.perf_counter() - overall_start
    logger.info(f"Stage 8 completed in {runtime:.2f} seconds.")

if __name__ == "__main__":
    run_stage8()
