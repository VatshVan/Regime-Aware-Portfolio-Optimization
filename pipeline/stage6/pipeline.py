import time
import json
import pandas as pd
import numpy as np

from . import config
from . import utils
from . import weights
from . import bootstrapping

def run_stage6():
    logger = utils.setup_logging("Stage6")
    logger.info("Starting Stage 6: Sample Weights & Sequential Bootstrapping")
    
    overall_start = time.perf_counter()
    
    logger.info("Loading Labels and Price Matrix...")
    labels_df = pd.read_parquet(config.TRIPLE_BARRIER_PATH)
    close_df = pd.read_parquet(config.CLOSE_MATRIX_PATH)
    
    # Drop rows from labels that have NaN EventTime or ExitTime
    # (Just in case some barrier paths weren't resolved due to missing prices)
    initial_len = len(labels_df)
    labels_df = labels_df.dropna(subset=['EventTime', 'ExitTime'])
    if len(labels_df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(labels_df)} labels due to missing exit times.")
    
    # 1. Sample Weights
    logger.info("Calculating Average Uniqueness and Return Attributed Weights...")
    start_t = time.perf_counter()
    weighted_df = weights.calculate_sample_weights(labels_df, close_df)
    
    logger.info(f"Sample weights calculated in {time.perf_counter() - start_t:.2f} seconds.")
    logger.info("Saving sample weights...")
    
    # Extract only the weight components to keep things modular (or keep all)
    out_weights = weighted_df[['EventTime', 'Ticker', 'Uniqueness', 'SampleWeight']]
    out_weights.to_parquet(config.SAMPLE_WEIGHTS_PATH)
    
    # Generate uniqueness stats
    stats_df = pd.DataFrame({
        "Metric": ["Mean Uniqueness", "Median Uniqueness", "Mean Sample Weight", "Median Sample Weight"],
        "Value": [
            weighted_df['Uniqueness'].mean(),
            weighted_df['Uniqueness'].median(),
            weighted_df['SampleWeight'].mean(),
            weighted_df['SampleWeight'].median()
        ]
    })
    stats_df.to_csv(config.UNIQUENESS_STATS_PATH, index=False)
    
    # 2. Sequential Bootstrapping
    logger.info("Starting Sequential Bootstrapping...")
    start_t = time.perf_counter()
    
    sample_size = config.BOOTSTRAP_SAMPLES if config.BOOTSTRAP_SAMPLES else len(labels_df)
    
    bootstrapped_idx = bootstrapping.seq_bootstrap(
        labels_df=labels_df,
        close_df=close_df,
        sample_size=sample_size,
        random_seed=config.RANDOM_SEED,
        num_batches=100
    )
    
    logger.info(f"Bootstrapping completed in {time.perf_counter() - start_t:.2f} seconds.")
    
    # Save the indices (mapping to the rows of the labels_df / out_weights)
    pd.DataFrame({'bootstrapped_index': bootstrapped_idx}).to_parquet(config.BOOTSTRAPPED_INDICES_PATH)
    
    # 3. Metadata
    metadata = {
        "pipeline_version": "6.0",
        "total_labels": len(labels_df),
        "mean_uniqueness": float(weighted_df['Uniqueness'].mean()),
        "bootstrap_sample_size": int(sample_size),
        "bootstrap_random_seed": config.RANDOM_SEED
    }
    with open(config.WEIGHTS_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
        
    runtime = time.perf_counter() - overall_start
    logger.info(f"Stage 6 completed in {runtime:.2f} seconds.")

if __name__ == "__main__":
    run_stage6()
