import time
import json
import logging
import pandas as pd

from . import config
from . import utils
from . import volatility
from . import cusum

def run_stage3():
    logger = utils.setup_logging("Stage3")
    logger.info("Starting Stage 3: Financial Data Preprocessing (Volatility & CUSUM)")
    
    overall_start = time.perf_counter()
    timings = {}
    
    # ---------------------------------------------------------
    # 1. Load Input Data
    # ---------------------------------------------------------
    logger.info(f"Loading close matrix from {config.CLOSE_MATRIX_PATH}")
    close_matrix = pd.read_parquet(config.CLOSE_MATRIX_PATH)
    logger.info(f"Loaded close matrix with shape: {close_matrix.shape}")
    
    # ---------------------------------------------------------
    # 2. Compute Returns
    # ---------------------------------------------------------
    logger.info("Computing daily returns (AFML exact index alignment)...")
    start = time.perf_counter()
    
    returns_matrix = volatility.compute_daily_returns(close_matrix)
    
    if config.SAVE_RETURNS:
        returns_matrix.to_parquet(config.LOG_RETURNS_PATH)
        logger.info(f"Saved returns to {config.LOG_RETURNS_PATH}")
        
    timings["Returns"] = time.perf_counter() - start
    logger.info(f"Returns computed in {timings['Returns']:.2f} s")
    
    utils.clear_memory()
    
    # ---------------------------------------------------------
    # 3. Compute Volatility
    # ---------------------------------------------------------
    logger.info(f"Estimating daily volatility (span={config.VOLATILITY_SPAN})...")
    start = time.perf_counter()
    
    vol_matrix = volatility.estimate_volatility(returns_matrix, span=config.VOLATILITY_SPAN)
    vol_matrix.to_parquet(config.DAILY_VOLATILITY_PATH)
    
    vol_summary = volatility.generate_volatility_summary(vol_matrix)
    vol_summary.to_csv(config.VOLATILITY_SUMMARY_PATH, index=False)
    
    timings["Volatility"] = time.perf_counter() - start
    logger.info(f"Volatility estimated in {timings['Volatility']:.2f} s")
    
    utils.clear_memory()
    
    # ---------------------------------------------------------
    # 4. Symmetric CUSUM Event Detection
    # ---------------------------------------------------------
    logger.info(f"Applying Symmetric CUSUM filter (multiplier={config.CUSUM_THRESHOLD_MULTIPLIER})...")
    start = time.perf_counter()
    
    events_df = cusum.compute_cusum_events(close_matrix.loc[returns_matrix.index], returns_matrix, vol_matrix, multiplier=config.CUSUM_THRESHOLD_MULTIPLIER)
    
    logger.info("Validating events...")
    cusum.validate_events(events_df)
    
    events_df.to_parquet(config.CUSUM_EVENTS_PATH)
    
    event_stats = cusum.generate_event_statistics(events_df)
    event_stats.to_csv(config.EVENT_STATISTICS_PATH, index=False)
    
    timings["CUSUM"] = time.perf_counter() - start
    logger.info(f"CUSUM processed in {timings['CUSUM']:.2f} s")
    
    total_events = len(events_df)
    num_stocks_with_events = len(event_stats)
    avg_events_per_stock = total_events / num_stocks_with_events if num_stocks_with_events > 0 else 0
    
    del close_matrix, returns_matrix, vol_matrix, events_df
    utils.clear_memory()
    
    # ---------------------------------------------------------
    # 5. Metadata and Diagnostics
    # ---------------------------------------------------------
    timings["Total"] = time.perf_counter() - overall_start
    
    metadata = {
        "pipeline_version": "3.0",
        "volatility_span": config.VOLATILITY_SPAN,
        "cusum_threshold_multiplier": config.CUSUM_THRESHOLD_MULTIPLIER,
        "use_log_returns": config.USE_LOG_RETURNS,
        "total_events": total_events,
        "average_events_per_stock": avg_events_per_stock,
        "runtime_seconds": timings
    }
    
    with open(config.EVENT_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info("="*40)
    logger.info("STAGE 3 RUNTIME SUMMARY")
    logger.info("="*40)
    for k, v in timings.items():
        logger.info(f"{k.ljust(15)}: {v:.2f} s")
    logger.info("="*40)
    
if __name__ == "__main__":
    run_stage3()
