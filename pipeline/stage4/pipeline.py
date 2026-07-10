import time
import json
import logging
import numpy as np
import pandas as pd

from . import config
from . import utils
from . import barriers
from . import labeling

def run_stage4():
    logger = utils.setup_logging("Stage4")
    logger.info("Starting Stage 4: Triple Barrier Method")
    
    overall_start = time.perf_counter()
    
    # 1. Loading
    logger.info("Loading CUSUM events and input matrices...")
    events_df = pd.read_parquet(config.CUSUM_EVENTS_PATH)
    close_df = pd.read_parquet(config.CLOSE_MATRIX_PATH)
    high_df = pd.read_parquet(config.HIGH_MATRIX_PATH)
    low_df = pd.read_parquet(config.LOW_MATRIX_PATH)
    vol_df = pd.read_parquet(config.DAILY_VOLATILITY_PATH)
    
    # Ensure all DataFrames are perfectly aligned by index
    common_idx = close_df.index.intersection(vol_df.index).intersection(high_df.index).intersection(low_df.index)
    close_df = close_df.loc[common_idx]
    high_df = high_df.loc[common_idx]
    low_df = low_df.loc[common_idx]
    vol_df = vol_df.loc[common_idx]
    
    # Ensure times are aligned
    times = close_df.index.values
    time_to_idx = {t: i for i, t in enumerate(times)}
    
    collision_policy_map = {"conservative": 1, "optimistic": 2, "ignore": 3, "raise": 4}
    c_pol = collision_policy_map.get(config.COLLISION_POLICY, 1)
    
    all_labels = []
    
    tickers = events_df["Ticker"].unique()
    logger.info(f"Processing {len(tickers)} tickers sequentially...")
    
    # 2. Sequential Processing Loop
    for ticker in tickers:
        ticker_events = events_df[events_df["Ticker"] == ticker]
        
        # Get event indices
        e_times = ticker_events["EventTime"].values
        e_indices = np.array([time_to_idx[t] for t in e_times], dtype=np.int32)
        
        # Extract 1D numpy arrays for this ticker
        c_arr = np.ascontiguousarray(close_df[ticker].values, dtype=np.float64)
        h_arr = np.ascontiguousarray(high_df[ticker].values, dtype=np.float64)
        l_arr = np.ascontiguousarray(low_df[ticker].values, dtype=np.float64)
        v_arr = np.ascontiguousarray(vol_df[ticker].values, dtype=np.float64)
        
        # 3. Numba Kernel
        exit_idx, exit_price, pts, sls, hits = barriers.triple_barrier_kernel(
            e_indices, c_arr, h_arr, l_arr, v_arr,
            config.PT_MULTIPLIER, config.SL_MULTIPLIER, config.VERTICAL_BARRIER_DAYS, c_pol
        )
        
        # Map indices back to times
        exit_times = times[exit_idx]
        
        v_indices = np.minimum(e_indices + config.VERTICAL_BARRIER_DAYS, len(times) - 1)
        v_times = times[v_indices]
        e_prices = c_arr[e_indices]
        e_vols = v_arr[e_indices]
        
        # 4. Label DataFrame Construction
        labels_df = labeling.construct_labels_dataframe(
            ticker=ticker,
            event_times=e_times,
            event_prices=e_prices,
            vols=e_vols,
            exit_times=exit_times,
            exit_prices=exit_price,
            pts=pts,
            sls=sls,
            hits=hits,
            vertical_times=v_times
        )
        
        all_labels.append(labels_df)
        
        # 5. Instant Garbage Collection
        del ticker_events, e_indices, c_arr, h_arr, l_arr, v_arr, exit_idx, exit_price, pts, sls, hits, labels_df
        utils.clear_memory()
        
    logger.info("Concatenating all labels...")
    final_labels = pd.concat(all_labels, ignore_index=True)
    
    # 6. Validation
    logger.info("Validating Triple Barrier labels...")
    labeling.validate_labels(final_labels)
    
    # 7. Saving
    logger.info("Saving results...")
    final_labels.to_parquet(config.TRIPLE_BARRIER_LABELS_PATH)
    
    barrier_paths = final_labels[["Ticker", "EventTime", "ProfitBarrier", "StopLossBarrier", "VerticalBarrierTime"]]
    barrier_paths.to_parquet(config.BARRIER_PATHS_PATH)
    
    stats_df = labeling.generate_statistics(final_labels)
    stats_df.to_csv(config.LABEL_STATISTICS_PATH, index=False)
    
    runtime = time.perf_counter() - overall_start
    logger.info(f"Stage 4 completed in {runtime:.2f} seconds.")
    
    # 8. Metadata
    metadata = {
        "pipeline_version": "4.0",
        "pt_multiplier": config.PT_MULTIPLIER,
        "sl_multiplier": config.SL_MULTIPLIER,
        "vertical_barrier_days": config.VERTICAL_BARRIER_DAYS,
        "collision_policy": config.COLLISION_POLICY,
        "total_events_processed": len(events_df),
        "total_labels_generated": len(final_labels),
        "runtime_seconds": runtime
    }
    
    with open(config.LABEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Generated exactly {len(final_labels)} labels for {len(events_df)} events.")

if __name__ == "__main__":
    run_stage4()
