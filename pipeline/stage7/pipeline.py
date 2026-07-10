import time
import json
import pandas as pd
import numpy as np

from . import config
from . import utils
from .purged_cv import PurgedKFold

def run_stage7():
    logger = utils.setup_logging("Stage7")
    logger.info("Starting Stage 7: Purged & Embargo Cross Validation")
    
    overall_start = time.perf_counter()
    
    logger.info("Loading Labels...")
    labels_df = pd.read_parquet(config.TRIPLE_BARRIER_PATH)
    
    # Drop rows from labels that have NaN EventTime or ExitTime
    initial_len = len(labels_df)
    labels_df = labels_df.dropna(subset=['EventTime', 'ExitTime'])
    if len(labels_df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(labels_df)} labels due to missing exit times.")
    
    logger.info(f"Applying Purged {config.N_FOLDS}-Fold CV with {config.EMBARGO_PCT*100}% embargo...")
    
    pkf = PurgedKFold(n_splits=config.N_FOLDS, embargo_pct=config.EMBARGO_PCT)
    folds = pkf.split(labels_df)
    
    # Restructure folds into a DataFrame for easy querying
    fold_records = []
    
    total_train = 0
    total_test = 0
    
    for i, (train_idx, test_idx) in enumerate(folds):
        logger.info(f"Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")
        
        total_train += len(train_idx)
        total_test += len(test_idx)
        
        # We store them as long-format
        # Instead of storing millions of rows of integers, we can just save it as a dataframe
        # of (Fold, Split, Index)
        
        train_df = pd.DataFrame({'Label_Index': train_idx})
        train_df['Fold'] = i + 1
        train_df['Split'] = 'Train'
        
        test_df = pd.DataFrame({'Label_Index': test_idx})
        test_df['Fold'] = i + 1
        test_df['Split'] = 'Test'
        
        fold_records.append(train_df)
        fold_records.append(test_df)
        
    final_folds_df = pd.concat(fold_records, ignore_index=True)
    
    logger.info("Saving folds...")
    final_folds_df.to_parquet(config.FOLDS_PATH)
    
    # Extract embargo regions explicitly for diagnostics
    unique_dates = np.sort(labels_df['EventTime'].unique())
    embargo_size = int(len(unique_dates) * config.EMBARGO_PCT)
    
    embargo_regions = []
    test_date_chunks = np.array_split(np.arange(len(unique_dates)), config.N_FOLDS)
    
    for i, chunk in enumerate(test_date_chunks):
        if len(chunk) == 0:
            continue
        test_end = unique_dates[chunk[-1]]
        embargo_end_idx = min(chunk[-1] + embargo_size, len(unique_dates) - 1)
        embargo_end = unique_dates[embargo_end_idx]
        
        embargo_regions.append({
            "Fold": i + 1,
            "Test_End": test_end,
            "Embargo_End": embargo_end,
            "Embargo_Days": embargo_end_idx - chunk[-1]
        })
        
    embargo_df = pd.DataFrame(embargo_regions)
    embargo_df.to_parquet(config.EMBARGO_REGIONS_PATH)
    
    # Metadata
    metadata = {
        "pipeline_version": "7.0",
        "n_folds": config.N_FOLDS,
        "embargo_pct": config.EMBARGO_PCT,
        "embargo_size_in_dates": embargo_size,
        "avg_train_size": total_train / config.N_FOLDS,
        "avg_test_size": total_test / config.N_FOLDS
    }
    with open(config.CV_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
        
    runtime = time.perf_counter() - overall_start
    logger.info(f"Stage 7 completed in {runtime:.2f} seconds.")

if __name__ == "__main__":
    run_stage7()
