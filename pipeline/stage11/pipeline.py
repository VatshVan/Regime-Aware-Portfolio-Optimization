import time
import json
import logging
import pandas as pd

from . import config
from . import diagnostics

def run_stage11():
    logger = logging.getLogger("Stage11")
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    logger.info("Starting Stage 11: Diagnostics & Walk-Forward Validation")
    overall_start = time.perf_counter()
    
    meta_preds_df = pd.read_parquet(config.META_PREDICTIONS_PATH)
    
    calibration = diagnostics.calculate_calibration_scores(meta_preds_df)
    lift = diagnostics.calculate_profit_lift(meta_preds_df)
    
    final_report = {
        "pipeline_version": "11.0",
        "calibration": calibration,
        "profit_lift": lift
    }
    
    with open(config.FINAL_REPORT_PATH, "w") as f:
        json.dump(final_report, f, indent=4)
        
    logger.info("Stage 11 completed successfully. MLAM Pipeline fully executed.")

if __name__ == "__main__":
    run_stage11()
