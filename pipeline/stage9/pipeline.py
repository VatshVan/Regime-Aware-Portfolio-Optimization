import time
import json
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, matthews_corrcoef

from . import config
from . import training
from . import inference

def run_stage9():
    logger = logging.getLogger("Stage9")
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    logger.info("Starting Stage 9: Primary Model Training")
    
    overall_start = time.perf_counter()
    
    # 1. Build Dataset
    master_df = training.build_master_dataset()
    
    # Identify feature columns
    exclude_cols = ['Datetime', 'EventTime', 'Ticker', 'Label', 'Label_Index', 'SampleWeight', 'bootstrapped_index']
    potential_features = [c for c in master_df.columns if c not in exclude_cols]
    
    # Strictly enforce numeric types for ML
    import numpy as np
    numeric_cols = master_df.select_dtypes(include=[np.number, bool]).columns.tolist()
    feature_cols = [c for c in potential_features if c in numeric_cols]
    
    logger.info(f"Identified {len(feature_cols)} features for training.")
    
    # 2. Train Models
    start_t = time.perf_counter()
    trained_models = training.train_models(master_df, feature_cols)
    logger.info(f"Training completed in {time.perf_counter() - start_t:.2f} seconds.")
    
    # 3. Out-of-Sample Inference
    start_t = time.perf_counter()
    predictions_df = inference.run_inference(master_df, trained_models, feature_cols)
    logger.info(f"Inference completed in {time.perf_counter() - start_t:.2f} seconds.")
    
    # 4. Save Predictions
    predictions_df.to_parquet(config.PREDICTIONS_PATH)
    
    # 5. Calculate Metrics
    metrics_report = {}
    y_true = predictions_df['Label'].values
    
    for model_name in config.MODELS_TO_TRAIN:
        y_pred = predictions_df[f"{model_name}_pred"].values
        
        # Calculate standard metrics
        metrics_report[model_name] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        }
        
    # 6. Save Metadata
    metadata = {
        "pipeline_version": "9.0",
        "n_samples": len(master_df),
        "n_features": len(feature_cols),
        "multi_class": config.MULTI_CLASS,
        "models": config.MODELS_TO_TRAIN,
        "metrics": metrics_report,
        "random_seed": config.RANDOM_SEED
    }
    
    with open(config.TRAINING_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
        
    runtime = time.perf_counter() - overall_start
    logger.info(f"Stage 9 completed in {runtime:.2f} seconds.")

if __name__ == "__main__":
    import logging
    run_stage9()
