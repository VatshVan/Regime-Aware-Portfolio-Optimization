import time
import json
import logging
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

from . import config
from . import labels
from . import meta_model

def run_stage10():
    logger = logging.getLogger("Stage10")
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    logger.info("Starting Stage 10: Meta Labeling")
    overall_start = time.perf_counter()
    
    # 1. Load Primary Predictions
    logger.info(f"Loading primary predictions. Using {config.PRIMARY_MODEL} as base.")
    predictions_df = pd.read_parquet(config.PREDICTIONS_PATH)
    
    # 2. Generate Meta Labels
    meta_labels_df = labels.generate_meta_labels(predictions_df, config.PRIMARY_MODEL)
    pos_ratio = meta_labels_df['Meta_Label'].mean()
    logger.info(f"Generated Meta Labels. Base Model Accuracy: {pos_ratio:.2%}")
    
    # 3. Build Meta Features
    meta_features_df = meta_model.build_meta_features(predictions_df, config.PRIMARY_MODEL)
    
    # 4. Train Meta Model and Predict
    start_t = time.perf_counter()
    meta_predictions_df = meta_model.train_and_infer_meta_model(meta_features_df, meta_labels_df, config.PRIMARY_MODEL)
    logger.info(f"Meta Training and Inference completed in {time.perf_counter() - start_t:.2f} seconds.")
    
    # 5. Save Meta Predictions
    meta_predictions_df.to_parquet(config.META_PREDICTIONS_PATH)
    
    # 6. Calculate Metrics
    y_true = meta_predictions_df['Meta_Label'].values
    y_pred = meta_predictions_df['Meta_Pred'].values
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True)
    }
    
    # 7. Save Metadata
    metadata = {
        "pipeline_version": "10.0",
        "primary_model": config.PRIMARY_MODEL,
        "n_samples": len(meta_predictions_df),
        "metrics": metrics,
        "random_seed": config.RANDOM_SEED
    }
    
    with open(config.META_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
        
    runtime = time.perf_counter() - overall_start
    logger.info(f"Stage 10 completed in {runtime:.2f} seconds.")

if __name__ == "__main__":
    run_stage10()
