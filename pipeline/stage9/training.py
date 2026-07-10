import logging
import gc
import numpy as np
import pandas as pd
from typing import Dict, Any

from . import config
from .models import get_model

logger = logging.getLogger("Stage9.Training")

def build_master_dataset() -> pd.DataFrame:
    """
    Merges all canonical sources (features, labels, regimes, weights) into a single 
    master DataFrame aligned by (Datetime/EventTime, Ticker).
    """
    logger.info("Building master dataset...")
    
    # 1. Load Labels
    labels_df = pd.read_parquet(config.TRIPLE_BARRIER_PATH).reset_index(drop=True)
    labels_df['Label_Index'] = labels_df.index
    labels_df['Datetime'] = labels_df['EventTime'].dt.tz_localize(None).dt.normalize()
    
    if not config.MULTI_CLASS:
        # Binarize: drop 0s
        labels_df = labels_df[labels_df['Label'] != 0].copy()
    
    # 2. Load Features (Memory Optimized via Chunking)
    # feature_matrix is ~7GB. We load it in chunks and immediately inner join 
    # to drop unused rows before accumulating in RAM.
    import pyarrow.parquet as pq
    
    logger.info("Iterating through feature matrix in chunks to prevent OOM...")
    labels_keys = labels_df[['Datetime', 'Ticker']].copy()
    
    parquet_file = pq.ParquetFile(config.FEATURE_MATRIX_PATH)
    merged_chunks = []
    
    for batch in parquet_file.iter_batches(batch_size=50000):
        df_chunk = batch.to_pandas()
        if isinstance(df_chunk.index, pd.MultiIndex):
            df_chunk = df_chunk.reset_index()
            
        df_chunk['Datetime'] = df_chunk['Datetime'].dt.tz_localize(None).dt.normalize()
        
        chunk_merged = labels_keys.merge(df_chunk, on=['Datetime', 'Ticker'], how='inner')
        if not chunk_merged.empty:
            merged_chunks.append(chunk_merged)
            
    features_df = pd.concat(merged_chunks, ignore_index=True)
    del merged_chunks
    gc.collect()
    
    # Cast to float32 to save memory during training
    float_cols = features_df.select_dtypes(include=['float64']).columns
    features_df[float_cols] = features_df[float_cols].astype(np.float32)
    
    # We join features based on Datetime and Ticker
    master_df = labels_df.merge(features_df, on=['Datetime', 'Ticker'], how='inner')
    
    # 3. Load Regimes
    regimes_df = pd.read_parquet(config.REGIME_LABELS_PATH)
    regimes_df.index = regimes_df.index.tz_localize(None).normalize()
    master_df = master_df.merge(regimes_df[['Regime_Label']], left_on='Datetime', right_index=True, how='left')
    
    # 4. Load Sample Weights
    weights_df = pd.read_parquet(config.SAMPLE_WEIGHTS_PATH)
    master_df = master_df.merge(weights_df[['EventTime', 'Ticker', 'SampleWeight']], on=['EventTime', 'Ticker'], how='left')
    
    # Drop rows where critical features are NaN
    master_df = master_df.dropna(subset=['SampleWeight', 'Regime_Label'])
    
    logger.info(f"Master dataset built with {len(master_df)} records.")
    return master_df

def train_models(master_df: pd.DataFrame, feature_cols: list) -> Dict[str, list]:
    """
    Executes Purged K-Fold training loop across requested models.
    """
    folds_df = pd.read_parquet(config.FOLDS_PATH)
    boot_idx = pd.read_parquet(config.BOOTSTRAPPED_INDICES_PATH)['bootstrapped_index'].values
    
    trained_models = {model_name: [] for model_name in config.MODELS_TO_TRAIN}
    
    n_folds = folds_df['Fold'].max()
    
    for fold in range(1, n_folds + 1):
        logger.info(f"--- Training Fold {fold} ---")
        
        # 1. Extract CV Indices
        fold_train_label_idx = folds_df[(folds_df['Fold'] == fold) & (folds_df['Split'] == 'Train')]['Label_Index'].values
        
        # 2. Restrict Bootstrapped Indices to Training Set to prevent leakage
        # We only keep drawn samples that fall in this fold's purged training set
        fold_train_boot_idx = boot_idx[np.isin(boot_idx, fold_train_label_idx)]
        
        # 3. Slice Master DataFrame
        # master_df has 'Label_Index' which corresponds to the original labels_df index
        # We need to construct the actual training dataframe using the bootstrapped indices
        # Since master_df might have dropped rows (NaNs), we set index to Label_Index for easy sampling
        master_indexed = master_df.set_index('Label_Index')
        
        # Find valid bootstrap indices that still exist in the master dataset
        valid_boot_idx = fold_train_boot_idx[np.isin(fold_train_boot_idx, master_indexed.index)]
        
        train_df = master_indexed.loc[valid_boot_idx]
        
        # 4. Prepare X, y, w
        X_train = train_df[feature_cols].values
        y_train = train_df['Label'].values
        w_train = train_df['SampleWeight'].values
        
        # Remap labels {-1, 0, 1} -> {0, 1, 2} for XGBoost/LGBM
        # XGBoost requires classes to start from 0
        y_train_mapped = y_train + 1 if config.MULTI_CLASS else (y_train == 1).astype(int)
        
        for model_name in config.MODELS_TO_TRAIN:
            logger.info(f"Fitting {model_name}...")
            model = get_model(model_name)
            
            if model_name in ["xgboost", "lightgbm", "catboost"]:
                # Passing sample_weight directly
                model.fit(X_train, y_train_mapped, sample_weight=w_train)
            else:
                model.fit(X_train, y_train_mapped, sample_weight=w_train)
                
            trained_models[model_name].append(model)
            
        del train_df, X_train, y_train, w_train
        gc.collect()
        
    return trained_models
