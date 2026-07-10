import logging
import pandas as pd
import numpy as np
import gc
from sklearn.ensemble import RandomForestClassifier

from . import config

logger = logging.getLogger("Stage10.MetaModel")

def build_meta_features(predictions_df: pd.DataFrame, primary_model: str) -> pd.DataFrame:
    """
    Constructs the feature set for the meta model.
    Meta Features = Original Features + Primary Model Probabilities
    """
    import pyarrow.parquet as pq
    
    logger.info("Iterating through feature matrix to construct meta features...")
    labels_keys = predictions_df[['Datetime', 'Ticker']].copy()
    labels_keys['Datetime'] = labels_keys['Datetime'].dt.tz_localize(None).dt.normalize()
    
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
    
    float_cols = features_df.select_dtypes(include=['float64']).columns
    features_df[float_cols] = features_df[float_cols].astype(np.float32)
    
    # Merge Features back onto predictions
    # predictions_df Datetime might be naive already, but let's ensure it matches
    preds_copy = predictions_df.copy()
    preds_copy['Datetime'] = preds_copy['Datetime'].dt.tz_localize(None).dt.normalize()
    
    meta_X = preds_copy.merge(features_df, on=['Datetime', 'Ticker'], how='inner')
    
    return meta_X

def train_and_infer_meta_model(meta_features_df: pd.DataFrame, meta_labels_df: pd.DataFrame, primary_model: str) -> pd.DataFrame:
    """
    Trains a Random Forest Meta Classifier on the out-of-sample primary predictions
    to determine whether to take the bet or pass.
    """
    # Join labels onto features
    df = meta_features_df.merge(meta_labels_df[['Datetime', 'Ticker', 'Meta_Label']], on=['Datetime', 'Ticker'], how='inner')
    
    exclude_cols = ['Datetime', 'EventTime', 'Ticker', 'Label', 'Label_Index', 'SampleWeight', 'bootstrapped_index', 'Fold', 'Meta_Label']
    # Also exclude raw predictions from other models if present
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.endswith("_pred")]
    numeric_cols = df.select_dtypes(include=[np.number, bool]).columns.tolist()
    feature_cols = [c for c in feature_cols if c in numeric_cols]
    
    n_folds = df['Fold'].max()
    all_predictions = []
    
    for fold in range(1, n_folds + 1):
        logger.info(f"--- Meta Training Fold {fold} ---")
        
        train_df = df[df['Fold'] != fold]
        test_df = df[df['Fold'] == fold]
        
        X_train = train_df[feature_cols].values
        y_train = train_df['Meta_Label'].values
        
        X_test = test_df[feature_cols].values
        
        # Meta Model: Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',
            n_jobs=-1,
            random_state=config.RANDOM_SEED
        )
        
        rf.fit(X_train, y_train)
        
        fold_results = test_df[['Datetime', 'Ticker', 'Meta_Label', 'Fold']].copy()
        
        probs = rf.predict_proba(X_test)
        fold_results['Meta_Prob_0'] = probs[:, 0]
        fold_results['Meta_Prob_1'] = probs[:, 1]
        
        fold_results['Meta_Pred'] = (probs[:, 1] > 0.5).astype(int)
        
        all_predictions.append(fold_results)
        
    final_df = pd.concat(all_predictions, ignore_index=True)
    return final_df
