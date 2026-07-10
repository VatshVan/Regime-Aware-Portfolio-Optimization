import logging
import gc
import numpy as np
import pandas as pd
from typing import Dict, List

from . import config

logger = logging.getLogger("Stage9.Inference")

def run_inference(master_df: pd.DataFrame, trained_models: Dict[str, list], feature_cols: list) -> tuple:
    """
    Evaluates the trained models strictly on out-of-sample (embargoed) test sets.
    Accumulates the OOS predictions into a unified Series.
    """
    folds_df = pd.read_parquet(config.FOLDS_PATH)
    n_folds = folds_df['Fold'].max()
    
    # We will build a list of prediction records, then concat
    all_predictions = []
    
    master_indexed = master_df.set_index('Label_Index')
    
    for fold in range(1, n_folds + 1):
        logger.info(f"--- Inferencing Fold {fold} ---")
        
        fold_test_label_idx = folds_df[(folds_df['Fold'] == fold) & (folds_df['Split'] == 'Test')]['Label_Index'].values
        valid_test_idx = fold_test_label_idx[np.isin(fold_test_label_idx, master_indexed.index)]
        
        test_df = master_indexed.loc[valid_test_idx]
        X_test = test_df[feature_cols].values
        
        # Base record structure: identifying info + ground truth
        fold_results = test_df[['Datetime', 'Ticker', 'Label']].copy()
        fold_results['Fold'] = fold
        
        for model_name in config.MODELS_TO_TRAIN:
            model = trained_models[model_name][fold - 1]
            
            # Predict Probabilities
            # Shape is (n_samples, n_classes)
            probs = model.predict_proba(X_test)
            
            if config.MULTI_CLASS:
                # Classes mapped as {0: -1, 1: 0, 2: 1}
                fold_results[f"{model_name}_prob_-1"] = probs[:, 0]
                fold_results[f"{model_name}_prob_0"] = probs[:, 1]
                fold_results[f"{model_name}_prob_1"] = probs[:, 2]
                
                # Class prediction
                preds_mapped = np.argmax(probs, axis=1)
                fold_results[f"{model_name}_pred"] = preds_mapped - 1
            else:
                # Binary mapped as {0: -1, 1: 1}
                fold_results[f"{model_name}_prob_-1"] = probs[:, 0]
                fold_results[f"{model_name}_prob_1"] = probs[:, 1]
                
                preds_mapped = np.argmax(probs, axis=1)
                fold_results[f"{model_name}_pred"] = np.where(preds_mapped == 1, 1, -1)
                
        all_predictions.append(fold_results)
        
    final_df = pd.concat(all_predictions, ignore_index=True)
    return final_df
