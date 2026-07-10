import pandas as pd
import numpy as np

def generate_meta_labels(predictions_df: pd.DataFrame, primary_model: str) -> pd.DataFrame:
    """
    Generates binary meta-labels. 
    1 if the primary model correctly predicted the triple barrier sign, 0 otherwise.
    """
    meta_df = predictions_df[['Datetime', 'Ticker', 'Label', 'Fold', f"{primary_model}_pred"]].copy()
    
    # Meta Label: 1 for Correct, 0 for Incorrect
    meta_df['Meta_Label'] = (meta_df['Label'] == meta_df[f"{primary_model}_pred"]).astype(int)
    
    return meta_df
