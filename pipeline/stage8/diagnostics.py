import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from .hmm import _get_canonical_mapping

def generate_diagnostics(model: GaussianHMM, scaler: StandardScaler, config):
    """
    Extracts canonical transition matrices and state statistics from the global HMM.
    """
    mapping = _get_canonical_mapping(model)
    n_states = model.n_components
    
    # 1. Map Transition Matrix
    # transmat_[i, j] = probability of going from i to j
    raw_transmat = model.transmat_
    canonical_transmat = np.zeros_like(raw_transmat)
    
    for orig_i in range(n_states):
        can_i = mapping[orig_i]
        for orig_j in range(n_states):
            can_j = mapping[orig_j]
            canonical_transmat[can_i, can_j] = raw_transmat[orig_i, orig_j]
            
    # Save Transition Matrix
    cols = [f"To_Regime_{i}" for i in range(n_states)]
    idx = [f"From_Regime_{i}" for i in range(n_states)]
    trans_df = pd.DataFrame(canonical_transmat, index=idx, columns=cols)
    trans_df.to_csv(config.TRANSITION_MATRIX_PATH)
    
    # 2. Map State Statistics (Means and Variances)
    # Inverse transform means
    raw_means = model.means_
    orig_means = scaler.inverse_transform(raw_means)
    
    stats_records = []
    
    for orig_i in range(n_states):
        can_i = mapping[orig_i]
        
        # Calculate trace of covariance as a proxy for total regime volatility
        total_variance = np.trace(model.covars_[orig_i])
        
        record = {
            "Regime": can_i,
            "Total_Variance": total_variance
        }
        
        for f_idx, f_name in enumerate(config.HMM_FEATURES):
            record[f"{f_name}_mean"] = orig_means[orig_i, f_idx]
            
        stats_records.append(record)
        
    stats_df = pd.DataFrame(stats_records)
    stats_df = stats_df.sort_values(by="Regime").reset_index(drop=True)
    stats_df.to_csv(config.STATE_STATS_PATH, index=False)
    
    return trans_df, stats_df
