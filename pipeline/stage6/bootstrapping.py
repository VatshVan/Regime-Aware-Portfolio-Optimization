import numpy as np
import pandas as pd
import numba as nb
import logging

from .weights import _compute_uniqueness_kernel

logger = logging.getLogger("Stage6.Bootstrapping")

def seq_bootstrap(labels_df: pd.DataFrame, close_df: pd.DataFrame, sample_size: int, random_seed: int = 42, num_batches: int = 100):
    """
    Implements a highly optimized, batched Sequential Bootstrapping algorithm.
    Drawing millions of samples sequentially is computationally intractable O(N * M).
    We batch the draws to update the probability distribution `num_batches` times, 
    preserving the uniqueness objective while running in milliseconds.
    """
    np.random.seed(random_seed)
    
    unique_dates = close_df.index.values
    date_to_idx = {date: i for i, date in enumerate(unique_dates)}
    
    start_idx = np.array([date_to_idx.get(t, -1) for t in labels_df['EventTime'].values])
    end_idx = np.array([date_to_idx.get(t, -1) for t in labels_df['ExitTime'].values])
    
    num_dates = len(unique_dates)
    num_labels = len(labels_df)
    
    # phi tracks the concurrency of the *already drawn* samples
    phi = np.zeros(num_dates, dtype=np.float64)
    
    bootstrapped_indices = []
    
    batch_size = max(1, sample_size // num_batches)
    remaining = sample_size
    
    logger.info(f"Starting Sequential Bootstrapping: {sample_size} samples across {num_batches} batches.")
    
    for b in range(num_batches):
        if remaining <= 0:
            break
            
        current_batch_size = min(batch_size, remaining)
        
        # 1. Evaluate uniqueness if each label were added to the current draws
        # c_t for label i would be phi + 1
        c_t = phi + 1.0
        
        # 2. Compute uniqueness u_i for all labels using Numba kernel
        u_i = _compute_uniqueness_kernel(start_idx, end_idx, c_t)
        
        # 3. Convert uniqueness to probabilities
        u_sum = np.sum(u_i)
        if u_sum == 0 or np.isnan(u_sum):
            # Fallback to uniform if something completely zeroes out
            p_i = np.ones(num_labels) / num_labels
        else:
            p_i = u_i / u_sum
            
        # 4. Draw a batch of indices according to p_i
        # We draw with replacement within the batch
        chosen = np.random.choice(num_labels, size=current_batch_size, p=p_i, replace=True)
        bootstrapped_indices.extend(chosen)
        
        # 5. Update phi with the newly drawn samples
        # To optimize updating phi, we can just use np.add.at or a quick Numba loop
        _update_phi(phi, start_idx[chosen], end_idx[chosen])
        
        remaining -= current_batch_size
        
        if (b + 1) % 20 == 0:
            logger.info(f"  Bootstrap progress: {len(bootstrapped_indices)} / {sample_size}")
            
    return np.array(bootstrapped_indices)

@nb.njit(cache=True)
def _update_phi(phi, chosen_starts, chosen_ends):
    for i in range(len(chosen_starts)):
        s = chosen_starts[i]
        e = chosen_ends[i]
        if s >= 0 and e < len(phi) and s <= e:
            phi[s:e+1] += 1.0
