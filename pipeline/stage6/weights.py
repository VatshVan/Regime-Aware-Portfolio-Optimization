import numpy as np
import pandas as pd
import numba as nb

@nb.njit(cache=True)
def _compute_concurrency_kernel(start_indices, end_indices, num_dates):
    """
    Numba kernel to rapidly compute concurrency c_t.
    start_indices and end_indices are the integer positions of EventTime and ExitTime.
    """
    c_t = np.zeros(num_dates, dtype=np.float64)
    n = len(start_indices)
    
    for i in range(n):
        s = start_indices[i]
        e = end_indices[i]
        if s >= 0 and e < num_dates and s <= e:
            # Add 1 to all dates in the span [s, e]
            c_t[s:e+1] += 1.0
            
    return c_t

@nb.njit(cache=True)
def _compute_uniqueness_kernel(start_indices, end_indices, c_t):
    """
    Numba kernel to compute average uniqueness for each label.
    u_i = mean(1 / c_t) over [start_indices[i], end_indices[i]]
    """
    n = len(start_indices)
    u_i = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        s = start_indices[i]
        e = end_indices[i]
        if s >= 0 and e < len(c_t) and s <= e:
            # Harmonic-like average of concurrency
            span_len = e - s + 1
            span_sum = 0.0
            for t in range(s, e+1):
                span_sum += 1.0 / c_t[t]
            u_i[i] = span_sum / span_len
            
    return u_i

def calculate_sample_weights(labels_df: pd.DataFrame, close_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates López de Prado's Average Uniqueness and Return Attributed Sample Weights.
    """
    # 1. Map timestamps to integer indices
    unique_dates = close_df.index.values
    date_to_idx = {date: i for i, date in enumerate(unique_dates)}
    
    start_idx = np.array([date_to_idx.get(t, -1) for t in labels_df['EventTime'].values])
    end_idx = np.array([date_to_idx.get(t, -1) for t in labels_df['ExitTime'].values])
    
    # 2. Compute Concurrency (c_t)
    c_t = _compute_concurrency_kernel(start_idx, end_idx, len(unique_dates))
    
    # 3. Compute Average Uniqueness (u_i)
    u_i = _compute_uniqueness_kernel(start_idx, end_idx, c_t)
    
    # 4. Return Attribution
    # We want to weight by absolute log return, scaled by uniqueness
    ret = np.abs(np.log(labels_df['ExitPrice'] / labels_df['EventPrice']))
    
    # If a return is exactly 0, give it a tiny weight so it isn't completely zeroed out
    ret = np.maximum(ret, 1e-6)
    
    sample_weights = u_i * ret
    
    # Optional: Standardize weights so they sum to the number of samples
    # (Matches standard Scikit-Learn convention)
    # sample_weights = sample_weights * (len(sample_weights) / sample_weights.sum())
    
    result = labels_df.copy()
    result['Uniqueness'] = u_i
    result['SampleWeight'] = sample_weights
    
    return result
