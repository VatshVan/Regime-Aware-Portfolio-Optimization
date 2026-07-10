import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Dict, List, Any
# Add pipeline root to path for config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pipeline.stage1.config as config

FEATURES_DIR = config.PROCESSED_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def get_weights_ffd(d: float, tau: float = 1e-4) -> np.ndarray:
    """
    Generate FFD (Fixed-Width Window) weights dynamically based on tau threshold.
    """
    w = [1.0]
    k = 1
    while True:
        w_new = -w[-1] / k * (d - k + 1)
        if abs(w_new) < tau:
            break
        w.append(w_new)
        k += 1
    return np.array(w).reshape(-1, 1)

def frac_diff_ffd(series: pd.Series, d: float, tau: float = 1e-4) -> pd.Series:
    """
    Apply Fixed-Width Window Fractional Differencing.
    Uses numpy.correlate for high-performance rolling dot product.
    """
    w = get_weights_ffd(d, tau)
    width = len(w)
    
    # Reverse weights for rolling dot product
    w_rev = w[::-1].flatten()
    
    valid_series = series.dropna()
    if len(valid_series) < width:
        return pd.Series(index=series.index, dtype=float)
        
    res = np.correlate(valid_series.values, w_rev, mode='valid')
    
    out_idx = valid_series.index[width - 1:]
    out = pd.Series(index=series.index, dtype=float)
    out.loc[out_idx] = res
    return out

def is_stationary(series: pd.Series, p_val_thresh: float = 0.05) -> Tuple[bool, float, float]:
    """
    Test stationarity using Augmented Dickey-Fuller (ADF) test.
    """
    clean = series.dropna()
    if len(clean) < 30: 
        return False, 1.0, 0.0
    
    try:
        adf = adfuller(clean.values, regression='c', autolag='AIC') # maxlag=1, 
        p_val = adf[1]
        stat = adf[0]
        return p_val < p_val_thresh, p_val, stat
    except Exception:
        return False, 1.0, 0.0

def find_optimal_d(series: pd.Series, name: str, tau: float = 1e-4, p_val_thresh: float = 0.05) -> dict:
    """
    Find optimal d in [0, 1] using grid search to balance stationarity and memory.
    """
    stat, p_val, adf_stat = is_stationary(series, p_val_thresh)
    if stat:
        return {"Feature": name, "Optimal_d": 0.0, "p_value": p_val, "adf_stat": adf_stat}
        
    d_opt = 1.0
    best_p = 1.0
    best_stat = 0.0
    
    # Coarse search
    for d in np.arange(0.1, 1.1, 0.1):
        fd = frac_diff_ffd(series, d, tau)
        stat, p_val, adf_stat = is_stationary(fd, p_val_thresh)
        if stat:
            d_opt = d
            best_p = p_val
            best_stat = adf_stat
            break
            
    # Refine search backwards
    if d_opt < 1.0:
        for d_ref in np.arange(d_opt - 0.09, d_opt, 0.01):
            fd = frac_diff_ffd(series, d_ref, tau)
            stat, p_val, adf_stat = is_stationary(fd, p_val_thresh)
            if stat:
                d_opt = d_ref
                best_p = p_val
                best_stat = adf_stat
                break

    return {"Feature": name, "Optimal_d": round(d_opt, 2), "p_value": best_p, "adf_stat": best_stat}

def process_column(args):
    col_name, series_data, tau = args
    series = pd.Series(series_data)
    
    # Fill zero or negative with small positive if we assume prices
    # Though fracdiff applies perfectly well to negative prices
    
    opt = find_optimal_d(series, col_name, tau)
    d = opt['Optimal_d']
    
    if d == 0.0:
        fd_series = series.values
    else:
        fd_series = frac_diff_ffd(series, d, tau).values
        
    return opt, fd_series

def run_frac_diff(matrix_path: Path, out_name: str, tau: float = 1e-4):
    print(f"\n--- Processing {matrix_path.name} ---")
    df = pd.read_parquet(matrix_path)
    
    if "macro" in out_name.lower():
        df = df.ffill()
    
    print(f"Starting FFD and ADF search for {len(df.columns)} features...")
    
    args = [(col, df[col].values, tau) for col in df.columns]
    
    results = []
    series_dict = {}
    
    cores = max(1, cpu_count() - 2)
    with Pool(cores) as p:
        for opt, fd_vals in tqdm(p.imap(process_column, args), total=len(args)):
            results.append(opt)
            series_dict[opt['Feature']] = fd_vals
            
    # Build summary
    opt_df = pd.DataFrame(results)
    csv_path = FEATURES_DIR / f"{out_name}_optimal_d.csv"
    opt_df.to_csv(csv_path, index=False)
    print(f"Saved Optimal d configurations to {csv_path.name}")
    
    # Build dataframe
    fd_matrix = pd.DataFrame(series_dict, index=df.index)
    pq_path = FEATURES_DIR / f"{out_name}_fracdiff.parquet"
    fd_matrix.to_parquet(pq_path)
    print(f"Saved Fractional Diff Matrix to {pq_path.name}")
    
    # Console summary
    avg_d = opt_df['Optimal_d'].mean()
    stationary_count = (opt_df['p_value'] < 0.05).sum()
    print(f"Average d required: {avg_d:.2f}")
    print(f"Strictly Stationary count: {stationary_count}/{len(opt_df)}")

if __name__ == "__main__":
    import multiprocessing
    # Required for Windows multiprocessing safety
    multiprocessing.freeze_support()
    
    run_frac_diff(config.DAILY_DIR / "close_matrix.parquet", "equity_close")
    
    macro_path = config.DAILY_DIR / "macro_matrix.parquet"
    if macro_path.exists():
        run_frac_diff(macro_path, "macro")
