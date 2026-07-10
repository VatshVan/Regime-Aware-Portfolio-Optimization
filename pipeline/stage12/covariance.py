import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf, OAS, EmpiricalCovariance

def estimate_covariance(
    returns_df: pd.DataFrame, 
    method: str = 'ledoit_wolf',
    min_periods: int = 126
) -> pd.DataFrame:
    """
    Estimates the covariance matrix of asset returns using robust shrinkage methods.
    
    Args:
        returns_df: DataFrame of historical returns (Rows: Datetime, Cols: Tickers).
        method: 'ledoit_wolf' (default), 'oas', or 'sample'.
        min_periods: Minimum number of historical observations required.
        
    Returns:
        pd.DataFrame: Covariance matrix (Tickers x Tickers).
    """
    if len(returns_df) < min_periods:
        # Fallback to identity matrix scaled by average variance if not enough data
        avg_var = returns_df.var().mean() if not returns_df.empty else 0.0001
        if np.isnan(avg_var): avg_var = 0.0001
        n_assets = returns_df.shape[1]
        return pd.DataFrame(np.eye(n_assets) * avg_var, index=returns_df.columns, columns=returns_df.columns)
        
    # Drop columns that are completely all NaNs to prevent solver failure
    clean_returns = returns_df.dropna(axis=1, how='all')
    clean_returns = clean_returns.fillna(0) # Fill intermittent NaNs with 0 (no return)
    
    if method == 'ledoit_wolf':
        model = LedoitWolf()
    elif method == 'oas':
        model = OAS()
    elif method == 'sample':
        model = EmpiricalCovariance()
    else:
        raise ValueError(f"Unknown covariance method: {method}")
        
    try:
        model.fit(clean_returns.values)
        cov_matrix = model.covariance_
    except Exception as e:
        # Fallback if convergence fails
        cov_matrix = clean_returns.cov().values
        
    # Convert back to DataFrame, re-aligning with original tickers if any were dropped
    cov_df = pd.DataFrame(cov_matrix, index=clean_returns.columns, columns=clean_returns.columns)
    
    # Reindex to original columns, filling missing with 0 cov and avg var on diag
    full_cov_df = cov_df.reindex(index=returns_df.columns, columns=returns_df.columns).fillna(0)
    
    vals = full_cov_df.values.copy()
    np.fill_diagonal(vals, vals.diagonal() + 1e-6) # Add tiny ridge for invertibility
    full_cov_df = pd.DataFrame(vals, index=full_cov_df.index, columns=full_cov_df.columns)
    
    return full_cov_df
