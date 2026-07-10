import numpy as np
import pandas as pd
from . import config

def compute_daily_returns(close_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily returns using Lopez de Prado's index search method.
    This exactly aligns the current day with the trading day closest to 1 day prior,
    handling holidays and weekends inherently.
    """
    # Find the index of the closest date 1 day prior
    idx = close_matrix.index.searchsorted(close_matrix.index - pd.Timedelta(days=1))
    idx = idx[idx > 0]
    
    # Map current timestamps to the previous timestamps
    curr_times = close_matrix.index[close_matrix.shape[0] - idx.shape[0]:]
    prev_times = close_matrix.index[idx - 1]
    
    # Extract matrices
    curr_prices = close_matrix.loc[curr_times].values
    prev_prices = close_matrix.loc[prev_times].values
    
    # Compute returns (Log or Percentage)
    if config.USE_LOG_RETURNS:
        # Avoid log of zero or negative prices by masking
        with np.errstate(divide='ignore', invalid='ignore'):
            rets = np.log(curr_prices / prev_prices)
    else:
        rets = (curr_prices / prev_prices) - 1
        
    return pd.DataFrame(rets, index=curr_times, columns=close_matrix.columns)

def estimate_volatility(returns_matrix: pd.DataFrame, span: int) -> pd.DataFrame:
    """
    Estimates daily volatility using an exponentially weighted moving standard deviation,
    as specified in Advances in Financial Machine Learning.
    """
    return returns_matrix.ewm(span=span).std()

def generate_volatility_summary(vol_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Generates diagnostic statistics for the volatility matrix.
    """
    summary = []
    for col in vol_matrix.columns:
        series = vol_matrix[col].dropna()
        if len(series) == 0:
            continue
            
        summary.append({
            "Ticker": col,
            "Mean Volatility": series.mean(),
            "Median": series.median(),
            "Std": series.std(),
            "Min": series.min(),
            "Max": series.max(),
            "Coverage %": (len(series) / len(vol_matrix)) * 100.0
        })
        
    return pd.DataFrame(summary)
