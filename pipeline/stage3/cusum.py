import numpy as np
import pandas as pd
from numba import njit
from . import config

@njit(cache=True, fastmath=False, parallel=False)
def _cusum_kernel(prices: np.ndarray, returns: np.ndarray, volatility: np.ndarray, multiplier: float):
    T, N = returns.shape
    
    # Preallocate output arrays with max possible size
    max_events = T * N
    out_t = np.zeros(max_events, dtype=np.int32)
    out_asset = np.zeros(max_events, dtype=np.int32)
    out_dir = np.zeros(max_events, dtype=np.int8)
    
    out_price = np.zeros(max_events, dtype=np.float64)
    out_ret = np.zeros(max_events, dtype=np.float64)
    out_vol = np.zeros(max_events, dtype=np.float64)
    out_thresh = np.zeros(max_events, dtype=np.float64)
    
    count = 0
    
    for i in range(N):
        sPos = 0.0
        sNeg = 0.0
        for t in range(1, T):
            y_t = returns[t, i]
            vol_t = volatility[t, i]
            p_t = prices[t, i]
            
            # Skip invalid or zero-volatility data
            if np.isnan(y_t) or np.isnan(vol_t) or vol_t <= 1e-8:
                continue
                
            h_t = vol_t * multiplier
            
            # Lopez de Prado Symmetric CUSUM recurrence
            sPos = max(0.0, sPos + y_t)
            sNeg = min(0.0, sNeg + y_t)
            
            if sPos > h_t:
                sPos = 0.0
                out_t[count] = t
                out_asset[count] = i
                out_dir[count] = 1
                out_price[count] = p_t
                out_ret[count] = y_t
                out_vol[count] = vol_t
                out_thresh[count] = h_t
                count += 1
            elif sNeg < -h_t:
                sNeg = 0.0
                out_t[count] = t
                out_asset[count] = i
                out_dir[count] = -1
                out_price[count] = p_t
                out_ret[count] = y_t
                out_vol[count] = vol_t
                out_thresh[count] = h_t
                count += 1
                
    return (out_t[:count], out_asset[:count], out_dir[:count],
            out_price[:count], out_ret[:count], out_vol[:count], out_thresh[:count])


def compute_cusum_events(prices: pd.DataFrame, returns: pd.DataFrame, volatility: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    """
    Computes Symmetric CUSUM events using a highly optimized Numba kernel.
    Input DataFrames are converted to contiguous NumPy arrays before processing.
    """
    # Ensure all dataframes are perfectly aligned before numpy extraction
    # Since they were derived from each other, they should be, but it's safe to assert
    assert prices.shape == returns.shape == volatility.shape
    
    times = prices.index.values
    tickers = prices.columns.values
    
    # Extract contiguous numpy arrays
    p_arr = np.ascontiguousarray(prices.values, dtype=np.float64)
    r_arr = np.ascontiguousarray(returns.values, dtype=np.float64)
    v_arr = np.ascontiguousarray(volatility.values, dtype=np.float64)
    
    # Run CUSUM kernel
    t_idx, asset_idx, dirs, event_prices, event_rets, event_vols, event_threshs = _cusum_kernel(p_arr, r_arr, v_arr, multiplier)
    
    # Reconstruct DataFrame
    events_df = pd.DataFrame({
        "Ticker": tickers[asset_idx],
        "EventTime": times[t_idx],
        "EventPrice": event_prices,
        "Return": event_rets,
        "Volatility": event_vols,
        "Threshold": event_threshs,
        "Direction": np.where(dirs == 1, "Positive", "Negative")
    })
    
    # Sort chronologically by ticker
    events_df = events_df.sort_values(["Ticker", "EventTime"]).reset_index(drop=True)
    return events_df

def validate_events(events_df: pd.DataFrame):
    """
    Validates the generated CUSUM events.
    """
    if events_df.duplicated(subset=["Ticker", "EventTime"]).any():
        raise ValueError("Duplicate events detected for the same Ticker and EventTime.")
    
    if (events_df["Threshold"] <= 0).any():
        raise ValueError("Non-positive thresholds detected.")
        
    if events_df["Volatility"].isna().any():
        raise ValueError("NaN Volatility found in events.")
        
def generate_event_statistics(events_df: pd.DataFrame, trading_days_per_year: int = 252) -> pd.DataFrame:
    """
    Generates diagnostic statistics for the CUSUM events.
    """
    stats = []
    
    for ticker, group in events_df.groupby("Ticker"):
        num_events = len(group)
        if num_events == 0:
            continue
            
        intervals = group["EventTime"].diff().dt.days.dropna()
        
        pos_events = (group["Direction"] == "Positive").sum()
        neg_events = (group["Direction"] == "Negative").sum()
        
        # Calculate lifespan of this asset in the event series to estimate events/year
        lifespan_days = (group["EventTime"].max() - group["EventTime"].min()).days
        events_per_year = (num_events / lifespan_days * trading_days_per_year) if lifespan_days > 0 else 0.0
        
        stats.append({
            "Ticker": ticker,
            "Number of Events": num_events,
            "Events / Year": events_per_year,
            "Positive Events": pos_events,
            "Negative Events": neg_events,
            "Positive %": (pos_events / num_events) * 100.0,
            "Negative %": (neg_events / num_events) * 100.0,
            "Average Interval": intervals.mean(),
            "Median Interval": intervals.median(),
            "Minimum Interval": intervals.min(),
            "Maximum Interval": intervals.max(),
            "Average Threshold": group["Threshold"].mean()
        })
        
    return pd.DataFrame(stats)
