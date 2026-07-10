import numpy as np
import pandas as pd

def construct_labels_dataframe(
    ticker: str,
    event_times: np.ndarray,
    event_prices: np.ndarray,
    vols: np.ndarray,
    exit_times: np.ndarray,
    exit_prices: np.ndarray,
    pts: np.ndarray,
    sls: np.ndarray,
    hits: np.ndarray,
    vertical_times: np.ndarray
) -> pd.DataFrame:
    
    # Map hits to strings and labels
    # 1=PT, 2=SL, 3=VERT
    hit_map = {1: 'PT', 2: 'SL', 3: 'VERTICAL'}
    label_map = {1: 1, 2: -1, 3: 0}
    
    barrier_hits = np.array([hit_map[h] for h in hits])
    labels = np.array([label_map[h] for h in hits])
    
    # Log Returns
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = np.log(exit_prices / event_prices)
        
    holding_periods = pd.Series(exit_times) - pd.Series(event_times)
        
    df = pd.DataFrame({
        "Ticker": ticker,
        "EventTime": event_times,
        "VerticalBarrierTime": vertical_times,
        "EventPrice": event_prices,
        "ProfitBarrier": pts,
        "StopLossBarrier": sls,
        "DailyVolatility": vols,
        "HoldingPeriod": holding_periods.dt.days,
        "ExitTime": exit_times,
        "ExitPrice": exit_prices,
        "BarrierHit": barrier_hits,
        "Label": labels,
        "Return": returns
    })
    
    # Drop rows where we couldn't resolve a valid path (NaN prices)
    df = df.dropna(subset=["ExitPrice", "EventPrice", "ProfitBarrier", "StopLossBarrier"]).reset_index(drop=True)
    
    return df

def validate_labels(df: pd.DataFrame):
    """
    Validates the generated Triple Barrier labels.
    """
    if len(df) == 0:
        return
        
    if not df["Label"].isin([-1, 0, 1]).all():
        raise ValueError("Invalid labels found outside {-1, 0, 1}.")
        
    if (df["ExitTime"] < df["EventTime"]).any():
        raise ValueError("ExitTime cannot be before EventTime.")
        
    if (df["ExitPrice"] <= 0).any():
        raise ValueError("ExitPrice must be > 0.")
        
    if (df["EventPrice"] <= 0).any():
        raise ValueError("EventPrice must be > 0.")
        
    # Check PT bounds
    pt_mask = df["ProfitBarrier"].notna()
    if (df.loc[pt_mask, "ProfitBarrier"] <= df.loc[pt_mask, "EventPrice"]).any():
        raise ValueError("ProfitBarrier must be > EventPrice.")
        
    # Check SL bounds
    sl_mask = df["StopLossBarrier"].notna()
    if (df.loc[sl_mask, "StopLossBarrier"] >= df.loc[sl_mask, "EventPrice"]).any():
        raise ValueError("StopLossBarrier must be < EventPrice.")
        
    if df["Label"].isna().any():
        raise ValueError("NaN labels found.")
        
    if df["ExitPrice"].isna().any():
        raise ValueError("NaN exit prices found.")
        
    if df.duplicated(subset=["Ticker", "EventTime"]).any():
        raise ValueError("Duplicate events for a ticker found.")
        
def generate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates statistics for the Triple Barrier labels.
    """
    stats = []
    for ticker, group in df.groupby("Ticker"):
        n = len(group)
        if n == 0:
            continue
            
        pt_hits = (group["BarrierHit"] == 'PT').sum()
        sl_hits = (group["BarrierHit"] == 'SL').sum()
        vert_hits = (group["BarrierHit"] == 'VERTICAL').sum()
        
        avg_vol = group["DailyVolatility"].mean()
        avg_width = (group["ProfitBarrier"] - group["StopLossBarrier"]).mean() if avg_vol > 0 else 0
        
        stats.append({
            "Ticker": ticker,
            "Number of Events": n,
            "Positive Labels": pt_hits,
            "Negative Labels": sl_hits,
            "Neutral Labels": vert_hits,
            "PT hits": pt_hits,
            "SL hits": sl_hits,
            "Vertical hits": vert_hits,
            "Positive %": (pt_hits / n) * 100.0,
            "Negative %": (sl_hits / n) * 100.0,
            "Neutral %": (vert_hits / n) * 100.0,
            "Average Holding Days": group["HoldingPeriod"].mean(),
            "Median Holding Days": group["HoldingPeriod"].median(),
            "Average Return": group["Return"].mean(),
            "Median Return": group["Return"].median(),
            "Average Volatility": avg_vol,
            "Average Barrier Width": avg_width
        })
        
    return pd.DataFrame(stats)
