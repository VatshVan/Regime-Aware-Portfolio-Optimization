import pandas as pd
from typing import List

def generate_rebalance_dates(
    start_date: pd.Timestamp, 
    end_date: pd.Timestamp, 
    freq: str = 'W-FRI'
) -> List[pd.Timestamp]:
    """
    Generates a list of target rebalancing dates.
    'W-FRI' = Weekly on Friday
    'M' = Month end
    'BM' = Business month end
    'Q' = Quarter end
    """
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    return dates.tolist()

def get_current_regime(regimes_df: pd.DataFrame, current_date: pd.Timestamp) -> int:
    """
    Gets the active HMM regime at the given date.
    Assumes regimes_df is indexed by Datetime.
    """
    # Get the latest regime observation on or before current_date
    past_regimes = regimes_df[regimes_df.index <= current_date]
    if past_regimes.empty:
        return 1 # Default to Bull/Normal
    
    # Typically regimes_df has a 'Regime_Label' column (0 or 1)
    # The HMM mapping sorted by variance puts the high variance state at 0
    return int(past_regimes.iloc[-1]['Regime_Label'])

def align_portfolio_weights(old_weights: pd.Series, target_weights: pd.Series) -> pd.Series:
    """
    Aligns the old and new weights, preserving un-traded assets as 0.0.
    Returns the target_weights fully indexed.
    """
    all_assets = old_weights.index.union(target_weights.index)
    return target_weights.reindex(all_assets).fillna(0.0)
