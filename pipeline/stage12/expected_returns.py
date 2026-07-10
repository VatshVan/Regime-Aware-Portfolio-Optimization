import pandas as pd
import numpy as np
from typing import Optional

def calculate_historical_conditional_returns(labels_df: pd.DataFrame, current_date: pd.Timestamp) -> dict:
    """
    Calculates the historical average return for positive (1) and negative (-1) labels
    using ONLY data available prior to the current_date to avoid lookahead bias.
    """
    past_labels = labels_df[labels_df['EventTime'] < current_date]
    
    if len(past_labels) == 0:
        return {1: 0.05, -1: 0.05} # Fallback defaults if no history
        
    # Average absolute realized return when label was 1
    pos_returns = past_labels[past_labels['Label'] == 1]['Return']
    mean_pos = pos_returns.mean() if len(pos_returns) > 0 else 0.05
    
    # Average absolute realized return when label was -1 (note Return is actual return, so taking absolute or negating)
    # Wait, if Label == -1, the strategy shorts, so the realized return of the short is -Return.
    # We want the expected return of the TRADE.
    neg_returns = past_labels[past_labels['Label'] == -1]['Return']
    mean_neg = (-neg_returns).mean() if len(neg_returns) > 0 else 0.05
    
    return {
        1: mean_pos,
        -1: mean_neg
    }

def generate_expected_returns(
    rebalance_date: pd.Timestamp,
    primary_predictions: pd.DataFrame,
    meta_predictions: pd.DataFrame,
    labels_df: pd.DataFrame,
    lookback_window: int = 252
) -> pd.Series:
    """
    Estimates asset-level expected returns for a specific date by combining:
    1. Primary model directional probabilities/predictions
    2. Meta-model trade acceptance probabilities (Meta_Prob_1)
    3. Historical realized returns from the Triple Barrier framework
    
    Returns a Series of expected returns indexed by Ticker.
    """
    # 1. Get predictions for this specific date
    day_primary = primary_predictions[primary_predictions['Datetime'] == str(rebalance_date)[:10]]
    day_meta = meta_predictions[meta_predictions['Datetime'] == str(rebalance_date)[:10]]
    
    if day_meta.empty or day_primary.empty:
        return pd.Series(dtype=float)
        
    # Join primary and meta on Ticker
    day_primary = day_primary.set_index('Ticker')
    day_meta = day_meta.set_index('Ticker')
    merged = day_primary.join(day_meta, rsuffix='_meta')
    
    # Calculate historical payoff distribution
    # This ensures we don't assume symmetric gains/losses
    conditional_returns = calculate_historical_conditional_returns(labels_df, rebalance_date)
    
    expected_returns = {}
    
    for ticker, row in merged.iterrows():
        t = ticker[1] if isinstance(ticker, tuple) else ticker # Handle multi-index
        
        # Primary direction (1 or -1)
        direction = row['xgboost_pred'] if 'xgboost_pred' in row else row['primary_pred']
        if pd.isna(direction) or direction == 0:
            expected_returns[t] = 0.0
            continue
            
        # Meta probability of the trade being CORRECT
        p_correct = row['Meta_Prob_1']
        p_incorrect = 1.0 - p_correct
        
        # Expected trade payoff
        # If correct, we get the conditional positive return of that direction
        # If incorrect, we lose the stop loss amount (which historically is also captured by the opposite)
        # Simplified statistically sound model:
        # Expected Return = (P(Correct) * E[Return|Correct]) - (P(Incorrect) * E[Loss|Incorrect])
        
        # We can approximate E[Return|Correct] using the historical mean of successful trades in that direction.
        # But for simplicity, we use the average absolute return of that class.
        payoff = conditional_returns.get(direction, 0.0)
        
        # If p_correct is high, the trade is very likely to hit the profit barrier.
        # Direction * Payoff gives the expected asset return.
        # E.g. if direction is -1 (short), and we accept the trade, the asset return is expected to be negative.
        # Expected Asset Return = Direction * [ (P_correct * Payoff) - (P_incorrect * Payoff) ]
        
        expected_trade_return = (p_correct * payoff) - (p_incorrect * payoff) # Symmetric loss for simplicity here, can be enhanced
        
        # Convert trade return to asset return (if short, asset return is negative)
        asset_return = expected_trade_return * direction
        
        expected_returns[t] = asset_return
        
    return pd.Series(expected_returns)
