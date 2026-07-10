import pandas as pd
import numpy as np

class TransactionCostEngine:
    def __init__(
        self,
        commission_bps: float = 1.0,
        slippage_bps: float = 5.0,
        bid_ask_spread_bps: float = 2.0
    ):
        self.commission_rate = commission_bps / 10000.0
        self.slippage_rate = slippage_bps / 10000.0
        self.spread_rate = (bid_ask_spread_bps / 2.0) / 10000.0 # Half-spread crossed per trade
        
    def calculate_costs(self, old_weights: pd.Series, new_weights: pd.Series, portfolio_value: float) -> dict:
        """
        Calculates the transaction costs incurred by rebalancing from old_weights to new_weights.
        
        Args:
            old_weights: Series of asset weights before rebalance
            new_weights: Series of target asset weights after rebalance
            portfolio_value: Total equity of the portfolio before costs
            
        Returns:
            dict containing turnover, commission, slippage, spread, total_cost, and net_value
        """
        # Align series to ensure all assets are accounted for
        all_assets = old_weights.index.union(new_weights.index)
        old_aligned = old_weights.reindex(all_assets).fillna(0.0)
        new_aligned = new_weights.reindex(all_assets).fillna(0.0)
        
        # Calculate weight delta (absolute change in weights)
        weight_delta = np.abs(new_aligned - old_aligned)
        
        # Total turnover (sum of absolute weight changes)
        # Note: 100% turnover means selling 50% and buying 50% (sum is 1.0)
        turnover = weight_delta.sum()
        
        # Traded value in dollars
        traded_value = turnover * portfolio_value
        
        # Compute costs
        commission_cost = traded_value * self.commission_rate
        slippage_cost = traded_value * self.slippage_rate
        spread_cost = traded_value * self.spread_rate
        
        total_cost = commission_cost + slippage_cost + spread_cost
        
        return {
            "turnover": turnover,
            "commission": commission_cost,
            "slippage": slippage_cost,
            "spread_cost": spread_cost,
            "total_cost": total_cost,
            "net_portfolio_value": portfolio_value - total_cost
        }
