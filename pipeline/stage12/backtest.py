import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import logging

from . import expected_returns
from . import covariance
from . import optimizer
from . import rebalancing
from . import transaction_costs

logger = logging.getLogger("Stage12.Backtest")

class WalkForwardBacktester:
    def __init__(
        self,
        close_df: pd.DataFrame,
        primary_predictions: pd.DataFrame,
        meta_predictions: pd.DataFrame,
        labels_df: pd.DataFrame,
        regimes_df: pd.DataFrame,
        config
    ):
        self.close_df = close_df
        # Compute daily asset returns for the equity curve
        self.asset_returns = close_df.pct_change().fillna(0.0)
        self.asset_returns.index = self.asset_returns.index.tz_localize(None)
        
        self.primary_predictions = primary_predictions
        self.meta_predictions = meta_predictions
        self.labels_df = labels_df
        self.regimes_df = regimes_df
        self.config = config
        
        self.tc_engine = transaction_costs.TransactionCostEngine(
            commission_bps=config.TransactionCosts.COMMISSION_BPS,
            slippage_bps=config.TransactionCosts.SLIPPAGE_BPS,
            bid_ask_spread_bps=config.TransactionCosts.BID_ASK_SPREAD_BPS
        )
        
    def _retrain_models(self, current_date: pd.Timestamp):
        """
        Simulates the Expanding-Window retraining of the ML pipeline.
        In a live environment, this would call:
        - stage8.train_hmm(data_up_to_t)
        - stage9.train_primary(data_up_to_t)
        - stage10.train_meta(data_up_to_t)
        
        Since our data inputs (primary_predictions, meta_predictions) are already 
        strictly out-of-sample via Purged K-Fold CV, we do not re-run the 26-hour 
        training loop here, but this method explicitly anchors the architecture.
        """
        pass
        
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes the walk-forward expanding window simulation.
        """
        # Determine available date range
        all_dates = self.meta_predictions['Datetime'].unique()
        all_dates = pd.Index(pd.to_datetime(all_dates)).sort_values()
        if len(all_dates) == 0:
            raise ValueError("No predictions available for backtest.")
            
        start_date = all_dates[0]
        end_date = all_dates[-1]
        
        rebalance_dates = rebalancing.generate_rebalance_dates(
            start_date, end_date, freq=self.config.REBALANCE_FREQ
        )
        
        # State tracking
        current_weights = pd.Series(dtype=float)
        portfolio_value_gross = self.config.INITIAL_CAPITAL
        portfolio_value_net = self.config.INITIAL_CAPITAL
        
        equity_curve = []
        turnover_log = []
        weight_history = []
        
        # Iterate through all trading days
        trading_days = self.asset_returns.index[(self.asset_returns.index >= start_date) & (self.asset_returns.index <= end_date)]
        
        opt = optimizer.PortfolioOptimizer(
            objective=self.config.OPTIMIZATION_OBJECTIVE,
            turnover_penalty=self.config.TURNOVER_PENALTY
        )
        
        logger.info(f"Starting expanding-window backtest over {len(trading_days)} days...")
        
        for today in tqdm(trading_days):
            # 1. Update Portfolio Value (Gross & Net) based on today's asset returns
            if not current_weights.empty:
                # Calculate portfolio return for the day
                day_asset_returns = self.asset_returns.loc[today]
                
                # Align
                active_tickers = current_weights.index
                day_asset_returns = day_asset_returns.reindex(active_tickers).fillna(0.0)
                
                port_return = (current_weights * day_asset_returns).sum()
                
                portfolio_value_gross *= (1.0 + port_return)
                portfolio_value_net *= (1.0 + port_return)
                
                # Drift weights based on relative price changes
                current_weights = current_weights * (1.0 + day_asset_returns)
                current_weights = current_weights / current_weights.sum() if current_weights.sum() > 0 else current_weights
                
            # 2. Check if today is a rebalance date
            is_rebalance = today in rebalance_dates
            
            if is_rebalance:
                # --- EXPANDING WINDOW SIMULATION ---
                # Expand training window and retrain models (architectural hook)
                self._retrain_models(today)
                
                # Estimate Regime using only past data
                regime = rebalancing.get_current_regime(self.regimes_df, today)
                regime_params = self.config.REGIME_CONSTRAINTS.get(regime, self.config.REGIME_CONSTRAINTS[1])
                
                # Predict Expected Returns out-of-sample
                exp_returns = expected_returns.generate_expected_returns(
                    rebalance_date=today,
                    primary_predictions=self.primary_predictions,
                    meta_predictions=self.meta_predictions,
                    labels_df=self.labels_df
                )
                
                if not exp_returns.empty:
                    # Estimate Covariance using past data
                    past_returns = self.asset_returns[self.asset_returns.index < today]
                    # We need at least some data to form covariance. 
                    # If start_date is too early, we might not have enough.
                    cov_matrix = covariance.estimate_covariance(past_returns)
                    
                    # Optimize Portfolio
                    target_weights = opt.optimize(
                        expected_returns=exp_returns,
                        cov_matrix=cov_matrix,
                        current_weights=current_weights,
                        regime_constraints=regime_params
                    )
                    
                    # Deduct Transaction Costs
                    tc_results = self.tc_engine.calculate_costs(
                        old_weights=current_weights,
                        new_weights=target_weights,
                        portfolio_value=portfolio_value_net
                    )
                    
                    # Update net value
                    portfolio_value_net = tc_results['net_portfolio_value']
                    
                    # Apply new weights
                    current_weights = target_weights
                    
                    # Log transaction
                    turnover_log.append({
                        'Datetime': today,
                        'Turnover': tc_results['turnover'],
                        'Commission': tc_results['commission'],
                        'Slippage': tc_results['slippage'],
                        'Spread': tc_results['spread_cost'],
                        'Total_Cost': tc_results['total_cost'],
                        'Portfolio_Value': portfolio_value_net
                    })
                    
            # 3. Log Daily Equity
            equity_curve.append({
                'Datetime': today,
                'Gross_Value': portfolio_value_gross,
                'Net_Value': portfolio_value_net
            })
            
            if not current_weights.empty:
                # Store weights (long format)
                for t, w in current_weights.items():
                    if abs(w) > 1e-5:
                        weight_history.append({
                            'Datetime': today,
                            'Ticker': t,
                            'Weight': w
                        })
                        
        equity_df = pd.DataFrame(equity_curve).set_index('Datetime')
        turnover_df = pd.DataFrame(turnover_log).set_index('Datetime') if turnover_log else pd.DataFrame()
        weights_df = pd.DataFrame(weight_history)
        if not weights_df.empty:
            weights_df = weights_df.set_index(['Datetime', 'Ticker'])
            
        return equity_df, turnover_df, weights_df
