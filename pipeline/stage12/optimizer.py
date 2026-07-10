import cvxpy as cp
import numpy as np
import pandas as pd

class PortfolioOptimizer:
    def __init__(self, objective: str = 'max_sharpe', turnover_penalty: float = 0.005):
        """
        Args:
            objective: 'max_sharpe', 'min_variance', 'risk_parity'
            turnover_penalty: L1 penalty multiplier for weight changes
        """
        self.objective_type = objective
        self.turnover_penalty = turnover_penalty
        
    def optimize(
        self, 
        expected_returns: pd.Series, 
        cov_matrix: pd.DataFrame, 
        current_weights: pd.Series,
        regime_constraints: dict
    ) -> pd.Series:
        """
        Solves the convex optimization problem for portfolio weights.
        """
        # Ensure alignment
        tickers = expected_returns.index
        mu = expected_returns.values
        Sigma = cov_matrix.loc[tickers, tickers].values
        n = len(tickers)
        
        # Current weights aligned to active tickers
        w_curr = np.zeros(n)
        if not current_weights.empty:
            for i, t in enumerate(tickers):
                w_curr[i] = current_weights.get(t, 0.0)
                
        # Variables
        w = cp.Variable(n)
        
        # Default constraints
        constraints = [
            cp.sum(w) <= 1.0 - regime_constraints.get('cash_allocation', 0.0),
            cp.sum(w) >= 0.0 # Fully invested up to cash limit or allowed shorting
        ]
        
        # Max leverage / gross exposure
        # If long-only, w >= 0. If long-short, cp.norm(w, 1) <= max_leverage
        max_leverage = regime_constraints.get('max_leverage', 1.0)
        if max_leverage == 1.0:
            constraints.append(w >= 0.0)
        else:
            constraints.append(cp.norm(w, 1) <= max_leverage)
            
        # Max asset weight
        max_asset = regime_constraints.get('max_asset_weight', 0.10)
        constraints.append(w <= max_asset)
        constraints.append(w >= -max_asset)
        
        # Overwrite objective based on regime if forced
        active_objective = regime_constraints.get('objective', self.objective_type)
        
        # Objective Functions
        ret = mu.T @ w
        risk = cp.quad_form(w, Sigma)
        turnover = cp.norm(w - w_curr, 1)
        
        if active_objective == 'min_variance':
            # Minimize variance + turnover penalty
            objective = cp.Minimize(risk + self.turnover_penalty * turnover)
            
        elif active_objective == 'max_sharpe':
            # Maximize Risk-Adjusted Return (Mean-Variance formulation)
            # gamma is risk aversion. Standard is roughly 2.0 to 5.0
            gamma = 2.0 
            objective = cp.Maximize(ret - gamma * risk - self.turnover_penalty * turnover)
            
        elif active_objective == 'risk_parity':
            # Simplified risk parity via min variance with lower bounds
            # For true risk parity, we need a nonlinear solver, but we can approximate
            # by penalizing deviation from equal risk contribution or just equal weight lower bound
            constraints.append(w >= (1.0 / n) * 0.2) # Soft equal weight anchor
            objective = cp.Minimize(risk + self.turnover_penalty * turnover)
            
        else:
            objective = cp.Maximize(ret - self.turnover_penalty * turnover)
            
        # Solve
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.ECOS) # Robust solver
            if w.value is None:
                prob.solve(solver=cp.SCS) # Fallback
                
            weights = w.value
            # Clean up tiny numerical artifacts
            weights[np.abs(weights) < 1e-4] = 0.0
        except Exception as e:
            # Fallback to current weights or equal weight if solver fails
            weights = w_curr if np.sum(np.abs(w_curr)) > 0 else np.ones(n) / n
            
        return pd.Series(weights, index=tickers)
