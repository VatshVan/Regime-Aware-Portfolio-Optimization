import pandas as pd
import numpy as np

def calculate_cagr(equity_series: pd.Series) -> float:
    if len(equity_series) < 2: return 0.0
    years = (equity_series.index[-1] - equity_series.index[0]).days / 365.25
    if years <= 0: return 0.0
    return (equity_series.iloc[-1] / equity_series.iloc[0]) ** (1 / years) - 1.0

def calculate_drawdowns(equity_series: pd.Series) -> pd.Series:
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    return drawdown

def calculate_metrics(equity_df: pd.DataFrame, turnover_df: pd.DataFrame, benchmark_df: pd.DataFrame = None) -> dict:
    """
    Computes strict institutional performance metrics.
    """
    gross_eq = equity_df['Gross_Value']
    net_eq = equity_df['Net_Value']
    
    # Returns
    gross_returns = gross_eq.pct_change().dropna()
    net_returns = net_eq.pct_change().dropna()
    
    # CAGRs
    gross_cagr = calculate_cagr(gross_eq)
    net_cagr = calculate_cagr(net_eq)
    cost_drag = gross_cagr - net_cagr
    
    # Risk
    annual_vol = net_returns.std() * np.sqrt(252)
    sharpe = (net_returns.mean() * 252) / (annual_vol if annual_vol > 0 else 1e-6)
    
    # Sortino (downside risk)
    downside_returns = net_returns[net_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = (net_returns.mean() * 252) / (downside_vol if downside_vol > 0 else 1e-6)
    
    # Max Drawdown
    drawdowns = calculate_drawdowns(net_eq)
    max_dd = drawdowns.min()
    calmar = net_cagr / abs(max_dd) if max_dd < 0 else np.nan
    
    # Turnover Costs
    if not turnover_df.empty:
        total_costs = turnover_df['Total_Cost'].sum()
        annual_turnover = turnover_df['Turnover'].mean() * (252 / (turnover_df.index.to_series().diff().dt.days.mean() or 1))
        avg_turnover = turnover_df['Turnover'].mean()
        avg_holding_period = 252.0 / annual_turnover if annual_turnover > 0 else np.nan
    else:
        total_costs = 0.0
        annual_turnover = 0.0
        avg_turnover = 0.0
        avg_holding_period = np.nan
        
    # Alpha, Beta, Information Ratio
    alpha, beta, ir = np.nan, np.nan, np.nan
    if benchmark_df is not None and not benchmark_df.empty:
        # We will use Equal_Weight as the market proxy for Alpha/Beta since it contains exactly the universe
        if 'Equal_Weight' in benchmark_df.columns:
            bench_ret = benchmark_df['Equal_Weight'].pct_change().dropna()
            # Align dates
            aligned = pd.concat([net_returns, bench_ret], axis=1).dropna()
            if len(aligned) > 2:
                port_r = aligned.iloc[:, 0]
                bench_r = aligned.iloc[:, 1]
                cov = np.cov(port_r, bench_r)[0, 1]
                var = np.var(bench_r)
                beta = cov / var if var > 0 else np.nan
                alpha = (port_r.mean() - beta * bench_r.mean()) * 252
                
                # IR
                tracking_error = (port_r - bench_r).std() * np.sqrt(252)
                ir = (net_cagr - calculate_cagr(benchmark_df['Equal_Weight'])) / tracking_error if tracking_error > 0 else np.nan
                
    # Win Rate (Days with positive returns)
    win_rate = (net_returns > 0).mean()
        
    return {
        "Gross_CAGR": gross_cagr,
        "Net_CAGR": net_cagr,
        "Cost_Drag_CAGR": cost_drag,
        "Annual_Return": net_cagr,
        "Annual_Volatility": annual_vol,
        "Sharpe_Ratio": sharpe,
        "Sortino_Ratio": sortino,
        "Max_Drawdown": max_dd,
        "Calmar_Ratio": calmar,
        "Information_Ratio": ir,
        "Alpha": alpha,
        "Beta": beta,
        "Total_Transaction_Costs": total_costs,
        "Annual_Turnover": annual_turnover,
        "Average_Rebalance_Turnover": avg_turnover,
        "Win_Rate": win_rate,
        "Average_Holding_Period": avg_holding_period
    }

def calculate_benchmark_returns(
    close_df: pd.DataFrame, 
    start_date: pd.Timestamp, 
    end_date: pd.Timestamp,
    macro_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Generates returns for standard benchmarks: S&P500, Equal Weight, Risk Parity, 60/40, Min Variance.
    Simplified approximations for the demonstration.
    """
    close_idx_tz_naive = close_df.index.tz_localize(None)
    trading_days = close_idx_tz_naive[(close_idx_tz_naive >= start_date) & (close_idx_tz_naive <= end_date)]
    
    # We must index close_df with its original timezone index, but assign naive back
    mask = (close_idx_tz_naive >= start_date) & (close_idx_tz_naive <= end_date)
    asset_returns = close_df.loc[mask].pct_change().fillna(0.0)
    asset_returns.index = asset_returns.index.tz_localize(None)
    
    benchmarks = pd.DataFrame(index=trading_days)
    
    # 1. Equal Weight
    benchmarks['Equal_Weight'] = asset_returns.mean(axis=1)
    
    # 2. Minimum Variance Proxy (Inversely proportional to variance)
    var = asset_returns.var()
    inv_var = 1.0 / var
    inv_var = inv_var / inv_var.sum()
    benchmarks['Min_Variance'] = (asset_returns * inv_var).sum(axis=1)
    
    # 3. Risk Parity Proxy (Inversely proportional to standard deviation)
    std = asset_returns.std()
    inv_std = 1.0 / std
    inv_std = inv_std / inv_std.sum()
    benchmarks['Risk_Parity'] = (asset_returns * inv_std).sum(axis=1)
    
    # Convert returns to equity curves
    for c in benchmarks.columns:
        benchmarks[c] = (1.0 + benchmarks[c]).cumprod()
        
    return benchmarks
