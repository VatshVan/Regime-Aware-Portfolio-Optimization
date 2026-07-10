import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')

def plot_equity_curve(equity_df: pd.DataFrame, benchmark_df: pd.DataFrame, save_path: str = None):
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df.index, equity_df['Gross_Value'] / equity_df['Gross_Value'].iloc[0], label='Gross Return', linestyle='--')
    plt.plot(equity_df.index, equity_df['Net_Value'] / equity_df['Net_Value'].iloc[0], label='Net Return', linewidth=2)
    
    for c in benchmark_df.columns:
        plt.plot(benchmark_df.index, benchmark_df[c], label=c, alpha=0.6)
        
    plt.title("Portfolio Equity Curve vs Benchmarks")
    plt.yscale('log')
    plt.ylabel("Cumulative Return (Log Scale)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.close()

def plot_drawdowns(equity_df: pd.DataFrame, save_path: str = None):
    net_eq = equity_df['Net_Value']
    roll_max = net_eq.cummax()
    drawdown = (net_eq - roll_max) / roll_max
    
    plt.figure(figsize=(12, 4))
    plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    plt.title("Portfolio Drawdown")
    plt.ylabel("Drawdown %")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.close()

def plot_rolling_metrics(equity_df: pd.DataFrame, window: int = 126, save_path: str = None):
    returns = equity_df['Net_Value'].pct_change().dropna()
    
    roll_ann_ret = returns.rolling(window).mean() * 252
    roll_ann_vol = returns.rolling(window).std() * np.sqrt(252)
    roll_sharpe = roll_ann_ret / (roll_ann_vol + 1e-6)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    axes[0].plot(roll_ann_ret.index, roll_ann_ret, color='blue')
    axes[0].set_title(f"Rolling {window}-Day Annualized Return")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(roll_ann_vol.index, roll_ann_vol, color='orange')
    axes[1].set_title(f"Rolling {window}-Day Annualized Volatility")
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(roll_sharpe.index, roll_sharpe, color='green')
    axes[2].set_title(f"Rolling {window}-Day Sharpe Ratio")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.close()
    
def plot_regime_overlay(equity_df: pd.DataFrame, regimes_df: pd.DataFrame, save_path: str = None):
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df.index, equity_df['Net_Value'] / equity_df['Net_Value'].iloc[0], color='black', label='Net Return')
    
    # Overlay Regimes (Assuming 0 is Crisis/Bear, 1 is Bull)
    # Reindex regimes to match equity index
    aligned_regimes = regimes_df.reindex(equity_df.index, method='ffill')
    
    if 'Regime_Label' in aligned_regimes.columns:
        plt.fill_between(aligned_regimes.index, 0, 1, where=(aligned_regimes['Regime_Label'] == 0), 
                         color='red', alpha=0.2, transform=plt.gca().get_xaxis_transform(), label='Crisis/Bear Regime')
        plt.fill_between(aligned_regimes.index, 0, 1, where=(aligned_regimes['Regime_Label'] == 1), 
                         color='green', alpha=0.1, transform=plt.gca().get_xaxis_transform(), label='Bull Regime')
                         
    plt.title("Portfolio Performance with HMM Regime Overlay")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.close()

def plot_comparison_bar_charts(comparison_df: pd.DataFrame, save_dir: str):
    import os
    from pathlib import Path
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_to_plot = ['Net_CAGR', 'Sharpe_Ratio', 'Max_Drawdown', 'Annual_Turnover', 'Alpha']
    
    for metric in metrics_to_plot:
        if metric in comparison_df.columns:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=comparison_df.index, y=comparison_df[metric], palette='viridis')
            plt.title(f"Comparison of {metric} Across Evaluation Periods")
            plt.ylabel(metric)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(save_dir / f"comparison_{metric}.png")
            plt.close()
