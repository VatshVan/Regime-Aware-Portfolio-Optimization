from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PORTFOLIO_DIR = PROCESSED_DIR / "portfolio"

# Ensure output directory exists
PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

# Input Paths from previous stages
CLOSE_MATRIX_PATH = PROCESSED_DIR / "daily" / "close_matrix.parquet"
LABELS_PATH = PROCESSED_DIR / "labels" / "triple_barrier_labels.parquet"
HMM_REGIMES_PATH = PROCESSED_DIR / "regimes" / "regime_labels.parquet"
META_PREDICTIONS_PATH = PROCESSED_DIR / "meta_models" / "meta_predictions.parquet"

# Evaluation Periods
EVALUATION_PERIODS = {
    'full_history': ('2000-01-01', '2026-12-31'),
    'dotcom_gfc': ('2000-01-01', '2010-12-31'),
    'bull_market': ('2010-01-01', '2020-12-31'),
    'recent_market': ('2021-01-01', '2026-12-31')
}

def get_portfolio_paths(run_name: str):
    run_dir = PORTFOLIO_DIR / run_name
    fig_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'WEIGHTS': run_dir / "portfolio_weights.parquet",
        'PORTFOLIO_RETURNS': run_dir / "portfolio_returns.parquet",
        'BENCHMARK_RETURNS': run_dir / "benchmark_returns.parquet",
        'TRANSACTION_LOG': run_dir / "transaction_log.parquet",
        'PERFORMANCE_SUMMARY': run_dir / "performance_summary.csv",
        'METADATA': run_dir / "optimization_metadata.json",
        'FIGURES_DIR': fig_dir
    }

# Comparison Path
COMPARISON_SUMMARY_PATH = PORTFOLIO_DIR / "backtest_comparison.csv"
COMPARISON_FIG_DIR = PORTFOLIO_DIR / "comparison_figures"
COMPARISON_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Rebalancing Config
REBALANCE_FREQ = 'ME' # 'W-FRI' (Weekly), 'ME' (Month-end)
INITIAL_CAPITAL = 100_000_000.0

# Transaction Costs Model
class TransactionCosts:
    COMMISSION_BPS = 1.0     # 1 basis point per trade
    SLIPPAGE_BPS = 5.0       # 5 basis points average slippage
    BID_ASK_SPREAD_BPS = 2.0 # 2 basis points spread cost

# Optimizer Config
OPTIMIZATION_OBJECTIVE = "max_sharpe" # Options: max_sharpe, min_variance, risk_parity
TURNOVER_PENALTY = 0.005 # L1 penalty factor

# Regime Specific Constraints
REGIME_CONSTRAINTS = {
    0: {  # High-Volatility / Crisis
        "name": "Bear/Crisis",
        "max_leverage": 1.0,
        "max_asset_weight": 0.05,
        "cash_allocation": 0.50, # Minimum 50% in cash
        "objective": "min_variance" # Force defensive objective
    },
    1: {  # Normal / Bull
        "name": "Bull",
        "max_leverage": 1.0, # Long only
        "max_asset_weight": 0.10,
        "cash_allocation": 0.0,
        "objective": "max_sharpe"
    }
}
