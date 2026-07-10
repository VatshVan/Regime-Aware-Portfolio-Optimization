import time
import pandas as pd
import json
import logging

from . import config
from . import backtest
from . import performance
from . import visualization

def run_stage12():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - Stage12 - %(levelname)s - %(message)s")
    logger = logging.getLogger("Stage12")
    logger.info("Starting Stage 12: Dynamic Portfolio Optimization & Backtest")
    
    overall_start = time.perf_counter()
    
    # 1. Loading Data
    logger.info("Loading required datasets...")
    close_df = pd.read_parquet(config.CLOSE_MATRIX_PATH)
    labels_df = pd.read_parquet(config.LABELS_PATH)
    if 'EventTime' in labels_df.columns:
        labels_df['EventTime'] = pd.to_datetime(labels_df['EventTime']).dt.tz_localize(None)
    regimes_df = pd.read_parquet(config.HMM_REGIMES_PATH)
    regimes_df.index = regimes_df.index.tz_localize(None)
    
    # Load out-of-sample predictions
    meta_predictions = pd.read_parquet(config.META_PREDICTIONS_PATH)
    
    # Primary predictions path
    try:
        primary_predictions = pd.read_parquet(config.PROCESSED_DIR / "models" / "predictions.parquet")
    except Exception as e:
        logger.error(f"Failed to load primary predictions: {e}")
        return
        
    # We need macro_df for benchmarks (it doesn't have SP500, but we handle that in performance.py)
    try:
        macro_df = pd.read_parquet(config.PROCESSED_DIR / "daily" / "macro_matrix.parquet")
    except Exception:
        macro_df = None
        
    all_metrics = []
    
    # 2. Run Expanding Window Backtest for Each Evaluation Period
    for run_name, (start_dt, end_dt) in config.EVALUATION_PERIODS.items():
        logger.info(f"=== Starting Evaluation Period: {run_name} ({start_dt} to {end_dt}) ===")
        
        # Determine paths
        paths = config.get_portfolio_paths(run_name)
        
        # Filter predictions for the testing window
        # For the backtester to work correctly, it needs the predictions *within* the evaluation window.
        period_start = pd.to_datetime(start_dt)
        period_end = pd.to_datetime(end_dt)
        
        period_meta_predictions = meta_predictions[
            (pd.to_datetime(meta_predictions['Datetime']) >= period_start) & 
            (pd.to_datetime(meta_predictions['Datetime']) <= period_end)
        ]
        
        period_primary_predictions = primary_predictions[
            (pd.to_datetime(primary_predictions['Datetime']) >= period_start) & 
            (pd.to_datetime(primary_predictions['Datetime']) <= period_end)
        ]
        
        if period_meta_predictions.empty:
            logger.warning(f"No predictions found for {run_name}. Skipping.")
            continue
            
        logger.info(f"Initializing Walk-Forward Backtester for {run_name}...")
        bt = backtest.WalkForwardBacktester(
            close_df=close_df,
            primary_predictions=period_primary_predictions,
            meta_predictions=period_meta_predictions,
            labels_df=labels_df,
            regimes_df=regimes_df,
            config=config
        )
        
        logger.info(f"Executing Backtest for {run_name}...")
        equity_df, turnover_df, weights_df = bt.run()
        
        if equity_df.empty:
            logger.error(f"Backtest yielded empty equity curve for {run_name}. Skipping.")
            continue
            
        actual_start_date = equity_df.index[0]
        actual_end_date = equity_df.index[-1]
        
        # Calculate Benchmarks specifically over this period
        logger.info(f"Calculating Benchmark Returns for {run_name}...")
        benchmark_df = performance.calculate_benchmark_returns(close_df, actual_start_date, actual_end_date, macro_df)
        
        # Calculate Metrics
        logger.info(f"Computing Performance Metrics for {run_name}...")
        metrics = performance.calculate_metrics(equity_df, turnover_df, benchmark_df)
        
        # Save metrics for comparison
        metrics['Evaluation_Period'] = run_name
        all_metrics.append(metrics)
        
        # Log metrics
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"{k}: {v:.4f}")
            else:
                logger.info(f"{k}: {v}")
                
        # Visualizations
        logger.info(f"Generating Visualizations for {run_name}...")
        fig_dir = paths['FIGURES_DIR']
        visualization.plot_equity_curve(equity_df, benchmark_df, save_path=str(fig_dir / "equity_curve.png"))
        visualization.plot_drawdowns(equity_df, save_path=str(fig_dir / "drawdowns.png"))
        visualization.plot_rolling_metrics(equity_df, save_path=str(fig_dir / "rolling_metrics.png"))
        visualization.plot_regime_overlay(equity_df, regimes_df, save_path=str(fig_dir / "regime_overlay.png"))
        
        # Saving Outputs
        logger.info(f"Saving processed portfolio outputs for {run_name}...")
        if not weights_df.empty:
            weights_df.to_parquet(paths['WEIGHTS'])
        equity_df.to_parquet(paths['PORTFOLIO_RETURNS'])
        benchmark_df.to_parquet(paths['BENCHMARK_RETURNS'])
        if not turnover_df.empty:
            turnover_df.to_parquet(paths['TRANSACTION_LOG'])
            
        pd.Series(metrics).to_csv(paths['PERFORMANCE_SUMMARY'])
        
        metadata = {
            "start_date": str(actual_start_date),
            "end_date": str(actual_end_date),
            "rebalance_freq": config.REBALANCE_FREQ,
            "objective": config.OPTIMIZATION_OBJECTIVE,
            "commission_bps": config.TransactionCosts.COMMISSION_BPS,
            "slippage_bps": config.TransactionCosts.SLIPPAGE_BPS
        }
        with open(paths['METADATA'], 'w') as f:
            json.dump(metadata, f, indent=4)
            
    # 7. Comparison Report
    if all_metrics:
        logger.info("Generating Cross-Period Comparison Report...")
        comparison_df = pd.DataFrame(all_metrics).set_index('Evaluation_Period')
        comparison_df.to_csv(config.COMPARISON_SUMMARY_PATH)
        
        # Generate summary visualizations across all periods
        visualization.plot_comparison_bar_charts(comparison_df, config.COMPARISON_FIG_DIR)
        
    runtime = time.perf_counter() - overall_start
    logger.info(f"Stage 12 completed in {runtime:.2f} seconds.")
    
if __name__ == "__main__":
    run_stage12()
