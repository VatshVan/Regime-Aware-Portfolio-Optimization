import time
import pandas as pd
import numpy as np

from . import config
from . import utils
from . import features
from . import transforms
from . import diagnostics

def run_stage5():
    logger = utils.setup_logging("Stage5")
    logger.info("Starting Stage 5: Feature Engineering")
    
    overall_start = time.perf_counter()
    
    # 1. Loading
    logger.info("Loading Canonical Matrices...")
    close_df = pd.read_parquet(config.CLOSE_MATRIX_PATH)
    high_df = pd.read_parquet(config.HIGH_MATRIX_PATH)
    low_df = pd.read_parquet(config.LOW_MATRIX_PATH)
    open_df = pd.read_parquet(config.OPEN_MATRIX_PATH)
    vol_df = pd.read_parquet(config.VOLUME_MATRIX_PATH)
    macro_df = pd.read_parquet(config.MACRO_MATRIX_PATH)
    
    # Optional loaded features from Stage 3
    logger.info("Loading pre-computed fractional differencing and volatility...")
    eq_frac = pd.read_parquet(config.EQUITY_FRACDIFF_PATH)
    macro_frac = pd.read_parquet(config.MACRO_FRACDIFF_PATH)
    daily_vol = pd.read_parquet(config.DAILY_VOLATILITY_PATH)
    
    # 2. Extract Benchmarks
    benchmarks_to_extract = [
        config.BENCHMARK_SP500, config.BENCHMARK_NASDAQ, config.BENCHMARK_DOLLAR,
        config.BENCHMARK_GOLD, config.BENCHMARK_BRENT, config.BENCHMARK_VIX,
        config.BENCHMARK_TREASURY
    ]
    benchmarks_df = transforms.extract_benchmarks(macro_df, benchmarks_to_extract)
    
    tickers = close_df.columns
    logger.info(f"Generating features for {len(tickers)} equities sequentially...")
    
    all_features = []
    
    # 3. Sequential Feature Generation
    for ticker in tickers:
        c = close_df[ticker]
        h = high_df[ticker]
        l = low_df[ticker]
        o = open_df[ticker]
        v = vol_df[ticker]
        
        # Generates features per ticker
        feat_df = features.generate_all_features(
            ticker, c, h, l, o, v, benchmarks_df, config
        )
        all_features.append(feat_df)
        
        # Explicit memory management
        del c, h, l, o, v, feat_df
        
    utils.clear_memory()
    
    # 4. Finalize Feature Matrix (Long Format)
    logger.info("Concatenating into long-format panel data...")
    feature_matrix = transforms.finalize_feature_matrix(all_features)
    del all_features
    utils.clear_memory()
    
    # 5. Join pre-computed features (Fractional Diff, Daily Vol)
    logger.info("Joining pre-computed Fractional Diff & Volatility...")
    
    # Melt wide matrices to long format before joining
    def melt_wide(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
        df_long = df.stack().reset_index()
        df_long.columns = ['Datetime', 'Ticker', value_name]
        return df_long.set_index(['Datetime', 'Ticker'])
        
    eq_frac_long = melt_wide(eq_frac, 'fracdiff_0.16') # Generic naming or specific
    daily_vol_long = melt_wide(daily_vol, 'daily_volatility')
    
    feature_matrix = feature_matrix.join(eq_frac_long, how='left')
    feature_matrix = feature_matrix.join(daily_vol_long, how='left')
    
    del eq_frac_long, daily_vol_long, eq_frac, daily_vol
    utils.clear_memory()
    
    # 6. Join Macro Features
    # Macro features are just by Datetime, so we join on level=0
    logger.info("Joining Macro features and Macro Fractional Diff...")
    feature_matrix = feature_matrix.join(macro_df, on='Datetime', how='left')
    
    # Rename macro frac columns to avoid collisions
    macro_frac.columns = [f"{c}_fracdiff" for c in macro_frac.columns]
    feature_matrix = feature_matrix.join(macro_frac, on='Datetime', how='left')
    
    del macro_df, macro_frac
    utils.clear_memory()
    
    # 7. Diagnostics
    logger.info("Generating Diagnostics...")
    diagnostics.generate_feature_diagnostics(feature_matrix, config)
    
    # 8. Saving
    logger.info("Saving canonical feature matrix...")
    feature_matrix.to_parquet(config.FEATURE_MATRIX_PATH)
    
    runtime = time.perf_counter() - overall_start
    logger.info(f"Stage 5 completed in {runtime:.2f} seconds.")
    
if __name__ == "__main__":
    run_stage5()
