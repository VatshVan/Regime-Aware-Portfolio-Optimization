from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Inputs
DAILY_DIR = PROCESSED_DIR / "daily"
FEATURES_DIR = PROCESSED_DIR / "features"

CLOSE_MATRIX_PATH = DAILY_DIR / "close_matrix.parquet"
OPEN_MATRIX_PATH = DAILY_DIR / "open_matrix.parquet"
HIGH_MATRIX_PATH = DAILY_DIR / "high_matrix.parquet"
LOW_MATRIX_PATH = DAILY_DIR / "low_matrix.parquet"
VOLUME_MATRIX_PATH = DAILY_DIR / "volume_matrix.parquet"
MACRO_MATRIX_PATH = DAILY_DIR / "macro_matrix.parquet"

EQUITY_FRACDIFF_PATH = FEATURES_DIR / "equity_close_fracdiff.parquet"
MACRO_FRACDIFF_PATH = FEATURES_DIR / "macro_fracdiff.parquet"
DAILY_VOLATILITY_PATH = FEATURES_DIR / "daily_volatility.parquet"

# Outputs
FEATURE_MATRIX_PATH = FEATURES_DIR / "feature_matrix.parquet"
FEATURE_METADATA_PATH = FEATURES_DIR / "feature_metadata.json"
FEATURE_STATISTICS_PATH = FEATURES_DIR / "feature_statistics.csv"
FEATURE_CORRELATIONS_PATH = FEATURES_DIR / "feature_correlations.parquet"

# Feature Parameters
HORIZONS = [5, 10, 20, 60]
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2.0
ATR_WINDOW = 14
VOLUME_MA_WINDOW = 20

# Cross-Asset Benchmarks (if available in macro_matrix)
BENCHMARK_SP500 = "sp500_index" # Will fallback or map based on availability
BENCHMARK_NASDAQ = "nasdaq_index"
BENCHMARK_DOLLAR = "us_dollar_index"
BENCHMARK_GOLD = "gold"
BENCHMARK_BRENT = "brent_crude"
BENCHMARK_VIX = "vix"
BENCHMARK_TREASURY = "treasury_10y"
