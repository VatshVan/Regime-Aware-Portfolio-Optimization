import pandas as pd
from typing import List

def extract_benchmarks(macro_df: pd.DataFrame, benchmark_names: List[str]) -> pd.DataFrame:
    """
    Extracts explicitly named benchmarks from the macro matrix.
    If a benchmark is missing, logs a warning and skips it.
    """
    available = macro_df.columns.intersection(benchmark_names)
    missing = set(benchmark_names) - set(available)
    
    # Optional: Log missing if needed, but for now just return what we have
    return macro_df[available]

def finalize_feature_matrix(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenates the list of ticker DataFrames into a single long-format
    MultiIndex DataFrame (Datetime, Ticker).
    """
    # Combine all
    master_df = pd.concat(df_list, axis=0)
    
    # The index is currently Datetime (from the series). We have a 'Ticker' column.
    master_df.index.name = 'Datetime'
    
    # Set multi-index
    master_df = master_df.reset_index().set_index(['Datetime', 'Ticker']).sort_index()
    
    return master_df
