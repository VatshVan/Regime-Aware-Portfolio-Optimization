import json
import pandas as pd
import numpy as np

def generate_feature_diagnostics(df: pd.DataFrame, config):
    """
    Generates statistics, correlations, and metadata for the final feature matrix.
    """
    stats = []
    
    # Sub-sample for faster correlation & std
    sample_df = df.sample(n=min(100000, len(df)), random_state=42) if len(df) > 100000 else df
    
    for col in df.columns:
        if col == "Ticker" or df[col].dtype == object:
            continue
            
        missing_pct = df[col].isna().mean() * 100.0
        
        try:
            col_std = float(sample_df[col].std())
        except:
            col_std = np.nan
            
        stats.append({
            "Feature": col,
            "Missing %": missing_pct,
            "Constant": col_std == 0.0,
            "Sample_Std": col_std
        })
        
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(config.FEATURE_STATISTICS_PATH, index=False)
    
    # Correlations
    # Drop NaNs to prevent pandas from doing slow pairwise comparisons
    corr_matrix = sample_df.dropna().corr(numeric_only=True)
    corr_matrix.to_parquet(config.FEATURE_CORRELATIONS_PATH)
    
    # Metadata
    metadata = {
        "pipeline_version": "5.0",
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "unique_tickers": len(df.index.get_level_values('Ticker').unique()),
        "features_generated": len(df.columns),
        "horizons": config.HORIZONS
    }
    
    with open(config.FEATURE_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
