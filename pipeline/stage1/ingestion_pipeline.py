"""
Stage 2: Ingestion & Canonical Matrix Construction (v2.0)

Features:
- Schema-based dataset detection
- Nuanced macro feature extraction
- Robust MultiIndex & Long-format equity handling
- Enhanced validation (OHLC invariants)
- Advanced missing value classification (Weekends)
- Detailed Metadata & Asset Registries
- Manifest generation for downstream caching
- Memory optimized processing
- Hard/Soft warning separation and detailed diagnostic reports
"""

import os
import sys
import gc
import json
import logging
import hashlib
import warnings
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .. import config

# =============================================================================
# SETUP LOGGING
# =============================================================================
def setup_logging() -> logging.Logger:
    for d in [config.DIAGNOSTICS_DIR, config.DAILY_DIR, config.HOURLY_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        
    logger = logging.getLogger("IngestionPipeline")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        fh = logging.FileHandler(config.LOG_FILE, mode='w')
        fh.setLevel(logging.WARNING)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
    return logger

logger = setup_logging()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_git_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode('utf-8').strip()
    except Exception:
        return "unknown"

def hash_dataframe(df: pd.DataFrame) -> str:
    try:
        h = hashlib.md5()
        h.update(str(df.shape).encode('utf-8'))
        h.update(str(df.columns.tolist()).encode('utf-8'))
        if not df.empty:
            h.update(str(df.index[0]).encode('utf-8'))
            h.update(str(df.index[-1]).encode('utf-8'))
        return h.hexdigest()
    except Exception:
        return "hash_error"

def detect_schema(df: pd.DataFrame) -> str:
    if isinstance(df.columns, pd.MultiIndex):
        return "Equity_Wide_MultiIndex"
        
    cols = [str(c).lower() for c in df.columns]
    
    if "ticker" in cols or "symbol" in cols or "asset" in cols:
        return "Equity_Long"
        
    if all(c in cols for c in ["open", "high", "low", "close"]):
        return "OHLCV"
        
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 1:
        return "Single_Numeric"
    else:
        return "Multi_Numeric"

def get_time_col(df: pd.DataFrame) -> Optional[str]:
    time_cols = [c for c in df.columns if str(c).lower() in ['date', 'datetime', 'time', 'timestamp']]
    return time_cols[0] if time_cols else None

def get_ticker_col(df: pd.DataFrame) -> Optional[str]:
    cols = [c for c in df.columns if str(c).lower() in ['ticker', 'symbol', 'asset']]
    return cols[0] if cols else None

# =============================================================================
# CORE PIPELINE FUNCTIONS
# =============================================================================

def validate(df: pd.DataFrame, name: str, schema: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Validation with sanity checks, separating hard failures from soft anomalies."""
    hard_failures = []
    soft_warnings = []
    total_rows = len(df)
    
    def add_soft(violation_type: str, mask: pd.Series):
        num_violations = mask.sum()
        if num_violations > 0:
            examples = df[mask].index[:5].tolist()
            soft_warnings.append({
                "dataset": name,
                "type": violation_type,
                "rows_checked": total_rows,
                "violations": num_violations,
                "percentage": round(100 * num_violations / max(1, total_rows), 2),
                "examples": [str(x) for x in examples]
            })

    # HARD FAILURES
    if df.columns.duplicated().any():
        hard_failures.append("Duplicate columns found.")
        
    if total_rows <= 1:
        hard_failures.append("Insufficient data (<= 1 row).")

    if schema == "Equity_Long":
        t_col = get_time_col(df)
        sym_col = get_ticker_col(df)
        if t_col and sym_col and df.duplicated(subset=[t_col, sym_col]).any():
            hard_failures.append("Duplicate (Date, Ticker) pairs found.")
    elif schema == "Equity_Wide_MultiIndex":
        if df.index.duplicated().any():
            hard_failures.append("Duplicate timestamps found in index.")
    else:
        t_col = get_time_col(df)
        if t_col:
            if df.duplicated(subset=[t_col]).any():
                hard_failures.append("Duplicate timestamps found.")
        elif isinstance(df.index, pd.DatetimeIndex):
            if df.index.duplicated().any():
                hard_failures.append("Duplicate timestamps in index.")

    # SOFT WARNINGS (OHLC Consistency & Price Sanity)
    if schema in ["Equity_Long", "OHLCV"]:
        col_map = {str(c).lower(): c for c in df.columns}
        if all(x in col_map for x in ["open", "high", "low", "close"]):
            o, h, l, c = col_map['open'], col_map['high'], col_map['low'], col_map['close']
            
            temp = df[[o, h, l, c]].apply(pd.to_numeric, errors='coerce')
            
            mask_high = temp[h] < temp[[o, c]].max(axis=1)
            add_soft("Vendor Inconsistency (High < max(Open, Close))", mask_high)
            
            mask_low = temp[l] > temp[[o, c]].min(axis=1)
            add_soft("Vendor Inconsistency (Low > min(Open, Close))", mask_low)
            
            mask_price = (temp[o] <= 0) | (temp[c] <= 0)
            add_soft("Informational Note (Price <= 0)", mask_price)
            
        if 'volume' in col_map:
            v = col_map['volume']
            temp_v = pd.to_numeric(df[v], errors='coerce')
            mask_vol = (temp_v < 0)
            add_soft("Negative Volume", mask_vol)

    return hard_failures, soft_warnings

def normalize(df: pd.DataFrame, schema: str) -> pd.DataFrame:
    if schema != "Equity_Wide_MultiIndex":
        col_map = {str(c).lower(): c for c in df.columns}
        rename_dict = {}
        for std in config.OHLCV_COLS:
            if std.lower() in col_map:
                rename_dict[col_map[std.lower()]] = std
        if rename_dict:
            df = df.rename(columns=rename_dict)
            
        adj_cols = [c for c in df.columns if 'adj' in str(c).lower()]
        if adj_cols:
            df = df.drop(columns=adj_cols)

    if not isinstance(df.index, pd.DatetimeIndex):
        t_col = get_time_col(df)
        if t_col:
            df[t_col] = pd.to_datetime(df[t_col])
            df = df.set_index(t_col)
            
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize(config.MASTER_TIMEZONE)
        else:
            df.index = df.index.tz_convert(config.MASTER_TIMEZONE)
            
        # Normalize to midnight to ensure perfect cross-dataset alignment
        df.index = df.index.normalize()
        # Remove duplicate index entries caused by daylight saving or bad data
        if schema != "Equity_Long":
            df = df[~df.index.duplicated(keep='last')]
            
    if schema != "Equity_Wide_MultiIndex":
        for col in df.columns:
            if col in ["Open", "High", "Low", "Close"]:
                df[col] = df[col].astype("float32")
            elif col == "Volume":
                try:
                    df[col] = df[col].astype("Int64")
                except:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype("float32")
                    
    return df

def extract_macro_features(df: pd.DataFrame, name: str, schema: str) -> pd.DataFrame:
    extracted = pd.DataFrame(index=df.index)
    
    if schema == "OHLCV":
        if "Close" in df.columns:
            extracted[f"{name}_Close"] = df["Close"]
        else:
            cols = df.select_dtypes(include=[np.number]).columns
            if len(cols) > 0:
                extracted[f"{name}_{cols[0]}"] = df[cols[0]]
                
    elif schema in ["Single_Numeric", "Multi_Numeric"]:
        cols = df.select_dtypes(include=[np.number]).columns
        for c in cols:
            col_name = f"{name}_{c}" if len(cols) > 1 else name
            extracted[col_name] = df[c]
            
    return extracted

def build_equity_matrices_from_long(df: pd.DataFrame, master_calendar: Optional[pd.DatetimeIndex] = None) -> Dict[str, pd.DataFrame]:
    ticker_col = get_ticker_col(df)
    
    # Drop perfectly identical rows first
    df = df.drop_duplicates()
    
    # Check for (Datetime, Ticker) duplicates with differing values
    df_reset = df.reset_index()
    index_col = df.index.name if df.index.name else 'index'
    
    dup_mask = df_reset.duplicated(subset=[index_col, ticker_col], keep=False)
    if dup_mask.any():
        import logging
        logging.getLogger(__name__).warning(f"Found {dup_mask.sum()} duplicate rows with identical (Datetime, Ticker). Aggregating deterministically.")
        
        agg_rules = {}
        for c in df_reset.columns:
            if c in [index_col, ticker_col]:
                continue
            c_lower = str(c).lower()
            if c_lower == 'open':
                agg_rules[c] = 'first'
            elif c_lower == 'high':
                agg_rules[c] = 'max'
            elif c_lower == 'low':
                agg_rules[c] = 'min'
            elif c_lower == 'close':
                agg_rules[c] = 'last'
            elif c_lower == 'volume':
                agg_rules[c] = 'sum'
            else:
                agg_rules[c] = 'last'
                
        df_reset = df_reset.groupby([index_col, ticker_col]).agg(agg_rules).reset_index()
        df = df_reset.set_index(index_col)
        
    matrices = {}
    for feature in config.OHLCV_COLS:
        if feature in df.columns:
            matrix = df.pivot(columns=ticker_col, values=feature)
            if master_calendar is not None:
                matrix = matrix.reindex(master_calendar)
            matrices[feature] = matrix
    return matrices

def build_equity_matrices_from_wide(df: pd.DataFrame, master_calendar: Optional[pd.DatetimeIndex] = None) -> Dict[str, pd.DataFrame]:
    matrices = {}
    levels = df.columns.levels
    price_level = 0
    for i, lvl in enumerate(levels):
        if any(c.lower() == 'close' for c in lvl if isinstance(c, str)):
            price_level = i
            break
            
    for feature in config.OHLCV_COLS:
        actual_feature = next((c for c in levels[price_level] if str(c).lower() == feature.lower()), None)
        if actual_feature is not None:
            if price_level == 0:
                matrix = df[actual_feature]
            else:
                matrix = df.xs(actual_feature, level=price_level, axis=1)
                
            if master_calendar is not None:
                matrix = matrix.reindex(master_calendar)
            matrices[feature] = matrix
            
    return matrices

def analyze_missing_data(df: pd.DataFrame, category: str) -> pd.DataFrame:
    report = []
    for col in df.columns:
        series = df[col]
        total_missing = series.isna().sum()
        
        if total_missing > 0:
            first_valid = series.first_valid_index()
            last_valid = series.last_valid_index()
            
            pre_missing = series.loc[:first_valid].iloc[:-1].isna().sum() if first_valid else 0
            post_missing = series.loc[last_valid:].iloc[1:].isna().sum() if last_valid else 0
            structural_missing = pre_missing + post_missing
            
            internal_missing = total_missing - structural_missing
            
            if first_valid and last_valid:
                active_slice = series.loc[first_valid:last_valid]
                missing_mask = active_slice.isna()
                missing_dates = active_slice[missing_mask].index
                
                weekend_missing = sum(1 for d in missing_dates if d.weekday() >= 5)
                other_internal = internal_missing - weekend_missing
                
                missing_ratio = other_internal / max(1, len(active_slice))
                
                if missing_ratio > 0.5 and category != "Equity":
                    cause = "Release Frequency (e.g. Monthly/Quarterly)"
                elif weekend_missing > other_internal:
                    cause = "Weekends / Non-Trading Days"
                elif category == "Equity" and other_internal > 0:
                    cause = "Different exchange holiday or halt"
                else:
                    cause = "Unexpected Missing"
            else:
                cause = "Structural (Empty)"
                weekend_missing = 0
                other_internal = 0
                
            report.append({
                "Feature": col,
                "Category": category,
                "TotalMissing": total_missing,
                "StructuralMissing": structural_missing,
                "WeekendMissing": weekend_missing,
                "OtherInternal": other_internal,
                "CoveragePct": round(100 * (1 - total_missing / len(df)), 2),
                "ProbableCause": cause
            })
            
    return pd.DataFrame(report)

def run_pipeline():
    logger.info("Starting Ingestion Pipeline v2.0")
    
    if not config.RAW_DIR.exists():
        logger.error(f"Raw directory not found: {config.RAW_DIR}")
        return
        
    dataset_summaries = []
    calendar_reports = []
    all_soft_warnings = []
    manifest = {}
    
    macro_matrices_list = []
    daily_close_matrix = None
    hourly_close_matrix = None
    master_calendar_daily = None
    
    parquet_files = list(config.RAW_DIR.rglob("*.parquet"))
    logger.info(f"Discovered {len(parquet_files)} parquet files.")
    
    for path in tqdm(parquet_files, desc="Processing Datasets"):
        name = path.stem
        
        try:
            df = pd.read_parquet(path)
            schema = detect_schema(df)
            
            # Validate
            hard_failures, soft_warnings = validate(df, name, schema)
            if hard_failures:
                logger.warning(f"[{name}] Hard failures detected, skipping: {hard_failures}")
                continue
                
            all_soft_warnings.extend(soft_warnings)
            
            # Normalize
            df = normalize(df, schema)
            
            # Calendar diagnostic
            t_idx = df.index
            if isinstance(t_idx, pd.DatetimeIndex):
                start = t_idx.min()
                end = t_idx.max()
                trading_days = len(t_idx.unique().normalize())
                
                all_days = pd.date_range(start=start.normalize(), end=end.normalize(), freq='D') if start and end else pd.DatetimeIndex([])
                weekends = sum(1 for d in all_days if d.weekday() >= 5)
                
                weekdays = len(all_days) - weekends
                holidays = max(0, weekdays - trading_days)
                
                calendar_reports.append({
                    "Dataset": name,
                    "Calendar": schema,
                    "Trading Days": trading_days,
                    "Weekends": weekends,
                    "Holiday Count": holidays,
                    "Timezone": str(t_idx.tz) if t_idx.tz else "None",
                    "Start": str(start),
                    "End": str(end)
                })
            
            # Manifest entry
            manifest[name] = {
                "rows": len(df),
                "columns": len(df.columns),
                "hash": hash_dataframe(df),
                "schema": schema,
                "processed_time": datetime.now().isoformat()
            }
            
            # Process and Extract
            if schema in ["Equity_Long", "Equity_Wide_MultiIndex"]:
                freq = "hourly" if "hour" in name.lower() or "intraday" in str(path).lower() else "daily"
                
                if freq == "daily":
                    if master_calendar_daily is None and isinstance(df.index, pd.DatetimeIndex):
                        if schema == "Equity_Long":
                            t_col = get_time_col(df)
                            if t_col:
                                master_calendar_daily = df.index.unique().sort_values()
                            else:
                                master_calendar_daily = df.index.unique().sort_values()
                        else:
                            master_calendar_daily = df.index.sort_values()
                            
                    matrices = build_equity_matrices_from_long(df, master_calendar_daily) if schema == "Equity_Long" else build_equity_matrices_from_wide(df, master_calendar_daily)
                    
                    for feature, mat in matrices.items():
                        out_path = config.DAILY_DIR / f"{feature.lower()}_matrix.parquet"
                        mat.to_parquet(out_path)
                        if feature.lower() == 'close':
                            daily_close_matrix = mat
                            
                else:
                    matrices = build_equity_matrices_from_long(df, None) if schema == "Equity_Long" else build_equity_matrices_from_wide(df, None)
                    for feature, mat in matrices.items():
                        out_path = config.HOURLY_DIR / f"{feature.lower()}_matrix.parquet"
                        mat.to_parquet(out_path)
                        if feature.lower() == 'close':
                            hourly_close_matrix = mat
            else:
                extracted = extract_macro_features(df, name, schema)
                macro_matrices_list.append(extracted)
                
            dataset_summaries.append({
                "Dataset": name,
                "Schema": schema,
                "Rows": len(df),
                "Columns": len(df.columns),
                "StartDate": str(df.index.min()) if isinstance(df.index, pd.DatetimeIndex) else None,
                "EndDate": str(df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else None
            })
            
            del df
            if 'extracted' in locals():
                del extracted
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")
            
    pd.DataFrame(dataset_summaries).to_csv(config.DIAG_SUMMARY, index=False)
    if calendar_reports:
        pd.DataFrame(calendar_reports).to_csv(config.DIAGNOSTICS_DIR / "calendar_report.csv", index=False)
    
    macro_matrix = None
    if macro_matrices_list and master_calendar_daily is not None:
        logger.info("Assembling Master Macro Matrix...")
        aligned = [m.reindex(master_calendar_daily) for m in macro_matrices_list]
        macro_matrix = pd.concat(aligned, axis=1)
        macro_matrix.to_parquet(config.DAILY_DIR / "macro_matrix.parquet")
        
        del aligned
        del macro_matrices_list
        gc.collect()
        
    logger.info("Generating Missing Data Diagnostics...")
    missing_reports = []
    
    if daily_close_matrix is not None:
        missing_reports.append(analyze_missing_data(daily_close_matrix, "Equity"))
    if macro_matrix is not None:
        missing_reports.append(analyze_missing_data(macro_matrix, "Macro"))
        
    if missing_reports:
        missing_df = pd.concat(missing_reports, ignore_index=True)
        missing_df.to_csv(config.DIAG_MISSING, index=False)
        
        coverage_df = missing_df[['Feature', 'Category', 'CoveragePct']].sort_values(by='CoveragePct')
        coverage_df.to_csv(config.DIAG_COVERAGE, index=False)
        
    logger.info("Generating Feature Statistics...")
    if macro_matrix is not None:
        desc = macro_matrix.describe(percentiles=[.05, .95]).T
        desc.index.name = 'Feature'
        desc.reset_index(inplace=True)
        desc.to_csv(config.DIAG_FEATURE_STATS, index=False)
        
    if daily_close_matrix is not None:
        logger.info("Generating Asset Registry...")
        registry = []
        lifetimes = []
        
        constituents = None
        const_path = config.RAW_DIR / "us_stocks" / "constituents.csv"
        if not const_path.exists():
            const_path = config.RAW_DIR / "sp500" / "constituents.csv"
            
        if const_path.exists():
            constituents = pd.read_csv(const_path)
            constituents.columns = [c.lower() for c in constituents.columns]
            
        for ticker in daily_close_matrix.columns:
            series = daily_close_matrix[ticker]
            first_valid = series.first_valid_index()
            last_valid = series.last_valid_index()
            coverage = 100 * (series.notna().sum() / len(series))
            
            asset_info = {
                "Ticker": ticker,
                "AssetType": "Equity",
                "Exchange": "US",
                "Country": "US",
                "Currency": "USD",
                "Frequency": "Daily",
                "Category": "us_stocks",
                "FirstValid": str(first_valid) if first_valid else None,
                "LastValid": str(last_valid) if last_valid else None,
                "CoveragePct": round(coverage, 2),
                "Sector": "Unknown",
                "Industry": "Unknown"
            }
            
            if constituents is not None:
                match = pd.DataFrame()
                if 'symbol' in constituents.columns:
                    match = constituents[constituents['symbol'] == ticker]
                elif 'ticker' in constituents.columns:
                    match = constituents[constituents['ticker'] == ticker]
                    
                if not match.empty:
                    row = match.iloc[0]
                    asset_info["Sector"] = row.get("sector", row.get("gics sector", "Unknown"))
                    asset_info["Industry"] = row.get("industry", row.get("gics sub-industry", "Unknown"))
                    
            registry.append(asset_info)
            lifetimes.append({
                "Ticker": ticker,
                "FirstValid": first_valid,
                "LastValid": last_valid,
                "ActiveDays": series.notna().sum()
            })
            
        registry_df = pd.DataFrame(registry)
        registry_df.to_parquet(config.DAILY_DIR / "asset_registry.parquet")
        pd.DataFrame(lifetimes).to_csv(config.DIAG_ASSET_LIFETIMES, index=False)
        
    logger.info("Generating Validation Report...")
    with open(config.DIAGNOSTICS_DIR / "validation.txt", "w") as f:
        f.write("=== Ingestion Pipeline Validation Report ===\n\n")
        if not all_soft_warnings:
            f.write("No soft warnings detected.\n")
        else:
            # Group into anomalies and notes
            anomalies = [w for w in all_soft_warnings if 'Vendor Inconsistency' in w['type']]
            notes = [w for w in all_soft_warnings if 'Informational' in w['type']]
            others = [w for w in all_soft_warnings if w not in anomalies and w not in notes]
            
            if anomalies:
                f.write("--- Vendor Anomalies ---\n")
                for w in anomalies:
                    f.write(f"Dataset: {w['dataset']}\n")
                    f.write(f"Issue: {w['type']}\n")
                    f.write(f"Rows checked: {w['rows_checked']}\n")
                    f.write(f"Violations: {w['violations']} ({w['percentage']}%)\n")
                    f.write(f"Examples: {', '.join(w['examples'])}\n\n")
                    
            if notes:
                f.write("--- Informational Notes ---\n")
                for w in notes:
                    f.write(f"Dataset: {w['dataset']}\n")
                    f.write(f"Note: {w['type']}\n")
                    f.write(f"Occurrences: {w['violations']} ({w['percentage']}%)\n")
                    f.write(f"Examples: {', '.join(w['examples'])}\n\n")
                    
            if others:
                f.write("--- Other Warnings ---\n")
                for w in others:
                    f.write(f"Dataset: {w['dataset']}\n")
                    f.write(f"Issue: {w['type']}\n")
                    f.write(f"Violations: {w['violations']} ({w['percentage']}%)\n\n")

    logger.info("Generating Metadata and Manifest...")
    with open(config.MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=4)
        
    metadata = {
        "PipelineVersion": config.PIPELINE_VERSION,
        "DataVersion": config.DATA_VERSION,
        "CreationTimestamp": datetime.now().isoformat(),
        "GitHash": get_git_hash(),
        "PythonVersion": sys.version,
        "PandasVersion": pd.__version__,
        "MasterTimezone": config.MASTER_TIMEZONE,
        "NumberEquities": len(daily_close_matrix.columns) if daily_close_matrix is not None else 0,
        "NumberMacroFeatures": len(macro_matrix.columns) if macro_matrix is not None else 0
    }
    with open(config.DAILY_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info("Ingestion Pipeline v2.0 Completed Successfully.")

if __name__ == "__main__":
    run_pipeline()
