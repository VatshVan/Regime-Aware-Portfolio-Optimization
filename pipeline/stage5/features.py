import numpy as np
import pandas as pd
from typing import Dict

def compute_price_features(c: pd.Series, h: pd.Series, l: pd.Series, horizons: list) -> pd.DataFrame:
    features = {}
    
    # Log Returns
    features['log_ret_1d'] = np.log(c / c.shift(1))
    
    # Multi-Horizon Returns
    for h_days in horizons:
        features[f'log_ret_{h_days}d'] = np.log(c / c.shift(h_days))
        
        # Rolling High/Low/Range
        r_high = h.rolling(h_days).max()
        r_low = l.rolling(h_days).min()
        features[f'rolling_high_{h_days}d'] = r_high / c - 1
        features[f'rolling_low_{h_days}d'] = r_low / c - 1
        features[f'rolling_range_{h_days}d'] = (r_high - r_low) / c
        
        # Price Breakout
        features[f'breakout_{h_days}d'] = (c - r_high.shift(1)) / c
        
    return pd.DataFrame(features, index=c.index)

def compute_momentum_features(c: pd.Series, horizons: list) -> pd.DataFrame:
    features = {}
    
    for h_days in horizons:
        # Rate of Change (Simple Momentum)
        features[f'roc_{h_days}d'] = c / c.shift(h_days) - 1
        
        # Price Relative to Moving Average
        sma = c.rolling(h_days).mean()
        features[f'price_to_sma_{h_days}d'] = c / sma - 1
        
    # Moving Average Crossovers
    if 5 in horizons and 20 in horizons:
        sma_5 = c.rolling(5).mean()
        sma_20 = c.rolling(20).mean()
        features['sma_cross_5_20'] = sma_5 / sma_20 - 1
        
    return pd.DataFrame(features, index=c.index)

def compute_volatility_features(c: pd.Series, h: pd.Series, l: pd.Series, horizons: list, atr_win: int) -> pd.DataFrame:
    features = {}
    
    log_ret = np.log(c / c.shift(1))
    
    for h_days in horizons:
        # Realized Volatility (Rolling Std of returns)
        features[f'realized_vol_{h_days}d'] = log_ret.rolling(h_days).std() * np.sqrt(252)
        
        # EWMA Volatility
        features[f'ewma_vol_{h_days}d'] = log_ret.ewm(span=h_days).std() * np.sqrt(252)
        
    # True Range for ATR
    prev_c = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(atr_win).mean()
    features[f'atr_{atr_win}d_norm'] = atr / c
    
    return pd.DataFrame(features, index=c.index)

def compute_trend_features(c: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    features = {}
    
    ema_fast = c.ewm(span=fast, adjust=False).mean()
    ema_slow = c.ewm(span=slow, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    
    # Normalize by price to make it stationary-ish
    features['macd'] = macd / c
    features['macd_signal'] = macd_signal / c
    features['macd_hist'] = macd_hist / c
    
    return pd.DataFrame(features, index=c.index)

def compute_volume_features(v: pd.Series, w: int) -> pd.DataFrame:
    features = {}
    
    v_ma = v.rolling(w).mean()
    v_std = v.rolling(w).std()
    
    features[f'vol_ma_{w}d'] = v_ma
    features[f'rel_vol_{w}d'] = v / v_ma
    features[f'vol_zscore_{w}d'] = (v - v_ma) / v_std
    
    if w >= 20:
        v_ma_short = v.rolling(5).mean()
        features['vol_momentum'] = v_ma_short / v_ma
        
    return pd.DataFrame(features, index=v.index)

def compute_cross_asset_features(c: pd.Series, benchmarks: pd.DataFrame) -> pd.DataFrame:
    features = {}
    
    # Relative Strength (Ratio of log returns or simply normalized price ratios)
    # Price ratio normalized: (c / c_shift) - (bench / bench_shift)
    ret_1d = c / c.shift(1) - 1
    
    for col in benchmarks.columns:
        b_series = benchmarks[col]
        b_ret = b_series / b_series.shift(1) - 1
        
        # Relative return
        features[f'rel_ret_{col}'] = ret_1d - b_ret
        
        # 20d Relative Strength
        features[f'rs_{col}_20d'] = (c / c.shift(20)) / (b_series / b_series.shift(20)) - 1
        
    return pd.DataFrame(features, index=c.index)

def generate_all_features(
    ticker: str, 
    c: pd.Series, h: pd.Series, l: pd.Series, o: pd.Series, v: pd.Series, 
    benchmarks: pd.DataFrame, 
    config
) -> pd.DataFrame:
    """Orchestrates all feature generations for a single asset."""
    
    dfs = []
    
    dfs.append(compute_price_features(c, h, l, config.HORIZONS))
    dfs.append(compute_momentum_features(c, config.HORIZONS))
    dfs.append(compute_volatility_features(c, h, l, config.HORIZONS, config.ATR_WINDOW))
    dfs.append(compute_trend_features(c, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL))
    dfs.append(compute_volume_features(v, config.VOLUME_MA_WINDOW))
    dfs.append(compute_cross_asset_features(c, benchmarks))
    
    # Combine
    out = pd.concat(dfs, axis=1)
    
    # Add Ticker identifier for the long format
    out['Ticker'] = ticker
    
    # Drop rows where everything is NaN (very early history)
    # We will keep rows that have at least some features, but the first 60 days will have many NaNs.
    # We do NOT drop NaNs here because ML models (like XGBoost) can handle them, and we need to keep time indices intact.
    
    return out
