import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("Stage8.HMM")

def _get_canonical_mapping(model: GaussianHMM) -> np.ndarray:
    """
    Solves the label switching problem by sorting states based on their 
    overall variance (trace of covariance matrix).
    Returns an array where mapped_state = mapping[original_state]
    State 0: Low Volatility
    State 1: Medium Volatility
    State 2: High Volatility
    """
    # model.covars_ shape is (n_states, n_features, n_features)
    state_vars = np.array([np.trace(cov) for cov in model.covars_])
    
    # argsort gives the indices that would sort the array
    # e.g., if vars are [10, 2, 5], argsort is [1, 2, 0]
    # This means state 1 is the lowest, state 2 is middle, state 0 is highest.
    # So canonical state for 1 is 0, for 2 is 1, for 0 is 2.
    order = np.argsort(state_vars)
    
    mapping = np.zeros(model.n_components, dtype=int)
    for canonical_idx, original_idx in enumerate(order):
        mapping[original_idx] = canonical_idx
        
    return mapping

def generate_expanding_regimes(macro_df: pd.DataFrame, config) -> tuple:
    """
    Generates strictly out-of-sample regime probabilities using an expanding window HMM.
    Returns (regime_probs_df, final_model)
    """
    # 1. Prepare Features
    # Fill any missing macro data with forward fill
    features_df = macro_df[config.HMM_FEATURES].ffill()
    
    # Convert S&P 500 price to returns for stationarity
    features_df['sp500_Close'] = features_df['sp500_Close'].pct_change()
    features_df = features_df.dropna()
    
    if len(features_df) < config.EXPANDING_WINDOW_MIN:
        raise ValueError("Not enough data to support the expanding window minimum.")
        
    dates = features_df.index.values
    X_raw = features_df.values
    
    n_samples = len(X_raw)
    n_states = config.N_STATES
    
    # Store out-of-sample probabilities
    # Shape: (n_samples, n_states)
    oos_probs = np.full((n_samples, n_states), np.nan)
    
    current_model = None
    canonical_mapping = None
    
    logger.info(f"Starting expanding window HMM over {n_samples} days...")
    
    # Standardize globally? No, standardize on expanding window to prevent lookahead!
    # But for simplicity and speed, we will maintain an expanding scaler.
    
    for t in range(config.EXPANDING_WINDOW_MIN, n_samples):
        
        # Check if we need to retrain
        if t == config.EXPANDING_WINDOW_MIN or t % config.RETRAIN_FREQUENCY == 0:
            # Fit on [0 : t]
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_raw[:t])
            
            model = GaussianHMM(
                n_components=n_states, 
                covariance_type="full", 
                n_iter=100,
                random_state=config.RANDOM_SEED
            )
            model.fit(X_train)
            current_model = model
            canonical_mapping = _get_canonical_mapping(model)
            
        # Predict exactly for day t (using information strictly up to t)
        # We pass X[0 : t+1] and extract the probability of the *last* state
        # Note: X_train scaler must be applied to X[0:t+1]
        X_eval = scaler.transform(X_raw[:t+1])
        
        # predict_proba returns (n_samples, n_states)
        probs = current_model.predict_proba(X_eval)
        last_prob = probs[-1]
        
        # Map to canonical states
        mapped_prob = np.zeros(n_states)
        for original_idx in range(n_states):
            c_idx = canonical_mapping[original_idx]
            mapped_prob[c_idx] = last_prob[original_idx]
            
        oos_probs[t] = mapped_prob
        
        if t % 1000 == 0:
            logger.info(f"  Processed {t} / {n_samples} days...")

    # Create Output DataFrame
    cols = [f"Regime_{i}_Prob" for i in range(n_states)]
    probs_df = pd.DataFrame(oos_probs, index=features_df.index, columns=cols)
    
    # Drop rows where we didn't predict (the initial window)
    probs_df = probs_df.dropna(how='all')
    
    # Assign discrete label as the argmax
    probs_df['Regime_Label'] = probs_df[cols].values.argmax(axis=1)
    
    return probs_df, current_model, scaler
