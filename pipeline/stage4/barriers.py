import numpy as np
from numba import njit

# Collision Policies Map:
# 1 = conservative (SL first)
# 2 = optimistic (PT first)
# 3 = ignore (treat as vertical)

@njit(cache=True)
def triple_barrier_kernel(
    event_indices: np.ndarray,
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    volatilities: np.ndarray,
    pt_multi: float,
    sl_multi: float,
    v_days: int,
    collision_policy: int
):
    """
    Highly optimized Numba kernel for evaluating Triple Barrier paths.
    Evaluates one asset's events sequentially.
    """
    n_events = len(event_indices)
    max_idx = len(close_prices) - 1
    
    out_exit_idx = np.zeros(n_events, dtype=np.int32)
    out_exit_price = np.zeros(n_events, dtype=np.float64)
    out_pt = np.zeros(n_events, dtype=np.float64)
    out_sl = np.zeros(n_events, dtype=np.float64)
    out_hit = np.zeros(n_events, dtype=np.int8)  # 1=PT, 2=SL, 3=VERT
    
    for i in range(n_events):
        e_idx = event_indices[i]
        
        # Ensure we don't exceed array bounds
        if e_idx >= max_idx:
            # Event is exactly at the end of the series, immediate vertical hit
            e_price = close_prices[e_idx]
            vol = volatilities[e_idx]
            out_exit_idx[i] = e_idx
            out_exit_price[i] = e_price
            out_pt[i] = e_price * (1.0 + pt_multi * vol) if vol > 0 else np.nan
            out_sl[i] = e_price * (1.0 - sl_multi * vol) if vol > 0 else np.nan
            out_hit[i] = 3
            continue
            
        e_price = close_prices[e_idx]
        vol = volatilities[e_idx]
        
        # If volatility is invalid, skip or just trigger vertical immediately
        if np.isnan(vol) or vol <= 0 or np.isnan(e_price) or e_price <= 0:
            out_exit_idx[i] = e_idx
            out_exit_price[i] = e_price
            out_pt[i] = np.nan
            out_sl[i] = np.nan
            out_hit[i] = 3
            continue
            
        pt = e_price * (1.0 + pt_multi * vol)
        sl = e_price * (1.0 - sl_multi * vol)
        
        out_pt[i] = pt
        out_sl[i] = sl
        
        v_idx = min(e_idx + v_days, max_idx)
        
        hit_type = 3  # Default to Vertical
        exit_idx = v_idx
        exit_price = close_prices[v_idx]
        
        # Evaluate forward path
        for t in range(e_idx + 1, v_idx + 1):
            c_t = close_prices[t]
            h_t = high_prices[t] if not np.isnan(high_prices[t]) else c_t
            l_t = low_prices[t] if not np.isnan(low_prices[t]) else c_t
            
            # Check barrier breaches
            hit_pt = h_t >= pt
            hit_sl = l_t <= sl
            
            if hit_pt and hit_sl:
                if collision_policy == 1:
                    hit_type = 2 # Conservative (SL)
                    exit_idx = t
                    exit_price = sl
                    break
                elif collision_policy == 2:
                    hit_type = 1 # Optimistic (PT)
                    exit_idx = t
                    exit_price = pt
                    break
                elif collision_policy == 3:
                    # Ignore implies we don't break, continue evaluating
                    pass
                else:
                    # Raise / undefined
                    hit_type = 2
                    exit_idx = t
                    exit_price = sl
                    break
            elif hit_pt:
                hit_type = 1
                exit_idx = t
                exit_price = pt
                break
            elif hit_sl:
                hit_type = 2
                exit_idx = t
                exit_price = sl
                break
                
        out_hit[i] = hit_type
        out_exit_idx[i] = exit_idx
        out_exit_price[i] = exit_price
        
    return out_exit_idx, out_exit_price, out_pt, out_sl, out_hit
