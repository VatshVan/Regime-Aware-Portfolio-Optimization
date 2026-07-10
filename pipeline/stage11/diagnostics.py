import logging
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from . import config

logger = logging.getLogger("Stage11.Diagnostics")

def calculate_calibration_scores(meta_predictions_df: pd.DataFrame) -> dict:
    """
    Calculates the Brier Score and Log Loss for the Meta Model's probability predictions.
    A lower Brier score indicates better calibration.
    """
    y_true = meta_predictions_df['Meta_Label'].values
    y_prob = meta_predictions_df['Meta_Prob_1'].values
    
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob))
    auc = float(roc_auc_score(y_true, y_prob))
    
    logger.info(f"Calibration metrics: Brier={brier:.4f}, LogLoss={ll:.4f}, AUC={auc:.4f}")
    
    return {
        "brier_score": brier,
        "log_loss": ll,
        "roc_auc": auc
    }

def calculate_profit_lift(meta_predictions_df: pd.DataFrame) -> dict:
    """
    Calculates the hypothetical profit lift achieved by applying the Meta Label filter.
    Base Hit Rate vs Filtered Hit Rate.
    """
    base_accuracy = meta_predictions_df['Meta_Label'].mean()
    
    # Filter where meta model predicts 1 (Take the bet)
    taken_bets = meta_predictions_df[meta_predictions_df['Meta_Pred'] == 1]
    filtered_accuracy = taken_bets['Meta_Label'].mean() if len(taken_bets) > 0 else 0.0
    
    bet_retention = len(taken_bets) / len(meta_predictions_df)
    
    logger.info(f"Profit Lift: Base Accuracy={base_accuracy:.2%}, Filtered={filtered_accuracy:.2%} (Retained {bet_retention:.2%} of bets)")
    
    return {
        "base_accuracy": float(base_accuracy),
        "filtered_accuracy": float(filtered_accuracy),
        "lift": float(filtered_accuracy - base_accuracy),
        "bet_retention_ratio": float(bet_retention),
        "total_bets_offered": len(meta_predictions_df),
        "total_bets_taken": len(taken_bets)
    }
