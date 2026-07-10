from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

def get_model(model_name: str, random_seed: int = 42):
    """
    Returns an uninitialized instance of the requested model,
    configured for multi-class classification and memory efficiency.
    """
    if model_name == "logistic":
        # Logistic Regression (L2 penalty)
        return LogisticRegression(
            penalty='l2', 
            C=1.0, 
            solver='saga', 
            max_iter=1000, 
            multi_class='multinomial', 
            random_state=random_seed,
            n_jobs=-1
        )
        
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=50,
            random_state=random_seed,
            n_jobs=-1,
            class_weight='balanced'
        )
        
    elif model_name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            tree_method='hist',  # Fast histogram optimized
            random_state=random_seed,
            n_jobs=-1,
            objective='multi:softprob'
        )
        
    elif model_name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_seed,
            n_jobs=-1,
            objective='multiclass',
            verbose=-1
        )
        
    elif model_name == "catboost":
        return CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=random_seed,
            thread_count=-1,
            loss_function='MultiClass',
            verbose=False
        )
        
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
