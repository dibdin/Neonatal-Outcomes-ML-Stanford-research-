from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
import numpy as np
from scipy import stats

def compute_auc(y_true, y_pred):
    """Compute AUC score for binary classification"""
    try:
        auc = roc_auc_score(y_true, y_pred)
        return auc
    except ValueError:
        return np.nan  # AUC is not defined if only one class is present

def compute_rmse(y_true, y_pred):
    """Compute Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_mae(y_true, y_pred):
    """Compute Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)

def compute_confidence_interval(data, confidence=0.95):
    """Compute confidence interval for a list of values"""
    if len(data) < 2:
        return np.mean(data), np.nan, np.nan
    
    mean_val = np.mean(data)
    std_err = stats.sem(data)
    ci_lower, ci_upper = stats.t.interval(confidence, len(data)-1, loc=mean_val, scale=std_err)
    
    return mean_val, ci_lower, ci_upper

def compute_metrics_by_gestational_age(y_true, y_pred, preterm_cutoff=37):
    """Compute MAE and RMSE separately for preterm and term babies"""
    # Convert to binary preterm outcome
    is_preterm = y_true < preterm_cutoff
    
    # Separate data
    preterm_mask = is_preterm
    term_mask = ~is_preterm
    
    # Compute metrics for preterm babies
    preterm_mae = compute_mae(y_true[preterm_mask], y_pred[preterm_mask]) if np.sum(preterm_mask) > 0 else np.nan
    preterm_rmse = compute_rmse(y_true[preterm_mask], y_pred[preterm_mask]) if np.sum(preterm_mask) > 0 else np.nan
    
    # Compute metrics for term babies
    term_mae = compute_mae(y_true[term_mask], y_pred[term_mask]) if np.sum(term_mask) > 0 else np.nan
    term_rmse = compute_rmse(y_true[term_mask], y_pred[term_mask]) if np.sum(term_mask) > 0 else np.nan
    
    # Overall metrics
    overall_mae = compute_mae(y_true, y_pred)
    overall_rmse = compute_rmse(y_true, y_pred)
    
    return {
        'overall': {'mae': overall_mae, 'rmse': overall_rmse},
        'preterm': {'mae': preterm_mae, 'rmse': preterm_rmse, 'count': np.sum(preterm_mask)},
        'term': {'mae': term_mae, 'rmse': term_rmse, 'count': np.sum(term_mask)}
    }

def evaluate_all(y_true, y_pred):
    """Return a dictionary with all metrics"""
    metrics =  {
        "AUC": compute_auc(y_true, y_pred),
        "RMSE": compute_rmse(y_true, y_pred),
        "MAE": compute_mae(y_true, y_pred),
    }
    return metrics

def average_auc(auc):
    avg_auc = np.mean(auc)
    return avg_auc