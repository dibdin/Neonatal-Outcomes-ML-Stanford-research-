"""
Performance metrics and evaluation utilities for gestational age prediction.

This module provides functions for:
- Computing regression metrics (MAE, RMSE)
- Computing classification metrics (AUC)
- Confidence interval calculations
- Gestational age-specific metric computation
- Comprehensive model evaluation

Author: Diba Dindoust
Date: 07/01/2025
"""

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
import numpy as np
from scipy import stats


def compute_auc(y_true, y_pred):
    """
    Compute AUC score for binary classification.
    
    Args:
        y_true (array): True binary labels
        y_pred (array): Predicted probabilities or scores
        
    Returns:
        float: AUC score or np.nan if not defined
    """
    try:
        auc = roc_auc_score(y_true, y_pred)
        return auc
    except ValueError:
        return np.nan  # AUC is not defined if only one class is present


def compute_rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error.
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        
    Returns:
        float: RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_mae(y_true, y_pred):
    """
    Compute Mean Absolute Error.
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        
    Returns:
        float: MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def compute_confidence_interval(data, confidence=0.95):
    """
    Compute confidence interval for a list of values.
    
    Args:
        data (list): List of values
        confidence (float): Confidence level (default: 0.95)
        
    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """
    if len(data) < 2:
        return np.mean(data), np.nan, np.nan
    
    mean_val = np.mean(data)
    std_err = stats.sem(data)
    ci_lower, ci_upper = stats.t.interval(confidence, len(data)-1, loc=mean_val, scale=std_err)
    
    return mean_val, ci_lower, ci_upper


def compute_metrics_by_gestational_age(y_true, y_pred, preterm_cutoff=37):
    """
    Compute MAE and RMSE separately for preterm and term babies.
    
    Args:
        y_true (array): True gestational ages
        y_pred (array): Predicted gestational ages
        preterm_cutoff (int): Gestational age cutoff for preterm classification
        
    Returns:
        dict: Dictionary with metrics for overall, preterm, and term groups
    """
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
    """
    Return a dictionary with all metrics.
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values
        
    Returns:
        dict: Dictionary containing AUC, RMSE, and MAE
    """
    metrics = {
        "AUC": compute_auc(y_true, y_pred),
        "RMSE": compute_rmse(y_true, y_pred),
        "MAE": compute_mae(y_true, y_pred),
    }
    return metrics


def average_auc(auc):
    """
    Compute average AUC from a list of AUC values.
    
    Args:
        auc (list): List of AUC values
        
    Returns:
        float: Average AUC
    """
    avg_auc = np.mean(auc)
    return avg_auc