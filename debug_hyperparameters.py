#!/usr/bin/env python3
"""
Debug script to test hyperparameter selection and see what's in best_params_
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.datasets import make_classification

def test_elasticnet_cv():
    """Test ElasticNet CV to see what's in best_params_"""
    
    # Create synthetic data
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, 
                              n_redundant=2, random_state=42)
    
    print("=== TESTING ELASTICNET CV ===")
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Define the same grid as in src/model.py
    param_grid = {
        'C': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
        'l1_ratio': [0.3, 0.5, 0.7, 0.9]
    }
    
    print(f"Grid C values: {param_grid['C']}")
    print(f"Grid l1_ratio values: {param_grid['l1_ratio']}")
    
    # Create the model
    elasticnet = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=2000,
        random_state=42
    )
    
    # Create CV model
    cv_model = GridSearchCV(
        elasticnet,
        param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    # Fit the model
    cv_model.fit(X, y)
    
    print(f"\nBest score: {cv_model.best_score_:.4f}")
    print(f"Best parameters: {cv_model.best_params_}")
    
    # Check if C=0.0 appears
    if cv_model.best_params_.get('C') == 0.0:
        print("⚠️  WARNING: C=0.0 detected in best_params_!")
    else:
        print(f"✅ C value is {cv_model.best_params_.get('C')} (not 0.0)")
    
    # Check all scores
    print(f"\nAll CV scores:")
    for params, score in zip(cv_model.cv_results_['params'], cv_model.cv_results_['mean_test_score']):
        print(f"  C={params['C']}, l1_ratio={params['l1_ratio']}: {score:.4f}")

def test_lasso_cv():
    """Test Lasso CV to see what's in best_params_"""
    
    # Create synthetic data
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, 
                              n_redundant=2, random_state=42)
    
    print("\n=== TESTING LASSO CV ===")
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Define the same grid as in src/model.py
    param_grid = {
        'C': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    }
    
    print(f"Grid C values: {param_grid['C']}")
    
    # Create the model
    lasso = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=2000,
        random_state=42
    )
    
    # Create CV model
    cv_model = GridSearchCV(
        lasso,
        param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    # Fit the model
    cv_model.fit(X, y)
    
    print(f"\nBest score: {cv_model.best_score_:.4f}")
    print(f"Best parameters: {cv_model.best_params_}")
    
    # Check if C=0.0 appears
    if cv_model.best_params_.get('C') == 0.0:
        print("⚠️  WARNING: C=0.0 detected in best_params_!")
    else:
        print(f"✅ C value is {cv_model.best_params_.get('C')} (not 0.0)")
    
    # Check all scores
    print(f"\nAll CV scores:")
    for params, score in zip(cv_model.cv_results_['params'], cv_model.cv_results_['mean_test_score']):
        print(f"  C={params['C']}: {score:.4f}")

if __name__ == "__main__":
    test_elasticnet_cv()
    test_lasso_cv() 