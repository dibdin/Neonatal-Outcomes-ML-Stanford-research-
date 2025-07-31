#!/usr/bin/env python3
"""
Test script to verify the improved grid configuration for ElasticNet and Lasso.
"""

import numpy as np
import pandas as pd
from src.data_loader import load_and_process_data, split_data
from src.model import get_classification_model
from sklearn.model_selection import GridSearchCV

def test_improved_grid():
    """Test the improved grid configuration."""
    
    print("=== TESTING IMPROVED GRID CONFIGURATION ===\n")
    
    # Load sample data
    X, y = load_and_process_data('cord', model_type='biomarker', data_option=1, target_type='gestational_age')
    print(f"Data loaded: X shape {X.shape}, y shape {y.shape}")
    
    # Create binary classification target
    from src.config import PRETERM_CUTOFF
    y_binary = (y < PRETERM_CUTOFF).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y_binary, test_size=0.2, random_state=42)
    
    # Test both models
    for model_type in ['elasticnet_cv', 'lasso_cv']:
        print(f"\n--- Testing {model_type} ---")
        
        # Get model
        model, base_estimator = get_classification_model(model_type)
        
        # Check grid parameters
        if hasattr(model, 'param_grid'):
            print(f"Grid parameters: {model.param_grid}")
        else:
            print("Model doesn't have param_grid attribute")
        
        # Check for C=0.0 in the grid
        if hasattr(model, 'param_grid') and 'C' in model.param_grid:
            c_values = model.param_grid['C']
            print(f"C values: {c_values}")
            
            if 0.0 in c_values:
                print("❌ WARNING: C=0.0 found in grid!")
            else:
                print("✅ No C=0.0 found in grid")
            
            # Check for clustering around 0.1-1.0
            low_c_values = [c for c in c_values if 0.05 <= c <= 2.0]
            print(f"Values in 0.05-2.0 range: {low_c_values}")
            print(f"Number of values in range: {len(low_c_values)}")
            
            if len(low_c_values) >= 6:
                print("✅ Good granularity around 0.1-1.0 range")
            else:
                print("⚠️  Limited granularity around 0.1-1.0 range")
        
        # Test training (small subset for speed)
        try:
            # Use smaller subset for testing
            X_train_small = X_train.iloc[:100]
            y_train_small = y_train.iloc[:100]
            
            print(f"Training on subset: {X_train_small.shape}")
            model.fit(X_train_small, y_train_small)
            
            # Check best parameters
            if hasattr(model, 'best_params_'):
                print(f"Best parameters: {model.best_params_}")
                
                # Check for C=0.0 in best parameters
                if 'C' in model.best_params_ and model.best_params_['C'] == 0.0:
                    print("❌ WARNING: Best C is 0.0!")
                else:
                    print("✅ Best C is not 0.0")
            else:
                print("Model doesn't have best_params_ attribute")
                
        except Exception as e:
            print(f"❌ Training error: {e}")
        
        print("✓ Test completed")

if __name__ == "__main__":
    test_improved_grid() 