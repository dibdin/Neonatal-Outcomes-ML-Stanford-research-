import pickle
import numpy as np
import pandas as pd
from src.data_loader import load_and_process_data
from src.model import get_model, train_model, predict_model

def test_hyperparameter_recording():
    """Test that hyperparameters are being recorded correctly for CV models."""
    
    print("=== Testing Hyperparameter Recording ===\n")
    
    # Load sample data
    X, y = load_and_process_data('cord', model_type='biomarker', data_option=1, target_type='gestational_age')
    print(f"Data loaded: X shape {X.shape}, y shape {y.shape}")
    
    # Test both lasso_cv and elasticnet_cv
    for model_type in ['lasso_cv', 'elasticnet_cv']:
        print(f"\n--- Testing {model_type} ---")
        
        # Get model
        model, base_estimator = get_model(model_type)
        print(f"Model type: {type(model)}")
        
        # Split data
        from src.data_loader import split_data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
        
        # Train model
        trained_model, _, selected_features = train_model(X_train, y_train, model, base_estimator)
        
        # Extract hyperparameters
        if hasattr(trained_model, 'best_params_'):
            print(f"Best parameters: {trained_model.best_params_}")
            alpha = trained_model.best_params_.get('alpha')
            l1_ratio = trained_model.best_params_.get('l1_ratio')
            print(f"Optimized alpha: {alpha}")
            print(f"Optimized l1_ratio: {l1_ratio}")
        elif hasattr(trained_model, 'best_estimator_'):
            print(f"Best estimator: {type(trained_model.best_estimator_)}")
            if hasattr(trained_model.best_estimator_, 'alpha'):
                alpha = trained_model.best_estimator_.alpha
                print(f"Optimized alpha: {alpha}")
            if hasattr(trained_model.best_estimator_, 'l1_ratio'):
                l1_ratio = trained_model.best_estimator_.l1_ratio
                print(f"Optimized l1_ratio: {l1_ratio}")
        
        # Make predictions
        y_pred = predict_model(trained_model, X_test, base_estimator)
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}")
        
        print("✓ Test completed successfully")

if __name__ == "__main__":
    test_hyperparameter_recording() 