#!/usr/bin/env python3
"""
Test ElasticNetCV with cleaned feature set
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from src.data_loader import load_and_process_data
import warnings
warnings.filterwarnings('ignore')

def test_elasticnet_cv():
    """Test ElasticNetCV with cleaned features"""
    
    print("=== ELASTICNET CV TEST WITH CLEANED FEATURES ===\n")
    
    # Load cleaned data
    print("🔍 Loading cleaned biomarker data...")
    X, y, data_df = load_and_process_data(
        dataset_type='cord',
        model_type='biomarker',
        data_option=1,
        dropna=False,
        random_state=48,
        target_type='gestational_age',
        return_dataframe=True
    )
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    
    # Step 2: Fit ElasticNetCV with cross-validation
    print("\n=== STEP 2: FITTING ELASTICNET CV ===")
    
    # Use ElasticNetCV with expanded l1_ratio grid
    model = ElasticNetCV(
        cv=3,
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        random_state=42,
        max_iter=5000,
        fit_intercept=True,
        n_jobs=-1
    )
    
    print("Fitting ElasticNetCV model...")
    model.fit(X, y)
    
    # Step 3: Evaluate model output
    print("\n=== STEP 3: MODEL EVALUATION ===")
    
    # Print selected features and non-zero coefficients
    print("\n📊 Selected Features and Coefficients:")
    feature_coefs = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    })
    
    # Sort by absolute coefficient value
    feature_coefs = feature_coefs.sort_values('abs_coefficient', ascending=False)
    
    # Show non-zero coefficients
    non_zero_features = feature_coefs[feature_coefs['abs_coefficient'] > 0]
    print(f"\nSelected features (non-zero coefficients): {len(non_zero_features)}")
    for _, row in non_zero_features.head(20).iterrows():
        print(f"  {row['feature']}: {row['coefficient']:.4f}")
    
    if len(non_zero_features) > 20:
        print(f"  ... and {len(non_zero_features) - 20} more features")
    
    # Model parameters
    print(f"\n🎛️  Model Parameters:")
    print(f"  Alpha (regularization): {model.alpha_:.6f}")
    print(f"  L1 ratio: {model.l1_ratio_:.3f}")
    print(f"  Intercept: {model.intercept_:.4f}")
    
    # Cross-validated performance
    print(f"\n📈 Cross-validated Performance:")
    cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
    cv_rmse = cross_val_score(model, X, y, cv=3, scoring='neg_root_mean_squared_error')
    
    print(f"  R² scores: {cv_scores}")
    print(f"  Mean R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  RMSE scores: {-cv_rmse}")
    print(f"  Mean RMSE: {-cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
    
    # Feature importance analysis
    print(f"\n🔍 Feature Importance Analysis:")
    print(f"  Total features: {len(X.columns)}")
    print(f"  Selected features: {len(non_zero_features)}")
    print(f"  Selection rate: {len(non_zero_features)/len(X.columns)*100:.1f}%")
    
    # Top features by importance
    top_features = feature_coefs.head(10)
    print(f"\n🏆 Top 10 Most Important Features:")
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:20s}: {row['coefficient']:8.4f}")
    
    # Coefficient distribution
    coef_abs = np.abs(model.coef_)
    print(f"\n📊 Coefficient Distribution:")
    print(f"  Mean |coefficient|: {coef_abs.mean():.6f}")
    print(f"  Max |coefficient|: {coef_abs.max():.6f}")
    print(f"  Features with |coef| > 0.01: {(coef_abs > 0.01).sum()}")
    print(f"  Features with |coef| > 0.1: {(coef_abs > 0.1).sum()}")
    
    # Save results
    results = {
        'model': model,
        'feature_coefficients': feature_coefs,
        'non_zero_features': non_zero_features,
        'cv_r2_scores': cv_scores,
        'cv_rmse_scores': -cv_rmse,
        'alpha': model.alpha_,
        'l1_ratio': model.l1_ratio_,
        'intercept': model.intercept_,
        'n_selected_features': len(non_zero_features),
        'selection_rate': len(non_zero_features)/len(X.columns)
    }
    
    print(f"\n✅ ElasticNetCV test completed successfully!")
    print(f"   Model selected {len(non_zero_features)} features from {len(X.columns)} total")
    print(f"   Cross-validated R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return results

if __name__ == "__main__":
    results = test_elasticnet_cv() 