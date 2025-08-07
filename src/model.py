"""
Machine learning model utilities for gestational age prediction.

This module provides model creation, training, and prediction functions for:
- Lasso regression
- Elastic Net regression  
- STABL (Stability Selection) feature selection
- Cross-validation model selection

Author: Diba Dindoust
Date: 07/01/2025
"""

from stabl.stabl import Stabl, plot_stabl_path, plot_fdr_graph, export_stabl_to_csv, save_stabl_results
from stabl.preprocessing import LowInfoFilter
from stabl.visualization import boxplot_features, scatterplot_features, plot_roc, boxplot_binary_predictions
from stabl.adaptive import ALogitLasso

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, KFold
import numpy as np

# Global configuration
random_state = 42

# STABL default parameters
STABL_LAMBDA_GRID = {"alpha": [0.01]}
STABL_N_BOOTSTRAPS = 20


def get_model(model_type):
    """
    Get a model instance based on the specified type.
    
    Args:
        model_type (str): Type of model ('stabl', 'elasticnet', 'lasso', 'elasticnet_cv', 'lasso_cv')
        
    Returns:
        tuple: (model, base_estimator) where base_estimator is None for non-STABL models
        
    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type == "stabl":
        # STABL with cross-validation - STABL handles its own hyperparameter optimization
        lasso = Lasso(max_iter=10000, fit_intercept=True, tol=1e-4, random_state=random_state)
        base_estimator = clone(lasso)
        stabl = Stabl(
            base_estimator=base_estimator,
            lambda_grid={"alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},
            n_lambda=10,
            artificial_type="random_permutation",
            artificial_proportion=1,
            n_bootstraps=500,
            random_state=random_state,
            verbose=1
        )
        return stabl, base_estimator
    elif model_type == "elasticnet":
        return ElasticNet(alpha=0.1), None
    elif model_type == "lasso":
        return Lasso(alpha=0.1, max_iter=2000), None
    elif model_type == "elasticnet_cv":
        # Elastic Net with cross-validation for optimal alpha and l1_ratio
        param_grid = {
            'elasticnet__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # Expanded grid
        }
        
        # Create pipeline with preprocessing
        pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5, weights='uniform')),
            ('scaler', StandardScaler()),
            ('elasticnet', ElasticNet(
                max_iter=10000,  # Increased for better convergence
                fit_intercept=True,  # Explicitly set
                tol=1e-4,  # Relaxed tolerance for better convergence
                random_state=random_state
            ))
        ])
        
        cv_model = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=KFold(n_splits=5, shuffle=True, random_state=random_state),  # 5-fold CV for regression
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        return cv_model, None
    elif model_type == "lasso_cv":
        # Lasso with cross-validation for optimal alpha, l1_ratio implicitly set at 1.0
        param_grid = {
            'lasso__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        }
        
        # Create pipeline with preprocessing
        pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5, weights='uniform')),
            ('scaler', StandardScaler()),
            ('lasso', Lasso(
                max_iter=10000,  # Increased for better convergence
                fit_intercept=True,  # Explicitly set
                tol=1e-4,  # Relaxed tolerance for better convergence
                random_state=random_state
            ))
        ])
        
        cv_model = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=KFold(n_splits=5, shuffle=True, random_state=random_state),  # 5-fold CV for regression
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        return cv_model, None
    else:
        raise ValueError("Unknown model type")


def get_classification_model(model_type):
    """
    Get a classification model instance based on the specified type.
    Args:
        model_type (str): Type of model ('stabl', 'elasticnet', 'lasso')
    Returns:
        tuple: (model, base_estimator) where base_estimator is None for non-STABL models
    """
    if model_type == "stabl":
        # STABL with cross-validation - STABL handles its own hyperparameter optimization
        logit_lasso = LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=10000,  # Increased for better convergence
            fit_intercept=True,  # Explicitly set
            tol=1e-4,  # Relaxed tolerance for better convergence
            class_weight="balanced",  # Handle class imbalance
            random_state=random_state
        )
        model = Stabl(
            base_estimator=clone(logit_lasso),
            lambda_grid={"C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]},
            n_lambda=10,
            artificial_type="random_permutation",
            artificial_proportion=1,
            n_bootstraps=500,
            random_state=random_state,
            verbose=1
        )
        return model, logit_lasso
    elif model_type == "elasticnet":
        # Use LogisticRegression with elasticnet penalty
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            max_iter=2000,
            random_state=42
        )
        return model, None
    elif model_type == "elasticnet_cv":
        # Use LogisticRegression with elasticnet penalty and CV
        # Log-scale grid for C to avoid clustering, with finer granularity around 0.1-1.0
        param_grid = {
            'logisticregression__C': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
            'logisticregression__l1_ratio': [0.3, 0.5, 0.7, 0.9]  # Exclude 1.0 as that's for lasso
        }
        
        # Create pipeline with preprocessing
        pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5, weights='uniform')),
            ('scaler', StandardScaler()),
            ('logisticregression', LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                max_iter=10000,  # Increased for better convergence
                fit_intercept=True,  # Explicitly set
                tol=1e-4,  # Relaxed tolerance for better convergence
                class_weight="balanced",  # Handle class imbalance
                random_state=42
            ))
        ])
        
        cv_model = GridSearchCV(
            pipeline,
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),  # Increased CV folds
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        return cv_model, None
    elif model_type == "lasso":
        # Use LogisticRegression with L1 penalty
        model = LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )
        return model, None
    elif model_type == "lasso_cv":
        # Use LogisticRegression with L1 penalty and CV, l1_ratio is implicitly set at 1.0
        # Log-scale grid for C to avoid clustering, with finer granularity around 0.1-1.0
        param_grid = {
            'logisticregression__C': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
        }
        
        # Create pipeline with preprocessing
        pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5, weights='uniform')),
            ('scaler', StandardScaler()),
            ('logisticregression', LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=10000,  # Increased for better convergence
                fit_intercept=True,  # Explicitly set
                tol=1e-4,  # Relaxed tolerance for better convergence
                class_weight="balanced",  # Handle class imbalance
                random_state=42
            ))
        ])
        
        cv_model = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),  # Use stratification for classification
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        return cv_model, None
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(X_train, y_train, model, base_estimator=None):
    """
    Train a model on the provided data.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets
        model: Model instance to train
        base_estimator: Base estimator for STABL (None for other models)
        
    Returns:
        tuple: (trained_model, base_estimator, selected_feature_names)
    """
    if base_estimator is None:
        # Regular model like ElasticNet or Lasso
        model.fit(X_train, y_train)
        return model, None, X_train.columns
    else:
        # STABL feature selection, train base estimator (e.g. ElasticNet)
        print(f"Before STABL: X_train shape = {X_train.shape}")
        X_train_selected = model.fit_transform(X_train, y_train)
        print(f"After STABL: X_train_selected shape = {X_train_selected.shape}")
        if X_train_selected.shape[1] == 0:
            raise RuntimeError("STABL selected zero features. This run will be stopped.")
        base_estimator.fit(X_train_selected, y_train)
        # Get selected feature names after STABL selection
        if hasattr(X_train, 'columns'):
            selected_mask = model.get_support()
            selected_feature_names = X_train.columns[selected_mask]
        else:
            selected_feature_names = None
        return model, base_estimator, selected_feature_names


def predict_model(model, X_test, base_estimator=None):
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        base_estimator: Base estimator for STABL (None for other models)
        
    Returns:
        array: Model predictions
    """
    if base_estimator is None:
        return model.predict(X_test)
    else:
        # Use STABL-selected features (no fallback if shape is 0)
        X_test_selected = model.transform(X_test)
        return base_estimator.predict(X_test_selected)


def compare_all_models(X_train, y_train, X_test, y_test):
    """
    Compare Lasso vs Elastic Net vs STABL performance.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training targets
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test targets
        
    Returns:
        dict: Results dictionary with performance metrics and selected features
    """
    print("Comparing Lasso vs Elastic Net vs STABL...")
    
    # Lasso with CV
    print("Training Lasso with cross-validation...")
    lasso_cv, _ = get_model("lasso_cv")
    lasso_cv.fit(X_train, y_train)
    lasso_pred = lasso_cv.predict(X_test)
    lasso_alpha = lasso_cv.best_params_['alpha']
    lasso_coef = lasso_cv.best_estimator_.coef_
    lasso_n_features = np.sum(lasso_coef != 0)
    
    # Elastic Net with CV
    print("Training Elastic Net with cross-validation...")
    enet_cv, _ = get_model("elasticnet_cv")
    enet_cv.fit(X_train, y_train)
    enet_pred = enet_cv.predict(X_test)
    enet_alpha = enet_cv.best_params_['alpha']
    enet_l1_ratio = enet_cv.best_params_['l1_ratio']
    enet_coef = enet_cv.best_estimator_.coef_
    enet_n_features = np.sum(enet_coef != 0)
    
    # STABL
    print("Training STABL...")
    stabl, base_estimator = get_model("stabl")
    X_train_selected = stabl.fit_transform(X_train, y_train)
    base_estimator.fit(X_train_selected, y_train)
    X_test_selected = stabl.transform(X_test)
    stabl_pred = base_estimator.predict(X_test_selected)
    stabl_n_features = X_train_selected.shape[1]
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = {
        'lasso': {
            'predictions': lasso_pred,
            'alpha': lasso_alpha,
            'coefficients': lasso_coef,
            'n_features': lasso_n_features,
            'mse': mean_squared_error(y_test, lasso_pred),
            'mae': mean_absolute_error(y_test, lasso_pred),
            'r2': r2_score(y_test, lasso_pred)
        },
        'elasticnet': {
            'predictions': enet_pred,
            'alpha': enet_alpha,
            'l1_ratio': enet_l1_ratio,
            'coefficients': enet_coef,
            'n_features': enet_n_features,
            'mse': mean_squared_error(y_test, enet_pred),
            'mae': mean_absolute_error(y_test, enet_pred),
            'r2': r2_score(y_test, enet_pred)
        },
        'stabl': {
            'predictions': stabl_pred,
            'n_features': stabl_n_features,
            'mse': mean_squared_error(y_test, stabl_pred),
            'mae': mean_absolute_error(y_test, stabl_pred),
            'r2': r2_score(y_test, stabl_pred)
        }
    }
    
    # Print comparison
    print(f"\n=== Model Comparison ===")
    print(f"Lasso (alpha={lasso_alpha:.4f}):")
    print(f"  - Features selected: {lasso_n_features}")
    print(f"  - MSE: {results['lasso']['mse']:.4f}")
    print(f"  - MAE: {results['lasso']['mae']:.4f}")
    print(f"  - R²: {results['lasso']['r2']:.4f}")
    
    print(f"\nElastic Net (alpha={enet_alpha:.4f}, l1_ratio={enet_l1_ratio:.2f}):")
    print(f"  - Features selected: {enet_n_features}")
    print(f"  - MSE: {results['elasticnet']['mse']:.4f}")
    print(f"  - MAE: {results['elasticnet']['mae']:.4f}")
    print(f"  - R²: {results['elasticnet']['r2']:.4f}")
    
    print(f"\nSTABL:")
    print(f"  - Features selected: {stabl_n_features}")
    print(f"  - MSE: {results['stabl']['mse']:.4f}")
    print(f"  - MAE: {results['stabl']['mae']:.4f}")
    print(f"  - R²: {results['stabl']['r2']:.4f}")
    
    # Determine winner
    mse_scores = {
        'Lasso': results['lasso']['mse'],
        'Elastic Net': results['elasticnet']['mse'],
        'STABL': results['stabl']['mse']
    }
    best_model = min(mse_scores, key=mse_scores.get)
    print(f"\n🏆 {best_model} performs best (lowest MSE: {mse_scores[best_model]:.4f})")
    
    return results