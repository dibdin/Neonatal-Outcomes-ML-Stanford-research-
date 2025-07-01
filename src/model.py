from stabl.stabl import Stabl, plot_stabl_path, plot_fdr_graph, export_stabl_to_csv, save_stabl_results
from stabl.preprocessing import LowInfoFilter
from stabl.visualization import boxplot_features, scatterplot_features, plot_roc, boxplot_binary_predictions
from stabl.adaptive import ALogitLasso

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np

random_state = 42

def get_model(model_type):
    if model_type == "stabl":
        base_estimator = ElasticNet(alpha=0.0001, l1_ratio=0.5, max_iter=2000)
        stabl = Stabl(
            base_estimator=base_estimator,
            lambda_grid={"alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1]},
            n_bootstraps=25,
            artificial_type="knockoff",
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
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]  # 1.0 is equivalent to Lasso
        }
        elasticnet = ElasticNet(max_iter=2000, random_state=random_state)
        cv_model = GridSearchCV(
            elasticnet, 
            param_grid, 
            cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        return cv_model, None
    elif model_type == "lasso_cv":
        # Lasso with cross-validation for optimal alpha
        param_grid = {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
        }
        lasso = Lasso(max_iter=2000, random_state=random_state)
        cv_model = GridSearchCV(
            lasso, 
            param_grid, 
            cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        return cv_model, None
    else:
        raise ValueError("Unknown model type")

def train_model(X_train, y_train, model, base_estimator=None):
    if base_estimator is None:
        # Regular model like ElasticNet or Lasso
        model.fit(X_train, y_train)
        return model, None
    else:
        # STABL feature selection, train base estimator (e.g. ElasticNet)
        print(f"Before STABL: X_train shape = {X_train.shape}")
        X_train_selected = model.fit_transform(X_train, y_train)
        print(f"After STABL: X_train_selected shape = {X_train_selected.shape}")
        
        if X_train_selected.shape[1] == 0:
            print("WARNING: STABL selected 0 features! Falling back to using all features with ElasticNet.")
            print("This run will use all 53 features instead of STABL-selected features.")
            # Fall back to using all features
            base_estimator.fit(X_train, y_train)
            return model, base_estimator
        else:
            base_estimator.fit(X_train_selected, y_train)
            return model, base_estimator

def predict_model(model, X_test, base_estimator=None):
    if base_estimator is None:
        return model.predict(X_test)
    else:
        # Check if STABL selected any features
        X_test_selected = model.transform(X_test)
        if X_test_selected.shape[1] == 0:
            # STABL selected 0 features, use all features
            return base_estimator.predict(X_test)
        else:
            # Use STABL-selected features
            return base_estimator.predict(X_test_selected)

def compare_all_models(X_train, y_train, X_test, y_test):
    """
    Compare Lasso vs Elastic Net vs STABL performance.
    Returns results dictionary with performance metrics and selected features.
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
    
    if X_train_selected.shape[1] == 0:
        print("WARNING: STABL selected 0 features! Using all features.")
        stabl_pred = base_estimator.predict(X_test)
        stabl_n_features = X_train.shape[1]
    else:
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