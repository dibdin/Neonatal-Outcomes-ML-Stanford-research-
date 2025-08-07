"""
Gestational Age Prediction and Preterm Birth Classification (from regression to classification)

This script implements a comprehensive machine learning pipeline for predicting 
gestational age and classifying preterm births using three different model approaches:
1. Clinical model: Uses clinical/demographic features (columns 145, 146, 159 + interactions)
2. Biomarker model: Uses serum biomarker features (columns 30-141)
3. Combined model: Uses both clinical and biomarker features

The pipeline includes:
- Multiple model training runs with cross-validation
- Performance evaluation with confidence intervals
- Separate analysis for preterm vs term babies
- Comprehensive visualization and reporting
- Model comparison and best model identification

Author: Diba Dindoust
Date: 07/01/2025
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
import pickle
from sklearn.preprocessing import StandardScaler
import os # Added for os.makedirs
import shutil

from src.data_loader import split_data, load_and_process_data
from src.model import train_model, predict_model, get_model
from src.metrics import (
    compute_mae, compute_rmse, 
    compute_metrics_by_gestational_age, compute_confidence_interval
)
from src.utils import (
    count_high_weight_biomarkers, create_metrics_table,
    plot_metrics_with_confidence_intervals, plot_summary_auc_by_dataset_and_model,
    plot_summary_mae_by_dataset_and_model, plot_summary_rmse_by_dataset_and_model,
    plot_true_vs_predicted_scatter, plot_biomarker_frequency_heel_vs_cord,
    plot_feature_frequency, save_all_as_pickle
)
from src.config import N_REPEATS, TEST_SIZE, PRETERM_CUTOFF

DATA_OPTION_LABELS = {1: 'both_samples', 2: 'heel_all', 3: 'cord_all'}

stabl_heel_biomarker_zero_feature_runs = []

def run_single_model(model_name, data_type, dataset_type, model_type, data_option=1, data_option_label='both_samples', target_type='gestational_age'):
    """
    Run a single model configuration and return results.
    
    Args:
        model_name (str): Name of the model (Clinical, Biomarker, Combined)
        data_type (str): Type of data to use (clinical, biomarker, combined)
        dataset_type (str): Dataset type (cord, heel)
        model_type (str): Model algorithm (lasso_cv, elasticnet_cv, stabl)
        data_option (int): Data loading option (1, 2, or 3)
        data_option_label (str): Label for data option
        target_type (str): Target variable type ('gestational_age' or 'birth_weight')
    
    Returns:
        dict: Model results and summary statistics
    """
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name} ({data_type.upper()} DATA) - {model_type.upper()} ON {dataset_type.upper()} [{data_option_label}]")
    print(f"{'='*60}")
    
    # Load and prepare data for this model
    X, y = load_and_process_data(dataset_type, model_type=data_type, data_option=data_option, target_type=target_type)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Initialize results storage
    maes = []
    rmses = []
    aucs = []
    run_predictions = []
    run_classification_predictions = []
    all_coefficients = []  # Collect coefficients for all runs
    all_classification_coefficients = []  # Collect coefficients for classification

    # --- REGRESSION TASK (GA prediction) ---
    n_stabl_regression_skipped = 0
    for i in range(N_REPEATS):
        try:
            # Split data
            X_train, X_test, y_train, y_test = split_data(X, y, test_size=TEST_SIZE, random_state=42 + i)
            
            if model_type == 'stabl':
                from sklearn.pipeline import Pipeline
                from sklearn.feature_selection import VarianceThreshold
                from stabl.preprocessing import LowInfoFilter
                from sklearn.impute import SimpleImputer
                # Build preprocessing pipeline as in the official STABL notebook
                preprocessing = Pipeline([
                    ("variance_threshold", VarianceThreshold(threshold=0)),
                    ("low_info_filter", LowInfoFilter(max_nan_fraction=0.2)),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("std", StandardScaler()),
                ])
                X_train_processed = preprocessing.fit_transform(X_train)
                X_test_processed = preprocessing.transform(X_test)
                # Convert back to DataFrame for compatibility
                X_train_scaled = pd.DataFrame(X_train_processed, columns=X_train.columns[preprocessing.named_steps['variance_threshold'].get_support()][preprocessing.named_steps['low_info_filter'].get_support()], index=X_train.index)
                X_test_scaled = pd.DataFrame(X_test_processed, columns=X_train_scaled.columns, index=X_test.index)
            # Get and train model
            model, base_estimator = get_model(model_type)
            if model_type == 'stabl':
                from sklearn.pipeline import Pipeline
                from sklearn.feature_selection import VarianceThreshold
                from stabl.preprocessing import LowInfoFilter
                from sklearn.impute import SimpleImputer
                # Build preprocessing pipeline as in the official STABL notebook
                preprocessing = Pipeline([
                    ("variance_threshold", VarianceThreshold(threshold=0)),
                    ("low_info_filter", LowInfoFilter(max_nan_fraction=0.2)),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("std", StandardScaler()),
                ])
                X_train_processed = preprocessing.fit_transform(X_train)
                X_test_processed = preprocessing.transform(X_test)
                # Convert back to DataFrame for compatibility
                X_train_scaled = pd.DataFrame(X_train_processed, columns=X_train.columns[preprocessing.named_steps['variance_threshold'].get_support()][preprocessing.named_steps['low_info_filter'].get_support()], index=X_train.index)
                X_test_scaled = pd.DataFrame(X_test_processed, columns=X_train_scaled.columns, index=X_test.index)
                print('\n[DEBUG] STABL input X_train_scaled shape:', X_train_scaled.shape)
                print('[DEBUG] First 5 rows of X_train_scaled:\n', X_train_scaled.head())
                variances = X_train_scaled.var(axis=0)
                print('[DEBUG] Feature variances:')
                print('  Number of features with zero variance:', (variances == 0).sum())
                print('  Min variance:', variances.min())
                print('  Max variance:', variances.max())
                print('  Mean variance:', variances.mean())
                print('[DEBUG] y_train stats: min', y_train.min(), 'max', y_train.max(), 'mean', y_train.mean(), 'std', y_train.std())
                trained_model, base_estimator, selected_feature_names = train_model(X_train_scaled, y_train, model, base_estimator)
                n_selected = len(selected_feature_names) if selected_feature_names is not None else 0
                print(f"[STABL] Number of features selected for {model_name} ({data_type}) on {dataset_type}: {n_selected}")
                print(f"[STABL] Selected features: {list(selected_feature_names) if selected_feature_names is not None else 'None'}")
                if base_estimator is not None and hasattr(base_estimator, 'coef_'):
                    print(f"[STABL] Base estimator coefficients: {base_estimator.coef_}")
                y_preds = predict_model(model, X_test_scaled, base_estimator)
            else:
                # For CV models, preprocessing is handled in the pipeline
                # For non-CV models, apply preprocessing manually
                if model_type in ['lasso_cv', 'elasticnet_cv']:
                    # Pipeline handles preprocessing automatically
                    trained_model, _, selected_feature_names = train_model(X_train, y_train, model, None)
                    y_preds = predict_model(trained_model, X_test, None)
                    # For CV models, use original data for feature reference
                    X_train_scaled = X_train
                    X_test_scaled = X_test
                else:
                    # Manual preprocessing for non-pipeline models
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
                    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
                    trained_model, _, selected_feature_names = train_model(X_train_scaled, y_train, model, None)
                    y_preds = predict_model(trained_model, X_test_scaled, None)
            
            # Calculate regression metrics
            mae = mean_absolute_error(y_test, y_preds)
            rmse = np.sqrt(mean_squared_error(y_test, y_preds))
            
            # Extract optimized hyperparameters for CV models
            optimized_alpha = None
            optimized_l1_ratio = None
            if model_type in ['lasso_cv', 'elasticnet_cv']:
                if hasattr(trained_model, 'best_params_'):
                    # DEBUG: Print the full best_params_ to see what's actually there
                    print(f"    DEBUG - Full best_params_: {trained_model.best_params_}")
                    # Extract from pipeline structure
                    if model_type == 'lasso_cv':
                        optimized_alpha = trained_model.best_params_.get('lasso__alpha')
                    elif model_type == 'elasticnet_cv':
                        optimized_alpha = trained_model.best_params_.get('elasticnet__alpha')
                        optimized_l1_ratio = trained_model.best_params_.get('elasticnet__l1_ratio')
                    print(f"    Optimized hyperparameters - Alpha: {optimized_alpha}, L1_ratio: {optimized_l1_ratio}")
                    # DEBUG: Check if alpha is exactly 0.0 and if it's in the grid
                    if optimized_alpha == 0.0:
                        print(f"    ⚠️  WARNING: Alpha=0.0 detected! This should not be in the grid.")
                        print(f"    DEBUG - Grid alpha values: [0.001, 0.01, 0.1, 1.0, 10.0]")
                elif hasattr(trained_model, 'best_estimator_'):
                    # For LogisticRegression, alpha is inverse of C
                    if hasattr(trained_model.best_estimator_, 'C'):
                        optimized_alpha = 1.0 / trained_model.best_estimator_.C
                    if hasattr(trained_model.best_estimator_, 'l1_ratio'):
                        optimized_l1_ratio = trained_model.best_estimator_.l1_ratio
                    print(f"    Optimized hyperparameters - Alpha (1/C): {optimized_alpha}, L1_ratio: {optimized_l1_ratio}")
            
            # Store regression results
            maes.append(mae)
            rmses.append(rmse)
            
            # Store regression predictions for this run
            run_predictions.append({
                'true': y_test.values,
                'pred': y_preds,
                'mae': mae,
                'rmse': rmse,
                'optimized_alpha': optimized_alpha,
                'optimized_l1_ratio': optimized_l1_ratio
            })
            
            # Save regression model outputs per run
            if target_type == 'gestational_age':
                model_outputs = {
                    'GA true': y_test.values,
                    'GA preds': y_preds,
                    'MAE': mae,
                    'RMSE': rmse,
                    'features': X_train_scaled,
                    'coef': trained_model.coef_ if hasattr(trained_model, 'coef_') else None
                }
            else:  # target_type == 'birth_weight'
                model_outputs = {
                    'BW true': y_test.values,
                    'BW preds': y_preds,
                    'MAE': mae,
                    'RMSE': rmse,
                    'features': X_train_scaled,
                    'coef': trained_model.coef_ if hasattr(trained_model, 'coef_') else None
                }
            
            # Collect coefficients for frequency plot (biomarker/combined only)
            if model_name.lower() in ['biomarker', 'combined']:
                # For CV models, extract coefficients from the best estimator
                if model_type in ['lasso_cv', 'elasticnet_cv']:
                    if hasattr(trained_model, 'best_estimator_'):
                        # For CV models, the best_estimator_ is a pipeline
                        # We need to extract coefficients from the final step (LogisticRegression for classification, Lasso/ElasticNet for regression)
                        best_estimator = trained_model.best_estimator_
                        if hasattr(best_estimator, 'named_steps'):
                            # It's a pipeline - get the final step
                            final_step_name = list(best_estimator.named_steps.keys())[-1]
                            final_step = best_estimator.named_steps[final_step_name]
                            if hasattr(final_step, 'coef_'):
                                coef = final_step.coef_
                                print(f"    DEBUG - CV model pipeline final step '{final_step_name}' coefs shape: {coef.shape}")
                                print(f"    DEBUG - CV model pipeline final step coefs: {coef}")
                            else:
                                print(f"    DEBUG - No coef_ found in pipeline final step '{final_step_name}'")
                                coef = np.zeros(X_train_scaled.shape[1])
                        else:
                            # Direct estimator (not pipeline)
                            if hasattr(best_estimator, 'coef_'):
                                coef = best_estimator.coef_
                                print(f"    DEBUG - CV model best estimator coefs shape: {coef.shape}")
                                print(f"    DEBUG - CV model best estimator coefs: {coef}")
                            else:
                                print(f"    DEBUG - No coef_ found in CV model best estimator")
                                coef = np.zeros(X_train_scaled.shape[1])
                    else:
                        print(f"    DEBUG - No best_estimator_ found for CV model")
                        coef = np.zeros(X_train_scaled.shape[1])
                else:
                    # For non-CV models, extract directly from trained_model
                    if hasattr(trained_model, 'coef_') and trained_model.coef_ is not None:
                        coef = trained_model.coef_
                        print(f"    DEBUG - Non-CV model coefs shape: {coef.shape}")
                        print(f"    DEBUG - Non-CV model coefs: {coef}")
                    else:
                        print(f"    DEBUG - No coef_ found for non-CV model")
                        coef = np.zeros(X_train_scaled.shape[1])
                
                # Ensure full length, pad with zeros if needed
                if len(coef) < X_train_scaled.shape[1]:
                    coef = np.pad(coef, (0, X_train_scaled.shape[1] - len(coef)), 'constant')
                    print(f"    DEBUG - Padded coefs shape: {coef.shape}")
                
                all_coefficients.append(coef)
                print(f"    DEBUG - All coefficients length: {len(all_coefficients)}")
                print(f"    DEBUG - Current coef: {coef}")
            # Save regression model outputs per run
            filename = f"outputs/models/{data_option_label}_{dataset_type}_{model_type}_{model_name.lower()}_run{i + 1}_model_outputs.pkl"
            save_all_as_pickle(model_outputs, filename)
            
            # --- CLASSIFICATION TASK (Preterm vs Term or SGA vs Normal) ---
            # Only run classification for lasso and elasticnet (including CV versions)
            if model_type in ['lasso_cv', 'elasticnet_cv']:
                # Create binary classification targets based on target_type
                if target_type == 'gestational_age':
                    # Original: Preterm vs Term classification
                    y_train_binary = (y_train < PRETERM_CUTOFF).astype(int)
                    y_test_binary = (y_test < PRETERM_CUTOFF).astype(int)
                    classification_type = 'preterm'
                else:  # target_type == 'birth_weight'
                    # SGA classification using Intergrowth-21 reference values
                    from src.sga_classification import create_sga_targets_intergrowth21
                    # Create SGA targets for both 10th and 3rd percentile using Intergrowth-21
                    sga_10th_train = create_sga_targets_intergrowth21(y_train, data_option, dataset_type, '10th_percentile')
                    sga_10th_test = create_sga_targets_intergrowth21(y_test, data_option, dataset_type, '10th_percentile')
                    sga_3rd_train = create_sga_targets_intergrowth21(y_train, data_option, dataset_type, '3rd_percentile')
                    sga_3rd_test = create_sga_targets_intergrowth21(y_test, data_option, dataset_type, '3rd_percentile')
                    # Use 3rd percentile for main classification (Intergrowth-21 standard)
                    y_train_binary = sga_3rd_train
                    y_test_binary = sga_3rd_test
                    classification_type = 'sga_3rd_intergrowth21'
                    # Store both classifications for later analysis
                    sga_classifications = {
                        'sga_10th_train': sga_10th_train,
                        'sga_10th_test': sga_10th_test,
                        'sga_3rd_train': sga_3rd_train,
                        'sga_3rd_test': sga_3rd_test
                    }
                # Print class balance for train and test splits
                def print_class_balance(y, label=""):
                    unique, counts = np.unique(y[~np.isnan(y)], return_counts=True)
                    print(f"    {label} class balance: {dict(zip(unique, counts))} (SGA/preterm proportion: {np.nanmean(y):.3f})")
                print_class_balance(y_train_binary, "Train")
                print_class_balance(y_test_binary, "Test")
                
                # Get classification model
                from src.model import get_classification_model
                class_model, class_base_estimator = get_classification_model(model_type)
                
                # Train classification model
                if model_type == 'stabl':
                    # Apply the same preprocessing pipeline for STABL classification
                    from sklearn.pipeline import Pipeline
                    from sklearn.feature_selection import VarianceThreshold
                    from stabl.preprocessing import LowInfoFilter
                    from sklearn.impute import SimpleImputer
                    
                    # Build preprocessing pipeline as specified
                    preprocessing = Pipeline([
                        ("variance_threshold", VarianceThreshold(threshold=0)),  # Removing 0 variance features
                        ("low_info_filter", LowInfoFilter(max_nan_fraction=0.2)),
                        ("imputer", SimpleImputer(strategy="median")),  # Imputing missing values with median
                        ("std", StandardScaler())  # Z-scoring features
                    ])
                    
                    # Apply preprocessing to classification data
                    X_train_class_processed = preprocessing.fit_transform(X_train)
                    X_test_class_processed = preprocessing.transform(X_test)
                    
                    # Convert back to DataFrame for compatibility
                    X_train_class_scaled = pd.DataFrame(X_train_class_processed, 
                        columns=X_train.columns[preprocessing.named_steps['variance_threshold'].get_support()][preprocessing.named_steps['low_info_filter'].get_support()], 
                        index=X_train.index)
                    X_test_class_scaled = pd.DataFrame(X_test_class_processed, 
                        columns=X_train_class_scaled.columns, 
                        index=X_test.index)
                    
                    print(f'\n[DEBUG] STABL classification input X_train_class_scaled shape:', X_train_class_scaled.shape)
                    
                    trained_class_model, class_base_estimator, class_selected_feature_names = train_model(X_train_class_scaled, y_train_binary, class_model, class_base_estimator)
                    y_class_preds = predict_model(class_model, X_test_class_scaled, class_base_estimator)
                else:
                    trained_class_model, _, class_selected_feature_names = train_model(X_train_scaled, y_train_binary, class_model, None)
                    y_class_preds = predict_model(trained_class_model, X_test_scaled, None)
                
                # Calculate classification metrics
                from src.metrics import compute_auc
                auc = compute_auc(y_test_binary, y_class_preds)
                
                # Extract optimized hyperparameters for CV classification models
                class_optimized_alpha = None
                class_optimized_l1_ratio = None
                if model_type in ['lasso_cv', 'elasticnet_cv']:
                    if hasattr(trained_class_model, 'best_params_'):
                        # DEBUG: Print the full best_params_ to see what's actually there
                        print(f"    DEBUG - Full best_params_: {trained_class_model.best_params_}")
                        class_optimized_alpha = trained_class_model.best_params_.get('C')
                        class_optimized_l1_ratio = trained_class_model.best_params_.get('l1_ratio')
                        print(f"    Classification optimized hyperparameters - C: {class_optimized_alpha}, L1_ratio: {class_optimized_l1_ratio}")
                        # DEBUG: Check if C is exactly 0.0 and if it's in the grid
                        if class_optimized_alpha == 0.0:
                            print(f"    ⚠️  WARNING: C=0.0 detected! This should not be in the grid.")
                            print(f"    DEBUG - Grid C values: [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]")
                    elif hasattr(trained_class_model, 'best_estimator_'):
                        if hasattr(trained_class_model.best_estimator_, 'C'):
                            class_optimized_alpha = trained_class_model.best_estimator_.C
                        if hasattr(trained_class_model.best_estimator_, 'l1_ratio'):
                            class_optimized_l1_ratio = trained_class_model.best_estimator_.l1_ratio
                        print(f"    Classification optimized hyperparameters - C: {class_optimized_alpha}, L1_ratio: {class_optimized_l1_ratio}")
                
                # Store classification results
                aucs.append(auc)
                
                # Store classification predictions for this run
                run_classification_predictions.append({
                    'true': y_test_binary if isinstance(y_test_binary, np.ndarray) else y_test_binary.values,
                    'pred': y_class_preds,
                    'auc': auc,
                    'optimized_C': class_optimized_alpha,
                    'optimized_l1_ratio': class_optimized_l1_ratio
                })
                
                # Collect classification coefficients for frequency plot (biomarker/combined only)
                if model_name.lower() in ['biomarker', 'combined']:
                    # For CV classification models, extract coefficients from the best estimator
                    if model_type in ['lasso_cv', 'elasticnet_cv']:
                        if hasattr(trained_class_model, 'best_estimator_'):
                            # For CV models, the best_estimator_ is a pipeline
                            # We need to extract coefficients from the final step (LogisticRegression)
                            best_estimator = trained_class_model.best_estimator_
                            if hasattr(best_estimator, 'named_steps'):
                                # It's a pipeline - get the final step (LogisticRegression)
                                final_step_name = list(best_estimator.named_steps.keys())[-1]
                                final_step = best_estimator.named_steps[final_step_name]
                                if hasattr(final_step, 'coef_'):
                                    class_coef = final_step.coef_
                                    print(f"    DEBUG - CV classification model pipeline final step '{final_step_name}' coefs shape: {class_coef.shape}")
                                    print(f"    DEBUG - CV classification model pipeline final step coefs: {class_coef}")
                                else:
                                    print(f"    DEBUG - No coef_ found in CV classification pipeline final step '{final_step_name}'")
                                    class_coef = np.zeros(X_train_scaled.shape[1])
                            else:
                                # Direct estimator (not pipeline)
                                if hasattr(best_estimator, 'coef_'):
                                    class_coef = best_estimator.coef_
                                    print(f"    DEBUG - CV classification model best estimator coefs shape: {class_coef.shape}")
                                    print(f"    DEBUG - CV classification model best estimator coefs: {class_coef}")
                                else:
                                    print(f"    DEBUG - No coef_ found in CV classification model best estimator")
                                    class_coef = np.zeros(X_train_scaled.shape[1])
                        else:
                            print(f"    DEBUG - No best_estimator_ found for CV classification model")
                            class_coef = np.zeros(X_train_scaled.shape[1])
                    else:
                        # For non-CV models (STABL), extract directly from trained_class_model
                        if hasattr(trained_class_model, 'coef_') and trained_class_model.coef_ is not None:
                            class_coef = trained_class_model.coef_
                            print(f"    DEBUG - Non-CV classification model coefs shape: {class_coef.shape}")
                            print(f"    DEBUG - Non-CV classification model coefs: {class_coef}")
                        else:
                            print(f"    DEBUG - No coef_ found for non-CV classification model")
                            class_coef = np.zeros(X_train_scaled.shape[1])
                    
                    # Ensure full length, pad with zeros if needed
                    if len(class_coef) < X_train_scaled.shape[1]:
                        class_coef = np.pad(class_coef, (0, X_train_scaled.shape[1] - len(class_coef)), 'constant')
                        print(f"    DEBUG - Padded classification coefs shape: {class_coef.shape}")
                    
                    all_classification_coefficients.append(class_coef)
                    print(f"    DEBUG - All classification coefficients length: {len(all_classification_coefficients)}")
                    print(f"    DEBUG - Current classification coef: {class_coef}")
                
                # Save classification model outputs per run
                if target_type == 'gestational_age':
                    class_model_outputs = {
                        'Preterm true': y_test_binary if isinstance(y_test_binary, np.ndarray) else y_test_binary.values,
                        'Preterm preds': y_class_preds,
                        'AUC': auc,
                        'features': X_train_scaled,
                        'coef': trained_class_model.coef_ if hasattr(trained_class_model, 'coef_') else None
                    }
                else:  # target_type == 'birth_weight'
                    class_model_outputs = {
                        'SGA true': y_test_binary if isinstance(y_test_binary, np.ndarray) else y_test_binary.values,
                        'SGA preds': y_class_preds,
                        'AUC': auc,
                        'features': X_train_scaled,
                        'coef': trained_class_model.coef_ if hasattr(trained_class_model, 'coef_') else None
                    }
                class_filename = f"outputs/models/{data_option_label}_{dataset_type}_{model_type}_{model_name.lower()}_run{i + 1}_classification_outputs.pkl"
                save_all_as_pickle(class_model_outputs, class_filename)
                
                print(f"  Run {i+1}: MAE = {mae:.3f}, RMSE = {rmse:.3f}, AUC = {auc:.3f}")
            else:
                print(f"  Run {i+1}: Skipping classification for STABL model.")
        except RuntimeError as e:
            if model_type == 'stabl' and 'zero features' in str(e).lower():
                print(f"[STABL] Skipping run {i+1} for {model_name} ({data_type}) on {dataset_type}: zero features selected.")
                n_stabl_regression_skipped += 1
                continue
            else:
                raise
    
    # Print summary of skipped STABL runs
    if model_type == 'stabl':
        print(f"\n[SUMMARY] STABL regression: Zero features selected in {n_stabl_regression_skipped} out of {N_REPEATS} runs for {model_name} ({data_type}) on {dataset_type}.\n")
    
    # Calculate summary statistics for regression
    mae_mean = np.mean(maes)
    mae_std = np.std(maes)
    mae_ci_lower = mae_mean - 1.96 * mae_std / np.sqrt(len(maes))
    mae_ci_upper = mae_mean + 1.96 * mae_std / np.sqrt(len(maes))
    
    rmse_mean = np.mean(rmses)
    rmse_std = np.std(rmses)
    rmse_ci_lower = rmse_mean - 1.96 * rmse_std / np.sqrt(len(rmses))
    rmse_ci_upper = rmse_mean + 1.96 * rmse_std / np.sqrt(len(rmses))
    
    # Calculate summary statistics for classification
    auc_mean = np.mean([auc for auc in aucs if not np.isnan(auc)])
    auc_std = np.std([auc for auc in aucs if not np.isnan(auc)])
    if len([auc for auc in aucs if not np.isnan(auc)]) > 0:
        auc_ci_lower = auc_mean - 1.96 * auc_std / np.sqrt(len([auc for auc in aucs if not np.isnan(auc)]))
        auc_ci_upper = auc_mean + 1.96 * auc_std / np.sqrt(len([auc for auc in aucs if not np.isnan(auc)]))
    else:
        auc_ci_lower = np.nan
        auc_ci_upper = np.nan
    
    # Store results
    results = {
        'maes': maes,
        'rmses': rmses,
        'aucs': aucs,
        'predictions': run_predictions,
        'classification_predictions': run_classification_predictions,
        'summary': {
            'mae_mean': mae_mean,
            'mae_ci_lower': mae_ci_lower,
            'mae_ci_upper': mae_ci_upper,
            'rmse_mean': rmse_mean,
            'rmse_ci_lower': rmse_ci_lower,
            'rmse_ci_upper': rmse_ci_upper,
            'auc_mean': auc_mean,
            'auc_ci_lower': auc_ci_lower,
            'auc_ci_upper': auc_ci_upper
        }
    }
    # Add coefficients and feature names for biomarker/combined models
    if model_name.lower() in ['biomarker', 'combined']:
        results['all_coefficients'] = all_coefficients
        results['all_classification_coefficients'] = all_classification_coefficients
        results['feature_names'] = list(X.columns)
    
    return results


def generate_model_outputs(model_name, data_type, dataset_type, model_type, results, all_fpr, tpr_list, auc_mean, data_option_label, target_type='gestational_age'):
    """Generate all output files for a single model."""
    
    # Biomarker frequency plot generation removed - now handled by organized plot structure

    # Create performance metrics table (regression and classification)
    print(f"\nCreating performance metrics table and plots for {model_name} model ({model_type} on {dataset_type})...")
    
    # Create metrics table for both regression and classification
    metrics_data = {
        'Metric': ['MAE', 'RMSE', 'AUC'],
        'Mean': [results['summary']['mae_mean'], results['summary']['rmse_mean'], results['summary']['auc_mean']],
        'CI_Lower': [results['summary']['mae_ci_lower'], results['summary']['rmse_ci_lower'], results['summary']['auc_ci_lower']],
        'CI_Upper': [results['summary']['mae_ci_upper'], results['summary']['rmse_ci_upper'], results['summary']['auc_ci_upper']]
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df['CI_95%'] = metrics_df.apply(lambda row: f"[{row['CI_Lower']:.3f}, {row['CI_Upper']:.3f}]" if not np.isnan(row['CI_Lower']) else "N/A", axis=1)
    metrics_df['Mean_CI'] = metrics_df.apply(lambda row: f"{row['Mean']:.3f} {row['CI_95%']}" if not np.isnan(row['Mean']) else "N/A", axis=1)
    
    # Save table
    os.makedirs(f"outputs/tables", exist_ok=True)
    metrics_df.to_csv(f"outputs/tables/{data_option_label}_{dataset_type}_{model_type}_{model_name.lower()}_performance_metrics.csv", index=False)
    print(f"Performance metrics table saved to outputs/tables/{data_option_label}_{dataset_type}_{model_type}_{model_name.lower()}_performance_metrics.csv")

    # Create bar plots with confidence intervals (regression and classification)
    plot_metrics_with_confidence_intervals(
        results['maes'], results['rmses'], 
        [], [],  # Empty lists for preterm/term metrics
        filename=f"outputs/plots/{data_option_label}_{dataset_type}_{model_type}_{model_name.lower()}_metrics_with_ci_{target_type}.png"
    )
    print(f"Performance plots with confidence intervals saved to outputs/plots/{data_option_label}_{dataset_type}_{model_type}_{model_name.lower()}_metrics_with_ci_{target_type}.png")
    
    # Create ROC curve if we have classification results
    if 'classification_predictions' in results and results['classification_predictions']:
        from src.utils import plot_roc_curve
        # Aggregate predictions for ROC curve
        all_true = []
        all_pred = []
        for pred in results['classification_predictions']:
            all_true.extend(pred['true'])
            all_pred.extend(pred['pred'])
        
        if len(set(all_true)) > 1:  # Only plot if we have both classes
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(all_true, all_pred)
            auc = results['summary']['auc_mean']
            
            # Determine classification type for filename
            if target_type == 'gestational_age':
                classification_type = 'preterm'
            else:  # target_type == 'birth_weight'
                classification_type = 'sga'
            
            plot_roc_curve(
                fpr, tpr, auc,
                filename=f"outputs/plots/{data_option_label}_{dataset_type}_{model_type}_{model_name.lower()}_roc_curve_{classification_type}_{target_type}.png"
            )
            print(f"ROC curve saved to outputs/plots/{data_option_label}_{dataset_type}_{model_type}_{model_name.lower()}_roc_curve_{classification_type}_{target_type}.png")


def print_detailed_summary(model_name, dataset_type, model_type, results):
    """Print detailed performance summary for a model."""
    print(f"\n{model_name} Model - Final Performance Summary")
    print("="*60)
    print(f"Dataset: {dataset_type.upper()}")
    print(f"Model: {model_type.upper()}")
    print(f"Number of runs: {N_REPEATS}")
    
    summary = results['summary']
    print("\nOverall Performance:")
    print(f"  MAE: {summary['mae_mean']:.3f} (95% CI: [{summary['mae_ci_lower']:.3f}, {summary['mae_ci_upper']:.3f}])")
    print(f"  RMSE: {summary['rmse_mean']:.3f} (95% CI: [{summary['rmse_ci_lower']:.3f}, {summary['rmse_ci_upper']:.3f}])")
    if not np.isnan(summary['auc_mean']):
        print(f"  AUC: {summary['auc_mean']:.3f} (95% CI: [{summary['auc_ci_lower']:.3f}, {summary['auc_ci_upper']:.3f}])")
    else:
        print(f"  AUC: N/A (insufficient data)")

    print("="*60)


def create_model_comparison(all_results, model_type, dataset_type):
    """Create and save model comparison table."""
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON SUMMARY FOR {model_type.upper()} ON {dataset_type.upper()}")
    print(f"{'='*80}")

    # Create comparison table
    comparison_data = []
    for model_name, results in all_results.items():
        summary = results['summary']
        comparison_data.append({
            'Model': model_name,
            'MAE': f"{summary['mae_mean']:.3f}",
            'MAE_CI': f"[{summary['mae_ci_lower']:.3f}, {summary['mae_ci_upper']:.3f}]",
            'RMSE': f"{summary['rmse_mean']:.3f}",
            'RMSE_CI': f"[{summary['rmse_ci_lower']:.3f}, {summary['rmse_ci_upper']:.3f}]",
            'AUC': f"{summary['auc_mean']:.3f}" if not np.isnan(summary['auc_mean']) else "N/A",
            'AUC_CI': f"[{summary['auc_ci_lower']:.3f}, {summary['auc_ci_upper']:.3f}]" if not np.isnan(summary['auc_mean']) else "N/A"
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))

    # Save comparison table
    os.makedirs("outputs/tables", exist_ok=True)
    comparison_df.to_csv(f"outputs/tables/{dataset_type}_{model_type}_model_comparison.csv", index=False)
    print(f"\nModel comparison saved to: outputs/tables/{dataset_type}_{model_type}_model_comparison.csv")

    # Find best models
    best_mae_model = min(all_results.items(), key=lambda x: x[1]['summary']['mae_mean'])
    best_rmse_model = min(all_results.items(), key=lambda x: x[1]['summary']['rmse_mean'])
    
    # Find best AUC model (only if we have valid AUC values)
    valid_auc_models = [(name, results) for name, results in all_results.items() if not np.isnan(results['summary']['auc_mean'])]
    if valid_auc_models:
        best_auc_model = max(valid_auc_models, key=lambda x: x[1]['summary']['auc_mean'])
        print(f"\nBest Models:")
        print(f"  Best MAE: {best_mae_model[0]} ({best_mae_model[1]['summary']['mae_mean']:.3f})")
        print(f"  Best RMSE: {best_rmse_model[0]} ({best_rmse_model[1]['summary']['rmse_mean']:.3f})")
        print(f"  Best AUC: {best_auc_model[0]} ({best_auc_model[1]['summary']['auc_mean']:.3f})")
    else:
        print(f"\nBest Models:")
        print(f"  Best MAE: {best_mae_model[0]} ({best_mae_model[1]['summary']['mae_mean']:.3f})")
        print(f"  Best RMSE: {best_rmse_model[0]} ({best_rmse_model[1]['summary']['rmse_mean']:.3f})")
        print(f"  Best AUC: N/A (no valid classification results)")


def main(target_type='gestational_age'):
    """
    Main execution function.
    
    Args:
        target_type (str): Target variable type ('gestational_age' or 'birth_weight')
    """
    # Clear scatter plot directories to force regeneration
    import shutil
    import os
    
    print(f"\n{'='*80}")
    print("CLEARING SCATTER PLOT DIRECTORIES TO FORCE REGENERATION")
    print(f"{'='*80}")
    
    for target in [
        "outputs/plots/gestational_age/scatter_plots",
        "outputs/plots/birth_weight/scatter_plots"
    ]:
        if os.path.exists(target):
            shutil.rmtree(target, ignore_errors=True)
            print(f"✅ Cleared directory: {target}")
        os.makedirs(target, exist_ok=True)
        print(f"✅ Created directory: {target}")
    
    # Configuration
    dataset_types = ['heel', 'cord']
    model_types = ['lasso_cv', 'elasticnet_cv', 'stabl']
    model_configs = [
        {'name': 'Clinical', 'data_type': 'clinical', 'allowed_models': ['lasso_cv', 'elasticnet_cv', 'stabl']},
        {'name': 'Biomarker', 'data_type': 'biomarker', 'allowed_models': ['lasso_cv', 'elasticnet_cv', 'stabl']},
        {'name': 'Combined', 'data_type': 'combined', 'allowed_models': ['lasso_cv', 'elasticnet_cv', 'stabl']}
    ]
    
    print(f"\n{'='*80}")
    print(f"RUNNING PIPELINE FOR TARGET: {target_type.upper()}")
    print(f"{'='*80}")
    summary_rows = []
    all_results = {}
    for data_option in [1, 2, 3]:
        data_option_label = DATA_OPTION_LABELS[data_option]
        print(f"\n{'#'*80}")
        print(f"RUNNING DATA OPTION {data_option}: {data_option_label}")
        print(f"{'#'*80}")
        
        # Determine which dataset types to use based on data option
        if data_option == 1:
            # Option 1: Use both heel and cord datasets (patients with both samples)
            current_dataset_types = ['heel', 'cord']
        elif data_option == 2:
            # Option 2: Use only heel dataset (all heel samples)
            current_dataset_types = ['heel']
        elif data_option == 3:
            # Option 3: Use only cord dataset (all cord samples)
            current_dataset_types = ['cord']
        else:
            raise ValueError(f"Invalid data_option: {data_option}")
        
        print(f"📊 Using dataset types: {current_dataset_types}")
        
        for dataset_type in current_dataset_types:
            print(f"\n{'='*80}")
            print(f"TRAINING ON {dataset_type.upper()} DATASET [{data_option_label}]")
            print(f"{'='*80}")
            for model_type in model_types:
                print(f"\n{'='*80}")
                print(f"TRAINING WITH {model_type.upper()} MODEL ON {dataset_type.upper()} DATA [{data_option_label}]")
                print(f"{'='*80}")
                model_results = {}
                for config in model_configs:
                    model_name = config['name']
                    data_type = config['data_type']
                    allowed_models = config['allowed_models']
                    if model_type not in allowed_models:
                        print(f"Skipping {model_name} model with {model_type} (not allowed)")
                        continue
                    results = run_single_model(model_name, data_type, dataset_type, model_type, data_option, data_option_label, target_type)
                    model_results[model_name] = results
                    all_results[f"{data_option_label}_{dataset_type}_{model_type}_{model_name}"] = results
                    # Update output/plot filenames in generate_model_outputs as well
                    generate_model_outputs(model_name, data_type, dataset_type, model_type, results, None, None, results['summary']['auc_mean'], data_option_label, target_type)
                    print_detailed_summary(model_name, dataset_type, model_type, results)
                    # --- Collect summary row for MAE, RMSE, AUC (for summary plots) ---
                    mae_mean = results['summary']['mae_mean']
                    mae_ci_lower = results['summary']['mae_ci_lower']
                    mae_ci_upper = results['summary']['mae_ci_upper']
                    rmse_mean = results['summary']['rmse_mean']
                    rmse_ci_lower = results['summary']['rmse_ci_lower']
                    rmse_ci_upper = results['summary']['rmse_ci_upper']
                    auc_mean = results['summary']['auc_mean']
                    auc_ci_lower = results['summary']['auc_ci_lower']
                    auc_ci_upper = results['summary']['auc_ci_upper']
                    summary_rows.append({
                        'Dataset': dataset_type,
                        'Model': model_name,
                        'ModelType': model_type,  # Add model type to distinguish between Lasso, ElasticNet, STABL
                        'DataOption': data_option_label,  # Add data option to distinguish between both_samples, heel_all, cord_all
                        'MAE': mae_mean,
                        'MAE_CI_Lower': mae_ci_lower,
                        'MAE_CI_Upper': mae_ci_upper,
                        'RMSE': rmse_mean,
                        'RMSE_CI_Lower': rmse_ci_lower,
                        'RMSE_CI_Upper': rmse_ci_upper,
                        'AUC': auc_mean,
                        'AUC_CI_Lower': auc_ci_lower,
                        'AUC_CI_Upper': auc_ci_upper
                    })

                # Create model comparison for this model type (only if we have results)
                if model_results:
                    create_model_comparison(model_results, model_type, f"{data_option_label}_{dataset_type}")

                    # Final summary for this model type
                    print(f"\n{'='*80}")
                    print(f"ALL MODELS COMPLETED FOR {model_type.upper()} ON {dataset_type.upper()}!")
                    print("="*80)
                    print("Generated files:")
                    for model_name in model_results.keys():
                        print(f"  {model_name} model:")
                        print(f"    - Performance table: outputs/tables/{data_option_label}_{dataset_type}_{model_type}_{model_name.lower()}_performance_metrics.csv")
                        print(f"    - Performance plots: outputs/plots/{data_option_label}_{dataset_type}_{model_type}_{model_name.lower()}_metrics_with_ci.png")
                        # print(f"    - ROC curve: outputs/plots/{dataset_type}_{model_type}_{model_name.lower()}_roc_avg_over_runs.png") # Removed ROC curve
                        if model_name == 'Biomarker':
                            print(f"    - Biomarker frequency: outputs/plots/{data_option_label}_{dataset_type}_{model_type}_{model_name.lower()}_biomarker_frequency.png")
                    print(f"  - Model comparison: outputs/tables/{data_option_label}_{dataset_type}_{model_type}_model_comparison.csv")
                    print("="*80)
                else:
                    print(f"\n{'='*80}")
                    print(f"NO MODELS TRAINED FOR {model_type.upper()} ON {dataset_type.upper()}!")
                    print("="*80)

    # --- After all training, generate separate summary plots for each model type and data option ---
    summary_df = pd.DataFrame(summary_rows)
    print('summary_df contents:')
    print(summary_df)
    
    # Create separate plots for each model type and data option (including AUC)
    for model_type in ['lasso_cv', 'elasticnet_cv', 'stabl']:
        for data_option in [1, 2, 3]:
            data_option_label = DATA_OPTION_LABELS[data_option]
            
            # Filter data for this model type and data option
            model_df = summary_df[(summary_df['ModelType'] == model_type) & (summary_df['DataOption'] == data_option_label)].copy()
            if not model_df.empty:
                # Group by Dataset and Model (feature sets) for this specific model type
                model_df_grouped = model_df.groupby(["Dataset", "Model"], as_index=False).mean(numeric_only=True)
                # Capitalize model type for label (remove _cv suffix)
                model_type_label = model_type.replace('_cv', '').capitalize()
                
                # Create summary plots for this specific data option
                plot_summary_mae_by_dataset_and_model(
                    model_df_grouped, 
                    filename=f"outputs/plots/summary_mae_by_dataset_and_model_{model_type}_{data_option_label}_{target_type}.png",
                    model_type_label=f"{model_type_label} ({data_option_label})"
                )
                plot_summary_rmse_by_dataset_and_model(
                    model_df_grouped, 
                    filename=f"outputs/plots/summary_rmse_by_dataset_and_model_{model_type}_{data_option_label}_{target_type}.png",
                    model_type_label=f"{model_type_label} ({data_option_label})"
                )
                # Add AUC summary plot
                plot_summary_auc_by_dataset_and_model(
                    model_df_grouped, 
                    filename=f"outputs/plots/summary_auc_by_dataset_and_model_{model_type}_{data_option_label}_{target_type}.png",
                    model_type_label=f"{model_type_label} ({data_option_label})"
                )
    
    # Also create the original aggregated plots for comparison
    for model_type in ['lasso_cv', 'elasticnet_cv', 'stabl']:
        # Filter data for this model type
        model_df = summary_df[summary_df['ModelType'] == model_type].copy()
        if not model_df.empty:
            # Group by Dataset and Model (feature sets) for this specific model type
            model_df_grouped = model_df.groupby(["Dataset", "Model"], as_index=False).mean(numeric_only=True)
            # Capitalize model type for label (remove _cv suffix)
            model_type_label = model_type.replace('_cv', '').capitalize()
            plot_summary_mae_by_dataset_and_model(
                model_df_grouped, 
                filename=f"outputs/plots/summary_mae_by_dataset_and_model_{model_type}_{target_type}.png",
                model_type_label=model_type_label
            )
            plot_summary_rmse_by_dataset_and_model(
                model_df_grouped, 
                filename=f"outputs/plots/summary_rmse_by_dataset_and_model_{model_type}_{target_type}.png",
                model_type_label=model_type_label
            )
            # Add AUC summary plot
            plot_summary_auc_by_dataset_and_model(
                model_df_grouped, 
                filename=f"outputs/plots/summary_auc_by_dataset_and_model_{model_type}_{target_type}.png",
                model_type_label=model_type_label
            )
    
    # --- Create combined scatter plots with all model types (Clinical, Biomarker, Combined) ---
    # Create scatter plots for each data option that show all model types together
    for data_option in [1, 2, 3]:
        data_option_label = DATA_OPTION_LABELS[data_option]
        
        # Determine which datasets to process based on data option
        if data_option == 1:
            # Option 1: both samples - process both heel and cord
            datasets_to_process = ['heel', 'cord']
        elif data_option == 2:
            # Option 2: all heel - only process heel
            datasets_to_process = ['heel']
        elif data_option == 3:
            # Option 3: all cord - only process cord
            datasets_to_process = ['cord']
        
        for dataset in datasets_to_process:
            # Collect predictions from all model types for this dataset and data option
            pred_rows = []
            
            for model_type in ['lasso_cv', 'elasticnet_cv', 'stabl']:
                for model_name in ['Clinical', 'Biomarker', 'Combined']:
                    result_key = f"{data_option_label}_{dataset}_{model_type}_{model_name}"
                    
                    if result_key in all_results:
                        result = all_results[result_key]
                        for run in result.get('predictions', []):
                            true_vals = run.get('true', [])
                            pred_vals = run.get('pred', [])
                            for t, p in zip(true_vals, pred_vals):
                                # Determine column names based on target type
                                if target_type == 'birth_weight':
                                    true_col = 'True_BW'
                                    pred_col = 'Pred_BW'
                                else:  # gestational_age
                                    true_col = 'True_GA'
                                    pred_col = 'Pred_GA'
                                
                                pred_rows.append({
                                    true_col: float(t),
                                    pred_col: float(p),
                                    'Model': model_name,
                                    'ModelType': model_type,
                                    'Dataset': dataset,
                                    'DataOption': data_option_label
                                })
            
            if pred_rows:
                preds_df = pd.DataFrame(pred_rows)
                plot_true_vs_predicted_scatter(
                    preds_df,
                    filename=f"outputs/plots/true_vs_predicted_scatter_{data_option_label}_{dataset}_{target_type}.png",
                    target_type=target_type
                )
                print(f"Combined scatter plot for {data_option_label} {dataset} saved: true_vs_predicted_scatter_{data_option_label}_{dataset}_{target_type}.png")
    
    # --- Create separate scatter plots per model type and data option ---
    print(f"\n{'='*80}")
    print("GENERATING SEPARATE SCATTER PLOTS PER MODEL TYPE")
    print(f"{'='*80}")
    
    for model_type in ['lasso_cv', 'elasticnet_cv', 'stabl']:
        model_display_name = model_type.replace('_cv', '').capitalize()
        print(f"\n--- Generating separate scatter plots for {model_display_name} ---")
        
        for data_option in [1, 2, 3]:
            data_option_label = DATA_OPTION_LABELS[data_option]
            
            # Determine which datasets to process based on data option
            if data_option == 1:
                datasets_to_process = ['heel', 'cord']
            elif data_option == 2:
                datasets_to_process = ['heel']
            elif data_option == 3:
                datasets_to_process = ['cord']
            
            # For both_samples (data_option = 1), combine heel and cord data on one plot
            if data_option == 1:
                # Collect predictions for both heel and cord datasets
                pred_rows = []
                
                for dataset in datasets_to_process:
                    for model_name in ['Clinical', 'Biomarker', 'Combined']:
                        result_key = f"{data_option_label}_{dataset}_{model_type}_{model_name}"
                        
                        if result_key in all_results:
                            result = all_results[result_key]
                            for run in result.get('predictions', []):
                                true_vals = run.get('true', [])
                                pred_vals = run.get('pred', [])
                                for t, p in zip(true_vals, pred_vals):
                                    # Determine column names based on target type
                                    if target_type == 'birth_weight':
                                        true_col = 'True_BW'
                                        pred_col = 'Pred_BW'
                                    else:  # gestational_age
                                        true_col = 'True_GA'
                                        pred_col = 'Pred_GA'
                                    
                                    pred_rows.append({
                                        true_col: float(t),
                                        pred_col: float(p),
                                        'Model': model_name,
                                        'Dataset': dataset,
                                        'DataOption': data_option_label
                                    })
                
                if pred_rows:
                    preds_df = pd.DataFrame(pred_rows)
                    plot_true_vs_predicted_scatter(
                        preds_df,
                        filename=f"outputs/plots/true_vs_predicted_scatter_{model_display_name}_{data_option_label}_{target_type}.png",
                        target_type=target_type
                    )
                    print(f"  {model_display_name} scatter plot for {data_option_label} (heel + cord combined) saved")
                else:
                    print(f"  No predictions found for {model_display_name} on {data_option_label}")
            
            else:
                # For other data options (heel_all, cord_all), also combine heel and cord data on one plot
                # Collect predictions for both heel and cord datasets
                pred_rows = []
                
                for dataset in ['heel', 'cord']:  # Always include both datasets
                    for model_name in ['Clinical', 'Biomarker', 'Combined']:
                        result_key = f"{data_option_label}_{dataset}_{model_type}_{model_name}"
                        
                        if result_key in all_results:
                            result = all_results[result_key]
                            for run in result.get('predictions', []):
                                true_vals = run.get('true', [])
                                pred_vals = run.get('pred', [])
                                for t, p in zip(true_vals, pred_vals):
                                    # Determine column names based on target type
                                    if target_type == 'birth_weight':
                                        true_col = 'True_BW'
                                        pred_col = 'Pred_BW'
                                    else:  # gestational_age
                                        true_col = 'True_GA'
                                        pred_col = 'Pred_GA'
                                    
                                    pred_rows.append({
                                        true_col: float(t),
                                        pred_col: float(p),
                                        'Model': model_name,
                                        'Dataset': dataset,
                                        'DataOption': data_option_label
                                    })
                
                if pred_rows:
                    preds_df = pd.DataFrame(pred_rows)
                    plot_true_vs_predicted_scatter(
                        preds_df,
                        filename=f"outputs/plots/true_vs_predicted_scatter_{model_display_name}_{data_option_label}_{target_type}.png",
                        target_type=target_type
                    )
                    print(f"  {model_display_name} scatter plot for {data_option_label} (heel + cord combined) saved")
                else:
                    print(f"  No predictions found for {model_display_name} on {data_option_label}")
    
    # Also create overall combined scatter plots for each model type across all data options
    for model_type in ['lasso_cv', 'elasticnet_cv']:
        # Collect predictions from all data options and datasets for this model type
        pred_rows = []
        
        for data_option in [1, 2, 3]:
            data_option_label = DATA_OPTION_LABELS[data_option]
            
            # Determine which datasets to process based on data option
            if data_option == 1:
                datasets_to_process = ['heel', 'cord']
            elif data_option == 2:
                datasets_to_process = ['heel']
            elif data_option == 3:
                datasets_to_process = ['cord']
            
            for dataset in datasets_to_process:
                for model_name in ['Clinical', 'Biomarker', 'Combined']:
                    result_key = f"{data_option_label}_{dataset}_{model_type}_{model_name}"
                    
                    if result_key in all_results:
                        result = all_results[result_key]
                        for run in result.get('predictions', []):
                            true_vals = run.get('true', [])
                            pred_vals = run.get('pred', [])
                            for t, p in zip(true_vals, pred_vals):
                                # Determine column names based on target type
                                if target_type == 'birth_weight':
                                    true_col = 'True_BW'
                                    pred_col = 'Pred_BW'
                                else:  # gestational_age
                                    true_col = 'True_GA'
                                    pred_col = 'Pred_GA'
                                
                                pred_rows.append({
                                    true_col: float(t),
                                    pred_col: float(p),
                                    'Model': model_name,
                                    'ModelType': model_type,
                                    'Dataset': dataset,
                                    'DataOption': data_option_label
                                })
        
        if pred_rows:
            preds_df = pd.DataFrame(pred_rows)
            # Create model-specific scatter plot with clean model name
            model_display_name = model_type.replace('_cv', '').capitalize()
            plot_true_vs_predicted_scatter(
                preds_df,
                filename=f"outputs/plots/true_vs_predicted_scatter_{model_display_name}_{target_type}.png",
                target_type=target_type
            )
            print(f"Model-specific scatter plot for {model_display_name} saved: true_vs_predicted_scatter_{model_display_name}_{target_type}.png")
    
    # --- Create overall combined scatter plots that combine heel_all and cord_all data ---
    print(f"\n{'='*80}")
    print("GENERATING OVERALL COMBINED SCATTER PLOTS (HEEL_ALL + CORD_ALL)")
    print(f"{'='*80}")
    
    for model_type in ['lasso_cv', 'elasticnet_cv', 'stabl']:
        model_display_name = model_type.replace('_cv', '').capitalize()
        print(f"\n--- Creating overall combined plot for {model_display_name} ---")
        
        # Collect predictions from both heel_all and cord_all data options
        pred_rows = []
        
        for data_option_label in ['heel_all', 'cord_all']:
            print(f"  Processing {data_option_label}...")
            
            for dataset in ['heel', 'cord']:  # Include both datasets
                for model_name in ['Clinical', 'Biomarker', 'Combined']:
                    result_key = f"{data_option_label}_{dataset}_{model_type}_{model_name}"
                    
                    if result_key in all_results:
                        result = all_results[result_key]
                        for run in result.get('predictions', []):
                            true_vals = run.get('true', [])
                            pred_vals = run.get('pred', [])
                            for t, p in zip(true_vals, pred_vals):
                                # Determine column names based on target type
                                if target_type == 'birth_weight':
                                    true_col = 'True_BW'
                                    pred_col = 'Pred_BW'
                                else:  # gestational_age
                                    true_col = 'True_GA'
                                    pred_col = 'Pred_GA'
                                
                                pred_rows.append({
                                    true_col: float(t),
                                    pred_col: float(p),
                                    'Model': model_name,
                                    'Dataset': dataset,
                                    'DataOption': data_option_label
                                })
        
        if pred_rows:
            preds_df = pd.DataFrame(pred_rows)
            plot_true_vs_predicted_scatter(
                preds_df,
                filename=f"outputs/plots/true_vs_predicted_scatter_{model_display_name}_overall_{target_type}.png",
                target_type=target_type
            )
            print(f"  {model_display_name} overall combined scatter plot saved")
            print(f"  Total predictions: {len(pred_rows)}")
            
            # Count predictions by data option and dataset
            heel_all_heel = sum(1 for row in pred_rows if row['DataOption'] == 'heel_all' and row['Dataset'] == 'heel')
            heel_all_cord = sum(1 for row in pred_rows if row['DataOption'] == 'heel_all' and row['Dataset'] == 'cord')
            cord_all_heel = sum(1 for row in pred_rows if row['DataOption'] == 'cord_all' and row['Dataset'] == 'heel')
            cord_all_cord = sum(1 for row in pred_rows if row['DataOption'] == 'cord_all' and row['Dataset'] == 'cord')
            
            print(f"  Heel_all heel: {heel_all_heel}")
            print(f"  Heel_all cord: {heel_all_cord}")
            print(f"  Cord_all heel: {cord_all_heel}")
            print(f"  Cord_all cord: {cord_all_cord}")
        else:
            print(f"  No predictions found for {model_display_name}")
    
    # Create the final overall scatter plot with all models combined
    all_pred_rows = []
    for data_option in [1, 2, 3]:
        data_option_label = DATA_OPTION_LABELS[data_option]
        
        # Determine which datasets to process based on data option
        if data_option == 1:
            datasets_to_process = ['heel', 'cord']
        elif data_option == 2:
            datasets_to_process = ['heel']
        elif data_option == 3:
            datasets_to_process = ['cord']
        
        for dataset in datasets_to_process:
            for model_type in ['lasso_cv', 'elasticnet_cv', 'stabl']:
                for model_name in ['Clinical', 'Biomarker', 'Combined']:
                    result_key = f"{data_option_label}_{dataset}_{model_type}_{model_name}"
                    
                    if result_key in all_results:
                        result = all_results[result_key]
                        for run in result.get('predictions', []):
                            true_vals = run.get('true', [])
                            pred_vals = run.get('pred', [])
                            for t, p in zip(true_vals, pred_vals):
                                # Determine column names based on target type
                                if target_type == 'birth_weight':
                                    true_col = 'True_BW'
                                    pred_col = 'Pred_BW'
                                else:  # gestational_age
                                    true_col = 'True_GA'
                                    pred_col = 'Pred_GA'
                                
                                all_pred_rows.append({
                                    true_col: float(t),
                                    pred_col: float(p),
                                    'Model': model_name,
                                    'ModelType': model_type,
                                    'Dataset': dataset,
                                    'DataOption': data_option_label
                                })
    
    if all_pred_rows:
        all_preds_df = pd.DataFrame(all_pred_rows)
        plot_true_vs_predicted_scatter(all_preds_df, filename=f"outputs/plots/true_vs_predicted_scatter_{target_type}.png", target_type=target_type)
        print(f"Final overall scatter plot saved: true_vs_predicted_scatter_{target_type}.png")
    else:
        print("No predictions found for scatter plot!")

    # Create biomarker frequency plots
    # plot_biomarker_frequency_heel_vs_cord(all_results, filename="outputs/plots/biomarker_frequency.png") # This line is removed

    # Create biomarker frequency comparison plots (Heel vs Cord) for each model type
    for model_type in ['lasso_cv', 'elasticnet_cv', 'stabl']:
        # Option 1: Heel vs Cord comparison using both_samples data
        option1_results = {k: v for k, v in all_results.items() if 'both_samples' in k}
        if option1_results:
            plot_biomarker_frequency_heel_vs_cord(
                option1_results, 
                model_type, 
                filename=f"outputs/plots/biomarker_frequency_heel_vs_cord_{model_type}_{target_type}_option1_both_samples.png",
                target_type=target_type
            )
            print(f"Option 1 (both_samples) heel vs cord comparison plot saved: biomarker_frequency_heel_vs_cord_{model_type}_{target_type}_option1_both_samples.png")
        
        # Options 2+3: Heel vs Cord comparison using heel_all and cord_all data
        option2_3_results = {k: v for k, v in all_results.items() if 'heel_all' in k or 'cord_all' in k}
        if option2_3_results:
            plot_biomarker_frequency_heel_vs_cord(
                option2_3_results, 
                model_type, 
                filename=f"outputs/plots/biomarker_frequency_heel_vs_cord_{model_type}_{target_type}_options2_3_heel_cord_all.png",
                target_type=target_type
            )
            print(f"Options 2+3 (heel_all + cord_all) heel vs cord comparison plot saved: biomarker_frequency_heel_vs_cord_{model_type}_{target_type}_options2_3_heel_cord_all.png")

    # --- Generate biomarker frequency plots for BEST models only (regression and classification) ---
    print(f"\n{'='*80}")
    print("GENERATING BIOMARKER FREQUENCY PLOTS FOR BEST MODELS")
    print(f"{'='*80}")
    
    # Find the best regression model and best classification model for each dataset type
    best_models = {
        'Clinical': {'regression': None, 'classification': None},
        'Biomarker': {'regression': None, 'classification': None},
        'Combined': {'regression': None, 'classification': None}
    }
    
    # Initialize best performance metrics for each dataset type
    best_rmse = {'Clinical': float('inf'), 'Biomarker': float('inf'), 'Combined': float('inf')}
    best_auc = {'Clinical': 0.0, 'Biomarker': 0.0, 'Combined': 0.0}
    
    for data_option in [1, 2, 3]:
        data_option_label = DATA_OPTION_LABELS[data_option]
        
        # Determine which datasets to process based on data option
        if data_option == 1:
            datasets_to_process = ['heel', 'cord']
        elif data_option == 2:
            datasets_to_process = ['heel']
        elif data_option == 3:
            datasets_to_process = ['cord']
        
        for dataset in datasets_to_process:
            for model_type in ['lasso_cv', 'elasticnet_cv', 'stabl']:
                for model_name in ['Clinical', 'Biomarker', 'Combined']:
                    result_key = f"{data_option_label}_{dataset}_{model_type}_{model_name}"
                    
                    if result_key not in all_results:
                        continue
                    
                    results = all_results[result_key]
                    
                    # Check for best regression model for this dataset type (lowest RMSE)
                    if 'summary' in results and 'rmse_mean' in results['summary']:
                        rmse = results['summary']['rmse_mean']
                        if rmse < best_rmse[model_name]:
                            best_rmse[model_name] = rmse
                            best_models[model_name]['regression'] = {
                                'key': result_key,
                                'results': results,
                                'model_type': model_type,
                                'model_name': model_name,
                                'data_option': data_option_label,
                                'dataset': dataset,
                                'rmse': rmse
                            }
                    
                    # Check for best classification model for this dataset type (highest AUC)
                    if 'summary' in results and 'auc_mean' in results['summary']:
                        auc = results['summary']['auc_mean']
                        if auc > best_auc[model_name]:
                            best_auc[model_name] = auc
                            best_models[model_name]['classification'] = {
                                'key': result_key,
                                'results': results,
                                'model_type': model_type,
                                'model_name': model_name,
                                'data_option': data_option_label,
                                'dataset': dataset,
                                'auc': auc
                            }
    
    # Generate plots for each dataset type's best models
    for dataset_type in ['Clinical', 'Biomarker', 'Combined']:
        print(f"\n{'='*60}")
        print(f"BEST MODELS FOR {dataset_type.upper()} DATASET")
        print(f"{'='*60}")
        
        # Best regression model for this dataset type
        if best_models[dataset_type]['regression']:
            best_reg = best_models[dataset_type]['regression']
            print(f"\n--- Best {dataset_type} Regression Model ---")
            print(f"Model: {best_reg['model_type']} {best_reg['model_name']}")
            print(f"Dataset: {best_reg['dataset']} ({best_reg['data_option']})")
            print(f"RMSE: {best_reg['rmse']:.4f}")
            
            results = best_reg['results']
            if 'all_coefficients' in results and results['all_coefficients'] is not None:
                freq = count_high_weight_biomarkers(results['all_coefficients'], results['feature_names'], threshold=0.01)
                
                plot_feature_frequency(
                    results['feature_names'],
                    freq,
                    filename=f"outputs/plots/best_{dataset_type.lower()}_regression_biomarker_frequency_{target_type}.png",
                    model_name=f"{best_reg['model_type'].replace('_cv', '').capitalize()} {best_reg['model_name']}",
                    dataset_name=f"{best_reg['dataset'].capitalize()} ({best_reg['data_option']}) - {target_type} Regression (RMSE: {best_reg['rmse']:.4f})"
                )
                print(f"Best {dataset_type} regression biomarker frequency plot saved")
        
        # Best classification model for this dataset type
        if best_models[dataset_type]['classification']:
            best_cls = best_models[dataset_type]['classification']
            print(f"\n--- Best {dataset_type} Classification Model ---")
            print(f"Model: {best_cls['model_type']} {best_cls['model_name']}")
            print(f"Dataset: {best_cls['dataset']} ({best_cls['data_option']})")
            print(f"AUC: {best_cls['auc']:.4f}")
            
            results = best_cls['results']
            if 'all_classification_coefficients' in results and results['all_classification_coefficients'] is not None:
                classification_freq = count_high_weight_biomarkers(results['all_classification_coefficients'], results['feature_names'], threshold=0.01)
                
                # Determine classification type for title
                if target_type == 'gestational_age':
                    classification_type = 'preterm_classification'
                    classification_title = 'Preterm Classification'
                else:  # target_type == 'birth_weight'
                    classification_type = 'sga_classification'
                    classification_title = 'SGA Classification'
                
                plot_feature_frequency(
                    results['feature_names'],
                    classification_freq,
                    filename=f"outputs/plots/best_{dataset_type.lower()}_classification_biomarker_frequency_{classification_type}.png",
                    model_name=f"{best_cls['model_type'].replace('_cv', '').capitalize()} {best_cls['model_name']}",
                    dataset_name=f"{best_cls['dataset'].capitalize()} ({best_cls['data_option']}) - {classification_title} (AUC: {best_cls['auc']:.4f})"
                )
                print(f"Best {dataset_type} classification biomarker frequency plot saved")

    # --- Generate heel vs cord comparison plots for BEST models from each dataset type ---
    print(f"\n{'='*80}")
    print("GENERATING HEEL VS CORD COMPARISON PLOTS FOR BEST MODELS")
    print(f"{'='*80}")
    
    for dataset_type in ['Clinical', 'Biomarker', 'Combined']:
        print(f"\n{'='*60}")
        print(f"HEEL VS CORD COMPARISONS FOR {dataset_type.upper()} DATASET")
        print(f"{'='*60}")
        
        # Generate heel vs cord comparison for best regression model of this dataset type
        if best_models[dataset_type]['regression']:
            best_reg = best_models[dataset_type]['regression']
            print(f"\n--- Best {dataset_type} Regression Model Heel vs Cord Comparison ---")
            
            # Option 1: Heel vs Cord comparison using both_samples data
            option1_results = {k: v for k, v in all_results.items() if 'both_samples' in k and best_reg['model_name'] in k and best_reg['model_type'] in k}
            if option1_results:
                plot_biomarker_frequency_heel_vs_cord(
                    option1_results, 
                    best_reg['model_type'], 
                    filename=f"outputs/plots/best_{dataset_type.lower()}_regression_heel_vs_cord_{target_type}_option1_both_samples.png",
                    target_type=target_type
                )
                print(f"Best {dataset_type} regression heel vs cord comparison plot (both_samples) saved")
            
            # Options 2+3: Heel vs Cord comparison using heel_all and cord_all data
            option2_3_results = {k: v for k, v in all_results.items() if ('heel_all' in k or 'cord_all' in k) and best_reg['model_name'] in k and best_reg['model_type'] in k}
            if option2_3_results:
                plot_biomarker_frequency_heel_vs_cord(
                    option2_3_results, 
                    best_reg['model_type'], 
                    filename=f"outputs/plots/best_{dataset_type.lower()}_regression_heel_vs_cord_{target_type}_options2_3_heel_cord_all.png",
                    target_type=target_type
                )
                print(f"Best {dataset_type} regression heel vs cord comparison plot (heel_all + cord_all) saved")
        
        # Generate heel vs cord comparison for best classification model of this dataset type
        if best_models[dataset_type]['classification']:
            best_cls = best_models[dataset_type]['classification']
            print(f"\n--- Best {dataset_type} Classification Model Heel vs Cord Comparison ---")
            
            # Option 1: Heel vs Cord comparison using both_samples data
            option1_results = {k: v for k, v in all_results.items() if 'both_samples' in k and best_cls['model_name'] in k and best_cls['model_type'] in k}
            if option1_results:
                plot_biomarker_frequency_heel_vs_cord(
                    option1_results, 
                    best_cls['model_type'], 
                    filename=f"outputs/plots/best_{dataset_type.lower()}_classification_heel_vs_cord_{target_type}_option1_both_samples.png",
                    target_type=target_type
                )
                print(f"Best {dataset_type} classification heel vs cord comparison plot (both_samples) saved")
            
            # Options 2+3: Heel vs Cord comparison using heel_all and cord_all data
            option2_3_results = {k: v for k, v in all_results.items() if ('heel_all' in k or 'cord_all' in k) and best_cls['model_name'] in k and best_cls['model_type'] in k}
            if option2_3_results:
                plot_biomarker_frequency_heel_vs_cord(
                    option2_3_results, 
                    best_cls['model_type'], 
                    filename=f"outputs/plots/best_{dataset_type.lower()}_classification_heel_vs_cord_{target_type}_options2_3_heel_cord_all.png",
                    target_type=target_type
                )
                print(f"Best {dataset_type} classification heel vs cord comparison plot (heel_all + cord_all) saved")

    # --- Final summary ---
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*80}")
    print("All regression and classification models completed successfully!")
    print("Generated plots:")
    print("- Individual model type plots (MAE, RMSE, AUC)")
    print("- True vs predicted scatter plots")
    print("- ROC curves for classification")
    print("- Biomarker frequency comparison plots (Heel vs Cord)")
    print("- Best model biomarker frequency plots for each dataset type (Clinical, Biomarker, Combined)")
    print("- Heel vs cord comparison plots for best models from each dataset type")
    
    # Save all_results for inspection
    import pickle
    if target_type == 'gestational_age':
        with open("all_results_gestational_age.pkl", "wb") as f:
            pickle.dump(all_results, f)
        print("Saved all_results_gestational_age.pkl for coefficient inspection")
    elif target_type == 'birth_weight':
        with open("all_results_birth_weight.pkl", "wb") as f:
            pickle.dump(all_results, f)
        print("Saved all_results_birth_weight.pkl for coefficient inspection")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        target_type = sys.argv[1].lower()
        if target_type not in ['gestational_age', 'birth_weight']:
            print("Error: target_type must be 'gestational_age' or 'birth_weight'")
            print("Usage: python3 main.py [gestational_age|birth_weight]")
            sys.exit(1)
    else:
        target_type = 'gestational_age'  # Default
    
    # Run the pipeline
    main(target_type)