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
from sklearn.metrics import roc_auc_score, roc_curve
import pickle

from src.data_loader import split_data, load_and_process_data
from src.model import train_model, predict_model, get_model
from src.metrics import (
    average_auc, compute_mae, compute_rmse, 
    compute_metrics_by_gestational_age, compute_confidence_interval
)
from src.utils import (
    plot_feature_frequency, plot_roc_curve, save_all_as_pickle,
    count_high_weight_biomarkers, create_metrics_table,
    plot_metrics_with_confidence_intervals
)
from src.config import N_REPEATS, TEST_SIZE, PRETERM_CUTOFF


def run_single_model(model_name, data_type, dataset_type, model_type):
    """
    Run a single model configuration and return results.
    
    Args:
        model_name (str): Name of the model (Clinical, Biomarker, Combined)
        data_type (str): Type of data to use (clinical, biomarker, combined)
        dataset_type (str): Dataset type (cord, heel)
        model_type (str): Model algorithm (lasso, elasticnet, stabl)
    
    Returns:
        dict: Model results and summary statistics
    """
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name} ({data_type.upper()} DATA) - {model_type.upper()} ON {dataset_type.upper()}")
    print(f"{'='*60}")
    
    # Load and prepare data for this model
    X, y = load_and_process_data(dataset_type, model_type=data_type)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Initialize results storage
    aucs = []
    all_coefficients = []
    all_fpr = np.linspace(0, 1, 100)  # fixed FPR values for ROC averaging
    tpr_list = []
    maes = []
    rmses = []
    preterm_metrics_list = []
    term_metrics_list = []

    # Train and evaluate over multiple repeats
    for i in range(N_REPEATS):
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=TEST_SIZE, random_state=i)
        
        # Get and train model
        model, base_estimator = get_model(model_type)
        
        if model_type == 'stabl':
            trained_model, base_estimator = train_model(X_train, y_train, model, base_estimator)
            y_preds = predict_model(model, X_test, base_estimator)
        else:
            trained_model, _ = train_model(X_train, y_train, model, None)
            y_preds = predict_model(model, X_test)

        # Convert to binary preterm outcome for AUC calculation
        y_test_binary = (y_test < PRETERM_CUTOFF).astype(int)
        y_pred_binary = (y_preds < PRETERM_CUTOFF).astype(int)
        y_pred_score = -y_preds  # lower GA = higher preterm risk

        # Calculate metrics
        auc = roc_auc_score(y_test_binary, y_pred_score)
        aucs.append(auc)

        # Interpolate TPR at fixed FPR for ROC averaging
        fpr, tpr, _ = roc_curve(y_test_binary, y_pred_score)
        interp_tpr = np.interp(all_fpr, fpr, tpr)
        interp_tpr[0] = 0.0  # ensure start at 0
        tpr_list.append(interp_tpr)
       
        # Compute MAE and RMSE for gestational age prediction
        mae = compute_mae(y_test, y_preds)
        rmse = compute_rmse(y_test, y_preds)
        maes.append(mae)
        rmses.append(rmse)

        # Compute separate metrics for preterm and term babies
        metrics_by_ga = compute_metrics_by_gestational_age(y_test, y_preds, PRETERM_CUTOFF)
        preterm_metrics_list.append(metrics_by_ga['preterm'])
        term_metrics_list.append(metrics_by_ga['term'])

        # Print first 5 runs for monitoring
        if i < 5:
            print(f"Run {i + 1}: AUC = {auc:.3f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}")
            print(f"  Preterm (n={metrics_by_ga['preterm']['count']}): MAE = {metrics_by_ga['preterm']['mae']:.2f}, RMSE = {metrics_by_ga['preterm']['rmse']:.2f}")
            print(f"  Term (n={metrics_by_ga['term']['count']}): MAE = {metrics_by_ga['term']['mae']:.2f}, RMSE = {metrics_by_ga['term']['rmse']:.2f}")

        # Save outputs per run
        output_data = {
            "coef": base_estimator.coef_ if base_estimator is not None else trained_model.coef_,
            "GA preds": y_preds,
            "GA true": y_test,
            "features": X_test,
            "preterm preds": y_pred_binary,
            "preterm true": y_test_binary,
            "AUC": auc,
            "RMSE": rmse,
            "MAE": mae,
            "preterm_metrics": metrics_by_ga['preterm'],
            "term_metrics": metrics_by_ga['term']
        }
        filename = f"outputs/models/{dataset_type}_{model_type}_{model_name.lower()}_run{i + 1}_model_outputs.pkl"
        save_all_as_pickle(output_data, filename=filename)

        # Collect coefficients for feature importance analysis
        all_coefficients.append(base_estimator.coef_ if base_estimator is not None else trained_model.coef_)

    # Compute confidence intervals and summary statistics
    mae_mean, mae_ci_lower, mae_ci_upper = compute_confidence_interval(maes)
    rmse_mean, rmse_ci_lower, rmse_ci_upper = compute_confidence_interval(rmses)
    auc_mean = average_auc(aucs)

    # Store results
    results = {
        'aucs': aucs,
        'maes': maes,
        'rmses': rmses,
        'preterm_metrics_list': preterm_metrics_list,
        'term_metrics_list': term_metrics_list,
        'all_coefficients': all_coefficients,
        'feature_names': X.columns,
        'summary': {
            'auc_mean': auc_mean,
            'mae_mean': mae_mean,
            'mae_ci_lower': mae_ci_lower,
            'mae_ci_upper': mae_ci_upper,
            'rmse_mean': rmse_mean,
            'rmse_ci_lower': rmse_ci_lower,
            'rmse_ci_upper': rmse_ci_upper
        }
    }

    # Print summary
    print(f"\n{model_name} Model Summary ({model_type.upper()} on {dataset_type.upper()}):")
    print(f"  AUC: {auc_mean:.3f}")
    print(f"  MAE: {mae_mean:.3f} (95% CI: [{mae_ci_lower:.3f}, {mae_ci_upper:.3f}])")
    print(f"  RMSE: {rmse_mean:.3f} (95% CI: [{rmse_ci_lower:.3f}, {rmse_ci_upper:.3f}])")

    # Generate plots and tables
    generate_model_outputs(model_name, data_type, dataset_type, model_type, results, all_fpr, tpr_list, auc_mean)
    
    return results


def generate_model_outputs(model_name, data_type, dataset_type, model_type, results, all_fpr, tpr_list, auc_mean):
    """Generate all output files for a single model."""
    
    # Plot averaged ROC curve
    mean_tpr = np.mean(tpr_list, axis=0)
    plot_roc_curve(
        all_fpr, mean_tpr, auc_mean,
        filename=f"outputs/plots/{dataset_type}_{model_type}_{model_name.lower()}_roc_avg_over_runs.png"
    )

    # Biomarker frequency plot (only for biomarker model)
    if data_type == 'biomarker':
        frequency = count_high_weight_biomarkers(results['all_coefficients'], results['feature_names'])
        plot_feature_frequency(
            results['feature_names'], 
            frequency, 
            filename=f"outputs/plots/{dataset_type}_{model_type}_{model_name.lower()}_biomarker_frequency.png"
        )

    # Create performance metrics table
    print(f"\nCreating performance metrics table and plots for {model_name} model ({model_type} on {dataset_type})...")
    create_metrics_table(
        results['maes'], results['rmses'], 
        results['preterm_metrics_list'], results['term_metrics_list'],
        filename=f"outputs/tables/{dataset_type}_{model_type}_{model_name.lower()}_performance_metrics.csv"
    )
    print(f"Performance metrics table saved to outputs/tables/{dataset_type}_{model_type}_{model_name.lower()}_performance_metrics.csv")

    # Create bar plots with confidence intervals
    plot_metrics_with_confidence_intervals(
        results['maes'], results['rmses'], 
        results['preterm_metrics_list'], results['term_metrics_list'],
        filename=f"outputs/plots/{dataset_type}_{model_type}_{model_name.lower()}_metrics_with_ci.png"
    )
    print(f"Performance plots with confidence intervals saved to outputs/plots/{dataset_type}_{model_type}_{model_name.lower()}_metrics_with_ci.png")


def print_detailed_summary(model_name, dataset_type, model_type, results):
    """Print detailed performance summary for a model."""
    print(f"\n{model_name} Model - Final Performance Summary")
    print("="*60)
    print(f"Dataset: {dataset_type.upper()}")
    print(f"Model: {model_type.upper()}")
    print(f"Number of runs: {N_REPEATS}")
    print(f"Preterm cutoff: {PRETERM_CUTOFF} weeks")
    print(f"Number of features: {len(results['feature_names'])}")
    
    summary = results['summary']
    print("\nOverall Performance:")
    print(f"  MAE: {summary['mae_mean']:.3f} (95% CI: [{summary['mae_ci_lower']:.3f}, {summary['mae_ci_upper']:.3f}])")
    print(f"  RMSE: {summary['rmse_mean']:.3f} (95% CI: [{summary['rmse_ci_lower']:.3f}, {summary['rmse_ci_upper']:.3f}])")
    print(f"  AUC: {summary['auc_mean']:.3f}")

    # Preterm performance
    preterm_maes = [m['mae'] for m in results['preterm_metrics_list'] if not np.isnan(m['mae'])]
    preterm_rmses = [m['rmse'] for m in results['preterm_metrics_list'] if not np.isnan(m['rmse'])]
    if preterm_maes:
        preterm_mae_mean, preterm_mae_ci_lower, preterm_mae_ci_upper = compute_confidence_interval(preterm_maes)
        preterm_rmse_mean, preterm_rmse_ci_lower, preterm_rmse_ci_upper = compute_confidence_interval(preterm_rmses)
        avg_preterm_count = np.mean([m['count'] for m in results['preterm_metrics_list']])
        print(f"\nPreterm Performance (avg n={avg_preterm_count:.1f}):")
        print(f"  MAE: {preterm_mae_mean:.3f} (95% CI: [{preterm_mae_ci_lower:.3f}, {preterm_mae_ci_upper:.3f}])")
        print(f"  RMSE: {preterm_rmse_mean:.3f} (95% CI: [{preterm_rmse_ci_lower:.3f}, {preterm_rmse_ci_upper:.3f}])")

    # Term performance
    term_maes = [m['mae'] for m in results['term_metrics_list'] if not np.isnan(m['mae'])]
    term_rmses = [m['rmse'] for m in results['term_metrics_list'] if not np.isnan(m['rmse'])]
    if term_maes:
        term_mae_mean, term_mae_ci_lower, term_mae_ci_upper = compute_confidence_interval(term_maes)
        term_rmse_mean, term_rmse_ci_lower, term_rmse_ci_upper = compute_confidence_interval(term_rmses)
        avg_term_count = np.mean([m['count'] for m in results['term_metrics_list']])
        print(f"\nTerm Performance (avg n={avg_term_count:.1f}):")
        print(f"  MAE: {term_mae_mean:.3f} (95% CI: [{term_mae_ci_lower:.3f}, {term_mae_ci_upper:.3f}])")
        print(f"  RMSE: {term_rmse_mean:.3f} (95% CI: [{term_rmse_ci_lower:.3f}, {term_rmse_ci_upper:.3f}])")

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
            'AUC': f"{summary['auc_mean']:.3f}",
            'MAE': f"{summary['mae_mean']:.3f}",
            'MAE_CI': f"[{summary['mae_ci_lower']:.3f}, {summary['mae_ci_upper']:.3f}]",
            'RMSE': f"{summary['rmse_mean']:.3f}",
            'RMSE_CI': f"[{summary['rmse_ci_lower']:.3f}, {summary['rmse_ci_upper']:.3f}]",
            'Features': len(results['feature_names'])
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\nModel Performance Comparison:")
    print(comparison_df.to_string(index=False))

    # Save comparison table
    comparison_df.to_csv(f"outputs/tables/{dataset_type}_{model_type}_model_comparison.csv", index=False)
    print(f"\nModel comparison saved to: outputs/tables/{dataset_type}_{model_type}_model_comparison.csv")

    # Find best models
    best_auc_model = max(all_results.items(), key=lambda x: x[1]['summary']['auc_mean'])
    best_mae_model = min(all_results.items(), key=lambda x: x[1]['summary']['mae_mean'])
    best_rmse_model = min(all_results.items(), key=lambda x: x[1]['summary']['rmse_mean'])

    print(f"\nBest Models:")
    print(f"  Best AUC: {best_auc_model[0]} ({best_auc_model[1]['summary']['auc_mean']:.3f})")
    print(f"  Best MAE: {best_mae_model[0]} ({best_mae_model[1]['summary']['mae_mean']:.3f})")
    print(f"  Best RMSE: {best_rmse_model[0]} ({best_rmse_model[1]['summary']['rmse_mean']:.3f})")


def main():
    """Main execution function."""
    # Configuration
    dataset_types = ['heel', 'cord']  # Train on both datasets
    model_types = ['lasso', 'elasticnet', 'stabl']  # Run all model types

    # Define the 3 model configurations as requested by PI
    model_configs = [
        {'name': 'Clinical', 'data_type': 'clinical'},
        {'name': 'Biomarker', 'data_type': 'biomarker'},
        {'name': 'Combined', 'data_type': 'combined'}
    ]

    # Run each dataset type
    for dataset_type in dataset_types:
        print(f"\n{'='*80}")
        print(f"TRAINING ON {dataset_type.upper()} DATASET")
        print(f"{'='*80}")
        
        # Run each model type
        for model_type in model_types:
            print(f"\n{'='*80}")
            print(f"TRAINING WITH {model_type.upper()} MODEL ON {dataset_type.upper()} DATA")
            print(f"{'='*80}")
            
            # Store results for all models for this model type
            all_results = {}

            # Run each model configuration
            for config in model_configs:
                model_name = config['name']
                data_type = config['data_type']
                
                # Run the model
                results = run_single_model(model_name, data_type, dataset_type, model_type)
                all_results[model_name] = results
                
                # Print detailed summary
                print_detailed_summary(model_name, dataset_type, model_type, results)

            # Create model comparison for this model type
            create_model_comparison(all_results, model_type, dataset_type)

            # Final summary for this model type
            print(f"\n{'='*80}")
            print(f"ALL MODELS COMPLETED FOR {model_type.upper()} ON {dataset_type.upper()}!")
            print("="*80)
            print("Generated files:")
            for model_name in all_results.keys():
                print(f"  {model_name} model:")
                print(f"    - Performance table: outputs/tables/{dataset_type}_{model_type}_{model_name.lower()}_performance_metrics.csv")
                print(f"    - Performance plots: outputs/plots/{dataset_type}_{model_type}_{model_name.lower()}_metrics_with_ci.png")
                print(f"    - ROC curve: outputs/plots/{dataset_type}_{model_type}_{model_name.lower()}_roc_avg_over_runs.png")
                if model_name == 'Biomarker':
                    print(f"    - Biomarker frequency: outputs/plots/{dataset_type}_{model_type}_{model_name.lower()}_biomarker_frequency.png")
            print(f"  - Model comparison: outputs/tables/{dataset_type}_{model_type}_model_comparison.csv")
            print("="*80)


if __name__ == "__main__":
    main()