#!/usr/bin/env python3
"""
Script to run only the plotting sections from main.py using existing all_results.pkl data.
This is useful when the main script was interrupted after model training but before plotting.
"""

import pickle
import pandas as pd
import numpy as np
import os
from src.utils import plot_true_vs_predicted_scatter, plot_biomarker_frequency_heel_vs_cord, count_high_weight_biomarkers, plot_feature_frequency

def run_plotting_only(target_type='gestational_age'):
    """
    Run only the plotting sections from main.py using existing results.
    
    Args:
        target_type (str): Target variable type ('gestational_age' or 'birth_weight')
    """
    print(f"\n{'='*80}")
    print(f"RUNNING PLOTTING ONLY FOR TARGET: {target_type.upper()}")
    print(f"{'='*80}")
    
    # Load existing results
    print("Loading existing results from all_results.pkl...")
    with open('all_results.pkl', 'rb') as f:
        all_results = pickle.load(f)
    
    print(f"Loaded {len(all_results)} model results")
    
    # Data option labels
    DATA_OPTION_LABELS = {
        1: 'both_samples',
        2: 'heel_all', 
        3: 'cord_all'
    }
    
    # Create summary DataFrame from existing results
    print("Creating summary DataFrame...")
    summary_rows = []
    for key, result in all_results.items():
        if target_type in key:
            # Parse key to extract components
            parts = key.split('_')
            if len(parts) >= 5:  # Need at least 5 parts for full key
                # Handle data option labels
                if parts[1] == 'all':
                    data_option_label = parts[0] + '_' + parts[1]
                    dataset = parts[2]
                    model_type = parts[3] + '_' + parts[4]
                    model_name = parts[5]
                else:
                    data_option_label = parts[0] + '_' + parts[1]
                    dataset = parts[2]
                    model_type = parts[3] + '_' + parts[4]
                    model_name = parts[5]
                
                summary = result['summary']
                summary_rows.append({
                    'DataOption': data_option_label,
                    'Dataset': dataset,
                    'ModelType': model_type,
                    'Model': model_name,
                    'MAE': summary['mae_mean'],
                    'MAE_CI_Lower': summary['mae_ci_lower'],
                    'MAE_CI_Upper': summary['mae_ci_upper'],
                    'RMSE': summary['rmse_mean'],
                    'RMSE_CI_Lower': summary['rmse_ci_lower'],
                    'RMSE_CI_Upper': summary['rmse_ci_upper'],
                    'AUC': summary['auc_mean'],
                    'AUC_CI_Lower': summary['auc_ci_lower'],
                    'AUC_CI_Upper': summary['auc_ci_upper']
                })
    
    summary_df = pd.DataFrame(summary_rows)
    print(f"Created summary DataFrame with {len(summary_df)} rows")
    print("Summary DataFrame columns:", summary_df.columns.tolist())
    print("Summary DataFrame head:")
    print(summary_df.head())
    
    # Create scatter plots for each data option
    print("\n" + "="*80)
    print("GENERATING SCATTER PLOTS")
    print("="*80)
    
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
            # Collect predictions from all model types for this dataset and data option
            pred_rows = []
            
            for model_type in ['lasso_cv', 'elasticnet_cv']:
                for model_name in ['Clinical', 'Biomarker', 'Combined']:
                    result_key = f"{data_option_label}_{dataset}_{model_type}_{model_name}_{target_type}"
                    
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
            else:
                print(f"No prediction data found for {data_option_label} {dataset}")
                # Debug: Check what keys are available
                available_keys = [k for k in all_results.keys() if data_option_label in k and dataset in k and target_type in k]
                print(f"Available keys for {data_option_label} {dataset}: {available_keys}")
    
    # Create biomarker frequency plots
    print("\n" + "="*80)
    print("GENERATING BIOMARKER FREQUENCY PLOTS")
    print("="*80)
    
    # Generate best model biomarker frequency plots for each data option
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
            # Filter summary_df for biomarker models of this dataset and data option
            biomarker_df = summary_df[(summary_df['Dataset'] == dataset) & (summary_df['Model'] == 'Biomarker') & (summary_df['DataOption'] == data_option_label)]
            if biomarker_df.empty:
                print(f"No biomarker models found for {dataset} dataset in {data_option_label}.")
                continue
            
            # Find the row with the lowest RMSE (best regression performance)
            best_row = biomarker_df.loc[biomarker_df['RMSE'].idxmin()]
            best_rmse = best_row['RMSE']
            best_model_type = best_row['ModelType']
            
            # Get the results from all_results dictionary
            result_key = f"{data_option_label}_{dataset}_{best_model_type}_Biomarker_{target_type}"
            print(f"Looking for key: {result_key}")
            print(f"Available keys: {[k for k in all_results.keys() if 'Biomarker' in k and data_option_label in k and dataset in k]}")
            
            if result_key not in all_results:
                print(f"Could not find results for {result_key} in all_results.")
                continue
                
            results = all_results[result_key]
            if 'all_coefficients' not in results or 'feature_names' not in results:
                print(f"No coefficients or feature names found for best biomarker model in {dataset} for {data_option_label}.")
                continue
                
            freq = count_high_weight_biomarkers(results['all_coefficients'], results['feature_names'], threshold=0.01)
            
            # Debug: Print coefficient statistics
            print(f"\n[DEBUG] Coefficient statistics for {data_option_label} {dataset} {best_model_type} Biomarker:")
            all_coefs = np.array(results['all_coefficients'])
            print(f"  Shape: {all_coefs.shape}")
            print(f"  Min: {all_coefs.min():.6f}")
            print(f"  Max: {all_coefs.max():.6f}")
            print(f"  Mean: {all_coefs.mean():.6f}")
            print(f"  Std: {all_coefs.std():.6f}")
            print(f"  Non-zero coefficients: {np.count_nonzero(all_coefs)}")
            print(f"  Features with any non-zero coef: {np.count_nonzero(np.any(all_coefs != 0, axis=0))}")
            
            plot_feature_frequency(
                results['feature_names'],
                freq,
                filename=f"outputs/plots/best_model_biomarker_frequency_{data_option_label}_{dataset}_{target_type}.png",
                model_name=best_model_type.capitalize(),
                dataset_name=f"{dataset.capitalize()} ({data_option_label}) - {target_type}"
            )
            print(f"Biomarker frequency plot for best model in {dataset} ({data_option_label}) saved to outputs/plots/best_model_biomarker_frequency_{data_option_label}_{dataset}_{target_type}.png")
    
    print(f"\n{'='*80}")
    print("PLOTTING COMPLETE!")
    print(f"{'='*80}")
    print("Generated plots:")
    print("- True vs predicted scatter plots")
    print("- Best model biomarker frequency plots")
    
    # Run organize_plots.py to move plots to subdirectories
    print("\nOrganizing plots into subdirectories...")
    os.system("python3 organize_plots.py")

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        target_type = sys.argv[1].lower()
        if target_type not in ['gestational_age', 'birth_weight']:
            print("Error: target_type must be 'gestational_age' or 'birth_weight'")
            print("Usage: python3 run_plotting_only.py [gestational_age|birth_weight]")
            sys.exit(1)
    else:
        target_type = 'gestational_age'  # Default
    
    # Run the plotting
    run_plotting_only(target_type) 