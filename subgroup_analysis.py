#!/usr/bin/env python3
"""
Comprehensive Subgroup Analysis Script
=====================================

This script combines all subgroup analysis functionality:
1. Load existing results from all_results.pkl files
2. Calculate subgroup metrics (MAE/RMSE for preterm, term, SGA, normal)
3. Display detailed and summarized results
4. Save updated results with subgroup metrics

Usage:
    python3 subgroup_analysis.py [gestational_age|birth_weight]
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

def load_original_data_for_subgroups(task: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load original data to get gestational ages for subgroup classification.
    
    Args:
        task: 'gestational_age' or 'birth_weight'
        
    Returns:
        Tuple of (DataFrame, gestational_ages_array)
    """
    try:
        # Load the original data - try both possible file names
        data_files = [
            'data/Bangladeshcombineddataset_both_samples.csv',
            'data/BangladeshcombineddatasetJan252022.csv'
        ]
        
        df = None
        for file_path in data_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                print(f"Loaded data from: {file_path}")
                break
        
        if df is None:
            print("Error: No data file found!")
            return None, None
        
        # Get gestational ages for subgroup classification
        gestational_ages = df['gestational_age_weeks'].values
        
        return df, gestational_ages
    except Exception as e:
        print(f"Error loading original data: {e}")
        return None, None

def create_subgroup_targets(gestational_ages: np.ndarray, birth_weights: np.ndarray = None) -> Dict[str, np.ndarray]:
    """
    Create subgroup classification targets.
    
    Args:
        gestational_ages: Array of gestational ages
        birth_weights: Array of birth weights (optional)
        
    Returns:
        Dictionary with subgroup targets
    """
    subgroups = {}
    
    # Preterm classification (< 37 weeks)
    subgroups['preterm'] = (gestational_ages < 37).astype(int)
    
    # Term classification (≥ 37 weeks)
    subgroups['term'] = (gestational_ages >= 37).astype(int)
    
    if birth_weights is not None:
        # SGA classification (< 10th percentile)
        sga_threshold = np.percentile(birth_weights, 10)
        subgroups['sga'] = (birth_weights < sga_threshold).astype(int)
        
        # Normal classification (≥ 10th percentile)
        subgroups['normal'] = (birth_weights >= sga_threshold).astype(int)
    else:
        # For gestational age task, use birth weight from data
        try:
            # Try to load data file
            data_files = [
                'data/Bangladeshcombineddataset_both_samples.csv',
                'data/BangladeshcombineddatasetJan252022.csv'
            ]
            
            df = None
            for file_path in data_files:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    break
            
            if df is not None:
                birth_weights = df['birth_weight_kg'].values
                sga_threshold = np.percentile(birth_weights, 10)
                subgroups['sga'] = (birth_weights < sga_threshold).astype(int)
                subgroups['normal'] = (birth_weights >= sga_threshold).astype(int)
            else:
                # If birth weight not available, use gestational age as proxy
                subgroups['sga'] = (gestational_ages < 37).astype(int)  # Preterm as proxy for SGA
                subgroups['normal'] = (gestational_ages >= 37).astype(int)
        except:
            # If birth weight not available, use gestational age as proxy
            subgroups['sga'] = (gestational_ages < 37).astype(int)  # Preterm as proxy for SGA
            subgroups['normal'] = (gestational_ages >= 37).astype(int)
    
    return subgroups

def calculate_subgroup_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                             gestational_ages: np.ndarray, 
                             birth_weights: np.ndarray = None) -> Dict[str, Dict[str, float]]:
    """
    Calculate MAE and RMSE for different subgroups.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        gestational_ages: Gestational ages for subgroup classification
        birth_weights: Birth weights for SGA classification (optional)
        
    Returns:
        Dictionary with subgroup metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    # Create subgroup targets
    subgroups = create_subgroup_targets(gestational_ages, birth_weights)
    
    metrics = {}
    
    for subgroup_name, subgroup_target in subgroups.items():
        # Get indices for this subgroup
        subgroup_indices = np.where(subgroup_target == 1)[0]
        
        if len(subgroup_indices) > 0:
            # Ensure indices are within bounds of y_true/y_pred
            valid_indices = subgroup_indices[subgroup_indices < len(y_true)]
            
            if len(valid_indices) > 0:
                # Calculate metrics for this subgroup
                subgroup_y_true = y_true[valid_indices]
                subgroup_y_pred = y_pred[valid_indices]
                
                mae = mean_absolute_error(subgroup_y_true, subgroup_y_pred)
                rmse = np.sqrt(mean_squared_error(subgroup_y_true, subgroup_y_pred))
                
                metrics[subgroup_name] = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'sample_size': len(valid_indices)
                }
            else:
                metrics[subgroup_name] = {
                    'MAE': np.nan,
                    'RMSE': np.nan,
                    'sample_size': 0
                }
        else:
            metrics[subgroup_name] = {
                'MAE': np.nan,
                'RMSE': np.nan,
                'sample_size': 0
            }
    
    return metrics

def calculate_subgroup_metrics_for_existing_results(task: str) -> Dict[str, Any]:
    """
    Calculate subgroup metrics for existing results without rerunning the pipeline.
    
    Args:
        task: 'gestational_age' or 'birth_weight'
        
    Returns:
        Updated all_results dictionary with subgroup metrics
    """
    # Load existing results
    results_file = f'all_results_{task}.pkl'
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found!")
        return {}
    
    print(f"Loading existing results from {results_file}...")
    with open(results_file, 'rb') as f:
        all_results = pickle.load(f)
    
    # Load original data for subgroup classification
    print("Loading original data for subgroup classification...")
    df, gestational_ages = load_original_data_for_subgroups(task)
    
    if df is None or gestational_ages is None:
        print("Error: Could not load original data!")
        return all_results
    
    # Get birth weights if needed
    birth_weights = None
    if task == 'birth_weight':
        birth_weights = df['birth_weight_kg'].values
    
    print(f"Calculating subgroup metrics for {len(all_results)} models...")
    
    # Process each model's results
    for result_key, results in all_results.items():
        print(f"Processing {result_key}...")
        
        # Get true and predicted values
        if 'predictions' in results and len(results['predictions']) > 0:
            # Calculate subgroup metrics across all runs and average them
            all_subgroup_metrics = []
            
            for run_idx, pred_data in enumerate(results['predictions']):
                if 'true' in pred_data and 'pred' in pred_data:
                    y_true = pred_data['true']
                    y_pred = pred_data['pred']
                    
                    # Calculate subgroup metrics for this run
                    run_subgroup_metrics = calculate_subgroup_metrics(
                        y_true, y_pred, gestational_ages, birth_weights
                    )
                    all_subgroup_metrics.append(run_subgroup_metrics)
                    
                    # Show progress for large numbers of runs
                    if len(results['predictions']) > 50 and (run_idx + 1) % 20 == 0:
                        print(f"    - Processed {run_idx + 1}/{len(results['predictions'])} runs...")
            
            if all_subgroup_metrics:
                # Average the metrics across all runs
                averaged_subgroup_metrics = {}
                subgroup_names = all_subgroup_metrics[0].keys()
                
                for subgroup_name in subgroup_names:
                    mae_values = [run_metrics[subgroup_name]['MAE'] for run_metrics in all_subgroup_metrics 
                                 if not np.isnan(run_metrics[subgroup_name]['MAE'])]
                    rmse_values = [run_metrics[subgroup_name]['RMSE'] for run_metrics in all_subgroup_metrics 
                                  if not np.isnan(run_metrics[subgroup_name]['RMSE'])]
                    
                    if mae_values and rmse_values:
                        averaged_subgroup_metrics[subgroup_name] = {
                            'MAE': np.mean(mae_values),
                            'RMSE': np.mean(rmse_values),
                            'MAE_std': np.std(mae_values),
                            'RMSE_std': np.std(rmse_values),
                            'sample_size': all_subgroup_metrics[0][subgroup_name]['sample_size'],
                            'n_runs': len(mae_values)
                        }
                    else:
                        averaged_subgroup_metrics[subgroup_name] = {
                            'MAE': np.nan,
                            'RMSE': np.nan,
                            'MAE_std': np.nan,
                            'RMSE_std': np.nan,
                            'sample_size': 0,
                            'n_runs': 0
                        }
                
                # Add averaged subgroup metrics to results
                results['subgroup_metrics'] = averaged_subgroup_metrics
                
                print(f"  - Added averaged subgroup metrics for {len(averaged_subgroup_metrics)} subgroups across {len(all_subgroup_metrics)} runs")
            else:
                print(f"  - Warning: No valid predictions found for {result_key}")
        else:
            print(f"  - Warning: No predictions found for {result_key}")
    
    return all_results

def calculate_within_1_week_percentage(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate percentage of predictions within ±1 week of actual gestational age.
    
    Args:
        y_true: True gestational ages
        y_pred: Predicted gestational ages
        
    Returns:
        Percentage within ±1 week
    """
    within_1_week = np.abs(y_true - y_pred) <= 1.0
    return np.mean(within_1_week) * 100

def calculate_within_1_week_percentage_for_subgroups(y_true: np.ndarray, y_pred: np.ndarray, 
                                                   gestational_ages: np.ndarray, 
                                                   birth_weights: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate percentage within ±1 week for different subgroups.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        gestational_ages: Gestational ages for subgroup classification
        birth_weights: Birth weights for SGA classification (optional)
        
    Returns:
        Dictionary with subgroup percentages
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    # Create subgroup targets
    subgroups = create_subgroup_targets(gestational_ages, birth_weights)
    
    percentages = {}
    
    for subgroup_name, subgroup_target in subgroups.items():
        # Get indices for this subgroup
        subgroup_indices = np.where(subgroup_target == 1)[0]
        
        if len(subgroup_indices) > 0:
            # Ensure indices are within bounds of y_true/y_pred
            valid_indices = subgroup_indices[subgroup_indices < len(y_true)]
            
            if len(valid_indices) > 0:
                # Calculate percentage for this subgroup
                subgroup_y_true = y_true[valid_indices]
                subgroup_y_pred = y_pred[valid_indices]
                
                within_1_week = np.abs(subgroup_y_true - subgroup_y_pred) <= 1.0
                percentage = np.mean(within_1_week) * 100
                
                percentages[subgroup_name] = percentage
            else:
                percentages[subgroup_name] = np.nan
        else:
            percentages[subgroup_name] = np.nan
    
    return percentages

def print_comprehensive_results_table(all_results: Dict[str, Any], task: str):
    """
    Print comprehensive results table comparing prediction models and save to file.
    
    Args:
        all_results: Results dictionary with subgroup metrics
        task: Task name for display
    """
    print(f"\n{'='*120}")
    print(f"COMPREHENSIVE RESULTS TABLE - {task.upper()}")
    print(f"{'='*120}")
    
    # Create output directory if it doesn't exist
    os.makedirs('outputs/tables', exist_ok=True)
    
    # Collect all metrics for table
    table_data = []
    
    for result_key, results in all_results.items():
        if 'subgroup_metrics' not in results:
            continue
            
        # Parse result key to get model info
        # Example key: "both_samples_heel_lasso_cv_Clinical"
        parts = result_key.split('_')
        
        # Extract data option (first two parts for "both_samples", first part for others)
        if len(parts) >= 2 and parts[0] == "both" and parts[1] == "samples":
            data_option = "both_samples"
        elif len(parts) >= 2 and parts[0] == "heel" and parts[1] == "all":
            data_option = "heel_all"
        elif len(parts) >= 2 and parts[0] == "cord" and parts[1] == "all":
            data_option = "cord_all"
        else:
            data_option = parts[0] if len(parts) > 0 else "Unknown"
        
        # Extract model type (look for lasso_cv or elasticnet_cv)
        model = "Unknown"
        for i, part in enumerate(parts):
            if part in ['lasso', 'elasticnet'] and i + 1 < len(parts) and parts[i + 1] == 'cv':
                model = f"{part}_cv"
                break
        
        # Extract dataset (last part)
        dataset = parts[-1] if len(parts) > 0 else "Unknown"
        
        # Get overall metrics
        overall_mae = results.get('mae', np.nan)
        overall_rmse = results.get('rmse', np.nan)
        
        # If not found in results, try to get from predictions
        if np.isnan(overall_mae) and 'predictions' in results and len(results['predictions']) > 0:
            pred_data = results['predictions'][0]
            overall_mae = pred_data.get('mae', np.nan)
            overall_rmse = pred_data.get('rmse', np.nan)
        
        # Calculate overall within ±1 week percentage
        overall_within_1_week = np.nan
        if 'predictions' in results and len(results['predictions']) > 0:
            pred_data = results['predictions'][0]
            if 'true' in pred_data and 'pred' in pred_data:
                overall_within_1_week = calculate_within_1_week_percentage(
                    pred_data['true'], pred_data['pred']
                )
        
        # Get subgroup metrics
        preterm_mae = results['subgroup_metrics'].get('preterm', {}).get('MAE', np.nan)
        preterm_rmse = results['subgroup_metrics'].get('preterm', {}).get('RMSE', np.nan)
        sga_mae = results['subgroup_metrics'].get('sga', {}).get('MAE', np.nan)
        sga_rmse = results['subgroup_metrics'].get('sga', {}).get('RMSE', np.nan)
        
        # Calculate subgroup within ±1 week percentages
        preterm_within_1_week = np.nan
        sga_within_1_week = np.nan
        if 'predictions' in results and len(results['predictions']) > 0:
            pred_data = results['predictions'][0]
            if 'true' in pred_data and 'pred' in pred_data:
                # Load data for subgroup classification
                df, gestational_ages = load_original_data_for_subgroups(task)
                if df is not None:
                    birth_weights = None
                    if task == 'birth_weight':
                        birth_weights = df['birth_weight_kg'].values
                    
                    within_1_week_percentages = calculate_within_1_week_percentage_for_subgroups(
                        pred_data['true'], pred_data['pred'], gestational_ages, birth_weights
                    )
                    preterm_within_1_week = within_1_week_percentages.get('preterm', np.nan)
                    sga_within_1_week = within_1_week_percentages.get('sga', np.nan)
        
        table_data.append({
            'Data_Option': data_option,
            'Model': model,
            'Dataset': dataset,
            'Overall_MAE': overall_mae,
            'Overall_RMSE': overall_rmse,
            'Overall_Within_1_Week': overall_within_1_week,
            'Preterm_MAE': preterm_mae,
            'Preterm_RMSE': preterm_rmse,
            'Preterm_Within_1_Week': preterm_within_1_week,
            'SGA_MAE': sga_mae,
            'SGA_RMSE': sga_rmse,
            'SGA_Within_1_Week': sga_within_1_week
        })
    
    if not table_data:
        print("No data available for table generation.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Filter for specific data options and datasets
    data_options = ['both_samples', 'heel_all', 'cord_all']
    datasets = ['Clinical', 'Biomarker', 'Combined']
    
    # Prepare table content for file
    table_content = []
    table_content.append(f"COMPREHENSIVE RESULTS TABLE - {task.upper()}")
    table_content.append("=" * 120)
    table_content.append("")
    
    # Generate table content for each data option
    for data_option in data_options:
        table_content.append("=" * 120)
        table_content.append(f"DATA OPTION: {data_option.upper()}")
        table_content.append("=" * 120)
        table_content.append("")
        
        for dataset in datasets:
            table_content.append(f"{dataset.upper()} DATASET:")
            table_content.append("-" * 80)
            
            # Filter data
            subset = df[(df['Data_Option'] == data_option) & (df['Dataset'] == dataset)]
            
            if subset.empty:
                table_content.append("No data available for this combination.")
                table_content.append("")
                continue
            
            # Add table header
            table_content.append(f"{'Model':<15} {'Metric':<15} {'Overall':<20} {'Preterm':<20} {'SGA':<20}")
            table_content.append("-" * 90)
            
            # Add MAE rows
            for _, row in subset.iterrows():
                model = row['Model']
                mae_overall = f"{row['Overall_MAE']:.3f}" if not np.isnan(row['Overall_MAE']) else "N/A"
                mae_preterm = f"{row['Preterm_MAE']:.3f}" if not np.isnan(row['Preterm_MAE']) else "N/A"
                mae_sga = f"{row['SGA_MAE']:.3f}" if not np.isnan(row['SGA_MAE']) else "N/A"
                table_content.append(f"{model:<15} {'MAE':<15} {mae_overall:<20} {mae_preterm:<20} {mae_sga:<20}")
            
            # Add RMSE rows
            for _, row in subset.iterrows():
                model = row['Model']
                rmse_overall = f"{row['Overall_RMSE']:.3f}" if not np.isnan(row['Overall_RMSE']) else "N/A"
                rmse_preterm = f"{row['Preterm_RMSE']:.3f}" if not np.isnan(row['Preterm_RMSE']) else "N/A"
                rmse_sga = f"{row['SGA_RMSE']:.3f}" if not np.isnan(row['SGA_RMSE']) else "N/A"
                table_content.append(f"{model:<15} {'RMSE':<15} {rmse_overall:<20} {rmse_preterm:<20} {rmse_sga:<20}")
            
            # Add Within ±1 Week rows
            for _, row in subset.iterrows():
                model = row['Model']
                within_overall = f"{row['Overall_Within_1_Week']:.1f}%" if not np.isnan(row['Overall_Within_1_Week']) else "N/A"
                within_preterm = f"{row['Preterm_Within_1_Week']:.1f}%" if not np.isnan(row['Preterm_Within_1_Week']) else "N/A"
                within_sga = f"{row['SGA_Within_1_Week']:.1f}%" if not np.isnan(row['SGA_Within_1_Week']) else "N/A"
                table_content.append(f"{model:<15} {'Within ±1':<15} {within_overall:<20} {within_preterm:<20} {within_sga:<20}")
            
            table_content.append("")
    
    # Print to console
    for line in table_content:
        print(line)
    
    # Save to file
    filename = f"outputs/tables/comprehensive_results_table_{task}.txt"
    with open(filename, 'w') as f:
        for line in table_content:
            f.write(line + '\n')
    
    print(f"\n📄 Comprehensive results table saved to: {filename}")
    
    # Also save as CSV for easier analysis
    csv_filename = f"outputs/tables/comprehensive_results_table_{task}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"📊 Results data also saved as CSV: {csv_filename}")

def print_subgroup_summary(all_results: Dict[str, Any], task: str):
    """
    Print detailed subgroup metrics summary.
    
    Args:
        all_results: Results dictionary with subgroup metrics
        task: Task name for display
    """
    print(f"\n{'='*80}")
    print(f"SUBGROUP METRICS SUMMARY FOR {task.upper()}")
    print(f"{'='*80}")
    
    print("\nDetailed Subgroup Metrics:")
    print("-" * 80)
    
    # Collect all metrics for summary
    all_metrics = []
    
    for result_key, results in all_results.items():
        if 'subgroup_metrics' not in results:
            continue
            
        # Parse result key to get model info
        # Example key: "both_samples_heel_lasso_cv_Clinical"
        parts = result_key.split('_')
        
        # Extract data option (first two parts for "both_samples", first part for others)
        if len(parts) >= 2 and parts[0] == "both" and parts[1] == "samples":
            data_option = "both_samples"
        elif len(parts) >= 2 and parts[0] == "heel" and parts[1] == "all":
            data_option = "heel_all"
        elif len(parts) >= 2 and parts[0] == "cord" and parts[1] == "all":
            data_option = "cord_all"
        else:
            data_option = parts[0] if len(parts) > 0 else "Unknown"
        
        # Extract model type (look for lasso_cv or elasticnet_cv)
        model = "Unknown"
        for i, part in enumerate(parts):
            if part in ['lasso', 'elasticnet'] and i + 1 < len(parts) and parts[i + 1] == 'cv':
                model = f"{part}_cv"
                break
        
        # Extract dataset (last part)
        dataset = parts[-1] if len(parts) > 0 else "Unknown"
        
        # Get overall metrics
        overall_mae = results.get('mae', np.nan)
        overall_rmse = results.get('rmse', np.nan)
        
        # If not found in results, try to get from predictions
        if np.isnan(overall_mae) and 'predictions' in results and len(results['predictions']) > 0:
            pred_data = results['predictions'][0]
            overall_mae = pred_data.get('mae', np.nan)
            overall_rmse = pred_data.get('rmse', np.nan)
        
        # Print subgroup metrics
        for subgroup, metrics in results['subgroup_metrics'].items():
            mae = metrics.get('MAE', np.nan)
            rmse = metrics.get('RMSE', np.nan)
            mae_std = metrics.get('MAE_std', 0)
            rmse_std = metrics.get('RMSE_std', 0)
            sample_size = metrics.get('sample_size', 0)
            n_runs = metrics.get('n_runs', 0)
            
            if not np.isnan(mae) and sample_size > 0:
                # Calculate 95% confidence intervals
                mae_ci_lower = mae - 1.96 * mae_std / np.sqrt(n_runs) if n_runs > 0 else mae - 0.1
                mae_ci_upper = mae + 1.96 * mae_std / np.sqrt(n_runs) if n_runs > 0 else mae + 0.1
                rmse_ci_lower = rmse - 1.96 * rmse_std / np.sqrt(n_runs) if n_runs > 0 else rmse - 0.1
                rmse_ci_upper = rmse + 1.96 * rmse_std / np.sqrt(n_runs) if n_runs > 0 else rmse + 0.1
                
                print(f"{data_option:>12} {model:>12} {dataset:>9} {subgroup:>8} "
                      f"{mae:.3f} [{mae_ci_lower:.3f}, {mae_ci_upper:.3f}] "
                      f"{rmse:.3f} [{rmse_ci_lower:.3f}, {rmse_ci_upper:.3f}] "
                      f"{overall_mae:>8.3f} {overall_rmse:>8.3f} (n={n_runs})")
                
                all_metrics.append({
                    'Data_Option': data_option,
                    'Model': model,
                    'Dataset': dataset,
                    'Subgroup': subgroup,
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAE_std': mae_std,
                    'RMSE_std': rmse_std,
                    'MAE_ci_lower': mae_ci_lower,
                    'MAE_ci_upper': mae_ci_upper,
                    'RMSE_ci_lower': rmse_ci_lower,
                    'RMSE_ci_upper': rmse_ci_upper,
                    'Overall_MAE': overall_mae,
                    'Overall_RMSE': overall_rmse,
                    'n_runs': n_runs
                })
    
    # Create summary by subgroup
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        
        print(f"\n{'='*80}")
        print("SUMMARY BY SUBGROUP")
        print(f"{'='*80}")
        
        for subgroup in df['Subgroup'].unique():
            subgroup_data = df[df['Subgroup'] == subgroup].copy()
            
            print(f"\n{subgroup.upper()} BABIES:")
            print("-" * 40)
            
            # Find best performing models
            best_mae_idx = subgroup_data['MAE'].idxmin()
            best_rmse_idx = subgroup_data['RMSE'].idxmin()
            
            best_mae_row = subgroup_data.loc[best_mae_idx]
            best_rmse_row = subgroup_data.loc[best_rmse_idx]
            
            print(f"Best MAE: {best_mae_row['Model']} on {best_mae_row['Dataset']} "
                  f"({best_mae_row['Data_Option']}) - {best_mae_row['MAE']:.3f}")
            print(f"Best RMSE: {best_rmse_row['Model']} on {best_rmse_row['Dataset']} "
                  f"({best_rmse_row['Data_Option']}) - {best_rmse_row['RMSE']:.3f}")
            
            avg_mae = subgroup_data['MAE'].mean()
            avg_rmse = subgroup_data['RMSE'].mean()
            print(f"Average MAE: {avg_mae:.3f}")
            print(f"Average RMSE: {avg_rmse:.3f}")

def main():
    """Main function to run subgroup analysis."""
    if len(sys.argv) != 2:
        print("Usage: python3 subgroup_analysis.py [gestational_age|birth_weight]")
        sys.exit(1)
    
    task = sys.argv[1].lower()
    
    if task not in ['gestational_age', 'birth_weight']:
        print("Error: Task must be 'gestational_age' or 'birth_weight'")
        sys.exit(1)
    
    print(f"🔍 Subgroup Analysis for {task.replace('_', ' ').title()}")
    print("=" * 60)
    
    # Calculate subgroup metrics
    all_results = calculate_subgroup_metrics_for_existing_results(task)
    
    if not all_results:
        print("❌ No results found or error occurred!")
        sys.exit(1)
    
    # Print comprehensive results table
    print_comprehensive_results_table(all_results, task)
    
    # Print detailed summary
    print_subgroup_summary(all_results, task)
    
    # Save updated results
    output_file = f'all_results_{task}_with_subgroups.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n✅ Subgroup analysis complete!")
    print(f"📁 Results saved to: {output_file}")
    print(f"📊 Analyzed {len(all_results)} models")
    
    # Count total subgroup metrics
    total_subgroups = 0
    for results in all_results.values():
        if 'subgroup_metrics' in results:
            total_subgroups += len(results['subgroup_metrics'])
    
    print(f"🎯 Calculated metrics for {total_subgroups} model-subgroup combinations")

if __name__ == "__main__":
    main()
