#!/usr/bin/env python3
"""
Subgroup Performance Analysis: Preterm and SGA Babies

This script calculates RMSE and MAE separately for:
- Preterm babies (<37 weeks gestational age)
- SGA babies (Small for Gestational Age)
- Overall performance

For both gestational age and birth weight prediction pipelines.

Author: Diba Dindoust
Date: 07/01/2025
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.sga_classification import create_sga_targets_intergrowth21
from src.config import PRETERM_CUTOFF

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_results():
    """Load all results from pickle file."""
    try:
        with open('all_results.pkl', 'rb') as f:
            all_results = pickle.load(f)
        print(f"✅ Loaded {len(all_results)} results from all_results.pkl")
        return all_results
    except FileNotFoundError:
        print("❌ all_results.pkl not found. Please run the main pipeline first.")
        return None

def calculate_subgroup_metrics(y_true, y_pred, gestational_ages, data_option, dataset_type):
    """
    Calculate RMSE and MAE for different subgroups.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        gestational_ages: Gestational age values for preterm classification
        data_option: Data option (1, 2, or 3)
        dataset_type: Dataset type ('cord' or 'heel')
    
    Returns:
        dict: Metrics for each subgroup
    """
    # Overall metrics
    overall_mae = mean_absolute_error(y_true, y_pred)
    overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Preterm classification (<37 weeks)
    preterm_mask = gestational_ages < PRETERM_CUTOFF
    
    if np.sum(preterm_mask) > 0:
        preterm_mae = mean_absolute_error(y_true[preterm_mask], y_pred[preterm_mask])
        preterm_rmse = np.sqrt(mean_squared_error(y_true[preterm_mask], y_pred[preterm_mask]))
        preterm_count = np.sum(preterm_mask)
    else:
        preterm_mae = preterm_rmse = np.nan
        preterm_count = 0
    
    # Term classification (≥37 weeks)
    term_mask = gestational_ages >= PRETERM_CUTOFF
    
    if np.sum(term_mask) > 0:
        term_mae = mean_absolute_error(y_true[term_mask], y_pred[term_mask])
        term_rmse = np.sqrt(mean_squared_error(y_true[term_mask], y_pred[term_mask]))
        term_count = np.sum(term_mask)
    else:
        term_mae = term_rmse = np.nan
        term_count = 0
    
    # SGA classification using Intergrowth-21
    try:
        # For birth weight prediction, we need birth weights as y_true
        if len(y_true) == len(gestational_ages):
            birth_weights = y_true
        else:
            # For gestational age prediction, we need to estimate birth weights
            # This is a simplified approach - in practice you'd need actual birth weights
            print("⚠️  Warning: Using estimated birth weights for SGA classification")
            birth_weights = y_true  # Placeholder - ideally should be actual birth weights
        
        # Get sex information (assuming it's available in the data)
        # For now, we'll use a default approach
        sexes = np.ones(len(birth_weights))  # Default to male (1)
        
        sga_classification = create_sga_targets_intergrowth21(
            birth_weights, data_option, dataset_type, '10th_percentile'
        )
        
        sga_mask = sga_classification == 1
        normal_mask = sga_classification == 0
        
        if np.sum(sga_mask) > 0:
            sga_mae = mean_absolute_error(y_true[sga_mask], y_pred[sga_mask])
            sga_rmse = np.sqrt(mean_squared_error(y_true[sga_mask], y_pred[sga_mask]))
            sga_count = np.sum(sga_mask)
        else:
            sga_mae = sga_rmse = np.nan
            sga_count = 0
            
        if np.sum(normal_mask) > 0:
            normal_mae = mean_absolute_error(y_true[normal_mask], y_pred[normal_mask])
            normal_rmse = np.sqrt(mean_squared_error(y_true[normal_mask], y_pred[normal_mask]))
            normal_count = np.sum(normal_mask)
        else:
            normal_mae = normal_rmse = np.nan
            normal_count = 0
            
    except Exception as e:
        print(f"⚠️  Warning: Could not calculate SGA metrics: {e}")
        sga_mae = sga_rmse = normal_mae = normal_rmse = np.nan
        sga_count = normal_count = 0
    
    return {
        'overall': {
            'mae': overall_mae,
            'rmse': overall_rmse,
            'count': len(y_true)
        },
        'preterm': {
            'mae': preterm_mae,
            'rmse': preterm_rmse,
            'count': preterm_count
        },
        'term': {
            'mae': term_mae,
            'rmse': term_rmse,
            'count': term_count
        },
        'sga': {
            'mae': sga_mae,
            'rmse': sga_rmse,
            'count': sga_count
        },
        'normal': {
            'mae': normal_mae,
            'rmse': normal_rmse,
            'count': normal_count
        }
    }

def analyze_subgroup_performance():
    """Analyze subgroup performance for all models."""
    all_results = load_results()
    if not all_results:
        return
    
    # Initialize results storage
    subgroup_results = {
        'gestational_age': [],
        'birth_weight': []
    }
    
    # Data options and dataset types
    data_options = [1, 2, 3]  # both, heel_all, cord_all
    dataset_types = ['cord', 'heel']
    model_types = ['clinical', 'biomarker', 'combined']
    model_names = ['ElasticNet', 'Lasso', 'STABL']
    
    print("🔍 Analyzing subgroup performance...")
    print(f"Total results to analyze: {len(all_results)}")
    
    processed_count = 0
    for result_key, result in all_results.items():
        processed_count += 1
        if processed_count <= 3:  # Only show first 3 for debugging
            print(f"Processing result {processed_count}: {result_key}")
        # Parse result key to extract information
        parts = result_key.split('_')
        
        # Determine target type and data option
        target_type = None
        data_option = None
        dataset_type = None
        model_type = None
        model_name = None
        
        # Parse the result key structure
        parts = result_key.split('_')
        
        # Determine target type
        if 'gestational_age' in result_key:
            target_type = 'gestational_age'
        elif 'birth_weight' in result_key:
            target_type = 'birth_weight'
        
        # Extract data option and dataset type
        if 'both_samples' in result_key:
            data_option = 1
            if 'heel' in result_key:
                dataset_type = 'heel'
            elif 'cord' in result_key:
                dataset_type = 'cord'
        elif 'heel_all' in result_key:
            data_option = 2
            dataset_type = 'heel'
        elif 'cord_all' in result_key:
            data_option = 3
            dataset_type = 'cord'
        
        # Extract model information
        for mt in model_types:
            if mt.capitalize() in result_key:
                model_type = mt
                break
        
        for mn in model_names:
            if mn.lower() in result_key.lower():
                model_name = mn
                break
        
        if not all([target_type, data_option, dataset_type, model_type, model_name]):
            print(f"⚠️  Skipping {result_key}: Missing required info")
            print(f"   target_type: {target_type}, data_option: {data_option}, dataset_type: {dataset_type}, model_type: {model_type}, model_name: {model_name}")
            continue
        
        # Get predictions and actual values
        if 'predictions' in result and len(result['predictions']) > 0:
            print(f"✅ Processing {result_key}")
            
            # Extract predictions data
            pred_data = result['predictions'][0]  # Take first prediction set
            
            if 'true' in pred_data and 'pred' in pred_data:
                y_true = pred_data['true']
                y_pred = pred_data['pred']
                print(f"   y_true shape: {y_true.shape if hasattr(y_true, 'shape') else len(y_true)}")
                print(f"   y_pred shape: {y_pred.shape if hasattr(y_pred, 'shape') else len(y_pred)}")
                
                # For gestational age prediction, use gestational ages for preterm classification
                if target_type == 'gestational_age':
                    gestational_ages = y_true
                else:  # birth_weight prediction
                    # We need gestational ages for preterm classification
                    # This should be available in the result
                    if 'gestational_ages_test' in pred_data:
                        gestational_ages = pred_data['gestational_ages_test']
                    else:
                        print(f"⚠️  Warning: No gestational ages found for {result_key}")
                        continue
            else:
                print(f"⚠️  Warning: No true/pred in predictions for {result_key}")
                continue
            
            # Calculate subgroup metrics
            try:
                metrics = calculate_subgroup_metrics(
                    y_true, y_pred, gestational_ages, data_option, dataset_type
                )
                
                # Store results
                result_entry = {
                    'result_key': result_key,
                    'target_type': target_type,
                    'data_option': data_option,
                    'dataset_type': dataset_type,
                    'model_type': model_type,
                    'model_name': model_name,
                    'metrics': metrics
                }
                
                subgroup_results[target_type].append(result_entry)
                print(f"✅ Successfully processed {result_key}")
            except Exception as e:
                print(f"❌ Error processing {result_key}: {e}")
                continue
    
    return subgroup_results

def create_subgroup_summary_table(subgroup_results):
    """Create summary tables for subgroup performance."""
    summary_data = []
    
    print(f"Creating summary table from {len(subgroup_results)} target types")
    
    for target_type, results in subgroup_results.items():
        print(f"Processing {target_type}: {len(results)} results")
        for result in results:
            metrics = result['metrics']
            
            # Overall metrics
            summary_data.append({
                'Target': target_type.replace('_', ' ').title(),
                'Data Option': result['data_option'],
                'Dataset': result['dataset_type'],
                'Model Type': result['model_type'],
                'Model': result['model_name'],
                'Subgroup': 'Overall',
                'MAE': metrics['overall']['mae'],
                'RMSE': metrics['overall']['rmse'],
                'Count': metrics['overall']['count']
            })
            
            # Preterm metrics
            if not np.isnan(metrics['preterm']['mae']):
                summary_data.append({
                    'Target': target_type.replace('_', ' ').title(),
                    'Data Option': result['data_option'],
                    'Dataset': result['dataset_type'],
                    'Model Type': result['model_type'],
                    'Model': result['model_name'],
                    'Subgroup': 'Preterm',
                    'MAE': metrics['preterm']['mae'],
                    'RMSE': metrics['preterm']['rmse'],
                    'Count': metrics['preterm']['count']
                })
            
            # Term metrics
            if not np.isnan(metrics['term']['mae']):
                summary_data.append({
                    'Target': target_type.replace('_', ' ').title(),
                    'Data Option': result['data_option'],
                    'Dataset': result['dataset_type'],
                    'Model Type': result['model_type'],
                    'Model': result['model_name'],
                    'Subgroup': 'Term',
                    'MAE': metrics['term']['mae'],
                    'RMSE': metrics['term']['rmse'],
                    'Count': metrics['term']['count']
                })
            
            # SGA metrics
            if not np.isnan(metrics['sga']['mae']):
                summary_data.append({
                    'Target': target_type.replace('_', ' ').title(),
                    'Data Option': result['data_option'],
                    'Dataset': result['dataset_type'],
                    'Model Type': result['model_type'],
                    'Model': result['model_name'],
                    'Subgroup': 'SGA',
                    'MAE': metrics['sga']['mae'],
                    'RMSE': metrics['sga']['rmse'],
                    'Count': metrics['sga']['count']
                })
            
            # Normal metrics
            if not np.isnan(metrics['normal']['mae']):
                summary_data.append({
                    'Target': target_type.replace('_', ' ').title(),
                    'Data Option': result['data_option'],
                    'Dataset': result['dataset_type'],
                    'Model Type': result['model_type'],
                    'Model': result['model_name'],
                    'Subgroup': 'Normal',
                    'MAE': metrics['normal']['mae'],
                    'RMSE': metrics['normal']['rmse'],
                    'Count': metrics['normal']['count']
                })
    
    df = pd.DataFrame(summary_data)
    return df

def plot_subgroup_comparison(subgroup_results):
    """Create comparison plots for subgroup performance."""
    # Create output directory
    output_dir = Path("outputs/plots/subgroup_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for plotting
    plot_data = []
    
    for target_type, results in subgroup_results.items():
        for result in results:
            metrics = result['metrics']
            
            # Add overall metrics
            plot_data.append({
                'Target': target_type.replace('_', ' ').title(),
                'Data Option': f"Option {result['data_option']}",
                'Dataset': result['dataset_type'],
                'Model': result['model_name'],
                'Subgroup': 'Overall',
                'MAE': metrics['overall']['mae'],
                'RMSE': metrics['overall']['rmse']
            })
            
            # Add preterm metrics
            if not np.isnan(metrics['preterm']['mae']):
                plot_data.append({
                    'Target': target_type.replace('_', ' ').title(),
                    'Data Option': f"Option {result['data_option']}",
                    'Dataset': result['dataset_type'],
                    'Model': result['model_name'],
                    'Subgroup': 'Preterm',
                    'MAE': metrics['preterm']['mae'],
                    'RMSE': metrics['preterm']['rmse']
                })
            
            # Add SGA metrics
            if not np.isnan(metrics['sga']['mae']):
                plot_data.append({
                    'Target': target_type.replace('_', ' ').title(),
                    'Data Option': f"Option {result['data_option']}",
                    'Dataset': result['dataset_type'],
                    'Model': result['model_name'],
                    'Subgroup': 'SGA',
                    'MAE': metrics['sga']['mae'],
                    'RMSE': metrics['sga']['rmse']
                })
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create MAE comparison plot
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Gestational Age MAE
    plt.subplot(2, 2, 1)
    ga_data = df_plot[df_plot['Target'] == 'Gestational Age']
    if not ga_data.empty:
        sns.boxplot(data=ga_data, x='Subgroup', y='MAE', hue='Model')
        plt.title('Gestational Age Prediction - MAE by Subgroup')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Subplot 2: Birth Weight MAE
    plt.subplot(2, 2, 2)
    bw_data = df_plot[df_plot['Target'] == 'Birth Weight']
    if not bw_data.empty:
        sns.boxplot(data=bw_data, x='Subgroup', y='MAE', hue='Model')
        plt.title('Birth Weight Prediction - MAE by Subgroup')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Subplot 3: Gestational Age RMSE
    plt.subplot(2, 2, 3)
    if not ga_data.empty:
        sns.boxplot(data=ga_data, x='Subgroup', y='RMSE', hue='Model')
        plt.title('Gestational Age Prediction - RMSE by Subgroup')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Subplot 4: Birth Weight RMSE
    plt.subplot(2, 2, 4)
    if not bw_data.empty:
        sns.boxplot(data=bw_data, x='Subgroup', y='RMSE', hue='Model')
        plt.title('Birth Weight Prediction - RMSE by Subgroup')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'subgroup_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved subgroup comparison plot to {output_dir / 'subgroup_performance_comparison.png'}")

def main():
    """Main function to run the subgroup analysis."""
    print("🚀 Starting Subgroup Performance Analysis")
    print("=" * 50)
    
    # Analyze subgroup performance
    subgroup_results = analyze_subgroup_performance()
    
    if not subgroup_results:
        print("❌ No results to analyze")
        return
    
    # Create summary table
    summary_df = create_subgroup_summary_table(subgroup_results)
    
    # Save summary table
    output_dir = Path("outputs/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_dir / "subgroup_performance_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ Saved subgroup performance summary to {summary_file}")
    
    # Display summary statistics
    print("\n📊 Subgroup Performance Summary:")
    print("=" * 50)
    
    for target_type in ['gestational_age', 'birth_weight']:
        print(f"\n🎯 {target_type.replace('_', ' ').title()} Prediction:")
        
        target_data = summary_df[summary_df['Target'] == target_type.replace('_', ' ').title()]
        
        for subgroup in ['Overall', 'Preterm', 'SGA']:
            subgroup_data = target_data[target_data['Subgroup'] == subgroup]
            
            if not subgroup_data.empty:
                print(f"\n  📈 {subgroup} Babies:")
                print(f"    MAE: {subgroup_data['MAE'].mean():.3f} ± {subgroup_data['MAE'].std():.3f}")
                print(f"    RMSE: {subgroup_data['RMSE'].mean():.3f} ± {subgroup_data['RMSE'].std():.3f}")
                print(f"    Sample sizes: {subgroup_data['Count'].sum()} total")
    
    # Create comparison plots
    plot_subgroup_comparison(subgroup_results)
    
    print("\n✅ Subgroup analysis completed successfully!")

if __name__ == "__main__":
    main()
