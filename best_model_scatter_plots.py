#!/usr/bin/env python3
"""
Best Model Scatter Plots Generator

This script identifies the best regression and classification models from ElasticNet, Lasso, and STABL
and generates predicted vs actual scatter plots for each data option and dataset combination.

Author: Diba Dindoust
Date: 07/01/2025
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from adjustText import adjust_text

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_results():
    """Load all results from pickle file."""
    try:
        with open('all_results.pkl', 'rb') as f:
            all_results = pickle.load(f)
        print(f"✅ Loaded {len(all_results)} model results")
        return all_results
    except FileNotFoundError:
        print("❌ all_results.pkl not found. Please run the main pipeline first.")
        return None

def parse_result_key(key):
    """Parse result key to extract components."""
    # Format: data_option_dataset_model_type_model_name_target_type
    parts = key.split('_')
    
    # Handle different key formats
    if 'both_samples' in key:
        data_option = 'both_samples'
        if 'cord' in key:
            dataset = 'cord'
        elif 'heel' in key:
            dataset = 'heel'
        else:
            dataset = 'unknown'
    elif 'cord_all' in key:
        data_option = 'cord_all'
        dataset = 'cord'
    elif 'heel_all' in key:
        data_option = 'heel_all'
        dataset = 'heel'
    else:
        data_option = 'unknown'
        dataset = 'unknown'
    
    # Extract model type and name
    if 'elasticnet_cv' in key:
        model_type = 'elasticnet_cv'
        model_name = 'ElasticNet'
    elif 'lasso_cv' in key:
        model_type = 'lasso_cv'
        model_name = 'Lasso'
    elif 'stabl' in key:
        model_type = 'stabl'
        model_name = 'STABL'
    else:
        model_type = 'unknown'
        model_name = 'unknown'
    
    # Extract dataset type (Biomarker, Clinical, Combined)
    if 'Biomarker' in key:
        dataset_type = 'Biomarker'
    elif 'Clinical' in key:
        dataset_type = 'Clinical'
    elif 'Combined' in key:
        dataset_type = 'Combined'
    else:
        dataset_type = 'unknown'
    
    # Extract target type
    if 'gestational_age' in key:
        target_type = 'gestational_age'
    elif 'birth_weight' in key:
        target_type = 'birth_weight'
    else:
        target_type = 'unknown'
    
    return {
        'data_option': data_option,
        'dataset': dataset,
        'model_type': model_type,
        'model_name': model_name,
        'dataset_type': dataset_type,
        'target_type': target_type
    }

def get_best_models(all_results):
    """Identify the best regression and classification models for each combination."""
    best_models = {
        'regression': {},
        'classification': {}
    }
    
    # Group results by data option, dataset, and dataset type
    grouped_results = {}
    
    for key, result in all_results.items():
        parsed = parse_result_key(key)
        
        # Create grouping key
        group_key = f"{parsed['data_option']}_{parsed['dataset']}_{parsed['dataset_type']}"
        
        if group_key not in grouped_results:
            grouped_results[group_key] = {
                'regression': [],
                'classification': []
            }
        
        # Check if this is a valid result
        if 'summary' not in result:
            continue
            
        summary = result['summary']
        
        # For regression models (gestational_age and birth_weight)
        if parsed['target_type'] in ['gestational_age', 'birth_weight']:
            if 'mae_mean' in summary and not np.isnan(summary['mae_mean']):
                grouped_results[group_key]['regression'].append({
                    'key': key,
                    'result': result,
                    'parsed': parsed,
                    'mae': summary['mae_mean'],
                    'rmse': summary['rmse_mean']
                })
        
        # For classification models (preterm classification)
        if parsed['target_type'] == 'gestational_age':
            if 'auc_mean' in summary and not np.isnan(summary['auc_mean']):
                grouped_results[group_key]['classification'].append({
                    'key': key,
                    'result': result,
                    'parsed': parsed,
                    'auc': summary['auc_mean']
                })
    
    # Find best models for each group
    for group_key, group_data in grouped_results.items():
        # Best regression model (lowest MAE)
        if group_data['regression']:
            best_reg = min(group_data['regression'], key=lambda x: x['mae'])
            best_models['regression'][group_key] = best_reg
        
        # Best classification model (highest AUC)
        if group_data['classification']:
            best_cls = max(group_data['classification'], key=lambda x: x['auc'])
            best_models['classification'][group_key] = best_cls
    
    return best_models

def extract_predictions(result, task_type='regression'):
    """Extract true and predicted values from result."""
    if task_type == 'regression':
        if 'predictions' in result and result['predictions']:
            # Extract from predictions list
            true_values = []
            pred_values = []
            
            for pred_dict in result['predictions']:
                if 'true' in pred_dict and 'pred' in pred_dict:
                    true_values.append(pred_dict['true'])
                    pred_values.append(pred_dict['pred'])
            
            if true_values and pred_values:
                return np.array(true_values), np.array(pred_values)
    
    elif task_type == 'classification':
        if 'classification_predictions' in result and result['classification_predictions']:
            # Extract from classification_predictions list
            true_values = []
            pred_values = []
            
            for pred_dict in result['classification_predictions']:
                if 'true' in pred_dict and 'pred' in pred_dict:
                    true_values.append(pred_dict['true'])
                    pred_values.append(pred_dict['pred'])
            
            if true_values and pred_values:
                return np.array(true_values), np.array(pred_values)
    
    return None, None

def create_scatter_plot(true_values, pred_values, title, filename, task_type='regression'):
    """Create a scatter plot of predicted vs actual values."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(true_values, pred_values, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add perfect prediction line
    min_val = min(true_values.min(), pred_values.min())
    max_val = max(true_values.max(), pred_values.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
    
    # Calculate metrics
    if task_type == 'regression':
        mae = np.mean(np.abs(true_values - pred_values))
        rmse = np.sqrt(np.mean((true_values - pred_values)**2))
        r2 = np.corrcoef(true_values, pred_values)[0, 1]**2
        
        # Add metrics to plot
        metrics_text = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=12)
        
        # Set labels
        ax.set_xlabel('Actual Values', fontsize=14)
        ax.set_ylabel('Predicted Values', fontsize=14)
        
    elif task_type == 'classification':
        # For classification, we might have probabilities
        if len(np.unique(true_values)) == 2:  # Binary classification
            from sklearn.metrics import roc_auc_score, accuracy_score
            try:
                auc = roc_auc_score(true_values, pred_values)
                accuracy = accuracy_score(true_values, pred_values.round())
                
                metrics_text = f'AUC: {auc:.3f}\nAccuracy: {accuracy:.3f}'
                ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        verticalalignment='top', fontsize=12)
            except:
                pass
        
        ax.set_xlabel('Actual Class', fontsize=14)
        ax.set_ylabel('Predicted Probability', fontsize=14)
    
    # Set title and grid
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: {filename}")

def generate_all_scatter_plots(all_results):
    """Generate scatter plots for all best models."""
    print("🔍 Identifying best models...")
    best_models = get_best_models(all_results)
    
    # Create output directory
    output_dir = Path("outputs/plots/best_model_scatter_plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Generating scatter plots...")
    
    # Generate regression plots
    print("\n📈 REGRESSION MODELS:")
    for group_key, best_reg in best_models['regression'].items():
        parsed = best_reg['parsed']
        result = best_reg['result']
        
        # Extract predictions
        true_values, pred_values = extract_predictions(result, 'regression')
        
        if true_values is not None and pred_values is not None:
            # Create title
            title = f"Best Regression Model: {best_reg['parsed']['model_name']}\n"
            title += f"Data: {parsed['data_option'].replace('_', ' ').title()} | "
            title += f"Dataset: {parsed['dataset_type']} | "
            title += f"Target: {parsed['target_type'].replace('_', ' ').title()}\n"
            title += f"MAE: {best_reg['mae']:.3f} | RMSE: {best_reg['rmse']:.3f}"
            
            # Create filename
            filename = output_dir / f"regression_{parsed['data_option']}_{parsed['dataset_type']}_{parsed['target_type']}.png"
            
            create_scatter_plot(true_values, pred_values, title, filename, 'regression')
    
    # Generate classification plots
    print("\n🎯 CLASSIFICATION MODELS:")
    for group_key, best_cls in best_models['classification'].items():
        parsed = best_cls['parsed']
        result = best_cls['result']
        
        # Extract predictions
        true_values, pred_values = extract_predictions(result, 'classification')
        
        if true_values is not None and pred_values is not None:
            # Create title
            title = f"Best Classification Model: {best_cls['parsed']['model_name']}\n"
            title += f"Data: {parsed['data_option'].replace('_', ' ').title()} | "
            title += f"Dataset: {parsed['dataset_type']} | "
            title += f"Target: Preterm Classification\n"
            title += f"AUC: {best_cls['auc']:.3f}"
            
            # Create filename
            filename = output_dir / f"classification_{parsed['data_option']}_{parsed['dataset_type']}.png"
            
            create_scatter_plot(true_values, pred_values, title, filename, 'classification')
    
    # Generate summary report
    generate_summary_report(best_models, output_dir)
    
    print(f"\n✅ All scatter plots saved to: {output_dir}")

def generate_summary_report(best_models, output_dir):
    """Generate a summary report of the best models."""
    print("\n📋 Generating summary report...")
    
    # Create summary dataframes
    reg_data = []
    cls_data = []
    
    for group_key, best_reg in best_models['regression'].items():
        parsed = best_reg['parsed']
        reg_data.append({
            'Data Option': parsed['data_option'].replace('_', ' ').title(),
            'Dataset Type': parsed['dataset_type'],
            'Target': parsed['target_type'].replace('_', ' ').title(),
            'Best Model': best_reg['parsed']['model_name'],
            'MAE': best_reg['mae'],
            'RMSE': best_reg['rmse']
        })
    
    for group_key, best_cls in best_models['classification'].items():
        parsed = best_cls['parsed']
        cls_data.append({
            'Data Option': parsed['data_option'].replace('_', ' ').title(),
            'Dataset Type': parsed['dataset_type'],
            'Best Model': best_cls['parsed']['model_name'],
            'AUC': best_cls['auc']
        })
    
    # Create DataFrames
    reg_df = pd.DataFrame(reg_data)
    cls_df = pd.DataFrame(cls_data)
    
    # Save to CSV
    reg_df.to_csv(output_dir / "best_regression_models.csv", index=False)
    cls_df.to_csv(output_dir / "best_classification_models.csv", index=False)
    
    # Print summary
    print("\n🏆 BEST REGRESSION MODELS:")
    print(reg_df.to_string(index=False))
    
    print("\n🏆 BEST CLASSIFICATION MODELS:")
    print(cls_df.to_string(index=False))
    
    print(f"\n📄 Summary reports saved to:")
    print(f"   {output_dir / 'best_regression_models.csv'}")
    print(f"   {output_dir / 'best_classification_models.csv'}")

def main():
    """Main function to run the scatter plot generation."""
    print("🎯 Best Model Scatter Plots Generator")
    print("=" * 50)
    
    # Load results
    all_results = load_results()
    if all_results is None:
        return
    
    # Generate scatter plots
    generate_all_scatter_plots(all_results)
    
    print("\n🎉 Scatter plot generation complete!")

if __name__ == "__main__":
    main()
