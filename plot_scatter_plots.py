#!/usr/bin/env python3
"""
Script to generate scatter plots for predicted vs actual values.
Creates separate plots for each combination of:
- Model: elasticnet_cv, lasso_cv
- Data option: both_samples, heel_all, cord_all  
- Dataset: biomarker, clinical, combined
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_results():
    """Load all results from pickle files."""
    results = {}
    
    # Try to load the main results file
    if os.path.exists('all_results.pkl'):
        with open('all_results.pkl', 'rb') as f:
            results = pickle.load(f)
        print(f"✅ Loaded {len(results)} results from all_results.pkl")
    else:
        print("❌ all_results.pkl not found")
        return {}
    
    return results

def create_scatter_plot(y_true, y_pred, title, filename, target_type):
    """
    Create a scatter plot of predicted vs actual values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        filename: Output filename
        target_type: 'gestational_age' or 'birth_weight'
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Add metrics to plot
    metrics_text = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and title
    if target_type == 'gestational_age':
        ax.set_xlabel('Actual Gestational Age (weeks)', fontsize=14)
        ax.set_ylabel('Predicted Gestational Age (weeks)', fontsize=14)
    else:  # birth_weight
        ax.set_xlabel('Actual Birth Weight (kg)', fontsize=14)
        ax.set_ylabel('Predicted Birth Weight (kg)', fontsize=14)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {filename}")
    return mae, rmse, r2

def generate_all_scatter_plots():
    """Generate scatter plots for all combinations."""
    
    print("🔍 Loading results...")
    results = load_results()
    
    if not results:
        print("❌ No results found. Please run the pipeline first.")
        return
    
    # Define combinations
    models = ['elasticnet_cv', 'lasso_cv']
    data_options = ['both_samples', 'heel_all', 'cord_all']
    datasets = ['biomarker', 'clinical', 'combined']
    target_types = ['gestational_age', 'birth_weight']
    
    # Create output directory
    output_dir = "outputs/plots/scatter_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📊 Generating scatter plots for {len(models)} models × {len(data_options)} data options × {len(datasets)} datasets × {len(target_types)} targets = {len(models) * len(data_options) * len(datasets) * len(target_types)} total plots")
    
    # Track metrics for summary
    all_metrics = []
    
    # Generate plots for each combination
    for target_type in target_types:
        print(f"\n🎯 Processing {target_type} predictions...")
        
        for data_option in data_options:
            print(f"   📁 Data option: {data_option}")
            
            for dataset in datasets:
                print(f"      🧬 Dataset: {dataset}")
                
                for model in models:
                    print(f"         🤖 Model: {model}")
                    
                    # Construct result key - format: data_option_dataset_model_DatasetType_target_type
                    # Map data options to dataset types
                    if data_option == 'both_samples':
                        # For both_samples, we need to check both heel and cord
                        result_key_heel = f"{data_option}_heel_{model}_{dataset.title()}_{target_type}"
                        result_key_cord = f"{data_option}_cord_{model}_{dataset.title()}_{target_type}"
                        
                        # Try heel first, then cord
                        if result_key_heel in results:
                            result_key = result_key_heel
                        elif result_key_cord in results:
                            result_key = result_key_cord
                        else:
                            result_key = None
                    else:
                        # For heel_all and cord_all, use the data option as dataset type
                        dataset_type = 'heel' if data_option == 'heel_all' else 'cord'
                        result_key = f"{data_option}_{dataset_type}_{model}_{dataset.title()}_{target_type}"
                    
                    if result_key and result_key in results:
                        result = results[result_key]
                        
                        # Check if predictions exist
                        if 'predictions' in result and len(result['predictions']) > 0:
                            # Get predictions from the first run (or average across runs)
                            pred_data = result['predictions'][0]
                            
                            if 'true' in pred_data and 'pred' in pred_data:
                                y_true = pred_data['true']
                                y_pred = pred_data['pred']
                                
                                # Create title
                                title = f"{model.upper()} - {dataset.title()} - {data_option.replace('_', ' ').title()}\n{target_type.replace('_', ' ').title()}"
                                
                                # Create filename
                                filename = f"{output_dir}/{target_type}_{data_option}_{dataset}_{model}_scatter.png"
                                
                                # Generate plot
                                mae, rmse, r2 = create_scatter_plot(y_true, y_pred, title, filename, target_type)
                                
                                # Store metrics
                                all_metrics.append({
                                    'target_type': target_type,
                                    'data_option': data_option,
                                    'dataset': dataset,
                                    'model': model,
                                    'mae': mae,
                                    'rmse': rmse,
                                    'r2': r2
                                })
                            else:
                                print(f"            ❌ Missing 'true' or 'pred' in predictions")
                        else:
                            print(f"            ❌ No predictions found")
                    else:
                        print(f"            ❌ Result key '{result_key}' not found")
    
    # Create summary table
    if all_metrics:
        print(f"\n📋 Creating summary table...")
        metrics_df = pd.DataFrame(all_metrics)
        
        # Save summary
        summary_file = f"{output_dir}/scatter_plots_summary.csv"
        metrics_df.to_csv(summary_file, index=False)
        print(f"   ✅ Saved summary: {summary_file}")
        
        # Print summary
        print(f"\n📊 Summary of {len(metrics_df)} scatter plots:")
        print(metrics_df.groupby(['target_type', 'model'])[['mae', 'rmse', 'r2']].mean().round(3))
    
    print(f"\n✅ All scatter plots generated in: {output_dir}")

def main():
    """Main function."""
    print("🎨 Generating Scatter Plots for Predicted vs Actual Values")
    print("=" * 60)
    
    generate_all_scatter_plots()
    
    print("\n🎉 Scatter plot generation completed!")

if __name__ == "__main__":
    main()
