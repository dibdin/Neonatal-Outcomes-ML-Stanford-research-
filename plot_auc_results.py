#!/usr/bin/env python3
"""
ML Experiment Results Visualization Script
=========================================

This script creates publication-ready grouped bar plots with error bars
for ML experiment results across different pipelines, models, data options, and datasets.

Usage:
    python3 plot_auc_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_actual_data():
    """
    Load actual data from all_results.pkl file.
    
    Returns:
        pd.DataFrame: Actual data with the required columns
    """
    import pickle
    
    # Load the results
    with open('all_results.pkl', 'rb') as f:
        all_results = pickle.load(f)
    
    # Parse the data
    data = []
    
    for key, results in all_results.items():
        # Parse the key: format examples:
        # "both_samples_heel_lasso_cv_Clinical_gestational_age"
        # "heel_all_heel_lasso_cv_Clinical_gestational_age"
        parts = key.split('_')
        
                # Extract components based on the format
        if len(parts) >= 8:
            # Handle "both_samples" case
            if parts[0] == "both" and parts[1] == "samples":
                data_option = f"both_{parts[2]}"  # both_heel or both_cord
                dataset_type = parts[2]  # heel or cord
                model = parts[3]  # lasso or elasticnet  
                # Skip 'cv' part
                dataset = parts[5]  # Clinical, Biomarker, Combined
                pipeline = f"{parts[6]}_{parts[7]}"  # gestational_age or birth_weight
            # Handle "heel_all" or "cord_all" case  
            elif parts[1] == "all":
                data_option = f"{parts[0]}_all"  # heel_all or cord_all
                dataset_type = parts[2]  # heel or cord
                model = parts[3]  # lasso or elasticnet
                # Skip 'cv' part
                dataset = parts[5]  # Clinical, Biomarker, Combined
                pipeline = f"{parts[6]}_{parts[7]}"  # gestational_age or birth_weight
            else:
                continue  # Skip malformed keys
        else:
            continue  # Skip malformed keys
        
        # Clean up names
        dataset = dataset.lower()
        if dataset == 'biomarker':
            dataset = 'biomarkers'
        
        # Get AUC values for each run
        aucs = results.get('aucs', [])
        
        # Create rows for each run
        for run_idx, auc in enumerate(aucs):
            data.append({
                'pipeline': pipeline,
                'model': model,
                'data_option': data_option,
                'dataset': dataset,
                'run': run_idx + 1,
                'auc': float(auc)
            })
    
    df = pd.DataFrame(data)
    
    # Print data info
    print(f"Loaded {len(df)} data points from all_results.pkl")
    print(f"Pipelines: {df['pipeline'].unique()}")
    print(f"Models: {df['model'].unique()}")
    print(f"Data options: {df['data_option'].unique()}")
    print(f"Datasets: {df['dataset'].unique()}")
    
    return df

def calculate_statistics(df):
    """
    Calculate mean AUC and 95% confidence intervals for each combination.
    
    Args:
        df (pd.DataFrame): Input data with columns: pipeline, model, data_option, dataset, run, auc
        
    Returns:
        pd.DataFrame: Aggregated statistics
    """
    # Group by all factors except 'run' and calculate statistics
    stats_df = df.groupby(['pipeline', 'model', 'data_option', 'dataset']).agg({
        'auc': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    stats_df.columns = ['pipeline', 'model', 'data_option', 'dataset', 'mean_auc', 'std_auc', 'count']
    
    # Calculate standard error and confidence interval
    stats_df['se_auc'] = stats_df['std_auc'] / np.sqrt(stats_df['count'])
    stats_df['ci_95_lower'] = stats_df['mean_auc'] - (1.96 * stats_df['se_auc'])
    stats_df['ci_95_upper'] = stats_df['mean_auc'] + (1.96 * stats_df['se_auc'])
    
    # Ensure confidence intervals don't go below 0.5 or above 1.0
    stats_df['ci_95_lower'] = np.clip(stats_df['ci_95_lower'], 0.5, 1.0)
    stats_df['ci_95_upper'] = np.clip(stats_df['ci_95_upper'], 0.5, 1.0)
    
    return stats_df

def create_publication_plot(stats_df, save_path='outputs/plots/auc_results_comparison.png'):
    """
    Create a publication-ready grouped bar plot with error bars.
    
    Args:
        stats_df (pd.DataFrame): Statistics data from calculate_statistics()
        save_path (str): Path to save the plot
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots for each pipeline
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Fig 2. AUC scores for classification of preterm and SGA babies ', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Color palette for data options
    colors = {'both_heel': '#1f77b4', 'both_cord': '#4a90e2', 'heel_all': '#ff7f0e', 'cord_all': '#2ca02c'}
    
    # Plot for each pipeline
    for i, pipeline in enumerate(['gestational_age', 'birth_weight']):
        ax = axes[i]
        
        # Filter data for this pipeline
        pipeline_data = stats_df[stats_df['pipeline'] == pipeline].copy()
        
        # Create grouped bar plot
        # Ensure datasets are in the correct order
        dataset_order = ['clinical', 'biomarkers', 'combined']
        x_positions = np.arange(len(dataset_order))
        width = 0.25  # Width of bars
        
        # Plot bars for each data option
        for j, data_option in enumerate(['both_heel', 'both_cord', 'heel_all', 'cord_all']):
            # Get data for this data option
            data_subset = pipeline_data[pipeline_data['data_option'] == data_option]
            
            if not data_subset.empty:
                # Position bars for each model
                for k, model in enumerate(['elasticnet', 'lasso']):
                    model_data = data_subset[data_subset['model'] == model]
                    
                    if not model_data.empty:
                        # Calculate bar positions
                        x_pos = x_positions + (j - 1) * width + k * (width / 2)
                        
                        # Plot bars with error bars
                        bars = ax.bar(x_pos, model_data['mean_auc'], 
                                     width/2, 
                                     yerr=[model_data['mean_auc'] - model_data['ci_95_lower'],
                                           model_data['ci_95_upper'] - model_data['mean_auc']],
                                     capsize=5,
                                     color=colors[data_option], 
                                     alpha=0.8 if k == 0 else 0.6,
                                     label=f'{data_option} ({model})' if k == 0 else None)
        
        # Customize the subplot
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('AUC Score', fontsize=14, fontweight='bold')
        
        # Set title based on pipeline
        if pipeline == 'gestational_age':
            title = 'Preterm'
        elif pipeline == 'birth_weight':
            title = 'SGA'
        else:
            title = pipeline.replace("_", " ").title()
        
        ax.set_title(title, 
                     fontsize=16, fontweight='bold')
        
        # Set x-axis ticks and labels
        ax.set_xticks(x_positions)
        # Get the actual order of datasets from the data
        actual_datasets = pipeline_data['dataset'].unique()
        # Map to display names
        display_names = []
        for dataset in actual_datasets:
            if dataset == 'clinical':
                display_names.append('Clinical')
            elif dataset == 'biomarkers':
                display_names.append('Biomarkers')
            elif dataset == 'combined':
                display_names.append('Combined')
        ax.set_xticklabels(display_names, fontsize=12, rotation=0)
        
        # Set y-axis limits and add horizontal line at 0.5
        ax.set_ylim(0.45, 1.0)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                  label='Random Classifier (AUC=0.5)')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        
        # Create custom legend with alpha information
        from matplotlib.patches import Patch
        
        # Create legend elements
        legend_elements = []
        
        # Add data option colors with model transparency info
        for data_option in ['both_heel', 'both_cord', 'heel_all', 'cord_all']:
            color = colors[data_option]
            # Elasticnet (more opaque)
            legend_elements.append(Patch(facecolor=color, alpha=0.8, 
                                       label=f'{data_option} (ElasticNet, α=0.8)'))
            # Lasso (more transparent)
            legend_elements.append(Patch(facecolor=color, alpha=0.6, 
                                       label=f'{data_option} (Lasso, α=0.6)'))
        
        # Add horizontal line legend
        from matplotlib.lines import Line2D
        legend_elements.append(Line2D([0], [0], color='red', linestyle='--', 
                                     label='Random Classifier (AUC=0.5)'))
        
        # Add legend
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
                 loc='upper left', fontsize=9)
        
        # Add value labels on bars
        for j, data_option in enumerate(['both_heel', 'both_cord', 'heel_all', 'cord_all']):
            # Get data for this data option
            data_subset = pipeline_data[pipeline_data['data_option'] == data_option]
            
            if not data_subset.empty:
                for k, model in enumerate(['elasticnet', 'lasso']):
                    model_data = data_subset[data_subset['model'] == model]
                    
                    if not model_data.empty:
                        x_pos = x_positions + (j - 1) * width + k * (width / 2)
                        
                        # Add value labels
                        for x, y in zip(x_pos, model_data['mean_auc']):
                            ax.text(x, y + 0.005, f'{y:.3f}', 
                                   ha='center', va='bottom', fontsize=4, fontweight='normal')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {save_path}")
    
    # Show the plot
    plt.show()

def create_alternative_plot(stats_df, save_path='outputs/plots/auc_results_alternative.png'):
    """
    Create an alternative visualization using seaborn's built-in faceting.
    
    Args:
        stats_df (pd.DataFrame): Statistics data from calculate_statistics()
        save_path (str): Path to save the plot
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create the plot using seaborn
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('ML Model Performance Comparison (Alternative View)', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Plot for each pipeline
    for i, pipeline in enumerate(['gestational_age', 'birth_weight']):
        ax = axes[i]
        
        # Filter data for this pipeline
        pipeline_data = stats_df[stats_df['pipeline'] == pipeline].copy()
        
        # Create grouped bar plot using seaborn
        sns.barplot(data=pipeline_data, 
                   x='dataset', 
                   y='mean_auc', 
                   hue='data_option',
                   palette=['#1f77b4', '#4a90e2', '#ff7f0e', '#2ca02c'],
                   ax=ax,
                   capsize=0.1,
                   errwidth=2)
        
        # Customize the subplot
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('AUC Score', fontsize=14, fontweight='bold')
        
        # Set title based on pipeline
        if pipeline == 'gestational_age':
            title = 'Preterm'
        elif pipeline == 'birth_weight':
            title = 'SGA'
        else:
            title = pipeline.replace("_", " ").title()
        
        ax.set_title(title, 
                     fontsize=16, fontweight='bold')
        
        # Set x-axis labels
        ax.set_xticklabels(['Clinical', 'Biomarkers', 'Combined'], 
                          fontsize=12, rotation=0)
        
        # Set y-axis limits and add horizontal line at 0.5
        ax.set_ylim(0.45, 1.0)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                  label='Random Classifier (AUC=0.5)')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        ax.legend(title='Data Option', fontsize=10)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=8, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Alternative plot saved to: {save_path}")
    
    # Show the plot
    plt.show()

def print_summary_statistics(stats_df):
    """
    Print summary statistics for the results.
    
    Args:
        stats_df (pd.DataFrame): Statistics data from calculate_statistics()
    """
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Total combinations: {len(stats_df)}")
    print(f"  Mean AUC across all: {stats_df['mean_auc'].mean():.3f}")
    print(f"  Best performing combination:")
    best_idx = stats_df['mean_auc'].idxmax()
    best = stats_df.loc[best_idx]
    print(f"    {best['pipeline']} | {best['model']} | {best['data_option']} | {best['dataset']}")
    print(f"    AUC: {best['mean_auc']:.3f} (95% CI: {best['ci_95_lower']:.3f}-{best['ci_95_upper']:.3f})")
    
    # Statistics by pipeline
    print(f"\nBy Pipeline:")
    for pipeline in ['gestational_age', 'birth_weight']:
        pipeline_data = stats_df[stats_df['pipeline'] == pipeline]
        print(f"  {pipeline.replace('_', ' ').title()}:")
        print(f"    Mean AUC: {pipeline_data['mean_auc'].mean():.3f}")
        print(f"    Best: {pipeline_data['mean_auc'].max():.3f}")
        print(f"    Worst: {pipeline_data['mean_auc'].min():.3f}")
    
    # Statistics by model
    print(f"\nBy Model:")
    for model in ['elasticnet', 'lasso']:
        model_data = stats_df[stats_df['model'] == model]
        print(f"  {model.title()}:")
        print(f"    Mean AUC: {model_data['mean_auc'].mean():.3f}")
        print(f"    Best: {model_data['mean_auc'].max():.3f}")
        print(f"    Worst: {model_data['mean_auc'].min():.3f}")
    
    # Statistics by data option
    print(f"\nBy Data Option:")
    for data_option in ['both', 'heel_all', 'cord_all']:
        option_data = stats_df[stats_df['data_option'] == data_option]
        print(f"  {data_option}:")
        print(f"    Mean AUC: {option_data['mean_auc'].mean():.3f}")
        print(f"    Best: {option_data['mean_auc'].max():.3f}")
        print(f"    Worst: {option_data['mean_auc'].min():.3f}")
    
    # Statistics by dataset
    print(f"\nBy Dataset:")
    for dataset in ['clinical', 'biomarkers', 'combined']:
        dataset_data = stats_df[stats_df['dataset'] == dataset]
        print(f"  {dataset.title()}:")
        print(f"    Mean AUC: {dataset_data['mean_auc'].mean():.3f}")
        print(f"    Best: {dataset_data['mean_auc'].max():.3f}")
        print(f"    Worst: {dataset_data['mean_auc'].min():.3f}")

def main():
    """Main function to run the analysis and create plots."""
    print("🔍 ML Experiment Results Visualization")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    df = load_actual_data()
    print(f"   Loaded {len(df)} data points")
    print(f"   Unique combinations: {len(df.groupby(['pipeline', 'model', 'data_option', 'dataset']))}")
    
    # Calculate statistics
    print("\n📈 Calculating statistics...")
    stats_df = calculate_statistics(df)
    print(f"   Calculated statistics for {len(stats_df)} combinations")
    
    # Print summary statistics
    print_summary_statistics(stats_df)
    
    # Create plots
    print("\n🎨 Creating plots...")
    
    # Main plot
    print("   Creating main publication plot...")
    create_publication_plot(stats_df)
    
    # Alternative plot
    print("   Creating alternative plot...")
    create_alternative_plot(stats_df)
    
    print("\n✅ Analysis complete!")
    print("📁 Plots saved to outputs/plots/")

if __name__ == "__main__":
    main()
