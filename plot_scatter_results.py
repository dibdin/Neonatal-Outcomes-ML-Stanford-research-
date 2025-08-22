#!/usr/bin/env python3
"""
Multi-Run Experimental Results Scatter Plot Script
==================================================

This script creates publication-ready grouped scatter plots for multi-run experimental results
with faceting by datasets, showing individual run results and group means.

Usage:
    python3 plot_scatter_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_experimental_data():
    """
    Load experimental data from all_results.pkl and format for scatter plotting.
    
    Returns:
        pd.DataFrame: Formatted data with all required columns
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
        
        # Get metrics for each run
        aucs = results.get('aucs', [])
        rmses = results.get('rmses', [])
        
        # Create rows for each run
        for run_idx in range(len(aucs)):
            data.append({
                'task': 'classification',  # AUC is for classification
                'pipeline': pipeline,
                'model': model,
                'data_option': data_option,
                'dataset': dataset,
                'run': run_idx + 1,
                'auc': float(aucs[run_idx]) if run_idx < len(aucs) else np.nan,
                'rmse': float(rmses[run_idx]) if run_idx < len(rmses) else np.nan
            })
    
    df = pd.DataFrame(data)
    
    # Print data info
    print(f"Loaded {len(df)} data points from all_results.pkl")
    print(f"Tasks: {df['task'].unique()}")
    print(f"Pipelines: {df['pipeline'].unique()}")
    print(f"Models: {df['model'].unique()}")
    print(f"Data options: {df['data_option'].unique()}")
    print(f"Datasets: {df['dataset'].unique()}")
    
    return df

def create_grouped_scatter_plot(df, save_path='outputs/plots/experimental_results_scatter.png'):
    """
    Create a publication-ready grouped scatter plot with faceting by datasets.
    
    Args:
        df (pd.DataFrame): Formatted experimental data
        save_path (str): Path to save the plot
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots for each dataset
    datasets = ['clinical', 'biomarkers', 'combined']
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Multi-Run Experimental Results Across Datasets', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Define colors and markers
    model_colors = {'elasticnet': '#1f77b4', 'lasso': '#ff7f0e'}
    data_option_markers = {
        'both_heel': 'o',      # Circle
        'both_cord': 's',      # Square
        'heel_all': '^',       # Triangle up
        'cord_all': 'D'        # Diamond
    }
    
    # Plot for each dataset
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        
        # Filter data for this dataset
        dataset_data = df[df['dataset'] == dataset].copy()
        
        # Create scatter plot
        for model in ['elasticnet', 'lasso']:
            for data_option in ['both_heel', 'both_cord', 'heel_all', 'cord_all']:
                # Filter data
                subset = dataset_data[(dataset_data['model'] == model) & 
                                    (dataset_data['data_option'] == data_option)]
                
                if not subset.empty:
                    # Plot individual run results
                    ax.scatter(subset['pipeline'], subset['auc'], 
                              c=model_colors[model], 
                              marker=data_option_markers[data_option],
                              s=60, alpha=0.7, 
                              label=f'{model}_{data_option}' if i == 0 else "")
                    
                    # Add mean values as larger points
                    mean_auc = subset['auc'].mean()
                    ax.scatter(subset['pipeline'].iloc[0], mean_auc,
                              c=model_colors[model],
                              marker=data_option_markers[data_option],
                              s=200, alpha=0.9, edgecolors='black', linewidth=2)
        
        # Customize subplot
        ax.set_xlabel('Pipeline', fontsize=14, fontweight='bold')
        ax.set_ylabel('AUC Score', fontsize=14, fontweight='bold')
        ax.set_title(f'{dataset.title()} Dataset', fontsize=16, fontweight='bold')
        
        # Set x-axis labels
        ax.set_xticklabels(['Gestational\nAge', 'Birth\nWeight'], 
                          fontsize=12, rotation=0)
        
        # Set y-axis limits
        ax.set_ylim(0.5, 1.0)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at random classifier
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    
    # Create custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    # Create legend elements
    legend_elements = []
    
    # Add model colors
    for model, color in model_colors.items():
        legend_elements.append(Patch(facecolor=color, 
                                   label=f'{model.title()} Model'))
    
    # Add data option markers
    for data_option, marker in data_option_markers.items():
        legend_elements.append(Line2D([0], [0], marker=marker, color='gray', 
                                    linestyle='', markersize=10,
                                    label=f'{data_option.replace("_", " ").title()}'))
    
    # Add mean indicator
    legend_elements.append(Line2D([0], [0], marker='o', color='gray', 
                                linestyle='', markersize=15, markeredgecolor='black',
                                markeredgewidth=2, label='Mean Value'))
    
    # Add random classifier line
    legend_elements.append(Line2D([0], [0], color='red', linestyle='--', 
                                label='Random Classifier (AUC=0.5)'))
    
    # Add legend
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.05), ncol=5, fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save the plot
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Scatter plot saved to: {save_path}")
    
    # Show the plot
    plt.show()

def create_alternative_scatter_plot(df, save_path='outputs/plots/experimental_results_scatter_alt.png'):
    """
    Create an alternative scatter plot using seaborn for better statistical visualization.
    
    Args:
        df (pd.DataFrame): Formatted experimental data
        save_path (str): Path to save the plot
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Experimental Results with Statistical Summary', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Plot for each dataset
    for i, dataset in enumerate(['clinical', 'biomarkers', 'combined']):
        ax = axes[i]
        
        # Filter data for this dataset
        dataset_data = df[df['dataset'] == dataset].copy()
        
        # Create scatter plot with seaborn
        sns.scatterplot(data=dataset_data, x='pipeline', y='auc', 
                       hue='model', style='data_option',
                       s=100, alpha=0.7, ax=ax)
        
        # Add mean points
        means = dataset_data.groupby(['pipeline', 'model', 'data_option'])['auc'].mean().reset_index()
        sns.scatterplot(data=means, x='pipeline', y='auc', 
                       hue='model', style='data_option',
                       s=300, alpha=0.9, ax=ax, 
                       legend=False, edgecolor='black', linewidth=2)
        
        # Customize subplot
        ax.set_xlabel('Pipeline', fontsize=14, fontweight='bold')
        ax.set_ylabel('AUC Score', fontsize=14, fontweight='bold')
        ax.set_title(f'{dataset.title()} Dataset', fontsize=16, fontweight='bold')
        
        # Set x-axis labels
        ax.set_xticklabels(['Gestational\nAge', 'Birth\nWeight'], 
                          fontsize=12, rotation=0)
        
        # Set y-axis limits
        ax.set_ylim(0.5, 1.0)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at random classifier
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        
        # Remove legend from individual subplots
        ax.get_legend().remove()
    
    # Create unified legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', 
              bbox_to_anchor=(0.5, 0.05), ncol=4, fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save the plot
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Alternative scatter plot saved to: {save_path}")
    
    # Show the plot
    plt.show()

def print_statistical_summary(df):
    """
    Print comprehensive statistical summary of the experimental results.
    
    Args:
        df (pd.DataFrame): Formatted experimental data
    """
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Total runs: {len(df)}")
    print(f"  Mean AUC: {df['auc'].mean():.3f}")
    print(f"  Std AUC: {df['auc'].std():.3f}")
    
    # Statistics by task and pipeline
    print(f"\nBy Task and Pipeline:")
    for task in df['task'].unique():
        for pipeline in df['pipeline'].unique():
            subset = df[(df['task'] == task) & (df['pipeline'] == pipeline)]
            if not subset.empty:
                print(f"  {task.title()} - {pipeline.replace('_', ' ').title()}:")
                print(f"    Mean AUC: {subset['auc'].mean():.3f}")
                print(f"    Best: {subset['auc'].max():.3f}")
                print(f"    Worst: {subset['auc'].min():.3f}")
    
    # Statistics by model
    print(f"\nBy Model:")
    for model in df['model'].unique():
        subset = df[df['model'] == model]
        print(f"  {model.title()}:")
        print(f"    Mean AUC: {subset['auc'].mean():.3f}")
        print(f"    Best: {subset['auc'].max():.3f}")
        print(f"    Worst: {subset['auc'].min():.3f}")
    
    # Statistics by data option
    print(f"\nBy Data Option:")
    for data_option in df['data_option'].unique():
        subset = df[df['data_option'] == data_option]
        print(f"  {data_option.replace('_', ' ').title()}:")
        print(f"    Mean AUC: {subset['auc'].mean():.3f}")
        print(f"    Best: {subset['auc'].max():.3f}")
        print(f"    Worst: {subset['auc'].min():.3f}")
    
    # Statistics by dataset
    print(f"\nBy Dataset:")
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        print(f"  {dataset.title()}:")
        print(f"    Mean AUC: {subset['auc'].mean():.3f}")
        print(f"    Best: {subset['auc'].max():.3f}")
        print(f"    Worst: {subset['auc'].min():.3f}")

def main():
    """Main function to run the scatter plot analysis."""
    print("🔍 Multi-Run Experimental Results Scatter Plot Analysis")
    print("=" * 60)
    
    # Load data
    print("📊 Loading experimental data...")
    df = load_experimental_data()
    print(f"   Loaded {len(df)} data points")
    print(f"   Unique combinations: {len(df.groupby(['task', 'pipeline', 'model', 'data_option', 'dataset']))}")
    
    # Print statistical summary
    print_statistical_summary(df)
    
    # Create plots
    print("\n🎨 Creating scatter plots...")
    
    # Main scatter plot
    print("   Creating main grouped scatter plot...")
    create_grouped_scatter_plot(df)
    
    # Alternative scatter plot
    print("   Creating alternative scatter plot...")
    create_alternative_scatter_plot(df)
    
    print("\n✅ Scatter plot analysis complete!")
    print("📁 Plots saved to outputs/plots/")

if __name__ == "__main__":
    main()
