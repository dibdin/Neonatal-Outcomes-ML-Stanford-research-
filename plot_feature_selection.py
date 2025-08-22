#!/usr/bin/env python3
"""
Feature Selection Frequency Scatter Plot Script
==============================================

This script creates publication-ready grouped scatter plots showing feature selection
frequency across all experimental runs, with faceting by task and dataset.

Usage:
    python3 plot_feature_selection.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_feature_selection_data():
    """
    Load feature selection data from all_results.pkl and calculate selection frequencies.
    
    Returns:
        pd.DataFrame: Formatted data with feature selection frequencies
    """
    import pickle
    import numpy as np
    
    # Load the results
    with open('all_results.pkl', 'rb') as f:
        all_results = pickle.load(f)
    
    # Dictionary to store feature selection data
    feature_data = []
    
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
                model = parts[3]  # lasso or elasticnet  
                dataset = parts[5]  # Clinical, Biomarker, Combined
                pipeline = f"{parts[6]}_{parts[7]}"  # gestational_age or birth_weight
            # Handle "heel_all" or "cord_all" case  
            elif parts[1] == "all":
                data_option = f"{parts[0]}_all"  # heel_all or cord_all
                model = parts[3]  # lasso or elasticnet
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
        
        # Each experiment contains BOTH regression and classification data
        # We'll process both separately
        has_regression_coeffs = 'all_coefficients' in results and results['all_coefficients']
        has_classification_coeffs = 'all_classification_coefficients' in results and results['all_classification_coefficients']
        
        if not has_regression_coeffs and not has_classification_coeffs:
            continue
        
        # Extract feature selection data for REGRESSION task
        if has_regression_coeffs and 'feature_names' in results:
            coefficients = results['all_coefficients']
            feature_names = results['feature_names']
            
            # Process each run's coefficients
            for run_idx, run_coeff in enumerate(coefficients):
                coeff_array = np.array(run_coeff)
                
                # For each feature, check if it was selected (non-zero coefficient)
                for feat_idx, feature_name in enumerate(feature_names):
                    # Feature is selected if coefficient is non-zero
                    is_selected = int(abs(coeff_array[feat_idx]) > 1e-10)  # Use small threshold for numerical precision
                    
                    feature_data.append({
                        'task': 'regression',
                        'pipeline': pipeline,
                        'model': model,
                        'data_option': data_option,
                        'dataset': dataset,
                        'run': run_idx + 1,
                        'feature_name': feature_name,
                        'is_selected': is_selected
                    })
        
        # Extract feature selection data for CLASSIFICATION task
        if has_classification_coeffs and 'feature_names' in results:
            class_coefficients = results['all_classification_coefficients']
            feature_names = results['feature_names']
            
            # Process each run's classification coefficients
            for run_idx, run_coeff in enumerate(class_coefficients):
                coeff_array = np.array(run_coeff)
                
                # Handle different coefficient shapes (could be 1D or 2D)
                if coeff_array.ndim == 2:
                    # For multi-class classification, take the first class coefficients (positive class)
                    coeff_array = coeff_array[0, :]  # Take first row (first class)
                
                # Take only the number of features we have names for
                coeff_array = coeff_array[:len(feature_names)]
                
                # For each feature, check if it was selected (non-zero coefficient)
                for feat_idx, feature_name in enumerate(feature_names):
                    if feat_idx < len(coeff_array):
                        # Feature is selected if coefficient is non-zero
                        is_selected = int(abs(coeff_array[feat_idx]) > 1e-10)
                        
                        feature_data.append({
                            'task': 'classification',
                            'pipeline': pipeline,
                            'model': model,
                            'data_option': data_option,
                            'dataset': dataset,
                            'run': run_idx + 1,
                            'feature_name': feature_name,
                            'is_selected': is_selected
                        })
    
    df = pd.DataFrame(feature_data)
    
    if df.empty:
        print("⚠️  No feature selection data found in all_results.pkl")
        print("   Creating sample data for demonstration...")
        return create_sample_feature_data()
    
    print(f"✅ Loaded {len(df)} feature selection records from all_results.pkl")
    print(f"   Experiments with coefficient data: {df['pipeline'].nunique()} pipelines, {df['model'].nunique()} models")
    print(f"   Datasets: {df['dataset'].unique()}")
    print(f"   Data options: {df['data_option'].unique()}")
    print(f"   Tasks: {df['task'].unique()}")
    print(f"   Task counts: {df['task'].value_counts()}")
    
    # Calculate selection frequencies
    freq_df = calculate_selection_frequencies(df)
    
    return freq_df



def calculate_selection_frequencies(df):
    """
    Calculate feature selection frequencies across all runs.
    
    Args:
        df (pd.DataFrame): Raw feature selection data
        
    Returns:
        pd.DataFrame: Aggregated selection frequencies
    """
    print("📊 Calculating feature selection frequencies...")
    
    # Group by all dimensions and calculate mean selection frequency
    freq_df = df.groupby([
        'task', 'pipeline', 'model', 'data_option', 'dataset', 'feature_name'
    ])['is_selected'].mean().reset_index()
    
    # Rename column
    freq_df = freq_df.rename(columns={'is_selected': 'selection_frequency'})
    
    # Filter to only include features with frequency >= 0.5
    freq_df = freq_df[freq_df['selection_frequency'] >= 0.5].copy()
    
    print(f"   Found {len(freq_df)} feature combinations with frequency >= 0.5")
    print(f"   Unique features: {freq_df['feature_name'].nunique()}")
    
    return freq_df

def create_feature_selection_scatter_plot(df, save_path='outputs/plots/feature_selection_scatter.png'):
    """
    Create a publication-ready grouped scatter plot for feature selection frequencies.
    
    Args:
        df (pd.DataFrame): Feature selection frequency data
        save_path (str): Path to save the plot
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots for all four task combinations
    # We'll create a 2x2 grid for the four combinations
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    
    fig.suptitle('Fig 3. Feature Selection for all Classification (preterm/SGA) and Regression (gestational age/birth weight)', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Define colors and markers for different combinations
    model_colors = {'elasticnet': '#1f77b4', 'lasso': '#ff7f0e'}
    data_option_markers = {
        'both_heel': 'o',      # Circle
        'both_cord': 's',      # Square
        'heel_all': '^',       # Triangle up
        'cord_all': 'D'        # Diamond
    }
    
    # Define the four task combinations we want to plot
    # Top row: Classification (GA left, Birth Weight right)
    # Bottom row: Regression (GA left, Birth Weight right)
    task_combinations = [
        ('classification', 'gestational_age', 'Preterm'),    # Top-left
        ('classification', 'birth_weight', 'SGA'),          # Top-right
        ('regression', 'gestational_age', 'Gestational Age'),           # Bottom-left
        ('regression', 'birth_weight', 'Birth Weight')                  # Bottom-right
    ]
    
    # Plot for each task combination
    for idx, (task, pipeline, title) in enumerate(task_combinations):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Filter data for this task and pipeline combination
        task_data = df[(df['task'] == task) & (df['pipeline'] == pipeline)].copy()
        
        if not task_data.empty:
            # Calculate number of runs with frequency=1 for each feature and order by this metric
            feature_perfect_runs = task_data.groupby('feature_name').apply(
                lambda x: (x['selection_frequency'] == 1.0).sum()
            ).sort_values(ascending=True)  # Changed to ascending=True to reverse order
            ordered_features = feature_perfect_runs.index.tolist()
            
            # Create scatter plot with smart positioning for overlapping dots
            # Process all data points by feature and frequency to handle overlaps properly
            for feature_name in ordered_features:
                feature_data = task_data[task_data['feature_name'] == feature_name]
                y_pos = ordered_features.index(feature_name)
                
                # Group by frequency to handle overlapping points
                for freq in feature_data['selection_frequency'].unique():
                    freq_subset = feature_data[feature_data['selection_frequency'] == freq]
                    
                    if len(freq_subset) == 1:
                        # Single point, no overlap
                        row = freq_subset.iloc[0]
                        ax.scatter(freq, y_pos, 
                                  c=model_colors[row['model']], 
                                  marker=data_option_markers[row['data_option']],
                                  s=15, alpha=0.7, 
                                  label=f"{row['model']}_{row['data_option']}" if idx == 0 else "")
                    else:
                        # Multiple points with same frequency, spread them horizontally
                        spread_width = 0.08  # Width of spread
                        x_positions = []
                        for idx_inner, (_, row) in enumerate(freq_subset.iterrows()):
                            # Calculate horizontal offset
                            offset = (idx_inner - (len(freq_subset) - 1) / 2) * spread_width / len(freq_subset)
                            x_pos = freq + offset
                            x_positions.append(x_pos)
                            
                            ax.scatter(x_pos, y_pos, 
                                      c=model_colors[row['model']], 
                                      marker=data_option_markers[row['data_option']],
                                      s=15, alpha=0.7, 
                                      label=f"{row['model']}_{row['data_option']}" if idx == 0 else "")
                        
                        # Draw a thin line through the spread points
                        if len(x_positions) > 1:
                            ax.plot([min(x_positions), max(x_positions)], [y_pos, y_pos], 
                                   color='gray', linewidth=0.5, alpha=0.5, zorder=0)
            
            # Customize subplot
            ax.set_xlabel('Selection Frequency', fontsize=14, fontweight='bold')
            ax.set_ylabel('Features', fontsize=14, fontweight='bold')
            ax.set_title(f'{title}', fontsize=16, fontweight='bold')
            
            # Set x-axis limits (extend beyond 1 for better visibility)
            ax.set_xlim(0.55, 1.05)
            
            # Set y-axis ticks and labels for all plotted features (bigger font)
            if ordered_features:
                ax.set_yticks(range(len(ordered_features)))
                ax.set_yticklabels(ordered_features, fontsize=7)
                ax.set_ylim(-0.5, len(ordered_features) - 0.5)
        else:
            # No data for this combination
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{title}', fontsize=16, fontweight='bold')
            ax.set_xlim(0.55, 1.05)
            ax.set_ylim(0, 1)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add vertical line at 0.5 threshold
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold (0.5)')
    
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
    
    # Add threshold line
    legend_elements.append(Line2D([0], [0], color='red', linestyle='--', 
                                label='Selection Threshold (0.5)'))
    
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
    print(f"✅ Feature selection scatter plot saved to: {save_path}")
    
    # Show the plot
    plt.show()

def print_feature_summary(df):
    """
    Print comprehensive summary of feature selection results.
    
    Args:
        df (pd.DataFrame): Feature selection frequency data
    """
    print("\n" + "="*80)
    print("FEATURE SELECTION SUMMARY")
    print("="*80)
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Total feature combinations: {len(df)}")
    print(f"  Mean selection frequency: {df['selection_frequency'].mean():.3f}")
    print(f"  Features with frequency >= 0.8: {len(df[df['selection_frequency'] >= 0.8])}")
    print(f"  Features with frequency >= 0.9: {len(df[df['selection_frequency'] >= 0.9])}")
    
    # Statistics by task
    print(f"\nBy Task:")
    for task in df['task'].unique():
        subset = df[df['task'] == task]
        print(f"  {task.title()}:")
        print(f"    Mean frequency: {subset['selection_frequency'].mean():.3f}")
        print(f"    High frequency features (>=0.8): {len(subset[subset['selection_frequency'] >= 0.8])}")
    
    # Statistics by model
    print(f"\nBy Model:")
    for model in df['model'].unique():
        subset = df[df['model'] == model]
        print(f"  {model.title()}:")
        print(f"    Mean frequency: {subset['selection_frequency'].mean():.3f}")
        print(f"    High frequency features (>=0.8): {len(subset[subset['selection_frequency'] >= 0.8])}")
    
    # Statistics by dataset
    print(f"\nBy Dataset:")
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        print(f"  {dataset.title()}:")
        print(f"    Mean frequency: {subset['selection_frequency'].mean():.3f}")
        print(f"    High frequency features (>=0.8): {len(subset[subset['selection_frequency'] >= 0.8])}")
    
    # Top features by selection frequency
    print(f"\nTop 10 Features by Selection Frequency:")
    top_features = df.nlargest(10, 'selection_frequency')
    for idx, row in top_features.iterrows():
        print(f"  {row['feature_name']}: {row['selection_frequency']:.3f} "
              f"({row['task']}, {row['model']}, {row['data_option']})")

def main():
    """Main function to run the feature selection analysis."""
    print("🔍 Feature Selection Frequency Analysis")
    print("=" * 50)
    
    # Load data
    print("📊 Loading feature selection data...")
    df = load_feature_selection_data()
    print(f"   Loaded {len(df)} feature combinations with frequency >= 0.5")
    
    # Print summary
    print_feature_summary(df)
    
    # Create plots
    print("\n🎨 Creating feature selection plots...")
    
    # Main scatter plot
    print("   Creating main feature selection scatter plot...")
    create_feature_selection_scatter_plot(df)
    
    print("\n✅ Feature selection analysis complete!")
    print("📁 Plots saved to outputs/plots/")

if __name__ == "__main__":
    main()
