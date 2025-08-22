#!/usr/bin/env python3
"""
Feature Selection Frequency Comparison for Best Models (Heel vs Cord)

This script creates a grouped scatter plot comparing feature selection frequencies
between heel and cord data for the best performing regression and classification models.

Author: Diba Dindoust
Date: 08/20/2025
"""

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from src.utils import count_high_weight_biomarkers
from matplotlib.patches import Rectangle

def load_all_results(results_file="all_results.pkl"):
    """
    Load and validate the all_results.pkl file.
    
    Args:
        results_file (str): Path to the results file
        
    Returns:
        dict: Loaded results data
    """
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file {results_file} not found.")
    
    with open(results_file, 'rb') as f:
        all_results = pickle.load(f)
    
    print(f"✅ Loaded {len(all_results)} result entries from {results_file}")
    return all_results

def identify_best_models(all_results):
    """
    Identify the best performing models for each task and pipeline.
    
    Args:
        all_results (dict): Results data from pickle file
        
    Returns:
        dict: Best models organized by task and pipeline
    """
    print("🔍 Identifying best performing models...")
    
    best_models = {
        'regression': {
            'gestational_age': None,
            'birth_weight': None
        },
        'classification': {
            'gestational_age': None,
            'birth_weight': None
        }
    }
    
    # Track performance metrics for each model
    model_performance = []
    
    for key, results in all_results.items():
        # Parse key to extract components
        if 'both_samples' in key and 'Biomarker' in key:
            # Extract model type
            if 'elasticnet' in key:
                model_type = 'elasticnet'
            elif 'lasso' in key:
                model_type = 'lasso'
            else:
                continue
            
            # Extract pipeline
            if 'gestational_age' in key:
                pipeline = 'gestational_age'
            elif 'birth_weight' in key:
                pipeline = 'birth_weight'
            else:
                continue
            
            # Extract dataset type
            if 'heel' in key:
                dataset_type = 'heel'
            elif 'cord' in key:
                dataset_type = 'cord'
            else:
                continue
            
            # Get performance metrics
            if 'maes' in results and 'rmses' in results and 'aucs' in results:
                mae = np.mean(results['maes'])
                rmse = np.mean(results['rmses'])
                auc = np.mean(results['aucs'])
                
                model_performance.append({
                    'key': key,
                    'model_type': model_type,
                    'pipeline': pipeline,
                    'dataset_type': dataset_type,
                    'mae': mae,
                    'rmse': rmse,
                    'auc': auc,
                    'results': results
                })
    
    # Find best models for each task and pipeline
    for task in ['regression', 'classification']:
        for pipeline in ['gestational_age', 'birth_weight']:
            # Filter models for this task and pipeline
            if task == 'regression':
                # For regression, minimize RMSE
                task_models = [m for m in model_performance if m['pipeline'] == pipeline]
                if task_models:
                    best_model = min(task_models, key=lambda x: x['rmse'])
                    best_models[task][pipeline] = best_model
                    print(f"  Best {task} {pipeline}: {best_model['model_type']} (RMSE: {best_model['rmse']:.4f})")
            else:
                # For classification, maximize AUC
                task_models = [m for m in model_performance if m['pipeline'] == pipeline]
                if task_models:
                    best_model = max(task_models, key=lambda x: x['auc'])
                    best_models[task][pipeline] = best_model
                    print(f"  Best {task} {pipeline}: {best_model['model_type']} (AUC: {best_model['auc']:.4f})")
    
    return best_models

def extract_feature_frequencies(best_models, all_results):
    """
    Extract feature selection frequencies for heel vs cord comparison.
    
    Args:
        best_models (dict): Best models identified
        all_results (dict): All results data
        
    Returns:
        pd.DataFrame: Feature frequency comparison data
    """
    print("📊 Extracting feature selection frequencies...")
    
    comparison_data = []
    
    # Extract data for both biomarker and combined datasets
    for dataset_type in ['Biomarker', 'Combined']:
        for task in ['regression', 'classification']:
            for pipeline in ['gestational_age', 'birth_weight']:
                best_model = best_models[task][pipeline]
                if best_model is None:
                    continue
                
                model_type = best_model['model_type']
                pipeline_name = pipeline
                
                # Get heel and cord results for this best model and dataset type
                heel_key = f"both_samples_heel_{model_type}_cv_{dataset_type}_{pipeline}"
                cord_key = f"both_samples_cord_{model_type}_cv_{dataset_type}_{pipeline}"
                
                heel_result = all_results.get(heel_key)
                cord_result = all_results.get(cord_key)
                
                if not (heel_result and cord_result):
                    print(f"    ⚠️  Missing heel or cord data for {task} {pipeline} {dataset_type}")
                    continue
                
                # Calculate feature frequencies
                if task == 'regression':
                    heel_coefs = heel_result['all_coefficients']
                    cord_coefs = cord_result['all_coefficients']
                else:
                    heel_coefs = heel_result['all_classification_coefficients']
                    cord_coefs = cord_result['all_classification_coefficients']
                
                heel_feature_names = heel_result['feature_names']
                cord_feature_names = cord_result['feature_names']
                
                # Calculate frequencies
                heel_freq = count_high_weight_biomarkers(heel_coefs, heel_feature_names, threshold=0.01)
                cord_freq = count_high_weight_biomarkers(cord_coefs, cord_feature_names, threshold=0.01)
                
                # Create frequency dictionaries
                heel_freq_dict = dict(zip(heel_feature_names, heel_freq))
                cord_freq_dict = dict(zip(cord_feature_names, cord_freq))
                
                # Get union of all features
                all_features = set(heel_feature_names) | set(cord_feature_names)
                
                # Only include features with frequency >= 0.5 in at least one dataset
                for feature in all_features:
                    heel_freq_val = heel_freq_dict.get(feature, 0)
                    cord_freq_val = cord_freq_dict.get(feature, 0)
                    
                    if heel_freq_val >= 0.5 or cord_freq_val >= 0.5:
                        comparison_data.append({
                            'task': task,
                            'pipeline': pipeline_name,
                            'model': model_type,
                            'data_option': 'both_samples',
                            'dataset': dataset_type.lower(),
                            'feature_name': feature,
                            'heel_frequency': heel_freq_val,
                            'cord_frequency': cord_freq_val
                        })
    
    # Also add heel_all and cord_all data for both dataset types
    for dataset_type in ['Biomarker', 'Combined']:
        for task in ['regression', 'classification']:
            for pipeline in ['gestational_age', 'birth_weight']:
                best_model = best_models[task][pipeline]
                if best_model is None:
                    continue
                
                model_type = best_model['model_type']
                pipeline_name = pipeline
                
                # Get heel_all and cord_all results
                heel_key = f"heel_all_heel_{model_type}_cv_{dataset_type}_{pipeline}"
                cord_key = f"cord_all_cord_{model_type}_cv_{dataset_type}_{pipeline}"
                
                heel_result = all_results.get(heel_key)
                cord_result = all_results.get(cord_key)
                
                if not (heel_result and cord_result):
                    continue
                
                # Calculate feature frequencies
                if task == 'regression':
                    heel_coefs = heel_result['all_coefficients']
                    cord_coefs = cord_result['all_coefficients']
                else:
                    heel_coefs = heel_result['all_classification_coefficients']
                    cord_coefs = cord_result['all_classification_coefficients']
                
                heel_feature_names = heel_result['feature_names']
                cord_feature_names = cord_result['feature_names']
                
                # Calculate frequencies
                heel_freq = count_high_weight_biomarkers(heel_coefs, heel_feature_names, threshold=0.01)
                cord_freq = count_high_weight_biomarkers(cord_coefs, cord_feature_names, threshold=0.01)
                
                # Create frequency dictionaries
                heel_freq_dict = dict(zip(heel_feature_names, heel_freq))
                cord_freq_dict = dict(zip(cord_feature_names, cord_freq))
                
                # Get union of all features
                all_features = set(heel_feature_names) | set(cord_feature_names)
                
                # Only include features with frequency >= 0.5 in at least one dataset
                for feature in all_features:
                    heel_freq_val = heel_freq_dict.get(feature, 0)
                    cord_freq_val = cord_freq_dict.get(feature, 0)
                    
                    if heel_freq_val >= 0.5 or cord_freq_val >= 0.5:
                        comparison_data.append({
                            'task': task,
                            'pipeline': pipeline_name,
                            'model': model_type,
                            'data_option': 'heel_cord_all',
                            'dataset': dataset_type.lower(),
                            'feature_name': feature,
                            'heel_frequency': heel_freq_val,
                            'cord_frequency': cord_freq_val
                        })
    
    df = pd.DataFrame(comparison_data)
    
    if df.empty:
        print("⚠️  No comparison data found")
        return None
    
    print(f"✅ Extracted {len(df)} feature frequency comparisons")
    print(f"   Tasks: {df['task'].unique()}")
    print(f"   Pipelines: {df['pipeline'].unique()}")
    print(f"   Models: {df['model'].unique()}")
    print(f"   Data options: {df['data_option'].unique()}")
    
    return df



def create_best_models_comparison_plot(df, save_path='outputs/plots/best_models_heel_vs_cord.png'):
    """
    Create a grouped scatter plot comparing heel vs cord feature selection frequencies.
    
    Args:
        df (pd.DataFrame): Feature frequency comparison data
        save_path (str): Path to save the plot
    """
    print("🎨 Creating best models comparison plot...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with 2x2 subplots: gestational age (top row), birth weight (bottom row)
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    
    fig.suptitle('Fig 4. Feature Selection Frequency (Heel vs Cord) — Best Regression and Classification Models', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Define colors for each unique combination
    # Create a color palette for different combinations
    import matplotlib.colors as mcolors
    
    # Define base colors for model + data option + dataset combinations
    # Same color across different plots (classification vs regression) for same combination
    combination_colors = {
        # Biomarkers dataset
        ('elasticnet', 'both_samples', 'biomarkers'): '#1f77b4',      # Blue
        ('lasso', 'both_samples', 'biomarkers'): '#ff7f0e',           # Orange
        ('elasticnet', 'heel_cord_all', 'biomarkers'): '#2ca02c',     # Green
        ('lasso', 'heel_cord_all', 'biomarkers'): '#d62728',          # Red
        
        # Combined dataset
        ('elasticnet', 'both_samples', 'combined'): '#9467bd',        # Purple
        ('lasso', 'both_samples', 'combined'): '#8c564b',             # Brown
        ('elasticnet', 'heel_cord_all', 'combined'): '#e377c2',       # Pink
        ('lasso', 'heel_cord_all', 'combined'): '#17becf'             # Cyan (much more distinct from red)
    }
    
    model_markers = {
        'elasticnet': 'o',  # Circle
        'lasso': 's'        # Square
    }
    
    data_option_alphas = {
        'both_samples': 0.9,
        'heel_cord_all': 0.4
    }
    
    # Plot for each pipeline (gestational age and birth weight)
    for pipeline_idx, pipeline in enumerate(['gestational_age', 'birth_weight']):
        # Filter data for this pipeline
        pipeline_data = df[df['pipeline'] == pipeline].copy()
        
        if pipeline_data.empty:
            # Handle empty data for both subplots in this row
            for task_idx, task in enumerate(['classification', 'regression']):
                ax = axes[pipeline_idx, task_idx]
                ax.text(0.5, 0.5, f'No {pipeline.replace("_", " ").title()} {task} data available', 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=14, color='gray')
                
                # Set title based on pipeline and task
                if pipeline == 'gestational_age' and task == 'classification':
                    title = 'Preterm'
                elif pipeline == 'birth_weight' and task == 'classification':
                    title = 'SGA'
                elif pipeline == 'gestational_age' and task == 'regression':
                    title = 'Gestational Age'
                elif pipeline == 'birth_weight' and task == 'regression':
                    title = 'Birth Weight'
                else:
                    title = f'{pipeline.replace("_", " ").title()} - {task.capitalize()}'
                
                ax.set_title(title, fontsize=16, fontweight='bold')
            continue
        
        # Plot for each task (classification and regression)
        for task_idx, task in enumerate(['classification', 'regression']):
            ax = axes[pipeline_idx, task_idx]
            
            # Filter data for this task and pipeline
            task_data = pipeline_data[pipeline_data['task'] == task].copy()
            
            if task_data.empty:
                ax.text(0.5, 0.5, f'No {task} data available', 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=14, color='gray')
                
                # Set title based on pipeline and task
                if pipeline == 'gestational_age' and task == 'classification':
                    title = 'Preterm'
                elif pipeline == 'birth_weight' and task == 'classification':
                    title = 'SGA'
                elif pipeline == 'gestational_age' and task == 'regression':
                    title = 'Gestational Age'
                elif pipeline == 'birth_weight' and task == 'regression':
                    title = 'Birth Weight'
                else:
                    title = f'{pipeline.replace("_", " ").title()} - {task.capitalize()}'
                
                ax.set_title(title, fontsize=16, fontweight='bold')
                continue
            
            # Create scatter plot
            legend_elements = []
            legend_labels = []
            
            # Collect all points and their positions for smart label positioning

            all_points_data = []  # Store (x, y, feature, color, marker, alpha) for each point
            
            for model in task_data['model'].unique():
                for data_option in task_data['data_option'].unique():
                    for dataset in task_data['dataset'].unique():
                        # Filter data for this combination
                        mask = ((task_data['model'] == model) & 
                               (task_data['data_option'] == data_option) &
                               (task_data['dataset'] == dataset))
                        
                        subset = task_data[mask]
                        
                        if not subset.empty:
                            # Create unique identifier for legend
                            legend_key = f"{model.title()} {data_option.replace('_', ' ').title()} {dataset.title()}"
                            
                            # Get color for this specific combination (same color across plots for same model + data option + dataset)
                            color_key = (model, data_option, dataset)
                            color = combination_colors.get(color_key, '#666666')  # Default gray if not found
                            
                            marker = model_markers[model]
                            alpha = data_option_alphas[data_option]
                            
                            # Handle overlapping dots by spreading them in circular clusters
                            heel_freqs = subset['heel_frequency'].values
                            cord_freqs = subset['cord_frequency'].values
                            feature_names = subset['feature_name'].values
                            
                            # Group overlapping points
                            from collections import defaultdict
                            point_groups = defaultdict(list)
                            
                            # Round coordinates to group nearby points (tolerance of 0.02)
                            tolerance = 0.02
                            for i, (heel, cord, feature) in enumerate(zip(heel_freqs, cord_freqs, feature_names)):
                                rounded_heel = round(heel / tolerance) * tolerance
                                rounded_cord = round(cord / tolerance) * tolerance
                                point_groups[(rounded_heel, rounded_cord)].append((heel, cord, feature, i))
                            
                            # Process each group and collect positions
                            for (center_heel, center_cord), points in point_groups.items():
                                if len(points) == 1:
                                    # Single point - plot normally
                                    heel, cord, feature, _ = points[0]
                                    all_points_data.append((heel, cord, feature, color, marker, alpha))

                                else:
                                    # Multiple overlapping points - spread horizontally with line
                                    print(f"🔍 CHECKPOINT: Found {len(points)} overlapping points at ({center_heel:.3f}, {center_cord:.3f})")
                                    print(f"   Features: {[p[2] for p in points]}")
                                    
                                    # Adjust spread width based on number of overlapping points
                                    if len(points) > 3:
                                        spread_width = 0.1  # Wider spread for many overlapping points
                                    else:
                                        spread_width = 0.02  # Narrower spread for few overlapping points
                                    
                                    print(f"   Spread width: {spread_width}")
                                    
                                    # Plot dots spread horizontally along the line
                                    for i, (heel, cord, feature, _) in enumerate(points):
                                        # Calculate horizontal position along the line
                                        # Spread evenly across the line width
                                        offset_x = (i - (len(points)-1)/2) * (spread_width / (len(points)-1))
                                        
                                        # Store the spread position
                                        spread_x = center_heel + offset_x
                                        spread_y = center_cord
                                        
                                        print(f"   Point {i}: {feature} -> ({spread_x:.3f}, {spread_y:.3f}) [offset: {offset_x:.3f}]")
                                        
                                        all_points_data.append((spread_x, spread_y, feature, color, marker, alpha))

                                    
                                    print(f"   ✅ CHECKPOINT: Dots spread horizontally across {spread_width:.3f} units")
                                    print()
                            
                            # Add to legend only once per combination
                            if legend_key not in legend_labels:
                                # Create a dummy scatter for legend
                                dummy_scatter = ax.scatter([], [], c=color, marker=marker, s=30, alpha=alpha, label=legend_key)
                                legend_elements.append(dummy_scatter)
                                legend_labels.append(legend_key)
            
            # Draw horizontal lines for clusters (plot lines first, then points on top)
            for model in task_data['model'].unique():
                for data_option in task_data['data_option'].unique():
                    for dataset in task_data['dataset'].unique():
                        mask = ((task_data['model'] == model) & 
                               (task_data['data_option'] == data_option) &
                               (task_data['dataset'] == dataset))
                        
                        subset = task_data[mask]
                        
                        if not subset.empty:
                            heel_freqs = subset['heel_frequency'].values
                            cord_freqs = subset['cord_frequency'].values
                            feature_names = subset['feature_name'].values
                            
                            # Group overlapping points
                            point_groups = defaultdict(list)
                            tolerance = 0.02
                            for i, (heel, cord, feature) in enumerate(zip(heel_freqs, cord_freqs, feature_names)):
                                rounded_heel = round(heel / tolerance) * tolerance
                                rounded_cord = round(cord / tolerance) * tolerance
                                point_groups[(rounded_heel, rounded_cord)].append((heel, cord, feature, i))
                            
                            # Draw lines for clusters with multiple points
                            for (center_heel, center_cord), points in point_groups.items():
                                if len(points) > 1:
                                    # Adjust spread width based on number of overlapping points
                                    if len(points) > 3:
                                        spread_width = 0.1  # Wider spread for many overlapping points
                                    else:
                                        spread_width = 0.02  # Narrower spread for few overlapping points
                                    
                                    # Draw horizontal line through the cluster
                                    line_x_start = center_heel - spread_width/2
                                    line_x_end = center_heel + spread_width/2
                                    ax.plot([line_x_start, line_x_end], [center_cord, center_cord], 
                                           color='black', alpha=0.2, linewidth=2.0, zorder=0)
            
            # Filter out points with x < 0.5 or y < 0.5
            filtered_points_data = []
            for x, y, feature, color, marker, alpha in all_points_data:
                if x >= 0.5 and y >= 0.5:
                    filtered_points_data.append((x, y, feature, color, marker, alpha))
            
            # Group overlapping points for label positioning
            from collections import defaultdict
            point_groups = defaultdict(list)
            tolerance = 0.02  # Same tolerance as used for spreading dots
            
            for i, (x, y, feature, color, marker, alpha) in enumerate(filtered_points_data):
                # Round to tolerance to group nearby coordinates
                rounded_x = round(x / tolerance) * tolerance
                rounded_y = round(y / tolerance) * tolerance
                point_groups[(rounded_x, rounded_y)].append((x, y, feature, color, marker, alpha, i))
            
            # Now plot all points and add labels with smart positioning
            for (center_x, center_y), points in point_groups.items():
                if len(points) == 1:
                    # Single point - plot normally with label above
                    x, y, feature, color, marker, alpha, i = points[0]
                    ax.scatter(x, y, c=color, marker=marker, s=30, alpha=alpha)
                    ax.annotate(feature, (x, y),
                              xytext=(0, 8), textcoords='offset points',
                              fontsize=5, alpha=0.8, ha='center', va='bottom')
                else:
                    # Multiple overlapping points - plot all points, stack labels above center
                    for x, y, feature, color, marker, alpha, i in points:
                        ax.scatter(x, y, c=color, marker=marker, s=30, alpha=alpha)
                    
                    # Stack labels vertically above the center point
                    for i, (x, y, feature, color, marker, alpha, idx) in enumerate(points):
                        offset = 8 + (i * 12)  # Stack labels with consistent spacing: 8, 20, 32, 44, ...
                        ax.annotate(feature, (center_x, center_y),
                                  xytext=(0, offset), textcoords='offset points',
                                  fontsize=5, alpha=0.8, ha='center', va='bottom')
            
            # Add reference lines
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Threshold (0.5)')
            ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            # Add diagonal line
            ax.plot([0.5, 1], [0.5, 1], 'k:', alpha=0.5, linewidth=1, label='Perfect agreement')
            
            # Set labels and title
            ax.set_xlabel('Heel Feature Selection Frequency', fontsize=14, fontweight='bold')
            ax.set_ylabel('Cord Feature Selection Frequency', fontsize=14, fontweight='bold')
            
            # Set title based on pipeline and task
            if pipeline == 'gestational_age' and task == 'classification':
                title = 'Preterm'
            elif pipeline == 'birth_weight' and task == 'classification':
                title = 'SGA'
            elif pipeline == 'gestational_age' and task == 'regression':
                title = 'Gestational Age'
            elif pipeline == 'birth_weight' and task == 'regression':
                title = 'Birth Weight'
            else:
                title = f'{pipeline.replace("_", " ").title()} - {task.capitalize()}'
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            
            # Set axis limits and ticks (0.4 to 1.1 to add space before 0.5 and after 1)
            ax.set_xlim(0.4, 1.1)
            ax.set_ylim(0.4, 1.1)
            ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            
            # Add grid
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
            # Add legend
            ax.legend(loc='upper left', fontsize=10)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Best models comparison plot saved to: {save_path}")

def print_comparison_summary(df):
    """
    Print summary statistics for the comparison data.
    
    Args:
        df (pd.DataFrame): Feature frequency comparison data
    """
    print("\n" + "="*80)
    print("BEST MODELS COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\nTotal feature comparisons: {len(df)}")
    print(f"Unique features: {df['feature_name'].nunique()}")
    
    print("\nBy Task:")
    for task in df['task'].unique():
        task_data = df[df['task'] == task]
        print(f"  {task.capitalize()}: {len(task_data)} comparisons")
        print(f"    Mean heel frequency: {task_data['heel_frequency'].mean():.3f}")
        print(f"    Mean cord frequency: {task_data['cord_frequency'].mean():.3f}")
        print(f"    Correlation: {task_data['heel_frequency'].corr(task_data['cord_frequency']):.3f}")
    
    print("\nBy Pipeline:")
    for pipeline in df['pipeline'].unique():
        pipeline_data = df[df['pipeline'] == pipeline]
        print(f"  {pipeline.replace('_', ' ').title()}: {len(pipeline_data)} comparisons")
        print(f"    Mean heel frequency: {pipeline_data['heel_frequency'].mean():.3f}")
        print(f"    Mean cord frequency: {pipeline_data['cord_frequency'].mean():.3f}")
    
    print("\nBy Model:")
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        print(f"  {model.title()}: {len(model_data)} comparisons")
        print(f"    Mean heel frequency: {model_data['heel_frequency'].mean():.3f}")
        print(f"    Mean cord frequency: {model_data['cord_frequency'].mean():.3f}")
    
    print("\nTop 10 Features by Average Frequency:")
    df['avg_frequency'] = (df['heel_frequency'] + df['cord_frequency']) / 2
    top_features = df.nlargest(10, 'avg_frequency')
    for _, row in top_features.iterrows():
        print(f"  {row['feature_name']}: {row['avg_frequency']:.3f} "
              f"({row['task']}, {row['pipeline']}, {row['model']})")

def main():
    """
    Main function to run the best models comparison analysis.
    """
    print("🔍 Best Models Feature Selection Comparison Analysis")
    print("="*60)
    
    # Load results
    all_results = load_all_results()
    
    # Identify best models
    best_models = identify_best_models(all_results)
    
    # Extract feature frequencies
    comparison_df = extract_feature_frequencies(best_models, all_results)
    
    if comparison_df is not None:
        # Print summary
        print_comparison_summary(comparison_df)
        
        # Create plot
        create_best_models_comparison_plot(comparison_df)
        
        print("\n" + "="*60)
        print("✅ BEST MODELS COMPARISON ANALYSIS COMPLETE!")
        print("="*60)
        print("📁 Check outputs/plots/best_models_heel_vs_cord.png for the generated plot.")
    else:
        print("❌ No comparison data available. Analysis failed.")

if __name__ == "__main__":
    main()
