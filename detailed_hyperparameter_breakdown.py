#!/usr/bin/env python3
"""
Detailed hyperparameter breakdown by dataset type for gestational age pipeline.
"""

import re
from collections import defaultdict, Counter

def extract_detailed_hyperparameters():
    """Extract hyperparameters with proper dataset identification."""
    
    print("=== DETAILED HYPERPARAMETER BREAKDOWN ===\n")
    print("Gestational Age Pipeline - By Dataset Type\n")
    
    # Read the log file
    try:
        with open('gestational_age_output.log', 'r') as f:
            log_content = f.read()
    except FileNotFoundError:
        print("❌ gestational_age_output.log not found!")
        return
    
    # Extract model sections with context
    model_sections = re.findall(r'MODEL: (\w+) \((\w+)\) - (\w+) ON (\w+)', log_content)
    
    # Extract hyperparameters with context
    regression_pattern = r'Optimized hyperparameters - Alpha: ([\d.]+), L1_ratio: ([^,\n]+)'
    classification_pattern = r'Classification optimized hyperparameters - C: ([\d.]+), L1_ratio: ([^,\n]+)'
    
    regression_matches = re.findall(regression_pattern, log_content)
    classification_matches = re.findall(classification_pattern, log_content)
    
    # Group results by dataset and model type
    results = {
        'Clinical': {'Lasso': [], 'ElasticNet': []},
        'Biomarker': {'Lasso': [], 'ElasticNet': []},
        'Combined': {'Lasso': [], 'ElasticNet': []}
    }
    
    class_results = {
        'Clinical': {'Lasso': [], 'ElasticNet': []},
        'Biomarker': {'Lasso': [], 'ElasticNet': []},
        'Combined': {'Lasso': [], 'ElasticNet': []}
    }
    
    # Process regression results
    for i, (alpha, l1_ratio) in enumerate(regression_matches):
        alpha_val = float(alpha)
        
        # Determine model type
        if l1_ratio.strip() == 'None':
            model_type = 'Lasso'
            l1_ratio_val = 1.0
        else:
            model_type = 'ElasticNet'
            l1_ratio_val = float(l1_ratio)
        
        # Determine dataset type (approximate based on position)
        dataset_type = "Biomarker"  # Default
        if i < len(model_sections):
            dataset_type = model_sections[i][1]  # Clinical, Biomarker, or Combined
        
        results[dataset_type][model_type].append({
            'alpha': alpha_val,
            'l1_ratio': l1_ratio_val
        })
    
    # Process classification results
    for i, (c_val, l1_ratio) in enumerate(classification_matches):
        c_val = float(c_val)
        
        # Determine model type
        if l1_ratio.strip() == 'None':
            model_type = 'Lasso'
            l1_ratio_val = 1.0
        else:
            model_type = 'ElasticNet'
            l1_ratio_val = float(l1_ratio)
        
        # Determine dataset type
        dataset_type = "Biomarker"  # Default
        if i < len(model_sections):
            dataset_type = model_sections[i][1]
        
        class_results[dataset_type][model_type].append({
            'C': c_val,
            'l1_ratio': l1_ratio_val
        })
    
    # Display results by dataset
    for dataset in ['Clinical', 'Biomarker', 'Combined']:
        print(f"🔹 {dataset.upper()} DATASET\n")
        print("-" * 60)
        
        # Regression results
        print("📊 REGRESSION:")
        for model_type in ['Lasso', 'ElasticNet']:
            runs = results[dataset][model_type]
            if not runs:
                continue
                
            alphas = [r['alpha'] for r in runs]
            l1_ratios = [r['l1_ratio'] for r in runs]
            
            print(f"  {model_type} CV:")
            print(f"    Runs: {len(runs)}")
            print(f"    Alpha range: {min(alphas):.4f} to {max(alphas):.4f}")
            print(f"    Alpha mean: {sum(alphas)/len(alphas):.4f}")
            
            # Most common alpha
            alpha_counts = Counter(alphas)
            most_common_alpha = alpha_counts.most_common(1)[0]
            print(f"    Most common alpha: {most_common_alpha[0]:.4f} ({most_common_alpha[1]} times)")
            
            if model_type == 'ElasticNet':
                print(f"    L1_ratio range: {min(l1_ratios):.1f} to {max(l1_ratios):.1f}")
                print(f"    L1_ratio mean: {sum(l1_ratios)/len(l1_ratios):.2f}")
                
                l1_counts = Counter(l1_ratios)
                most_common_l1 = l1_counts.most_common(1)[0]
                print(f"    Most common L1_ratio: {most_common_l1[0]:.1f} ({most_common_l1[1]} times)")
            else:
                print(f"    L1_ratio: Fixed at 1.0")
            print()
        
        # Classification results
        print("📊 CLASSIFICATION:")
        for model_type in ['Lasso', 'ElasticNet']:
            runs = class_results[dataset][model_type]
            if not runs:
                continue
                
            c_vals = [r['C'] for r in runs]
            l1_ratios = [r['l1_ratio'] for r in runs]
            
            print(f"  {model_type} CV:")
            print(f"    Runs: {len(runs)}")
            print(f"    C range: {min(c_vals):.1f} to {max(c_vals):.1f}")
            print(f"    C mean: {sum(c_vals)/len(c_vals):.2f}")
            
            # Most common C
            c_counts = Counter(c_vals)
            most_common_c = c_counts.most_common(1)[0]
            print(f"    Most common C: {most_common_c[0]:.1f} ({most_common_c[1]} times)")
            
            if model_type == 'ElasticNet':
                print(f"    L1_ratio range: {min(l1_ratios):.1f} to {max(l1_ratios):.1f}")
                print(f"    L1_ratio mean: {sum(l1_ratios)/len(l1_ratios):.2f}")
                
                l1_counts = Counter(l1_ratios)
                most_common_l1 = l1_counts.most_common(1)[0]
                print(f"    Most common L1_ratio: {most_common_l1[0]:.1f} ({most_common_l1[1]} times)")
            else:
                print(f"    L1_ratio: Fixed at 1.0")
            print()
        
        print("=" * 80)
        print()
    
    # Overall summary
    print("🎯 OVERALL SUMMARY\n")
    print("=" * 80)
    
    total_regression = sum(len(runs) for dataset in results.values() for runs in dataset.values())
    total_classification = sum(len(runs) for dataset in class_results.values() for runs in dataset.values())
    
    print(f"Total regression runs: {total_regression}")
    print(f"Total classification runs: {total_classification}")
    
    # Collect all values for overall statistics
    all_alphas = []
    all_c_vals = []
    all_l1_ratios = []
    
    for dataset in results.values():
        for model_type, runs in dataset.items():
            all_alphas.extend([r['alpha'] for r in runs])
            if model_type == 'ElasticNet':
                all_l1_ratios.extend([r['l1_ratio'] for r in runs])
    
    for dataset in class_results.values():
        for model_type, runs in dataset.items():
            all_c_vals.extend([r['C'] for r in runs])
            if model_type == 'ElasticNet':
                all_l1_ratios.extend([r['l1_ratio'] for r in runs])
    
    if all_alphas:
        print(f"\nOverall Alpha statistics:")
        print(f"  Range: {min(all_alphas):.4f} to {max(all_alphas):.4f}")
        print(f"  Mean: {sum(all_alphas)/len(all_alphas):.4f}")
        print(f"  Most common: {Counter(all_alphas).most_common(1)[0]}")
    
    if all_c_vals:
        print(f"\nOverall C statistics:")
        print(f"  Range: {min(all_c_vals):.1f} to {max(all_c_vals):.1f}")
        print(f"  Mean: {sum(all_c_vals)/len(all_c_vals):.2f}")
        print(f"  Most common: {Counter(all_c_vals).most_common(1)[0]}")
    
    if all_l1_ratios:
        print(f"\nOverall L1_ratio statistics (ElasticNet only):")
        print(f"  Range: {min(all_l1_ratios):.1f} to {max(all_l1_ratios):.1f}")
        print(f"  Mean: {sum(all_l1_ratios)/len(all_l1_ratios):.2f}")
        print(f"  Most common: {Counter(all_l1_ratios).most_common(1)[0]}")

if __name__ == "__main__":
    extract_detailed_hyperparameters() 