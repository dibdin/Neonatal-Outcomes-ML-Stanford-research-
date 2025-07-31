#!/usr/bin/env python3
"""
Comprehensive hyperparameter summary for gestational age pipeline.
"""

import re
from collections import defaultdict

def extract_hyperparameters():
    """Extract all hyperparameters from the gestational age log."""
    
    print("=== COMPREHENSIVE HYPERPARAMETER SUMMARY ===\n")
    print("Gestational Age Pipeline - All Models and Datasets\n")
    
    # Read the log file
    try:
        with open('gestational_age_output.log', 'r') as f:
            log_content = f.read()
    except FileNotFoundError:
        print("❌ gestational_age_output.log not found!")
        return
    
    # Extract all hyperparameter lines
    regression_pattern = r'Optimized hyperparameters - Alpha: ([\d.]+), L1_ratio: ([^,\n]+)'
    classification_pattern = r'Classification optimized hyperparameters - C: ([\d.]+), L1_ratio: ([^,\n]+)'
    
    regression_matches = re.findall(regression_pattern, log_content)
    classification_matches = re.findall(classification_pattern, log_content)
    
    # Group by model type and dataset
    results = defaultdict(list)
    
    # Find model sections and extract context
    model_sections = re.findall(r'MODEL: (\w+) \((\w+)\) - (\w+) ON (\w+)', log_content)
    
    print("📊 REGRESSION HYPERPARAMETERS\n")
    print("=" * 80)
    
    # Process regression results
    for i, (alpha, l1_ratio) in enumerate(regression_matches):
        # Determine if it's Lasso or ElasticNet based on l1_ratio
        if l1_ratio.strip() == 'None':
            model_type = 'Lasso CV'
            l1_ratio_val = 1.0  # Lasso has l1_ratio = 1.0
        else:
            model_type = 'ElasticNet CV'
            l1_ratio_val = float(l1_ratio)
        
        alpha_val = float(alpha)
        
        # Try to determine dataset from context (this is approximate)
        if i < len(model_sections):
            data_type = model_sections[i][1]  # Clinical, Biomarker, or Combined
        else:
            data_type = "Unknown"
        
        key = f"{model_type}_{data_type}"
        results[key].append({
            'alpha': alpha_val,
            'l1_ratio': l1_ratio_val,
            'model_type': model_type,
            'data_type': data_type
        })
    
    # Display regression results
    for key, runs in results.items():
        if not runs:
            continue
            
        model_type = runs[0]['model_type']
        data_type = runs[0]['data_type']
        
        alphas = [r['alpha'] for r in runs]
        l1_ratios = [r['l1_ratio'] for r in runs]
        
        print(f"🔹 {model_type} - {data_type.upper()} DATA")
        print(f"   Total runs: {len(runs)}")
        print(f"   Alpha range: {min(alphas):.4f} to {max(alphas):.4f}")
        print(f"   Alpha mean: {sum(alphas)/len(alphas):.4f}")
        
        if model_type == 'ElasticNet CV':
            print(f"   L1_ratio range: {min(l1_ratios):.1f} to {max(l1_ratios):.1f}")
            print(f"   L1_ratio mean: {sum(l1_ratios)/len(l1_ratios):.2f}")
        else:
            print(f"   L1_ratio: Fixed at 1.0 (Lasso)")
        
        # Most common values
        from collections import Counter
        alpha_counts = Counter(alphas)
        most_common_alpha = alpha_counts.most_common(1)[0]
        print(f"   Most common alpha: {most_common_alpha[0]:.4f} ({most_common_alpha[1]} times)")
        
        if model_type == 'ElasticNet CV':
            l1_counts = Counter(l1_ratios)
            most_common_l1 = l1_counts.most_common(1)[0]
            print(f"   Most common L1_ratio: {most_common_l1[0]:.1f} ({most_common_l1[1]} times)")
        
        print()
    
    print("\n📊 CLASSIFICATION HYPERPARAMETERS\n")
    print("=" * 80)
    
    # Process classification results
    class_results = defaultdict(list)
    
    for i, (c_val, l1_ratio) in enumerate(classification_matches):
        # Determine if it's Lasso or ElasticNet based on l1_ratio
        if l1_ratio.strip() == 'None':
            model_type = 'Lasso CV'
            l1_ratio_val = 1.0  # Lasso has l1_ratio = 1.0
        else:
            model_type = 'ElasticNet CV'
            l1_ratio_val = float(l1_ratio)
        
        c_val = float(c_val)
        
        # Try to determine dataset from context
        if i < len(model_sections):
            data_type = model_sections[i][1]
        else:
            data_type = "Unknown"
        
        key = f"{model_type}_{data_type}_Classification"
        class_results[key].append({
            'C': c_val,
            'l1_ratio': l1_ratio_val,
            'model_type': model_type,
            'data_type': data_type
        })
    
    # Display classification results
    for key, runs in class_results.items():
        if not runs:
            continue
            
        model_type = runs[0]['model_type']
        data_type = runs[0]['data_type']
        
        c_vals = [r['C'] for r in runs]
        l1_ratios = [r['l1_ratio'] for r in runs]
        
        print(f"🔹 {model_type} - {data_type.upper()} DATA (Classification)")
        print(f"   Total runs: {len(runs)}")
        print(f"   C range: {min(c_vals):.1f} to {max(c_vals):.1f}")
        print(f"   C mean: {sum(c_vals)/len(c_vals):.2f}")
        
        if model_type == 'ElasticNet CV':
            print(f"   L1_ratio range: {min(l1_ratios):.1f} to {max(l1_ratios):.1f}")
            print(f"   L1_ratio mean: {sum(l1_ratios)/len(l1_ratios):.2f}")
        else:
            print(f"   L1_ratio: Fixed at 1.0 (Lasso)")
        
        # Most common values
        from collections import Counter
        c_counts = Counter(c_vals)
        most_common_c = c_counts.most_common(1)[0]
        print(f"   Most common C: {most_common_c[0]:.1f} ({most_common_c[1]} times)")
        
        if model_type == 'ElasticNet CV':
            l1_counts = Counter(l1_ratios)
            most_common_l1 = l1_counts.most_common(1)[0]
            print(f"   Most common L1_ratio: {most_common_l1[0]:.1f} ({most_common_l1[1]} times)")
        
        print()
    
    print("\n🎯 SUMMARY INSIGHTS\n")
    print("=" * 80)
    
    # Overall statistics
    total_regression = sum(len(runs) for runs in results.values())
    total_classification = sum(len(runs) for runs in class_results.values())
    
    print(f"Total regression runs analyzed: {total_regression}")
    print(f"Total classification runs analyzed: {total_classification}")
    
    # Find most common hyperparameters across all models
    all_alphas = []
    all_c_vals = []
    all_l1_ratios = []
    
    for runs in results.values():
        all_alphas.extend([r['alpha'] for r in runs])
        if runs[0]['model_type'] == 'ElasticNet CV':
            all_l1_ratios.extend([r['l1_ratio'] for r in runs])
    
    for runs in class_results.values():
        all_c_vals.extend([r['C'] for r in runs])
        if runs[0]['model_type'] == 'ElasticNet CV':
            all_l1_ratios.extend([r['l1_ratio'] for r in runs])
    
    if all_alphas:
        print(f"\nOverall Alpha statistics:")
        print(f"  Range: {min(all_alphas):.4f} to {max(all_alphas):.4f}")
        print(f"  Mean: {sum(all_alphas)/len(all_alphas):.4f}")
    
    if all_c_vals:
        print(f"\nOverall C statistics:")
        print(f"  Range: {min(all_c_vals):.1f} to {max(all_c_vals):.1f}")
        print(f"  Mean: {sum(all_c_vals)/len(all_c_vals):.2f}")
    
    if all_l1_ratios:
        print(f"\nOverall L1_ratio statistics (ElasticNet only):")
        print(f"  Range: {min(all_l1_ratios):.1f} to {max(all_l1_ratios):.1f}")
        print(f"  Mean: {sum(all_l1_ratios)/len(all_l1_ratios):.2f}")

if __name__ == "__main__":
    extract_hyperparameters() 