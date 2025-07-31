#!/usr/bin/env python3
"""
Extract all hyperparameters directly from the training log
"""

import re
from collections import Counter, defaultdict

def extract_all_hyperparameters():
    """Extract all hyperparameters from the log file"""
    
    print("=== ALL HYPERPARAMETERS FROM TRAINING LOG ===\n")
    
    with open('gestational_age_output.log', 'r') as f:
        log_content = f.read()
    
    # Extract all regression hyperparameters
    regression_matches = re.findall(r'Optimized hyperparameters - Alpha: ([\d.]+), L1_ratio: ([^,\n]+)', log_content)
    
    # Extract all classification hyperparameters
    classification_matches = re.findall(r'Classification optimized hyperparameters - C: ([\d.]+), L1_ratio: ([^,\n]+)', log_content)
    
    print(f"Total regression entries: {len(regression_matches)}")
    print(f"Total classification entries: {len(classification_matches)}")
    
    # Analyze regression hyperparameters
    print("\n" + "="*60)
    print("📊 REGRESSION HYPERPARAMETERS")
    print("="*60)
    
    regression_alphas = []
    regression_l1_ratios = []
    
    for alpha_str, l1_ratio_str in regression_matches:
        alpha = float(alpha_str)
        l1_ratio = l1_ratio_str.strip()
        
        regression_alphas.append(alpha)
        if l1_ratio != 'None':
            regression_l1_ratios.append(float(l1_ratio))
    
    print(f"Alpha values found: {len(regression_alphas)}")
    print(f"Alpha range: {min(regression_alphas):.4f} to {max(regression_alphas):.4f}")
    print(f"Alpha mean: {sum(regression_alphas)/len(regression_alphas):.4f}")
    
    alpha_counts = Counter(regression_alphas)
    print(f"Most common alpha values:")
    for alpha, count in alpha_counts.most_common(10):
        print(f"  {alpha:.4f}: {count} times")
    
    print(f"\nL1_ratio values found: {len(regression_l1_ratios)}")
    if regression_l1_ratios:
        print(f"L1_ratio range: {min(regression_l1_ratios):.1f} to {max(regression_l1_ratios):.1f}")
        print(f"L1_ratio mean: {sum(regression_l1_ratios)/len(regression_l1_ratios):.2f}")
        
        l1_counts = Counter(regression_l1_ratios)
        print(f"Most common L1_ratio values:")
        for l1_ratio, count in l1_counts.most_common(10):
            print(f"  {l1_ratio:.1f}: {count} times")
    else:
        print("All L1_ratios were 'None' (Lasso models)")
    
    # Analyze classification hyperparameters
    print("\n" + "="*60)
    print("📊 CLASSIFICATION HYPERPARAMETERS")
    print("="*60)
    
    classification_c_vals = []
    classification_l1_ratios = []
    
    for c_str, l1_ratio_str in classification_matches:
        c_val = float(c_str)
        l1_ratio = l1_ratio_str.strip()
        
        classification_c_vals.append(c_val)
        if l1_ratio != 'None':
            classification_l1_ratios.append(float(l1_ratio))
    
    print(f"C values found: {len(classification_c_vals)}")
    print(f"C range: {min(classification_c_vals):.1f} to {max(classification_c_vals):.1f}")
    print(f"C mean: {sum(classification_c_vals)/len(classification_c_vals):.2f}")
    
    c_counts = Counter(classification_c_vals)
    print(f"Most common C values:")
    for c_val, count in c_counts.most_common(10):
        print(f"  {c_val:.1f}: {count} times")
    
    print(f"\nL1_ratio values found: {len(classification_l1_ratios)}")
    if classification_l1_ratios:
        print(f"L1_ratio range: {min(classification_l1_ratios):.1f} to {max(classification_l1_ratios):.1f}")
        print(f"L1_ratio mean: {sum(classification_l1_ratios)/len(classification_l1_ratios):.2f}")
        
        l1_counts = Counter(classification_l1_ratios)
        print(f"Most common L1_ratio values:")
        for l1_ratio, count in l1_counts.most_common(10):
            print(f"  {l1_ratio:.1f}: {count} times")
    else:
        print("All L1_ratios were 'None' (Lasso models)")
    
    # Check for any 0.0 values
    print("\n" + "="*60)
    print("🔍 CHECKING FOR C=0.0 OR ALPHA=0.0")
    print("="*60)
    
    zero_c_count = sum(1 for c in classification_c_vals if c == 0.0)
    zero_alpha_count = sum(1 for a in regression_alphas if a == 0.0)
    
    print(f"C=0.0 count: {zero_c_count}")
    print(f"Alpha=0.0 count: {zero_alpha_count}")
    
    if zero_c_count == 0:
        print("✅ No C=0.0 values found in classification")
    else:
        print(f"⚠️  Found {zero_c_count} C=0.0 values in classification")
    
    if zero_alpha_count == 0:
        print("✅ No Alpha=0.0 values found in regression")
    else:
        print(f"⚠️  Found {zero_alpha_count} Alpha=0.0 values in regression")
    
    # Show all unique values
    print("\n" + "="*60)
    print("📋 ALL UNIQUE VALUES")
    print("="*60)
    
    print(f"Unique Alpha values: {sorted(set(regression_alphas))}")
    print(f"Unique C values: {sorted(set(classification_c_vals))}")
    if regression_l1_ratios:
        print(f"Unique L1_ratio values (regression): {sorted(set(regression_l1_ratios))}")
    if classification_l1_ratios:
        print(f"Unique L1_ratio values (classification): {sorted(set(classification_l1_ratios))}")

if __name__ == "__main__":
    extract_all_hyperparameters() 