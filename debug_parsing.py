#!/usr/bin/env python3
"""
Debug script to see what's happening with model type categorization
"""

import re
from collections import defaultdict

def debug_parsing():
    """Debug the parsing logic"""
    
    with open('gestational_age_output.log', 'r') as f:
        log_content = f.read()
    
    # Find all model section positions in the log
    model_positions = []
    for match in re.finditer(r'MODEL: (\w+) \(([^)]+)\) - (\w+) ON (\w+)', log_content):
        model_positions.append({
            'position': match.start(),
            'model_name': match.group(1),
            'data_type': match.group(2),
            'model_type': match.group(3),
            'dataset': match.group(4)
        })
    
    # Find all classification hyperparameter positions
    classification_positions = []
    for match in re.finditer(r'Classification optimized hyperparameters - C: ([\d.]+), L1_ratio: ([^,\n]+)', log_content):
        classification_positions.append({
            'position': match.start(),
            'C': float(match.group(1)),
            'l1_ratio': match.group(2).strip()
        })
    
    # Match hyperparameters to their nearest model section
    def find_nearest_model_section(hyperparam_pos, model_positions):
        """Find the model section that comes before this hyperparameter."""
        nearest_model = None
        for model in model_positions:
            if model['position'] < hyperparam_pos:
                if nearest_model is None or model['position'] > nearest_model['position']:
                    nearest_model = model
        return nearest_model
    
    # Group by dataset and model type
    results = defaultdict(lambda: defaultdict(list))
    
    for class_pos in classification_positions:
        model_section = find_nearest_model_section(class_pos['position'], model_positions)
        
        if model_section:
            data_type = model_section['data_type']
            # Map the data type to the correct key
            if data_type == 'CLINICAL DATA':
                data_type = 'Clinical'
            elif data_type == 'BIOMARKER DATA':
                data_type = 'Biomarker'
            elif data_type == 'COMBINED DATA':
                data_type = 'Combined'
            
            # Determine model type based on L1_ratio
            if class_pos['l1_ratio'] == 'None':
                model_type = 'Lasso'
                l1_ratio_val = 1.0
            else:
                model_type = 'ElasticNet'
                l1_ratio_val = float(class_pos['l1_ratio'])
            
            results[data_type][model_type].append({
                'C': class_pos['C'],
                'l1_ratio': l1_ratio_val
            })
    
    # Print debug info
    print("=== DEBUG PARSING ===")
    for dataset in ['Clinical', 'Biomarker', 'Combined']:
        print(f"\n{dataset} DATASET:")
        for model_type in ['Lasso', 'ElasticNet']:
            runs = results[dataset][model_type]
            if runs:
                c_vals = [r['C'] for r in runs]
                print(f"  {model_type}: {len(runs)} runs")
                print(f"    C values: {c_vals[:10]}...")  # Show first 10
                print(f"    C range: {min(c_vals)} to {max(c_vals)}")
                print(f"    C=0.0 count: {sum(1 for c in c_vals if c == 0.0)}")
                if model_type == 'ElasticNet':
                    l1_vals = [r['l1_ratio'] for r in runs]
                    print(f"    L1_ratio values: {l1_vals[:10]}...")
            else:
                print(f"  {model_type}: No runs")

if __name__ == "__main__":
    debug_parsing() 