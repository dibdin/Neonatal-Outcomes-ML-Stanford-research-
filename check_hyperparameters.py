#!/usr/bin/env python3
"""
Script to check the optimized hyperparameters from the gestational age pipeline results.
"""

import pickle
import glob
import os
from collections import defaultdict

def check_hyperparameters():
    """Extract and display optimized hyperparameters from model outputs."""
    
    print("=== Checking Optimized Hyperparameters ===\n")
    
    # Find all model output files
    model_files = glob.glob("outputs/models/*_cv_*_run*_model_outputs.pkl")
    
    if not model_files:
        print("No CV model output files found!")
        return
    
    print(f"Found {len(model_files)} CV model output files\n")
    
    # Group by model type and data type
    results = defaultdict(list)
    
    for file_path in model_files:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extract filename components
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            
            # Parse model type and data type
            if 'elasticnet_cv' in filename:
                model_type = 'elasticnet_cv'
            elif 'lasso_cv' in filename:
                model_type = 'lasso_cv'
            else:
                continue
                
            # Find data type (clinical, biomarker, combined)
            data_type = None
            for part in parts:
                if part in ['clinical', 'biomarker', 'combined']:
                    data_type = part
                    break
            
            # Find run number
            run_num = None
            for part in parts:
                if part.startswith('run') and part[3:].isdigit():
                    run_num = int(part[3:])
                    break
            
            if data_type and run_num:
                key = f"{model_type}_{data_type}"
                
                # Extract hyperparameters if available
                alpha = data.get('optimized_alpha')
                l1_ratio = data.get('optimized_l1_ratio')
                
                results[key].append({
                    'run': run_num,
                    'alpha': alpha,
                    'l1_ratio': l1_ratio,
                    'file': filename
                })
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Display results
    for model_key, runs in results.items():
        print(f"--- {model_key.upper()} ---")
        
        # Sort by run number
        runs.sort(key=lambda x: x['run'])
        
        for run in runs:
            alpha_str = f"{run['alpha']:.4f}" if run['alpha'] is not None else "N/A"
            l1_str = f"{run['l1_ratio']:.4f}" if run['l1_ratio'] is not None else "N/A"
            
            print(f"  Run {run['run']}: Alpha = {alpha_str}, L1_ratio = {l1_str}")
        
        # Calculate statistics
        alphas = [r['alpha'] for r in runs if r['alpha'] is not None]
        l1_ratios = [r['l1_ratio'] for r in runs if r['l1_ratio'] is not None]
        
        if alphas:
            print(f"  Alpha stats: min={min(alphas):.4f}, max={max(alphas):.4f}, mean={sum(alphas)/len(alphas):.4f}")
        if l1_ratios:
            print(f"  L1_ratio stats: min={min(l1_ratios):.4f}, max={max(l1_ratios):.4f}, mean={sum(l1_ratios)/len(l1_ratios):.4f}")
        
        print()

if __name__ == "__main__":
    check_hyperparameters() 