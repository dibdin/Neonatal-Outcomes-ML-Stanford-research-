#!/usr/bin/env python3
"""
Fast test version of the pipeline with reduced scope
"""

import pandas as pd
import numpy as np
from main import run_single_model, generate_model_outputs, print_detailed_summary
from src.config import TEST_SIZE, PRETERM_CUTOFF
import warnings
warnings.filterwarnings('ignore')

def test_fast_pipeline():
    """Test pipeline with reduced scope for faster execution"""
    
    print("=== FAST PIPELINE TEST (REDUCED SCOPE) ===\n")
    
    # Reduced configuration for testing
    dataset_types = ['cord']  # Only test cord data
    model_types = ['elasticnet_cv']  # Only test ElasticNet
    model_configs = [
        {'name': 'Biomarker', 'data_type': 'biomarker', 'allowed_models': ['elasticnet_cv']}
    ]
    
    print(f"Testing with reduced scope:")
    print(f"  Dataset types: {dataset_types}")
    print(f"  Model types: {model_types}")
    print(f"  Model configs: {[config['name'] for config in model_configs]}")
    print(f"  Data options: [1] (both_samples only)")
    
    summary_rows = []
    all_results = {}
    
    # Test only data option 1
    for data_option in [1]:
        data_option_label = 'both_samples'
        print(f"\n{'#'*80}")
        print(f"TESTING DATA OPTION {data_option}: {data_option_label}")
        print(f"{'#'*80}")
        
        for dataset_type in dataset_types:
            print(f"\n{'='*80}")
            print(f"TESTING ON {dataset_type.upper()} DATASET [{data_option_label}]")
            print(f"{'='*80}")
            
            for model_type in model_types:
                print(f"\n{'='*80}")
                print(f"TESTING WITH {model_type.upper()} MODEL ON {dataset_type.upper()} DATA [{data_option_label}]")
                print(f"{'='*80}")
                
                model_results = {}
                for config in model_configs:
                    model_name = config['name']
                    data_type = config['data_type']
                    allowed_models = config['allowed_models']
                    
                    if model_type not in allowed_models:
                        print(f"Skipping {model_name} model with {model_type} (not allowed)")
                        continue
                    
                    print(f"\n🔄 Running {model_name} model...")
                    results = run_single_model(model_name, data_type, dataset_type, model_type, data_option, data_option_label, 'gestational_age')
                    model_results[model_name] = results
                    all_results[f"{data_option_label}_{dataset_type}_{model_type}_{model_name}"] = results
                    
                    # Generate outputs
                    generate_model_outputs(model_name, data_type, dataset_type, model_type, results, None, None, results['summary']['auc_mean'], data_option_label, 'gestational_age')
                    print_detailed_summary(model_name, dataset_type, model_type, results)
                    
                    # Collect summary
                    mae_mean = results['summary']['mae_mean']
                    rmse_mean = results['summary']['rmse_mean']
                    auc_mean = results['summary']['auc_mean']
                    
                    summary_rows.append({
                        'Dataset': dataset_type,
                        'Model': model_name,
                        'ModelType': model_type,
                        'DataOption': data_option_label,
                        'MAE': mae_mean,
                        'RMSE': rmse_mean,
                        'AUC': auc_mean
                    })
                
                # Print results
                if model_results:
                    print(f"\n{'='*80}")
                    print(f"FAST TEST COMPLETED FOR {model_type.upper()} ON {dataset_type.upper()}!")
                    print("="*80)
                    for model_name, results in model_results.items():
                        print(f"  {model_name} Results:")
                        print(f"    MAE: {results['summary']['mae_mean']:.4f}")
                        print(f"    RMSE: {results['summary']['rmse_mean']:.4f}")
                        print(f"    AUC: {results['summary']['auc_mean']:.4f}")
                    print("="*80)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FAST PIPELINE TEST SUMMARY")
    print("="*80)
    for row in summary_rows:
        print(f"  {row['Dataset']} | {row['Model']} | {row['ModelType']}:")
        print(f"    MAE: {row['MAE']:.4f}, RMSE: {row['RMSE']:.4f}, AUC: {row['AUC']:.4f}")
    print("="*80)
    
    return summary_rows

if __name__ == "__main__":
    results = test_fast_pipeline() 