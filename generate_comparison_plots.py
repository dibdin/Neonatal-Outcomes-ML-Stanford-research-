#!/usr/bin/env python3
"""
Generate comparison plots for both gestational age and birth weight analyses.

This script generates the heel vs cord biomarker frequency comparison plots
for both target types, data options, and model types.
"""

import pickle
import os
from src.utils import plot_biomarker_frequency_heel_vs_cord

def generate_comparison_plots(target_type, all_results):
    """Generate comparison plots for a specific target type."""
    
    print(f"\nGenerating comparison plots for {target_type}...")
    
    # Filter results for this target type
    target_results = {}
    for key, value in all_results.items():
        if target_type in key:
            target_results[key] = value
    
    if not target_results:
        print(f"  No results found for {target_type}. Skipping.")
        return
    
    # Generate comparison plots for each model type
    for model_type in ['lasso', 'elasticnet']:
        print(f"  Processing {model_type}...")
        
        # Option 1: Heel vs Cord comparison using both_samples data
        option1_results = {k: v for k, v in target_results.items() if 'both_samples' in k}
        if option1_results:
            plot_biomarker_frequency_heel_vs_cord(
                option1_results, 
                model_type, 
                filename=f"outputs/plots/biomarker_frequency_heel_vs_cord_{model_type}_{target_type}_option1_both_samples.png",
                target_type=target_type
            )
            print(f"    ✓ Option 1 (both_samples) plot generated")
        
        # Options 2+3: Heel vs Cord comparison using heel_all and cord_all data
        option2_3_results = {k: v for k, v in target_results.items() if 'heel_all' in k or 'cord_all' in k}
        if option2_3_results:
            plot_biomarker_frequency_heel_vs_cord(
                option2_3_results, 
                model_type, 
                filename=f"outputs/plots/biomarker_frequency_heel_vs_cord_{model_type}_{target_type}_options2_3_heel_cord_all.png",
                target_type=target_type
            )
            print(f"    ✓ Options 2+3 (heel_all + cord_all) plot generated")

def main():
    """Generate comparison plots for both target types."""
    
    print("Generating Heel vs Cord Comparison Plots")
    print("=" * 50)
    
    # Load the combined results file
    results_file = "all_results.pkl"
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found.")
        return
    
    with open(results_file, 'rb') as f:
        all_results = pickle.load(f)
    
    print(f"Loaded {len(all_results)} result entries")
    
    # Generate plots for both target types
    for target_type in ['gestational_age', 'birth_weight']:
        generate_comparison_plots(target_type, all_results)
    
    print("\n" + "=" * 50)
    print("COMPARISON PLOTS GENERATED!")
    print("=" * 50)
    print("Generated files:")
    print("  Gestational Age:")
    print("    • biomarker_frequency_heel_vs_cord_lasso_gestational_age_option1_both_samples.png")
    print("    • biomarker_frequency_heel_vs_cord_lasso_gestational_age_options2_3_heel_cord_all.png")
    print("    • biomarker_frequency_heel_vs_cord_elasticnet_gestational_age_option1_both_samples.png")
    print("    • biomarker_frequency_heel_vs_cord_elasticnet_gestational_age_options2_3_heel_cord_all.png")
    print("  Birth Weight:")
    print("    • biomarker_frequency_heel_vs_cord_lasso_birth_weight_option1_both_samples.png")
    print("    • biomarker_frequency_heel_vs_cord_lasso_birth_weight_options2_3_heel_cord_all.png")
    print("    • biomarker_frequency_heel_vs_cord_elasticnet_birth_weight_option1_both_samples.png")
    print("    • biomarker_frequency_heel_vs_cord_elasticnet_birth_weight_options2_3_heel_cord_all.png")
    print("\nNow run: python3 organize_plots.py")

if __name__ == "__main__":
    main() 