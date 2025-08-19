#!/usr/bin/env python3
"""
Script to regenerate biomarker frequency plots using existing results.
This script reads the saved results and recreates all biomarker frequency plots
with the updated top 50% labeling without rerunning the entire pipeline.
"""

import pickle
import os
import sys
from src.utils import plot_biomarker_frequency_heel_vs_cord

def load_results(target_type):
    """Load results from pickle file."""
    filename = f"all_results_{target_type}.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            results = pickle.load(f)
        print(f"✅ Loaded {len(results)} results from {filename}")
        return results
    else:
        print(f"❌ {filename} not found")
        return None

def regenerate_biomarker_plots(target_type):
    """Regenerate all biomarker frequency plots for a given target type."""
    
    print(f"🔄 Regenerating biomarker frequency plots for {target_type}")
    print("=" * 60)
    
    # Load existing results
    all_results = load_results(target_type)
    if not all_results:
        return
    
    # Create output directory
    os.makedirs("outputs/plots", exist_ok=True)
    
    # Define model types to process
    model_types = ['lasso_cv', 'elasticnet_cv', 'stabl']
    
    # Track generated plots
    generated_plots = []
    
    # Generate plots for each model type
    for model_type in model_types:
        print(f"\n🎯 Processing {model_type}...")
        
        # Option 1: Heel vs Cord comparison using both_samples data
        option1_results = {k: v for k, v in all_results.items() if 'both_samples' in k and model_type in k}
        if option1_results:
            filename = f"outputs/plots/biomarker_frequency_heel_vs_cord_{model_type}_{target_type}_option1_both_samples.png"
            try:
                plot_biomarker_frequency_heel_vs_cord(
                    option1_results, 
                    model_type, 
                    filename=filename,
                    target_type=target_type
                )
                generated_plots.append(filename)
                print(f"   ✅ Generated: {filename}")
            except Exception as e:
                print(f"   ❌ Error generating {filename}: {e}")
        
        # Options 2+3: Heel vs Cord comparison using heel_all and cord_all data
        option2_3_results = {k: v for k, v in all_results.items() if ('heel_all' in k or 'cord_all' in k) and model_type in k}
        if option2_3_results:
            filename = f"outputs/plots/biomarker_frequency_heel_vs_cord_{model_type}_{target_type}_options2_3_heel_cord_all.png"
            try:
                plot_biomarker_frequency_heel_vs_cord(
                    option2_3_results, 
                    model_type, 
                    filename=filename,
                    target_type=target_type
                )
                generated_plots.append(filename)
                print(f"   ✅ Generated: {filename}")
            except Exception as e:
                print(f"   ❌ Error generating {filename}: {e}")
    
    # Generate best model plots
    print(f"\n🏆 Generating best model plots...")
    
    # Find best models for each dataset type
    best_models = {
        'Clinical': {'regression': None, 'classification': None},
        'Biomarker': {'regression': None, 'classification': None},
        'Combined': {'regression': None, 'classification': None}
    }
    
    # Initialize best performance metrics
    best_rmse = {'Clinical': float('inf'), 'Biomarker': float('inf'), 'Combined': float('inf')}
    best_auc = {'Clinical': 0.0, 'Biomarker': 0.0, 'Combined': 0.0}
    
    # Find best models
    for key, result in all_results.items():
        if 'summary' not in result:
            continue
            
        # Extract model info from key
        parts = key.split('_')
        if len(parts) >= 4:
            model_name = parts[-1]  # Clinical, Biomarker, or Combined
            model_type = parts[-2]  # lasso_cv, elasticnet_cv, stabl
            
            if model_name in best_models:
                # Check for best regression model (lowest RMSE)
                if 'rmse_mean' in result['summary']:
                    rmse = result['summary']['rmse_mean']
                    if rmse < best_rmse[model_name]:
                        best_rmse[model_name] = rmse
                        best_models[model_name]['regression'] = {
                            'key': key,
                            'model_type': model_type,
                            'rmse': rmse
                        }
                
                # Check for best classification model (highest AUC)
                if 'auc_mean' in result['summary']:
                    auc = result['summary']['auc_mean']
                    if auc > best_auc[model_name]:
                        best_auc[model_name] = auc
                        best_models[model_name]['classification'] = {
                            'key': key,
                            'model_type': model_type,
                            'auc': auc
                        }
    
    # Generate best model plots
    for dataset_type in ['Clinical', 'Biomarker', 'Combined']:
        print(f"\n📊 Processing best {dataset_type} models...")
        
        # Best regression model
        if best_models[dataset_type]['regression']:
            best_reg = best_models[dataset_type]['regression']
            print(f"   Best {dataset_type} regression: {best_reg['model_type']} (RMSE: {best_reg['rmse']:.4f})")
            
            # Option 1 plots
            option1_results = {k: v for k, v in all_results.items() if 'both_samples' in k and dataset_type in k and best_reg['model_type'] in k}
            if option1_results:
                filename = f"outputs/plots/best_{dataset_type.lower()}_regression_heel_vs_cord_{target_type}_option1_both_samples.png"
                try:
                    plot_biomarker_frequency_heel_vs_cord(
                        option1_results, 
                        best_reg['model_type'], 
                        filename=filename,
                        target_type=target_type
                    )
                    generated_plots.append(filename)
                    print(f"      ✅ Generated: {filename}")
                except Exception as e:
                    print(f"      ❌ Error generating {filename}: {e}")
            
            # Options 2+3 plots
            option2_3_results = {k: v for k, v in all_results.items() if ('heel_all' in k or 'cord_all' in k) and dataset_type in k and best_reg['model_type'] in k}
            if option2_3_results:
                filename = f"outputs/plots/best_{dataset_type.lower()}_regression_heel_vs_cord_{target_type}_options2_3_heel_cord_all.png"
                try:
                    plot_biomarker_frequency_heel_vs_cord(
                        option2_3_results, 
                        best_reg['model_type'], 
                        filename=filename,
                        target_type=target_type
                    )
                    generated_plots.append(filename)
                    print(f"      ✅ Generated: {filename}")
                except Exception as e:
                    print(f"      ❌ Error generating {filename}: {e}")
        
        # Best classification model
        if best_models[dataset_type]['classification']:
            best_cls = best_models[dataset_type]['classification']
            print(f"   Best {dataset_type} classification: {best_cls['model_type']} (AUC: {best_cls['auc']:.4f})")
            
            # Option 1 plots
            option1_results = {k: v for k, v in all_results.items() if 'both_samples' in k and dataset_type in k and best_cls['model_type'] in k}
            if option1_results:
                filename = f"outputs/plots/best_{dataset_type.lower()}_classification_heel_vs_cord_{target_type}_option1_both_samples.png"
                try:
                    plot_biomarker_frequency_heel_vs_cord(
                        option1_results, 
                        best_cls['model_type'], 
                        filename=filename,
                        target_type=target_type
                    )
                    generated_plots.append(filename)
                    print(f"      ✅ Generated: {filename}")
                except Exception as e:
                    print(f"      ❌ Error generating {filename}: {e}")
            
            # Options 2+3 plots
            option2_3_results = {k: v for k, v in all_results.items() if ('heel_all' in k or 'cord_all' in k) and dataset_type in k and best_cls['model_type'] in k}
            if option2_3_results:
                filename = f"outputs/plots/best_{dataset_type.lower()}_classification_heel_vs_cord_{target_type}_options2_3_heel_cord_all.png"
                try:
                    plot_biomarker_frequency_heel_vs_cord(
                        option2_3_results, 
                        best_cls['model_type'], 
                        filename=filename,
                        target_type=target_type
                    )
                    generated_plots.append(filename)
                    print(f"      ✅ Generated: {filename}")
                except Exception as e:
                    print(f"      ❌ Error generating {filename}: {e}")
    
    # Summary
    print(f"\n🎉 Regeneration complete!")
    print(f"📊 Generated {len(generated_plots)} biomarker frequency plots:")
    for plot in generated_plots:
        print(f"   📄 {plot}")
    
    return generated_plots

def main():
    """Main function."""
    print("🔄 Biomarker Frequency Plot Regenerator")
    print("=" * 50)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        target_type = sys.argv[1].lower()
        if target_type not in ['gestational_age', 'birth_weight']:
            print("Error: target_type must be 'gestational_age' or 'birth_weight'")
            print("Usage: python3 regenerate_biomarker_plots.py [gestational_age|birth_weight]")
            sys.exit(1)
        target_types = [target_type]
    else:
        # Try both target types
        target_types = ['gestational_age', 'birth_weight']
    
    # Regenerate plots for each target type
    for target_type in target_types:
        print(f"\n🎯 Processing {target_type}...")
        regenerate_biomarker_plots(target_type)
    
    print(f"\n✅ All biomarker frequency plots regenerated with top 50% labeling!")

if __name__ == "__main__":
    main()
