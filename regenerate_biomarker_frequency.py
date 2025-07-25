"""
Regenerate biomarker frequency and SHAP-based preterm/term scatter plots for best models.

This script generates:
- Biomarker selection frequency plots for the best biomarker model in each dataset
- SHAP-based preterm/term scatter plots for the best biomarker model in each dataset

Usage:
    python regenerate_biomarker_frequency.py

Outputs are saved in outputs/plots/.
"""

import os
import glob
import pickle
import numpy as np
from src.utils import plot_feature_frequency

# --- Biomarker Frequency Plot ---
def generate_biomarker_frequency_plot(dataset):
    """
    Generate biomarker selection frequency plot for the best biomarker model in the given dataset.
    Only produce the average plot across all runs for each model/dataset.
    """
    # Find all model output files for this dataset
    files = glob.glob(f"outputs/models/{dataset}_*_biomarker_run*_model_outputs.pkl")
    if not files:
        print(f"No model output files found for {dataset}.")
        return
    all_coefs = []
    feature_names = None
    for model_file in files:
        with open(model_file, 'rb') as f:
            output = pickle.load(f)
        coefs = output['coef']
        # Ensure coefs is a list or tuple for vstack
        if not isinstance(coefs, (list, tuple)):
            coefs = [coefs]
        # Filter out None values
        coefs = [c for c in coefs if c is not None]
        if len(coefs) == 0:
            print(f"[WARNING] Skipping {model_file}: all coefficients are None.")
            continue
        all_coefs.extend(coefs)
        # Robustly get feature names
        if feature_names is None:
            if 'selected_feature_names' in output:
                feature_names = output['selected_feature_names']
            elif 'feature_names' in output:
                feature_names = output['feature_names']
            elif 'features' in output and hasattr(output['features'], 'columns'):
                feature_names = output['features'].columns
    if not all_coefs or feature_names is None:
        print(f"[WARNING] No valid coefficients or feature names for {dataset}.")
        return
    # Count frequency of nonzero coefficients across all runs
    freq = np.sum(np.abs(np.vstack(all_coefs)) >= 0.1, axis=0)
    freq_normalized = freq / len(all_coefs)
    output_file = f"outputs/plots/best_model_biomarker_frequency_{dataset}.png"
    # Use dataset name for plot
    plot_feature_frequency(feature_names, freq_normalized, filename=output_file, model_name="All", dataset_name=dataset.capitalize())
    print(f"Biomarker frequency plot saved to: {output_file}")

if __name__ == "__main__":
    print("Regenerating biomarker frequency plots for all available model files...")
    print("="*60)
    for dataset in ['heel', 'cord']:
        generate_biomarker_frequency_plot(dataset)
    print("="*60)
    print("Plot regeneration complete!")
    print("Generated files:")
    print("  - outputs/plots/best_model_biomarker_frequency_heel.png")
    print("  - outputs/plots/best_model_biomarker_frequency_cord.png") 