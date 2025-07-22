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
    """
    # Find best model output file
    files = glob.glob(f"outputs/models/{dataset}_*_biomarker_run*_model_outputs.pkl")
    if not files:
        print(f"No model output files found for {dataset}.")
        return
    # Find file with highest AUC
    best_file = max(files, key=lambda f: pickle.load(open(f, 'rb'))['AUC'])
    with open(best_file, 'rb') as f:
        output = pickle.load(f)
    coefs = output['coef']
    feature_names = output['selected_feature_names'] if 'selected_feature_names' in output else output['feature_names']
    # Count frequency of nonzero coefficients
    freq = np.sum(np.abs(np.vstack(coefs)) >= 0.1, axis=0)
    freq_normalized = freq / len(coefs)
    output_file = f"outputs/plots/best_model_biomarker_frequency_{dataset}.png"
    # Extract model type from filename
    model_type = best_file.split('_')[1].capitalize()
    plot_feature_frequency(feature_names, freq_normalized, filename=output_file, model_name=model_type, dataset_name=dataset.capitalize())
    print(f"Biomarker frequency plot saved to: {output_file}")

# --- SHAP-based Preterm/Term Scatter Plot ---
def generate_shap_preterm_term_scatter(dataset, top_n=10):
    """
    Generate a SHAP-based scatter plot of feature importance (preterm vs term) for the best biomarker model.
    X-axis: mean(|SHAP|) for term
    Y-axis: mean(|SHAP|) for preterm
    """
    from src.data_loader import load_and_process_data
    from src.utils import plot_shap_preterm_term_scatter
    import shap
    # Find best model output file
    files = glob.glob(f"outputs/models/{dataset}_*_biomarker_run*_model_outputs.pkl")
    if not files:
        print(f"No model output files found for {dataset}.")
        return
    # Find file with highest AUC
    best_file = max(files, key=lambda f: pickle.load(open(f, 'rb'))['AUC'])
    with open(best_file, 'rb') as f:
        output = pickle.load(f)
    model = output['model'] if 'model' in output else None
    X = output['features']
    y = output['GA true']
    feature_names = output['selected_feature_names'] if 'selected_feature_names' in output else output['feature_names']
    # Load data and split by preterm/term
    data, _ = load_and_process_data(dataset)
    preterm_mask = data['preterm'] == 1
    term_mask = data['preterm'] == 0
    # Compute SHAP values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    abs_shap = np.abs(shap_values.values)
    # Mean absolute SHAP per group
    mean_shap_preterm = abs_shap[preterm_mask].mean(axis=0)
    mean_shap_term = abs_shap[term_mask].mean(axis=0)
    output_file = f"outputs/plots/best_model_shap_preterm_term_scatter_{dataset}.png"
    plot_shap_preterm_term_scatter(mean_shap_term, mean_shap_preterm, feature_names, output_file, top_n=top_n)
    print(f"SHAP-based preterm/term scatter plot saved to: {output_file}")

if __name__ == "__main__":
    print("Regenerating biomarker frequency and SHAP-based preterm/term scatter plots for best models...")
    print("="*60)
    for dataset in ['heel', 'cord']:
        generate_biomarker_frequency_plot(dataset)
        generate_shap_preterm_term_scatter(dataset)
    print("="*60)
    print("Plot regeneration complete!")
    print("Generated files:")
    print("  - outputs/plots/best_model_biomarker_frequency_heel.png")
    print("  - outputs/plots/best_model_shap_preterm_term_scatter_heel.png")
    print("  - outputs/plots/best_model_biomarker_frequency_cord.png")
    print("  - outputs/plots/best_model_shap_preterm_term_scatter_cord.png") 