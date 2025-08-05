#!/usr/bin/env python3
"""
Generate comparison plots for both gestational age and birth weight analyses.

This script generates the heel vs cord biomarker frequency comparison plots
for both target types and model types, using the new all_results.pkl key format.
"""

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from src.utils import count_high_weight_biomarkers
from collections import defaultdict

def plot_biomarker_frequency_heel_vs_cord_no_intersect(heel_result, cord_result, model_type, filename, min_freq=0.5, top_n=10, use_classification=False):
    """
    Plot all biomarkers above 50% frequency for cord (x axis) and heel (y axis).
    Label only the top N biomarkers by max frequency, with clean deterministic placement.
    """
    # Use classification coefficients if requested
    if use_classification:
        heel_coefs = heel_result.get('all_classification_coefficients')
        cord_coefs = cord_result.get('all_classification_coefficients')
        if not (heel_coefs and cord_coefs):
            print(f"      ✗ Skipping classification plot: missing classification coefficients.")
            return
    else:
        heel_coefs = heel_result['all_coefficients']
        cord_coefs = cord_result['all_coefficients']
    heel_feature_names = heel_result['feature_names']
    cord_feature_names = cord_result['feature_names']
    heel_freq = count_high_weight_biomarkers(heel_coefs, heel_feature_names, threshold=0.01)
    heel_freq = np.array(heel_freq)
    cord_freq = count_high_weight_biomarkers(cord_coefs, cord_feature_names, threshold=0.01)
    cord_freq = np.array(cord_freq)
    # Build a union of all feature names
    all_features = set(heel_feature_names) | set(cord_feature_names)
    # Build frequency dicts
    heel_freq_dict = dict(zip(heel_feature_names, heel_freq))
    cord_freq_dict = dict(zip(cord_feature_names, cord_freq))
    # Only keep features above min_freq in either dataset
    features_to_plot = [f for f in all_features if (heel_freq_dict.get(f,0) >= min_freq) or (cord_freq_dict.get(f,0) >= min_freq)]
    x = [cord_freq_dict.get(f, 0) for f in features_to_plot]
    y = [heel_freq_dict.get(f, 0) for f in features_to_plot]
    max_freq = [max(xi, yi) for xi, yi in zip(x, y)]
    # Determine which features to label: top N by max frequency
    if len(features_to_plot) > top_n:
        top_indices = np.argsort(max_freq)[-top_n:][::-1]
    else:
        top_indices = np.arange(len(features_to_plot))
    # Group points by (rounded) coordinates for clustering
    coord_to_indices = defaultdict(list)
    for idx in top_indices:
        # Round to 2 decimals for grouping
        coord = (round(x[idx], 2), round(y[idx], 2))
        coord_to_indices[coord].append(idx)
    # Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    scatter = ax.scatter(x, y, c='black', s=30, alpha=0.8, zorder=3)
    ax.plot([0, 1], [0, 1], ':', color='black', alpha=0.7, linewidth=1.5, zorder=1)
    # Smart label placement
    for coord, indices in coord_to_indices.items():
        n = len(indices)
        # If more than 3 points at the same location, label only the top 3 by frequency
        if n > 3:
            indices = sorted(indices, key=lambda i: max_freq[i], reverse=True)[:3]
            n = 3
        for j, idx in enumerate(indices):
            f = features_to_plot[idx]
            xval, yval = x[idx], y[idx]
            # Edge logic: for x near 0 or 1, always fan out vertically above with arrow pointing up
            if xval < 0.08 or xval > 0.92:
                label_x = 0.03 if xval < 0.08 else 0.97
                offset_y = 0.08 + 0.08 * (j - (n-1)/2)
                txt = ax.text(
                    label_x, yval + offset_y, f, fontsize=8, fontfamily='sans-serif', color='black', alpha=0.9,
                    ha='center', va='bottom', clip_on=False, zorder=5
                )
                ax.annotate('', xy=(xval, yval), xytext=(label_x, yval + offset_y),
                            arrowprops=dict(arrowstyle='-', color='black', alpha=0.9, lw=1.2, clip_on=False),
                            annotation_clip=False)
            elif yval > 0.92:
                # For top edge, fan out labels downwards (below the point), never above y=1
                offset_y = 0.08 + 0.08 * (j - (n-1)/2)
                label_y = max(yval - offset_y, 0)  # Ensure label stays within plot
                label_y = min(label_y, 0.98)  # Clip to y=0.98 max
                txt = ax.text(
                    xval, label_y, f, fontsize=8, fontfamily='sans-serif', color='black', alpha=0.9,
                    ha='center', va='top', clip_on=False, zorder=5
                )
                ax.annotate('', xy=(xval, yval), xytext=(xval, label_y),
                            arrowprops=dict(arrowstyle='-', color='black', alpha=0.9, lw=1.2, clip_on=False),
                            annotation_clip=False)
            elif yval < 0.08:
                # For bottom edge, fan out labels upwards (above the point)
                offset_y = 0.08 + 0.08 * (j - (n-1)/2)
                txt = ax.text(
                    xval, yval + offset_y, f, fontsize=8, fontfamily='sans-serif', color='black', alpha=0.9,
                    ha='center', va='bottom', clip_on=False, zorder=5
                )
                ax.annotate('', xy=(xval, yval), xytext=(xval, yval + offset_y),
                            arrowprops=dict(arrowstyle='-', color='black', alpha=0.9, lw=1.2, clip_on=False),
                            annotation_clip=False)
            else:
                offset_x = 0.05
                offset_y = 0.05 + 0.08 * (j - (n-1)/2)
                txt = ax.text(
                    xval + offset_x, yval + offset_y, f, fontsize=8, fontfamily='sans-serif', color='black', alpha=0.9,
                    ha='left', va='bottom', clip_on=False, zorder=5
                )
                ax.annotate('', xy=(xval, yval), xytext=(xval + offset_x, yval + offset_y),
                            arrowprops=dict(arrowstyle='-', color='black', alpha=0.9, lw=1.2, clip_on=False),
                            annotation_clip=False)
    ax.set_xlabel('Cord Biomarker Frequency', fontsize=14, fontweight='bold')
    ax.set_ylabel('Heel Biomarker Frequency', fontsize=14, fontweight='bold')
    ax.set_title(f'Biomarker Frequency Comparison (>{min_freq*100:.0f}%): {model_type.capitalize()}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, linestyle='-', alpha=0.3, linewidth=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Add extra top margin for the title
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"      ✓ Plot saved: {filename}")

def generate_comparison_plots(target_type, all_results):
    print(f"\nGenerating comparison plots for {target_type}...")
    for model_type in ['lasso', 'elasticnet']:
        print(f"  Processing {model_type}...")
        # Find heel and cord results for both_samples
        heel_key = f"both_samples_heel_{model_type}_cv_Biomarker_{target_type}"
        cord_key = f"both_samples_cord_{model_type}_cv_Biomarker_{target_type}"
        heel_result = all_results.get(heel_key)
        cord_result = all_results.get(cord_key)
        if heel_result and cord_result:
            outdir = f'outputs/plots/{target_type}/biomarker_frequency_cordvsheel/'
            os.makedirs(outdir, exist_ok=True)
            # Regression plot
            filename = f"{outdir}heel_vs_cord_biomarker_frequency_option1_both_samples_{target_type}.png"
            plot_biomarker_frequency_heel_vs_cord_no_intersect(heel_result, cord_result, model_type, filename)
            # Classification plot (if available)
            filename_cls = f"{outdir}heel_vs_cord_biomarker_frequency_option1_both_samples_{target_type}_classification.png"
            plot_biomarker_frequency_heel_vs_cord_no_intersect(heel_result, cord_result, model_type, filename_cls, use_classification=True)
        else:
            print(f"    ✗ Missing heel or cord result for both_samples {model_type} {target_type}")
        # Find heel_all and cord_all results
        heel_key2 = f"heel_all_heel_{model_type}_cv_Biomarker_{target_type}"
        cord_key2 = f"cord_all_cord_{model_type}_cv_Biomarker_{target_type}"
        heel_result2 = all_results.get(heel_key2)
        cord_result2 = all_results.get(cord_key2)
        if heel_result2 and cord_result2:
            outdir = f'outputs/plots/{target_type}/biomarker_frequency_cordvsheel/'
            os.makedirs(outdir, exist_ok=True)
            # Regression plot
            filename = f"{outdir}heel_vs_cord_biomarker_frequency_options2_3_heel_cord_all_{target_type}.png"
            plot_biomarker_frequency_heel_vs_cord_no_intersect(heel_result2, cord_result2, model_type, filename)
            # Classification plot (if available)
            filename_cls = f"{outdir}heel_vs_cord_biomarker_frequency_options2_3_heel_cord_all_{target_type}_classification.png"
            plot_biomarker_frequency_heel_vs_cord_no_intersect(heel_result2, cord_result2, model_type, filename_cls, use_classification=True)
        else:
            print(f"    ✗ Missing heel_all or cord_all result for {model_type} {target_type}")

def main():
    print("Generating Heel vs Cord Comparison Plots")
    print("=" * 50)
    results_file = "all_results.pkl"
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found.")
        return
    with open(results_file, 'rb') as f:
        all_results = pickle.load(f)
    print(f"Loaded {len(all_results)} result entries")
    for target_type in ['gestational_age', 'birth_weight']:
        generate_comparison_plots(target_type, all_results)
    print("\n" + "=" * 50)
    print("COMPARISON PLOTS GENERATED!")
    print("=" * 50)
    print("Check outputs/plots/<target_type>/biomarker_frequency_cordvsheel/ for the generated plots.")

if __name__ == "__main__":
    main() 