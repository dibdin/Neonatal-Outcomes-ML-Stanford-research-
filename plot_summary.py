#!/usr/bin/env python3
"""
Script to provide a summary of organized plots and help navigate them.
"""

import os
from pathlib import Path
import glob

def print_plot_summary():
    """Print a summary of all organized plots."""
    
    plots_dir = Path("outputs/plots")
    if not plots_dir.exists():
        print("Error: outputs/plots directory not found!")
        return
    
    print("=" * 80)
    print("PLOT ORGANIZATION SUMMARY")
    print("=" * 80)
    
    # Count files in each directory
    total_files = len(list(plots_dir.rglob("*.png")))
    print(f"Total plots: {total_files}")
    print()
    
    # Gestational Age plots
    ga_dir = plots_dir / "gestational_age"
    if ga_dir.exists():
        ga_files = len(list(ga_dir.rglob("*.png")))
        print(f"📊 GESTATIONAL AGE ANALYSIS ({ga_files} plots)")
        print("-" * 50)
        
        for subdir in ["performance_metrics", "roc_curves", "scatter_plots", "biomarker_frequency", "biomarker_frequency_cordvsheel", "summary_plots"]:
            subdir_path = ga_dir / subdir
            if subdir_path.exists():
                count = len(list(subdir_path.glob("*.png")))
                print(f"  {subdir.replace('_', ' ').title()}: {count} plots")
        print()
    
    # Birth Weight plots
    bw_dir = plots_dir / "birth_weight"
    if bw_dir.exists():
        bw_files = len(list(bw_dir.rglob("*.png")))
        print(f"⚖️  BIRTH WEIGHT ANALYSIS ({bw_files} plots)")
        print("-" * 50)
        
        for subdir in ["performance_metrics", "roc_curves", "scatter_plots", "biomarker_frequency", "biomarker_frequency_cordvsheel", "summary_plots"]:
            subdir_path = bw_dir / subdir
            if subdir_path.exists():
                count = len(list(subdir_path.glob("*.png")))
                print(f"  {subdir.replace('_', ' ').title()}: {count} plots")
        print()
    
    print("=" * 80)
    print("QUICK NAVIGATION GUIDE")
    print("=" * 80)
    print("📁 Main directories:")
    print("  outputs/plots/gestational_age/     - Gestational age analysis")
    print("  outputs/plots/birth_weight/        - Birth weight analysis")
    print()
    print("📊 Subdirectories in each main directory:")
    print("  performance_metrics/  - MAE, RMSE, AUC plots with confidence intervals")
    print("  roc_curves/          - ROC curves for classification tasks")
    print("  scatter_plots/       - True vs predicted scatter plots")
    print("  biomarker_frequency/ - Biomarker importance frequency plots")
    print("  biomarker_frequency_cordvsheel/ - Heel vs Cord biomarker comparison plots")
    print("  summary_plots/       - Summary plots comparing models")
    print()
    print("🔍 Key plot types to look for:")
    print("  • *_metrics_with_ci.png     - Performance metrics")
    print("  • *_roc_curve_*.png         - ROC curves")
    print("  • true_vs_predicted_scatter_*.png - Regression scatter plots")
    print("  • best_model_biomarker_frequency_*.png - Best model biomarker analysis")
    print("  • summary_*.png             - Summary comparison plots")
    print()
    print("📖 For detailed information, see: outputs/plots/README.md")

def list_key_plots():
    """List some key plots that are most important to look at."""
    
    print("\n" + "=" * 80)
    print("KEY PLOTS TO EXAMINE")
    print("=" * 80)
    
    plots_dir = Path("outputs/plots")
    
    # Gestational Age key plots
    print("📊 GESTATIONAL AGE - Key Performance Plots:")
    ga_summary = list(plots_dir.glob("gestational_age/summary_plots/summary_*.png"))
    for plot in sorted(ga_summary):
        print(f"  • {plot.name}")
    
    # Birth Weight key plots
    print("\n⚖️  BIRTH WEIGHT - Key Performance Plots:")
    bw_summary = list(plots_dir.glob("birth_weight/summary_plots/summary_*.png"))
    for plot in sorted(bw_summary):
        print(f"  • {plot.name}")
    
    # Best model biomarker plots
    print("\n🧬 BEST MODEL BIOMARKER ANALYSIS:")
    best_models = list(plots_dir.glob("*/biomarker_frequency/best_model_*.png"))
    for plot in sorted(best_models):
        print(f"  • {plot.name}")
    
    # Main scatter plots
    print("\n📈 MAIN SCATTER PLOTS:")
    scatter_plots = list(plots_dir.glob("*/scatter_plots/true_vs_predicted_scatter_*.png"))
    for plot in sorted(scatter_plots):
        print(f"  • {plot.name}")
    
    # Heel vs Cord comparison plots
    print("\n🔄 HEEL VS CORD COMPARISON PLOTS:")
    heel_vs_cord_plots = list(plots_dir.glob("*/biomarker_frequency_cordvsheel/*.png"))
    for plot in sorted(heel_vs_cord_plots):
        print(f"  • {plot.name}")

if __name__ == "__main__":
    print_plot_summary()
    list_key_plots() 