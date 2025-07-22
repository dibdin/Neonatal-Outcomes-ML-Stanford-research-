"""
Summary of the three pipeline approaches.

This script provides an overview of the data distribution and sample sizes
for the three different pipeline approaches.
"""

import pandas as pd
import numpy as np

def analyze_pipeline_data():
    """
    Analyze and summarize the data for all three pipelines.
    """
    print("=== THREE PIPELINE APPROACH SUMMARY ===")
    print()
    
    # Load both datasets
    both_samples_df = pd.read_csv('data/Bangladeshcombineddataset_both_samples.csv')
    original_df = pd.read_csv('data/BangladeshcombineddatasetJan252022.csv')
    
    print("1. BOTH SAMPLES PIPELINE (Current)")
    print("   Data source: Bangladeshcombineddataset_both_samples.csv")
    print(f"   Total samples: {len(both_samples_df)}")
    print(f"   Cord samples: {len(both_samples_df[both_samples_df['Source'] == 'CORD'])}")
    print(f"   Heel samples: {len(both_samples_df[both_samples_df['Source'] == 'HEEL'])}")
    
    # Gestational age analysis
    preterm_both = both_samples_df['gestational_age_weeks'] < 37
    print(f"   Preterm (<37 weeks): {preterm_both.sum()} ({preterm_both.mean()*100:.1f}%)")
    print(f"   Term (≥37 weeks): {(~preterm_both).sum()} ({(~preterm_both).mean()*100:.1f}%)")
    print(f"   Mean GA: {both_samples_df['gestational_age_weeks'].mean():.1f} weeks")
    print()
    
    print("2. HEEL-ONLY PIPELINE (New)")
    print("   Data source: BangladeshcombineddatasetJan252022.csv (filtered to HEEL)")
    heel_df = original_df[original_df['Source'] == 'HEEL']
    print(f"   Total samples: {len(heel_df)}")
    
    preterm_heel = heel_df['gestational_age_weeks'] < 37
    print(f"   Preterm (<37 weeks): {preterm_heel.sum()} ({preterm_heel.mean()*100:.1f}%)")
    print(f"   Term (≥37 weeks): {(~preterm_heel).sum()} ({(~preterm_heel).mean()*100:.1f}%)")
    print(f"   Mean GA: {heel_df['gestational_age_weeks'].mean():.1f} weeks")
    print()
    
    print("3. CORD-ONLY PIPELINE (New)")
    print("   Data source: BangladeshcombineddatasetJan252022.csv (filtered to CORD)")
    cord_df = original_df[original_df['Source'] == 'CORD']
    print(f"   Total samples: {len(cord_df)}")
    
    preterm_cord = cord_df['gestational_age_weeks'] < 37
    print(f"   Preterm (<37 weeks): {preterm_cord.sum()} ({preterm_cord.mean()*100:.1f}%)")
    print(f"   Term (≥37 weeks): {(~preterm_cord).sum()} ({(~preterm_cord).mean()*100:.1f}%)")
    print(f"   Mean GA: {cord_df['gestational_age_weeks'].mean():.1f} weeks")
    print()
    
    print("=== COMPARISON ===")
    print(f"Sample size increase:")
    print(f"  Heel-only vs Both-samples: {len(heel_df)} vs {len(both_samples_df[both_samples_df['Source'] == 'HEEL'])} (+{len(heel_df) - len(both_samples_df[both_samples_df['Source'] == 'HEEL'])} samples)")
    print(f"  Cord-only vs Both-samples: {len(cord_df)} vs {len(both_samples_df[both_samples_df['Source'] == 'CORD'])} (+{len(cord_df) - len(both_samples_df[both_samples_df['Source'] == 'CORD'])} samples)")
    print()
    
    print("Preterm prevalence:")
    print(f"  Both-samples: {preterm_both.mean()*100:.1f}%")
    print(f"  Heel-only: {preterm_heel.mean()*100:.1f}%")
    print(f"  Cord-only: {preterm_cord.mean()*100:.1f}%")
    print()
    
    print("=== PIPELINE IMPLEMENTATION STATUS ===")
    print("✓ Current pipeline (both_samples): Preserved and unchanged")
    print("✓ Heel-only pipeline: Implemented in main_extended.py")
    print("✓ Cord-only pipeline: Implemented in main_extended.py")
    print("✓ Extended data loader: src/data_loader_extended.py")
    print("✓ Test script: test_extended_pipelines.py")
    print()
    print("To run all pipelines:")
    print("  python3 main_extended.py")
    print()
    print("To run individual pipeline:")
    print("  python3 -c \"from main_extended import run_single_model_extended; run_single_model_extended('Biomarker', 'biomarker', 'heel_only', 'lasso')\"")

if __name__ == "__main__":
    analyze_pipeline_data() 