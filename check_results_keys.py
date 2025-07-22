#!/usr/bin/env python3
"""
Check what keys are in the all_results.pkl file and determine target type.
"""

import pickle
import numpy as np

# Load the results file
with open("all_results.pkl", 'rb') as f:
    all_results = pickle.load(f)

print(f"Total results: {len(all_results)}")
print("\nKeys in all_results:")
for i, key in enumerate(sorted(all_results.keys())):
    print(f"  {i+1:2d}. {key}")

print("\nSample result structure:")
sample_key = list(all_results.keys())[0]
sample_result = all_results[sample_key]
print(f"Key: {sample_key}")
print(f"Keys in result: {list(sample_result.keys())}")

# Check if this is gestational age or birth weight by looking at the data
if 'predictions' in sample_result:
    preds = sample_result['predictions']
    print(f"\nPredictions type: {type(preds)}")
    if isinstance(preds, list) and len(preds) > 0:
        # Get the first prediction to check its structure
        first_pred = preds[0]
        print(f"First prediction type: {type(first_pred)}")
        if isinstance(first_pred, dict):
            print(f"First prediction keys: {list(first_pred.keys())}")
            if 'true' in first_pred and 'pred' in first_pred:
                true_vals = first_pred['true']
                pred_vals = first_pred['pred']
                print(f"True values range: {np.min(true_vals):.1f} to {np.max(true_vals):.1f}")
                print(f"Predicted values range: {np.min(pred_vals):.1f} to {np.max(pred_vals):.1f}")
                if np.max(true_vals) > 50:  # Likely gestational age (weeks)
                    print("This appears to be GESTATIONAL AGE data (values in weeks)")
                else:  # Likely birth weight (kg)
                    print("This appears to be BIRTH WEIGHT data (values in kg)") 