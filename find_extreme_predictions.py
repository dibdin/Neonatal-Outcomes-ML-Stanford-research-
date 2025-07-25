import pickle
import numpy as np

# === USER CONFIGURATION ===
# Path to the file containing predictions (adjust as needed)
results_pkl = "all_results.pkl"
model_key = "both_samples_cord_elasticnet_Biomarker_birth_weight"  # Adjust as needed
extreme_threshold = 10  # kg

# === LOAD RESULTS ===
with open(results_pkl, 'rb') as f:
    all_results = pickle.load(f)

if model_key not in all_results:
    print(f"Model key '{model_key}' not found in all_results.")
    exit(1)

results = all_results[model_key]

# === FIND EXTREME PREDICTIONS ===
extreme_indices = []
for run_idx, run in enumerate(results.get('predictions', [])):
    preds = np.array(run.get('pred', []))
    trues = np.array(run.get('true', []))
    for i, pred in enumerate(preds):
        if pred > extreme_threshold:
            print(f"Extreme prediction found: Run {run_idx}, Index {i}, Predicted: {pred}, True: {trues[i] if i < len(trues) else 'N/A'}")
            extreme_indices.append((run_idx, i, pred, trues[i] if i < len(trues) else 'N/A'))

if not extreme_indices:
    print(f"No predictions above {extreme_threshold} kg found.")
else:
    print(f"Total extreme predictions found: {len(extreme_indices)}") 