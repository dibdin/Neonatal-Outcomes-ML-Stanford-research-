import pickle
import numpy as np

results_pkl = "all_results.pkl"
reasonable_min = 0  # kg
reasonable_max = 10  # kg

with open(results_pkl, 'rb') as f:
    all_results = pickle.load(f)

found_issue = False
for model_key, results in all_results.items():
    for run_idx, run in enumerate(results.get('predictions', [])):
        preds = np.array(run.get('pred', []))
        trues = np.array(run.get('true', []))
        # Check for length mismatch
        if len(preds) != len(trues):
            print(f"Length mismatch: Model key: {model_key}, Run {run_idx}, len(pred)={len(preds)}, len(true)={len(trues)}")
            found_issue = True
        # Check for NaN or inf
        if np.any(np.isnan(preds)) or np.any(np.isnan(trues)):
            print(f"NaN found: Model key: {model_key}, Run {run_idx}")
            found_issue = True
        if np.any(np.isinf(preds)) or np.any(np.isinf(trues)):
            print(f"Inf found: Model key: {model_key}, Run {run_idx}")
            found_issue = True
        # Check for values outside reasonable range
        for i, (pred, true) in enumerate(zip(preds, trues)):
            if pred < reasonable_min or pred > reasonable_max:
                print(f"Unreasonable pred: Model key: {model_key}, Run {run_idx}, Index {i}, Pred: {pred}, True: {true}")
                found_issue = True
            if true < reasonable_min or true > reasonable_max:
                print(f"Unreasonable true: Model key: {model_key}, Run {run_idx}, Index {i}, Pred: {pred}, True: {true}")
                found_issue = True
if not found_issue:
    print("No plotting errors or unreasonable values found in any model key.") 