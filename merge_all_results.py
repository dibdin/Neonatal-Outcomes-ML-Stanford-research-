import pickle

# Load gestational age results
with open('all_results_gestational_age.pkl', 'rb') as f:
    ga_results = pickle.load(f)

# Load birth weight results
with open('all_results_birth_weight.pkl', 'rb') as f:
    bw_results = pickle.load(f)

# Merge the dictionaries, adding target type as a suffix to each key
merged_results = {}
for k, v in ga_results.items():
    merged_results[f"{k}_gestational_age"] = v
for k, v in bw_results.items():
    merged_results[f"{k}_birth_weight"] = v

# Save the merged results
with open('all_results.pkl', 'wb') as f:
    pickle.dump(merged_results, f)

print("Merged results saved to all_results.pkl with target type in keys") 