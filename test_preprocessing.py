import pandas as pd
import numpy as np
from src.data_loader import load_and_process_data

print("=== TESTING DATA PREPROCESSING ===\n")

for target_type in ['gestational_age', 'birth_weight']:
    print(f"--- Target: {target_type} ---")
    try:
        X, y, df = load_and_process_data(
            dataset_type='cord',
            model_type='combined',
            data_option=1,
            target_type=target_type,
            return_dataframe=True
        )
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"First 5 targets: {y.head().values}")
        print(f"Feature means (should be ~0):\n{X.mean().head()}\n")
        print(f"Feature stds (should be ~1):\n{X.std().head()}\n")
        if target_type == 'birth_weight':
            # Check for invalid birth weights
            invalid = (df['birth_weight_kg'] == 99.9) | (df['birth_weight_kg'] < 0.5) | (df['birth_weight_kg'] > 6)
            print(f"Invalid birth_weight_kg rows remaining: {invalid.sum()}")
        print()
    except Exception as e:
        print(f"Error for target_type={target_type}: {e}\n")
print("=== DONE ===") 