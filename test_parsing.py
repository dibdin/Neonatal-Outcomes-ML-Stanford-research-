#!/usr/bin/env python3
"""
Test script to debug the parsing issue
"""

import re

# Sample log entries
sample_log = """
    Classification optimized hyperparameters - C: 0.01, L1_ratio: 0.5
    Classification optimized hyperparameters - C: 0.1, L1_ratio: None
    Classification optimized hyperparameters - C: 1.0, L1_ratio: 0.3
    Classification optimized hyperparameters - C: 0.001, L1_ratio: 0.7
"""

# Test the regex pattern
pattern = r'Classification optimized hyperparameters - C: ([\d.]+), L1_ratio: ([^,\n]+)'

matches = re.findall(pattern, sample_log)

print("=== TESTING PARSING ===")
for i, (c_val, l1_ratio) in enumerate(matches):
    c_float = float(c_val)
    print(f"Match {i+1}: C='{c_val}' -> float={c_float}, L1_ratio='{l1_ratio}'")
    if c_float == 0.0:
        print(f"  ⚠️  WARNING: C=0.0 detected!")
    else:
        print(f"  ✅ C={c_float} (not 0.0)")

# Test with actual log entries
print("\n=== TESTING WITH ACTUAL LOG ===")
with open('gestational_age_output.log', 'r') as f:
    log_content = f.read()

# Find a few actual entries
matches = re.findall(pattern, log_content)
print(f"Found {len(matches)} classification entries")

# Check first 10 matches
for i, (c_val, l1_ratio) in enumerate(matches[:10]):
    c_float = float(c_val)
    print(f"Entry {i+1}: C='{c_val}' -> float={c_float}, L1_ratio='{l1_ratio}'")
    if c_float == 0.0:
        print(f"  ⚠️  WARNING: C=0.0 detected!")
    else:
        print(f"  ✅ C={c_float} (not 0.0)")

# Count how many are actually 0.0
zero_count = 0
for c_val, l1_ratio in matches:
    c_float = float(c_val)
    if c_float == 0.0:
        zero_count += 1

print(f"\nTotal entries: {len(matches)}")
print(f"Entries with C=0.0: {zero_count}")
print(f"Percentage with C=0.0: {zero_count/len(matches)*100:.1f}%") 