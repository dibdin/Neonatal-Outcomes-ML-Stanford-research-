#!/usr/bin/env python3
"""
Debug floating point precision issues
"""

import re
from collections import Counter

def debug_precision():
    """Debug floating point precision"""
    
    with open('gestational_age_output.log', 'r') as f:
        log_content = f.read()
    
    # Find all classification hyperparameter positions
    classification_positions = []
    for match in re.finditer(r'Classification optimized hyperparameters - C: ([\d.]+), L1_ratio: ([^,\n]+)', log_content):
        classification_positions.append({
            'position': match.start(),
            'C': float(match.group(1)),
            'l1_ratio': match.group(2).strip()
        })
    
    # Test with Clinical ElasticNet values
    clinical_elasticnet = []
    for class_pos in classification_positions:
        if class_pos['l1_ratio'] != 'None':  # ElasticNet
            clinical_elasticnet.append(class_pos['C'])
    
    print("=== DEBUGGING PRECISION ===")
    print(f"Clinical ElasticNet C values (first 10): {clinical_elasticnet[:10]}")
    print(f"All unique C values: {sorted(set(clinical_elasticnet))}")
    print(f"Min C: {min(clinical_elasticnet)}")
    print(f"Max C: {max(clinical_elasticnet)}")
    print(f"C=0.0 count: {sum(1 for c in clinical_elasticnet if c == 0.0)}")
    print(f"C<0.001 count: {sum(1 for c in clinical_elasticnet if c < 0.001)}")
    
    # Check if any values are very close to 0
    very_small = [c for c in clinical_elasticnet if c < 0.001]
    print(f"Values < 0.001: {very_small[:10]}")
    
    # Test Counter
    c_counts = Counter(clinical_elasticnet)
    print(f"Counter most common: {c_counts.most_common(5)}")
    
    # Test if 0.0 is somehow being added
    print(f"\nTesting if 0.0 is in the list: {0.0 in clinical_elasticnet}")
    print(f"Testing if 0.0 is in the set: {0.0 in set(clinical_elasticnet)}")

if __name__ == "__main__":
    debug_precision() 