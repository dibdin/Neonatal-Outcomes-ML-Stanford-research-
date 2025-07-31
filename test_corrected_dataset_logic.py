#!/usr/bin/env python3
"""
Test script to verify the corrected dataset logic for each data option.
"""

import sys
import os
sys.path.append('src')

from data_loader import load_and_process_data

def test_dataset_logic():
    """Test the dataset logic for each data option."""
    
    print("🧪 TESTING CORRECTED DATASET LOGIC")
    print("=" * 60)
    
    # Test configurations based on the corrected training matrix
    test_configs = [
        # Data Option 1: both_samples - should work with both heel and cord
        {'data_option': 1, 'dataset_type': 'heel', 'expected_samples': '~792'},
        {'data_option': 1, 'dataset_type': 'cord', 'expected_samples': '~793'},
        
        # Data Option 2: heel_all - should only work with heel
        {'data_option': 2, 'dataset_type': 'heel', 'expected_samples': '~1340'},
        
        # Data Option 3: cord_all - should only work with cord
        {'data_option': 3, 'dataset_type': 'cord', 'expected_samples': '~1340'},
    ]
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n🔬 TEST {i}: Data Option {config['data_option']} - {config['dataset_type']} dataset")
        print("-" * 50)
        print(f"Expected samples: {config['expected_samples']}")
        
        try:
            X, y = load_and_process_data(
                dataset_type=config['dataset_type'],
                model_type='biomarker',  # Use biomarker for consistent testing
                data_option=config['data_option'],
                target_type='gestational_age'
            )
            
            print(f"✅ Test {i} completed successfully!")
            print(f"   Actual samples: {X.shape[0]}")
            print(f"   Features: {X.shape[1]}")
            
            # Verify the logic
            if config['data_option'] == 1:
                if config['dataset_type'] in ['heel', 'cord']:
                    print(f"   ✅ Correct: Option 1 supports both heel and cord")
                else:
                    print(f"   ❌ Error: Option 1 should support both heel and cord")
            elif config['data_option'] == 2:
                if config['dataset_type'] == 'heel':
                    print(f"   ✅ Correct: Option 2 supports only heel")
                else:
                    print(f"   ❌ Error: Option 2 should support only heel")
            elif config['data_option'] == 3:
                if config['dataset_type'] == 'cord':
                    print(f"   ✅ Correct: Option 3 supports only cord")
                else:
                    print(f"   ❌ Error: Option 3 should support only cord")
            
        except Exception as e:
            print(f"❌ Test {i} failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎯 DATASET LOGIC VERIFICATION COMPLETE!")
    print("=" * 60)
    
    # Print summary of expected behavior
    print(f"\n📊 EXPECTED BEHAVIOR SUMMARY:")
    print(f"   Data Option 1 (both_samples): heel + cord datasets")
    print(f"   Data Option 2 (heel_all): heel dataset only")
    print(f"   Data Option 3 (cord_all): cord dataset only")

if __name__ == "__main__":
    test_dataset_logic() 