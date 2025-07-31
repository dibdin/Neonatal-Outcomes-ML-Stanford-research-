#!/usr/bin/env python3
"""
Test script to verify preprocessing logging is working correctly.
"""

import sys
import os
sys.path.append('src')

from data_loader import load_and_process_data

def test_preprocessing_logging():
    """Test the preprocessing pipeline with enhanced logging."""
    
    print("🧪 TESTING PREPROCESSING LOGGING")
    print("=" * 60)
    
    # Test different model types
    test_configs = [
        {'model_type': 'clinical', 'dataset_type': 'heel', 'data_option': 1},
        {'model_type': 'biomarker', 'dataset_type': 'heel', 'data_option': 1},
        {'model_type': 'combined', 'dataset_type': 'heel', 'data_option': 1},
    ]
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n🔬 TEST {i}: {config['model_type'].upper()} MODEL")
        print("-" * 40)
        
        try:
            X, y = load_and_process_data(
                dataset_type=config['dataset_type'],
                model_type=config['model_type'],
                data_option=config['data_option'],
                target_type='gestational_age'
            )
            
            print(f"✅ Test {i} completed successfully!")
            print(f"   Final X shape: {X.shape}")
            print(f"   Final y shape: {y.shape}")
            
        except Exception as e:
            print(f"❌ Test {i} failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🎯 ALL TESTS COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    test_preprocessing_logging() 