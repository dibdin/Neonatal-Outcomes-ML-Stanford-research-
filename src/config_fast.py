"""
Fast configuration parameters for testing the pipeline.

This module contains reduced parameters for faster testing.

Author: Diba Dindoust
Date: 07/01/2025
"""

# Model training parameters (reduced for testing)
N_REPEATS = 1  # Reduced from 2 for faster testing
TEST_SIZE = 0.2  # Fraction of data to use for testing (20%)
PRETERM_CUTOFF = 37  # Gestational age cutoff for preterm classification (weeks)

# Data options (reduced scope)
DATA_OPTION_LABELS = {
    1: 'both_samples'
}

# Model types (reduced scope)
FAST_MODEL_TYPES = ['elasticnet_cv']

# Dataset types (reduced scope)
FAST_DATASET_TYPES = ['cord']

# Model configs (reduced scope)
FAST_MODEL_CONFIGS = [
    {'name': 'Biomarker', 'data_type': 'biomarker', 'allowed_models': ['elasticnet_cv']}
] 