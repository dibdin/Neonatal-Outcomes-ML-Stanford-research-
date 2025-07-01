"""
Configuration parameters for the gestational age prediction pipeline.

This module contains all the configurable parameters used throughout the project.
"""

# Model training parameters
N_REPEATS = 100  # Number of training runs for statistical significance
TEST_SIZE = 0.2  # Fraction of data to use for testing (20%)
PRETERM_CUTOFF = 37  # Gestational age cutoff for preterm classification (weeks)