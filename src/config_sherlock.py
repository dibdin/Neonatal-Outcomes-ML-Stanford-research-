"""
Configuration parameters for the gestational age prediction pipeline (Sherlock version).

This module contains configurable parameters optimized for Sherlock cluster execution.

Author: Diba Dindoust
Date: 07/01/2025
"""

# Model training parameters (reduced for Sherlock to avoid timeouts)
N_REPEATS = 10  # Number of training runs for statistical significance (reduced from 100)
TEST_SIZE = 0.2  # Fraction of data to use for testing (20%)
PRETERM_CUTOFF = 37  # Gestational age cutoff for preterm classification (weeks)
