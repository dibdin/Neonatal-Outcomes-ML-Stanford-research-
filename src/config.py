"""Pipeline configuration for the gestational age prediction study.

Modify these constants to reproduce the analysis under different settings.
"""

RANDOM_SEED = 42          # Seed for all random number generators
N_REPEATS = 100           # Number of repeated train/test splits
TEST_SIZE = 0.2           # Fraction of data held out per split
PRETERM_CUTOFF = 37       # Gestational age threshold (weeks) for preterm classification

USE_CLASS_BALANCING = True        # Apply class-weight balancing in logistic regression
USE_STRATIFIED_SPLITTING = True   # Use stratified splits to preserve class ratio

from pathlib import Path
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAINING_CSV_BOTH = _DATA_DIR / "Bangladesh_children_with_both_samples.csv"
TRAINING_CSV_ALL  = _DATA_DIR / "BangladeshcombineddatasetJan252022.csv"
