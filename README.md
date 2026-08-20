# Gestational Age Prediction and Preterm Birth Classification

Machine learning pipeline for predicting gestational age and classifying preterm births using serum biomarkers and clinical features from a training cohort (Bangladesh) and an external validation cohort (Zimbabwe).

## Overview

The pipeline trains regularised regression and classification models (ElasticNet) across three feature sets:

- **Clinical model** — birth weight, sex, multiple birth, pre-pregnancy BMI, maternal age at delivery
- **Biomarker model** — 92 newborn screening serum biomarkers (amino acids, acylcarnitines, haemoglobin fractions, metabolic analytes)
- **Combined model** — clinical + biomarker features

Each model is evaluated on both heel-prick and cord-blood sample sub-types. Training uses 100 repeated random train/test splits (80/20) for robust performance estimation. The preterm cutoff is 37 completed weeks of gestation.

## Project Structure

```
serum_biomarkers/
├── src/
│   ├── config.py                  # Global constants (seed, cutoff, data paths)
│   ├── feature_sets.py            # Feature lists and data loading functions
│   ├── model.py                   # Model definitions and pipeline construction
│   ├── classification.py          # Binary classification training and evaluation
│   ├── regression.py              # Continuous GA regression training and evaluation
│   ├── cohort_descriptives.py     # Cohort summary statistics
│   ├── plot_utils.py              # Shared plotting utilities
│   ├── plotting.py                # Publication-ready figure generation
│   ├── imputation_audit.py        # Missing-data audit across cohorts
│   ├── compare_cohort_overlap.py  # Feature overlap between cohorts
│   ├── compare_feature_quality.py # Feature quality comparison
│   ├── compare_sample_subsets.py  # Heel vs cord sub-type comparison
│   └── prepare_validation_dataset.py  # External validation data preparation
├── data/                          # Data files (not included in repo)
├── outputs/                       # Generated results and figures
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/dibdin/Neonatal-Outcomes-ML-Stanford-research-.git
cd Neonatal-Outcomes-ML-Stanford-research-
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Run classification (preterm vs term)
python -m src.classification

# Run regression (continuous gestational age)
python -m src.regression

# Generate publication figures
python -m src.plotting

# Cohort descriptive statistics
python -m src.cohort_descriptives
```

## Configuration

Edit `src/config.py` to set data paths and key parameters:

```python
RANDOM_SEED    = 42
PRETERM_CUTOFF = 37   # weeks; samples < 37 weeks classified as preterm
N_REPEATS      = 100  # number of repeated train/test splits
TEST_SIZE      = 0.2
```

## Data

The pipeline expects two CSV files (not included):

- **Training cohort** — heel-prick and cord-blood samples with `gestational_age_weeks`, clinical features, and biomarker analytes; a `Source` column (`HEEL` / `CORD`) identifies sample type.
- **External validation cohort** — same feature columns; sample type is encoded in the `study_id` suffix (`-C` = cord, `-H` = heel).

## Performance Metrics

Models are evaluated using MAE, RMSE, and AUC (for binary preterm classification), reported as means with 95% confidence intervals across the 100 repeated splits.

## Citation

If you use this code, please cite the associated publication (details to be added upon acceptance).

## Acknowledgements

This work was supported by the Maternal and Child Health Research Institute (MCHRI) and conducted at Stanford University.
