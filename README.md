# Gestational Age Prediction and Preterm Birth Classification

This repository contains a comprehensive machine learning pipeline for predicting gestational age and classifying preterm births using serum biomarkers and clinical data from cord and heel blood samples.

Link to poster presentation at MCHRI DRIVE 2025 Symposium: https://drive.google.com/file/d/1zbxuDT7oj7iz5E8p28a_RDRSphCZUlfZ/view?usp=sharing

## Overview

The project implements multiple model approaches to predict gestational age and identify preterm births:

1. **Clinical Model**: Uses clinical/demographic features (columns 145, 146, 159 + pairwise interactions)
2. **Biomarker Model**: Uses serum biomarker features (columns 30-141)
3. **Combined Model**: Uses both clinical and biomarker features

## Features

- **Multiple Model Types**: Support for Lasso, ElasticNet, and STABL algorithms
- **Comprehensive Evaluation**: MAE, RMSE, and AUC metrics with confidence intervals
- **Subgroup Analysis**: Performance evaluation for preterm vs term babies and SGA vs normal babies
- **Feature Importance**: Biomarker frequency analysis and feature selection patterns
- **Advanced Visualization**: ROC curves, performance plots, scatter plots, and comparison tables
- **Reproducible Results**: Multiple training runs with statistical significance
- **Sherlock Integration**: Optimized for high-performance computing clusters
- **Hyperparameter Analysis**: Comprehensive analysis of model hyperparameters across runs
- **Heel vs Cord Comparison**: Detailed comparison between heel and cord blood samples

## Project Structure

```
serum_biomarkers/
├── main.py                           # Main execution script
├── run_sherlock.sh                   # Sherlock batch job script
├── run_sherlock_optimized.sh         # Optimized Sherlock script
├── run_sherlock_parallel.sh          # Parallel processing script
├── sync_to_sherlock.sh               # Sync files to Sherlock
├── subgroup_analysis.py              # Subgroup analysis (preterm/term, SGA/normal)
├── analyze_hyperparameters.py        # Hyperparameter analysis
├── comprehensive_hyperparameter_summary.py # Comprehensive hyperparameter summary
├── pipeline_summary.py               # Pipeline summary and statistics
├── plot_summary.py                   # Summary plots generation
├── plot_auc_results.py               # AUC results visualization
├── plot_best_models_comparison.py    # Best models comparison plots
├── plot_feature_selection.py         # Feature selection analysis
├── plot_scatter_plots.py             # Scatter plots generation
├── best_model_scatter_plots.py       # Best model scatter plots
├── generate_comparison_plots.py      # Heel vs cord comparison plots
├── organize_plots.py                 # Plot organization and structure
├── merge_all_results.py              # Merge results from different analyses
├── regenerate_biomarker_plots.py     # Regenerate biomarker frequency plots
├── src/                              # Source code modules
│   ├── config.py                     # Configuration parameters
│   ├── data_loader.py                # Data loading and preprocessing
│   ├── model.py                      # Model definitions and training
│   ├── metrics.py                    # Performance metrics calculation
│   └── utils.py                      # Utility functions and plotting
├── data/                             # Data files (not included in repo)
├── outputs/                          # Generated outputs
│   ├── models/                       # Model outputs and coefficients
│   ├── plots/                        # Generated plots and visualizations
│   │   ├── gestational_age/          # Gestational age analysis plots
│   │   │   ├── performance_metrics/  # MAE, RMSE, AUC plots
│   │   │   ├── roc_curves/          # ROC curves for classification
│   │   │   ├── scatter_plots/       # True vs predicted scatter plots
│   │   │   ├── biomarker_frequency/ # Biomarker importance plots
│   │   │   ├── biomarker_frequency_cordvsheel/ # Heel vs cord comparisons
│   │   │   └── summary_plots/       # Summary comparison plots
│   │   ├── birth_weight/             # Birth weight analysis plots
│   │   │   ├── performance_metrics/  # MAE, RMSE, AUC plots
│   │   │   ├── roc_curves/          # ROC curves for SGA classification
│   │   │   ├── scatter_plots/       # True vs predicted scatter plots
│   │   │   ├── biomarker_frequency/ # Biomarker importance plots
│   │   │   ├── biomarker_frequency_cordvsheel/ # Heel vs cord comparisons
│   │   │   └── summary_plots/       # Summary comparison plots
│   │   ├── best_model_scatter_plots/ # Best model scatter plots
│   │   ├── scatter_plots/           # General scatter plots
│   │   ├── feature_selection_scatter.png # Feature selection analysis
│   │   ├── best_models_heel_vs_cord.png # Best models comparison
│   │   ├── auc_results_comparison.png # AUC results comparison
│   │   ├── auc_results_alternative.png # Alternative AUC visualization
│   │   └── README.md                # Plot organization guide
│   └── tables/                       # Performance metrics tables
│       ├── *_performance_metrics.csv # Detailed performance metrics
│       └── *_model_comparison.csv   # Model comparison tables
├── notebooks/                        # Jupyter notebooks for exploration
├── requirements.txt                  # Python dependencies
├── FINAL_IMPLEMENTATION_SUMMARY.md   # Implementation summary
├── IMPLEMENTATION_SUMMARY.md         # Development progress
└── README.md                         # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/serum_biomarkers.git
cd serum_biomarkers
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Local Execution

Run the complete pipeline with default settings:

```bash
python main.py gestational_age
python main.py birth_weight
```

### Sherlock High-Performance Computing

For large-scale analysis on Sherlock cluster:

```bash
# Submit batch job
./run_sherlock.sh

# Or use optimized version
./run_sherlock_optimized.sh

# Or use parallel processing
./run_sherlock_parallel.sh
```

### Individual Scripts

#### Subgroup Analysis
```bash
python subgroup_analysis.py gestational_age
python subgroup_analysis.py birth_weight
```

#### Hyperparameter Analysis
```bash
python analyze_hyperparameters.py
python comprehensive_hyperparameter_summary.py
```

#### Plot Generation
```bash
python plot_summary.py
python plot_auc_results.py
python plot_best_models_comparison.py
python plot_feature_selection.py
python plot_scatter_plots.py
python best_model_scatter_plots.py
```

#### Plot Organization
```bash
python organize_plots.py
```

### Configuration

Modify the configuration in `src/config.py`:

```python
# Model training parameters
N_REPEATS = 100  # Number of training runs for statistical significance
TEST_SIZE = 0.2  # Fraction of data to use for testing (20%)
PRETERM_CUTOFF = 37  # Gestational age cutoff for preterm classification (weeks)
SGA_PERCENTILE = 10  # Percentile cutoff for SGA classification
```

## Output Files

The pipeline generates comprehensive outputs:

### Performance Metrics Tables
- `*_performance_metrics.csv` - Detailed performance metrics for each model
- `*_model_comparison.csv` - Comparison tables across models

### Visualization Plots

#### Performance Metrics
- `*_metrics_with_ci.png` - Performance plots with confidence intervals
- `*_roc_curve_*.png` - ROC curves for classification tasks
- `true_vs_predicted_scatter_*.png` - True vs predicted scatter plots

#### Feature Analysis
- `best_model_biomarker_frequency_*.png` - Best model biomarker analysis
- `heel_vs_cord_biomarker_frequency_*.png` - Heel vs cord biomarker comparisons
- `feature_selection_scatter.png` - Feature selection frequency analysis

#### Summary Plots
- `summary_auc_by_dataset_and_model_*.png` - Summary AUC plots
- `summary_mae_by_dataset_and_model_*.png` - Summary MAE plots
- `summary_rmse_by_dataset_and_model_*.png` - Summary RMSE plots
- `best_models_heel_vs_cord.png` - Best models comparison
- `auc_results_comparison.png` - AUC results comparison

### Data Files
- `all_results_gestational_age.pkl` - Gestational age analysis results
- `all_results_birth_weight.pkl` - Birth weight analysis results
- `all_results.pkl` - Merged results from both analyses
- `all_results_*_with_subgroups.pkl` - Results with subgroup analysis

## Data Requirements

The pipeline expects a CSV file with the following structure:
- **Biomarker columns**: Columns 30-141 (serum biomarkers)
- **Clinical columns**: Columns 145, 146, 159 (demographic/clinical features)
- **Target variables**: `gestational_age_weeks`, `birth_weight`
- **Source column**: `Source` (with values 'CORD' or 'HEEL')

## Model Details

### Clinical Model
- **Features**: 6 total (3 base clinical + 3 pairwise interactions)
- **Purpose**: Baseline performance using only clinical/demographic data

### Biomarker Model
- **Features**: 111 serum biomarkers
- **Purpose**: Identify most predictive biomarkers for gestational age

### Combined Model
- **Features**: 117 total (111 biomarkers + 6 clinical)
- **Purpose**: Optimal performance using all available data

## Performance Metrics

The pipeline evaluates models using:

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
- **RMSE (Root Mean Square Error)**: Square root of average squared differences
- **AUC (Area Under Curve)**: Performance in classification tasks
- **Confidence Intervals**: 95% confidence intervals for all metrics
- **Subgroup Analysis**: Performance breakdown for:
  - Preterm (<37 weeks) vs term (≥37 weeks) babies
  - SGA (<10th percentile) vs normal (≥10th percentile) babies

## Subgroup Analysis

The pipeline includes comprehensive subgroup analysis:

- **Preterm vs Term**: Performance comparison for gestational age prediction
- **SGA vs Normal**: Performance comparison for birth weight prediction
- **Heel vs Cord**: Comparison between different blood sample types
- **Model-specific Analysis**: Performance breakdown by model type and dataset

## Hyperparameter Analysis

Comprehensive analysis of model hyperparameters:

- **Alpha Values**: Regularization strength analysis
- **L1 Ratio**: ElasticNet mixing parameter analysis
- **Cross-validation Results**: Stability analysis across folds
- **Feature Selection Patterns**: Analysis of selected features across runs

## Sherlock Integration

Optimized for Stanford's Sherlock high-performance computing cluster:

- **Batch Job Scripts**: Automated job submission and management
- **Resource Optimization**: Memory and CPU allocation
- **Parallel Processing**: Multi-core analysis capabilities
- **File Synchronization**: Automated file transfer and organization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{serum_biomarkers,
  title={Gestational Age Prediction and Preterm Birth Classification},
  author={},
  year={2025},
  url={https://github.com/yourusername/serum_biomarkers}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact [].

## Acknowledgments

- MCHRI (Maternal and Child Health Research Institute) for funding and support
- Research team for data collection and preprocessing
- Stanford Research Computing for Sherlock cluster access
- Open source community for the tools and libraries used in this project
