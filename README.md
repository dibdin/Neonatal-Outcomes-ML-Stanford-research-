# Gestational Age Prediction and Preterm Birth Classification

This repository contains a comprehensive machine learning pipeline for predicting gestational age and classifying preterm births using serum biomarkers and clinical data.

## Overview

The project implements three different model approaches to predict gestational age and identify preterm births:

1. **Clinical Model**: Uses clinical/demographic features (columns 145, 146, 159 + pairwise interactions)
2. **Biomarker Model**: Uses serum biomarker features (columns 30-141)
3. **Combined Model**: Uses both clinical and biomarker features

## Features

- **Multiple Model Types**: Support for Lasso, ElasticNet, and STABL algorithms
- **Comprehensive Evaluation**: MAE, RMSE, and AUC metrics with confidence intervals
- **Separate Analysis**: Performance evaluation for preterm vs term babies
- **Feature Importance**: Biomarker frequency analysis for interpretability
- **Visualization**: ROC curves, performance plots, and comparison tables
- **Reproducible Results**: Multiple training runs with statistical significance

## Project Structure

```
serum_biomarkers/
├── main.py                 # Main execution script
├── src/                    # Source code modules
│   ├── config.py          # Configuration parameters
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── model.py           # Model definitions and training
│   ├── metrics.py         # Performance metrics calculation
│   └── utils.py           # Utility functions and plotting
├── data/                   # Data files (not included in repo)
├── outputs/                # Generated outputs
│   ├── models/            # Model outputs and coefficients
│   ├── plots/             # Generated plots and visualizations
│   └── tables/            # Performance metrics tables
├── notebooks/              # Jupyter notebooks for exploration
└── README.md              # This file
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

### Basic Usage

Run the complete pipeline with default settings:

```bash
python main.py
```

This will:
- Train all three models (Clinical, Biomarker, Combined)
- Generate performance metrics with confidence intervals
- Create visualizations and comparison tables
- Save all outputs to the `outputs/` directory

### Configuration

Modify the configuration in `main.py`:

```python
# Configuration
dataset_type = 'heel'  # 'cord' or 'heel'
model_type = 'lasso'   # 'lasso', 'elasticnet', or 'stabl'
```

### Output Files

The pipeline generates the following outputs:

**For each model:**
- `{model}_performance_metrics.csv` - Detailed performance metrics
- `{model}_metrics_with_ci.png` - Performance plots with confidence intervals
- `{model}_roc_avg_over_runs.png` - ROC curves averaged over runs
- `{model}_biomarker_frequency.png` - Feature importance (biomarker model only)

**Overall:**
- `model_comparison.csv` - Comparison table of all models

## Data Requirements

The pipeline expects a CSV file with the following structure:
- **Biomarker columns**: Columns 30-141 (serum biomarkers)
- **Clinical columns**: Columns 145, 146, 159 (demographic/clinical features)
- **Target variable**: `gestational_age_weeks`
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

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual gestational age
- **RMSE (Root Mean Square Error)**: Square root of average squared differences
- **AUC (Area Under Curve)**: Performance in preterm vs term classification
- **Confidence Intervals**: 95% confidence intervals for all metrics
- **Separate Analysis**: Performance breakdown for preterm (<37 weeks) vs term (≥37 weeks) babies

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
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/serum_biomarkers}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

## Acknowledgments

- MCHRI (Maternal and Child Health Research Institute) for funding and support
- Research team for data collection and preprocessing
- Open source community for the tools and libraries used in this project