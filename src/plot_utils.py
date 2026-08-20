"""Visualisation and analysis utilities for the gestational age prediction pipeline.

This module provides comprehensive utilities for:
- Plotting and visualization (ROC curves, feature frequencies, SHAP plots)
- Model saving and loading
- Performance metrics calculation and visualization
- Data processing and analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
import numpy as np
import os
import pickle
from adjustText import adjust_text


def save_model(model, path):
    """Save a trained model to disk using joblib."""
    joblib.dump(model, path)


def load_model(path):
    """Load a trained model from disk using joblib."""
    return joblib.load(path)


def save_plot(fig, filename):
    """Save a matplotlib figure to disk."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename)
    plt.close(fig)


def plot_roc_curve(fpr, tpr, auc, filename="roc_curve.png"):
    """
    Plot ROC curve with AUC value.

    Args:
        fpr (array): False positive rates
        tpr (array): True positive rates
        auc (float): Area under the curve
        filename (str): Output file path
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot ROC curve with a distinct color and label
    ax.plot(fpr, tpr, color='#2E86AB', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')

    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], linestyle='--', color='#A23B72', alpha=0.7, linewidth=1.5, label='Random Classifier (AUC = 0.5)')

    # Customize plot
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold', pad=20)

    # Add legend with better positioning
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Set axis limits and ticks
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Add AUC value as text annotation
    ax.text(0.6, 0.2, f'AUC = {auc:.3f}', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))

    save_plot(fig, filename)


def plot_feature_frequency(feature_names, freq, filename, min_freq=0.5, model_name="", dataset_name=""):
    """
    Plot a vertical dot plot of biomarkers with selection frequencies above a minimum threshold.

    Args:
        feature_names: list or array of biomarker names
        freq: array of frequencies (same order as feature_names)
        filename: output file path
        min_freq: minimum frequency threshold (default 0.5 = 50%)
        model_name: name of the model (e.g., "ElasticNet")
        dataset_name: name of the dataset (e.g., "Heel")
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Convert to pandas Series for easy sorting and selection
    freq_series = pd.Series(freq, index=feature_names)
    freq_series = freq_series.sort_values(ascending=False)
    freq_series = freq_series[freq_series >= min_freq]  # Filter by minimum frequency threshold

    # Create figure with single subplot (only left side)
    fig, ax = plt.subplots(1, 1, figsize=(12, max(8, int(len(freq_series) * 0.4))))

    # Create color gradient based on frequency values
    colors = plt.cm.viridis(freq_series.values)

    # Dot plot of top features with color gradient
    y_pos = np.arange(len(freq_series))
    scatter = ax.scatter(freq_series.values, y_pos, c=freq_series.values, cmap='viridis',
                        s=100, zorder=3, alpha=0.8, edgecolors='black', linewidth=0.5)

    # Add colorbar to show frequency scale
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Selection Frequency', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(freq_series.index, fontsize=11)
    ax.set_xlabel('Selection Frequency', fontsize=14, fontweight='bold')
    ax.set_ylabel('Biomarkers', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()  # Most frequent at top

    # Add grid for better readability
    ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=0, color='gray')

    # Remove spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    # Create title with model name if provided
    if model_name and dataset_name:
        title = f'Biomarker Selection Frequency ≥ {min_freq*100:.0f}% - {model_name} on {dataset_name}'
    elif model_name:
        title = f'Biomarker Selection Frequency ≥ {min_freq*100:.0f}% - {model_name}'
    else:
        title = f'Biomarker Selection Frequency ≥ {min_freq*100:.0f}%'

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Add frequency threshold line
    ax.axvline(x=min_freq, color='red', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Threshold ({min_freq*100:.0f}%)')
    ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_predictions(actual, predicted, filename="pred_vs_actual.png", error_bars=None):
    """
    Plot predicted probability vs actual outcome.

    Args:
        actual (array): Actual outcomes
        predicted (array): Predicted probabilities
        filename (str): Output file path
        error_bars (array, optional): Error bars for predictions
    """
    fig, ax = plt.subplots()
    sns.barplot(x=["Preterm", "Term"], y=predicted, yerr=error_bars, ax=ax)
    ax.set_title("Predicted vs. Actual Preterm Births")
    ax.set_ylabel("Predicted Probability")
    save_plot(fig, filename)


def count_high_weight_biomarkers(coef_list, feature_names, threshold=0.1, abs_val=True):
    """
    Count how often each biomarker has a coefficient above threshold across runs.

    Args:
        coef_list (list): List of coefficient arrays from different runs
        feature_names (list): List of feature names
        threshold (float): Threshold for counting as "selected"
        abs_val (bool): Whether to use absolute values

    Returns:
        pd.Series: Frequency count for each feature (normalized to 0-1)
    """
    # Initialize frequency counter
    freq = {feature: 0 for feature in feature_names}
    n_runs = len(coef_list)

    for coef in coef_list:
        if isinstance(coef, pd.Series):
            coef = coef.values

        # Handle case where coef might be a 2D array (e.g., from classification)
        if coef.ndim > 1:
            coef = coef.flatten()

        for i, weight in enumerate(coef):
            # Ensure weight is a scalar
            if hasattr(weight, '__iter__') and not isinstance(weight, (str, bytes)):
                weight = weight[0] if len(weight) > 0 else 0

            value = abs(weight) if abs_val else weight
            if value >= threshold:
                freq[feature_names[i]] += 1

    # Normalize by number of runs to get frequency between 0 and 1
    normalized_freq = {feature: count / n_runs for feature, count in freq.items()}

    # Return a pandas Series with all features, including those with 0 frequency
    return pd.Series(normalized_freq).reindex(feature_names, fill_value=0)


def save_all_as_pickle(obj, filename="full_model_output.pkl"):
    """Save an object to disk using pickle."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def plot_mae(maes, filename="outputs/plots/mae_over_runs.png"):
    """Plot MAE over multiple runs."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(range(1, len(maes) + 1), maes, marker='o')
    ax.set_title("MAE Over Runs")
    ax.set_xlabel("Run")
    ax.set_ylabel("MAE")
    fig.tight_layout()
    save_plot(fig, filename)


def plot_rmse(rmses, filename="outputs/plots/rmse_over_runs.png"):
    """Plot RMSE over multiple runs."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(range(1, len(rmses) + 1), rmses, marker='o', color='orange')
    ax.set_title("RMSE Over Runs")
    ax.set_xlabel("Run")
    ax.set_ylabel("RMSE")
    fig.tight_layout()
    save_plot(fig, filename)


def create_metrics_table(maes, rmses, preterm_metrics_list, term_metrics_list, filename="outputs/tables/performance_metrics.csv"):
    """
    Create a comprehensive table with MAE/RMSE metrics and confidence intervals.

    Args:
        maes (list): List of MAE values across runs
        rmses (list): List of RMSE values across runs
        preterm_metrics_list (list): List of preterm-specific metrics
        term_metrics_list (list): List of term-specific metrics
        filename (str): Output file path

    Returns:
        pd.DataFrame: Formatted metrics table
    """
    from src.metrics import compute_confidence_interval

    def _safe_ci(values):
        values = [float(v) for v in values if np.isfinite(v)]
        if not values:
            return (np.nan, np.nan, np.nan)
        mean = float(np.mean(values))
        if len(values) < 2:
            return (mean, np.nan, np.nan)
        m, lo, hi = compute_confidence_interval(values)
        return (float(m), float(lo), float(hi))

    # Compute confidence intervals for overall metrics
    mae_mean, mae_ci_lower, mae_ci_upper = _safe_ci(maes)
    rmse_mean, rmse_ci_lower, rmse_ci_upper = _safe_ci(rmses)

    # Extract preterm and term metrics
    preterm_maes = [m['mae'] for m in preterm_metrics_list if not np.isnan(m['mae'])]
    preterm_rmses = [m['rmse'] for m in preterm_metrics_list if not np.isnan(m['rmse'])]
    term_maes = [m['mae'] for m in term_metrics_list if not np.isnan(m['mae'])]
    term_rmses = [m['rmse'] for m in term_metrics_list if not np.isnan(m['rmse'])]

    # Compute confidence intervals for preterm and term
    preterm_mae_mean, preterm_mae_ci_lower, preterm_mae_ci_upper = _safe_ci(preterm_maes)
    preterm_rmse_mean, preterm_rmse_ci_lower, preterm_rmse_ci_upper = _safe_ci(preterm_rmses)
    term_mae_mean, term_mae_ci_lower, term_mae_ci_upper = _safe_ci(term_maes)
    term_rmse_mean, term_rmse_ci_lower, term_rmse_ci_upper = _safe_ci(term_rmses)

    # Create table data
    table_data = {
        'Metric': ['MAE', 'RMSE', 'MAE (Preterm)', 'RMSE (Preterm)', 'MAE (Term)', 'RMSE (Term)'],
        'Mean': [mae_mean, rmse_mean, preterm_mae_mean, preterm_rmse_mean, term_mae_mean, term_rmse_mean],
        'CI_Lower': [mae_ci_lower, rmse_ci_lower, preterm_mae_ci_lower, preterm_rmse_ci_lower, term_mae_ci_lower, term_rmse_ci_lower],
        'CI_Upper': [mae_ci_upper, rmse_ci_upper, preterm_mae_ci_upper, preterm_rmse_ci_upper, term_mae_ci_upper, term_rmse_ci_upper],
        'Sample_Size': [len(maes), len(rmses), len(preterm_maes), len(preterm_rmses), len(term_maes), len(term_rmses)]
    }

    df = pd.DataFrame(table_data)

    # Format confidence intervals
    df['CI_95%'] = df.apply(lambda row: f"[{row['CI_Lower']:.3f}, {row['CI_Upper']:.3f}]", axis=1)
    df['Mean_CI'] = df.apply(lambda row: f"{row['Mean']:.3f} {row['CI_95%']}", axis=1)

    # Save table
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)

    return df


def plot_metrics_with_confidence_intervals(maes, rmses, preterm_metrics_list, term_metrics_list, filename="outputs/plots/metrics_with_ci.png"):
    """
    Create bar plots for MAE/RMSE with confidence intervals.

    Args:
        maes (list): List of MAE values across runs
        rmses (list): List of RMSE values across runs
        preterm_metrics_list (list): List of preterm-specific metrics
        term_metrics_list (list): List of term-specific metrics
        filename (str): Output file path
    """
    from src.metrics import compute_confidence_interval

    def _safe_ci(values):
        values = [float(v) for v in values if np.isfinite(v)]
        if not values:
            return (np.nan, np.nan, np.nan)
        mean = float(np.mean(values))
        if len(values) < 2:
            return (mean, np.nan, np.nan)
        m, lo, hi = compute_confidence_interval(values)
        return (float(m), float(lo), float(hi))

    # Compute confidence intervals
    mae_mean, mae_ci_lower, mae_ci_upper = _safe_ci(maes)
    rmse_mean, rmse_ci_lower, rmse_ci_upper = _safe_ci(rmses)

    # Extract preterm and term metrics
    preterm_maes = [m['mae'] for m in preterm_metrics_list if not np.isnan(m['mae'])]
    preterm_rmses = [m['rmse'] for m in preterm_metrics_list if not np.isnan(m['rmse'])]
    term_maes = [m['mae'] for m in term_metrics_list if not np.isnan(m['mae'])]
    term_rmses = [m['rmse'] for m in term_metrics_list if not np.isnan(m['rmse'])]

    preterm_mae_mean, preterm_mae_ci_lower, preterm_mae_ci_upper = _safe_ci(preterm_maes)
    preterm_rmse_mean, preterm_rmse_ci_lower, preterm_rmse_ci_upper = _safe_ci(preterm_rmses)
    term_mae_mean, term_mae_ci_lower, term_mae_ci_upper = _safe_ci(term_maes)
    term_rmse_mean, term_rmse_ci_lower, term_rmse_ci_upper = _safe_ci(term_rmses)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Define consistent color palette
    colors = {
        'Overall': '#2E86AB',     # Blue
        'Preterm': '#A23B72',     # Purple
        'Term': '#F18F01',        # Orange
    }

    # MAE plot
    categories = ['Overall', 'Preterm', 'Term']
    mae_means = [mae_mean, preterm_mae_mean, term_mae_mean]
    mae_ci_lowers = [mae_ci_lower, preterm_mae_ci_lower, term_mae_ci_lower]
    mae_ci_uppers = [mae_ci_upper, preterm_mae_ci_upper, term_mae_ci_upper]

    def _yerr_or_none(mean, lo, hi):
        if np.isfinite(lo) and np.isfinite(hi):
            return [[mean - lo], [hi - mean]]
        return None  # draw bar without error bars

    # MAE
    bars1 = []
    for cat, mean, lo, hi in zip(categories, mae_means, mae_ci_lowers, mae_ci_uppers):
        if np.isfinite(mean):
            yerr = _yerr_or_none(mean, lo, hi)
            b = ax1.bar([cat], [mean], yerr=yerr, capsize=5,
                        color=colors[cat], alpha=0.8, edgecolor='black', linewidth=1)
            bars1.extend(b)
            ax1.text(b[0].get_x() + b[0].get_width()/2., b[0].get_height() + 0.01,
                     (f'{mean:.2f}\n[{lo:.2f}, {hi:.2f}]' if np.isfinite(lo) and np.isfinite(hi) else f'{mean:.2f}'),
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_title('Mean Absolute Error (MAE) with 95% CI', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('MAE (weeks)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Population Group', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    # RMSE plot
    rmse_means = [rmse_mean, preterm_rmse_mean, term_rmse_mean]
    rmse_ci_lowers = [rmse_ci_lower, preterm_rmse_ci_lower, term_rmse_ci_lower]
    rmse_ci_uppers = [rmse_ci_upper, preterm_rmse_ci_upper, term_rmse_ci_upper]

    # RMSE — same pattern with ax2 and rmse_* variables
    bars2 = []
    for cat, mean, lo, hi in zip(categories, rmse_means, rmse_ci_lowers, rmse_ci_uppers):
        if np.isfinite(mean):
            yerr = _yerr_or_none(mean, lo, hi)
            b = ax2.bar([cat], [mean], yerr=yerr, capsize=5,
                        color=colors[cat], alpha=0.8, edgecolor='black', linewidth=1)
            bars2.extend(b)
            ax2.text(b[0].get_x() + b[0].get_width()/2., b[0].get_height() + 0.01,
                     (f'{mean:.2f}\n[{lo:.2f}, {hi:.2f}]' if np.isfinite(lo) and np.isfinite(hi) else f'{mean:.2f}'),
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_title('Root Mean Square Error (RMSE) with 95% CI', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('RMSE (weeks)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Population Group', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[cat], alpha=0.8, edgecolor='black')
                      for cat in categories]
    fig.legend(legend_elements, categories, loc='upper center', bbox_to_anchor=(0.5, 0.02),
              ncol=3, fontsize=12, title='Population Groups', title_fontsize=13, framealpha=0.9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def _normalize_auc_df(df):
    df = df.copy()

    # Prefer FeatureSet; if missing, map from Model (for old CSVs)
    if "FeatureSet" not in df.columns and "Model" in df.columns:
        df["FeatureSet"] = df["Model"]

    # Normalize FeatureSet labels
    df["FeatureSet"] = (
        df["FeatureSet"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("Biomarkers", "Biomarker")
        .str.replace("biomarker", "Biomarker")
        .str.replace("clinical", "Clinical")
        .str.replace("combined", "Combined")
    )

    # Normalize Dataset to Title case (Heel/Cord)
    if "Dataset" in df.columns:
        df["Dataset"] = (
            df["Dataset"].astype(str).str.strip().str.lower()
            .map({"heel": "Heel", "cord": "Cord"})
            .fillna(df["Dataset"].astype(str).str.title())
        )

    # Keep only the columns we need
    need = ["Dataset", "FeatureSet", "AUC", "AUC_CI_Lower", "AUC_CI_Upper"]
    df = df.loc[:, [c for c in need if c in df.columns]]

    # If multiple rows exist for a (Dataset, FeatureSet) pair,
    # choose the best AUC or average — here we keep the max AUC.
    df = (
        df.sort_values("AUC", ascending=False)
          .drop_duplicates(subset=["Dataset", "FeatureSet"], keep="first")
    )

    # Enforce numeric types (guards against strings)
    for c in ["AUC", "AUC_CI_Lower", "AUC_CI_Upper"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# Build the final grid the plotter expects
def build_result_df(heel_df_raw, cord_df_raw):
    heel_df = _normalize_auc_df(heel_df_raw)
    cord_df = _normalize_auc_df(cord_df_raw)

    merged = pd.concat([heel_df, cord_df], ignore_index=True)

    # Create a canonical template to left-join into
    template = pd.MultiIndex.from_product(
        [["Heel", "Cord"], ["Clinical", "Biomarker", "Combined"]],
        names=["Dataset", "FeatureSet"]
    ).to_frame(index=False)

    # Left-join on (Dataset, FeatureSet)
    result_df = (
        template.merge(
            merged,
            how="left",
            on=["Dataset", "FeatureSet"],
            validate="one_to_one",
        )
        .rename(columns={"FeatureSet": "FeatureSet", "Dataset": "Dataset"})
    )

    # Optional: show any gaps
    missing = result_df[result_df["AUC"].isna()]
    if not missing.empty:
        print("[WARN][AUC_MISSING] Missing AUC for combos:\n", missing[["Dataset","FeatureSet"]])

    return result_df

def _prepare_summary_for_plot(summary_df, value_col, lower_col, upper_col):
    """Helper function to prepare summary data for plotting."""
    # Check if we have FeatureSet or Model column
    if 'FeatureSet' in summary_df.columns:
        model_col = 'FeatureSet'
    elif 'Model' in summary_df.columns:
        model_col = 'Model'
    else:
        raise ValueError("Neither 'FeatureSet' nor 'Model' column found in summary_df")

    # Normalize the input data to lowercase keys for robust joining
    df_norm = summary_df.copy()

    # Create lowercase join keys
    if 'Dataset' in df_norm.columns:
        df_norm['dataset_key'] = df_norm['Dataset'].astype(str).str.strip().str.lower()
    else:
        df_norm['dataset_key'] = np.nan

    if model_col in df_norm.columns:
        df_norm['featureset_key'] = df_norm[model_col].astype(str).str.strip().str.lower()
        # Normalize feature set names
        df_norm['featureset_key'] = df_norm['featureset_key'].map({
            'clinical': 'clinical',
            'biomarker': 'biomarker',
            'biomarkers': 'biomarker',
            'combined': 'combined'
        }).fillna(df_norm['featureset_key'])
    else:
        df_norm['featureset_key'] = np.nan

    # Drop rows with missing keys
    df_norm = df_norm.dropna(subset=['dataset_key', 'featureset_key'])

    # If multiple rows per (dataset,featureset), keep the max value
    if value_col in df_norm.columns:
        df_norm = (
            df_norm.sort_values(value_col, ascending=False)
            .drop_duplicates(subset=['dataset_key', 'featureset_key'], keep='first')
        )

    # Create canonical template (lowercase)
    template = pd.MultiIndex.from_product(
        [['heel', 'cord'], ['clinical', 'biomarker', 'combined']],
        names=['dataset_key', 'featureset_key']
    ).to_frame(index=False)

    # Left-join on lowercase keys
    result_df = template.merge(
        df_norm, how='left', on=['dataset_key', 'featureset_key'], validate='one_to_one'
    )

    # Anti-join: which template combos were missing?
    missing = template.merge(
        df_norm[['dataset_key', 'featureset_key']].drop_duplicates(),
        how='left', on=['dataset_key', 'featureset_key'], indicator=True
    ).query("_merge == 'left_only'")[['dataset_key', 'featureset_key']]
    if not missing.empty:
        print(f"[WARN][{value_col}_MISSING] No data found for these (dataset,featureset):\n", missing)

    # Create display columns (Title Case)
    result_df['Dataset'] = result_df['dataset_key'].map({'heel': 'heel', 'cord': 'cord'})
    result_df['FeatureSet'] = result_df['featureset_key'].map({
        'clinical': 'Clinical',
        'biomarker': 'Biomarker',
        'combined': 'Combined'
    })

    # Fill missing values with np.nan (not 0)
    result_df[value_col] = result_df[value_col].fillna(np.nan)
    result_df[lower_col] = result_df[lower_col].fillna(np.nan)
    result_df[upper_col] = result_df[upper_col].fillna(np.nan)

    return result_df


def plot_summary_auc_by_dataset_and_model(summary_df, filename="outputs/plots/summary_auc_by_dataset_and_model.png", model_type_label="Dataset"):
    """
    Create a summary plot showing AUC by dataset and model type.

    Args:
        summary_df (pd.DataFrame): Summary statistics DataFrame
        filename (str): Output file path
        model_type_label (str): Label for the x-axis (e.g., 'ElasticNet')
    """
    # Normalize the input data
    normalized_df = _normalize_auc_df(summary_df)

    # Prepare data for plotting using the old method for now (since we have single dataset)
    plot_df = _prepare_summary_for_plot(normalized_df, 'AUC', 'AUC_CI_Lower', 'AUC_CI_Upper')

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set up the plot with consistent color scheme
    datasets = ['heel', 'cord']
    models = ['Clinical', 'Biomarker', 'Combined']

    # Define consistent color palette
    colors = {
        'Clinical': '#2E86AB',    # Blue
        'Biomarker': '#A23B72',   # Purple
        'Combined': '#F18F01',    # Orange
    }

    n_datasets = len(datasets)
    n_models = len(models)
    x = np.arange(n_datasets)
    width = 0.25

    # Create bars for each model
    # Determine which column to use for model/feature set
    if 'FeatureSet' in plot_df.columns:
        model_col = 'FeatureSet'
    elif 'Model' in plot_df.columns:
        model_col = 'Model'
    else:
        raise ValueError("Neither 'FeatureSet' nor 'Model' column found in plot_df")

    for i, model in enumerate(models):
        auc_vals = []
        auc_errs = []
        for dataset in datasets:
            row = plot_df[(plot_df['Dataset'] == dataset) & (plot_df[model_col] == model)]
            if not row.empty and not np.isnan(row['AUC'].values[0]):
                auc = row['AUC'].values[0]
                lower = row['AUC_CI_Lower'].values[0]
                upper = row['AUC_CI_Upper'].values[0]
                auc_vals.append(auc)
                auc_errs.append([[auc - lower], [upper - auc]])
            else:
                auc_vals.append(np.nan)
                auc_errs.append([[0], [0]])

        positions = x + (i - (n_models-1)/2) * width
        bars = ax.bar(positions, auc_vals, width, label=model, color=colors[model],
                     yerr=np.array(auc_errs).squeeze().T, capsize=5, alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels on bars
        for bar, value in zip(bars, auc_vals):
            if not np.isnan(value):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Customize the plot
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Area Under Curve (AUC)', fontsize=14, fontweight='bold')
    ax.set_title(f'AUC by Dataset and Model Type - {model_type_label}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in datasets], fontsize=12)

    # Add legend with better styling
    ax.legend(title='Model Type', title_fontsize=13, fontsize=12, loc='upper right', framealpha=0.9)

    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim(0, 1.1)

    # Add horizontal line at AUC = 0.5 (random classifier)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Random Classifier (AUC = 0.5)')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_summary_metric_by_dataset_and_model(summary_df, metric, filename):
    """
    Create a summary plot showing a specific metric by dataset and model type.

    Args:
        summary_df (pd.DataFrame): Summary statistics DataFrame
        metric (str): Metric to plot ('MAE' or 'RMSE')
        filename (str): Output file path
    """
    # Prepare data for plotting
    plot_df = _prepare_summary_for_plot(summary_df, metric, f'{metric}_CI_Lower', f'{metric}_CI_Upper')

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set up the plot
    x = np.arange(len(plot_df))
    width = 0.25

    # Create bars for each dataset
    datasets = ['heel', 'cord']
    colors = ['skyblue', 'lightcoral']

    for i, dataset in enumerate(datasets):
        dataset_data = plot_df[plot_df['Dataset'] == dataset]
        metric_values = dataset_data[metric].values
        metric_lower = dataset_data[f'{metric}_CI_Lower'].values
        metric_upper = dataset_data[f'{metric}_CI_Upper'].values

        # Calculate error bars
        yerr = np.vstack([metric_values - metric_lower, metric_upper - metric_values])

        # Create bars
        bars = ax.bar(x + i*width, metric_values, width, label=dataset.title(),
                     color=colors[i], yerr=yerr, capsize=5)

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)

    # Customize the plot
    ax.set_xlabel('Model Type')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} by Dataset and Model Type')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(['Clinical', 'Biomarker', 'Combined'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_summary_mae_by_dataset_and_model(summary_df, filename, model_type_label):
    """
    Create a summary plot showing MAE by dataset and model type.
    Args:
        summary_df (pd.DataFrame): Summary statistics DataFrame
        filename (str): Output file path
        model_type_label (str): Label for the x-axis (e.g., 'ElasticNet')
    """
    plot_df = _prepare_summary_for_plot(summary_df, 'MAE', 'MAE_CI_Lower', 'MAE_CI_Upper')
    datasets = ['heel', 'cord']

    # Define consistent color palette
    colors = {
        'Clinical': '#2E86AB',    # Blue
        'Biomarker': '#A23B72',   # Purple
        'Combined': '#F18F01',    # Orange
    }

    if model_type_label.lower() == 'stabl':
        models = ['Biomarker', 'Combined']
        model_colors = {model: colors[model] for model in models}
    else:
        models = ['Clinical', 'Biomarker', 'Combined']
        model_colors = colors

    n_datasets = len(datasets)
    n_models = len(models)
    x = np.arange(n_datasets)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, model in enumerate(models):
        mae_vals = []
        mae_errs = []
        for dataset in datasets:
            row = plot_df[(plot_df['Dataset'] == dataset) & (plot_df['FeatureSet'] == model)]
            if not row.empty:
                mae = row['MAE'].values[0]
                lower = row['MAE_CI_Lower'].values[0]
                upper = row['MAE_CI_Upper'].values[0]
                mae_vals.append(mae)
                mae_errs.append([[mae - lower], [upper - mae]])
            else:
                mae_vals.append(np.nan)
                mae_errs.append([[0], [0]])

        positions = x + (i - (n_models-1)/2) * width
        bars = ax.bar(positions, mae_vals, width, label=model, color=model_colors[model],
                     yerr=np.array(mae_errs).squeeze().T, capsize=5, alpha=0.8, edgecolor='black', linewidth=1)

        for bar, value in zip(bars, mae_vals):
            if not np.isnan(value):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
    ax.set_title(f'MAE by Dataset and Model Type - {model_type_label}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in datasets], fontsize=12)
    ax.legend(title='Model Type', title_fontsize=13, fontsize=12, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_summary_rmse_by_dataset_and_model(summary_df, filename, model_type_label):
    """
    Create a summary plot showing RMSE by dataset and model type.
    Args:
        summary_df (pd.DataFrame): Summary statistics DataFrame
        filename (str): Output file path
        model_type_label (str): Label for the x-axis (e.g., 'ElasticNet')
    """
    plot_df = _prepare_summary_for_plot(summary_df, 'RMSE', 'RMSE_CI_Lower', 'RMSE_CI_Upper')
    datasets = ['heel', 'cord']

    # Define consistent color palette
    colors = {
        'Clinical': '#2E86AB',    # Blue
        'Biomarker': '#A23B72',   # Purple
        'Combined': '#F18F01',    # Orange
    }

    if model_type_label.lower() == 'stabl':
        models = ['Biomarker', 'Combined']
        model_colors = {model: colors[model] for model in models}
    else:
        models = ['Clinical', 'Biomarker', 'Combined']
        model_colors = colors

    n_datasets = len(datasets)
    n_models = len(models)
    x = np.arange(n_datasets)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, model in enumerate(models):
        rmse_vals = []
        rmse_errs = []
        for dataset in datasets:
            row = plot_df[(plot_df['Dataset'] == dataset) & (plot_df['FeatureSet'] == model)]
            if not row.empty:
                rmse = row['RMSE'].values[0]
                lower = row['RMSE_CI_Lower'].values[0]
                upper = row['RMSE_CI_Upper'].values[0]
                rmse_vals.append(rmse)
                rmse_errs.append([[rmse - lower], [upper - rmse]])
            else:
                rmse_vals.append(np.nan)
                rmse_errs.append([[0], [0]])

        positions = x + (i - (n_models-1)/2) * width
        bars = ax.bar(positions, rmse_vals, width, label=model, color=model_colors[model],
                     yerr=np.array(rmse_errs).squeeze().T, capsize=5, alpha=0.8, edgecolor='black', linewidth=1)

        for bar, value in zip(bars, rmse_vals):
            if not np.isnan(value):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Root Mean Square Error (RMSE)', fontsize=14, fontweight='bold')
    ax.set_title(f'RMSE by Dataset and Model Type - {model_type_label}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in datasets], fontsize=12)
    ax.legend(title='Model Type', title_fontsize=13, fontsize=12, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_true_vs_predicted_scatter(all_preds_df, filename="outputs/plots/true_vs_predicted_scatter.png", target_type="gestational_age"):
    """
    Create a scatter plot of true vs predicted values with proper color coding for Clinical, Biomarker, and Combined models.

    Args:
        all_preds_df (pd.DataFrame): DataFrame with true and predicted values
        filename (str): Output file path
        target_type (str): Target type ('gestational_age' or 'birth_weight')
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define color mapping for model types with consistent colors
    model_color = {
        'Clinical': '#2E86AB',      # Blue
        'Biomarker': '#A23B72',     # Purple
        'Combined': '#F18F01',      # Orange
    }

    # Define marker mapping for datasets
    dataset_marker = {
        'heel': 'o',
        'cord': 's',
    }

    # Define model type mapping for cleaner labels
    modeltype_name_map = {
        'elasticnet': 'ElasticNet',
        'elasticnet': 'ElasticNet',
        'stabl': 'STABL'
    }

    # Determine grouping columns
    if 'ModelType' in all_preds_df.columns:
        group_cols = ['Model', 'ModelType', 'Dataset']
    else:
        group_cols = ['Model', 'Dataset']

    # Create legend handles and labels
    legend_handles = []
    legend_labels = []

    for i, (name, group) in enumerate(all_preds_df.groupby(group_cols)):
        if 'ModelType' in all_preds_df.columns:
            model, model_type, dataset = name
            model_display = model
            modeltype_display = modeltype_name_map.get(model_type.lower(), model_type.title())
            label = f'{model_display} ({modeltype_display}, {dataset.title()})'
            color = model_color.get(model, '#666666')
            marker = dataset_marker.get(dataset.lower(), 'o')
        else:
            model, dataset = name
            model_display = model
            label = f'{model_display} ({dataset.title()})'
            color = model_color.get(model, '#666666')
            marker = dataset_marker.get(dataset.lower(), 'o')

        # Determine column names based on target type
        if target_type == 'birth_weight':
            true_col = 'True_BW'
            pred_col = 'Pred_BW'
            xlabel = 'True Birth Weight (kg)'
            ylabel = 'Predicted Birth Weight (kg)'
            title = 'True vs Predicted Birth Weight'
        else:  # gestational_age
            true_col = 'True_GA'
            pred_col = 'Pred_GA'
            xlabel = 'True Gestational Age (weeks)'
            ylabel = 'Predicted Gestational Age (weeks)'
            title = 'True vs Predicted Gestational Age'

        # Create scatter plot
        scatter = ax.scatter(group[true_col], group[pred_col],
                           alpha=0.7, s=50, c=color, marker=marker,
                           label=label, edgecolors='white', linewidth=0.5)

        # Store legend handle
        legend_handles.append(scatter)

    # Add diagonal line for perfect prediction
    min_val = min(all_preds_df[true_col].min(), all_preds_df[pred_col].min())
    max_val = max(all_preds_df[true_col].max(), all_preds_df[pred_col].max())
    diagonal_line = ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='Perfect Prediction')
    legend_handles.append(diagonal_line[0])

    # Customize plot
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Add legend with better positioning and styling
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left',
             fontsize=11, framealpha=0.9, title='Models', title_fontsize=12)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Add R² value as text annotation (guarded)
    clean = all_preds_df[[true_col, pred_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) > 1 and clean[true_col].nunique() > 1 and clean[pred_col].nunique() > 1:
        from scipy.stats import pearsonr
        r, _ = pearsonr(clean[true_col], clean[pred_col])
        if np.isfinite(r):
            ax.text(0.05, 0.95, f'R² = {r**2:.3f}', transform=ax.transAxes,
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7),
                    verticalalignment='top')
        else:
            pass
    else:
        pass

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_biomarker_preterm_term_scatter(term_freq, preterm_freq, feature_names, filename, top_n=10):
    """
    Plot scatter plot of biomarker selection frequency (term vs preterm).

    Args:
        term_freq (array): Term frequencies
        preterm_freq (array): Preterm frequencies
        feature_names (list): List of feature names
        filename (str): Output file path
        top_n (int): Number of top features to label
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create DataFrame for easy manipulation
    df = pd.DataFrame({
        'feature': feature_names,
        'term_freq': term_freq,
        'preterm_freq': preterm_freq
    })

    # Calculate total frequency for ranking
    df['total_freq'] = df['term_freq'] + df['preterm_freq']

    # Sort by total frequency and get top N
    df_sorted = df.sort_values('total_freq', ascending=False)
    top_features = df_sorted.head(top_n)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot all features as gray dots
    ax.scatter(df['term_freq'], df['preterm_freq'], alpha=0.6, color='lightgray', s=30, label='All features')

    # Plot top features as red dots with labels
    ax.scatter(top_features['term_freq'], top_features['preterm_freq'],
              color='red', s=60, zorder=5, label=f'Top {top_n} features')

    # Add labels for top features
    for _, row in top_features.iterrows():
        ax.annotate(row['feature'],
                   (row['term_freq'], row['preterm_freq']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, alpha=0.8)

    # Add diagonal line for equal frequency
    max_val = max(df['term_freq'].max(), df['preterm_freq'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal frequency')

    # Customize plot
    ax.set_xlabel('Frequency in Term Pregnancies')
    ax.set_ylabel('Frequency in Preterm Pregnancies')
    ax.set_title('Biomarker Selection Frequency: Term vs Preterm')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_biomarker_frequency_heel_vs_cord(model_results, model_type, filename, target_type='gestational_age'):
    """
    Create a scatter plot comparing biomarker frequencies between heel and cord datasets.
    Args:
        model_results (dict): Dictionary containing model results
        model_type (str): Type of model ('elasticnet')
        filename (str): Output filename for the plot
        target_type (str): Target type ('gestational_age' or 'birth_weight')
    Note:
        Automatically labels the top 50% of biomarkers by total frequency (heel + cord).
    """
    # Extract biomarker selection frequencies (proportion of runs with non-zero coef) for heel and cord datasets
    eps = 1e-12
    heel_select_count = {}
    heel_total_count = {}
    cord_select_count = {}
    cord_total_count = {}
    for key, result in model_results.items():
        if 'heel' in key and model_type in key:
            if 'all_coefficients' in result and result['all_coefficients'] is not None:
                all_coefs = result['all_coefficients']
                if len(all_coefs) > 0:  # Check if all_coefs is not empty
                    feature_names = result.get('feature_names', [f'Feature_{i}' for i in range(len(all_coefs[0]))])
                    for run_coefs in all_coefs:
                        for i, (name, val) in enumerate(zip(feature_names, run_coefs)):
                            heel_total_count[name] = heel_total_count.get(name, 0) + 1
                            if np.abs(val) > eps:
                                heel_select_count[name] = heel_select_count.get(name, 0) + 1
        elif 'cord' in key and model_type in key:
            if 'all_coefficients' in result and result['all_coefficients'] is not None:
                all_coefs = result['all_coefficients']
                if len(all_coefs) > 0:  # Check if all_coefs is not empty
                    feature_names = result.get('feature_names', [f'Feature_{i}' for i in range(len(all_coefs[0]))])
                    for run_coefs in all_coefs:
                        for i, (name, val) in enumerate(zip(feature_names, run_coefs)):
                            cord_total_count[name] = cord_total_count.get(name, 0) + 1
                            if np.abs(val) > eps:
                                cord_select_count[name] = cord_select_count.get(name, 0) + 1
    heel_mean = {name: (heel_select_count.get(name, 0) / heel_total_count[name]) for name in heel_total_count if heel_total_count[name] > 0}
    cord_mean = {name: (cord_select_count.get(name, 0) / cord_total_count[name]) for name in cord_total_count if cord_total_count[name] > 0}
    common_biomarkers = set(heel_mean.keys()) & set(cord_mean.keys())
    if not common_biomarkers:
        print(f"No common biomarkers found for {model_type}")
        return
    plot_data = []
    for biomarker in common_biomarkers:
        plot_data.append({
            'Biomarker': biomarker,
            'Heel_Frequency': heel_mean[biomarker],
            'Cord_Frequency': cord_mean[biomarker]
        })
    df = pd.DataFrame(plot_data)
    df['Total_Frequency'] = df['Heel_Frequency'] + df['Cord_Frequency']
    df = df.sort_values('Total_Frequency', ascending=False)

    # A. Increase figure size
    fig, ax = plt.subplots(figsize=(18, 14))

    scatter = ax.scatter(df['Heel_Frequency'], df['Cord_Frequency'],
                        c=df['Total_Frequency'], cmap='viridis', s=60, alpha=0.7,
                        edgecolors='black', linewidth=0.5, zorder=3)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Total Selection Frequency (Heel + Cord)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    threshold = 0.5
    high_freq = df[df['Total_Frequency'] > threshold]
    ax.scatter(high_freq['Heel_Frequency'], high_freq['Cord_Frequency'],
               c=high_freq['Total_Frequency'], cmap='viridis', s=120, alpha=0.9,
               edgecolors='red', linewidth=2, zorder=5, label=f'Frequency > {threshold*100:.0f}%')

    # D. Label the top 50% of biomarkers by total frequency
    # Calculate how many biomarkers represent the top 50%
    total_biomarkers = len(df)
    top_50_percent_count = max(1, int(total_biomarkers * 0.5))  # At least 1 biomarker

    # Get the top 50% biomarkers by total frequency
    to_label = df.head(top_50_percent_count)

    # Highlight labeled points for clarity
    ax.scatter(
        to_label['Heel_Frequency'], to_label['Cord_Frequency'],
        s=140, facecolors='none', edgecolors='red', linewidths=1.5, zorder=6
    )

    # Label with leader lines pointing to each dot; alternate offsets to reduce overlap
    offsets = [(14, 10), (14, -10), (-14, 10), (-14, -10)]
    for idx, (_, row) in enumerate(to_label.iterrows()):
        dx, dy = offsets[idx % len(offsets)]
        ha = 'left' if dx >= 0 else 'right'
        va = 'bottom' if dy >= 0 else 'top'
        ax.annotate(
            row['Biomarker'],
            xy=(row['Heel_Frequency'], row['Cord_Frequency']),
            xytext=(dx, dy), textcoords='offset points',
            fontsize=9, color='red', alpha=0.95,
            ha=ha, va=va,
            arrowprops=dict(arrowstyle='-', color='red', lw=0.8, shrinkA=0, shrinkB=0),
            zorder=7
        )

    # Fix scale to [0, 1]
    ax.plot([0, 1.0], [0, 1.0], '--', color='red', alpha=0.8, linewidth=2,
            label='Equal frequency line')
    mid_val = 0.5
    ax.axhline(y=mid_val, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(x=mid_val, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.set_xlabel('Selection Frequency (Heel)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Selection Frequency (Cord)', fontsize=14, fontweight='bold')
    target_title = target_type.replace('_', ' ').title()
    ax.set_title(f'Biomarker Frequency Comparison: Heel vs Cord\n{model_type.upper()} - {target_title}',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    correlation = df['Heel_Frequency'].corr(df['Cord_Frequency'])
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes,
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7),
            verticalalignment='top')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_summary_auc_combined_heel_cord(
    heel_df,
    cord_df,
    filename="outputs/plots/summary_auc_by_dataset_and_model_combined_heel_cord.png",
    model_type_label="ElasticNet (CV)",
    metric_label="AUC",
):
    """
    Create a combined summary plot showing AUC by dataset (Heel, Cord) and feature set.
    NOTE: This version expects incoming summary frames to have either:
        - FeatureSet column with values in {"Clinical","Biomarker","Combined"}, or
        - Model column that actually contains those same values (older CSVs).
    We normalize both, prefer FeatureSet, and fall back to Model if needed.
    Includes robust debug prints to help diagnose empty/NaN plots.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Defensive copies
    heel_df = heel_df.copy()
    cord_df = cord_df.copy()

    # === Normalization helpers ===
    def _ensure_featureset(df):
        # If FeatureSet missing but Model present, treat Model as FeatureSet (old CSV pattern)
        if "FeatureSet" not in df.columns and "Model" in df.columns:
            df["FeatureSet"] = df["Model"]

        # Normalize capitalization and plural forms
        if "FeatureSet" in df.columns:
            df["FeatureSet"] = (
                df["FeatureSet"]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.replace("Biomarkers", "Biomarker")
                .str.replace("biomarker", "Biomarker")
                .str.replace("clinical", "Clinical")
                .str.replace("combined", "Combined")
            )

        # Normalize Dataset capitalization if present
        if "Dataset" in df.columns:
            df["Dataset"] = (
                df["Dataset"]
                .astype(str).str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
            # Accept Heel/Cord or heel/cord or mixed
            df["Dataset"] = df["Dataset"].str.lower().map({"heel": "Heel", "cord": "Cord"}).fillna(
                df["Dataset"].str.title()
            )

        return df

    heel_df = _ensure_featureset(heel_df)
    cord_df = _ensure_featureset(cord_df)

    # Add dataset tag if missing (some sources might not include it)
    if "Dataset" not in heel_df.columns:
        heel_df["Dataset"] = "Heel"
    if "Dataset" not in cord_df.columns:
        cord_df["Dataset"] = "Cord"

    # Keep only the columns we need; tolerate both old and new schemas
    needed_cols = ["Dataset", "FeatureSet", metric_label, f"{metric_label}_CI_Lower", f"{metric_label}_CI_Upper"]
    present_cols = [c for c in needed_cols if c in heel_df.columns]
    if "FeatureSet" not in present_cols:
        print("[WARN] plot_summary_auc_combined_heel_cord: FeatureSet column still missing in heel_df after normalization.")
    present_cols_cord = [c for c in needed_cols if c in cord_df.columns]
    if "FeatureSet" not in present_cols_cord:
        print("[WARN] plot_summary_auc_combined_heel_cord: FeatureSet column still missing in cord_df after normalization.")

    # Concatenate and then drop duplicates by (Dataset, FeatureSet) — keep last
    plot_df = pd.concat([heel_df[heel_df.columns.intersection(needed_cols)],
                         cord_df[cord_df.columns.intersection(needed_cols)]],
                        ignore_index=True)

    # De-duplicate rows if multiple summaries accidentally exist (fixes duplication bug)
    if {"Dataset", "FeatureSet"}.issubset(plot_df.columns):
        before = len(plot_df)
        plot_df = plot_df.drop_duplicates(subset=["Dataset", "FeatureSet"], keep="last")
        after = len(plot_df)
        if after < before:
            print(f"[INFO] De-duplicated AUC rows: {before} -> {after} by (Dataset, FeatureSet)")

    # Validate we have the expected categories
    expected_sets = {"Clinical", "Biomarker", "Combined"}
    observed_sets = set(plot_df.get("FeatureSet", pd.Series(dtype=str)).dropna().unique())
    missing_sets = expected_sets - observed_sets
    if missing_sets:
        print(f"[WARN] Missing FeatureSet categories in AUC summary: {sorted(missing_sets)} "
              f"(observed={sorted(observed_sets)})")

    # Prepare plotting
    models = ["Clinical", "Biomarker", "Combined"]  # these are feature sets
    colors = {
        "Clinical": "#2E86AB",    # Blue
        "Biomarker": "#A23B72",   # Purple
        "Combined": "#F18F01",    # Orange
    }
    datasets = ["Heel", "Cord"]
    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 8))

    any_bar_drawn = False
    for i, model in enumerate(models):
        auc_vals = []
        auc_errs = []
        for dataset in datasets:
            try:
                mask = (plot_df["Dataset"] == dataset) & (plot_df["FeatureSet"] == model)
                row = plot_df.loc[mask]
                if not row.empty:
                    auc = float(row.iloc[-1][metric_label])
                    lower = float(row.iloc[-1][f"{metric_label}_CI_Lower"])
                    upper = float(row.iloc[-1][f"{metric_label}_CI_Upper"])
                    if np.isnan(auc) or np.isnan(lower) or np.isnan(upper):
                        print(f"[WARN] NaN metric in row for dataset={dataset}, feature={model}: {row.iloc[-1].to_dict()}")
                        auc_vals.append(np.nan)
                        auc_errs.append([[0.0], [0.0]])
                    else:
                        auc_vals.append(auc)
                        auc_errs.append([[auc - lower], [upper - auc]])
                        any_bar_drawn = True
                else:
                    print(f"[WARN] No AUC row found for dataset={dataset}, feature={model}")
                    auc_vals.append(np.nan)
                    auc_errs.append([[0.0], [0.0]])
            except Exception as e:
                print(f"[ERROR] Selecting row for dataset={dataset}, feature={model} failed: {e}")
                auc_vals.append(np.nan)
                auc_errs.append([[0.0], [0.0]])

        positions = x + (i - (len(models) - 1) / 2) * width
        try:
            yerr = np.array(auc_errs, dtype=float).squeeze().T  # shape (2, n_bars)
            bars = ax.bar(
                positions,
                auc_vals,
                width,
                label=model,
                color=colors.get(model, None),
                yerr=yerr,
                capsize=5,
                alpha=0.8,
                edgecolor="black",
                linewidth=1,
            )
            for bar, value in zip(bars, auc_vals):
                if value == value:  # not NaN
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        fontweight="bold",
                    )
        except Exception as e:
            print(f"[ERROR] Plotting bars for feature set {model} failed: {e}")

    if not any_bar_drawn:
        print("[ERROR] No bars were drawn. Check that your CSVs contain rows with FeatureSet ∈ {Clinical, Biomarker, Combined}.")
        print("[HINT] If your CSVs only have a 'Model' column with those values, this function should have auto-populated FeatureSet from Model.")

    ax.set_xlabel("Dataset", fontsize=14, fontweight="bold")
    ax.set_ylabel(f"{metric_label} (with CI)", fontsize=14, fontweight="bold")
    ax.set_title(
        f"{metric_label} by Dataset and Feature Set - {model_type_label} (heel_all vs cord_all)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([d for d in datasets], fontsize=12)
    ax.legend(title="Feature Set", title_fontsize=13, fontsize=12, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)

    if metric_label.upper() == "AUC":
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, linewidth=2, label="Random Classifier (AUC = 0.5)")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    try:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved AUC plot to {filename}")
    except Exception as e:
        print(f"[ERROR] Saving AUC plot failed: {e}")
    finally:
        plt.close()


def plot_auc_classification_comparison(
    all_results: dict,
    classification_type: str = "preterm",
    save_path: str = "outputs/run_all/plots/auc_classification_comparison.png",
    title: str = "AUC Scores for Classification Performance"
) -> None:
    """
    Create AUC bar plots for classification tasks (preterm or SGA).

    Args:
        all_results: Dictionary containing results from all model runs
        classification_type: "preterm" or "sga"
        save_path: Path to save the plot
        title: Plot title
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Extract AUC data from all_results
    auc_data = []

    # Filter for data option 1 (both_samples) only
    for result_key, results in all_results.items():
        if 'both_samples' not in result_key:
            continue

        # Parse the result key to extract components
        parts = result_key.split('_')
        if len(parts) >= 4:
            data_option_label = parts[0] + '_' + parts[1]  # both_samples
            dataset_type = parts[2]  # heel or cord
            model_type = parts[3]    # elasticnet_cv
            model_name = parts[4]    # Clinical, Biomarker, or Combined

            # Get AUC data
            if 'summary' in results and 'auc_mean' in results['summary']:
                auc_mean = results['summary']['auc_mean']
                auc_ci_lower = results['summary']['auc_ci_lower']
                auc_ci_upper = results['summary']['auc_ci_upper']

                if not np.isnan(auc_mean):
                    auc_data.append({
                        'Dataset': dataset_type.title(),
                        'FeatureSet': model_name,
                        'ModelType': model_type.replace('_cv', '').title(),
                        'AUC': auc_mean,
                        'AUC_CI_Lower': auc_ci_lower,
                        'AUC_CI_Upper': auc_ci_upper
                    })

    if not auc_data:
        print(f"[WARN] No AUC data found for {classification_type} classification")
        return

    # Create DataFrame
    df = pd.DataFrame(auc_data)

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors for model types
    colors = {
        'ElasticNet': '#1f77b4',
        'Elasticnet': '#ff7f0e'
    }

    # Define positions for bars
    datasets = ['Heel', 'Cord']
    featuresets = ['Clinical', 'Biomarker', 'Combined']
    model_types = ['ElasticNet']

    # Calculate bar positions
    n_datasets = len(datasets)
    n_featuresets = len(featuresets)
    n_model_types = len(model_types)

    bar_width = 0.35
    x_positions = np.arange(n_datasets * n_featuresets)

    # Plot bars
    for i, dataset in enumerate(datasets):
        for j, featureset in enumerate(featuresets):
            pos_idx = i * n_featuresets + j

            for k, model_type in enumerate(model_types):
                # Filter data for this combination
                subset = df[
                    (df['Dataset'] == dataset) &
                    (df['FeatureSet'] == featureset) &
                    (df['ModelType'] == model_type)
                ]

                if not subset.empty:
                    auc_val = subset['AUC'].iloc[0]
                    ci_lower = subset['AUC_CI_Lower'].iloc[0]
                    ci_upper = subset['AUC_CI_Upper'].iloc[0]

                    # Calculate error bar values
                    yerr_lower = auc_val - ci_lower if not np.isnan(ci_lower) else 0
                    yerr_upper = ci_upper - auc_val if not np.isnan(ci_upper) else 0

                    # Position bars side by side for each model type
                    x_pos = pos_idx + (k - 0.5) * bar_width / n_model_types

                    bars = ax.bar(
                        x_pos, auc_val,
                        width=bar_width / n_model_types,
                        yerr=[[yerr_lower], [yerr_upper]],
                        capsize=3,
                        color=colors[model_type],
                        alpha=0.8,
                        edgecolor='black',
                        linewidth=0.5,
                        label=model_type if i == 0 and j == 0 else ""
                    )

                    # Add value labels on top of bars
                    ax.text(x_pos, auc_val + yerr_upper + 0.01, f'{auc_val:.3f}',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Customize plot
    ax.set_xlabel('Dataset × Feature Set', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{title} - {classification_type.title()} Classification',
                fontsize=14, fontweight='bold', pad=20)

    # Set x-axis labels
    x_labels = []
    for dataset in datasets:
        for featureset in featuresets:
            x_labels.append(f'{dataset}\n{featureset}')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=10)

    # Add horizontal line at AUC = 0.5 (random classifier)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7,
               label='Random Classifier (AUC=0.5)')

    # Set y-axis limits
    ax.set_ylim(0.4, 1.0)

    # Add legend
    ax.legend(loc='upper left', fontsize=10)

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')

    # Adjust layout and save
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"AUC classification plot saved: {save_path}")


def plot_auc_sga_classification(
    all_results: dict,
    save_path: str = "outputs/run_all/plots/auc_sga_classification.png",
    title: str = "AUC Scores for SGA Classification Performance"
) -> None:
    """Create AUC bar plot for SGA classification."""
    plot_auc_classification_comparison(
        all_results=all_results,
        classification_type="sga",
        save_path=save_path,
        title=title
    )


def plot_auc_preterm_classification(
    all_results: dict,
    save_path: str = "outputs/run_all/plots/auc_preterm_classification.png",
    title: str = "AUC Scores for Preterm Classification Performance"
) -> None:
    """Create AUC bar plot for preterm classification."""
    plot_auc_classification_comparison(
        all_results=all_results,
        classification_type="preterm",
        save_path=save_path,
        title=title
    )
