import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
import numpy as np
import os
import pickle

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def save_plot(fig, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename)

def plot_roc_curve(fpr, tpr, auc, filename="roc_curve.png"):
    """Plot ROC curve."""
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    save_plot(fig, filename)

def plot_feature_frequency(feature_names, frequencies, filename="feature_frequency.png"):
    """Plot frequency of selected features across model iterations."""
    # Increase figure size to accommodate all biomarkers
    fig, ax = plt.subplots(figsize=(12, max(8, len(feature_names) * 0.15)))
    
    # Use matplotlib's horizontal bar plot to ensure all features are shown
    # including those with 0 frequency
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, frequencies)
    
    # Set y-axis ticks and labels to feature names
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    
    ax.set_xlabel("Frequency of Selection")
    ax.set_ylabel("Biomarker")
    ax.set_title("Feature Selection Frequency")
    
    # Add grid for better readability
    ax.grid(axis='x', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    save_plot(fig, filename)

def plot_predictions(actual, predicted, filename="pred_vs_actual.png", error_bars=None):
    """Plot predicted probability vs actual outcome (e.g., preterm vs term)."""
    fig, ax = plt.subplots()
    sns.barplot(x=["Preterm", "Term"], y=predicted, yerr=error_bars, ax=ax)
    ax.set_title("Predicted vs. Actual Preterm Births")
    ax.set_ylabel("Predicted Probability")
    save_plot(fig, filename)

def count_high_weight_biomarkers(coef_list, feature_names, threshold=0.1, abs_val=True):
    # Initialize frequency counter
    freq = {feature: 0 for feature in feature_names}

    for coef in coef_list:
        if isinstance(coef, pd.Series):
            coef = coef.values

        for i, weight in enumerate(coef):
            value = abs(weight) if abs_val else weight
            if value >= threshold:
                freq[feature_names[i]] += 1
    
    # Return a pandas Series with all features, including those with 0 frequency
    return pd.Series(freq).reindex(feature_names, fill_value=0)

    # Convert to DataFrame for easy visualization
    freq_df = pd.DataFrame({
        'biomarker': list(freq.keys()),
        'frequency': list(freq.values())
    }).sort_values(by='frequency', ascending=False)

    return freq_df

def save_all_as_pickle(obj, filename="full_model_output.pkl"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def plot_mae(maes, filename="outputs/plots/mae_over_runs.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(range(1, len(maes) + 1), maes, marker='o')
    ax.set_title("MAE Over Runs")
    ax.set_xlabel("Run")
    ax.set_ylabel("MAE")
    fig.tight_layout()
    fig.savefig(filename)

def plot_rmse(rmses, filename="outputs/plots/rmse_over_runs.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(range(1, len(rmses) + 1), rmses, marker='o', color='orange')
    ax.set_title("RMSE Over Runs")
    ax.set_xlabel("Run")
    ax.set_ylabel("RMSE")
    fig.tight_layout()
    fig.savefig(filename)

def create_metrics_table(maes, rmses, preterm_metrics_list, term_metrics_list, filename="outputs/tables/performance_metrics.csv"):
    """Create a comprehensive table with MAE/RMSE metrics and confidence intervals"""
    from src.metrics import compute_confidence_interval
    
    # Compute confidence intervals for overall metrics
    mae_mean, mae_ci_lower, mae_ci_upper = compute_confidence_interval(maes)
    rmse_mean, rmse_ci_lower, rmse_ci_upper = compute_confidence_interval(rmses)
    
    # Extract preterm and term metrics
    preterm_maes = [m['mae'] for m in preterm_metrics_list if not np.isnan(m['mae'])]
    preterm_rmses = [m['rmse'] for m in preterm_metrics_list if not np.isnan(m['rmse'])]
    term_maes = [m['mae'] for m in term_metrics_list if not np.isnan(m['mae'])]
    term_rmses = [m['rmse'] for m in term_metrics_list if not np.isnan(m['rmse'])]
    
    # Compute confidence intervals for preterm and term
    preterm_mae_mean, preterm_mae_ci_lower, preterm_mae_ci_upper = compute_confidence_interval(preterm_maes)
    preterm_rmse_mean, preterm_rmse_ci_lower, preterm_rmse_ci_upper = compute_confidence_interval(preterm_rmses)
    term_mae_mean, term_mae_ci_lower, term_mae_ci_upper = compute_confidence_interval(term_maes)
    term_rmse_mean, term_rmse_ci_lower, term_rmse_ci_upper = compute_confidence_interval(term_rmses)
    
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
    """Create bar plots for MAE/RMSE with confidence intervals"""
    from src.metrics import compute_confidence_interval
    
    # Compute confidence intervals
    mae_mean, mae_ci_lower, mae_ci_upper = compute_confidence_interval(maes)
    rmse_mean, rmse_ci_lower, rmse_ci_upper = compute_confidence_interval(rmses)
    
    # Extract preterm and term metrics
    preterm_maes = [m['mae'] for m in preterm_metrics_list if not np.isnan(m['mae'])]
    preterm_rmses = [m['rmse'] for m in preterm_metrics_list if not np.isnan(m['rmse'])]
    term_maes = [m['mae'] for m in term_metrics_list if not np.isnan(m['mae'])]
    term_rmses = [m['rmse'] for m in term_metrics_list if not np.isnan(m['rmse'])]
    
    preterm_mae_mean, preterm_mae_ci_lower, preterm_mae_ci_upper = compute_confidence_interval(preterm_maes)
    preterm_rmse_mean, preterm_rmse_ci_lower, preterm_rmse_ci_upper = compute_confidence_interval(preterm_rmses)
    term_mae_mean, term_mae_ci_lower, term_mae_ci_upper = compute_confidence_interval(term_maes)
    term_rmse_mean, term_rmse_ci_lower, term_rmse_ci_upper = compute_confidence_interval(term_rmses)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MAE plot
    categories = ['Overall', 'Preterm', 'Term']
    mae_means = [mae_mean, preterm_mae_mean, term_mae_mean]
    mae_ci_lowers = [mae_ci_lower, preterm_mae_ci_lower, term_mae_ci_lower]
    mae_ci_uppers = [mae_ci_upper, preterm_mae_ci_upper, term_mae_ci_upper]
    mae_errors = [mae_ci_upper - mae_ci_lower, preterm_mae_ci_upper - preterm_mae_ci_lower, term_mae_ci_upper - term_mae_ci_lower]
    
    bars1 = ax1.bar(categories, mae_means, yerr=mae_errors, capsize=5, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('Mean Absolute Error (MAE) with 95% CI')
    ax1.set_ylabel('MAE (weeks)')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars1, mae_means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom')
    
    # RMSE plot
    rmse_means = [rmse_mean, preterm_rmse_mean, term_rmse_mean]
    rmse_ci_lowers = [rmse_ci_lower, preterm_rmse_ci_lower, term_rmse_ci_lower]
    rmse_ci_uppers = [rmse_ci_upper, preterm_rmse_ci_upper, term_rmse_ci_upper]
    rmse_errors = [rmse_ci_upper - rmse_ci_lower, preterm_rmse_ci_upper - preterm_rmse_ci_lower, term_rmse_ci_upper - term_rmse_ci_lower]
    
    bars2 = ax2.bar(categories, rmse_means, yerr=rmse_errors, capsize=5, color=['orange', 'salmon', 'lightblue'])
    ax2.set_title('Root Mean Squared Error (RMSE) with 95% CI')
    ax2.set_ylabel('RMSE (weeks)')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars2, rmse_means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_plot(fig, filename)
    
    return fig