"""Plotting utilities for classification and regression results.

Generates volcano plots (univariate analysis), AUC curves, feature selection
summaries, hyperparameter distributions, and prediction scatter plots for both
training cohort and external validation cohort results.

Usage:
    python src/plotting.py            # Training cohort plots
    python src/plotting.py --validate # External validation cohort plots
"""

import argparse
import ast
import os
from collections import defaultdict
from pathlib import Path
import sys
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from adjustText import adjust_text

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import PRETERM_CUTOFF
from src.feature_sets import (
    BIOMARKER_FEATURES,
    EXTERNAL_VALIDATION_CSV,
    filter_validation_by_sample_type,
)

VALIDATION_SUBDIR = "validation_zimbabwe"


def _config_run_dir(
    outdir: Path,
    model_name: str,
    data_option: int,
    model_type: str,
    data_type: str,
    validation: bool = False,
) -> Path:
    """Return the output subdirectory for a given run configuration."""
    if data_option == 1:
        base = outdir / f"{model_name}__opt{data_option}__{model_type}__{data_type}"
    else:
        base = outdir / f"{model_name}__opt{data_option}__{model_type}"
    if validation:
        return base / VALIDATION_SUBDIR
    return base


def _plots_save_dir(outdir: Path, validation: bool) -> Path:
    """Return the directory where plots are saved, creating it if needed."""
    if validation:
        save_dir = _project_root / "outputs" / "plots" / VALIDATION_SUBDIR
    else:
        save_dir = outdir
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def _result_subdir_name(subdir: Path, validation: bool) -> Optional[str]:
    """Parent config directory name for a training or validation result folder."""
    if validation:
        if subdir.name != VALIDATION_SUBDIR or subdir.parent.name.count("__") < 2:
            return None
        return subdir.parent.name
    return subdir.name

def boxplot_irt_dataset_preterm_term() -> None:

    """
    Plot boxplots for IRT from the dataset, one box for preterm and one for term babies.
    """
    df = pd.read_csv("data/Bangladesh_children_with_both_samples.csv")
    preterm_irt = df[df["gestational_age_weeks"] < PRETERM_CUTOFF]["IRT"].dropna()
    term_irt = df[df["gestational_age_weeks"] >= PRETERM_CUTOFF]["IRT"].dropna()

    plt.figure(figsize=(10, 5))
    plt.boxplot([preterm_irt, term_irt], labels=["Preterm", "Term"], widths=0.5)
    plt.xlabel("Group")
    plt.ylabel("IRT")

    out_dir = _project_root / "outputs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "boxplot_irt_dataset_preterm_term.png")
    plt.close()

def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (same length as pvals)."""
    pvals = np.asarray(pvals, dtype=float)
    m = int(pvals.size)
    if m == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    ranks = np.arange(1, m + 1, dtype=float)
    bh = ranked * m / ranks
    bh_adj = np.minimum.accumulate(bh[::-1])[::-1]
    bh_adj = np.minimum(bh_adj, 1.0)
    out = np.empty(m)
    out[order] = bh_adj
    return out


def plot_average_auc(outdir: Path, validation: bool = False) -> None:
    """
    Scan all runs under outputs/classification/, compute average AUC per
    (model, data_option, feature_set, sample_source), and create two bar plots:
      - Data option 1
      - Data options 2 and 3

    When validation=True, reads auc_all.csv from each .../validation_zimbabwe/ folder.
    """
    FEATURE_SETS = ["clinical", "biomarker", "combined"]

    # Subdir names are either:
    #   - data_option 1:  {model_name}__opt1__{feature_set}__{sample_source}  (sample_source in {heel, cord})
    #   - data_option 2/3:{model_name}__opt{2,3}__{feature_set}              (no sample_source suffix)
    all_subdirs = list(outdir.glob("*"))
    entries = []
    for subdir in all_subdirs:
        try:
            config_name = _result_subdir_name(subdir, validation=False)
            if validation:
                if not subdir.is_dir():
                    continue
                val_dir = subdir / VALIDATION_SUBDIR
                config_name = _result_subdir_name(val_dir, validation=True)
                if config_name is None:
                    continue
                parts = config_name.split("__")
            else:
                if config_name is None:
                    continue
                parts = subdir.name.split("__")
            # Handle both 3-part and 4-part names
            if len(parts) == 4:
                model_name, opt_str, model_type, sample_source = parts
            elif len(parts) == 3:
                model_name, opt_str, model_type = parts
                sample_source = None
            else:
                continue
            if not opt_str.startswith("opt"):
                continue
            # Only plot ElasticNet results
            if model_name != "elasticnet_cv":
                continue
            data_option = int(opt_str.replace("opt", ""))

            # For data_option 1 we require an explicit sample_source ('heel' or 'cord')
            if data_option == 1:
                if sample_source not in {"heel", "cord"}:
                    continue
            else:
                # For data_option 2/3, we don't distinguish heel/cord here
                # but keep the column for completeness
                if sample_source is None:
                    sample_source = "all"
            if validation:
                auc_path = subdir / VALIDATION_SUBDIR / "auc_all.csv"
            else:
                auc_path = subdir / "auc_all.csv"
            if not auc_path.exists():
                continue
            auc_all = pd.read_csv(auc_path)
            auc_vals = auc_all["auc"].to_numpy()
            if auc_vals.size == 0:
                continue
            avg_auc = float(auc_vals.mean())
            std_auc = float(auc_vals.std(ddof=1)) if auc_vals.size > 1 else 0.0
            se_auc = std_auc / np.sqrt(auc_vals.size) if auc_vals.size > 1 else 0.0
            ci_delta = 1.96 * se_auc
            ci_lower = avg_auc - ci_delta
            ci_upper = avg_auc + ci_delta
            entries.append(
                {
                    "model": model_name,
                    "data_option": data_option,
                    "feature_set": model_type,
                    "sample_source": sample_source,
                    "avg_auc": avg_auc,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            )
        except Exception:
            continue

    if not entries:
        label = "validation " if validation else ""
        print(f"[WARN] No {label}auc_all.csv files found under {outdir}")
        return

    df = pd.DataFrame(entries)
    save_dir = _plots_save_dir(outdir, validation)
    zim_prefix = "validation_" if validation else ""
    zim_title = " (External validation cohort)" if validation else ""

    # --- Plot for data option 1 (both_heel/both_cord) and for 2+3 (heel_all/cord_all) ---
    for do_group, do_titles, plot_title, fname in [
        (
            [1],
            {"heel": "both_heel", "cord": "both_cord"},
            f"AUC scores for classification of preterm babies (when babies with both heel and cord data are present){zim_title}",
            f"{zim_prefix}average_auc_data_option1.png",
        ),
        (
            [2, 3],
            {2: "heel_all", 3: "cord_all"},
            f"AUC scores for classification of preterm babies (when babies with heel or cord data are present){zim_title}",
            f"{zim_prefix}average_auc_data_option2_3.png",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        bar_width = 0.5
        bar_centers = []
        xtick_labels = []
        cluster_idx = 0
        for group_val, group_title in do_titles.items():
            if isinstance(group_val, int):
                cluster_df = df[df["data_option"] == group_val]
            else:
                cluster_df = df[(df["data_option"] == 1) & (df["sample_source"] == group_val)]
            x_pos = cluster_idx
            for d_idx, feature_set in enumerate(FEATURE_SETS):
                row = cluster_df[
                    (cluster_df["feature_set"] == feature_set) & (cluster_df["model"] == "elasticnet_cv")
                ]
                if row.empty:
                    continue
                row = row.iloc[0]
                mean_auc = row["avg_auc"]
                ci_lower = row["ci_lower"]
                ci_upper = row["ci_upper"]
                xpos = x_pos + d_idx * 1.1
                bar = ax.bar(
                    xpos,
                    mean_auc,
                    width=bar_width,
                    color="#1f77b4",
                    alpha=0.6,
                    edgecolor="black",
                    linewidth=0.8,
                )
                ax.bar_label(bar, labels=[f"{mean_auc:.2f}"], padding=3, fontsize=10)
                yerr_lower = mean_auc - ci_lower
                yerr_upper = ci_upper - mean_auc
                ax.errorbar(
                    xpos,
                    mean_auc,
                    yerr=[[yerr_lower], [yerr_upper]],
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=4,
                )
                bar_centers.append(xpos)
                xtick_labels.append(f"{group_title}\n{feature_set}")
            cluster_idx += 4

        if not bar_centers:
            print(f"[INFO] No data to plot for {fname}")
            plt.close(fig)
            continue

        ax.set_xticks(bar_centers)
        ax.set_xticklabels(xtick_labels, rotation=25, ha="right", fontsize=11)
        ax.set_ylabel("Average AUC", fontsize=14)
        ax.set_title(plot_title, fontsize=10, fontweight="bold")
        ax.set_ylim(0.5, 1.01)
        ax.grid(axis="y", alpha=0.3)

        save_path = save_dir / fname
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)



def plot_features_selected(
    data_option: int,
    model_type: str,
    model_name: str,
    data_type: str,
    outdir: Path,
    validation: bool = False,
) -> None:
    """
    For a given configuration of data_option, model_type, model_name, and data_type,
    plot how often each feature had a nonzero coefficient across runs, as a fraction
    in [0, 1] (number of runs with nonzero / total runs).

    Directory layout matches classification / regression:
      - data_option 1: ...__{model_type}__{heel|cord}
      - data_option 2/3: ...__{model_type} (no heel/cord suffix)
    """

    run_dir = _config_run_dir(outdir, model_name, data_option, model_type, data_type, validation)
    coef_csv = run_dir / "coefficients_all.csv"

    if not coef_csv.exists():
        print(
            f"[WARN] plot_features_selected: missing {coef_csv} for "
            f"{model_type} {model_name} data_option={data_option}, data_type={data_type}"
        )
        return

    coef_df = pd.read_csv(coef_csv)
    if coef_df.empty or "coefficient names" not in coef_df.columns or "coefficient values" not in coef_df.columns:
        print(f"[WARN] plot_features_selected: empty or malformed {coef_csv}")
        return

    if "run" in coef_df.columns:
        n_runs = int(coef_df["run"].nunique())
    else:
        n_runs = len(coef_df)
    n_runs = max(1, n_runs)

    # Count runs in which each feature appears with nonzero coefficient
    feature_counts: dict[str, int] = defaultdict(int)
    for _, row in coef_df.iterrows():
        pairs = _coef_pairs_from_row(row["coefficient names"], row["coefficient values"])
        seen_this_run = {fname for fname, _ in pairs}
        for fname in seen_this_run:
            feature_counts[fname] += 1

    if not feature_counts:
        print(
            f"[INFO] plot_features_selected: no nonzero coefficients parsed for "
            f"{run_dir} (n_rows={len(coef_df)}, n_runs={n_runs}). "
            f"If you expected plots, check that regression.py wrote nonzero rows to coefficients_all.csv "
            f"(regression coef extraction)."
        )
        return

    # Fraction of runs [0, 1]
    freq_by_feature = {f: feature_counts[f] / n_runs for f in feature_counts}
    sorted_items = sorted(freq_by_feature.items(), key=lambda x: x[1], reverse=True)
    top_k = 20
    sorted_items = sorted_items[:top_k]
    features, freqs = zip(*sorted_items)

    # Plot as lollipop / dot chart
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(features))))
    y_pos = list(range(len(features)))
    ax.hlines(y_pos, 0, list(freqs), colors="#4C72B0", linewidth=1.5, alpha=0.6)
    ax.scatter(list(freqs), y_pos, color="#1f4e79", s=60, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(features))
    ax.invert_yaxis()  # highest-count feature at the top
    ax.set_xlabel("Selection frequency (fraction of runs)", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_xlim(0.0, 1.0)
    zim_title = ", External validation cohort" if validation else ""
    ax.set_title(
        f"Top {len(features)} features by selection frequency\n"
        f"{model_type}, {model_name}, data_option {data_option} (n_runs={n_runs}){zim_title}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    save_dir = _plots_save_dir(outdir, validation)
    zim_prefix = "validation_" if validation else ""
    # Save (include data_type for opt1 so heel/cord do not overwrite each other)
    if data_option == 1:
        out_fn = save_dir / f"{zim_prefix}features_selected_{model_type}_{model_name}_{data_option}_{data_type}.png"
    else:
        out_fn = save_dir / f"{zim_prefix}features_selected_{model_type}_{model_name}_{data_option}.png"
    fig.savefig(out_fn, dpi=300)
    plt.close(fig)


def plot_hyperparameters(
    data_option: int,
    model_type: str,
    model_name: str,
    data_type: str,
    outdir: Path,
    validation: bool = False,
) -> None:
    """
    Plot hyperparameters across runs for a given regression configuration.
    Mirrors regression directory naming:
      - opt1:  {model_name}__opt1__{model_type}__{data_type}
      - opt2/3:{model_name}__opt{2,3}__{model_type}
    """
    run_dir = _config_run_dir(outdir, model_name, data_option, model_type, data_type, validation)
    hyperparams_csv = run_dir / "hyperparameters_all.csv"
    if not hyperparams_csv.exists():
        print(f"[WARN] Missing hyperparameters file: {hyperparams_csv}")
        return

    # Define hyperparameters for each model type
    MODEL_HYPERPARAMS = {
        "lasso_cv":  ["C", "penalty", "alpha"],
        "elasticnet_cv": ["C", "penalty", "l1_ratio", "alpha"]
    }
    # Decide which keys are present
    hp_names = MODEL_HYPERPARAMS.get(model_name, [])

    hp_df = pd.read_csv(hyperparams_csv)

    # Only keep columns actually present in file (and among allowed)
    present_hps = [hp for hp in hp_names if hp in hp_df.columns]
    n_hps = len(present_hps)
    if n_hps == 0:
        print(f"[INFO] No recognized hyperparameters found in {hyperparams_csv}")
        return

    fig, axs = plt.subplots(n_hps, 1, figsize=(8, 3.2 * n_hps), squeeze=False)

    for i, hp in enumerate(present_hps):
        ax = axs[i, 0]
        # X-axis: run index (from 'run' column if present)
        if "run" in hp_df.columns:
            x = hp_df["run"]
        else:
            x = range(len(hp_df))
        # Connected points instead of discrete scatter
        ax.plot(
            x,
            hp_df[hp],
            marker="o",
            linestyle="-",
            color="#4C72B0",
            linewidth=1.5,
            markersize=4,
        )
        ax.set_xlabel("Run Number", fontsize=11)
        ax.set_ylabel(hp, fontsize=11)
        ax.set_title(f"{hp}", fontsize=13)
    
    zim_title = ", External validation cohort" if validation else ""
    plt.suptitle(
        f"Hyperparameters: {model_type}, {model_name}, data_option {data_option}{zim_title}",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_dir = _plots_save_dir(outdir, validation)
    zim_prefix = "validation_" if validation else ""
    if data_option == 1:
        out_fn = save_dir / f"{zim_prefix}hyperparameters_{model_type}_{model_name}_{data_option}_{data_type}.png"
    else:
        out_fn = save_dir / f"{zim_prefix}hyperparameters_{model_type}_{model_name}_{data_option}.png"
    plt.savefig(out_fn, dpi=300)
    plt.close()

def plot_all_predictions(
    data_option: int,
    model_type: str,
    model_name: str,
    data_type: str,
    validation: bool = False,
) -> None:
    """
    Plot the per-run mean predicted probability for a classification configuration.

    Reads predictions_all.csv and plots mean predicted value per run as a connected
    scatter (run index on x-axis), giving a quick sanity-check of stability across
    repeated splits.
    """
    run_dir = _config_run_dir(
        Path("outputs/classification"),
        model_name,
        data_option,
        model_type,
        data_type,
        validation,
    )
    preds_df = pd.read_csv(run_dir / "predictions_all.csv") if (run_dir / "predictions_all.csv").exists() else pd.DataFrame()
    if preds_df.empty:
        print(f"[WARN] Empty predictions file: {run_dir / 'predictions_all.csv'}")
        return

    # Plot: per-run trajectory of mean predicted value over runs
    plt.figure(figsize=(7, 5))

    if "run" in preds_df.columns:
        # Aggregate to one point per run: mean predicted
        grouped = preds_df.groupby("run")
        runs = grouped.size().index.to_numpy()
        mean_pred = grouped["y_pred"].mean().to_numpy()

        # Connected scatter (line with markers) for mean predicted
        plt.plot(
            runs,
            mean_pred,
            marker="s",
            linestyle="-",
            color="#ff7f0e",
            linewidth=1.5,
            markersize=4,
            label="Mean predicted",
        )
    else:
        # Fallback: treat each row as its own run index, plotting only predicted values
        x = np.arange(len(preds_df))
        plt.plot(
            x,
            preds_df["y_pred"],
            marker="s",
            linestyle="-",
            color="#ff7f0e",
            linewidth=1.5,
            markersize=4,
            label="Predicted",
        )

    plt.xlabel("Run", fontsize=12)
    plt.ylabel("Predicted Value", fontsize=12)
    zim_title = ", External validation cohort" if validation else ""
    plt.title(
        f"Per-run mean predicted value: {model_type}, {model_name}, data_option {data_option}{zim_title}",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=9)
    plt.tight_layout()

    save_dir = _plots_save_dir(Path("outputs/classification"), validation)
    zim_prefix = "validation_" if validation else ""
    if data_option == 1:
        out_fn = save_dir / f"{zim_prefix}predictions_{model_type}_{model_name}_{data_option}_{data_type}.png"
    else:
        out_fn = save_dir / f"{zim_prefix}predictions_{model_type}_{model_name}_{data_option}.png"
    plt.savefig(out_fn, dpi=300)
    plt.close()

def plot_regression_predictions(
    data_option: int,
    model_type: str,
    model_name: str,
    data_type: str,
    validation: bool = False,
) -> None:
    """
    Plot the predictions for all runs per configuration of data_option, model_type, and model_name, data_type.
    Creates a scatter plot: x axis is the true value, y axis is the predicted value.
    Title of the plot includes model_type, model_name, and data_option.
    The plot is saved as predictions_{model_type}_{model_name}_{data_option}.png within the appropriate subdirectory.
    """
    run_dir = _config_run_dir(
        Path("outputs/regression"),
        model_name,
        data_option,
        model_type,
        data_type,
        validation,
    )

    predictions_csv = run_dir / "predictions_all.csv"
    if not predictions_csv.exists():
        print(f"[WARN] Missing regression predictions file: {predictions_csv}")
        return

    preds_df = pd.read_csv(predictions_csv)

    # Plot: discrete scatter of true vs predicted
    plt.figure(figsize=(7, 5))

    plt.scatter(
        preds_df["y_true"],
        preds_df["y_pred"],
        alpha=0.6,
        s=22,
        color="#ff7f0e",
        label="Predicted",
    )

    # Make x and y axes share the same scale
    vmin = 24
    vmax = 45
    plt.xlim(vmin, vmax)
    plt.ylim(vmin, vmax)
    # Diagonal reference line (perfect prediction)
    plt.plot(
        [vmin, vmax],
        [vmin, vmax],
        "k--",
        linewidth=1.0,
        label="Ideal"
    )

    plt.xlabel("True Value", fontsize=12)
    plt.ylabel("Predicted Value", fontsize=12)
    zim_title = ", External validation cohort" if validation else ""
    plt.title(
        f"Per-run predictions: {model_type}, {model_name}, data_option {data_option}, data_type {data_type}{zim_title}",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=9)
    plt.tight_layout()

    save_dir = _plots_save_dir(Path("outputs/regression"), validation)
    zim_prefix = "validation_" if validation else ""
    if data_option == 1:
        out_fn = save_dir / f"{zim_prefix}predictions_{model_type}_{model_name}_{data_option}_{data_type}.png"
    else:
        out_fn = save_dir / f"{zim_prefix}predictions_{model_type}_{model_name}_{data_option}.png"
    plt.savefig(out_fn, dpi=300)
    plt.close()

def plot_mean_regression_metrics(
    data_option: int,
    model_type: str,
    model_name: str,
    data_type: str,
    validation: bool = False,
) -> None:
    
    """
    Scan all runs under outputs/regression/, compute average MAE and RMSE per
    (model, data_option, feature_set, sample_source), and create two bar plots:
      - Data option 1
      - Data options 2 and 3

    When validation=True, reads metrics from .../validation_zimbabwe/.
    """
    # Define mapping for color/opacity for each metric
    METRIC_STYLE = {
        "mean_mae": {"color": "#1f77b4", "alpha": 0.5, "label": "MAE"},
        "mean_rmse": {"color": "#ff7f0e", "alpha": 0.9, "label": "RMSE"},
    }
    FEATURE_SETS = ["clinical", "biomarker", "combined"]

    # Subdir names are either:
    #   - data_option 1:  {model_name}__opt1__{feature_set}__{sample_source}  (sample_source in {heel, cord})
    #   - data_option 2/3:{model_name}__opt{2,3}__{feature_set}              (no sample_source suffix)
    all_subdirs = list(Path("outputs/regression").glob("*"))
    entries = []
    for subdir in all_subdirs:
        try:
            if validation:
                if not subdir.is_dir():
                    continue
                config_name = _result_subdir_name(subdir / VALIDATION_SUBDIR, validation=True)
                if config_name is None:
                    continue
                parts = config_name.split("__")
            else:
                parts = subdir.name.split("__")
            # Handle both 3-part and 4-part names
            if len(parts) == 4:
                model_name, opt_str, model_type, sample_source = parts
            elif len(parts) == 3:
                model_name, opt_str, model_type = parts
                sample_source = None
            else:
                continue
            if not opt_str.startswith("opt"):
                continue
            data_option = int(opt_str.replace("opt", ""))

            # For data_option 1 we require an explicit sample_source ('heel' or 'cord')
            if data_option == 1:
                if sample_source not in {"heel", "cord"}:
                    continue
            else:
                # For data_option 2/3, we don't distinguish heel/cord here
                # but keep the column for completeness
                if sample_source is None:
                    sample_source = "all"

            if validation:
                mean_metrics_path = subdir / VALIDATION_SUBDIR / "mean_metrics_all.csv"
                per_run_path = subdir / VALIDATION_SUBDIR / "metrics_all.csv"
            else:
                mean_metrics_path = subdir / "mean_metrics_all.csv"
                per_run_path = subdir / "metrics_all.csv"

            if mean_metrics_path.exists():
                mean_metrics_df = pd.read_csv(mean_metrics_path)
                mean_mae = mean_metrics_df["mae_mean"].iloc[0]
                std_mae = mean_metrics_df["mae_std"].iloc[0]
                ci_lower_mae = mean_metrics_df["mae_ci_lower"].iloc[0]
                ci_upper_mae = mean_metrics_df["mae_ci_upper"].iloc[0]
                if (
                    "rmse_mean" in mean_metrics_df.columns
                    and "rmse_std" in mean_metrics_df.columns
                    and "rmse_ci_lower" in mean_metrics_df.columns
                    and "rmse_ci_upper" in mean_metrics_df.columns
                ):
                    mean_rmse = mean_metrics_df["rmse_mean"].iloc[0]
                    std_rmse = mean_metrics_df["rmse_std"].iloc[0]
                    ci_lower_rmse = mean_metrics_df["rmse_ci_lower"].iloc[0]
                    ci_upper_rmse = mean_metrics_df["rmse_ci_upper"].iloc[0]
                else:
                    if not per_run_path.exists():
                        continue
                    mrun = pd.read_csv(per_run_path)
                    if "rmse" in mrun.columns:
                        rs = mrun["rmse"].astype(float)
                    elif "mse" in mrun.columns:
                        rs = np.sqrt(mrun["mse"].astype(float))
                    else:
                        continue
                    n = max(1, len(rs))
                    mean_rmse = float(rs.mean())
                    std_rmse = float(rs.std(ddof=0)) if n > 1 else 0.0
                    ci_lower_rmse = mean_rmse - 1.96 * std_rmse / np.sqrt(n)
                    ci_upper_rmse = mean_rmse + 1.96 * std_rmse / np.sqrt(n)
            else:
                if not per_run_path.exists():
                    continue
                mrun = pd.read_csv(per_run_path)
                if "mae" not in mrun.columns:
                    continue
                mae_vals = mrun["mae"].astype(float)
                n = max(1, len(mae_vals))
                mean_mae = float(mae_vals.mean())
                std_mae = float(mae_vals.std(ddof=0)) if n > 1 else 0.0
                ci_lower_mae = mean_mae - 1.96 * std_mae / np.sqrt(n)
                ci_upper_mae = mean_mae + 1.96 * std_mae / np.sqrt(n)
                if "rmse" in mrun.columns:
                    rs = mrun["rmse"].astype(float)
                elif "mse" in mrun.columns:
                    rs = np.sqrt(mrun["mse"].astype(float))
                else:
                    continue
                mean_rmse = float(rs.mean())
                std_rmse = float(rs.std(ddof=0)) if n > 1 else 0.0
                ci_lower_rmse = mean_rmse - 1.96 * std_rmse / np.sqrt(n)
                ci_upper_rmse = mean_rmse + 1.96 * std_rmse / np.sqrt(n)
            entries.append(
                {
                    "model": model_name,
                    "data_option": data_option,
                    "feature_set": model_type,
                    "sample_source": sample_source,
                    "mean_mae": mean_mae,
                    "mean_rmse": mean_rmse,
                    "std_mae": std_mae,
                    "std_rmse": std_rmse,
                    "ci_lower_mae": ci_lower_mae,
                    "ci_upper_mae": ci_upper_mae,
                    "ci_lower_rmse": ci_lower_rmse,
                    "ci_upper_rmse": ci_upper_rmse,
                }
            )
        except Exception:
            continue

    if not entries:
        label = "validation " if validation else ""
        print(f"[WARN] No {label}regression metrics found under {Path('outputs/regression')}")
        return

    df = pd.DataFrame(entries)
    save_dir = _plots_save_dir(Path("outputs/regression"), validation)
    zim_prefix = "validation_" if validation else ""
    zim_title = " (External validation cohort)" if validation else ""

    plot_specs = [
        (
            [1],
            {"heel": "both_heel", "cord": "both_cord"},
            "Mean MAE and RMSE for all runs of all configurations with Data Option 1",
            "mean_metrics_data_option1",
        ),
        (
            [2, 3],
            {2: "heel_all", 3: "cord_all"},
            "Mean MAE and RMSE for all runs of all configurations with Data Option 2 and 3",
            "mean_metrics_data_option2_3",
        ),
    ]

    # One plot per model (elasticnet vs lasso), for training and validation.
    for plot_model in ["elasticnet_cv", "lasso_cv"]:
        model_df = df[df["model"] == plot_model]
        if model_df.empty:
            continue

        model_label = plot_model.replace("_cv", "")
        for _do_group, do_titles, plot_title_base, fname_stem in plot_specs:
            fname = f"{zim_prefix}{fname_stem}_{plot_model}.png"
            plot_title = f"{plot_title_base} ({model_label}){zim_title}"

            fig, ax = plt.subplots(figsize=(12, 6))
            bar_width = 0.36
            x_positions = []
            x_labels = []
            x_idx = 0
            for group_val, group_title in do_titles.items():
                if isinstance(group_val, int):
                    cluster_df = model_df[model_df["data_option"] == group_val]
                else:
                    cluster_df = model_df[
                        (model_df["data_option"] == 1) & (model_df["sample_source"] == group_val)
                    ]
                for feature_set in FEATURE_SETS:
                    row = cluster_df[cluster_df["feature_set"] == feature_set]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    mean_mae = row["mean_mae"]
                    mean_rmse = row["mean_rmse"]
                    ci_lower_mae = row["ci_lower_mae"]
                    ci_upper_mae = row["ci_upper_mae"]
                    ci_lower_rmse = row["ci_lower_rmse"]
                    ci_upper_rmse = row["ci_upper_rmse"]
                    mae_low = max(0.0, float(mean_mae - ci_lower_mae))
                    mae_high = max(0.0, float(ci_upper_mae - mean_mae))
                    rmse_low = max(0.0, float(mean_rmse - ci_lower_rmse))
                    rmse_high = max(0.0, float(ci_upper_rmse - mean_rmse))
                    x_center = x_idx
                    mae_x = x_center - bar_width / 2
                    rmse_x = x_center + bar_width / 2

                    mae_bar = ax.bar(
                        mae_x,
                        mean_mae,
                        width=bar_width,
                        color=METRIC_STYLE["mean_mae"]["color"],
                        alpha=METRIC_STYLE["mean_mae"]["alpha"],
                        edgecolor="black",
                        linewidth=0.8,
                        yerr=[[mae_low], [mae_high]],
                        capsize=4,
                        label=METRIC_STYLE["mean_mae"]["label"] if x_idx == 0 else "",
                    )
                    rmse_bar = ax.bar(
                        rmse_x,
                        mean_rmse,
                        width=bar_width,
                        color=METRIC_STYLE["mean_rmse"]["color"],
                        alpha=METRIC_STYLE["mean_rmse"]["alpha"],
                        edgecolor="black",
                        linewidth=0.8,
                        yerr=[[rmse_low], [rmse_high]],
                        capsize=4,
                        label=METRIC_STYLE["mean_rmse"]["label"] if x_idx == 0 else "",
                    )

                    ax.bar_label(mae_bar, labels=[f"{mean_mae:.2f}"], padding=2, fontsize=9)
                    ax.bar_label(rmse_bar, labels=[f"{mean_rmse:.2f}"], padding=2, fontsize=9)

                    x_positions.append(x_center)
                    x_labels.append(f"{group_title}\n{feature_set}")
                    x_idx += 1

            if not x_positions:
                print(f"[INFO] No mean metrics data to plot for {fname}")
                plt.close(fig)
                continue

            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=10)
            ax.set_ylabel("Average MAE and RMSE", fontsize=12)
            ax.set_title(plot_title, fontsize=12, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(save_dir / fname, dpi=300)
            plt.close(fig)

def plot_univariate_analysis(
    data_option: int,
    model_type: str,
    model_name: str,
    data_type: str,
    validation: bool = False,
) -> None:
    """
    For each feature: Wilcoxon rank-sum test (two independent samples) for preterm vs term,
    log2 fold change log2((mean_preterm + eps) / (mean_term + eps)), Benjamini–Hochberg FDR,
    and a volcano plot (-log10 p vs log2 FC).

    Saves PNG and CSV under outputs/plots/.
    Data loading matches classification.py (options 1–3).

    Only runs for model_type == 'biomarker'; clinical/combined are skipped
    (main() still passes all model_type values).
    """
    if model_type != "biomarker":
        return

    data_type_norm = str(data_type).strip().lower()
    if validation:
        df = pd.read_csv(EXTERNAL_VALIDATION_CSV)
        if data_option == 1:
            df = filter_validation_by_sample_type(df, data_type_norm)
        elif data_option in (2, 3):
            df = filter_validation_by_sample_type(df, "heel" if data_option == 2 else "cord")
        else:
            raise ValueError("data_option must be 1, 2, or 3.")
    elif data_option == 1:
        df = pd.read_csv(_project_root / "data" / "Bangladesh_children_with_both_samples.csv")
        if data_type_norm == "heel":
            df = df[df["Source"] == "HEEL"].copy()
        elif data_type_norm == "cord":
            df = df[df["Source"] == "CORD"].copy()
        else:
            raise ValueError("data_type must be 'heel' or 'cord' when data_option == 1.")
    elif data_option == 2:
        df = pd.read_csv(_project_root / "data" / "BangladeshcombineddatasetJan252022.csv")
        df = df[df["Source"] == "HEEL"].copy()
    elif data_option == 3:
        df = pd.read_csv(_project_root / "data" / "BangladeshcombineddatasetJan252022.csv")
        df = df[df["Source"] == "CORD"].copy()
    else:
        raise ValueError("data_option must be 1, 2, or 3.")

    target_col = "gestational_age_weeks"
    
    biomarker_features_to_keep = [
        "TSH", "N17P",
        "IRT", "GALT", "BIOT", "Ala", "Arg", "ASA", "ASA_Arg", "ASA_Orn", "Cit", "Cit_Arg", "Cit_Orn",
        "Cit_Tyr", "Gly", "LEU", "Leu_Ala", "Leu_Phe", "MET", "Met_Phe", "Orn", "Orn_Arg", "Orn_Cit",
        "Orn_Phe", "PHE", "PHE_TYR", "SUAC", "TYR", "Tyr_Phe", "Val", "Val_Ala", "Val_Phe", "C0",
        "C0_C16C18", "C0C2C3C16C18_Cit", "C2", "C3", "C3_C0", "C3_C16", "C3_C2", "C3_C4DC",
        "C3DC", "C4", "C4DC", "C4OH", "C5", "C5_C0", "C5_C2", "C5_C3", "C5_1", "C5DC", "C5DC_C16",
        "C5DC_C5OH", "C5DC_C8", "C5OH", "C5OH_C2", "C5OH_C5_1", "C5OH_C8", "C6", "C6DC", "C8", "C8_C10",
        "C8_C2", "C8_1", "C10", "C10_1", "C12", "C12_1", "C14", "C14OH", "C14_1", "C14_1_C12_1",
        "C14_1_C16", "C14_1_C4", "C14_2", "C16", "C16_1OH", "C16_1OH_C4DC", "C16OH", "C16OH_C16",
        "C18", "C18_1", "C18_1OH", "C18_2", "C18OH",
        "HGB___FAST", "HGB___F1", "HGB___F", "HGB___F_F1", "HGB___A", "HGB___FAST_F1", "HGB___Other",
    ]

    features_to_keep = biomarker_features_to_keep
    keep_cols = [c for c in features_to_keep + [target_col] if c in df.columns]
    df = df[keep_cols].copy()

    if target_col not in df.columns:
        raise KeyError(
            f"Expected target column '{target_col}' not found in dataframe columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[target_col])
    y_cont = df[target_col]
    y = (y_cont < PRETERM_CUTOFF).astype(int)

    
    if model_type == "biomarker":
        biomarker_cols = [c for c in biomarker_features_to_keep if c in df.columns]
        X = df[biomarker_cols].copy()

    min_per_group = 3
    eps = 1e-12
    rows = []
    for feature in X.columns:
        v = pd.to_numeric(X[feature], errors="coerce")
        mask = v.notna()
        y_m = y.loc[mask]
        v_m = v.loc[mask]
        pre = v_m[y_m == 1]
        term = v_m[y_m == 0]
        if len(pre) < min_per_group or len(term) < min_per_group:
            rows.append(
                {
                    "feature": feature,
                    "mean_preterm": np.nan,
                    "mean_term": np.nan,
                    "log2_fc": np.nan,
                    "p_value": np.nan,
                }
            )
            continue
        m_pre = float(pre.mean())
        m_term = float(term.mean())
        log2_fc = np.log2((m_pre + eps) / (m_term + eps))
        try:
            _, p_raw = stats.ranksums(pre.to_numpy(), term.to_numpy(), alternative="two-sided")
            if p_raw is None or (isinstance(p_raw, float) and np.isnan(p_raw)):
                p_raw = np.nan
        except ValueError:
            p_raw = np.nan
        rows.append(
            {
                "feature": feature,
                "mean_preterm": m_pre,
                "mean_term": m_term,
                "log2_fc": log2_fc,
                "p_value": float(p_raw) if p_raw == p_raw else np.nan,
            }
        )

    res = pd.DataFrame(rows)
    res["q_bh"] = np.nan
    m_p = res["p_value"].notna()
    if m_p.any():
        res.loc[m_p, "q_bh"] = _benjamini_hochberg(res.loc[m_p, "p_value"].to_numpy(dtype=float))

    out_dir = _project_root / "outputs" / "plots"
    if validation:
        out_dir = out_dir / VALIDATION_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    zim_prefix = "validation_" if validation else ""
    fname_base = f"{zim_prefix}univariate_analysis_{model_type}_{model_name}_{data_option}_{data_type_norm}"
    csv_path = out_dir / f"{fname_base}.csv"
    res.to_csv(csv_path, index=False)

    # Plot using BH-adjusted p-values (q-values) as requested.
    plot_mask = res["q_bh"].notna() & res["log2_fc"].notna()
    if not plot_mask.any():
        print(f"[WARN] No valid p-values for volcano plot: {fname_base}")
        return

    sub = res.loc[plot_mask].copy()
    q_plot = np.maximum(sub["q_bh"].to_numpy(dtype=float), 1e-300)
    neg_log_q = -np.log10(q_plot)
    x_fc = sub["log2_fc"].to_numpy(dtype=float)
    sig = np.isfinite(q_plot) & (q_plot < 0.05)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(
        x_fc[~sig],
        neg_log_q[~sig],
        s=26,
        alpha=0.55,
        c="#9aa0a6",
        edgecolors="none",
        label="Adjusted p >= 0.05",
    )
    if sig.any():
        ax.scatter(
            x_fc[sig],
            neg_log_q[sig],
            s=40,
            alpha=0.95,
            c="#e41a1c",
            edgecolors="black",
            linewidths=0.5,
            label="Adjusted p < 0.05 (BH)",
        )

    # Label the top 15 features: significant ones first (sorted by -log10(q)
    # then |log2FC|), falling back to the top 15 overall if nothing is significant.
    # Capping at 15 is the most effective way to prevent overlap on a dense volcano.
    TOP_N_LABELS = 15
    LABEL_FS = 12

    rank_df = pd.DataFrame({
        "neg_log_q": neg_log_q,
        "abs_fc": np.abs(x_fc),
        "is_sig": sig.astype(int),
    }, index=sub.index)
    rank_df = rank_df.sort_values(
        ["is_sig", "neg_log_q", "abs_fc"], ascending=[False, False, False]
    )
    label_indices = set(rank_df.index[:TOP_N_LABELS])

    texts = []
    tx, ty = [], []   # anchor coordinates for adjust_text target_x/y
    for i, (idx, r) in enumerate(sub.iterrows()):
        if idx not in label_indices:
            continue
        x_val = float(r["log2_fc"])
        y_val = -np.log10(max(float(r["q_bh"]), 1e-300))
        is_sig = float(r["q_bh"]) < 0.05
        texts.append(ax.text(
            x_val, y_val,
            str(r["feature"]),
            fontsize=LABEL_FS,
            color="#b2182b" if is_sig else "#5f6368",
            zorder=5,
        ))
        tx.append(x_val)
        ty.append(y_val)

    if texts:
        adjust_text(
            texts,
            x=x_fc,        # repel labels from ALL scatter points, not just labelled ones
            y=neg_log_q,
            target_x=tx,
            target_y=ty,
            ax=ax,
            force_text=(0.5, 0.8),
            force_static=(0.3, 0.5),
            force_explode=(0.5, 1.0),
            expand=(1.3, 1.5),
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6),
        )

    ax.axhline(-np.log10(0.05), color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel(r"log$_2$ fold change (preterm / term)", fontsize=12)
    ax.set_ylabel(r"$-$log$_{10}$(adjusted p)", fontsize=12)
    cohort = "External validation cohort" if validation else "Training cohort"
    ax.set_title(
        f"Volcano (Wilcoxon rank-sum): {model_type}, {model_name}, opt{data_option}, {data_type_norm}\n"
        f"{cohort}; GA < {PRETERM_CUTOFF} w = preterm",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    # Keep legend in a stable corner to avoid overlap with dense point labels.
    ax.legend(
        loc="upper right",
        fontsize=13,
        framealpha=0.95,
        markerscale=1.5,
        labelspacing=0.8,
        handletextpad=0.8,
    )    
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Keep legend area separate from plot area
    fig.savefig(
        out_dir / f"{fname_base}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.close(fig)

def _coef_pairs_from_row(names_cell, vals_cell) -> list[tuple[str, float]]:
    """
    One row from coefficients_all.csv may store sparse (feature, coef) pairs as lists
    or as stringified lists after CSV round-trip. Returns aligned (name, coef) pairs.
    """
    def _norm_list(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        s = str(x).strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
            return [parsed]
        except (ValueError, SyntaxError):
            return [s.strip("'\"")]

    names = _norm_list(names_cell)
    vals_raw = _norm_list(vals_cell)
    if not names:
        return []
    coefs = []
    for v in vals_raw:
        try:
            coefs.append(float(v))
        except (TypeError, ValueError):
            continue
    if len(names) != len(coefs):
        return []
    return [(str(n).strip(), c) for n, c in zip(names, coefs) if abs(c) > 0]


def plot_feature_selected_coe(
    data_option: int,
    model_type: str,
    model_name: str,
    data_type: str,
    outdir: Path,
    top_n: int = 20,
    validation: bool = False,
) -> None:
    """
    Plot mean nonzero logistic coefficients across runs (classification / regression).

    Only the top `top_n` features by |mean coefficient| are drawn. This is separate
    from `plot_features_selected`, which plots selection *counts* and keeps the top
    50% of ranked features (not a fixed 20).
    """

    run_dir = _config_run_dir(outdir, model_name, data_option, model_type, data_type, validation)
    coef_csv = run_dir / "coefficients_all.csv"

    if not coef_csv.exists():
        print(
            f"[WARN] plot_feature_selected_coe: missing {coef_csv} for "
            f"{model_type} {model_name} data_option={data_option}, data_type={data_type}"
        )
        return

    coef_df = pd.read_csv(coef_csv)
    if coef_df.empty or "coefficient names" not in coef_df.columns or "coefficient values" not in coef_df.columns:
        print(f"[WARN] plot_feature_selected_coe: empty or malformed {coef_csv}")
        return

    # Collect per-run coefficient values for each feature
    coef_by_feature: dict = defaultdict(list)
    for _, row in coef_df.iterrows():
        pairs = _coef_pairs_from_row(row["coefficient names"], row["coefficient values"])
        for fname, coef in pairs:
            coef_by_feature[fname].append(coef)

    if not coef_by_feature:
        print(
            f"[INFO] plot_feature_selected_coe: no nonzero coefficients to plot for "
            f"{run_dir}"
        )
        return

    # Select top_n features by |mean coefficient|
    mean_abs = {f: float(np.abs(np.mean(vals))) for f, vals in coef_by_feature.items()}
    sorted_features = sorted(mean_abs, key=mean_abs.__getitem__, reverse=True)
    k = min(int(top_n), len(sorted_features))
    top_features = sorted_features[:k]

    # Build data lists in sorted order (highest |mean| at top → invert y-axis)
    plot_data = [coef_by_feature[f] for f in top_features]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.38 * k)))
    bp = ax.boxplot(
        plot_data,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="#4C72B0", alpha=0.55),
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=1.0, color="#333333"),
        capprops=dict(linewidth=1.0, color="#333333"),
        flierprops=dict(marker="o", markersize=3, alpha=0.4, markerfacecolor="#4C72B0", linestyle="none"),
    )
    ax.set_yticks(range(1, k + 1))
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()  # highest |mean| at the top
    ax.axvline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Coefficient value across runs", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    zim_title = ", External validation cohort" if validation else ""
    ax.set_title(
        f"Top {k} features by |mean coefficient|\n{model_type}, {model_name}, "
        f"data_option={data_option}, {data_type if data_option == 1 else 'all'}{zim_title}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    zim_prefix = "validation_" if validation else ""
    fig.savefig(
        run_dir / f"{zim_prefix}feature_selected_coefficients_{model_type}_{model_name}_{data_option}_{data_type}.png",
        dpi=300,
    )
    plt.close(fig)


def run_all_plots(validation: bool = False) -> None:
    """Generate all plots for training or external validation results."""
    if not validation:
        boxplot_irt_dataset_preterm_term()

    for data_option in [1, 2, 3]:
        for model_type in ["clinical", "biomarker", "combined"]:
            for model_name in ["lasso_cv", "elasticnet_cv"]:
                if data_option == 1:
                    data_types = ["heel", "cord"]
                else:
                    data_types = ["all"]
                for data_type in data_types:
                    plot_features_selected(
                        data_option,
                        model_type,
                        model_name,
                        data_type,
                        Path("outputs/classification"),
                        validation=validation,
                    )
                    plot_features_selected(
                        data_option,
                        model_type,
                        model_name,
                        data_type,
                        Path("outputs/regression"),
                        validation=validation,
                    )
                    plot_hyperparameters(
                        data_option,
                        model_type,
                        model_name,
                        data_type,
                        Path("outputs/classification"),
                        validation=validation,
                    )
                    plot_hyperparameters(
                        data_option,
                        model_type,
                        model_name,
                        data_type,
                        Path("outputs/regression"),
                        validation=validation,
                    )
                    plot_regression_predictions(
                        data_option,
                        model_type,
                        model_name,
                        data_type,
                        validation=validation,
                    )
                    plot_all_predictions(
                        data_option,
                        model_type,
                        model_name,
                        data_type,
                        validation=validation,
                    )
                    plot_univariate_analysis(
                        data_option,
                        model_type,
                        model_name,
                        data_type,
                        validation=validation,
                    )
                    plot_feature_selected_coe(
                        data_option,
                        model_type,
                        model_name,
                        data_type,
                        Path("outputs/classification"),
                        validation=validation,
                    )
                    plot_feature_selected_coe(
                        data_option,
                        model_type,
                        model_name,
                        data_type,
                        Path("outputs/regression"),
                        validation=validation,
                    )

        plot_average_auc(Path("outputs/classification"), validation=validation)

    plot_mean_regression_metrics(1, "clinical", "lasso_cv", "heel", validation=validation)


def main():
    """Parse command-line arguments and invoke run_all_plots."""
    parser = argparse.ArgumentParser(description="Generate plots for Bangladesh training or Zimbabwe validation")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Plot results from outputs/*/.../validation_zimbabwe/ folders",
    )
    args = parser.parse_args()
    run_all_plots(validation=args.validate)
if __name__ == "__main__":
    main()