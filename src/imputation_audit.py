"""Count feature NaNs imputed by SimpleImputer in classification / regression pipelines.

Classification and regression use the same feature matrices, so counts are shared.

Usage:
  python3 src/imputation_audit.py
  python3 src/imputation_audit.py --data_option 1 --run_seed 42
  python3 src/imputation_audit.py --plot-only

Outputs (under outputs/imputation_counts/):
  - imputation_summary.csv / imputation_by_feature.csv
  - imputation_categorical_summary.csv / imputation_categorical_by_feature.csv
  - imputation_summary_opt*.png, imputation_categorical_opt*.png, etc.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import PRETERM_CUTOFF, RANDOM_SEED
from src.feature_sets import (
    CATEGORICAL_FEATURES,
    align_to_training_features,
    load_bangladesh_frame,
    load_zimbabwe_validation_frame,
)

DEFAULT_OUT_DIR = _project_root / "outputs" / "imputation_counts"

SPLIT_ORDER = ["train_80", "test_20", "validation"]
SPLIT_LABELS = {
    "train_80": "BD train (fit)",
    "test_20": "BD test (predict)",
    "validation": "ZW val (predict)",
}
SPLIT_COLORS = {
    "train_80": "#1f77b4",
    "test_20": "#ff7f0e",
    "validation": "#2ca02c",
}
COHORT_FOR_SPLIT = {
    "train_80": "Bangladesh",
    "test_20": "Bangladesh",
    "validation": "Zimbabwe",
}


def _config_label(row: pd.Series) -> str:
    """Return a display label combining model_type and data_type for a summary row."""
    if row["data_type"] == "all":
        return str(row["model_type"])
    return f"{row['model_type']}\n{row['data_type']}"


def _plot_slice(
    ax: plt.Axes,
    slice_df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    unit_note: str,
    use_log: bool = False,
) -> None:
    """Render a grouped bar chart for one metric across all configs and pipeline splits."""
    slice_df = slice_df.copy()
    slice_df["config"] = slice_df.apply(_config_label, axis=1)
    configs = slice_df[["model_type", "data_type", "config"]].drop_duplicates()
    configs = configs.sort_values(["model_type", "data_type"])
    config_labels = configs["config"].tolist()
    x = np.arange(len(config_labels))
    bar_width = 0.24

    for idx, split in enumerate(SPLIT_ORDER):
        cohort = COHORT_FOR_SPLIT[split]
        heights = []
        for _, cfg in configs.iterrows():
            row = slice_df[
                (slice_df["model_type"] == cfg["model_type"])
                & (slice_df["data_type"] == cfg["data_type"])
                & (slice_df["cohort"] == cohort)
                & (slice_df["split"] == split)
            ]
            heights.append(float(row[metric].iloc[0]) if not row.empty else 0.0)
        offset = (idx - 1) * bar_width
        bars = ax.bar(
            x + offset,
            heights,
            width=bar_width,
            color=SPLIT_COLORS[split],
            edgecolor="black",
            linewidth=0.6,
            label=SPLIT_LABELS[split],
        )
        for bar, val in zip(bars, heights):
            if val <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(val) if val >= 10 else val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=9)
    scale_note = "log10 scale" if use_log else "linear scale"
    ax.set_ylabel(f"{ylabel}\n({scale_note}; absolute counts, not %)", fontsize=10)
    ax.set_title(f"{title}\n[{unit_note}]", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    if use_log:
        ax.set_yscale("log")


def plot_imputation_summary(summary_df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Bar charts of imputation burden by config and pipeline stage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    plot_df = summary_df[
        summary_df["split"].isin(SPLIT_ORDER) & summary_df["cohort"].isin(COHORT_FOR_SPLIT.values())
    ].copy()

    for data_option in sorted(plot_df["data_option"].unique()):
        opt_df = plot_df[plot_df["data_option"] == data_option]
        data_types = sorted(opt_df["data_type"].unique())
        ncols = len(data_types)
        fig, axes = plt.subplots(2, ncols, figsize=(5.5 * ncols, 9), squeeze=False)

        for col_idx, data_type in enumerate(data_types):
            slice_df = opt_df[opt_df["data_type"] == data_type]
            dt_title = data_type if data_type != "all" else "all samples"
            use_log = slice_df["missing_cells"].max() >= 100

            _plot_slice(
                axes[0, col_idx],
                slice_df,
                metric="missing_cells",
                title=f"Imputed missing cells — opt{data_option}, {dt_title}",
                ylabel="Number of NaN cells filled by imputer",
                unit_note="unit: imputed cells (rows × features with NaN)",
                use_log=use_log,
            )
            _plot_slice(
                axes[1, col_idx],
                slice_df,
                metric="rows_with_any_missing",
                title=f"Samples with any missing data — opt{data_option}, {dt_title}",
                ylabel="Number of samples (rows) with ≥1 NaN",
                unit_note="unit: sample count (not % of cohort)",
                use_log=False,
            )

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(
            f"Imputation burden by model config (data option {data_option})\n"
            "Same for classification and regression; all bars are absolute counts (not percentages)",
            fontsize=12,
            fontweight="bold",
            y=1.08,
        )
        fig.tight_layout()
        path = out_dir / f"imputation_summary_opt{data_option}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved


def plot_top_missing_features(
    feature_df: pd.DataFrame,
    out_dir: Path,
    top_n: int = 15,
) -> list[Path]:
    """Top features with missing values on Zimbabwe validation (combined models)."""
    if feature_df.empty:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    zw_val = feature_df[
        (feature_df["cohort"] == "Zimbabwe") & (feature_df["split"] == "validation")
    ].copy()
    if zw_val.empty:
        return saved

    targets = zw_val[
        (zw_val["model_type"] == "combined") | (zw_val["model_type"] == "clinical")
    ]
    if targets.empty:
        targets = zw_val

    for (data_option, data_type), group in targets.groupby(["data_option", "data_type"]):
        combined = group[group["model_type"] == "combined"]
        plot_group = combined if not combined.empty else group
        top = (
            plot_group.groupby("feature", as_index=False)["n_missing"]
            .max()
            .sort_values("n_missing", ascending=False)
            .head(top_n)
        )
        if top.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(top))))
        ax.barh(top["feature"], top["n_missing"], color="#4C72B0", edgecolor="black", linewidth=0.5)
        ax.invert_yaxis()
        ax.set_xlabel(
            "Missing values imputed per feature\n"
            "(absolute cell count across validation samples; not %)",
            fontsize=11,
        )
        ax.set_ylabel("Feature", fontsize=11)
        dt_label = data_type if data_type != "all" else "all"
        ax.set_title(
            f"Top {len(top)} imputed features — Zimbabwe validation\n"
            f"data option {data_option}, {dt_label}, combined model "
            f"[scale: count of NaN cells, not %]",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"imputation_top_features_opt{data_option}_{dt_label}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    return saved


def _missing_stats(X: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """Return a dict of missingness statistics (cell counts, row counts, feature counts) for X."""
    if columns is not None:
        use_cols = [c for c in columns if c in X.columns]
        X = X[use_cols] if use_cols else X.iloc[:, 0:0]
    nan_mask = X.isna()
    n_rows, n_cols = X.shape
    total_cells = int(n_rows * n_cols)
    missing_cells = int(nan_mask.sum().sum())
    return {
        "n_rows": int(n_rows),
        "n_features": int(n_cols),
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "pct_missing_cells": round(100 * missing_cells / total_cells, 4) if total_cells else 0.0,
        "rows_with_any_missing": int(nan_mask.any(axis=1).sum()) if n_cols else 0,
        "features_with_any_missing": int(nan_mask.any(axis=0).sum()) if n_cols else 0,
    }


def _categorical_missing_stats(X: pd.DataFrame) -> dict:
    """Return missingness statistics restricted to the categorical feature columns."""
    stats = _missing_stats(X, CATEGORICAL_FEATURES)
    stats["categorical_features_in_matrix"] = stats.pop("n_features")
    stats["categorical_missing_cells"] = stats.pop("missing_cells")
    stats["categorical_pct_missing_cells"] = stats.pop("pct_missing_cells")
    stats["rows_with_any_categorical_missing"] = stats.pop("rows_with_any_missing")
    stats["categorical_features_with_any_missing"] = stats.pop("features_with_any_missing")
    return stats


def _feature_missing_rows(
    X: pd.DataFrame,
    data_option: int,
    data_type: str,
    model_type: str,
    cohort: str,
    split: str,
    features: list[str] | None = None,
) -> list[dict]:
    """Return one dict per feature that has at least one missing value in X."""
    if features is not None:
        cols = [c for c in features if c in X.columns]
        counts = X[cols].isna().sum() if cols else pd.Series(dtype=int)
    else:
        counts = X.isna().sum()
    counts = counts[counts > 0].sort_values(ascending=False)
    return [
        {
            "data_option": data_option,
            "data_type": data_type,
            "model_type": model_type,
            "cohort": cohort,
            "split": split,
            "feature": feature,
            "n_missing": int(n_missing),
        }
        for feature, n_missing in counts.items()
    ]


def _categorical_feature_missing_rows(
    X: pd.DataFrame,
    data_option: int,
    data_type: str,
    model_type: str,
    cohort: str,
    split: str,
) -> list[dict]:
    """Return one dict per categorical feature with its missing-value count in X."""
    rows = []
    for feature in CATEGORICAL_FEATURES:
        if feature not in X.columns:
            continue
        rows.append(
            {
                "data_option": data_option,
                "data_type": data_type,
                "model_type": model_type,
                "cohort": cohort,
                "split": split,
                "feature": feature,
                "n_missing": int(X[feature].isna().sum()),
            }
        )
    return rows


def plot_categorical_imputation(
    categorical_summary_df: pd.DataFrame,
    categorical_feature_df: pd.DataFrame,
    out_dir: Path,
) -> list[Path]:
    """Plot imputation counts for encoded categorical features (sex, multiple_birth)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    plot_summary = categorical_summary_df[
        (categorical_summary_df["split"].isin(SPLIT_ORDER))
        & (categorical_summary_df["model_type"].isin(["clinical", "combined"]))
    ].copy()
    if plot_summary.empty:
        return saved

    cat_colors = {"sex": "#9467bd", "multiple_birth": "#8c564b"}

    for data_option in sorted(plot_summary["data_option"].unique()):
        opt_df = plot_summary[plot_summary["data_option"] == data_option]
        data_types = sorted(opt_df["data_type"].unique())
        fig, axes = plt.subplots(2, len(data_types), figsize=(5.5 * len(data_types), 9), squeeze=False)

        for col_idx, data_type in enumerate(data_types):
            slice_df = opt_df[opt_df["data_type"] == data_type]
            clinical_df = slice_df[slice_df["model_type"] == "clinical"]
            if clinical_df.empty:
                clinical_df = slice_df[slice_df["model_type"] == "combined"]
            dt_title = data_type if data_type != "all" else "all samples"

            _plot_slice(
                axes[0, col_idx],
                clinical_df,
                metric="categorical_missing_cells",
                title=f"Categorical NaN cells imputed — opt{data_option}, {dt_title}",
                ylabel="Number of NaN cells (sex + multiple_birth)",
                unit_note="unit: imputed cells; median-filled codes 1/2 (not mode)",
                use_log=False,
            )
            _plot_slice(
                axes[1, col_idx],
                clinical_df,
                metric="rows_with_any_categorical_missing",
                title=f"Samples with missing categorical data — opt{data_option}, {dt_title}",
                ylabel="Number of samples with ≥1 NaN in sex/multiple_birth",
                unit_note="unit: sample count (not % of cohort)",
                use_log=False,
            )

            feat_slice = categorical_feature_df[
                (categorical_feature_df["data_option"] == data_option)
                & (categorical_feature_df["data_type"] == data_type)
                & (categorical_feature_df["model_type"] == "clinical")
                & (categorical_feature_df["feature"].isin(CATEGORICAL_FEATURES))
            ]
            if feat_slice.empty:
                feat_slice = categorical_feature_df[
                    (categorical_feature_df["data_option"] == data_option)
                    & (categorical_feature_df["data_type"] == data_type)
                    & (categorical_feature_df["model_type"] == "combined")
                    & (categorical_feature_df["feature"].isin(CATEGORICAL_FEATURES))
                ]

            inset = axes[0, col_idx].inset_axes([0.55, 0.55, 0.42, 0.42])
            if not feat_slice.empty:
                for split_idx, split in enumerate(SPLIT_ORDER):
                    cohort = COHORT_FOR_SPLIT[split]
                    split_feats = feat_slice[
                        (feat_slice["cohort"] == cohort) & (feat_slice["split"] == split)
                    ]
                    x_base = split_idx * (len(CATEGORICAL_FEATURES) + 0.5)
                    for f_idx, feature in enumerate(CATEGORICAL_FEATURES):
                        row = split_feats[split_feats["feature"] == feature]
                        val = float(row["n_missing"].iloc[0]) if not row.empty else 0.0
                        inset.bar(
                            x_base + f_idx,
                            val,
                            color=cat_colors.get(feature, "#999999"),
                            width=0.8,
                            edgecolor="black",
                            linewidth=0.4,
                        )
                inset.set_xticks(
                    [i * (len(CATEGORICAL_FEATURES) + 0.5) + 0.5 for i in range(len(SPLIT_ORDER))]
                )
                inset.set_xticklabels(["train", "test", "val"], fontsize=7)
                inset.set_ylabel("cells", fontsize=7)
                inset.set_title("By feature", fontsize=8)
                inset.tick_params(labelsize=7)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(
            f"Categorical imputation (sex, multiple_birth) — data option {data_option}\n"
            "Pre-imputation NaN counts; median imputation on numeric codes 1=male/multiple, 2=female/single; "
            "biomarker-only configs excluded (no categorical columns in X)",
            fontsize=11,
            fontweight="bold",
            y=1.10,
        )
        fig.tight_layout()
        path = out_dir / f"imputation_categorical_opt{data_option}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

        if not categorical_feature_df.empty:
            feat_plot = categorical_feature_df[
                (categorical_feature_df["data_option"] == data_option)
                & (categorical_feature_df["model_type"] == "clinical")
            ]
            if feat_plot.empty:
                feat_plot = categorical_feature_df[
                    categorical_feature_df["data_option"] == data_option
                ]
            plot_types = [dt for dt in sorted(feat_plot["data_type"].unique()) if dt != "all"]
            if not plot_types:
                plot_types = sorted(feat_plot["data_type"].unique())
            fig2, axes2 = plt.subplots(1, len(plot_types), figsize=(5 * len(plot_types), 4), squeeze=False)
            for ax_idx, dt in enumerate(plot_types):
                ax2 = axes2[0, ax_idx]
                x = np.arange(len(SPLIT_ORDER))
                width = 0.35
                for f_idx, feature in enumerate(CATEGORICAL_FEATURES):
                    heights = []
                    for split in SPLIT_ORDER:
                        cohort = COHORT_FOR_SPLIT[split]
                        row = feat_plot[
                            (feat_plot["data_type"] == dt)
                            & (feat_plot["cohort"] == cohort)
                            & (feat_plot["split"] == split)
                            & (feat_plot["feature"] == feature)
                        ]
                        heights.append(float(row["n_missing"].iloc[0]) if not row.empty else 0.0)
                    ax2.bar(
                        x + (f_idx - 0.5) * width,
                        heights,
                        width=width,
                        label=feature,
                        color=cat_colors.get(feature, "#999999"),
                        edgecolor="black",
                        linewidth=0.5,
                    )
                ax2.set_xticks(x)
                ax2.set_xticklabels([SPLIT_LABELS[s] for s in SPLIT_ORDER], fontsize=8, rotation=15, ha="right")
                ax2.set_ylabel("Missing cells (count, not %)", fontsize=9)
                ax2.set_title(f"{dt}", fontsize=10, fontweight="bold")
                ax2.legend(fontsize=8)
                ax2.grid(axis="y", alpha=0.3)
            fig2.suptitle(
                f"Categorical imputation by feature (sex, multiple_birth) — opt{data_option}, clinical model",
                fontsize=11,
                fontweight="bold",
            )
            fig2.tight_layout()
            path2 = out_dir / f"imputation_categorical_by_feature_opt{data_option}.png"
            fig2.savefig(path2, dpi=300, bbox_inches="tight")
            plt.close(fig2)
            saved.append(path2)

    return saved


def plot_imputation_outputs(
    summary_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    out_dir: Path,
    top_n: int = 15,
    categorical_summary_df: pd.DataFrame | None = None,
    categorical_feature_df: pd.DataFrame | None = None,
) -> list[Path]:
    """Generate all imputation summary plots."""
    paths = plot_imputation_summary(summary_df, out_dir)
    paths.extend(plot_top_missing_features(feature_df, out_dir, top_n=top_n))
    if categorical_summary_df is not None and not categorical_summary_df.empty:
        paths.extend(
            plot_categorical_imputation(
                categorical_summary_df,
                categorical_feature_df if categorical_feature_df is not None else pd.DataFrame(),
                out_dir,
            )
        )
    return paths


def collect_imputation_counts(
    data_options: list[int],
    model_types: list[str],
    run_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Iterate over all model configs and collect per-split imputation counts into DataFrames."""
    summary_rows: list[dict] = []
    feature_rows: list[dict] = []
    categorical_summary_rows: list[dict] = []
    categorical_feature_rows: list[dict] = []

    for data_option in data_options:
        data_types = ["heel", "cord"] if data_option == 1 else ["all"]
        for data_type in data_types:
            val_data_type = None if data_option in (2, 3) else data_type
            for model_type in model_types:
                X_bd, y_cont, feature_names = load_bangladesh_frame(
                    data_option, data_type, model_type
                )
                y_bin = (y_cont < PRETERM_CUTOFF).astype(int)
                X_train, X_test, _, _ = train_test_split(
                    X_bd,
                    y_bin,
                    test_size=0.2,
                    random_state=run_seed,
                    stratify=y_bin,
                )

                X_zw, _, _, _, _ = load_zimbabwe_validation_frame(
                    model_type, data_type=val_data_type
                )
                X_zw = align_to_training_features(X_zw, feature_names)

                for cohort, split, X in [
                    ("Bangladesh", "full", X_bd),
                    ("Bangladesh", "train_80", X_train),
                    ("Bangladesh", "test_20", X_test),
                    ("Zimbabwe", "validation", X_zw),
                ]:
                    stats = _missing_stats(X)
                    summary_rows.append(
                        {
                            "data_option": data_option,
                            "data_type": data_type,
                            "model_type": model_type,
                            "cohort": cohort,
                            "split": split,
                            **stats,
                        }
                    )
                    feature_rows.extend(
                        _feature_missing_rows(
                            X, data_option, data_type, model_type, cohort, split
                        )
                    )

                    cat_stats = _categorical_missing_stats(X)
                    categorical_summary_rows.append(
                        {
                            "data_option": data_option,
                            "data_type": data_type,
                            "model_type": model_type,
                            "cohort": cohort,
                            "split": split,
                            "n_rows": stats["n_rows"],
                            **cat_stats,
                        }
                    )
                    categorical_feature_rows.extend(
                        _categorical_feature_missing_rows(
                            X, data_option, data_type, model_type, cohort, split
                        )
                    )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(feature_rows),
        pd.DataFrame(categorical_summary_rows),
        pd.DataFrame(categorical_feature_rows),
    )


def main() -> None:
    """Parse CLI arguments, collect imputation counts, save CSVs, and optionally plot results."""
    parser = argparse.ArgumentParser(
        description="Count NaN cells filled by SimpleImputer for each model configuration."
    )
    parser.add_argument("--data_option", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--model_type",
        nargs="+",
        default=["clinical", "biomarker", "combined"],
        choices=["clinical", "biomarker", "combined"],
    )
    parser.add_argument(
        "--run_seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for the 80/20 Bangladesh train/test split (default: config RANDOM_SEED).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for summary and by-feature CSV outputs.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating PNG plots.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate plots from existing CSVs in --out_dir (skip recount).",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=15,
        help="Number of top features to show in feature missingness plots.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "imputation_summary.csv"
    feature_path = args.out_dir / "imputation_by_feature.csv"
    categorical_summary_path = args.out_dir / "imputation_categorical_summary.csv"
    categorical_feature_path = args.out_dir / "imputation_categorical_by_feature.csv"

    if args.plot_only:
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing {summary_path}; run without --plot-only first.")
        summary_df = pd.read_csv(summary_path)
        feature_df = pd.read_csv(feature_path) if feature_path.exists() else pd.DataFrame()
        categorical_summary_df = (
            pd.read_csv(categorical_summary_path)
            if categorical_summary_path.exists()
            else pd.DataFrame()
        )
        categorical_feature_df = (
            pd.read_csv(categorical_feature_path)
            if categorical_feature_path.exists()
            else pd.DataFrame()
        )
    else:
        summary_df, feature_df, categorical_summary_df, categorical_feature_df = (
            collect_imputation_counts(
                data_options=args.data_option,
                model_types=args.model_type,
                run_seed=args.run_seed,
            )
        )
        summary_df.to_csv(summary_path, index=False)
        feature_df.to_csv(feature_path, index=False)
        categorical_summary_df.to_csv(categorical_summary_path, index=False)
        categorical_feature_df.to_csv(categorical_feature_path, index=False)

        print("Imputation counts (NaN cells filled by SimpleImputer median)")
        print("Same for classification and regression.\n")
        print(summary_df.to_string(index=False))
        print(f"\nWrote summary: {summary_path}")
        print(f"Wrote by-feature: {feature_path}")
        print(f"\nCategorical features tracked: {', '.join(CATEGORICAL_FEATURES)}")
        print(categorical_summary_df.to_string(index=False))
        print(f"\nWrote categorical summary: {categorical_summary_path}")
        print(f"Wrote categorical by-feature: {categorical_feature_path}")

    if not args.no_plot:
        plot_paths = plot_imputation_outputs(
            summary_df,
            feature_df,
            args.out_dir,
            top_n=args.top_n,
            categorical_summary_df=categorical_summary_df,
            categorical_feature_df=categorical_feature_df,
        )
        for path in plot_paths:
            print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
