#!/usr/bin/env python3
"""Identifies samples present in one cohort but absent in another, computes preterm/term ratios by sample type, and plots gestational-age distributions."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import PRETERM_CUTOFF

MISSING_OUTPUT_CSV = Path("outputs/missing_samples_from_dataset1_not_in_dataset2.csv")
RATIO_OUTPUT_CSV   = Path("outputs/preterm_term_ratios_by_dataset_and_source.csv")

PLOT_HEEL_PATH = Path("outputs/scatter_ga_preterm_term_HEEL.png")
PLOT_CORD_PATH = Path("outputs/scatter_ga_preterm_term_CORD.png")

GA_COL = "gestational_age_weeks"


def normalize_source(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same Source normalization convention as in data_loader.py."""
    if "Source" not in df.columns:
        raise ValueError("Expected a 'Source' column in the dataset.")
    df["Source"] = (
        df["Source"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    return df


def make_key_series(df: pd.DataFrame) -> pd.Series:
    """
    Create a Series of composite keys for joining:
      (mother_id, child_id, Source)
    """
    for col in ["mother_id", "child_id", "Source"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in the dataset.")

    return pd.Series(
        list(zip(
            df["mother_id"].astype(str),
            df["child_id"].astype(str),
            df["Source"].astype(str),
        )),
        index=df.index,
    )


def compute_preterm_term_stats(df: pd.DataFrame, source: str) -> dict:
    """
    For a given dataframe and Source type (e.g., 'HEEL' or 'CORD'),
    compute counts of preterm vs term based on gestational_age_weeks < PRETERM_CUTOFF.

    Returns a dict with:
      total_samples, valid_ga, n_preterm, n_term,
      preterm_fraction, preterm_term_ratio
    """
    if GA_COL not in df.columns:
        raise ValueError(f"Expected gestational age column '{GA_COL}' in the dataset.")

    # Filter to desired Source (HEEL or CORD)
    sub = df[df["Source"] == source].copy()
    n_total = sub.shape[0]

    # Handle GA and preterm flag
    ga = pd.to_numeric(sub[GA_COL], errors="coerce")
    valid_ga_mask = ga.notna()
    n_valid_ga = int(valid_ga_mask.sum())

    is_preterm = ga < PRETERM_CUTOFF
    is_preterm = is_preterm & valid_ga_mask  # ensure we only count where GA is valid

    n_preterm = int(is_preterm.sum())
    n_term = int(n_valid_ga - n_preterm)

    # Fractions/ratios
    if n_valid_ga > 0:
        preterm_fraction = n_preterm / n_valid_ga
    else:
        preterm_fraction = float("nan")

    if n_term > 0:
        preterm_term_ratio = n_preterm / n_term
    else:
        preterm_term_ratio = float("nan")

    return {
        "total_samples": n_total,
        "valid_ga": n_valid_ga,
        "n_preterm": n_preterm,
        "n_term": n_term,
        "preterm_fraction": preterm_fraction,
        "preterm_term_ratio": preterm_term_ratio,
    }


def build_combined_for_plot(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Build a combined dataframe from dataset1 and dataset2 with columns:
      - gestational_age_weeks
      - dataset_label: 'training_cohort' or 'validation_cohort'
      - is_preterm: bool
      - Source: 'HEEL' or 'CORD'
    """
    # Prepare Dataset 1
    d1 = df1.copy()
    d1["dataset_label"] = "training_cohort"

    # Prepare Dataset 2
    d2 = df2.copy()
    d2["dataset_label"] = "validation_cohort"

    combined = pd.concat([d1, d2], axis=0, ignore_index=True)

    if GA_COL not in combined.columns:
        raise ValueError(f"Expected gestational age column '{GA_COL}' in the dataset.")

    ga = pd.to_numeric(combined[GA_COL], errors="coerce")
    combined["ga_numeric"] = ga
    combined["is_preterm"] = ga < PRETERM_CUTOFF  # True if preterm, False if term

    # Only keep rows with valid GA
    combined = combined[combined["ga_numeric"].notna()].copy()

    return combined


def plot_ga_preterm_term_by_source(combined: pd.DataFrame, source: str, out_path: Path):
    """
    Make a scatter plot for a given Source ('HEEL' or 'CORD'):
      - x-axis: ga_numeric
      - y-axis: 0 (term) / 1 (preterm), with jitter
      - color: dataset_label
      - marker: 'o' for term, 'x' for preterm
    """
    sub = combined[combined["Source"] == source].copy()
    if sub.empty:
        print(f"[WARN] No rows found for Source={source}; skipping plot.")
        return

    # Map preterm/term to y-position
    sub["y_base"] = np.where(sub["is_preterm"], 1.0, 0.0)
    # Add small vertical jitter so points don't overlap exactly
    rng = np.random.default_rng(42)
    sub["y_jitter"] = sub["y_base"] + rng.normal(loc=0.0, scale=0.03, size=len(sub))

    # Plot
    plt.figure(figsize=(10, 5))

    # Plot term vs preterm separately so we can use different markers
    for dataset_label, color in [("training_cohort", "tab:blue"),
                                 ("validation_cohort", "tab:orange")]:
        for is_preterm, marker, label_suffix in [
            (False, "o", f"term (>={PRETERM_CUTOFF}w)"),
            (True, "x", f"preterm (<{PRETERM_CUTOFF}w)"),
        ]:
            mask = (sub["dataset_label"] == dataset_label) & (sub["is_preterm"] == is_preterm)
            if not mask.any():
                continue
            plt.scatter(
                sub.loc[mask, "ga_numeric"],
                sub.loc[mask, "y_jitter"],
                alpha=0.7,
                label=f"{dataset_label} - {label_suffix}",
                marker=marker,
                s=30,
                color=color,
            )

    plt.yticks([0, 1], [f"Term (>={PRETERM_CUTOFF}w)", f"Preterm (<{PRETERM_CUTOFF}w)"])
    plt.xlabel("Gestational age (weeks)")
    plt.ylabel("Preterm vs term (with jitter)")
    plt.title(f"Gestational age distribution by dataset and outcome ({source} samples)")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path.resolve()}")


def main():
    """Load two cohort CSVs, identify missing samples, compute preterm ratios, and save plots."""
    parser = argparse.ArgumentParser(
        description="Compare two cohort datasets and identify sample overlap."
    )
    parser.add_argument("--dataset1", required=True, help="Path to the first (training) cohort CSV.")
    parser.add_argument("--dataset2", required=True, help="Path to the second (validation) cohort CSV.")
    args = parser.parse_args()

    dataset1_path = Path(args.dataset1)
    dataset2_path = Path(args.dataset2)

    print("Loading datasets...")
    df1 = pd.read_csv(dataset1_path)
    df2 = pd.read_csv(dataset2_path)

    # Normalize Source to upper-case 'HEEL' / 'CORD'
    df1 = normalize_source(df1)
    df2 = normalize_source(df2)

    # ---------- PART 1: Missing samples from Dataset 1 vs Dataset 2 ----------
    print("\n=== Part 1: Missing samples from Dataset 1 that are not in Dataset 2 ===")

    keys1 = set(make_key_series(df1))
    keys2 = set(make_key_series(df2))

    missing_keys = sorted(keys1 - keys2)
    print(f"Total samples in Dataset 1 but NOT in Dataset 2: {len(missing_keys)}")

    if missing_keys:
        key_series1 = make_key_series(df1)
        missing_df = df1.loc[key_series1.isin(missing_keys)].copy()

        # Create preterm flag for missing samples
        ga_missing = pd.to_numeric(missing_df[GA_COL], errors="coerce")
        missing_df["is_preterm"] = ga_missing < PRETERM_CUTOFF

        # Count missing samples by Source (HEEL / CORD)
        for src in ["HEEL", "CORD"]:
            sub = missing_df[missing_df["Source"] == src]
            ga = pd.to_numeric(sub[GA_COL], errors="coerce")
            valid_ga = ga.notna()

            n_total = sub.shape[0]
            n_valid_ga = int(valid_ga.sum())
            n_preterm = int((ga[valid_ga] < PRETERM_CUTOFF).sum())
            n_term = int(n_valid_ga - n_preterm)

            print(f"\n--- Missing {src} samples ---")
            print(f"Total missing {src} samples: {n_total}")
            print(f"Valid GA entries:            {n_valid_ga}")
            print(f"  Preterm (<{PRETERM_CUTOFF}w):            {n_preterm}")
            print(f"  Term (>={PRETERM_CUTOFF}w):              {n_term}")

        # Save detailed missing rows
        MISSING_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        missing_df.to_csv(MISSING_OUTPUT_CSV, index=False)
        print(f"\nSaved detailed missing-sample CSV to: {MISSING_OUTPUT_CSV.resolve()}")
    else:
        print("No missing samples detected (Dataset 2 covers all (mother, child, Source) in Dataset 1).")
        missing_df = pd.DataFrame([])  # empty, for completeness

    # ---------- PART 2: Preterm/Term ratios in each dataset, by Source ----------
    print("\n=== Part 2: Preterm/Term ratios by dataset and sample type (HEEL vs CORD) ===")

    rows = []
    for dataset_label, df in [("training_cohort", df1), ("validation_cohort", df2)]:
        for src in ["HEEL", "CORD"]:
            stats = compute_preterm_term_stats(df, src)
            row = {
                "dataset": dataset_label,
                "source": src,
                **stats,
            }
            rows.append(row)

            print(f"\n[{dataset_label} | {src}]")
            print(f"  Total samples:        {stats['total_samples']}")
            print(f"  Valid GA entries:     {stats['valid_ga']}")
            print(f"  Preterm (<{PRETERM_CUTOFF}w):       {stats['n_preterm']}")
            print(f"  Term (>={PRETERM_CUTOFF}w):         {stats['n_term']}")
            if stats['valid_ga'] > 0:
                print(f"  Preterm fraction:     {stats['preterm_fraction']:.3f}")
            else:
                print("  Preterm fraction:     NaN")
            if stats['n_term'] > 0:
                print(f"  Preterm/Term ratio:   {stats['preterm_term_ratio']:.3f}")
            else:
                print("  Preterm/Term ratio:   NaN (no term babies)")

    ratio_df = pd.DataFrame(rows)
    RATIO_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ratio_df.to_csv(RATIO_OUTPUT_CSV, index=False)
    print(f"\nSaved preterm/term ratio summary CSV to: {RATIO_OUTPUT_CSV.resolve()}")

    # ---------- PART 3: Scatter plots for HEEL and CORD ----------
    print("\n=== Part 3: Scatter plots of GA vs preterm/term by dataset ===")
    combined_for_plot = build_combined_for_plot(df1, df2)

    # HEEL-only plot
    plot_ga_preterm_term_by_source(combined_for_plot, source="HEEL", out_path=PLOT_HEEL_PATH)

    # CORD-only plot
    plot_ga_preterm_term_by_source(combined_for_plot, source="CORD", out_path=PLOT_CORD_PATH)


if __name__ == "__main__":
    main()
