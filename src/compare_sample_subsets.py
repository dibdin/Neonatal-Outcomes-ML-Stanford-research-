#!/usr/bin/env python3
"""Extra analyses comparing heel sample subsets: GA distributions, feature effect sizes,
birth weight distributions, GA vs birth weight scatter, and stratified AUC.

Uses composite sample ID: (mother_id, child_id, Source) matching the comparison script.

python src/compare_sample_subsets.py
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from src when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# Import feature lists from data_loader to match preprocessing
try:
    from src.data_loader import FEATURES_TO_DROP, BIOMARKER_COLS, DROP_SET
except ImportError:
    # Fallback if import fails
    FEATURES_TO_DROP = [
        "_17PAnd_Cort", "_17OHP", "Androstenedione", "Cortisol", "_11_DC", "_21_DC",
        "TREC2", "TBX1", "RNASEP"
    ]
    DROP_SET = set(FEATURES_TO_DROP)
    BIOMARKER_COLS = [
        "TSH","N17P","_17PAnd_Cort","_17OHP","Androstenedione","Cortisol","_11_DC","_21_DC",
        "IRT","GALT","BIOT","Ala","Arg","ASA","ASA_Arg","ASA_Orn","Cit","Cit_Arg","Cit_Orn",
        "Cit_Tyr","Gly","LEU","Leu_Ala","Leu_Phe","MET","Met_Phe","Orn","Orn_Arg","Orn_Cit",
        "Orn_Phe","PHE","PHE_TYR","SUAC","TYR","Tyr_Phe","Val","Val_Ala","Val_Phe","C0",
        "C0_C16C18","C0C2C3C16C18_Cit","C2","C3","C3_C0","C3_C16","C3_C2","C3_C4DC","MCA",
        "C3DC","C4","C4DC","C4OH","C5","C5_C0","C5_C2","C5_C3","C5_1","C5DC","C5DC_C16",
        "C5DC_C5OH","C5DC_C8","C5OH","C5OH_C2","C5OH_C5_1","C5OH_C8","C6","C6DC","C8","C8_C10",
        "C8_C2","C8_1","C10","C10_1","C12","C12_1","C14","C14OH","C14_1","C14_1_C12_1",
        "C14_1_C16","C14_1_C4","C14_2","C16","C16_1OH","C16_1OH_C4DC","C16OH","C16OH_C16",
        "C18","C18_1","C18_1OH","C18_2","C18OH","TREC","TREC2","TBX1","RNASEP",
        "HGB___FAST","HGB___F1","HGB___F","HGB___F_F1","HGB___A","HGB___FAST_F1","HGB___Other",
    ]
else:
    # If import succeeded, ensure DROP_SET is set
    DROP_SET = set(FEATURES_TO_DROP)

# Import preterm cutoff from config
try:
    from src.config import PRETERM_CUTOFF
except ImportError:
    PRETERM_CUTOFF = 37  # Fallback

GA_COL = "gestational_age_weeks"


# ----------------------------------------------------------------------
#  COMPOSITE KEY (mother_id, child_id, Source)
# ----------------------------------------------------------------------

def make_sample_key(df: pd.DataFrame, key_name="sample_key") -> pd.DataFrame:
    """
    Build composite sample identity:
        (mother_id, child_id, Source)
    Exactly matching the comparison script.
    """
    for col in ["mother_id", "child_id", "Source"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' for composite key.")

    df = df.copy()
    df[key_name] = list(
        zip(
            df["mother_id"].astype(str),
            df["child_id"].astype(str),
            df["Source"].astype(str).str.upper().str.strip(),
        )
    )
    return df


# ----------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------

def normalize_source(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Source to 'HEEL' / 'CORD' like in data_loader."""
    if "Source" not in df.columns:
        raise ValueError("Expected a 'Source' column in the dataset.")
    df = df.copy()
    df["Source"] = (
        df["Source"]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    return df


# ----------------------------------------------------------------------
#  Loading heel subsets
# ----------------------------------------------------------------------

def load_heel_subsets(full_csv: Path, both_csv: Path):
    """
    Load full and 'both-samples' datasets and return HEEL-only subsets with sample_key.
    Applies the same preprocessing steps as data_loader.py:
    1. Drop date_transfusion column
    2. Filter invalid birth weights
    3. Drop fixed problematic features
    """
    df_full = pd.read_csv(full_csv)
    df_both = pd.read_csv(both_csv)

    if "Source" not in df_full.columns or "Source" not in df_both.columns:
        raise ValueError("Both CSVs must have a 'Source' column indicating HEEL/CORD.")

    # Drop date_transfusion if present (matching data_loader.py)
    if 'date_transfusion' in df_full.columns:
        df_full = df_full.drop(columns=['date_transfusion'])
    if 'date_transfusion' in df_both.columns:
        df_both = df_both.drop(columns=['date_transfusion'])

    # Normalize Source column (matching analyze_heel_feature_quality.py)
    df_full = normalize_source(df_full)
    df_both = normalize_source(df_both)

    # Filter for HEEL only
    heel_full = df_full[df_full["Source"] == "HEEL"].copy()
    heel_both = df_both[df_both["Source"] == "HEEL"].copy()

    # Filter invalid birth weights (matching data_loader.py step 1)
    if 'birth_weight_kg' in heel_full.columns:
        invalid_bw = (
            (heel_full['birth_weight_kg'] == 99.9) |
            (heel_full['birth_weight_kg'] < 0.5) |
            (heel_full['birth_weight_kg'] > 6)
        )
        if invalid_bw.any():
            print(f"[heel_full] Filtering out {invalid_bw.sum()} rows with invalid birth_weight_kg values.")
            heel_full = heel_full[~invalid_bw]

    if 'birth_weight_kg' in heel_both.columns:
        invalid_bw = (
            (heel_both['birth_weight_kg'] == 99.9) |
            (heel_both['birth_weight_kg'] < 0.5) |
            (heel_both['birth_weight_kg'] > 6)
        )
        if invalid_bw.any():
            print(f"[heel_both] Filtering out {invalid_bw.sum()} rows with invalid birth_weight_kg values.")
            heel_both = heel_both[~invalid_bw]

    # Drop fixed problematic features (matching data_loader.py step 2)
    to_drop_full = [c for c in FEATURES_TO_DROP if c in heel_full.columns]
    to_drop_both = [c for c in FEATURES_TO_DROP if c in heel_both.columns]

    if to_drop_full:
        print(f"[heel_full] Dropping fixed features: {len(to_drop_full)} -> {to_drop_full}")
        heel_full = heel_full.drop(columns=to_drop_full)
    if to_drop_both:
        print(f"[heel_both] Dropping fixed features: {len(to_drop_both)} -> {to_drop_both}")
        heel_both = heel_both.drop(columns=to_drop_both)

    # Add composite ID (after preprocessing to ensure consistency)
    heel_full = make_sample_key(heel_full, "sample_key")
    heel_both = make_sample_key(heel_both, "sample_key")

    return heel_full, heel_both


# ----------------------------------------------------------------------
#  Feature Selection (matching analyze_heel_feature_quality.py)
# ----------------------------------------------------------------------

def pick_common_numeric_features(heel_full: pd.DataFrame,
                                 heel_both: pd.DataFrame,
                                 ga_col: str = GA_COL):
    """
    Return list of numeric feature names present in BOTH heel datasets,
    using BIOMARKER_COLS list (matching data_loader.py preprocessing).

    This matches data_loader.py biomarker feature selection:
    - Uses BIOMARKER_COLS list
    - Excludes features in DROP_SET (already dropped in load_heel_subsets)
    - Only includes features present in both datasets
    - Excludes gestational age target column
    - For non-biomarker numeric features, only includes: birth_weight_kg, sex, multiple_birth
    """
    # Use BIOMARKER_COLS list (matching data_loader.py biomarker selection)
    # Filter: must be in BIOMARKER_COLS, present in both datasets, and not in DROP_SET
    biomarker_cols = [
        c for c in BIOMARKER_COLS
        if c in heel_full.columns
        and c in heel_both.columns
        and c not in DROP_SET
    ]

    # Only include specified clinical numeric features (matching data_loader.py)
    allowed_clinical_features = ["birth_weight_kg", "sex", "multiple_birth", "gestational_age_weeks"]
    clinical_cols = [
        c for c in allowed_clinical_features
        if c in heel_full.columns
        and c in heel_both.columns
        and c != ga_col  # Exclude GA as it's the target
    ]

    # Combine biomarker and clinical features
    all_features = sorted(set(biomarker_cols) | set(clinical_cols))
    features = [c for c in all_features if c != ga_col and c not in DROP_SET]

    return features


# ----------------------------------------------------------------------
#  Analysis 3.1 — GA distributions
# ----------------------------------------------------------------------

def plot_gestational_age_distributions(heel_full, heel_both, out_dir: Path):
    """Plot overlapping histograms of gestational age for the two heel subsets."""
    ga_col = "gestational_age_weeks"
    if ga_col not in heel_full or ga_col not in heel_both:
        print(f"[3.1] SKIP: '{ga_col}' missing.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    ga_full = pd.to_numeric(heel_full[ga_col], errors="coerce").dropna()
    ga_both = pd.to_numeric(heel_both[ga_col], errors="coerce").dropna()

    plt.figure(figsize=(6,4))
    plt.hist(ga_full, bins=20, alpha=0.5, label="Heel – all (opt2)")
    plt.hist(ga_both, bins=20, alpha=0.5, label="Heel – both (opt1)")
    plt.axvline(37, color="k", linestyle="--")
    plt.xlabel("Gestational age (weeks)")
    plt.ylabel("Count")
    plt.title("GA distribution: heel all vs heel both")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "heel_ga_hist_full_vs_both.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[3.1] Saved GA histogram: {out_path}")


# ----------------------------------------------------------------------
#  Analysis 3.2 — Cohen's d
# ----------------------------------------------------------------------

def cohen_d(pre, term):
    """Compute Cohen's d effect size between preterm and term value arrays."""
    pre, term = np.asarray(pre), np.asarray(term)
    pre = pre[np.isfinite(pre)]
    term = term[np.isfinite(term)]

    if pre.size < 2 or term.size < 2:
        return np.nan

    m1, m2 = pre.mean(), term.mean()
    v1, v2 = pre.var(ddof=1), term.var(ddof=1)
    pooled = ((pre.size-1)*v1 + (term.size-1)*v2) / (pre.size + term.size - 2)

    if pooled <= 0:
        return np.nan

    return (m1 - m2) / np.sqrt(pooled)


def plot_feature_effect_sizes(heel_full, heel_both, features, out_dir: Path):
    """Compute and save Cohen's d for each feature comparing preterm vs term within each dataset."""
    ga_col = "gestational_age_weeks"
    if ga_col not in heel_full or ga_col not in heel_both:
        print("[3.2] SKIP: GA column missing.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for feat in features:
        for ds_name, df in [("heel_all_opt2", heel_full), ("heel_both_opt1", heel_both)]:
            if feat not in df.columns:
                print(f"[3.2] WARN: '{feat}' missing in {ds_name}.")
                continue

            ga = pd.to_numeric(df[ga_col], errors="coerce")
            pre_vals = pd.to_numeric(df.loc[ga < PRETERM_CUTOFF, feat], errors="coerce").dropna()
            term_vals = pd.to_numeric(df.loc[ga >= PRETERM_CUTOFF, feat], errors="coerce").dropna()

            d = cohen_d(pre_vals, term_vals)

            rows.append({
                "dataset": ds_name,
                "feature": feat,
                "n_preterm": pre_vals.size,
                "n_term": term_vals.size,
                "mean_preterm": pre_vals.mean() if pre_vals.size else np.nan,
                "mean_term": term_vals.mean() if term_vals.size else np.nan,
                "cohen_d": d
            })

    df_out = pd.DataFrame(rows)
    out_csv = out_dir / "heel_feature_effect_sizes_preterm_vs_term.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"[3.2] Saved: {out_csv}")


# ----------------------------------------------------------------------
#  Analysis 3.3 — Birth Weight distribution comparison
# ----------------------------------------------------------------------

def plot_birth_weight_distributions(heel_full, heel_both, out_dir: Path):
    """
    Plot birth weight distributions for heel_all vs heel_both datasets.
    Creates histogram and boxplot comparisons.
    """
    bw_col = "birth_weight_kg"

    if bw_col not in heel_full.columns or bw_col not in heel_both.columns:
        print(f"[3.3] SKIP: '{bw_col}' missing.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract numeric values, handling any non-numeric entries
    bw_full = pd.to_numeric(heel_full[bw_col], errors="coerce").dropna()
    bw_both = pd.to_numeric(heel_both[bw_col], errors="coerce").dropna()

    if len(bw_full) == 0 or len(bw_both) == 0:
        print(f"[3.3] SKIP: No valid birth weight data.")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Histogram comparison ---
    ax1 = axes[0]
    ax1.hist(bw_full, bins=30, alpha=0.6, label=f"Heel – all (opt2), n={len(bw_full)}",
             color='blue', edgecolor='black', linewidth=0.5)
    ax1.hist(bw_both, bins=30, alpha=0.6, label=f"Heel – both (opt1), n={len(bw_both)}",
             color='orange', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel("Birth weight (kg)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Birth weight distribution: heel all vs heel both", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add summary statistics as text
    stats_text = (
        f"Heel – all:\n"
        f"  Mean: {bw_full.mean():.3f} kg\n"
        f"  Median: {bw_full.median():.3f} kg\n"
        f"  Std: {bw_full.std():.3f} kg\n"
        f"\nHeel – both:\n"
        f"  Mean: {bw_both.mean():.3f} kg\n"
        f"  Median: {bw_both.median():.3f} kg\n"
        f"  Std: {bw_both.std():.3f} kg"
    )
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # --- Boxplot comparison ---
    ax2 = axes[1]
    box_data = [bw_full, bw_both]
    box_labels = [f"Heel – all (opt2)\nn={len(bw_full)}", f"Heel – both (opt1)\nn={len(bw_both)}"]
    bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                     widths=0.6, showmeans=True, meanline=True)

    # Color the boxes
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel("Birth weight (kg)", fontsize=12)
    ax2.set_title("Birth weight boxplot: heel all vs heel both", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = out_dir / "heel_birth_weight_hist_full_vs_both.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[3.3] Saved birth weight distribution plot: {out_path}")

    # Print summary statistics
    print(f"\n[3.3] Birth Weight Summary Statistics:")
    print(f"  Heel – all (opt2):")
    print(f"    n={len(bw_full)}, mean={bw_full.mean():.3f} kg, median={bw_full.median():.3f} kg, "
          f"std={bw_full.std():.3f} kg")
    print(f"    min={bw_full.min():.3f} kg, max={bw_full.max():.3f} kg")
    print(f"  Heel – both (opt1):")
    print(f"    n={len(bw_both)}, mean={bw_both.mean():.3f} kg, median={bw_both.median():.3f} kg, "
          f"std={bw_both.std():.3f} kg")
    print(f"    min={bw_both.min():.3f} kg, max={bw_both.max():.3f} kg")


# ----------------------------------------------------------------------
#  Analysis 3.4 — GA vs Birth Weight scatter plot
# ----------------------------------------------------------------------

def plot_ga_birthweight_scatter(heel_full, heel_both, out_dir: Path):
    """
    Plot gestational age vs birth weight for both heel_all and heel_both datasets.
    Also compute and display R² (squared Pearson correlation) between GA and BW
    for each dataset.
    """
    ga_col = "gestational_age_weeks"
    bw_col = "birth_weight_kg"

    if ga_col not in heel_full.columns or ga_col not in heel_both.columns:
        print(f"[3.4] SKIP: '{ga_col}' missing.")
        return
    if bw_col not in heel_full.columns or bw_col not in heel_both.columns:
        print(f"[3.4] SKIP: '{bw_col}' missing.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Extract numeric values, handling any non-numeric entries
    ga_full = pd.to_numeric(heel_full[ga_col], errors="coerce")
    bw_full = pd.to_numeric(heel_full[bw_col], errors="coerce")

    ga_both = pd.to_numeric(heel_both[ga_col], errors="coerce")
    bw_both = pd.to_numeric(heel_both[bw_col], errors="coerce")

    # Create mask for valid pairs (both GA and BW are not NaN)
    valid_full = ga_full.notna() & bw_full.notna()
    valid_both = ga_both.notna() & bw_both.notna()

    # Compute Pearson correlation and R^2 for each dataset
    r2_full = np.nan
    r2_both = np.nan
    try:
        if valid_full.sum() > 1:
            r_full = np.corrcoef(ga_full[valid_full], bw_full[valid_full])[0, 1]
            if np.isfinite(r_full):
                r2_full = float(r_full ** 2)
        if valid_both.sum() > 1:
            r_both = np.corrcoef(ga_both[valid_both], bw_both[valid_both])[0, 1]
            if np.isfinite(r_both):
                r2_both = float(r_both ** 2)
    except Exception as e:
        print(f"[3.4] WARN: Failed to compute R^2: {e}")

    plt.figure(figsize=(10, 6))

    # Plot both datasets with different colors and transparency
    label_full = f"Heel – all (opt2), n={int(valid_full.sum())}"
    if np.isfinite(r2_full):
        label_full += f" (R^2={r2_full:.2f})"
    plt.scatter(
        ga_full[valid_full],
        bw_full[valid_full],
        alpha=0.5,
        s=20,
        label=label_full,
        color="blue",
    )

    label_both = f"Heel – both (opt1), n={int(valid_both.sum())}"
    if np.isfinite(r2_both):
        label_both += f" (R^2={r2_both:.2f})"
    plt.scatter(
        ga_both[valid_both],
        bw_both[valid_both],
        alpha=0.5,
        s=20,
        label=label_both,
        color="orange",
    )

    # Add vertical line at preterm cutoff (preterm threshold)
    plt.axvline(PRETERM_CUTOFF, color="k", linestyle="--", linewidth=1, label=f"Preterm threshold ({PRETERM_CUTOFF} weeks)")

    plt.xlabel("Gestational age (weeks)", fontsize=12)
    plt.ylabel("Birth weight (kg)", fontsize=12)
    plt.title("Gestational age vs Birth weight: heel all vs heel both", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = out_dir / "heel_ga_vs_bw_scatter.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[3.4] Saved GA vs BW scatter plot: {out_path}")


# ----------------------------------------------------------------------
#  Analysis 3.5 — Stratified AUC
# ----------------------------------------------------------------------

def evaluate_stratified_auc(
    heel_full,
    heel_both,
    pred_csv: Path,
    proba_col: str,
    ytrue_col: str,
    out_dir: Path,
):
    """Compute and save AUC stratified by whether a baby had both samples or heel-only."""
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pred = pd.read_csv(pred_csv)

    # Attach composite key to prediction CSV (matching run_all)
    if "sample_key" not in df_pred.columns:
        # We require prediction CSV to already contain sample_key!
        raise ValueError(
            "[3.5] Prediction CSV MUST contain 'sample_key'. "
            "Regenerate the per-baby prediction file including composite sample_key."
        )

    # Build sets for grouping
    ids_full = set(heel_full["sample_key"])
    ids_both = set(heel_both["sample_key"])

    df_pred["_key"] = df_pred["sample_key"].apply(tuple)

    def grp(k):
        if k in ids_both:
            return "both_samples"
        elif k in ids_full:
            return "heel_only"
        else:
            return "unknown"

    df_pred["group"] = df_pred["_key"].map(grp)

    # Overall AUC
    overall_auc = roc_auc_score(df_pred[ytrue_col], df_pred[proba_col])
    print(f"[3.5] Overall AUC: {overall_auc:.3f}")

    rows = []
    for grp_name in ["both_samples", "heel_only"]:
        sub = df_pred[df_pred["group"] == grp_name]
        if len(sub) < 2 or sub[ytrue_col].nunique() < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(sub[ytrue_col], sub[proba_col])

        rows.append({"group": grp_name, "n": len(sub), "auc": auc})
        print(f"[3.5] {grp_name}: n={len(sub)}, AUC={auc}")

    out_csv = out_dir / "heel_opt2_auc_by_group.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[3.5] Saved AUC table: {out_csv}")


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

def main():
    """Parse arguments, load heel subsets, and run all comparative analyses."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_csv", required=True)
    ap.add_argument("--both_csv", required=True)
    ap.add_argument("--pred_csv")
    ap.add_argument("--features", default="birth_weight_kg")
    ap.add_argument("--proba_col", default="mean_proba")
    ap.add_argument("--ytrue_col", default="y_true")
    ap.add_argument("--out_dir", default="outputs/heel_extra_analysis")

    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    heel_full, heel_both = load_heel_subsets(Path(args.full_csv), Path(args.both_csv))

    # Use the same feature selection as analyze_heel_feature_quality.py
    # If features are provided via command line, use those; otherwise use all common features
    if args.features and args.features != "birth_weight_kg":  # Default is just birth_weight_kg
        features = [f.strip() for f in args.features.split(",") if f.strip()]
    else:
        # Use the same feature selection logic as analyze_heel_feature_quality.py
        features = pick_common_numeric_features(heel_full, heel_both, ga_col=GA_COL)
        print(f"[INFO] Using {len(features)} common features (biomarkers + clinical: birth_weight_kg, sex, multiple_birth)")

    print(f"[INFO] heel_all n={len(heel_full)}, heel_both n={len(heel_both)}")

    plot_gestational_age_distributions(heel_full, heel_both, out_dir)
    plot_feature_effect_sizes(heel_full, heel_both, features, out_dir)
    plot_birth_weight_distributions(heel_full, heel_both, out_dir)
    plot_ga_birthweight_scatter(heel_full, heel_both, out_dir)

    if args.pred_csv:
        evaluate_stratified_auc(
            heel_full,
            heel_both,
            pred_csv=Path(args.pred_csv),
            proba_col=args.proba_col,
            ytrue_col=args.ytrue_col,
            out_dir=out_dir,
        )
    else:
        print("[3.5] No pred_csv provided; skipping stratified AUC.")


if __name__ == "__main__":
    main()
