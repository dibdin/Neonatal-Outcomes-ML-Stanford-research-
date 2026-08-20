#!/usr/bin/env python3
"""Extra analyses comparing heel_all (opt2) vs heel_both (opt1): missingness per feature,
variance per feature, correlation with gestational age, PCA visualization, CV AUC, and
correlations with both GA and birth weight.

Only uses rows with Source == HEEL (after normalization).
Uses all numeric features common to both heel subsets, except gestational_age_weeks.

python src/compare_feature_quality.py
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import from src when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Import SGA classification with graceful fallback
from src.config import PRETERM_CUTOFF

try:
    from src.sga_classification import calculate_sga_classification_intergrowth21
    _HAS_SGA = True
except ImportError:
    _HAS_SGA = False

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


GA_COL = "gestational_age_weeks"


# ---------- helpers ---------- #

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


def load_heel_subsets(full_csv: Path, both_csv: Path):
    """
    Load full & both-samples CSVs and return HEEL-only subsets.
    Applies the same preprocessing steps as data_loader.py:
    1. Drop date_transfusion column
    2. Filter invalid birth weights
    3. Drop fixed problematic features
    """
    # Load and normalize Source
    df_full = normalize_source(pd.read_csv(full_csv))
    df_both = normalize_source(pd.read_csv(both_csv))

    # Drop date_transfusion if present (matching data_loader.py)
    if 'date_transfusion' in df_full.columns:
        df_full = df_full.drop(columns=['date_transfusion'])
    if 'date_transfusion' in df_both.columns:
        df_both = df_both.drop(columns=['date_transfusion'])

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

    return heel_full, heel_both


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


# ---------- Test A: Missingness per feature ---------- #

def compare_missingness(heel_full, heel_both, features, out_dir: Path):
    """Compute per-feature missingness fractions and save long and pivot CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for col in features + [GA_COL]:
        for ds_name, df in [("heel_all_opt2", heel_full), ("heel_both_opt1", heel_both)]:
            if col not in df.columns:
                continue
            series = df[col]
            n_total = len(series)
            n_missing = series.isna().sum()
            frac_missing = n_missing / n_total if n_total > 0 else np.nan
            rows.append({
                "dataset": ds_name,
                "feature": col,
                "n_total": n_total,
                "n_missing": n_missing,
                "frac_missing": frac_missing,
            })

    if not rows:
        print("[A] No missingness rows computed.")
        return

    df_long = pd.DataFrame(rows)

    # Pivot to compare side-by-side
    pivot = df_long.pivot(index="feature",
                          columns="dataset",
                          values="frac_missing")
    pivot["diff_heel_all_minus_both"] = (
        pivot.get("heel_all_opt2") - pivot.get("heel_both_opt1")
    )

    out_csv_long = out_dir / "A_missingness_long.csv"
    out_csv_pivot = out_dir / "A_missingness_pivot.csv"
    df_long.to_csv(out_csv_long, index=False)
    pivot.to_csv(out_csv_pivot)
    print(f"[A] Saved missingness tables:\n  {out_csv_long}\n  {out_csv_pivot}")


# ---------- Test B: Variance per feature ---------- #

def compare_variance(heel_full, heel_both, features, out_dir: Path):
    """
    Calculate variance per feature after imputation and scaling (matching Test D and E).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for ds_name, df in [("heel_all_opt2", heel_full), ("heel_both_opt1", heel_both)]:
        # Prepare features (same preprocessing as Test D)
        X = df[features].copy()

        # Convert categorical variables to numeric (matching data_loader.py)
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if col in X.columns:
                    X[col] = pd.Categorical(X[col]).codes

        # Impute + scale (matching Test D and data_loader.py)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_imp = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imp)

        # Convert back to DataFrame for easier processing
        X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=df.index)

        # Calculate variance for each feature after imputation and scaling
        for col in features:
            if col not in X_scaled_df.columns:
                continue
            vals = X_scaled_df[col].values
            var = float(np.var(vals, ddof=1)) if len(vals) > 1 else np.nan
            rows.append({
                "dataset": ds_name,
                "feature": col,
                "n": int(len(vals)),
                "variance": var,
            })

    if not rows:
        print("[B] No variance rows computed.")
        return

    df_long = pd.DataFrame(rows)
    pivot = df_long.pivot(index="feature",
                          columns="dataset",
                          values="variance")
    pivot["var_ratio_all_over_both"] = (
        pivot.get("heel_all_opt2") / pivot.get("heel_both_opt1")
    )

    out_csv_long = out_dir / "B_variance_long.csv"
    out_csv_pivot = out_dir / "B_variance_pivot.csv"
    df_long.to_csv(out_csv_long, index=False)
    pivot.to_csv(out_csv_pivot)
    print(f"[B] Saved variance tables:\n  {out_csv_long}\n  {out_csv_pivot}")


# ---------- Test C: Correlation of each feature with GA ---------- #

def _corr_with_ga_scaled(X_scaled_df: pd.DataFrame, ga_values: np.ndarray, feat: str):
    """Calculate Pearson correlation between a scaled feature and GA values."""
    if feat not in X_scaled_df.columns:
        return np.nan
    x = X_scaled_df[feat].values
    # Both x and ga should be aligned (same length after imputation)
    if len(x) != len(ga_values):
        return np.nan
    mask = np.isfinite(x) & np.isfinite(ga_values)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[mask], ga_values[mask])[0, 1])


def compare_ga_correlation(heel_full, heel_both, features, out_dir: Path):
    """
    Calculate correlation of each feature with GA after imputation and scaling (matching Test D and E).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process both datasets and store correlations
    corr_dict = {}

    for ds_name, df in [("heel_all_opt2", heel_full), ("heel_both_opt1", heel_both)]:
        # Prepare features (same preprocessing as Test D)
        X = df[features].copy()

        # Get GA values
        ga = pd.to_numeric(df[GA_COL], errors="coerce")

        # Convert categorical variables to numeric (matching data_loader.py)
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if col in X.columns:
                    X[col] = pd.Categorical(X[col]).codes

        # Impute + scale (matching Test D and data_loader.py)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_imp = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imp)

        # Convert back to DataFrame for easier processing
        X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=df.index)

        # Impute GA values to match the imputed feature rows
        ga_imp = ga.fillna(ga.median()).values

        # Calculate correlation for each feature
        for feat in features:
            corr = _corr_with_ga_scaled(X_scaled_df, ga_imp, feat)
            if feat not in corr_dict:
                corr_dict[feat] = {}
            corr_dict[feat][ds_name] = corr

    # Build rows from correlation dictionary
    rows = []
    for feat in features:
        corr_full = corr_dict.get(feat, {}).get("heel_all_opt2", np.nan)
        corr_both = corr_dict.get(feat, {}).get("heel_both_opt1", np.nan)
        rows.append({
            "feature": feat,
            "corr_full_heel_all_opt2": corr_full,
            "corr_both_heel_both_opt1": corr_both,
            "diff_corr_full_minus_both": (
                corr_full - corr_both
                if np.all(np.isfinite([corr_full, corr_both]))
                else np.nan
            ),
        })

    df_corr = pd.DataFrame(rows)
    out_csv = out_dir / "C_corr_with_ga.csv"
    df_corr.to_csv(out_csv, index=False)
    print(f"[C] Saved GA-correlation table: {out_csv}")

    # Scatter plot corr_full vs corr_both
    valid = df_corr.dropna(subset=["corr_full_heel_all_opt2",
                                   "corr_both_heel_both_opt1"])
    if not valid.empty:
        plt.figure(figsize=(6, 6))
        plt.scatter(valid["corr_full_heel_all_opt2"],
                    valid["corr_both_heel_both_opt1"],
                    alpha=0.7)
        lim = [-1, 1]
        plt.plot(lim, lim, "k--", linewidth=1)
        plt.xlim(lim)
        plt.ylim(lim)
        plt.xlabel("Corr(feature, GA) – heel_all (opt2)")
        plt.ylabel("Corr(feature, GA) – heel_both (opt1)")
        plt.title("Correlation of features with GA\nheel_all vs heel_both")
        plt.tight_layout()
        out_png = out_dir / "C_corr_with_ga_scatter.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[C] Saved corr scatter plot: {out_png}")
    else:
        print("[C] No valid correlations to plot.")


# ---------- Test D: PCA visualization ---------- #

def compare_pca_structure(heel_full, heel_both, features, out_dir: Path):
    """Run PCA on both heel subsets combined and separately; save plots and loading CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Combine for PCA
    df_full = heel_full.copy()
    df_full["dataset_label"] = "heel_all_opt2"
    df_both = heel_both.copy()
    df_both["dataset_label"] = "heel_both_opt1"
    combined = pd.concat([df_full, df_both], axis=0, ignore_index=True)

    if GA_COL not in combined.columns:
        print("[D] SKIP: GA column not found.")
        return

    # Only keep rows with any non-NaN in feature set
    X = combined[features].copy()

    # Convert categorical variables to numeric (matching data_loader.py)
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"[D] Converting categorical to numeric: {list(categorical_cols)}")
        for col in categorical_cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes

    # Impute + scale
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    combined["_PC1"] = X_pca[:, 0]
    combined["_PC2"] = X_pca[:, 1]
    ga = pd.to_numeric(combined[GA_COL], errors="coerce")
    combined["is_preterm"] = ga < PRETERM_CUTOFF

    # Calculate SGA status using Intergrowth-21 classification (from src/sga_classification.py)
    if _HAS_SGA and 'birth_weight_kg' in combined.columns and 'sex' in combined.columns:
        bw = pd.to_numeric(combined['birth_weight_kg'], errors="coerce")
        sex = pd.to_numeric(combined['sex'], errors="coerce")

        # Use Intergrowth-21 3rd percentile SGA classification
        # This function requires: birth_weights (kg), gestational_ages (weeks), sexes (1=male, 2=female)
        sga_classification = calculate_sga_classification_intergrowth21(
            birth_weights=bw.values,
            gestational_ages=ga.values,
            sexes=sex.values
        )
        combined["is_sga"] = pd.Series(sga_classification, index=combined.index).astype(bool)

        # Handle NaN values (samples outside Intergrowth-21 GA range 24-46 weeks)
        combined["is_sga"] = combined["is_sga"].fillna(False)
    else:
        combined["is_sga"] = False
        if not _HAS_SGA:
            print("[D] WARNING: sga_classification module not available; SGA status set to False")
        else:
            if 'birth_weight_kg' not in combined.columns:
                print("[D] WARNING: 'birth_weight_kg' column not found, cannot calculate SGA status")
            if 'sex' not in combined.columns:
                print("[D] WARNING: 'sex' column not found, cannot calculate SGA status")

    plt.figure(figsize=(7, 6))
    for ds_name, color in [("heel_all_opt2", "tab:blue"),
                           ("heel_both_opt1", "tab:orange")]:
        for is_pre, marker, label_suffix in [
            (False, "o", "term"),
            (True, "x", "preterm"),
        ]:
            mask = (combined["dataset_label"] == ds_name) & (combined["is_preterm"] == is_pre)
            if not mask.any():
                continue
            plt.scatter(
                combined.loc[mask, "_PC1"],
                combined.loc[mask, "_PC2"],
                alpha=0.6,
                s=25,
                color=color,
                marker=marker,
                label=f"{ds_name} – {label_suffix}",
            )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of heel features\n(heel_all vs heel_both)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_png = out_dir / "D_pca_heel_all_vs_both.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[D] Saved PCA plot: {out_png}")

    # ========================================================================
    # Extract loadings separately for each dataset
    # ========================================================================

    def extract_loadings_from_pca(pca_model, feature_names, dataset_name):
        """Helper function to extract and process PCA loadings."""
        n_features_pca = pca_model.components_.shape[1]

        # Match feature names to PCA components
        if len(feature_names) != n_features_pca:
            feature_names = feature_names[:n_features_pca]

        loadings_df = pd.DataFrame(
            pca_model.components_.T,  # Transpose: rows are features, columns are PCs
            columns=['PC1', 'PC2'],
            index=feature_names[:n_features_pca]
        )

        # Add absolute values for ranking
        loadings_df['abs_PC1'] = loadings_df['PC1'].abs()
        loadings_df['abs_PC2'] = loadings_df['PC2'].abs()

        # Sort by absolute loading values
        loadings_df_pc1_sorted = loadings_df.sort_values('abs_PC1', ascending=False)
        loadings_df_pc2_sorted = loadings_df.sort_values('abs_PC2', ascending=False)

        return loadings_df, loadings_df_pc1_sorted, loadings_df_pc2_sorted

    # Process heel_all dataset separately
    print("\n[D] Extracting PCA loadings for heel_all (opt2)...")
    X_full = heel_full[features].copy()
    categorical_cols_full = X_full.select_dtypes(include=['object']).columns
    if len(categorical_cols_full) > 0:
        for col in categorical_cols_full:
            if col in X_full.columns:
                X_full[col] = pd.Categorical(X_full[col]).codes

    imputer_full = SimpleImputer(strategy="median")
    scaler_full = StandardScaler()
    X_full_imp = imputer_full.fit_transform(X_full)
    X_full_scaled = scaler_full.fit_transform(X_full_imp)

    pca_full = PCA(n_components=2, random_state=42)
    pca_full.fit_transform(X_full_scaled)

    loadings_full, loadings_full_pc1, loadings_full_pc2 = extract_loadings_from_pca(
        pca_full, list(X_full.columns), "heel_all_opt2"
    )

    # Process heel_both dataset separately
    print("[D] Extracting PCA loadings for heel_both (opt1)...")
    X_both = heel_both[features].copy()
    categorical_cols_both = X_both.select_dtypes(include=['object']).columns
    if len(categorical_cols_both) > 0:
        for col in categorical_cols_both:
            if col in X_both.columns:
                X_both[col] = pd.Categorical(X_both[col]).codes

    imputer_both = SimpleImputer(strategy="median")
    scaler_both = StandardScaler()
    X_both_imp = imputer_both.fit_transform(X_both)
    X_both_scaled = scaler_both.fit_transform(X_both_imp)

    pca_both = PCA(n_components=2, random_state=42)
    pca_both.fit_transform(X_both_scaled)

    loadings_both, loadings_both_pc1, loadings_both_pc2 = extract_loadings_from_pca(
        pca_both, list(X_both.columns), "heel_both_opt1"
    )

    # Save separate loadings for each dataset
    out_csv_full = out_dir / "D_pca_loadings_heel_all_opt2.csv"
    loadings_full.to_csv(out_csv_full)
    print(f"[D] Saved heel_all loadings: {out_csv_full}")

    out_csv_both = out_dir / "D_pca_loadings_heel_both_opt1.csv"
    loadings_both.to_csv(out_csv_both)
    print(f"[D] Saved heel_both loadings: {out_csv_both}")

    # Save top contributors for each dataset
    out_csv_full_pc1 = out_dir / "D_pca_loadings_heel_all_opt2_top_PC1.csv"
    loadings_full_pc1.to_csv(out_csv_full_pc1)
    out_csv_full_pc2 = out_dir / "D_pca_loadings_heel_all_opt2_top_PC2.csv"
    loadings_full_pc2.to_csv(out_csv_full_pc2)

    out_csv_both_pc1 = out_dir / "D_pca_loadings_heel_both_opt1_top_PC1.csv"
    loadings_both_pc1.to_csv(out_csv_both_pc1)
    out_csv_both_pc2 = out_dir / "D_pca_loadings_heel_both_opt1_top_PC2.csv"
    loadings_both_pc2.to_csv(out_csv_both_pc2)

    # Print top contributors for each dataset
    print(f"\n[D] Top 10 features contributing to PC1 in heel_all (explained variance: {pca_full.explained_variance_ratio_[0]:.3f}):")
    for idx, (feat, row) in enumerate(loadings_full_pc1.head(10).iterrows()):
        print(f"  {idx+1:2d}. {feat:30s} loading={row['PC1']:7.4f} (abs={row['abs_PC1']:.4f})")

    print(f"\n[D] Top 10 features contributing to PC2 in heel_all (explained variance: {pca_full.explained_variance_ratio_[1]:.3f}):")
    for idx, (feat, row) in enumerate(loadings_full_pc2.head(10).iterrows()):
        print(f"  {idx+1:2d}. {feat:30s} loading={row['PC2']:7.4f} (abs={row['abs_PC2']:.4f})")

    print(f"\n[D] Top 10 features contributing to PC1 in heel_both (explained variance: {pca_both.explained_variance_ratio_[0]:.3f}):")
    for idx, (feat, row) in enumerate(loadings_both_pc1.head(10).iterrows()):
        print(f"  {idx+1:2d}. {feat:30s} loading={row['PC1']:7.4f} (abs={row['abs_PC1']:.4f})")

    print(f"\n[D] Top 10 features contributing to PC2 in heel_both (explained variance: {pca_both.explained_variance_ratio_[1]:.3f}):")
    for idx, (feat, row) in enumerate(loadings_both_pc2.head(10).iterrows()):
        print(f"  {idx+1:2d}. {feat:30s} loading={row['PC2']:7.4f} (abs={row['abs_PC2']:.4f})")

    # Create comparison visualizations for separate loadings
    print("\n[D] Creating comparison visualizations for separate dataset loadings...")

    # Comparison bar plots: side-by-side for each dataset
    n_top = 20
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # PC1 loadings comparison
    top_pc1_full = loadings_full_pc1.head(n_top)
    top_pc1_both = loadings_both_pc1.head(n_top)

    # Get common features for comparison
    common_features_pc1 = sorted(set(top_pc1_full.index) & set(top_pc1_both.index))[:n_top]

    ax1 = axes[0, 0]
    colors_full = ['red' if x < 0 else 'blue' for x in top_pc1_full.loc[common_features_pc1, 'PC1']]
    ax1.barh(range(len(common_features_pc1)), top_pc1_full.loc[common_features_pc1, 'PC1'],
             color=colors_full, alpha=0.7, label='heel_all_opt2')
    ax1.set_yticks(range(len(common_features_pc1)))
    ax1.set_yticklabels(common_features_pc1, fontsize=7)
    ax1.set_xlabel('Loading value', fontsize=10)
    ax1.set_title(f'PC1 Loadings: heel_all (opt2)\n(Explained Variance: {pca_full.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()

    ax2 = axes[0, 1]
    colors_both = ['red' if x < 0 else 'blue' for x in top_pc1_both.loc[common_features_pc1, 'PC1']]
    ax2.barh(range(len(common_features_pc1)), top_pc1_both.loc[common_features_pc1, 'PC1'],
             color=colors_both, alpha=0.7, label='heel_both_opt1')
    ax2.set_yticks(range(len(common_features_pc1)))
    ax2.set_yticklabels(common_features_pc1, fontsize=7)
    ax2.set_xlabel('Loading value', fontsize=10)
    ax2.set_title(f'PC1 Loadings: heel_both (opt1)\n(Explained Variance: {pca_both.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()

    # PC2 loadings comparison
    top_pc2_full = loadings_full_pc2.head(n_top)
    top_pc2_both = loadings_both_pc2.head(n_top)
    common_features_pc2 = sorted(set(top_pc2_full.index) & set(top_pc2_both.index))[:n_top]

    ax3 = axes[1, 0]
    colors_full_pc2 = ['red' if x < 0 else 'blue' for x in top_pc2_full.loc[common_features_pc2, 'PC2']]
    ax3.barh(range(len(common_features_pc2)), top_pc2_full.loc[common_features_pc2, 'PC2'],
             color=colors_full_pc2, alpha=0.7)
    ax3.set_yticks(range(len(common_features_pc2)))
    ax3.set_yticklabels(common_features_pc2, fontsize=7)
    ax3.set_xlabel('Loading value', fontsize=10)
    ax3.set_title(f'PC2 Loadings: heel_all (opt2)\n(Explained Variance: {pca_full.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax3.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.invert_yaxis()

    ax4 = axes[1, 1]
    colors_both_pc2 = ['red' if x < 0 else 'blue' for x in top_pc2_both.loc[common_features_pc2, 'PC2']]
    ax4.barh(range(len(common_features_pc2)), top_pc2_both.loc[common_features_pc2, 'PC2'],
             color=colors_both_pc2, alpha=0.7)
    ax4.set_yticks(range(len(common_features_pc2)))
    ax4.set_yticklabels(common_features_pc2, fontsize=7)
    ax4.set_xlabel('Loading value', fontsize=10)
    ax4.set_title(f'PC2 Loadings: heel_both (opt1)\n(Explained Variance: {pca_both.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax4.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()

    plt.tight_layout()
    out_png_comparison = out_dir / "D_pca_loadings_comparison_separate_datasets.png"
    plt.savefig(out_png_comparison, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[D] Saved comparison plot for separate datasets: {out_png_comparison}")

    # Scatter plot comparing PC1 loadings between datasets
    common_features_all = sorted(set(loadings_full.index) & set(loadings_both.index))
    if common_features_all:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # PC1 loadings comparison scatter
        pc1_full_vals = loadings_full.loc[common_features_all, 'PC1']
        pc1_both_vals = loadings_both.loc[common_features_all, 'PC1']

        ax1.scatter(pc1_full_vals, pc1_both_vals, alpha=0.6, s=30)
        lim_pc1 = [min(pc1_full_vals.min(), pc1_both_vals.min()) - 0.1,
                   max(pc1_full_vals.max(), pc1_both_vals.max()) + 0.1]
        ax1.plot(lim_pc1, lim_pc1, "k--", linewidth=1)
        ax1.set_xlim(lim_pc1)
        ax1.set_ylim(lim_pc1)
        ax1.set_xlabel('PC1 Loading - heel_all (opt2)', fontsize=10)
        ax1.set_ylabel('PC1 Loading - heel_both (opt1)', fontsize=10)
        ax1.set_title('PC1 Loadings Comparison\nheel_all vs heel_both', fontsize=11)
        ax1.grid(True, alpha=0.3)

        # PC2 loadings comparison scatter
        pc2_full_vals = loadings_full.loc[common_features_all, 'PC2']
        pc2_both_vals = loadings_both.loc[common_features_all, 'PC2']

        ax2.scatter(pc2_full_vals, pc2_both_vals, alpha=0.6, s=30)
        lim_pc2 = [min(pc2_full_vals.min(), pc2_both_vals.min()) - 0.1,
                   max(pc2_full_vals.max(), pc2_both_vals.max()) + 0.1]
        ax2.plot(lim_pc2, lim_pc2, "k--", linewidth=1)
        ax2.set_xlim(lim_pc2)
        ax2.set_ylim(lim_pc2)
        ax2.set_xlabel('PC2 Loading - heel_all (opt2)', fontsize=10)
        ax2.set_ylabel('PC2 Loading - heel_both (opt1)', fontsize=10)
        ax2.set_title('PC2 Loadings Comparison\nheel_all vs heel_both', fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_png_scatter_comparison = out_dir / "D_pca_loadings_scatter_comparison.png"
        plt.savefig(out_png_scatter_comparison, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[D] Saved loadings scatter comparison: {out_png_scatter_comparison}")

    # ========================================================================
    # Also extract loadings from combined PCA (for comparison)
    # ========================================================================
    print("\n[D] Extracting PCA loadings from combined dataset (for comparison)...")
    n_features_pca = pca.components_.shape[1]
    feature_names = list(X.columns)[:n_features_pca]

    if len(feature_names) != n_features_pca:
        feature_names = [f for f in X.columns if not X[f].isna().all()][:n_features_pca]

    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=feature_names[:n_features_pca]
    )

    # Add absolute values for ranking
    loadings_df['abs_PC1'] = loadings_df['PC1'].abs()
    loadings_df['abs_PC2'] = loadings_df['PC2'].abs()

    # Sort by absolute loading values to identify top contributors
    loadings_df_pc1_sorted = loadings_df.sort_values('abs_PC1', ascending=False)
    loadings_df_pc2_sorted = loadings_df.sort_values('abs_PC2', ascending=False)

    # Save full loadings table
    out_csv_loadings = out_dir / "D_pca_loadings_all_features.csv"
    loadings_df.to_csv(out_csv_loadings)
    print(f"[D] Saved PCA loadings (all features): {out_csv_loadings}")

    # Save top contributors for PC1
    out_csv_pc1 = out_dir / "D_pca_loadings_top_PC1.csv"
    loadings_df_pc1_sorted.to_csv(out_csv_pc1)
    print(f"[D] Saved top PC1 contributors: {out_csv_pc1}")

    # Save top contributors for PC2
    out_csv_pc2 = out_dir / "D_pca_loadings_top_PC2.csv"
    loadings_df_pc2_sorted.to_csv(out_csv_pc2)
    print(f"[D] Saved top PC2 contributors: {out_csv_pc2}")

    # Print top 10 contributors to each PC
    print(f"\n[D] Top 10 features contributing to PC1 (explained variance: {pca.explained_variance_ratio_[0]:.3f}):")
    for idx, (feat, row) in enumerate(loadings_df_pc1_sorted.head(10).iterrows()):
        print(f"  {idx+1:2d}. {feat:30s} loading={row['PC1']:7.4f} (abs={row['abs_PC1']:.4f})")

    print(f"\n[D] Top 10 features contributing to PC2 (explained variance: {pca.explained_variance_ratio_[1]:.3f}):")
    for idx, (feat, row) in enumerate(loadings_df_pc2_sorted.head(10).iterrows()):
        print(f"  {idx+1:2d}. {feat:30s} loading={row['PC2']:7.4f} (abs={row['abs_PC2']:.4f})")

    # Create visualization of top loadings
    n_top = 20  # Show top 20 features for each PC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # PC1 loadings bar plot
    top_pc1 = loadings_df_pc1_sorted.head(n_top)
    colors_pc1 = ['red' if x < 0 else 'blue' for x in top_pc1['PC1']]
    ax1.barh(range(len(top_pc1)), top_pc1['PC1'], color=colors_pc1, alpha=0.7)
    ax1.set_yticks(range(len(top_pc1)))
    ax1.set_yticklabels(top_pc1.index, fontsize=8)
    ax1.set_xlabel('Loading value', fontsize=10)
    ax1.set_title(f'Top {n_top} Features Contributing to PC1\n(Explained Variance: {pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()

    # PC2 loadings bar plot
    top_pc2 = loadings_df_pc2_sorted.head(n_top)
    colors_pc2 = ['red' if x < 0 else 'blue' for x in top_pc2['PC2']]
    ax2.barh(range(len(top_pc2)), top_pc2['PC2'], color=colors_pc2, alpha=0.7)
    ax2.set_yticks(range(len(top_pc2)))
    ax2.set_yticklabels(top_pc2.index, fontsize=8)
    ax2.set_xlabel('Loading value', fontsize=10)
    ax2.set_title(f'Top {n_top} Features Contributing to PC2\n(Explained Variance: {pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()

    plt.tight_layout()
    out_png_loadings = out_dir / "D_pca_loadings_barplot.png"
    plt.savefig(out_png_loadings, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[D] Saved PCA loadings bar plot: {out_png_loadings}")

    # Create biplot: PC1 vs PC2 loadings scatter plot
    n_top_biplot = 10  # Show top 10 features in biplot
    top_features_pc1 = set(loadings_df_pc1_sorted.head(n_top_biplot).index)
    top_features_pc2 = set(loadings_df_pc2_sorted.head(n_top_biplot).index)
    top_features = list(top_features_pc1 | top_features_pc2)  # Union of top features from both PCs

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot all features as points
    ax.scatter(loadings_df['PC1'], loadings_df['PC2'], alpha=0.3, s=20, color='gray', label='All features')

    # Highlight top features
    top_loadings = loadings_df.loc[top_features]
    ax.scatter(top_loadings['PC1'], top_loadings['PC2'], alpha=0.7, s=50, color='red', label=f'Top {len(top_features)} features')

    # Add feature labels for top contributors
    for feat in top_features:
        if feat in loadings_df.index:
            ax.annotate(
                feat,
                (loadings_df.loc[feat, 'PC1'], loadings_df.loc[feat, 'PC2']),
                fontsize=7,
                alpha=0.8,
                xytext=(3, 3),
                textcoords='offset points'
            )

    # Add axes lines
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_xlabel(f'PC1 Loading (Explained Variance: {pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 Loading (Explained Variance: {pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.set_title('PCA Loadings Biplot: PC1 vs PC2\n(Top contributing features labeled)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_png_biplot = out_dir / "D_pca_loadings_biplot.png"
    plt.savefig(out_png_biplot, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[D] Saved PCA loadings biplot: {out_png_biplot}")

    # Create heatmap of top 10 loadings for both PCs
    n_top_heatmap = 10
    top_pc1_features = loadings_df_pc1_sorted.head(n_top_heatmap).index.tolist()
    top_pc2_features = loadings_df_pc2_sorted.head(n_top_heatmap).index.tolist()
    all_top_features = sorted(list(set(top_pc1_features + top_pc2_features)))

    # Create a matrix for heatmap: rows are features, columns are PC1 and PC2
    heatmap_data = loadings_df.loc[all_top_features, ['PC1', 'PC2']]

    fig, ax = plt.subplots(figsize=(6, max(10, len(all_top_features) * 0.3)))
    im = ax.imshow(heatmap_data.values, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)

    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['PC1', 'PC2'], fontsize=10)
    ax.set_yticks(range(len(all_top_features)))
    ax.set_yticklabels(all_top_features, fontsize=7)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Loading Value', fontsize=10)

    # Add text annotations
    for i in range(len(all_top_features)):
        for j in range(2):
            text = ax.text(j, i, f'{heatmap_data.values[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=6)

    ax.set_title(f'PCA Loadings Heatmap: Top {len(all_top_features)} Features\n(PC1 and PC2)', fontsize=11)
    plt.tight_layout()

    out_png_heatmap = out_dir / "D_pca_loadings_heatmap.png"
    plt.savefig(out_png_heatmap, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[D] Saved PCA loadings heatmap: {out_png_heatmap}")

    # Create quadrant analysis plot (showing features by their PC1/PC2 contribution)
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot all features
    scatter = ax.scatter(loadings_df['PC1'], loadings_df['PC2'],
                        c=loadings_df['abs_PC1'] + loadings_df['abs_PC2'],
                        s=50, alpha=0.6, cmap='viridis', edgecolors='black', linewidth=0.5)

    # Add quadrant lines
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)

    # Label quadrants
    ax.text(0.02, 0.98, 'High PC1, High PC2', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(0.98, 0.98, 'High PC1, Low PC2', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(0.02, 0.02, 'Low PC1, High PC2', transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(0.98, 0.02, 'Low PC1, Low PC2', transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Label top features
    for feat in top_features[:10]:  # Label top 10 to avoid overcrowding
        if feat in loadings_df.index:
            ax.annotate(feat,
                       (loadings_df.loc[feat, 'PC1'], loadings_df.loc[feat, 'PC2']),
                       fontsize=7, alpha=0.8, xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel(f'PC1 Loading (Explained Variance: {pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 Loading (Explained Variance: {pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.set_title('PCA Loadings Quadrant Analysis\n(Color = Combined PC1+PC2 Contribution)', fontsize=12)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Combined |PC1| + |PC2|', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_png_quadrant = out_dir / "D_pca_loadings_quadrant.png"
    plt.savefig(out_png_quadrant, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[D] Saved PCA loadings quadrant plot: {out_png_quadrant}")

    # If SGA data is available, analyze loadings in context of SGA
    if 'is_sga' in combined.columns and combined['is_sga'].any():
        # Calculate mean PC1 and PC2 values for SGA vs non-SGA groups
        sga_stats = combined.groupby('is_sga')[['_PC1', '_PC2']].agg(['mean', 'std', 'count'])
        out_csv_sga = out_dir / "D_pca_sga_stats.csv"
        sga_stats.to_csv(out_csv_sga)
        print(f"[D] Saved SGA vs non-SGA PCA statistics: {out_csv_sga}")

        # Print summary
        print(f"\n[D] PCA separation by SGA status:")
        for sga_status, group_name in [(True, "SGA"), (False, "Non-SGA")]:
            mask = combined['is_sga'] == sga_status
            if mask.any():
                mean_pc1 = combined.loc[mask, '_PC1'].mean()
                mean_pc2 = combined.loc[mask, '_PC2'].mean()
                print(f"  {group_name:8s}: PC1={mean_pc1:7.3f}, PC2={mean_pc2:7.3f}, n={mask.sum()}")


# ---------- Test E: Simple logistic-regression AUC via CV ---------- #

def run_logistic_cv_auc(df: pd.DataFrame,
                        features,
                        ga_col: str = GA_COL,
                        n_splits: int = 5):
    """Stratified CV AUC for preterm vs term using given feature set."""
    # Target: preterm vs term
    ga = pd.to_numeric(df[ga_col], errors="coerce")
    mask = ga.notna()
    df = df.loc[mask].copy()
    y = (ga.loc[mask] < PRETERM_CUTOFF).astype(int).to_numpy()

    X = df[features].copy()

    # Convert categorical variables to numeric (matching data_loader.py)
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes

    # Drop columns that are all-NaN in this dataset
    keep_cols = [c for c in features if not X[c].isna().all()]
    X = X[keep_cols]

    if len(keep_cols) == 0:
        raise ValueError("No usable feature columns (all-NaN).")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        aucs.append(auc)

    aucs = np.asarray(aucs, dtype=float)
    return keep_cols, aucs


def compare_cv_auc(heel_full, heel_both, features, out_dir: Path):
    """Compute stratified cross-validation AUC for both heel subsets and save summary CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use the same feature list for both datasets (PCA feature set)
    print(f"[E] Using {len(features)} common numeric features for CV AUC.")

    rows = []
    for ds_name, df in [("heel_all_opt2", heel_full), ("heel_both_opt1", heel_both)]:
        try:
            used_features, aucs = run_logistic_cv_auc(df, features)
            mean_auc = float(aucs.mean())
            std_auc = float(aucs.std(ddof=1))
            print(f"[E] {ds_name}: mean AUC={mean_auc:.3f} +/- {std_auc:.3f} (n_splits={len(aucs)})")
            rows.append({
                "dataset": ds_name,
                "n_splits": len(aucs),
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "min_auc": float(aucs.min()),
                "max_auc": float(aucs.max()),
                "n_features_used": len(used_features),
            })
        except Exception as e:
            print(f"[E] ERROR running CV AUC for {ds_name}: {e}")

    if rows:
        out_csv = out_dir / "E_logistic_cv_auc_summary.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[E] Saved CV AUC summary: {out_csv}")
    else:
        print("[E] No AUC rows saved (check errors above).")


# ---------- Test F: Correlations with GA and Birth Weight ---------- #

def _corr_with_target_scaled(X_scaled_df: pd.DataFrame, target_values: np.ndarray, feat: str):
    """Calculate Pearson correlation between a scaled feature and an arbitrary target array."""
    if feat not in X_scaled_df.columns:
        return np.nan
    x = X_scaled_df[feat].values
    # Both x and target should be aligned (same length after imputation)
    if len(x) != len(target_values):
        return np.nan
    mask = np.isfinite(x) & np.isfinite(target_values)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(x[mask], target_values[mask])[0, 1])


def compare_ga_bw_correlation(heel_full, heel_both, features, out_dir: Path):
    """
    Calculate correlations of each feature with both gestational_age_weeks and birth_weight_kg.
    Creates visualizations comparing correlations for both targets.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    BW_COL = "birth_weight_kg"

    # Process both datasets and store correlations
    corr_dict = {}

    for ds_name, df in [("heel_all_opt2", heel_full), ("heel_both_opt1", heel_both)]:
        # Prepare features (same preprocessing as Test D)
        X = df[features].copy()

        # Get GA and BW values
        ga = pd.to_numeric(df[GA_COL], errors="coerce")
        bw = pd.to_numeric(df[BW_COL], errors="coerce") if BW_COL in df.columns else None

        # Convert categorical variables to numeric (matching data_loader.py)
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if col in X.columns:
                    X[col] = pd.Categorical(X[col]).codes

        # Impute + scale (matching Test D and data_loader.py)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_imp = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imp)

        # Convert back to DataFrame for easier processing
        X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=df.index)

        # Impute target values to match the imputed feature rows
        ga_imp = ga.fillna(ga.median()).values
        bw_imp = bw.fillna(bw.median()).values if bw is not None else None

        # Calculate correlation for each feature with GA
        for feat in features:
            if feat not in corr_dict:
                corr_dict[feat] = {}
            corr_ga = _corr_with_target_scaled(X_scaled_df, ga_imp, feat)
            corr_dict[feat][f"{ds_name}_GA"] = corr_ga

            # Calculate correlation for each feature with BW (if available)
            if bw_imp is not None:
                corr_bw = _corr_with_target_scaled(X_scaled_df, bw_imp, feat)
                corr_dict[feat][f"{ds_name}_BW"] = corr_bw

    # Build rows from correlation dictionary
    rows = []
    for feat in features:
        row = {"feature": feat}

        # GA correlations
        row["corr_GA_heel_all_opt2"] = corr_dict.get(feat, {}).get("heel_all_opt2_GA", np.nan)
        row["corr_GA_heel_both_opt1"] = corr_dict.get(feat, {}).get("heel_both_opt1_GA", np.nan)

        # BW correlations (if available)
        row["corr_BW_heel_all_opt2"] = corr_dict.get(feat, {}).get("heel_all_opt2_BW", np.nan)
        row["corr_BW_heel_both_opt1"] = corr_dict.get(feat, {}).get("heel_both_opt1_BW", np.nan)

        # Average correlations across datasets
        ga_vals = [row["corr_GA_heel_all_opt2"], row["corr_GA_heel_both_opt1"]]
        bw_vals = [row["corr_BW_heel_all_opt2"], row["corr_BW_heel_both_opt1"]]

        row["corr_GA_mean"] = np.nanmean([v for v in ga_vals if np.isfinite(v)]) if any(np.isfinite(v) for v in ga_vals) else np.nan
        row["corr_BW_mean"] = np.nanmean([v for v in bw_vals if np.isfinite(v)]) if any(np.isfinite(v) for v in bw_vals) else np.nan

        # Absolute values for ranking
        row["abs_corr_GA_mean"] = abs(row["corr_GA_mean"]) if np.isfinite(row["corr_GA_mean"]) else np.nan
        row["abs_corr_BW_mean"] = abs(row["corr_BW_mean"]) if np.isfinite(row["corr_BW_mean"]) else np.nan

        rows.append(row)

    df_corr = pd.DataFrame(rows)
    out_csv = out_dir / "F_corr_with_ga_and_bw.csv"
    df_corr.to_csv(out_csv, index=False)
    print(f"[F] Saved GA and BW correlation table: {out_csv}")

    # ========================================================================
    # Create visualizations
    # ========================================================================

    # 1. Top correlations bar plots (separate for each dataset and target)
    n_top = 20

    # Add absolute correlation columns for individual datasets
    df_corr['abs_corr_GA_heel_all'] = df_corr['corr_GA_heel_all_opt2'].abs()
    df_corr['abs_corr_GA_heel_both'] = df_corr['corr_GA_heel_both_opt1'].abs()
    df_corr['abs_corr_BW_heel_all'] = df_corr['corr_BW_heel_all_opt2'].abs()
    df_corr['abs_corr_BW_heel_both'] = df_corr['corr_BW_heel_both_opt1'].abs()

    # Filter valid correlations
    valid_ga = df_corr.dropna(subset=["corr_GA_mean"])
    valid_bw = df_corr.dropna(subset=["corr_BW_mean"])

    # GA correlations - separate plots for each dataset
    if not valid_ga.empty:
        # heel_all GA correlations
        valid_ga_all = df_corr.dropna(subset=["corr_GA_heel_all_opt2"])
        if not valid_ga_all.empty:
            top_ga_all = valid_ga_all.nlargest(n_top, "abs_corr_GA_heel_all")

            fig, ax = plt.subplots(figsize=(12, max(8, len(top_ga_all) * 0.3)))
            colors = ['red' if x < 0 else 'blue' for x in top_ga_all['corr_GA_heel_all_opt2']]
            ax.barh(range(len(top_ga_all)), top_ga_all['corr_GA_heel_all_opt2'], color=colors, alpha=0.7)
            ax.set_yticks(range(len(top_ga_all)))
            ax.set_yticklabels(top_ga_all['feature'], fontsize=8)
            ax.set_xlabel('Correlation with Gestational Age (weeks)', fontsize=11)
            ax.set_title(f'Top {len(top_ga_all)} Features Correlated with Gestational Age\n(heel_all, opt2)', fontsize=12)
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3, axis='x')
            ax.invert_yaxis()
            plt.tight_layout()
            out_png_ga_all = out_dir / "F_corr_with_ga_top_features_heel_all.png"
            plt.savefig(out_png_ga_all, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[F] Saved GA correlation bar plot (heel_all): {out_png_ga_all}")

        # heel_both GA correlations
        valid_ga_both = df_corr.dropna(subset=["corr_GA_heel_both_opt1"])
        if not valid_ga_both.empty:
            top_ga_both = valid_ga_both.nlargest(n_top, "abs_corr_GA_heel_both")

            fig, ax = plt.subplots(figsize=(12, max(8, len(top_ga_both) * 0.3)))
            colors = ['red' if x < 0 else 'blue' for x in top_ga_both['corr_GA_heel_both_opt1']]
            ax.barh(range(len(top_ga_both)), top_ga_both['corr_GA_heel_both_opt1'], color=colors, alpha=0.7)
            ax.set_yticks(range(len(top_ga_both)))
            ax.set_yticklabels(top_ga_both['feature'], fontsize=8)
            ax.set_xlabel('Correlation with Gestational Age (weeks)', fontsize=11)
            ax.set_title(f'Top {len(top_ga_both)} Features Correlated with Gestational Age\n(heel_both, opt1)', fontsize=12)
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3, axis='x')
            ax.invert_yaxis()
            plt.tight_layout()
            out_png_ga_both = out_dir / "F_corr_with_ga_top_features_heel_both.png"
            plt.savefig(out_png_ga_both, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[F] Saved GA correlation bar plot (heel_both): {out_png_ga_both}")

    # BW correlations - separate plots for each dataset
    if not valid_bw.empty:
        # heel_all BW correlations
        valid_bw_all = df_corr.dropna(subset=["corr_BW_heel_all_opt2"])
        if not valid_bw_all.empty:
            top_bw_all = valid_bw_all.nlargest(n_top, "abs_corr_BW_heel_all")

            fig, ax = plt.subplots(figsize=(12, max(8, len(top_bw_all) * 0.3)))
            colors = ['red' if x < 0 else 'blue' for x in top_bw_all['corr_BW_heel_all_opt2']]
            ax.barh(range(len(top_bw_all)), top_bw_all['corr_BW_heel_all_opt2'], color=colors, alpha=0.7)
            ax.set_yticks(range(len(top_bw_all)))
            ax.set_yticklabels(top_bw_all['feature'], fontsize=8)
            ax.set_xlabel('Correlation with Birth Weight (kg)', fontsize=11)
            ax.set_title(f'Top {len(top_bw_all)} Features Correlated with Birth Weight\n(heel_all, opt2)', fontsize=12)
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3, axis='x')
            ax.invert_yaxis()
            plt.tight_layout()
            out_png_bw_all = out_dir / "F_corr_with_bw_top_features_heel_all.png"
            plt.savefig(out_png_bw_all, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[F] Saved BW correlation bar plot (heel_all): {out_png_bw_all}")

        # heel_both BW correlations
        valid_bw_both = df_corr.dropna(subset=["corr_BW_heel_both_opt1"])
        if not valid_bw_both.empty:
            top_bw_both = valid_bw_both.nlargest(n_top, "abs_corr_BW_heel_both")

            fig, ax = plt.subplots(figsize=(12, max(8, len(top_bw_both) * 0.3)))
            colors = ['red' if x < 0 else 'blue' for x in top_bw_both['corr_BW_heel_both_opt1']]
            ax.barh(range(len(top_bw_both)), top_bw_both['corr_BW_heel_both_opt1'], color=colors, alpha=0.7)
            ax.set_yticks(range(len(top_bw_both)))
            ax.set_yticklabels(top_bw_both['feature'], fontsize=8)
            ax.set_xlabel('Correlation with Birth Weight (kg)', fontsize=11)
            ax.set_title(f'Top {len(top_bw_both)} Features Correlated with Birth Weight\n(heel_both, opt1)', fontsize=12)
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3, axis='x')
            ax.invert_yaxis()
            plt.tight_layout()
            out_png_bw_both = out_dir / "F_corr_with_bw_top_features_heel_both.png"
            plt.savefig(out_png_bw_both, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[F] Saved BW correlation bar plot (heel_both): {out_png_bw_both}")

    # 2. Scatter plot: GA correlation vs BW correlation
    if not valid_ga.empty and not valid_bw.empty:
        # Get features with both correlations
        both_valid = df_corr.dropna(subset=["corr_GA_mean", "corr_BW_mean"])

        if not both_valid.empty:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(both_valid["corr_GA_mean"], both_valid["corr_BW_mean"],
                      alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

            # Add diagonal line
            lim = [-1, 1]
            ax.plot(lim, lim, "k--", linewidth=1, alpha=0.5, label='y=x')

            # Label top features (by combined absolute correlation)
            both_valid_copy = both_valid.copy()
            both_valid_copy['combined_abs'] = both_valid_copy['abs_corr_GA_mean'] + both_valid_copy['abs_corr_BW_mean']
            top_combined = both_valid_copy.nlargest(10, 'combined_abs')

            for _, row in top_combined.iterrows():
                ax.annotate(row['feature'],
                           (row['corr_GA_mean'], row['corr_BW_mean']),
                           fontsize=7, alpha=0.8,
                           xytext=(5, 5), textcoords='offset points')

            ax.set_xlabel('Correlation with Gestational Age (weeks)', fontsize=11)
            ax.set_ylabel('Correlation with Birth Weight (kg)', fontsize=11)
            ax.set_title('Feature Correlations: GA vs Birth Weight\n(Top 10 features labeled)', fontsize=12)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            out_png_scatter = out_dir / "F_corr_ga_vs_bw_scatter.png"
            plt.savefig(out_png_scatter, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[F] Saved GA vs BW correlation scatter plot: {out_png_scatter}")

    # 3. Heatmap of correlations for both targets (top features)
    if not valid_ga.empty and not valid_bw.empty:
        # Get top features by combined absolute correlation
        both_valid = df_corr.dropna(subset=["corr_GA_mean", "corr_BW_mean"])
        if not both_valid.empty:
            both_valid_copy = both_valid.copy()
            both_valid_copy['combined_abs'] = both_valid_copy['abs_corr_GA_mean'] + both_valid_copy['abs_corr_BW_mean']
            top_features = both_valid_copy.nlargest(min(30, len(both_valid)), 'combined_abs')

            # Create heatmap data: rows are features, columns are GA and BW
            heatmap_data = top_features[['corr_GA_mean', 'corr_BW_mean']].T
            heatmap_data.columns = top_features['feature'].values

            fig, ax = plt.subplots(figsize=(max(12, len(top_features) * 0.4), 4))
            im = ax.imshow(heatmap_data.values, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)

            # Set ticks and labels
            ax.set_xticks(range(len(top_features)))
            ax.set_xticklabels(top_features['feature'].values, rotation=90, fontsize=7, ha='right')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['GA (weeks)', 'BW (kg)'], fontsize=10)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Correlation', fontsize=10)

            # Add text annotations
            for i in range(2):
                for j in range(len(top_features)):
                    text = ax.text(j, i, f'{heatmap_data.values[i, j]:.2f}',
                                 ha="center", va="center", color="white" if abs(heatmap_data.values[i, j]) > 0.5 else "black",
                                 fontsize=6, weight='bold' if abs(heatmap_data.values[i, j]) > 0.5 else 'normal')

            ax.set_title(f'Correlation Heatmap: Top {len(top_features)} Features\nwith Gestational Age and Birth Weight', fontsize=12)
            plt.tight_layout()

            out_png_heatmap = out_dir / "F_corr_ga_and_bw_heatmap.png"
            plt.savefig(out_png_heatmap, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[F] Saved GA and BW correlation heatmap: {out_png_heatmap}")

    # 4. Comparison plots: heel_all vs heel_both for GA and BW separately
    if not valid_ga.empty:
        # GA comparison scatter
        valid_ga_comparison = df_corr.dropna(subset=["corr_GA_heel_all_opt2", "corr_GA_heel_both_opt1"])
        if not valid_ga_comparison.empty:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(valid_ga_comparison["corr_GA_heel_all_opt2"],
                      valid_ga_comparison["corr_GA_heel_both_opt1"],
                      alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
            lim = [-1, 1]
            ax.plot(lim, lim, "k--", linewidth=1)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel("Corr(feature, GA) – heel_all (opt2)", fontsize=11)
            ax.set_ylabel("Corr(feature, GA) – heel_both (opt1)", fontsize=11)
            ax.set_title("GA Correlations: heel_all vs heel_both", fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out_png_ga_comp = out_dir / "F_corr_ga_heel_all_vs_both.png"
            plt.savefig(out_png_ga_comp, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[F] Saved GA correlation comparison plot: {out_png_ga_comp}")

    if not valid_bw.empty:
        # BW comparison scatter
        valid_bw_comparison = df_corr.dropna(subset=["corr_BW_heel_all_opt2", "corr_BW_heel_both_opt1"])
        if not valid_bw_comparison.empty:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(valid_bw_comparison["corr_BW_heel_all_opt2"],
                      valid_bw_comparison["corr_BW_heel_both_opt1"],
                      alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
            lim = [-1, 1]
            ax.plot(lim, lim, "k--", linewidth=1)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xlabel("Corr(feature, BW) – heel_all (opt2)", fontsize=11)
            ax.set_ylabel("Corr(feature, BW) – heel_both (opt1)", fontsize=11)
            ax.set_title("Birth Weight Correlations: heel_all vs heel_both", fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out_png_bw_comp = out_dir / "F_corr_bw_heel_all_vs_both.png"
            plt.savefig(out_png_bw_comp, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[F] Saved BW correlation comparison plot: {out_png_bw_comp}")

    print(f"[F] Completed correlation analysis with GA and Birth Weight")


# ---------- main ---------- #

def main():
    """Parse arguments, load heel subsets, and run all feature quality comparison tests."""
    ap = argparse.ArgumentParser(
        description="Tests A-F: feature quality comparison for heel_all vs heel_both."
    )
    ap.add_argument(
        "--full_csv",
        required=True,
        help="Path to full BangladeshcombineddatasetJan252022.csv",
    )
    ap.add_argument(
        "--both_csv",
        required=True,
        help="Path to Bangladesh_children_with_both_samples.csv",
    )
    ap.add_argument(
        "--out_dir",
        default="outputs/heel_feature_quality",
        help="Directory to save CSVs and plots.",
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    heel_full, heel_both = load_heel_subsets(Path(args.full_csv), Path(args.both_csv))
    print(f"[INFO] heel_all_opt2 n={len(heel_full)}, heel_both_opt1 n={len(heel_both)}")

    if GA_COL not in heel_full.columns or GA_COL not in heel_both.columns:
        raise ValueError(f"Expected gestational age column '{GA_COL}' in both datasets.")

    # Common numeric feature set
    features = pick_common_numeric_features(heel_full, heel_both, ga_col=GA_COL)
    print(f"[INFO] Common numeric features (excluding {GA_COL}): {len(features)}")

    # A - Missingness
    compare_missingness(heel_full, heel_both, features, out_dir)

    # B - Variance
    compare_variance(heel_full, heel_both, features, out_dir)

    # C - Corr(feature, GA)
    compare_ga_correlation(heel_full, heel_both, features, out_dir)

    # D - PCA
    compare_pca_structure(heel_full, heel_both, features, out_dir)

    # E - Simple logistic-regression CV AUC
    compare_cv_auc(heel_full, heel_both, features, out_dir)

    # F - Correlations with GA and Birth Weight
    compare_ga_bw_correlation(heel_full, heel_both, features, out_dir)


if __name__ == "__main__":
    main()
