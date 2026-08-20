"""Shared feature lists and dataset loading for Training cohort / external validation cohort."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.config import RANDOM_SEED, PRETERM_CUTOFF, TRAINING_CSV_BOTH, TRAINING_CSV_ALL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_COL = "gestational_age_weeks"
EXTERNAL_VALIDATION_CSV = PROJECT_ROOT / "data" / "ZimbabweNewborn_samefeatures_as_Bangladesh.csv"

CLINICAL_FEATURES = [
    "birth_weight_kg",
    "sex",
    "multiple_birth",
    "prepregnancy_bmi",
    "maternal_age_delivery",
]

# Encoded as numeric codes (1/2); imputed with median SimpleImputer like continuous features.
CATEGORICAL_FEATURES = [
    "sex",
    "multiple_birth",
]

BIOMARKER_FEATURES = [
    "TSH", "N17P", "IRT", "GALT", "BIOT", "Ala", "Arg", "ASA", "ASA_Arg", "ASA_Orn",
    "Cit", "Cit_Arg", "Cit_Orn", "Cit_Tyr", "Gly", "LEU", "Leu_Ala", "Leu_Phe", "MET",
    "Met_Phe", "Orn", "Orn_Arg", "Orn_Cit", "Orn_Phe", "PHE", "PHE_TYR", "SUAC", "TYR",
    "Tyr_Phe", "Val", "Val_Ala", "Val_Phe", "C0", "C0_C16C18", "C0C2C3C16C18_Cit", "C2",
    "C3", "C3_C0", "C3_C16", "C3_C2", "C3_C4DC", "C3DC", "C4", "C4DC", "C4OH", "C5", "C5_C0",
    "C5_C2", "C5_C3", "C5_1", "C5DC", "C5DC_C16", "C5DC_C5OH", "C5DC_C8", "C5OH", "C5OH_C2",
    "C5OH_C5_1", "C5OH_C8", "C6", "C6DC", "C8", "C8_C10", "C8_C2", "C8_1", "C10", "C10_1",
    "C12", "C12_1", "C14", "C14OH", "C14_1", "C14_1_C12_1", "C14_1_C16", "C14_1_C4", "C14_2",
    "C16", "C16_1OH", "C16_1OH_C4DC", "C16OH", "C16OH_C16", "C18", "C18_1", "C18_1OH", "C18_2",
    "C18OH", "HGB___FAST", "HGB___F1", "HGB___F", "HGB___F_F1", "HGB___A", "HGB___FAST_F1",
    "HGB___Other",
]


def validation_sample_type(study_id: str) -> Optional[str]:
    """Return 'CORD' or 'HEEL' based on the study ID suffix, or None if unrecognised."""
    compact = str(study_id).strip().upper().replace(" ", "")
    match = re.search(r"-(C|H)\d", compact)
    if not match:
        return None
    return "CORD" if match.group(1) == "C" else "HEEL"


def filter_validation_by_sample_type(df: pd.DataFrame, data_type: Optional[str]) -> pd.DataFrame:
    """Filter the external validation DataFrame to rows matching the requested sample type."""
    if data_type is None:
        return df.copy()
    data_type_norm = str(data_type).strip().lower()
    if data_type_norm not in ("heel", "cord"):
        raise ValueError("data_type must be 'heel', 'cord', or None.")
    want = "HEEL" if data_type_norm == "heel" else "CORD"
    if "study_id" not in df.columns:
        raise KeyError("study_id column required to filter validation data")
    mask = df["study_id"].map(validation_sample_type) == want
    return df.loc[mask].copy()


def feature_columns_for_model_type(model_type: str, available_columns) -> list:
    """Return the list of feature column names appropriate for the given model type."""
    if model_type == "clinical":
        cols = [c for c in CLINICAL_FEATURES if c in available_columns]
    elif model_type == "biomarker":
        cols = [c for c in BIOMARKER_FEATURES if c in available_columns]
    elif model_type == "combined":
        cols = [c for c in CLINICAL_FEATURES + BIOMARKER_FEATURES if c in available_columns]
    else:
        raise ValueError("model_type must be 'clinical', 'biomarker', or 'combined'.")
    return cols


def load_training_frame(
    data_option: int,
    data_type: str,
    model_type: str,
) -> Tuple[pd.DataFrame, pd.Series, list]:
    """Load training data; returns X, continuous GA target, feature names."""
    data_type_norm = str(data_type).strip().lower()
    if data_option == 1:
        df = pd.read_csv(TRAINING_CSV_BOTH)
        if data_type_norm == "heel":
            df = df[df["Source"] == "HEEL"].copy()
        elif data_type_norm == "cord":
            df = df[df["Source"] == "CORD"].copy()
        else:
            raise ValueError("data_type must be 'heel' or 'cord' when data_option == 1.")
    elif data_option == 2:
        df = pd.read_csv(TRAINING_CSV_ALL)
        df = df[df["Source"] == "HEEL"].copy()
    elif data_option == 3:
        df = pd.read_csv(TRAINING_CSV_ALL)
        df = df[df["Source"] == "CORD"].copy()
    else:
        raise ValueError("data_option must be 1, 2, or 3.")

    keep_cols = [c for c in CLINICAL_FEATURES + BIOMARKER_FEATURES + [TARGET_COL] if c in df.columns]
    df = df[keep_cols].copy()
    if TARGET_COL not in df.columns:
        raise KeyError(f"Expected target column '{TARGET_COL}' not found.")

    df = df.dropna(subset=[TARGET_COL])
    feature_names = feature_columns_for_model_type(model_type, df.columns)
    X = df[feature_names].copy()
    y = df[TARGET_COL]
    return X, y, feature_names


def load_validation_frame(
    model_type: str,
    data_type: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, list]:
    """
    Load external validation data.

    Returns X_val, y_binary, y_continuous, study_id, feature_names.
    """
    df = pd.read_csv(EXTERNAL_VALIDATION_CSV)
    df = filter_validation_by_sample_type(df, data_type)

    keep_cols = [
        c
        for c in ["study_id"] + CLINICAL_FEATURES + BIOMARKER_FEATURES + [TARGET_COL]
        if c in df.columns
    ]
    df = df[keep_cols].copy()
    if TARGET_COL not in df.columns:
        raise KeyError(f"Expected target column '{TARGET_COL}' not found in validation CSV.")

    df = df.dropna(subset=[TARGET_COL])
    feature_names = feature_columns_for_model_type(model_type, df.columns)
    X_val = df[feature_names].copy()
    y_cont = df[TARGET_COL]
    y_binary = (y_cont < PRETERM_CUTOFF).astype(int)
    study_id = df["study_id"] if "study_id" in df.columns else pd.Series([None] * len(df))
    return X_val, y_binary, y_cont, study_id, feature_names


def align_to_training_features(X: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Reindex validation features to match the training column order."""
    return X.reindex(columns=feature_names)
