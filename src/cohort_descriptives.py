"""Cohort descriptive statistics for the training and external validation datasets."""

import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import PRETERM_CUTOFF
from src.feature_sets import validation_sample_type

TRAINING_DATASET_PATH = _project_root / "data" / "Bangladesh_children_with_both_samples.csv"
VALIDATION_NEWBORN_DATASET_PATH = _project_root / "data" / "ZimbabweNewborn_samefeatures_as_Bangladesh.csv"
VALIDATION_MATERNAL_DATASET_PATH = _project_root / "data" / "ZimbabweMaternalOnly (1).csv"
DEFAULT_DATASET = TRAINING_DATASET_PATH


def prepare_validation_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize validation newborn tables to use ``study_id``."""
    out = df.copy()
    if "study_id" not in out.columns and "Study.ID" in out.columns:
        out = out.rename(columns={"Study.ID": "study_id"})
    return out


def parse_validation_baby_key(study_id) -> Optional[Tuple[str, int, Optional[int]]]:
    """Parse newborn study_id to (site_prefix, enrolment_number, twin_index)."""
    if pd.isna(study_id):
        return None
    compact = str(study_id).strip().upper().replace(" ", "")
    match = re.match(r"^([A-Z]+)-?(\d+)(?:-(C|H)(\d+))?(?:-(\d+))?$", compact)
    if not match:
        return None
    twin = int(match.group(5)) if match.group(5) else None
    return (match.group(1), int(match.group(2)), twin)


def parse_validation_mother_key(study_id) -> Optional[Tuple[str, int]]:
    """Parse newborn study_id to the mother-level (site_prefix, enrolment_number) key."""
    baby_key = parse_validation_baby_key(study_id)
    if baby_key is None:
        return None
    return (baby_key[0], baby_key[1])


def count_unique_mothers_and_babies(df: pd.DataFrame) -> tuple[int, int]:
    """
    Count unique mothers and babies in a training sample-level table.

    - Mothers: unique ``mother_id``
    - Babies: unique (``mother_id``, ``child_id``) pairs
      (``child_id`` encodes birth order / multiples within a mother)
    """
    if "mother_id" not in df.columns or "child_id" not in df.columns:
        raise KeyError("Expected columns 'mother_id' and 'child_id'.")

    n_mothers = int(df["mother_id"].nunique())
    baby_rows = df[["mother_id", "child_id"]].dropna()
    n_babies = int(baby_rows.drop_duplicates().shape[0])
    return n_mothers, n_babies


def babies_with_ga(df: pd.DataFrame) -> pd.DataFrame:
    """One row per baby with gestational age in weeks."""
    if "gestational_age_weeks" not in df.columns:
        raise KeyError("Expected column 'gestational_age_weeks'.")

    baby_df = df[["mother_id", "child_id", "gestational_age_weeks"]].copy()
    baby_df["gestational_age_weeks"] = pd.to_numeric(
        baby_df["gestational_age_weeks"], errors="coerce"
    )
    baby_df = baby_df.dropna(subset=["mother_id", "child_id"])
    # GA is identical across heel/cord rows for the same baby; take the first value.
    return (
        baby_df.groupby(["mother_id", "child_id"], as_index=False)["gestational_age_weeks"]
        .first()
    )


def count_preterm_term_babies(df: pd.DataFrame) -> dict[str, int]:
    """Count unique babies by preterm (< PRETERM_CUTOFF w) vs term (>= cutoff)."""
    babies = babies_with_ga(df)
    ga = babies["gestational_age_weeks"]
    n_valid = int(ga.notna().sum())
    n_preterm = int((ga < PRETERM_CUTOFF).sum())
    n_term = int((ga >= PRETERM_CUTOFF).sum())
    n_missing = int(ga.isna().sum())
    return {
        "n_babies_with_ga": n_valid,
        "n_preterm": n_preterm,
        "n_term": n_term,
        "n_missing_ga": n_missing,
    }


def print_preterm_term_babies(df: pd.DataFrame) -> None:
    """Print preterm/term counts and prevalence for the training dataset."""
    counts = count_preterm_term_babies(df)
    n_preterm = counts["n_preterm"]
    n_term = counts["n_term"]
    n_valid = counts["n_babies_with_ga"]
    n_missing = counts["n_missing_ga"]

    print(f"Unique babies with GA: {n_valid:,}")
    print(f"Unique preterm babies (GA < {PRETERM_CUTOFF} w): {n_preterm:,}")
    print(f"Unique term babies (GA >= {PRETERM_CUTOFF} w): {n_term:,}")
    if n_missing:
        print(f"Unique babies missing GA: {n_missing:,}")
    if n_valid:
        print(
            f"Preterm prevalence among unique babies: "
            f"{n_preterm / n_valid:.4f} ({n_preterm}/{n_valid})"
        )


def print_mothers_and_babies(df: pd.DataFrame) -> None:
    """Print mother and baby counts plus preterm/term breakdown for the training dataset."""
    n_mothers, n_babies = count_unique_mothers_and_babies(df)
    print(f"Unique mothers (mother_id): {n_mothers:,}")
    print(f"Unique babies (mother_id + child_id): {n_babies:,}")
    print(f"Total rows (samples): {len(df):,}")

    if "Source" in df.columns:
        n_heel_babies = df.loc[df["Source"] == "HEEL", ["mother_id", "child_id"]].dropna().drop_duplicates().shape[0]
        n_cord_babies = df.loc[df["Source"] == "CORD", ["mother_id", "child_id"]].dropna().drop_duplicates().shape[0]
        print(f"Unique babies with HEEL sample: {n_heel_babies:,}")
        print(f"Unique babies with CORD sample: {n_cord_babies:,}")

    print_preterm_term_babies(df)


def count_unique_subjects_validation(df: pd.DataFrame) -> tuple[int, int]:
    """
    Count unique mothers and babies in the validation newborn table.

    - Mothers: unique (site_prefix, enrolment_number) parsed from ``study_id``
    - Babies: unique (site_prefix, enrolment_number, twin_index) parsed from ``study_id``
      (heel and cord rows for the same baby share the same baby key)
    """
    if "study_id" not in df.columns:
        raise KeyError("Expected column 'study_id'.")

    mother_keys = df["study_id"].map(parse_validation_mother_key).dropna()
    baby_keys = df["study_id"].map(parse_validation_baby_key).dropna()
    return int(mother_keys.nunique()), int(baby_keys.nunique())


def babies_with_ga_validation(df: pd.DataFrame) -> pd.DataFrame:
    """One row per validation baby with gestational age in weeks."""
    if "study_id" not in df.columns or "gestational_age_weeks" not in df.columns:
        raise KeyError("Expected columns 'study_id' and 'gestational_age_weeks'.")

    baby_df = df[["study_id", "gestational_age_weeks"]].copy()
    baby_df["baby_key"] = baby_df["study_id"].map(parse_validation_baby_key)
    baby_df["gestational_age_weeks"] = pd.to_numeric(
        baby_df["gestational_age_weeks"], errors="coerce"
    )
    baby_df = baby_df.dropna(subset=["baby_key"])
    return (
        baby_df.groupby("baby_key", as_index=False)["gestational_age_weeks"]
        .first()
    )


def count_preterm_term_babies_validation(df: pd.DataFrame) -> dict[str, int]:
    """Count unique validation babies by preterm vs term status."""
    babies = babies_with_ga_validation(df)
    ga = babies["gestational_age_weeks"]
    n_valid = int(ga.notna().sum())
    n_preterm = int((ga < PRETERM_CUTOFF).sum())
    n_term = int((ga >= PRETERM_CUTOFF).sum())
    n_missing = int(ga.isna().sum())
    return {
        "n_babies_with_ga": n_valid,
        "n_preterm": n_preterm,
        "n_term": n_term,
        "n_missing_ga": n_missing,
    }


def summarize_preterm_term_validation(df: pd.DataFrame) -> None:
    """Print preterm/term counts and prevalence for the validation dataset."""
    counts = count_preterm_term_babies_validation(df)
    n_preterm = counts["n_preterm"]
    n_term = counts["n_term"]
    n_valid = counts["n_babies_with_ga"]
    n_missing = counts["n_missing_ga"]

    print(f"Unique babies with GA: {n_valid:,}")
    print(f"Unique preterm babies (GA < {PRETERM_CUTOFF} w): {n_preterm:,}")
    print(f"Unique term babies (GA >= {PRETERM_CUTOFF} w): {n_term:,}")
    if n_missing:
        print(f"Unique babies missing GA: {n_missing:,}")
    if n_valid:
        print(
            f"Preterm prevalence among unique babies: "
            f"{n_preterm / n_valid:.4f} ({n_preterm}/{n_valid})"
        )


def summarize_subjects_validation(df: pd.DataFrame) -> None:
    """Print mother, baby, and sample-type counts for the validation dataset."""
    df = prepare_validation_df(df)
    n_mothers, n_babies = count_unique_subjects_validation(df)
    n_unparsed = int(df["study_id"].map(parse_validation_baby_key).isna().sum())

    print(f"Unique mothers (study_id prefix + enrolment number): {n_mothers:,}")
    print(f"Unique babies (study_id baby key, heel/cord combined): {n_babies:,}")
    print(f"Total rows (samples): {len(df):,}")
    print(f"Unique study_id values: {df['study_id'].nunique():,}")
    if n_unparsed:
        print(f"Rows with unparseable study_id: {n_unparsed:,}")

    sample_types = df["study_id"].map(validation_sample_type)
    n_heel_babies = (
        df.loc[sample_types == "HEEL", "study_id"]
        .map(parse_validation_baby_key)
        .dropna()
        .nunique()
    )
    n_cord_babies = (
        df.loc[sample_types == "CORD", "study_id"]
        .map(parse_validation_baby_key)
        .dropna()
        .nunique()
    )
    print(f"Unique babies with HEEL sample: {n_heel_babies:,}")
    print(f"Unique babies with CORD sample: {n_cord_babies:,}")

    summarize_preterm_term_validation(df)


def summarize_validation_dataset(dataset_path: Path) -> None:
    """Load a validation dataset CSV and print subject-level descriptive statistics."""
    df = pd.read_csv(dataset_path)
    print(f"Dataset: {dataset_path.name}")
    summarize_subjects_validation(df)


def get_training_babies_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one deduplicated row per unique baby from the training dataset.

    The file has two rows per baby (HEEL and CORD samples).  All covariates
    are identical across those rows, so we collapse to one row per baby
    using (mother_id, child_id) before computing any stats.
    """
    if "mother_id" not in df.columns or "child_id" not in df.columns:
        raise KeyError("Expected columns 'mother_id' and 'child_id'.")
    return (
        df.dropna(subset=["mother_id", "child_id"])
        .drop_duplicates(subset=["mother_id", "child_id"])
        .reset_index(drop=True)
    )


def get_validation_babies_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one deduplicated row per unique baby from the validation newborn dataset.

    The newborn file has two rows per baby (cord + heel).  All covariates
    (GA, sex, BMI, etc.) are identical across those rows, so we collapse to
    one row per baby using the parsed baby key before computing any stats.
    """
    df = prepare_validation_df(df).copy()
    df["_baby_key"] = df["study_id"].map(parse_validation_baby_key)
    df = df.dropna(subset=["_baby_key"])
    return df.drop_duplicates(subset=["_baby_key"]).reset_index(drop=True)


def print_dataset_descriptives(df: pd.DataFrame) -> None:
    """Print gestational age, maternal age, sex, and BMI statistics for a baby-level dataframe."""
    ga = pd.to_numeric(df["gestational_age_weeks"], errors="coerce")
    n_valid_ga = int(ga.notna().sum())
    n_preterm = int((ga < PRETERM_CUTOFF).sum())
    preterm_prevalence = n_preterm / n_valid_ga if n_valid_ga else float("nan")
    print(f"Preterm prevalence (GA < {PRETERM_CUTOFF} w, non-null GA): {preterm_prevalence:.4f}")

    preterm_df = df[ga < PRETERM_CUTOFF]
    term_df = df[ga >= PRETERM_CUTOFF]

    for group_name, group in [("all", df), ("preterm", preterm_df), ("term", term_df)]:

        if "gestational_age_weeks" in group.columns:
            group_ga = pd.to_numeric(group["gestational_age_weeks"], errors="coerce")
            mean_ga = group_ga.mean()
            sd_ga = group_ga.std()
            print(f"Mean gestational age in {group_name}: {mean_ga:.2f} weeks, SD: {sd_ga:.2f} weeks")

        if "maternal_age_delivery" in group.columns:
            ma = pd.to_numeric(group["maternal_age_delivery"], errors="coerce")
            mean_maternal_age = ma.mean()
            sd_maternal_age = ma.std()
            print(f"Mean maternal age at delivery in {group_name}: {mean_maternal_age:.2f} years, SD: {sd_maternal_age:.2f} years")

        if "Age (years) (at enrolment)" in group.columns:
            ma = pd.to_numeric(group["Age (years) (at enrolment)"], errors="coerce")
            mean_maternal_age = ma.mean()
            sd_maternal_age = ma.std()
            print(f"Mean maternal age at delivery in {group_name}: {mean_maternal_age:.2f} years, SD: {sd_maternal_age:.2f} years")

        if "sex" in group.columns:
            sex = pd.to_numeric(group["sex"], errors="coerce")
            female_fraction = (sex == 2).mean()
            male_fraction = (sex == 1).mean()
            print(f"Female fraction in {group_name}: {female_fraction:.2f}, Male fraction in {group_name}: {male_fraction:.2f}")

        if "multiple_birth" in group.columns:
            multiple_birth = pd.to_numeric(group["multiple_birth"], errors="coerce")
            multiple_birth_fraction = (multiple_birth == 1).mean()
            single_birth_fraction = (multiple_birth == 2).mean()
            print(f"Multiple birth fraction in {group_name}: {multiple_birth_fraction:.4f}, Single birth fraction in {group_name}: {single_birth_fraction:.4f}")

        if "prepregnancy_bmi" in group.columns:
            prepregnancy_bmi = pd.to_numeric(group["prepregnancy_bmi"], errors="coerce")
            mean_prepregnancy_bmi = prepregnancy_bmi.mean()
            sd_prepregnancy_bmi = prepregnancy_bmi.std()
            print(f"Mean prepregnancy BMI in {group_name}: {mean_prepregnancy_bmi:.2f}, SD: {sd_prepregnancy_bmi:.2f}")


def main() -> None:
    """Print descriptive statistics for the training and validation cohorts."""
    training_df = pd.read_csv(TRAINING_DATASET_PATH)
    print(f"Dataset: {TRAINING_DATASET_PATH.name}")
    print_mothers_and_babies(training_df)
    print()
    training_babies_df = get_training_babies_df(training_df)
    print_dataset_descriptives(training_babies_df)

    print("\n" + "=" * 72 + "\n")

    validation_df = pd.read_csv(VALIDATION_NEWBORN_DATASET_PATH)
    print(f"Dataset: {VALIDATION_NEWBORN_DATASET_PATH.name}")
    summarize_subjects_validation(validation_df)
    print()
    validation_babies_df = get_validation_babies_df(validation_df)
    print_dataset_descriptives(validation_babies_df)


if __name__ == "__main__":
    main()
