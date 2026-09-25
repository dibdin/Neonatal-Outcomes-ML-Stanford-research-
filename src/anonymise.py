"""
Produce anonymised CSVs for the Bangladesh and Zimbabwe datasets.

Removes all identifying columns; adds:
  - participant_id  : sequential integer, one unique value per participant
                      (twins in Bangladesh get separate IDs)
  - sample_type     : "cord" or "heel", derived from the source column / study_id

Usage:
    python -m src.anonymise
"""

from pathlib import Path
import re
import sys

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

# Columns to keep (in order) after adding participant_id and sample_type.
# Must match the training pipeline's expectations.
KEEP_COLS = [
    "participant_id",
    "sample_type",
    "gestational_age_weeks",
    "birth_weight_kg",
    "sex",
    "multiple_birth",
    "prepregnancy_bmi",
    "maternal_age_delivery",
    "TSH", "N17P", "IRT", "GALT", "BIOT",
    "Ala", "Arg", "ASA", "ASA_Arg", "ASA_Orn",
    "Cit", "Cit_Arg", "Cit_Orn", "Cit_Tyr",
    "Gly", "LEU", "Leu_Ala", "Leu_Phe",
    "MET", "Met_Phe",
    "Orn", "Orn_Arg", "Orn_Cit", "Orn_Phe",
    "PHE", "PHE_TYR", "SUAC", "TYR", "Tyr_Phe",
    "Val", "Val_Ala", "Val_Phe",
    "C0", "C0_C16C18", "C0C2C3C16C18_Cit",
    "C2", "C3", "C3_C0", "C3_C16", "C3_C2", "C3_C4DC", "C3DC",
    "C4", "C4DC", "C4OH",
    "C5", "C5_C0", "C5_C2", "C5_C3", "C5_1",
    "C5DC", "C5DC_C16", "C5DC_C5OH", "C5DC_C8",
    "C5OH", "C5OH_C2", "C5OH_C5_1", "C5OH_C8",
    "C6", "C6DC",
    "C8", "C8_C10", "C8_C2", "C8_1",
    "C10", "C10_1", "C12", "C12_1",
    "C14", "C14OH", "C14_1", "C14_1_C12_1", "C14_1_C16", "C14_1_C4", "C14_2",
    "C16", "C16_1OH", "C16_1OH_C4DC", "C16OH", "C16OH_C16",
    "C18", "C18_1", "C18_1OH", "C18_2", "C18OH",
    "HGB___FAST", "HGB___F1", "HGB___F", "HGB___F_F1",
    "HGB___A", "HGB___FAST_F1", "HGB___Other",
]


def _base_study_id_bangladesh(study_id: str) -> str:
    """Strip the -C1 / -H1 (and optional twin suffix) from a Bangladesh Study_ID.

    '1-2-533-C1-2' -> '1-2-533'
    '1-1-005-H1'   -> '1-1-005'
    """
    base = re.sub(r"-[CH]\d+(?:-\d+)?$", "", str(study_id).strip())
    return base.rstrip("-")


def _sample_type_from_source(source: str) -> str:
    """Map Bangladesh Source values to 'cord' / 'heel'."""
    s = str(source).strip().upper()
    if s == "CORD":
        return "cord"
    if s == "HEEL":
        return "heel"
    raise ValueError(f"Unknown Source value: {source!r}")


def anonymise_bangladesh() -> pd.DataFrame:
    """
    Anonymise Bangladesh_children_with_both_samples.csv.

    participant_id is assigned per unique (base_study_id, child_id) combination
    so that twins (same base_study_id, different child_id) receive separate IDs.
    """
    src = _DATA_DIR / "Bangladesh_children_with_both_samples.csv"
    df = pd.read_csv(src)

    # Derive anonymisation keys
    df["_base_study_id"] = df["Study_ID"].map(_base_study_id_bangladesh)
    # child_id is 1.0 or 2.0 (NaN for singletons); treat NaN as 1
    df["_child_id"] = df["child_id"].fillna(1).astype(int)
    df["sample_type"] = df["Source"].map(_sample_type_from_source)

    # Assign sequential participant_id per unique (_base_study_id, _child_id)
    groups = (
        df[["_base_study_id", "_child_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    groups["participant_id"] = groups.index + 1
    df = df.merge(groups, on=["_base_study_id", "_child_id"], how="left")

    # Rename GA column if needed
    if "gestational_age_weeks" not in df.columns and "Gestational_Age_Weeks" in df.columns:
        df = df.rename(columns={"Gestational_Age_Weeks": "gestational_age_weeks"})

    keep = [c for c in KEEP_COLS if c in df.columns]
    out = df[keep].copy()
    return out


def _sample_type_from_study_id(study_id: str) -> str:
    """
    Derive sample type from a Zimbabwe study_id.

    The study_id encodes the sample type via -C (cord) or -H (heel):
      SM-0001-C1      -> cord
      SM-0001-H1      -> heel
      SM-0309-*HEEL*  -> heel  (non-standard but interpretable)
    """
    s = str(study_id).strip().upper()
    if "HEEL" in s:
        return "heel"
    if "CORD" in s:
        return "cord"
    match = re.search(r"-([CH])\d+", s)
    if match:
        return "cord" if match.group(1) == "C" else "heel"
    raise ValueError(f"Cannot determine sample type from study_id: {study_id!r}")


def _base_study_id_zimbabwe(study_id: str) -> str:
    """
    Return the participant key for a Zimbabwe study_id.

    The key is the enrolment ID plus the twin number (if any), so that
    twins (e.g. BD-0012-C1-1 and BD-0012-C1-2) get separate participant IDs
    while cord and heel from the same participant collapse to one key.

      SM-0001-C1      -> SM-0001        (singleton)
      SM0003-C1       -> SM0003         (compact format, singleton)
      BD-0012-C1-1    -> BD-0012-1      (twin child 1)
      BD-0012-C1-2    -> BD-0012-2      (twin child 2)
      RT-0072-H1-1    -> RT-0072-1      (twin child 1)
      SM-0309-*HEEL*  -> SM-0309        (non-standard, singleton)
    """
    s = str(study_id).strip().upper()
    # Non-standard format: strip from -*HEEL*/*CORD* onward
    s = re.sub(r"-\*?(HEEL|CORD)\*?.*$", "", s)
    # Capture optional trailing twin number after the sample-type suffix
    twin_match = re.search(r"-[CH]\d+-?(\d+)$", s)
    twin_suffix = f"-{twin_match.group(1)}" if twin_match else ""
    # Remove the sample-type suffix (and any trailing twin number)
    s = re.sub(r"-[CH]\d+.*$", "", s)
    return s + twin_suffix


def anonymise_zimbabwe() -> pd.DataFrame:
    """
    Anonymise ZimbabweNewborn_samefeatures_as_Bangladesh.csv.

    participant_id is assigned per unique base study_id (participant enrolment number).
    sample_type is derived from the -C / -H in the study_id, NOT by row position.
    """
    src = _DATA_DIR / "ZimbabweNewborn_samefeatures_as_Bangladesh.csv"
    df = pd.read_csv(src)

    # Drop exact study_id duplicates (data-entry errors); keep first occurrence.
    # e.g. EO-0018-H1 appears twice with different GA/BW values.
    n_before = len(df)
    df = df.drop_duplicates(subset=["study_id"], keep="first").copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  [INFO] Dropped {n_dropped} exact study_id duplicate row(s).")

    df["_base_study_id"] = df["study_id"].map(_base_study_id_zimbabwe)
    df["sample_type"] = df["study_id"].map(_sample_type_from_study_id)

    # Assign sequential participant_id per unique base study_id
    groups = (
        df[["_base_study_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    groups["participant_id"] = groups.index + 1
    df = df.merge(groups, on="_base_study_id", how="left")

    keep = [c for c in KEEP_COLS if c in df.columns]
    out = df[keep].copy()
    return out


def _validate(df: pd.DataFrame, label: str) -> None:
    """Check that every participant has at most one cord and one heel entry."""
    counts = df.groupby(["participant_id", "sample_type"]).size()
    bad = counts[counts > 1]
    if not bad.empty:
        print(f"[WARN] {label}: {len(bad)} participant/sample_type combos with >1 row:")
        print(bad.to_string())
    else:
        print(f"[OK]   {label}: all participants have at most one cord and one heel entry.")


def main() -> None:
    print("Anonymising Bangladesh data...")
    df_b = anonymise_bangladesh()
    out_b = _DATA_DIR / "bangladesh_anonymised.csv"
    df_b.to_csv(out_b, index=False)
    print(f"  Wrote {out_b}  ({df_b.shape[0]} rows, {df_b.shape[1]} cols, "
          f"{df_b['participant_id'].nunique()} participants)")
    _validate(df_b, "Bangladesh")

    print()
    print("Anonymising Zimbabwe data...")
    df_z = anonymise_zimbabwe()
    out_z = _DATA_DIR / "zimbabwe_anonymised.csv"
    df_z.to_csv(out_z, index=False)
    print(f"  Wrote {out_z}  ({df_z.shape[0]} rows, {df_z.shape[1]} cols, "
          f"{df_z['participant_id'].nunique()} participants)")
    _validate(df_z, "Zimbabwe")


if __name__ == "__main__":
    main()
