"""Transforms the raw external validation CSV into the shared feature space required by the training pipeline, including column renaming, clinical variable encoding, birth-weight unit conversion, and maternal data merging."""

from pathlib import Path
import re
from typing import Optional

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OUT_PATH = _PROJECT_ROOT / "data" / "ZimbabweNewborn_samefeatures_as_Bangladesh.csv"
_MATERNAL_PATH = _PROJECT_ROOT / "data" / "ZimbabweMaternalOnly (1).csv"
_MATERNAL_BMI_COL = "Body mass index (kg/m2) (preprenancy)"  # Note: column name "preprenancy" is a typo in the source CSV
_MATERNAL_SAMPLE_COL = "Sample type (Cord or Heel)"
_MATERNAL_BW_COL = "Birth weight (grams)"
_MATERNAL_DOB_COL = "Date of Birth (yyyy/mm/dd) "
_MATERNAL_SEX_COL = "Sex (M,F, Unknown)"
_NEWBORN_DOB_COL = "Birth_Date_Time"

# Match Bangladesh_children_with_both_samples.csv: 1=male, 2=female; 1=multiple, 2=single
_SEX_MAP = {"M": 1, "MALE": 1, "F": 2, "FEMALE": 2}
_MULTIPLE_BIRTH_MAP = {"YES": 1, "Y": 1, "NO": 2, "N": 2}

# After rename_dots_to_underscores: map Zimbabwe column names to Bangladesh-style names.
# study_id is validation-only (not in Bangladesh training features).
_COLUMN_RENAMES = {
    "Study_ID": "study_id",
}

FEATURES_TO_FIND = ["study_id",
        "gestational_age_weeks",
        "birth_weight_kg",
        "sex",
        "multiple_birth",
        "prepregnancy_bmi",
        "maternal_age_delivery",
        "TSH",
        "N17P",
        "IRT",
        "GALT",
        "BIOT",
        "Ala",
        "Arg",
        "ASA",
        "ASA_Arg",
        "ASA_Orn",
        "Cit",
        "Cit_Arg",
        "Cit_Orn",
        "Cit_Tyr",
        "Gly",
        "LEU",
        "Leu_Ala",
        "Leu_Phe",
        "MET",
        "Met_Phe",
        "Orn",
        "Orn_Arg",
        "Orn_Cit",
        "Orn_Phe",
        "PHE",
        "PHE_TYR",
        "SUAC",
        "TYR",
        "Tyr_Phe",
        "Val",
        "Val_Ala",
        "Val_Phe",
        "C0",
        "C0_C16C18",
        "C0C2C3C16C18_Cit",
        "C2",
        "C3",
        "C3_C0",
        "C3_C16",
        "C3_C2",
        "C3_C4DC",
        "C3DC",
        "C4",
        "C4DC",
        "C4OH",
        "C5",
        "C5_C0",
        "C5_C2",
        "C5_C3",
        "C5_1",
        "C5DC",
        "C5DC_C16",
        "C5DC_C5OH",
        "C5DC_C8",
        "C5OH",
        "C5OH_C2",
        "C5OH_C5_1",
        "C5OH_C8",
        "C6",
        "C6DC",
        "C8",
        "C8_C10",
        "C8_C2",
        "C8_1",
        "C10",
        "C10_1",
        "C12",
        "C12_1",
        "C14",
        "C14OH",
        "C14_1",
        "C14_1_C12_1",
        "C14_1_C16",
        "C14_1_C4",
        "C14_2",
        "C16",
        "C16_1OH",
        "C16_1OH_C4DC",
        "C16OH",
        "C16OH_C16",
        "C18",
        "C18_1",
        "C18_1OH",
        "C18_2",
        "C18OH",
        "HGB___FAST",
        "HGB___F1",
        "HGB___F",
        "HGB___F_F1",
        "HGB___A",
        "HGB___FAST_F1",
        "HGB___Other",
    ]

def _maternal_age_column(df: pd.DataFrame) -> str:
    """Find and return the maternal age column name from the Zimbabwe maternal CSV."""
    for col in df.columns:
        if "Age (years)" in col and "enrolment" in col:
            return col
    raise KeyError("Maternal age column not found in ZimbabweMaternalOnly CSV")


def parse_maternal_baby_key(raw_id) -> Optional[tuple]:
    """Parse maternal first-column ID to (prefix, number, twin)."""
    raw = str(raw_id).strip().upper()
    if raw in ("NAN", ""):
        return None
    twin = None
    twin_match = re.search(r"TWIN\s*(\d+)", raw)
    if twin_match:
        twin = int(twin_match.group(1))
    cleaned = re.sub(r"TWIN\s*\d+", "", raw).strip()
    match = re.match(r"^([A-Z]+)[\s-]+(\d+)(?:[\s-]+(\d+))?", cleaned)
    if not match:
        match = re.match(r"^([A-Z]+)-?(\d+)", cleaned.replace(" ", ""))
    if not match:
        return None
    if twin is None and match.lastindex and match.lastindex >= 3 and match.group(3):
        twin = int(match.group(3))
    return (match.group(1), int(match.group(2)), twin)


def parse_newborn_baby_key(study_id) -> Optional[tuple]:
    """Parse newborn study_id to (prefix, number, twin)."""
    if pd.isna(study_id):
        return None
    compact = str(study_id).strip().upper().replace(" ", "")
    match = re.match(r"^([A-Z]+)-?(\d+)(?:-(C|H)(\d+))?(?:-(\d+))?$", compact)
    if not match:
        return None
    twin = int(match.group(5)) if match.group(5) else None
    return (match.group(1), int(match.group(2)), twin)


def maternal_id_to_study_id(raw_id, sample_type=None) -> Optional[str]:
    """Convert maternal first-column ID to newborn study_id format (e.g. SM-0222-C1)."""
    key = parse_maternal_baby_key(raw_id)
    if not key:
        return None
    prefix, num, twin = key
    sample = str(sample_type or "").strip().upper()
    letter = "C" if sample == "CORD" else ("H" if sample == "HEEL" else "C")
    study_id = f"{prefix}-{num:04d}-{letter}1"
    if twin is not None:
        study_id = f"{study_id}-{twin}"
    return study_id


def normalize_study_id(study_id) -> Optional[str]:
    """Canonical newborn study_id (e.g. SM0003-C1 -> SM-0003-C1)."""
    if pd.isna(study_id):
        return None
    compact = str(study_id).strip().upper().replace(" ", "")
    match = re.match(r"^([A-Z]+)-?(\d+)(?:-(C|H)(\d+))?(?:-(\d+))?$", compact)
    if not match:
        return None
    twin = f"-{match.group(5)}" if match.group(5) else ""
    letter = match.group(3) or "C"
    sample_num = match.group(4) or "1"
    return f"{match.group(1)}-{int(match.group(2)):04d}-{letter}{sample_num}{twin}"


def newborn_base_study_id(study_id) -> Optional[str]:
    """Return the base study_id (prefix + enrolment number) without sample-type suffix."""
    normalized = normalize_study_id(study_id)
    if not normalized:
        return None
    match = re.match(r"^([A-Z]+-\d+)", normalized)
    return match.group(1) if match else None


def parse_prefix_num(study_id) -> Optional[tuple]:
    """Return (site_prefix, enrolment_number) from either a maternal or newborn ID."""
    key = parse_maternal_baby_key(study_id) or parse_newborn_baby_key(study_id)
    if not key:
        return None
    return (key[0], key[1])


def parse_maternal_ga(value):
    """Extract a numeric gestational age (weeks) from a free-text maternal GA field."""
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else None


def parse_maternal_dob(value):
    """Parse a date-of-birth string in yyyy/Month/dd or similar formats to a Timestamp."""
    text = str(value).strip()
    if not text or text.upper() == "NAN":
        return pd.NaT
    match = re.match(r"(\d{4})/([A-Za-z]+)/(\d+)", text)
    if match:
        for fmt in ("%Y %B %d", "%Y %b %d"):
            parsed = pd.to_datetime(
                f"{match.group(1)} {match.group(2)} {match.group(3)}",
                format=fmt,
                errors="coerce",
            )
            if pd.notna(parsed):
                return parsed
    return pd.to_datetime(text, errors="coerce")


def normalize_sex(value) -> Optional[str]:
    """Normalize a sex field to 'M', 'F', or None."""
    text = str(value).strip().upper()
    if text in ("M", "MALE", "1", "1.0"):
        return "M"
    if text in ("F", "FEMALE", "2", "2.0"):
        return "F"
    return text[:1] if text and text != "NAN" else None


def newborn_birth_weight_grams(series: pd.Series) -> pd.Series:
    """Convert birth weight to grams, auto-detecting if values are already in kg."""
    bw = pd.to_numeric(series, errors="coerce")
    if bw.median(skipna=True) <= 50:
        return (bw * 1000).round()
    return bw.round()


def load_maternal_table() -> pd.DataFrame:
    """Load maternal CSV; IDs always come from the first column."""
    df = pd.read_csv(_MATERNAL_PATH)
    age_col = _maternal_age_column(df)
    maternal_ids = df.iloc[:, 0]
    ga_col = "Gestational Age estimate by dates (weeks+ days)at enrolment"

    out = pd.DataFrame(
        {
            "maternal_id_raw": maternal_ids.astype(str).str.strip(),
            "baby_key": maternal_ids.map(parse_maternal_baby_key),
            "prefix_num": maternal_ids.map(parse_prefix_num),
            "study_id": [
                maternal_id_to_study_id(raw_id, sample)
                for raw_id, sample in zip(maternal_ids, df[_MATERNAL_SAMPLE_COL])
            ],
            "bw_g": pd.to_numeric(df[_MATERNAL_BW_COL], errors="coerce"),
            "dob": df[_MATERNAL_DOB_COL].map(parse_maternal_dob).dt.normalize(),
            "ga_w": df[ga_col].map(parse_maternal_ga),
            "sex_n": df[_MATERNAL_SEX_COL].map(normalize_sex),
            "sample_n": df[_MATERNAL_SAMPLE_COL].astype(str).str.strip().str.upper(),
            "prepregnancy_bmi": pd.to_numeric(df[_MATERNAL_BMI_COL], errors="coerce"),
            "maternal_age_delivery": pd.to_numeric(df[age_col], errors="coerce"),
        }
    )
    out["mat_row"] = range(len(out))
    out["prefix"] = out["prefix_num"].map(lambda x: x[0] if x else None)
    return out


def _build_newborn_match_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Build a lookup frame of normalised identifiers for each newborn row."""
    ga = pd.to_numeric(df_raw["gestational_age_weeks"], errors="coerce")
    return pd.DataFrame(
        {
            "study_id": df_raw["study_id"],
            "norm_id": df_raw["study_id"].map(normalize_study_id),
            "base_id": df_raw["study_id"].map(newborn_base_study_id),
            "baby_key": df_raw["study_id"].map(parse_newborn_baby_key),
            "prefix_num": df_raw["study_id"].map(parse_prefix_num),
            "prefix": df_raw["study_id"].map(
                lambda s: parse_prefix_num(s)[0] if parse_prefix_num(s) else None
            ),
            "bw_g": newborn_birth_weight_grams(df_raw["birth_weight_kg"]),
            "dob": pd.to_datetime(df_raw[_NEWBORN_DOB_COL], errors="coerce").dt.normalize(),
            "ga_w": ga,
            "ga_r": ga.round(),
            "sex_n": df_raw["sex"].map(normalize_sex),
        }
    )


def _build_maternal_lookups(mat: pd.DataFrame) -> dict:
    """Build a multi-key lookup dict mapping various identifier combinations to maternal row indices."""
    lookups = {}
    for _, row in mat.iterrows():
        row_idx = row["mat_row"]
        if row["study_id"]:
            lookups[("sid", row["study_id"])] = row_idx
        if row["prefix_num"]:
            lookups[("pn", row["prefix_num"])] = row_idx
        if pd.notna(row["bw_g"]) and pd.notna(row["dob"]):
            lookups[("bd", (row["bw_g"], row["dob"]))] = row_idx
        if pd.notna(row["bw_g"]) and pd.notna(row["ga_w"]):
            lookups[("bgs", (row["bw_g"], row["ga_w"], row["sex_n"]))] = row_idx
            lookups[("bg", (row["bw_g"], row["ga_w"]))] = row_idx
        if pd.notna(row["bw_g"]):
            lookups[("bw", row["bw_g"])] = row_idx
        if pd.notna(row["ga_w"]):
            lookups[("gs", (row["ga_w"], row["sex_n"]))] = row_idx
            lookups[("gsr", (round(row["ga_w"]), row["sex_n"]))] = row_idx
        if pd.notna(row["ga_w"]) and row["prefix"]:
            lookups[("pgs", (row["prefix"], round(row["ga_w"]), row["sex_n"]))] = row_idx
    return lookups


def _lookup_key_valid(key_name, value) -> bool:
    """Return True if value is a usable (non-null, non-None) lookup key."""
    if value is None:
        return False
    if isinstance(value, tuple):
        return not any(pd.isna(part) for part in value)
    return not pd.isna(value)


def _direct_maternal_match(row: pd.Series, lookups: dict) -> object:
    """Return the maternal row index for a newborn row using prioritised key fallbacks."""
    candidates = [
        ("sid", row["norm_id"]),
        ("pn", row["prefix_num"]),
        ("bd", (row["bw_g"], row["dob"])),
        ("bgs", (row["bw_g"], row["ga_w"], row["sex_n"])),
        ("bg", (row["bw_g"], row["ga_w"])),
        ("bw", row["bw_g"]),
        ("gs", (row["ga_w"], row["sex_n"])),
        ("gsr", (row["ga_r"], row["sex_n"])),
        ("pgs", (row["prefix"], row["ga_r"], row["sex_n"])),
    ]
    for key_name, value in candidates:
        if _lookup_key_valid(key_name, value) and (key_name, value) in lookups:
            return lookups[(key_name, value)]
    return pd.NA


def _propagate_maternal_matches(newborn: pd.DataFrame) -> pd.Series:
    """Propagate maternal matches from matched newborn rows to their unmatched siblings."""
    propagation_fields = [
        ("norm_id", "norm_id"),
        ("base_id", "base_id"),
        ("pn", "prefix_num"),
        ("bd", lambda r: (r["bw_g"], r["dob"])),
        ("bgs", lambda r: (r["bw_g"], r["ga_w"], r["sex_n"])),
        ("bg", lambda r: (r["bw_g"], r["ga_w"])),
        ("bw", lambda r: r["bw_g"]),
        ("gs", lambda r: (r["ga_w"], r["sex_n"])),
        ("gsr", lambda r: (r["ga_r"], r["sex_n"])),
        ("pgs", lambda r: (r["prefix"], r["ga_r"], r["sex_n"])),
    ]

    matched = newborn["mat_row"].copy()
    for _ in range(8):
        propagated = {}
        for _, row in newborn.iterrows():
            if pd.isna(row["mat_row"]):
                continue
            for field_name, getter in propagation_fields:
                value = getter(row) if callable(getter) else row[getter]
                if _lookup_key_valid(field_name, value):
                    propagated.setdefault(field_name, {})[value] = row["mat_row"]

        updated = []
        for _, row in newborn.iterrows():
            if pd.notna(row["mat_row"]):
                updated.append(row["mat_row"])
                continue
            found = pd.NA
            for field_name, getter in propagation_fields:
                value = getter(row) if callable(getter) else row[getter]
                if _lookup_key_valid(field_name, value):
                    found = propagated.get(field_name, {}).get(value, pd.NA)
                if pd.notna(found):
                    break
            updated.append(found)
        matched = pd.Series(updated, index=newborn.index, dtype="object")
        newborn["mat_row"] = matched
        if matched.notna().all():
            break
    return matched.astype("Int64")


def match_newborn_to_maternal(df_raw: pd.DataFrame) -> pd.Series:
    """
    Match every newborn row to a maternal record.

    Priority: normalized study_id, prefix+number, birth weight/date, then
    progressively broader clinical keys. Unmatched cord/heel pairs inherit the
    maternal record from a matched sibling via iterative propagation.
    """
    mat = load_maternal_table()
    lookups = _build_maternal_lookups(mat)
    newborn = _build_newborn_match_frame(df_raw)
    newborn["mat_row"] = newborn.apply(lambda row: _direct_maternal_match(row, lookups), axis=1)
    matched = _propagate_maternal_matches(newborn)

    if matched.isna().any():
        missing = newborn.loc[matched.isna(), "study_id"].tolist()
        raise ValueError(
            f"Could not match {matched.isna().sum()} newborn rows to maternal data. "
            f"Examples: {missing[:5]}"
        )
    return matched


def append_maternal_clinical(df_out: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """Attach prepregnancy_bmi and maternal_age_delivery to every newborn row."""
    if "study_id" not in df_out.columns:
        raise KeyError("study_id required to merge maternal clinical data")

    mat = load_maternal_table()
    matched_rows = match_newborn_to_maternal(df_raw)
    out = df_out.copy()
    out["prepregnancy_bmi"] = matched_rows.apply(lambda i: mat.loc[i, "prepregnancy_bmi"])
    out["maternal_age_delivery"] = matched_rows.apply(lambda i: mat.loc[i, "maternal_age_delivery"])
    return out


def rename_dots_to_underscores(df: pd.DataFrame) -> pd.DataFrame:
    """Replace '.' with '_' in all column names."""
    out = df.copy()
    out.columns = [str(c).replace(".", "_") for c in out.columns]
    return out


def apply_column_renames(df: pd.DataFrame) -> pd.DataFrame:
    """Rename validation columns to match training dataset naming where needed."""
    out = df.copy()
    rename = {src: dst for src, dst in _COLUMN_RENAMES.items() if src in out.columns}
    if rename:
        out = out.rename(columns=rename)
    return out


def encode_sex_numeric(series: pd.Series) -> pd.Series:
    """Encode sex as 1 = male, 2 = female (numeric, matching training CSV convention)."""
    as_num = pd.to_numeric(series, errors="coerce")
    as_str = series.astype(str).str.strip().str.upper()
    mapped = as_str.map(_SEX_MAP)
    out = mapped.fillna(as_num)
    unknown = out.isna() & series.notna()
    if unknown.any():
        bad = sorted(series[unknown].astype(str).unique().tolist())
        raise ValueError(f"Unmapped sex values: {bad}")
    return out.astype("float64")


def encode_multiple_birth_numeric(series: pd.Series) -> pd.Series:
    """Encode multiple_birth as 1 = multiple, 2 = single (numeric, matching training CSV convention)."""
    as_num = pd.to_numeric(series, errors="coerce")
    as_str = series.astype(str).str.strip().str.upper()
    mapped = as_str.map(_MULTIPLE_BIRTH_MAP)
    out = mapped.fillna(as_num)
    unknown = out.isna() & series.notna()
    if unknown.any():
        bad = sorted(series[unknown].astype(str).unique().tolist())
        raise ValueError(f"Unmapped multiple_birth values: {bad}")
    return out.astype("float64")


def apply_clinical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Apply numeric encoding for sex and multiple_birth columns to match training conventions."""
    out = df.copy()
    if "sex" in out.columns:
        out["sex"] = encode_sex_numeric(out["sex"])
    if "multiple_birth" in out.columns:
        out["multiple_birth"] = encode_multiple_birth_numeric(out["multiple_birth"])
    return out


def convert_birth_weight_to_kg(df: pd.DataFrame) -> pd.DataFrame:
    """Zimbabwe stores grams; Bangladesh CSV uses kilograms."""
    out = df.copy()
    if "birth_weight_kg" not in out.columns:
        return out
    bw = pd.to_numeric(out["birth_weight_kg"], errors="coerce")
    if bw.median(skipna=True) > 50:
        out["birth_weight_kg"] = bw / 1000.0
    return out


def build_samefeatures_dataset() -> pd.DataFrame:
    """Build the validation dataset aligned to the training feature space."""
    df = rename_dots_to_underscores(
        pd.read_csv(_PROJECT_ROOT / "data" / "ZimbabweNewborn_nomiss.csv")
    )
    df = apply_column_renames(df)
    cols = [c for c in FEATURES_TO_FIND if c in df.columns]
    if "gestational_age_weeks" not in cols:
        raise KeyError("gestational_age_weeks missing from Zimbabwe nomiss data")
    df_out = df[cols].copy()
    df_out = convert_birth_weight_to_kg(df_out)
    df_out = apply_clinical_encoding(df_out)
    df_out = append_maternal_clinical(df_out, df)
    ordered = [c for c in FEATURES_TO_FIND if c in df_out.columns]
    extra = [c for c in df_out.columns if c not in ordered]
    return df_out[ordered + extra]


def find_train_features_in_val(df_train: pd.DataFrame, df_val: pd.DataFrame) -> list:
    """Return the subset of FEATURES_TO_FIND that are present in the validation dataframe."""
    return [c for c in FEATURES_TO_FIND if c in df_val.columns]


def main():
    """Build the validation dataset CSV aligned to the training feature space and write to disk."""
    df_val = build_samefeatures_dataset()
    df_val.to_csv(_OUT_PATH, index=False)

    print(f"Wrote {_OUT_PATH} ({df_val.shape[0]} rows, {df_val.shape[1]} columns)\n")
    print("gestational_age_weeks:", df_val["gestational_age_weeks"].describe().to_dict())
    print("birth_weight_kg (kg):", df_val["birth_weight_kg"].describe().to_dict())
    print("sex (1=male, 2=female):")
    print(df_val["sex"].value_counts(dropna=False).sort_index())
    print("multiple_birth (1=multiple, 2=single):")
    print(df_val["multiple_birth"].value_counts(dropna=False).sort_index())

    if "prepregnancy_bmi" in df_val.columns:
        n_maternal_age = df_val["maternal_age_delivery"].notna().sum()
        n_maternal_bmi = df_val["prepregnancy_bmi"].notna().sum()
        print(
            f"\nMaternal data merged: {n_maternal_age}/{len(df_val)} rows with maternal_age_delivery, "
            f"{n_maternal_bmi}/{len(df_val)} with prepregnancy_bmi"
        )
        if n_maternal_age != len(df_val):
            print("WARNING: not all rows received maternal age")
        if n_maternal_bmi != len(df_val):
            print("NOTE: prepregnancy_bmi missing where maternal source BMI is blank")
        print("prepregnancy_bmi:", df_val["prepregnancy_bmi"].describe().to_dict())
        print("maternal_age_delivery:", df_val["maternal_age_delivery"].describe().to_dict())

    features_not_in_val = set(FEATURES_TO_FIND) - set(df_val.columns)
    print(f"\nColumns in FEATURES_TO_FIND but not in output: {sorted(features_not_in_val)}")

if __name__ == "__main__":
    main()
