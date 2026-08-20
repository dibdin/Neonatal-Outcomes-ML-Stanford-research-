"""Binary classification of gestational age (preterm vs. term).

Training (all option-1 configurations):
    python src/classification.py --all --data_option 1

External validation on external cohort:
    python src/classification.py --validate --all --data_option 1

Single configuration:
    python src/classification.py --data_option 1 --model_type biomarker --model_name lasso_cv --data_type cord
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import RANDOM_SEED, N_REPEATS, PRETERM_CUTOFF, TRAINING_CSV_BOTH, TRAINING_CSV_ALL
from src.model import get_classification_model
from src.feature_sets import (
    CLINICAL_FEATURES,
    BIOMARKER_FEATURES,
    TARGET_COL,
    load_training_frame,
    load_validation_frame,
    align_to_training_features,
)

OUTPUT_DIR = Path("outputs/classification")


def _run_dir(model_name: str, data_option: int, model_type: str, data_type: str) -> Path:
    """Return the output subdirectory for a given training configuration."""
    if data_option == 1:
        return OUTPUT_DIR / f"{model_name}__opt{data_option}__{model_type}__{data_type}"
    return OUTPUT_DIR / f"{model_name}__opt{data_option}__{model_type}"


def _extract_coef(model, step_name: str) -> list:
    """Extract the coefficient array from the named step of a fitted GridSearchCV pipeline."""
    coef_flat = []
    if hasattr(model, "best_estimator_"):
        est = model.best_estimator_
        if hasattr(est, "named_steps"):
            step = est.named_steps.get(step_name)
            if step is not None and hasattr(step, "coef_"):
                c = step.coef_
                coef_flat = c.flatten() if c.ndim > 1 else c
        elif hasattr(est, "coef_"):
            c = est.coef_
            coef_flat = c.flatten() if c.ndim > 1 else c
    elif hasattr(model, "coef_"):
        c = model.coef_
        coef_flat = c.flatten() if c.ndim > 1 else c
    return coef_flat


def run_validation(
    data_option: int = 1,
    model_type: str = "clinical",
    model_name: str = "elasticnet_cv",
    data_type: str = "heel",
    run_number: int = N_REPEATS,
) -> None:
    """External validation: models trained on training cohort splits, evaluated on external validation cohort."""
    X_train_full, y_cont, feature_names = load_training_frame(data_option, data_type, model_type)
    y_train_full = (y_cont < PRETERM_CUTOFF).astype(int)

    X_val, y_val, y_cont_val, study_id, val_feature_names = load_validation_frame(
        model_type, data_type=data_type if data_option == 1 else None
    )
    X_val = align_to_training_features(X_val, feature_names)

    missing_in_val = [c for c in feature_names if c not in val_feature_names]
    if missing_in_val:
        print(
            f"WARNING: {len(missing_in_val)} training features absent from validation set "
            f"(will be imputed as NaN): {missing_in_val}"
        )

    val_out_dir = _run_dir(model_name, data_option, model_type, data_type) / "validation_zimbabwe"
    val_out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": feature_names}).to_csv(val_out_dir / "feature_names.csv", index=False)

    auc_rows, predictions_all, hyperparameters_all, coefficients_all = [], [], [], []

    for i in range(run_number):
        X_train, _, y_train, _ = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.2,
            random_state=RANDOM_SEED + i,
            stratify=y_train_full,
        )

        model, _ = get_classification_model(model_name)
        model.fit(X_train, y_train)

        y_proba = (
            model.predict_proba(X_val)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_val)
        )

        auc = roc_auc_score(y_val, y_proba)
        auc_rows.append({"run": i, "auc": auc, "n_val": len(y_val)})

        predictions_all.append(pd.DataFrame({
            "run": i,
            "study_id": study_id.values,
            "y_true": y_val.values,
            "y_pred": y_proba,
            "gestational_age_weeks": y_cont_val.values,
        }))

        best_params = model.best_params_ if hasattr(model, "best_params_") else {}
        hyperparameters_all.append(pd.DataFrame({
            "run": i,
            "C": best_params.get("logisticregression__C", None),
            "l1_ratio": best_params.get("logisticregression__l1_ratio", None),
        }, index=[i]))

        coef_flat = _extract_coef(model, step_name="logisticregression")
        coefficients_all.append(pd.DataFrame({
            "run": i,
            "coefficient names": [col for col, coef in zip(feature_names, coef_flat) if coef != 0],
            "coefficient values": [coef for coef in coef_flat if coef != 0],
        }))

    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(val_out_dir / "auc_all.csv", index=False)
    pd.concat(predictions_all, ignore_index=True).to_csv(val_out_dir / "predictions_all.csv", index=False)
    if hyperparameters_all:
        pd.concat(hyperparameters_all, ignore_index=True).to_csv(
            val_out_dir / "hyperparameters_all.csv", index=False
        )
    if coefficients_all:
        pd.concat(coefficients_all, ignore_index=True).to_csv(
            val_out_dir / "coefficients_all.csv", index=False
        )

    auc_mean, auc_std = auc_df["auc"].mean(), auc_df["auc"].std()
    print(f"Validation AUC: mean={auc_mean:.4f}, std={auc_std:.4f}, n={len(y_val)}")
    print(f"Results saved to {val_out_dir}/")


def run_classification_model(
    data_option: int = 1,
    model_type: str = "clinical",
    model_name: str = "elasticnet_cv",
    data_type: str = "heel",
    run_number: int = N_REPEATS,
) -> None:
    """Train a regularised logistic regression classifier with repeated train/test splits."""
    data_type_norm = str(data_type).strip().lower()
    if data_option == 1:
        df = pd.read_csv(str(TRAINING_CSV_BOTH))
        if data_type_norm == "heel":
            df = df[df["Source"] == "HEEL"].copy()
        elif data_type_norm == "cord":
            df = df[df["Source"] == "CORD"].copy()
        else:
            raise ValueError("data_type must be 'heel' or 'cord' when data_option == 1.")
    elif data_option == 2:
        df = pd.read_csv(str(TRAINING_CSV_ALL))
        df = df[df["Source"] == "HEEL"].copy()
    elif data_option == 3:
        df = pd.read_csv(str(TRAINING_CSV_ALL))
        df = df[df["Source"] == "CORD"].copy()
    else:
        raise ValueError("data_option must be 1, 2, or 3.")

    keep_cols = [c for c in CLINICAL_FEATURES + BIOMARKER_FEATURES + [TARGET_COL] if c in df.columns]
    df = df[keep_cols].copy()
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found.")

    df = df.dropna(subset=[TARGET_COL])
    y_cont = df[TARGET_COL]
    y = (y_cont < PRETERM_CUTOFF).astype(int)

    if model_type == "clinical":
        X = df[[c for c in CLINICAL_FEATURES if c in df.columns]].copy()
    elif model_type == "biomarker":
        X = df[[c for c in BIOMARKER_FEATURES if c in df.columns]].copy()
    elif model_type == "combined":
        X = df[[c for c in CLINICAL_FEATURES + BIOMARKER_FEATURES if c in df.columns]].copy()
    else:
        raise ValueError("model_type must be 'clinical', 'biomarker', or 'combined'.")

    feature_names = list(X.columns)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"n_rows": X.shape[0], "n_cols": X.shape[1]}]).to_csv(
        OUTPUT_DIR / "raw_dataset_shape_and_missingness.csv", index=False
    )
    pd.DataFrame({"feature": feature_names}).to_csv(OUTPUT_DIR / "raw_feature_names.csv", index=False)

    run_dir_name = (
        f"{model_name}__opt{data_option}__{model_type}__{data_type}"
        if data_option == 1
        else f"{model_name}__opt{data_option}__{model_type}"
    )
    run_out_dir = OUTPUT_DIR / run_dir_name
    run_out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": feature_names}).to_csv(run_out_dir / "feature_names.csv", index=False)

    predictions_all, auc_all, hyperparameters_all, coefficients_all = [], [], [], []

    for i in range(run_number):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED + i, stratify=y,
        )

        model, _ = get_classification_model(model_name)
        model.fit(X_train, y_train)

        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )
        auc = roc_auc_score(y_test, y_proba)

        predictions_all.append(pd.DataFrame({"run": i, "y_true": y_test, "y_pred": y_proba}))
        auc_all.append(pd.DataFrame({"run": i, "auc": auc}, index=[i]))

        best_params = model.best_params_ if hasattr(model, "best_params_") else {}
        hyperparameters_all.append(pd.DataFrame({
            "run": i,
            "C": best_params.get("logisticregression__C", None),
            "l1_ratio": best_params.get("logisticregression__l1_ratio", None),
        }, index=[i]))

        coef_flat = _extract_coef(model, step_name="logisticregression")
        coefficients_all.append(pd.DataFrame({
            "run": i,
            "coefficient names": [col for col, coef in zip(feature_names, coef_flat) if coef != 0],
            "coefficient values": [coef for coef in coef_flat if coef != 0],
        }))

    pd.concat(predictions_all).to_csv(run_out_dir / "predictions_all.csv", index=False)
    pd.concat(auc_all).to_csv(run_out_dir / "auc_all.csv", index=False)
    pd.concat(hyperparameters_all).to_csv(run_out_dir / "hyperparameters_all.csv", index=False)
    pd.concat(coefficients_all).to_csv(run_out_dir / "coefficients_all.csv", index=False)

    auc_df = pd.read_csv(run_out_dir / "auc_all.csv")
    auc_mean, auc_std = auc_df["auc"].mean(), auc_df["auc"].std()
    n = len(auc_df)
    print(f"AUC: mean={auc_mean:.4f}, std={auc_std:.4f}")
    print(
        f"95% CI: [{auc_mean - 1.96 * auc_std / np.sqrt(n):.4f}, "
        f"{auc_mean + 1.96 * auc_std / np.sqrt(n):.4f}]"
    )

    print(f"Results saved to {run_out_dir}/")


def main():
    """Parse arguments and dispatch to training or external validation."""
    parser = argparse.ArgumentParser(
        description="Gestational age classification: training cohort / external validation cohort"
    )
    parser.add_argument("--validate", action="store_true", help="External validation on held-out cohort data")
    parser.add_argument("--data_option", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--model_type", default="clinical", choices=["clinical", "biomarker", "combined"])
    parser.add_argument("--model_name", default="elasticnet_cv", choices=["elasticnet_cv", "lasso_cv"])
    parser.add_argument("--data_type", default="heel", choices=["heel", "cord"])
    parser.add_argument("--run_number", type=int, default=N_REPEATS)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all model_type x model_name x data_type combinations",
    )
    args = parser.parse_args()

    if args.validate:
        configs = (
            [
                (mt, mn, dt)
                for mt in ["clinical", "biomarker", "combined"]
                for mn in ["elasticnet_cv", "lasso_cv"]
                for dt in ["heel", "cord"]
            ]
            if args.all
            else [(args.model_type, args.model_name, args.data_type)]
        )
        for model_type, model_name, data_type in configs:
            run_validation(args.data_option, model_type, model_name, data_type, args.run_number)
        return

    if args.all:
        data_types = (
            ["heel", "cord"] if args.data_option == 1
            else ["heel"] if args.data_option == 2
            else ["cord"]
        )
        for model_type in ["clinical", "biomarker", "combined"]:
            for model_name in ["elasticnet_cv", "lasso_cv"]:
                for data_type in data_types:
                    run_classification_model(
                        args.data_option, model_type, model_name, data_type, args.run_number
                    )
        return

    run_classification_model(
        args.data_option, args.model_type, args.model_name, args.data_type, args.run_number
    )


if __name__ == "__main__":
    main()
