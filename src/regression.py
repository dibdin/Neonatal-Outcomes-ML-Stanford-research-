"""Gestational age regression: continuous prediction of gestational age (weeks).

Training (all option-1 configurations):
    python src/regression.py --all --data_option 1

External validation on held-out cohort:
    python src/regression.py --all-validate --data_option 1

Single configuration:
    python src/regression.py --data_option 1 --model_type biomarker --model_name lasso_cv --data_type heel
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import RANDOM_SEED, N_REPEATS, TRAINING_CSV_BOTH, TRAINING_CSV_ALL
from src.model import get_model
from src.feature_sets import (
    CLINICAL_FEATURES,
    BIOMARKER_FEATURES,
    TARGET_COL,
    load_training_frame,
    load_validation_frame,
    align_to_training_features,
)

OUTPUT_DIR = Path("outputs/regression")


def _run_dir(model_name: str, data_option: int, model_type: str, data_type: str) -> Path:
    """Return the output subdirectory for a given training configuration."""
    if data_option == 1:
        return OUTPUT_DIR / f"{model_name}__opt{data_option}__{model_type}__{data_type}"
    return OUTPUT_DIR / f"{model_name}__opt{data_option}__{model_type}"


def _extract_coef(model, model_name: str) -> list:
    """Extract the coefficient array from a fitted GridSearchCV regression pipeline."""
    coef_flat = []
    step_name = "elasticnet" if model_name == "elasticnet_cv" else "lasso"
    if hasattr(model, "best_estimator_"):
        est = model.best_estimator_
        if hasattr(est, "named_steps"):
            step = est.named_steps.get(step_name)
            if step is not None and hasattr(step, "coef_"):
                c = step.coef_
                coef_flat = c.flatten() if getattr(c, "ndim", 1) > 1 else np.asarray(c).ravel()
            # Fallback: last pipeline step that has coef_
            if len(coef_flat) == 0:
                for _, s in reversed(list(est.named_steps.items())):
                    if s is not None and hasattr(s, "coef_"):
                        c = s.coef_
                        coef_flat = c.flatten() if getattr(c, "ndim", 1) > 1 else np.asarray(c).ravel()
                        break
    return coef_flat


def run_validation(
    data_option: int = 1,
    model_type: str = "clinical",
    model_name: str = "elasticnet_cv",
    data_type: str = "heel",
    run_number: int = N_REPEATS,
) -> None:
    """External validation: models trained on training cohort splits, evaluated on external validation cohort."""
    X_train_full, y_train_full, feature_names = load_training_frame(
        data_option, data_type, model_type
    )
    X_val, _, y_cont_val, study_id, val_feature_names = load_validation_frame(
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

    metrics_rows, predictions_all, hyperparameters_all, coefficients_all = [], [], [], []

    for i in range(run_number):
        X_train, _, y_train, _ = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_SEED + i,
        )

        model, _ = get_model(model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        mae = mean_absolute_error(y_cont_val, y_pred)
        mse = mean_squared_error(y_cont_val, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_cont_val, y_pred)
        metrics_rows.append({"run": i, "mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "n_val": len(y_cont_val)})

        predictions_all.append(pd.DataFrame({
            "run": i,
            "study_id": study_id.values,
            "y_true": y_cont_val.values,
            "y_pred": y_pred,
        }))

        best_params = model.best_params_ if hasattr(model, "best_params_") else {}
        hyperparameters_all.append(pd.DataFrame({
            "run": i,
            "alpha": (
                best_params.get("elasticnet__alpha") if model_name == "elasticnet_cv"
                else best_params.get("lasso__alpha")
            ),
            "l1_ratio": best_params.get("elasticnet__l1_ratio") if model_name == "elasticnet_cv" else None,
        }, index=[i]))

        coef_flat = _extract_coef(model, model_name)
        coefficients_all.append(pd.DataFrame({
            "run": i,
            "coefficient names": [col for col, coef in zip(feature_names, coef_flat) if coef != 0],
            "coefficient values": [coef for coef in coef_flat if coef != 0],
        }))

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(val_out_dir / "metrics_all.csv", index=False)
    pd.concat(predictions_all, ignore_index=True).to_csv(val_out_dir / "predictions_all.csv", index=False)
    if hyperparameters_all:
        pd.concat(hyperparameters_all, ignore_index=True).to_csv(
            val_out_dir / "hyperparameters_all.csv", index=False
        )
    if coefficients_all:
        pd.concat(coefficients_all, ignore_index=True).to_csv(
            val_out_dir / "coefficients_all.csv", index=False
        )

    n = max(1, len(metrics_df))
    mae_mean, mae_std = metrics_df["mae"].mean(), metrics_df["mae"].std()
    rmse_series = metrics_df["rmse"] if "rmse" in metrics_df.columns else np.sqrt(metrics_df["mse"].astype(float))
    rmse_mean = float(rmse_series.mean())
    rmse_std = float(rmse_series.std(ddof=0)) if len(rmse_series) > 1 else 0.0

    pd.DataFrame({
        "mae_mean": [mae_mean], "mae_std": [mae_std],
        "mae_ci_lower": [mae_mean - 1.96 * mae_std / np.sqrt(n)],
        "mae_ci_upper": [mae_mean + 1.96 * mae_std / np.sqrt(n)],
        "rmse_mean": [rmse_mean], "rmse_std": [rmse_std],
        "rmse_ci_lower": [rmse_mean - 1.96 * rmse_std / np.sqrt(n)],
        "rmse_ci_upper": [rmse_mean + 1.96 * rmse_std / np.sqrt(n)],
    }).to_csv(val_out_dir / "mean_metrics_all.csv", index=False)

    print(
        f"Validation: MAE={mae_mean:.4f}, RMSE={rmse_mean:.4f}, "
        f"R2={metrics_df['r2'].mean():.4f}, n={len(y_cont_val)}"
    )
    print(f"Results saved to {val_out_dir}/")


def run_regression_model(
    data_option: int = 1,
    model_type: str = "clinical",
    model_name: str = "elasticnet_cv",
    data_type: str = "heel",
    run_number: int = N_REPEATS,
) -> None:
    """Train a regularised regression model with repeated train/test splits."""
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
    y = df[TARGET_COL]

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

    predictions_all, metrics_all, hyperparameters_all, coefficients_all = [], [], [], []

    for i in range(run_number):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED + i,
        )

        model, _ = get_model(model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, y_pred)

        predictions_all.append(pd.DataFrame({"run": i, "y_true": y_test, "y_pred": y_pred}))
        metrics_all.append(pd.DataFrame({"run": i, "mae": mae, "mse": mse, "rmse": rmse, "r2": r2}, index=[i]))

        best_params = model.best_params_ if hasattr(model, "best_params_") else {}
        hyperparameters_all.append(pd.DataFrame({
            "run": i,
            "alpha": (
                best_params.get("elasticnet__alpha") if model_name == "elasticnet_cv"
                else best_params.get("lasso__alpha")
            ),
            "l1_ratio": best_params.get("elasticnet__l1_ratio") if model_name == "elasticnet_cv" else None,
        }, index=[i]))

        coef_flat = _extract_coef(model, model_name)
        coefficients_all.append(pd.DataFrame({
            "run": i,
            "coefficient names": [col for col, coef in zip(feature_names, coef_flat) if coef != 0],
            "coefficient values": [coef for coef in coef_flat if coef != 0],
        }))

    pd.concat(predictions_all, ignore_index=True).to_csv(run_out_dir / "predictions_all.csv", index=False)
    metrics_df = pd.concat(metrics_all, ignore_index=True)
    metrics_df.to_csv(run_out_dir / "metrics_all.csv", index=False)
    pd.concat(hyperparameters_all, ignore_index=True).to_csv(run_out_dir / "hyperparameters_all.csv", index=False)
    pd.concat(coefficients_all, ignore_index=True).to_csv(run_out_dir / "coefficients_all.csv", index=False)

    n = len(metrics_df)
    mae_mean, mae_std = metrics_df["mae"].mean(), metrics_df["mae"].std()
    mse_mean, mse_std = metrics_df["mse"].mean(), metrics_df["mse"].std()
    rmse_series = metrics_df["rmse"] if "rmse" in metrics_df.columns else np.sqrt(metrics_df["mse"].astype(float))
    rmse_mean = float(rmse_series.mean())
    rmse_std = float(rmse_series.std(ddof=0)) if n > 1 else 0.0
    r2_mean, r2_std = metrics_df["r2"].mean(), metrics_df["r2"].std()

    print(f"MAE: mean={mae_mean:.4f}, std={mae_std:.4f}, 95% CI [{mae_mean - 1.96*mae_std/np.sqrt(n):.4f}, {mae_mean + 1.96*mae_std/np.sqrt(n):.4f}]")
    print(f"RMSE: mean={rmse_mean:.4f}, std={rmse_std:.4f}")
    print(f"R2: mean={r2_mean:.4f}, std={r2_std:.4f}")

    pd.DataFrame({
        "mae_mean": mae_mean, "mae_std": mae_std,
        "mae_ci_lower": mae_mean - 1.96 * mae_std / np.sqrt(n),
        "mae_ci_upper": mae_mean + 1.96 * mae_std / np.sqrt(n),
        "mse_mean": mse_mean, "mse_std": mse_std,
        "mse_ci_lower": mse_mean - 1.96 * mse_std / np.sqrt(n),
        "mse_ci_upper": mse_mean + 1.96 * mse_std / np.sqrt(n),
        "rmse_mean": rmse_mean, "rmse_std": rmse_std,
        "rmse_ci_lower": rmse_mean - 1.96 * rmse_std / np.sqrt(n),
        "rmse_ci_upper": rmse_mean + 1.96 * rmse_std / np.sqrt(n),
        "r2_mean": r2_mean, "r2_std": r2_std,
        "r2_ci_lower": r2_mean - 1.96 * r2_std / np.sqrt(n),
        "r2_ci_upper": r2_mean + 1.96 * r2_std / np.sqrt(n),
    }, index=[0]).to_csv(run_out_dir / "mean_metrics_all.csv", index=False)

    print(f"Results saved to {run_out_dir}/")


def main():
    """Parse arguments and dispatch to training or external validation."""
    parser = argparse.ArgumentParser(
        description="Gestational age regression: training cohort / external validation cohort"
    )
    parser.add_argument("--validate", action="store_true", help="External validation on held-out cohort data")
    parser.add_argument("--data_option", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--model_type", default="clinical", choices=["clinical", "biomarker", "combined"])
    parser.add_argument("--model_name", default="elasticnet_cv", choices=["elasticnet_cv", "lasso_cv"])
    parser.add_argument("--data_type", default="heel", choices=["heel", "cord"])
    parser.add_argument("--run_number", type=int, default=N_REPEATS)
    parser.add_argument("--all", action="store_true", help="Run all training configurations")
    parser.add_argument(
        "--all-validate",
        action="store_true",
        help="Validate all model_type x model_name x data_type combinations",
    )
    args = parser.parse_args()

    if args.validate or args.all_validate:
        configs = (
            [
                (mt, mn, dt)
                for mn in ["elasticnet_cv", "lasso_cv"]
                for mt in ["clinical", "biomarker", "combined"]
                for dt in ["heel", "cord"]
            ]
            if args.all_validate
            else [(args.model_type, args.model_name, args.data_type)]
        )
        for model_type, model_name, data_type in configs:
            run_validation(args.data_option, model_type, model_name, data_type, args.run_number)
        return

    if args.all:
        for model_type in ["clinical", "biomarker", "combined"]:
            for model_name in ["lasso_cv", "elasticnet_cv"]:
                for data_type in ["heel", "cord"]:
                    run_regression_model(1, model_type, model_name, data_type)
        return

    run_regression_model(
        args.data_option, args.model_type, args.model_name, args.data_type, args.run_number
    )


if __name__ == "__main__":
    main()
