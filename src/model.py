"""Model definitions for the gestational age prediction pipeline.

Provides pipeline factories for regularised regression (Elastic Net)
and regularised logistic regression (Elastic Net penalty) via GridSearchCV.
Each pipeline applies median imputation and standard scaling before fitting.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
import numpy as np

from .config import RANDOM_SEED, USE_CLASS_BALANCING

_REGRESSION_ALPHA_GRID = np.logspace(-4, 1, 50)
_REGRESSION_L1_RATIO_GRID = [0, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

_CLASSIFICATION_C_GRID = np.logspace(-4, 4, 10)
_CLASSIFICATION_L1_RATIO_GRID = [0, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]


def get_model(model_type: str):
    """Return a GridSearchCV regression pipeline for the specified model type.

    Parameters
    ----------
    model_type : {'elasticnet_cv'}

    Returns
    -------
    tuple
        (GridSearchCV estimator, None)
    """
    if model_type == "elasticnet_cv":
        param_grid = {
            "elasticnet__alpha": _REGRESSION_ALPHA_GRID,
            "elasticnet__l1_ratio": _REGRESSION_L1_RATIO_GRID,
        }
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("elasticnet", ElasticNet(
                max_iter=1000, fit_intercept=True, tol=1e-4, random_state=RANDOM_SEED
            )),
        ]
        pipeline = Pipeline(steps)
        cv_model = GridSearchCV(
            pipeline,
            param_grid,
            cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
            scoring="neg_mean_squared_error",
            n_jobs=1,
            verbose=0,
        )
        return cv_model, None

    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. Supported: 'elasticnet_cv'."
        )


def get_classification_model(model_type: str):
    """Return a GridSearchCV classification pipeline for the specified model type.

    Parameters
    ----------
    model_type : {'elasticnet_cv'}
        'elasticnet_cv' uses an elastic-net penalty.

    Returns
    -------
    tuple
        (GridSearchCV estimator, None)
    """
    class_weight = "balanced" if USE_CLASS_BALANCING else None

    if model_type == "elasticnet_cv":
        param_grid = {
            "logisticregression__C": _CLASSIFICATION_C_GRID,
            "logisticregression__l1_ratio": _CLASSIFICATION_L1_RATIO_GRID,
        }
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logisticregression", LogisticRegression(
                penalty="elasticnet", solver="saga",
                max_iter=1000, fit_intercept=True, tol=1e-4,
                class_weight=class_weight, random_state=RANDOM_SEED,
            )),
        ])
        cv_model = GridSearchCV(
            pipeline,
            param_grid,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
            scoring="roc_auc",
            n_jobs=1,
            verbose=0,
        )
        return cv_model, None

    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. Supported: 'elasticnet_cv'."
        )
