from typing import Dict
import numpy as np
from sklearn import metrics


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Regression Report

    Parameters
    ----------
    y_true : np.ndarray
        _description_
    y_prednp : _type_
        _description_

    Returns
    -------
    Dict[str, float]
        Explained Variance Score, Max Error, Mean Absolute Error,
        Mean Squared Error, Median Absolute Error, R2 score,
        Mean Absolute Percentage Error
    """
    EVS = metrics.explained_variance_score(y_true, y_pred)
    ME = metrics.max_error(y_true, y_pred)
    MAE = metrics.mean_absolute_error(y_true, y_pred)
    MSE = metrics.mean_squared_error(y_true, y_pred)
    MedAE = metrics.median_absolute_error(y_true, y_pred)
    R2 = metrics.r2_score(y_true, y_pred)
    MAPE = metrics.mean_absolute_percentage_error(y_true, y_pred)
    out = {
        "EVS": EVS,
        "ME": ME,
        "MAE": MAE,
        "MSE": MSE,
        "MedAE": MedAE,
        "R2": R2,
        "MAPE": MAPE,
    }
    ANY_NEG = any(y_pred[y_pred < 0]) or any(y_true[y_true < 0])
    ANY_ZERO = any(y_pred[y_pred == 0]) or any(y_true[y_true == 0])
    if not ANY_NEG:
        MSLE = metrics.mean_squared_log_error(y_true, y_pred)
        out["MSLE"] = MSLE
        if not ANY_ZERO:
            MPD = metrics.mean_poisson_deviance(y_true, y_pred)
            out["MPD"] = MPD
            MGD = metrics.mean_gamma_deviance(y_true, y_pred)
            out["MGD"] = MGD
    return out
