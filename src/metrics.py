from math import sqrt
from typing import Tuple

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix


def report(predictions: np.array, targets: np.array) -> Tuple[float, float, float, float]:
    rmse_value = rmse(predictions, targets)
    mard_value = mard(predictions, targets)
    mae_value = mae(predictions, targets)
    mape_value = mape(predictions, targets)
    return rmse_value, mard_value, mae_value, mape_value


def rmse(predictions: np.array, targets: np.array) -> float:
    """Root-mean-square error"""
    return sqrt(mean_squared_error(predictions, targets))


def mae(predictions: np.array, targets: np.array) -> float:
    """Mean absolute error"""
    return mean_absolute_error(predictions, targets)


def mape(predictions: np.array, targets: np.array) -> float:
    """Mean absolute percentage error"""
    # From https://stackoverflow.com/a/47648179/118173
    mask = predictions != 0
    return (np.fabs(predictions - targets) / predictions)[mask].mean()

def mard(predictions: np.array, targets: np.array) -> float:
    """Mean absolute relative difference"""
    # From GLUNET
    mask = targets != 0
    return (np.fabs(targets - predictions) / targets)[mask].mean()

def confusion_hypo(predictions: np.array, targets: np.array):
    tn, fp, fn, tp = confusion_matrix(targets < (3.9*18), predictions < (3.9*18)).ravel()

    # Precision: Amount of correct positive out all positive predictions
    precision = tp / (tp + fp)

    # How many hypos are caught
    recall = tp / (tp + fn)

    # Harmonic mean
    F1 = 2 * precision * recall / (precision + recall)

    

    return recall, precision, F1, tn, fp, fn, tp
