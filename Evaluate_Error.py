import numpy as np
import math
from jiwer import wer


def evaluat_error(sp, act, acxtText=None, predText=None) -> object:
    r = np.squeeze(act)
    x = np.squeeze(sp)
    points = np.zeros(len(x))
    abs_r = np.zeros(len(x))
    abs_x = np.zeros(len(x))
    abs_r_x = np.zeros(len(x))
    abs_x_r = np.zeros(len(x))
    abs_r_x__r = np.zeros(len(x))
    for j in range(len(x)):
        points[j] = abs(x[j] - x[j - 1])
    for i in range(len(r)):
        abs_r[i] = abs(r[i])
    for i in range(len(r)):
        abs_x[i] = abs(x[i])
    for i in range(len(r)):
        abs_r_x[i] = abs(r[i] - x[i])
    for i in range(len(r)):
        abs_x_r[i] = abs(x[i] - r[i])
    for i in range(len(r)):
        abs_r_x__r[i] = abs((r[i] - x[i]) / r[i])
    md = (100 / len(x)) * sum(abs_r_x__r)
    smape = (1 / len(x)) * sum(abs_r_x / ((abs_r + abs_x) / 2))
    mase = sum(abs_r_x) / ((1 / (len(x) - 1)) * sum(points))
    mae = sum(abs_r_x) / len(r)
    rmse = (sum(abs_x_r ** 2) / len(r)) ** 0.5
    accuracy = (1 - (np.abs(sum(r - x)) / sum(r))) * 100
    EVAL_ERR = [md, smape, mase, mae, rmse, accuracy]
    return EVAL_ERR


def calculate_ser(y_true, y_pred):
    """
    Calculates the Standard Error of the Regression (SER).

    Args:
        y_true (array-like): The true values of the dependent variable.
        y_pred (array-like): The predicted values of the dependent variable.

    Returns:
        float: The SER value.
    """
    residuals = y_true - y_pred
    return np.sqrt(np.sum(residuals**2) / (len(y_true) - 2))


def evaluation_text(sp, act, acxtText=None, predText=None):
    A = np.squeeze(act)
    B = np.squeeze(sp)
    r = []
    x = []
    for i in range(len(A)):
        R = A[i]
        X = B[i]
        if R[0] == 0 or X[0] == 0:
            continue
        r.append(R)
        x.append(X)
    WER = wer(acxtText, predText)  # WER
    SER = calculate_ser(A, B)  # SER
    EVAL_ERR = [WER, SER]
    return EVAL_ERR