import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def metric(pred, true):
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(f'prediction shape {pred.shape} != target shape {true.shape}')
    return MAE(pred, true), MSE(pred, true), RMSE(pred, true)
