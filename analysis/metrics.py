"""Numerical prediction and row-phase-aligned matrix metrics."""

import numpy as np


def aligned_matrix_errors(estimate, truth):
    correlation = np.sum(estimate.conj() * truth, axis=1)
    aligned = estimate * np.exp(1j * np.angle(correlation))[:, None]
    row_errors = np.linalg.norm(aligned - truth, axis=1)
    absolute = float(np.linalg.norm(aligned - truth))
    norm = float(np.linalg.norm(truth))
    return row_errors, absolute, absolute / norm if norm > 0 else None


def prediction_metrics(prediction, target):
    residual = prediction - target
    mse = float(np.mean(residual ** 2))
    values = {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(residual)))}
    denominator = float(np.sum(target ** 2))
    if denominator > 0:
        values["nmse"] = float(np.sum(residual ** 2) / denominator)
    return values


