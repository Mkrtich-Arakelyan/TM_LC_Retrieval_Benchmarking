"""Numerical prediction and row-phase-aligned matrix metrics."""

import numpy as np

def mean_baseline(readings, fit_indices, held_indices, prediction, eligible, clipped=None):
    """Use fit data only; compare predictors on one identical held-out mask."""
    known_clipping = clipped is not None
    clipped = np.zeros_like(readings, bool) if clipped is None else np.asarray(clipped, bool)
    fit, held = readings[fit_indices], readings[held_indices]
    fit_ok, held_ok = ~clipped[fit_indices], ~clipped[held_indices]
    if prediction.shape != held.shape or np.shape(eligible) != (held.shape[1],):
        raise ValueError("Saved predictions/eligibility do not match recorded detector readings")
    means = np.array([np.mean(fit[fit_ok[:, j], j]) if fit_ok[:, j].any() else np.nan for j in range(held.shape[1])])
    baseline = np.broadcast_to(means, held.shape)
    common = held_ok & np.asarray(eligible, bool)[None, :] & np.isfinite(prediction) & np.isfinite(baseline)
    def scores(mask):
        if not mask.any():
            return {"fit_rmse": None, "mean_baseline_rmse": None, "fit_nmse": None, "mean_baseline_nmse": None, "result": "not scored"}
        squared = [(value[mask] - held[mask])**2 for value in (prediction, baseline)]
        mse = [float(np.mean(value)) for value in squared]
        denominator = max(float(np.mean(held[mask]**2)), 1e-12)
        return {"fit_rmse": float(np.sqrt(mse[0])), "mean_baseline_rmse": float(np.sqrt(mse[1])),
                "fit_nmse": mse[0]/denominator, "mean_baseline_nmse": mse[1]/denominator,
                "result": "tie" if mse[0] == mse[1] else "fit better" if mse[0] < mse[1] else "mean baseline better"}
    rows = []
    for j in range(held.shape[1]):
        mask = np.zeros_like(common)
        mask[:, j] = common[:, j]
        rows.append({"detector": j, "fitting_mean": float(means[j]) if np.isfinite(means[j]) else None,
            "eligible_fit": int(fit_ok[:, j].sum()), "heldout_usable": int(mask.sum()),
            "heldout_clipped": int((~held_ok[:, j]).sum()) if known_clipping else None, **scores(mask)})
    return {**scores(common), "usable": int(common.sum()), "total": int(common.size),
        "clipped": int((~held_ok).sum()) if known_clipping else None,
        "unscorable_unclipped": int((held_ok & ~common).sum()) if known_clipping else None, "per_detector": rows,
        "selection": "Per-detector mean of eligible fitting readings only; both predictors scored on the same eligible held-out readings. No refitting or holdout selection."
            + (" Clipping flags were not recorded; eligibility uses the available saved data." if not known_clipping else "")}



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


