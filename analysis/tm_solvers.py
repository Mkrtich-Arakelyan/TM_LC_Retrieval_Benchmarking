"""Shared TM fitting and bounded NPZ loading; no operation or workflow dispatch."""

from __future__ import annotations

import time
import zipfile

import numpy as np

from workflow.state import TMFitSettings


def _archive(path: str, keys: set[str]) -> dict[str, np.ndarray]:
    # Bound uncompressed input before allocating arrays; never unpickle datasets.
    with zipfile.ZipFile(path) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > 128 * 1024 * 1024:
            raise ValueError("Numerical archive exceeds 128 MiB uncompressed budget")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != keys:
            raise ValueError(f"Expected exactly these NPZ arrays: {sorted(keys)}; ground truth must be separate")
        return {key: archive[key].copy() for key in keys}


def intensity_loss_gradient(X, I, W):
    """Real-coordinate complex gradient: dL = Re(sum(conj(grad)*dW))."""
    field = X @ W
    residual = np.abs(field) ** 2 - I
    return float(np.mean(residual ** 2)), 4.0 / I.size * X.conj().T @ (residual * field)


def fit_methods(X_fit: np.ndarray, I_fit: np.ndarray, settings: TMFitSettings, *, deadline=float("inf")) -> dict:
    """Fit-only API. Holdout and ground truth cannot influence restart/stopping selection.

    Returns T matrices [M,D]. Both methods receive identical random-phase LS starts
    and choose their best restart/iterate using fit intensity MSE only.
    """
    if not np.isfinite(X_fit).all() or not np.isfinite(I_fit).all():
        raise ValueError("Fit values must be finite")
    initialization_started = time.perf_counter()
    scale = float(np.mean(I_fit))
    target = I_fit / scale if scale > 0 else I_fit.copy()
    inverse = np.linalg.pinv(X_fit)
    rng = np.random.default_rng(settings.seed)
    starts = [inverse @ (np.sqrt(target) * np.exp(1j * rng.uniform(-np.pi, np.pi, target.shape)))
              for _ in range(settings.restarts)]
    initialization_seconds = time.perf_counter() - initialization_started
    results = {}
    for method in ("ggs", "intensity_ls"):
        started = time.perf_counter()
        best = None
        histories = []
        stopping_reasons = []
        for restart, initial in enumerate(starts):
            W = initial.copy()
            loss, _ = intensity_loss_gradient(X_fit, target, W)
            history = [loss * scale ** 2]
            if best is None or loss < best[0]:
                best = (loss, W.copy(), restart, 0)
            stopping_reason = "iteration_budget"
            for iteration in range(1, settings.iterations + 1):
                if time.monotonic() > deadline:
                    raise TimeoutError("Reviewed numerical operation exceeded its cooperative time budget")
                if method == "ggs":
                    field = X_fit @ W
                    candidate = inverse @ (np.sqrt(target) * np.exp(1j * np.angle(field)))
                else:
                    _, gradient = intensity_loss_gradient(X_fit, target, W)
                    step = settings.step_size
                    candidate = W
                    for _ in range(30):
                        trial = W - step * gradient
                        trial_loss, _ = intensity_loss_gradient(X_fit, target, trial)
                        if np.isfinite(trial_loss) and trial_loss <= loss:
                            candidate = trial
                            break
                        step *= 0.5
                next_loss, _ = intensity_loss_gradient(X_fit, target, candidate)
                if not np.isfinite(next_loss):
                    raise ValueError("Non-finite fitting objective")
                history.append(next_loss * scale ** 2)
                if next_loss < best[0]:
                    best = (next_loss, candidate.copy(), restart, iteration)
                change = abs(loss - next_loss)
                W, loss = candidate, next_loss
                if change <= settings.tolerance * max(history[-2] / max(scale ** 2, 1e-300), 1e-15):
                    stopping_reason = "fit_objective_tolerance_or_no_descent"
                    break
            histories.append(history)
            stopping_reasons.append(stopping_reason)
        results[method] = {"T": best[1].T * np.sqrt(scale), "fit_mse": best[0] * scale ** 2,
            "selected_restart": best[2], "selected_iteration": best[3], "fit_histories": histories,
            "stopping_reasons": stopping_reasons, "fit_intensity_scale": scale,
            "shared_initialization_seconds": initialization_seconds,
            "runtime_seconds": time.perf_counter() - started}
    return results
