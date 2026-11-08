"""Reviewed, reference-free offline numerical operations. Never execute supplied code.

Rows of X are known input fields; columns of I are output channels:
I = abs(X @ T.T)**2. GS alternates amplitude replacement with least-squares
projection onto range(X). The baseline minimizes mean squared intensity residual.
"""

from __future__ import annotations

import json
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from config import AppConfig
from state import (AnalysisRequest, Artifact, ExecutionDetails, NumericalAnalysisResult,
                   TMFitSettings, WorkflowState)
from tools.python_execution import checked_inputs, fingerprint, validated_metrics

OPERATIONS = ("tm.validate_dataset", "tm.compare_reference_free")
ARRAY_KEYS = {"X", "intensities", "pattern_ids", "measurement_pattern_ids", "frame_ids"}


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1]
    arrays: Artifact
    data_origin: Literal["physical", "simulated", "test_only"]
    provenance: str = Field(min_length=1)
    fit_indices: list[StrictInt] = Field(min_length=1)
    holdout_indices: list[StrictInt] = Field(min_length=1)

    @model_validator(mode="after")
    def separate_splits(self):
        fit, holdout = set(self.fit_indices), set(self.holdout_indices)
        if len(fit) != len(self.fit_indices) or len(holdout) != len(self.holdout_indices) or fit & holdout:
            raise ValueError("Fit and holdout indices must be unique and disjoint")
        if not self.provenance.strip():
            raise ValueError("Dataset provenance description is required")
        return self


@dataclass
class Dataset:
    manifest: DatasetManifest
    X: np.ndarray
    intensities: np.ndarray


def registered_artifacts(state: WorkflowState, record_ids: list[UUID]) -> tuple[dict[UUID, Artifact], list[UUID]]:
    records = {r.id: r for group in (state.acquisitions, state.analyses) for r in group}
    found = {}
    acquisitions = set()
    def visit(value):
        if isinstance(value, dict):
            if {"id", "path", "description"} <= value.keys():
                artifact = Artifact.model_validate(value)
                if artifact.id in found and found[artifact.id] != artifact:
                    raise ValueError("Artifact ID has conflicting metadata")
                found[artifact.id] = artifact
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for key in record_ids:
        if key not in records:
            raise ValueError("Numerical inputs must reference existing acquisition/analysis records")
        record = records[key]
        if hasattr(record, "acquisition_ids"):
            if record.status != "completed":
                raise ValueError("Input analysis must be completed")
            acquisitions.update(record.acquisition_ids)
        else:
            acquisitions.add(record.id)
        visit(record.model_dump(mode="json"))
    for key in acquisitions:
        acquisition = next((a for a in state.acquisitions if a.id == key), None)
        if acquisition is None or (acquisition.status != "completed" and not (
            state.control.allow_partial_analysis and acquisition.status in ("partial", "failed", "aborted") and acquisition.raw_data)):
            raise ValueError("Ineligible upstream acquisition; partial data requires explicit authorization")
    return found, sorted(acquisitions, key=str)


def _checked(artifact: Artifact, config: AppConfig) -> Artifact:
    if artifact.sha256 is None:
        raise ValueError("Numerical input requires a recorded SHA-256")
    return checked_inputs([artifact], config)[0]


def _archive(path: str, keys: set[str]) -> dict[str, np.ndarray]:
    # Bound uncompressed input before allocating arrays; never unpickle datasets.
    with zipfile.ZipFile(path) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > 128 * 1024 * 1024:
            raise ValueError("Numerical archive exceeds 128 MiB uncompressed budget")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != keys:
            raise ValueError(f"Expected exactly these NPZ arrays: {sorted(keys)}; ground truth must be separate")
        return {key: archive[key].copy() for key in keys}


def load_dataset(manifest_artifact: Artifact, artifacts: dict[UUID, Artifact], config: AppConfig) -> Dataset:
    manifest_ref = _checked(manifest_artifact, config)
    if Path(manifest_ref.path).stat().st_size > 1024 * 1024:
        raise ValueError("Dataset manifest exceeds 1 MiB")
    manifest = DatasetManifest.model_validate_json(Path(manifest_ref.path).read_text(encoding="utf-8"))
    if artifacts.get(manifest.arrays.id) != manifest.arrays:
        raise ValueError("Dataset arrays must match an existing artifact ID and metadata")
    arrays_ref = _checked(manifest.arrays, config)
    arrays = _archive(arrays_ref.path, ARRAY_KEYS)
    X, I = arrays["X"], arrays["intensities"]
    if X.ndim != 2 or I.ndim != 2 or X.shape[0] != I.shape[0] or min(*X.shape, *I.shape) < 1:
        raise ValueError("X[N,D] and intensities[N,M] must have matching, nonempty pattern dimensions")
    if X.dtype.kind != "c" or I.dtype.kind not in "fiu":
        raise ValueError("X must be complex and intensities must be real numeric values")
    if X.size + I.size > 2_000_000 or X.shape[1] > 128 or I.shape[1] > 256:
        raise ValueError("Dataset exceeds reviewed small-analysis resource budget")
    if not np.isfinite(X).all() or not np.isfinite(I).all() or (I < 0).any():
        raise ValueError("Dataset values must be finite and intensities nonnegative")
    n = len(X)
    ids = []
    for key in ("pattern_ids", "measurement_pattern_ids", "frame_ids"):
        value = arrays[key]
        if value.shape != (n,) or value.dtype.kind not in "US" or any(not str(v).strip() for v in value):
            raise ValueError(f"{key} must contain one nonempty string ID per pattern/frame")
        value = value.astype(str)
        if len(set(value)) != n:
            raise ValueError(f"Duplicate IDs in {key}; split data cannot overlap")
        ids.append(value)
    if not np.array_equal(ids[0], ids[1]):
        raise ValueError("Pattern-to-frame pairing disagrees: pattern IDs do not match measurement pattern IDs")
    fit, holdout = manifest.fit_indices, manifest.holdout_indices
    if any(i < 0 or i >= n for i in fit + holdout) or set(fit + holdout) != set(range(n)):
        raise ValueError("Splits must partition all rows with valid indices")
    # Distinct labels must not conceal identical patterns across the split.
    fit_patterns = {np.asarray(X[i], dtype=np.complex128).tobytes() for i in fit}
    if any(np.asarray(X[i], dtype=np.complex128).tobytes() in fit_patterns for i in holdout):
        raise ValueError("Identical input patterns overlap fit and holdout")
    if np.linalg.matrix_rank(X[fit]) < X.shape[1]:
        raise ValueError("Fit input matrix is rank deficient; additional independent fit patterns are required")
    return Dataset(manifest, X.astype(np.complex128), I.astype(np.float64))


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


def aligned_matrix_errors(estimate, truth):
    correlation = np.sum(estimate.conj() * truth, axis=1)
    aligned = estimate * np.exp(1j * np.angle(correlation))[:, None]
    row_errors = np.linalg.norm(aligned - truth, axis=1)
    absolute = float(np.linalg.norm(aligned - truth))
    norm = float(np.linalg.norm(truth))
    return row_errors, absolute, absolute / norm if norm > 0 else None


def execute_operation(state: WorkflowState, request: AnalysisRequest, config: AppConfig,
                      *, attempt_number=1, run_id: UUID | None = None) -> NumericalAnalysisResult:
    """Dispatch only an enumerated reviewed function; no imports/calls from model strings."""
    if request.operation not in OPERATIONS:
        raise ValueError("An explicit supported numerical operation is required")
    request = AnalysisRequest.model_validate(request.model_dump())
    config = AppConfig.model_validate(config.model_dump())
    artifacts, acquisition_ids = registered_artifacts(state, request.input_record_ids)
    requested_ids = [request.dataset_artifact_id] + ([request.ground_truth_artifact_id] if request.ground_truth_artifact_id else [])
    if any(key not in artifacts for key in requested_ids):
        raise ValueError("Named operation references an unregistered input artifact ID")
    if not 1 <= attempt_number <= config.max_workflow_analysis_attempts:
        raise ValueError("Numerical attempt budget exceeded")
    root = (config.run_output_path / "analysis").resolve()
    if root.is_relative_to(config.numerical_input_path.resolve()) or config.numerical_input_path.resolve().is_relative_to(root):
        raise ValueError("Numerical input and output roots must be disjoint")
    directory = root / str(run_id or uuid4())
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    code_path = directory / "reviewed_implementation.py"
    code_path.write_bytes(source)  # audit copy only; never imported/executed
    code = Artifact(path=str(code_path), description="Reviewed tm_numerics implementation audit copy", sha256=fingerprint(code_path))
    common = dict(acquisition_ids=acquisition_ids, input_data=[artifacts[k] for k in requested_ids],
                  code=code, code_version=code.sha256, method_settings={"operation": request.operation,
                  **request.parameters.model_dump()}, random_seed=request.parameters.seed)
    started = time.monotonic()
    if not config.reviewed_numerics_enabled:
        return NumericalAnalysisResult(**common, execution=ExecutionDetails(status="unavailable",
            backend="reviewed_numpy", attempt_number=attempt_number, reason="Enable TM_REVIEWED_NUMERICS_ENABLED to run reviewed named operations; arbitrary code remains disabled"))
    try:
        dataset = load_dataset(artifacts[request.dataset_artifact_id], artifacts, config)
        common["input_data"] = [artifacts[request.dataset_artifact_id], dataset.manifest.arrays] + (
            [artifacts[request.ground_truth_artifact_id]] if request.ground_truth_artifact_id else [])
        common["data_origin"] = dataset.manifest.data_origin
        for key in acquisition_ids:
            acquisition = next(a for a in state.acquisitions if a.id == key)
            origins = [acquisition.data_origin, acquisition.hardware.data_origin if acquisition.hardware else None]
            if any(origin and origin != dataset.manifest.data_origin for origin in origins):
                raise ValueError("Dataset provenance conflicts with the source acquisition record")
        metrics = {"fit_pattern_count": len(dataset.manifest.fit_indices), "holdout_pattern_count": len(dataset.manifest.holdout_indices)}
        report = {"operation": request.operation, "data_origin": dataset.manifest.data_origin,
            "environment": {"numpy": np.__version__, "python": sys.version},
            "provenance": dataset.manifest.provenance, "settings": request.parameters.model_dump(),
            "dataset": artifacts[request.dataset_artifact_id].model_dump(mode="json"),
            "splits": {"fit": dataset.manifest.fit_indices, "holdout": dataset.manifest.holdout_indices},
            "selection": "fit intensity MSE only; holdout/ground truth excluded from fitting", "methods": {}}
        outputs = []
        def artifact(path, description):
            return Artifact(path=str(path), description=description, sha256=fingerprint(path))
        if request.operation == "tm.compare_reference_free":
            fit, hold = dataset.manifest.fit_indices, dataset.manifest.holdout_indices
            fitted = fit_methods(dataset.X[fit].copy(), dataset.intensities[fit].copy(), request.parameters,
                                 deadline=started + config.numerical_timeout_seconds)
            # Ground truth is loaded only AFTER all fitting and fit-only selection.
            truth = None
            if request.ground_truth_artifact_id:
                reference = _checked(artifacts[request.ground_truth_artifact_id], config)
                truth = _archive(reference.path, {"T"})["T"]
                if truth.shape != (dataset.intensities.shape[1], dataset.X.shape[1]) or truth.dtype.kind != "c" or not np.isfinite(truth).all():
                    raise ValueError("Ground-truth T must be finite complex [M,D]")
            for method, result in fitted.items():
                matrix = result.pop("T")
                prediction = np.abs(dataset.X[hold] @ matrix.T) ** 2
                errors = (prediction - dataset.intensities[hold]) ** 2
                metrics[f"{method}.holdout_mse"] = float(np.mean(errors))
                denominator = float(np.sum(dataset.intensities[hold] ** 2))
                if denominator > 0:
                    metrics[f"{method}.holdout_nmse"] = float(np.sum(errors) / denominator)
                metrics[f"{method}.fit_mse"] = result["fit_mse"]
                metrics[f"{method}.runtime_seconds"] = result["runtime_seconds"]
                arrays = {"T": matrix, "holdout_prediction": prediction, "holdout_squared_error": errors,
                          "holdout_indices": np.asarray(hold)}
                if truth is not None:
                    row, absolute, relative = aligned_matrix_errors(matrix, truth)
                    arrays["row_phase_aligned_errors"] = row
                    metrics[f"{method}.matrix_aligned_absolute_error"] = absolute
                    if relative is not None:
                        metrics[f"{method}.matrix_aligned_relative_error"] = relative
                path = directory / f"{method}.npz"
                np.savez(path, **arrays)
                outputs.append(artifact(path, f"Estimated {method} matrix and final held-out predictions/errors"))
                report["methods"][method] = result
        metrics["total_runtime_seconds"] = time.monotonic() - started
        report["metrics"] = validated_metrics(metrics)
        path = directory / "report.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        outputs.append(artifact(path, "Reviewed numerical operation settings, selection, provenance and metrics"))
        return NumericalAnalysisResult(**common, status="completed", metrics=validated_metrics(metrics), outputs=outputs,
            execution=ExecutionDetails(status="completed", backend="reviewed_numpy", attempt_number=attempt_number,
                exit_code=0, elapsed_seconds=time.monotonic() - started,
                stdout="Reviewed named NumPy operation executed; no model-generated Python executed"))
    except Exception as exc:
        return NumericalAnalysisResult(**common, status="failed", errors=[str(exc) or type(exc).__name__],
            execution=ExecutionDetails(status="failed", backend="reviewed_numpy", attempt_number=attempt_number,
                reason=str(exc), elapsed_seconds=time.monotonic() - started))
