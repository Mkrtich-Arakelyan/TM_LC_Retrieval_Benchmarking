"""Registered numerical datasets and saved-prediction loading; no dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from workflow.config import AppConfig
from workflow.state import Artifact, TMFitSettings, WorkflowState
from tools.python_execution import checked_inputs, contained_file, fingerprint
from analysis.tm_solvers import _archive


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


def saved_predictions(state, request, config, dataset):
    """Load hashed outputs from one completed reviewed fit; never call a solver."""
    sources = [a for a in state.analyses if a.id in request.input_record_ids and a.status == "completed"
        and a.execution is not None and a.execution.backend == "reviewed_numpy"
        and a.method_settings.get("operation") == "tm.compare_reference_free"
        and any(ref.id == request.dataset_artifact_id for ref in a.input_data)]
    if len(sources) != 1:
        raise ValueError("Saved prediction evaluation requires exactly one matching completed reviewed comparison")
    source = sources[0]
    settings = {k: v for k, v in source.method_settings.items() if k != "operation"}
    if TMFitSettings.model_validate(settings) != request.parameters:
        raise ValueError("Evaluation settings must identify the original fit settings; no refitting/tuning is allowed")
    if request.ground_truth_artifact_id and not any(ref.id == request.ground_truth_artifact_id for ref in source.input_data):
        raise ValueError("Evaluation truth must match the original comparison's separate truth artifact")
    fitted, references = {}, []
    hold = dataset.manifest.holdout_indices
    for method in ("ggs", "intensity_ls"):
        outputs = [a for a in source.outputs if Path(a.path).name == f"{method}.npz"]
        if len(outputs) != 1 or outputs[0].sha256 is None:
            raise ValueError("Saved method output requires one registered artifact and content hash")
        ref = outputs[0]
        path = contained_file(ref.path, (config.run_output_path / "analysis").resolve())
        if fingerprint(path) != ref.sha256:
            raise ValueError("Saved prediction artifact hash mismatch")
        keys = {"T", "holdout_prediction", "holdout_squared_error", "holdout_indices"}
        if f"{method}.matrix_aligned_absolute_error" in source.metrics:
            keys.add("row_phase_aligned_errors")
        arrays = _archive(str(path), keys)
        matrix, prediction = arrays["T"], arrays["holdout_prediction"]
        if (matrix.shape != (dataset.intensities.shape[1], dataset.X.shape[1]) or matrix.dtype.kind != "c"
                or prediction.shape != dataset.intensities[hold].shape or prediction.dtype.kind not in "fiu"
                or not np.isfinite(matrix).all() or not np.isfinite(prediction).all()
                or not np.array_equal(arrays["holdout_indices"], hold)
                or not np.allclose(prediction, np.abs(dataset.X[hold] @ matrix.T) ** 2, rtol=1e-12, atol=1e-14)):
            raise ValueError("Saved predictions/matrix/holdout indices do not match the registered dataset")
        fitted[method] = {"T": matrix, "saved_prediction": prediction,
            "fit_mse": source.metrics[f"{method}.fit_mse"],
            "runtime_seconds": source.metrics[f"{method}.runtime_seconds"],
            "fit_reused": True, "source_analysis_id": str(source.id),
            "runtime_description": "Original fit runtime, retained from the source comparison; no refit performed"}
        references.append(ref)
    return fitted, references


