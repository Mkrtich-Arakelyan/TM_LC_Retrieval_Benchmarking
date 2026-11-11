"""Saved-data TM evaluation; called only by the reviewed operation interface."""

import json
import sys
import time

import numpy as np

from workflow.state import Artifact, ExecutionDetails, NumericalAnalysisResult
from tools.python_execution import fingerprint, validated_metrics
from analysis.datasets import _checked, load_dataset, saved_predictions
from analysis.metrics import aligned_matrix_errors, prediction_metrics
from analysis.tm_solvers import _archive


def evaluate(state, request, config, artifacts, acquisition_ids, common, started, directory, *, fit_methods,
             attempt_number):
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
    if request.operation in ("tm.compare_reference_free", "tm.evaluate_saved_predictions"):
        fit, hold = dataset.manifest.fit_indices, dataset.manifest.holdout_indices
        if request.operation == "tm.compare_reference_free":
            fitted = fit_methods(dataset.X[fit].copy(), dataset.intensities[fit].copy(), request.parameters,
                                 deadline=started + config.numerical_timeout_seconds)
        else:
            fitted, saved_refs = saved_predictions(state, request, config, dataset)
            common["input_data"].extend(saved_refs)
            report["selection"] = "Saved fit and holdout predictions reused unchanged; evaluation only, no fitting or selection"
        # Ground truth is loaded only AFTER all fitting and fit-only selection.
        truth = None
        if request.ground_truth_artifact_id:
            reference = _checked(artifacts[request.ground_truth_artifact_id], config)
            truth = _archive(reference.path, {"T"})["T"]
            if truth.shape != (dataset.intensities.shape[1], dataset.X.shape[1]) or truth.dtype.kind != "c" or not np.isfinite(truth).all():
                raise ValueError("Ground-truth T must be finite complex [M,D]")
        for method, result in fitted.items():
            matrix = result.pop("T")
            prediction = result.pop("saved_prediction", None)
            if prediction is None:
                prediction = np.abs(dataset.X[hold] @ matrix.T) ** 2
            errors = (prediction - dataset.intensities[hold]) ** 2
            for name, value in prediction_metrics(prediction, dataset.intensities[hold]).items():
                metrics[f"{method}.holdout_{name}"] = value
            metrics[f"{method}.fit_mse"] = result["fit_mse"]
            metrics[f"{method}.runtime_seconds"] = result["runtime_seconds"]
            arrays = {"T": matrix, "holdout_prediction": prediction, "holdout_squared_error": errors,
                      "holdout_indices": np.asarray(hold)}
            if truth is not None:
                noiseless = np.abs(dataset.X[hold] @ truth.T) ** 2
                for name, value in prediction_metrics(prediction, noiseless).items():
                    metrics[f"{method}.holdout_noiseless_{name}"] = value
                row, absolute, relative = aligned_matrix_errors(matrix, truth)
                arrays["row_phase_aligned_errors"] = row
                metrics[f"{method}.matrix_aligned_absolute_error"] = absolute
                if relative is not None:
                    metrics[f"{method}.matrix_aligned_relative_error"] = relative
            path = directory / f"{method}.npz"
            np.savez(path, **arrays)
            outputs.append(artifact(path, f"Estimated {method} matrix and final held-out predictions/errors"))
            report["methods"][method] = result
        for target in ("holdout", "holdout_noiseless"):
            for metric in ("mse", "rmse", "mae", "nmse"):
                key = f"{target}_{metric}"
                if f"ggs.{key}" in metrics and f"intensity_ls.{key}" in metrics:
                    metrics[f"intensity_ls_minus_ggs.{key}"] = metrics[f"intensity_ls.{key}"] - metrics[f"ggs.{key}"]
    metrics["total_runtime_seconds"] = time.monotonic() - started
    report["metrics"] = validated_metrics(metrics)
    path = directory / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    outputs.append(artifact(path, "Reviewed numerical operation settings, selection, provenance and metrics"))
    return NumericalAnalysisResult(**common, status="completed", metrics=validated_metrics(metrics), outputs=outputs,
        execution=ExecutionDetails(status="completed", backend="reviewed_numpy", attempt_number=attempt_number,
            exit_code=0, elapsed_seconds=time.monotonic() - started,
            stdout="Reviewed named NumPy operation executed; no model-generated Python executed"))
