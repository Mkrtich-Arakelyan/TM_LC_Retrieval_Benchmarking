"""Reviewed intensity-only retrieval from saved bench measurements, without truth."""

import csv
import json
import time
from pathlib import Path

import numpy as np

from bench.config import BenchConfig
from bench.optical_bench import CAMERA_GAIN, detector_regions, pattern_to_camera
from bench.acquisition import extract_detectors, saved_artifact, write_json
from tools.python_execution import checked_inputs
from analysis.tm_solvers import _archive, fit_methods

BENCH_ARRAYS = {"phases", "frames", "pattern_ids", "detector_readings",
                "detector_saturation", "whole_frame_saturation"}


def load_bench_dataset(reference, artifacts, config):
    if reference.sha256 is None:
        raise ValueError("Bench manifest requires a recorded content hash")
    checked = checked_inputs([reference], config)[0]
    if Path(checked.path).stat().st_size > 1024 * 1024:
        raise ValueError("Bench manifest exceeds size budget")
    manifest = json.loads(Path(checked.path).read_text(encoding="utf-8"))
    if manifest.get("kind") != "synthetic-bench-v1" or manifest.get("data_origin") != "simulated":
        raise ValueError("Not a simulated bench manifest")
    from workflow.state import Artifact
    arrays_ref = Artifact.model_validate(manifest["arrays"])
    if artifacts.get(arrays_ref.id) != arrays_ref or arrays_ref.sha256 is None:
        raise ValueError("Bench arrays must be registered with matching identity and content hash")
    checked_arrays = checked_inputs([arrays_ref], config)[0]
    data = _archive(checked_arrays.path, BENCH_ARRAYS)
    cfg = BenchConfig.model_validate(manifest["config"])
    n = cfg.fit_patterns + cfg.holdout_patterns
    if manifest["fit_patterns"] != cfg.fit_patterns or manifest["holdout_patterns"] != cfg.holdout_patterns:
        raise ValueError("Bench split metadata conflicts with configured counts")
    if manifest["regions"] != [list(r) for r in detector_regions(cfg.frame_side)]:
        raise ValueError("Detector extraction geometry disagrees with the model contract")
    for field, expected in (("camera_gain", CAMERA_GAIN), ("background_counts", cfg.background_counts),
                            ("read_noise_std", cfg.read_noise_std)):
        if manifest[field] != expected:
            raise ValueError(f"Camera metadata mismatch: {field}")
    if not cfg.shutter_min <= manifest["shutter_seconds"] <= cfg.shutter_max or not cfg.power_min <= manifest["laser_power_mw"] <= cfg.power_max:
        raise ValueError("Calibrated settings outside reviewed model bounds")
    phases, frames, ids = data["phases"], data["frames"], data["pattern_ids"]
    if phases.shape != (n, cfg.input_regions) or phases.dtype.kind not in "fiu" or not np.isfinite(phases).all():
        raise ValueError("Invalid phase-only input patterns")
    if frames.shape != (n, cfg.frame_side, cfg.frame_side) or frames.dtype != np.uint8:
        raise ValueError("Expected one raw uint8 camera frame per pattern")
    expected_ids = [f"fit-{i:04}" if i < cfg.fit_patterns else f"holdout-{i-cfg.fit_patterns:04}" for i in range(n)]
    if ids.dtype.kind not in "US" or ids.tolist() != expected_ids:
        raise ValueError("Pattern/frame IDs or split ordering disagree")
    if len(np.unique(phases, axis=0)) != n:
        raise ValueError("Repeated patterns invalidate independent holdout")
    readings, saturation = extract_detectors(frames, manifest["regions"])
    for key, expected in (("detector_readings", readings), ("detector_saturation", saturation),
                          ("whole_frame_saturation", (frames == 255).mean(axis=(1, 2)))):
        if not np.array_equal(data[key], expected):
            raise ValueError(f"Saved {key} disagrees with raw frame extraction")
    return manifest, data, arrays_ref


def retrieve(manifest, data, settings, directory, *, deadline):
    """GGS is selected in advance, never using holdout scores or hidden truth.

    Each uniform detector region has one effective complex row, identifiable up to
    independent row phase for generic phase-only patterns. No truth comparison is
    performed. Clipped fit readings are excluded per detector, not treated as 255
    linear observations. Predictions include the same quantization and ROI mean.
    """
    from bench.plots import prediction_plot
    cfg = BenchConfig.model_validate(manifest["config"])
    fit_count = cfg.fit_patterns
    X = np.exp(1j * data["phases"])
    measured = data["detector_readings"]
    clipped = data["detector_saturation"] > 0
    estimates = np.zeros((4, cfg.input_regions), dtype=complex)
    identifiable = []
    fit_usable = []
    details = []
    for j in range(4):
        mask = ~clipped[:fit_count, j]
        usable = int(mask.sum())
        fit_usable.append(usable)
        valid = usable >= 8 * cfg.input_regions and np.linalg.matrix_rank(X[:fit_count][mask]) == cfg.input_regions
        identifiable.append(bool(valid))
        if valid:
            # Known synthetic camera offset is removed before intensity fitting.
            intensity = np.maximum(measured[:fit_count, j:j+1][mask] - manifest["background_counts"], 0)
            methods = fit_methods(X[:fit_count][mask], intensity, settings, deadline=deadline)
            selected = methods["ggs"]  # reference method fixed before observing holdout
            estimates[j] = selected["T"][0]
            details.append({k: v for k, v in selected.items() if k != "T"})
        else:
            details.append({"failure": "Insufficient unclipped independent fit measurements"})
    scale = manifest["camera_gain"] * manifest["shutter_seconds"] * manifest["laser_power_mw"]
    predicted_matrix = np.zeros((cfg.frame_side, cfg.frame_side, cfg.input_regions), dtype=complex)
    for j, (y0, y1, x0, x1) in enumerate(manifest["regions"]):
        predicted_matrix[y0:y1, x0:x1] = estimates[j] / np.sqrt(scale)
    # Prediction is the expected camera reading: do not add new random read noise.
    predicted_frames = np.array([pattern_to_camera(predicted_matrix, row,
        shutter_seconds=manifest["shutter_seconds"], laser_power_mw=manifest["laser_power_mw"],
        gain=manifest["camera_gain"], background=manifest["background_counts"]) for row in X[fit_count:]])
    predicted, predicted_sat = extract_detectors(predicted_frames, manifest["regions"])
    hold_measured, hold_clipped = measured[fit_count:], clipped[fit_count:]
    metrics = {"fit_pattern_count": fit_count, "holdout_pattern_count": cfg.holdout_patterns,
               "fit_usable_readings": sum(fit_usable), "fit_clipped_readings": int(clipped[:fit_count].sum()),
               "holdout_usable_readings": int((~hold_clipped).sum()), "holdout_clipped_readings": int(hold_clipped.sum())}
    passed = all(identifiable)
    for j in range(4):
        valid = ~hold_clipped[:, j]
        metrics[f"detector_{j}.fit_usable"] = fit_usable[j]
        metrics[f"detector_{j}.holdout_usable"] = int(valid.sum())
        metrics[f"detector_{j}.identifiable"] = int(identifiable[j])
        if valid.any() and identifiable[j]:
            error = predicted[valid, j] - hold_measured[valid, j]
            rmse = float(np.sqrt(np.mean(error**2)))
            nmse = float(np.mean(error**2) / max(np.mean(hold_measured[valid, j]**2), 1e-12))
            metrics[f"detector_{j}.holdout_rmse"] = rmse
            metrics[f"detector_{j}.holdout_nmse"] = nmse
            passed = passed and rmse <= 2 and nmse <= .02 and valid.sum() >= 8
        else:
            passed = False
    metrics["prediction_passed"] = int(passed)
    metrics["measurement_passed"] = int(all(identifiable) and all((~hold_clipped).sum(axis=0) >= 8))
    rows = []
    for i, pattern_id in enumerate(data["pattern_ids"][fit_count:]):
        for j in range(4):
            rows.append({"pattern_id": str(pattern_id), "detector": j,
                "measured_counts": float(hold_measured[i, j]), "predicted_counts": float(predicted[i, j]),
                "signed_error_counts": float(predicted[i, j] - hold_measured[i, j]),
                "absolute_error_counts": float(abs(predicted[i, j] - hold_measured[i, j])),
                "measured_clipped": bool(hold_clipped[i, j]), "predicted_clipped": bool(predicted_sat[i, j] > 0),
                "usable_for_score": bool(not hold_clipped[i, j] and identifiable[j])})
    with (directory / "heldout-predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    prediction_plot(hold_measured, predicted, hold_clipped, directory / "heldout-predictions.png")
    np.savez(directory / "estimated-tm.npz", T=estimates / np.sqrt(scale),
             identifiable=np.asarray(identifiable), holdout_prediction=predicted)
    report = {"method": "GGS with fit-only restart selection; intensity LS also calculated by shared fitter but not selected",
        "data_origin": "simulated", "metrics": metrics, "fit_usable_per_detector": fit_usable,
        "measurement_quality": "pass" if metrics["measurement_passed"] else "inconclusive",
        "prediction_quality": "pass" if passed else "fail_or_inconclusive",
        "thresholds": {"rmse_counts_max": 2, "nmse_max": .02, "usable_holdout_per_detector_min": 8},
        "ambiguity": "Independent global phase per detector row; true TM and internal fields never used for fitting/evaluation",
        "limitations": ["Uniform coherent response in each ROI", "Known synthetic offset subtracted", "Clipped holdout excluded from scores but retained in CSV", "No laboratory realism established"],
        "settings": {k: manifest[k] for k in ("shutter_seconds", "laser_power_mw", "background_counts", "read_noise_std")},
        "fitting": details}
    write_json(directory / "bench-evaluation.json", report)
    outputs = [saved_artifact(directory / name, description) for name, description in [
        ("heldout-predictions.csv", "Per-pattern/per-detector held-out readings, errors and clipping flags"),
        ("heldout-predictions.png", "Measured versus predicted held-out detector counts"),
        ("estimated-tm.npz", "Estimated matrix only, arbitrary independent row phases; no hidden truth"),
        ("bench-evaluation.json", "Separate measurement and prediction quality with usable measurement counts")]]
    if time.monotonic() > deadline:
        raise TimeoutError("Bench evaluation exceeded numerical deadline")
    return metrics, outputs
