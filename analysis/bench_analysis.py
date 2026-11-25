"""Reviewed intensity-only retrieval from saved bench measurements, without truth."""

import csv
import json
import time
from pathlib import Path

import numpy as np

from bench.config import BenchConfig
from bench.optical_bench import CAMERA_GAIN, pattern_to_camera
from bench.geometry import recorded_regions
from bench.acquisition import extract_detectors, saved_artifact, write_json
from tools.python_execution import checked_inputs
from analysis.tm_solvers import _archive, fit_methods
from analysis.metrics import mean_baseline

BENCH_ARRAYS = {"phases", "frames", "pattern_ids", "detector_readings",
                "detector_saturation", "whole_frame_saturation"}


def is_bench_manifest(reference, config):
    if reference.sha256 is None:
        raise ValueError("Dataset manifest requires a recorded content hash")
    checked = checked_inputs([reference], config)[0]
    if Path(checked.path).stat().st_size > 1024 * 1024:
        raise ValueError("Dataset manifest exceeds size budget")
    payload = json.loads(Path(checked.path).read_text(encoding="utf-8"))
    return isinstance(payload, dict) and payload.get("kind") == "synthetic-bench-v1"


def inspection_outputs(manifest, data, predicted, predicted_sat, identifiable, directory):
    """Describe saved observations/predictions; no fitting, selection, or hidden TM."""
    from bench.plots import (frame_preview, raw_frame_image, representative_grids,
                             heldout_error_plot, slm_pattern_plot)
    fit_count = manifest["fit_patterns"]
    regions = manifest["regions"]
    ids = manifest.get("detector_order", list(range(len(regions))))
    measured = data["detector_readings"][fit_count:]
    clipped = data["detector_saturation"][fit_count:] > 0
    rows, variation = [], []
    for j, detector in enumerate(ids):
        valid = ~clipped[:, j] & bool(identifiable[j])
        error = predicted[valid, j] - measured[valid, j]
        target = measured[valid, j]
        all_unclipped = data["detector_readings"][:, j][data["detector_saturation"][:, j] == 0]
        span = float(np.ptp(all_unclipped)) if len(all_unclipped) else None
        standard_deviation = float(np.std(all_unclipped)) if len(all_unclipped) else None
        variation.append({"detector": detector, "unclipped_patterns": len(all_unclipped),
                          "span_counts": span, "std_counts": standard_deviation,
                          "low_variation": bool(len(all_unclipped) >= 2 and span <= 1.)})
        rows.append({"detector": detector, "total_readings": len(measured),
                     "unclipped_readings": int((~clipped[:, j]).sum()),
                     "clipped_readings": int(clipped[:, j].sum()), "usable_for_score": int(valid.sum()),
                     "unscorable_unclipped": int((~clipped[:, j] & ~np.bool_(identifiable[j])).sum()),
                     "predicted_clipped_readings": int((predicted_sat[:, j] > 0).sum()),
                     "rmse_counts": float(np.sqrt(np.mean(error**2))) if len(error) else None,
                     "mae_counts": float(np.mean(np.abs(error))) if len(error) else None,
                     "nmse": float(np.mean(error**2) / max(np.mean(target**2), 1e-12)) if len(error) else None,
                     "max_abs_error_counts": float(np.max(np.abs(error))) if len(error) else None,
                     "all_patterns_span_counts": span, "all_patterns_std_counts": standard_deviation})
    with (directory / "heldout-detector-errors.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    heldout_error_plot(rows, directory / "heldout-detector-errors.png")
    y_positions = sorted({r[0] for r in regions})
    x_positions = sorted({r[2] for r in regions})
    example = {"pattern_id": str(data["pattern_ids"][fit_count]),
               "selection": "First saved held-out pattern; not chosen by error or clipping",
               "phase_radians": data["phases"][fit_count].tolist(),
               "detector_ids": ids, "regions": regions, "camera_crop": manifest.get("camera_crop"),
               "grid_positions": [[y_positions.index(r[0]), x_positions.index(r[2])] for r in regions],
               "measured_counts": measured[0].tolist(), "predicted_counts": predicted[0].tolist(),
               "difference_counts": (predicted[0] - measured[0]).tolist(),
               "measured_clipped": clipped[0].tolist(), "predicted_clipped": (predicted_sat[0] > 0).tolist(),
               "usable_for_score": ((~clipped[0]) & np.asarray(identifiable)).tolist(),
               "brightness_scale": [0, 255]}
    write_json(directory / "heldout-example.json", example)
    representative_grids(example, directory / "heldout-example-grids.png")
    frame = data["frames"][fit_count]
    np.save(directory / "heldout-camera-raw.npy", frame)
    raw_frame_image(frame, directory / "heldout-camera-raw.png")
    frame_preview(frame, regions, directory / "heldout-camera-preview.png", example["pattern_id"],
                  crop=manifest.get("camera_crop"))
    filenames = ["heldout-detector-errors.csv", "heldout-detector-errors.png", "heldout-example.json",
                 "heldout-example-grids.png", "heldout-camera-raw.npy", "heldout-camera-raw.png",
                 "heldout-camera-preview.png"]
    if manifest.get("geometry_version") == 2 and data["phases"].shape[1] == 16:
        slm_pattern_plot(data["phases"][fit_count], directory / "heldout-slm-pattern.png", example["pattern_id"])
        filenames.append("heldout-slm-pattern.png")
    return {"heldout_detector_errors": rows,
            "representative_pattern_id": example["pattern_id"],
            "measurement_variation": {"span_threshold_counts": 1., "scope": "All unclipped fit and held-out readings",
                                      "per_detector": variation,
                                      "low_variation_detectors": [v["detector"] for v in variation if v["low_variation"]]}}, filenames


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
    regions = recorded_regions(manifest, cfg.frame_side)
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
    readings, saturation = extract_detectors(frames, regions)
    for key, expected in (("detector_readings", readings), ("detector_saturation", saturation),
                          ("whole_frame_saturation", (frames == 255).mean(axis=(1, 2)))):
        if not np.array_equal(data[key], expected):
            raise ValueError(f"Saved {key} disagrees with raw frame extraction")
    return manifest, data, arrays_ref


def retrieve(manifest, data, settings, directory, *, deadline, compare=False):
    """GGS-only or matched comparison; never select using holdout or hidden truth.

    The legacy `identifiable` mask checks only fit eligibility (count and input
    rank), not unique recovery. Intensity-only patterns can retain mapping and
    amplitude ambiguities as well as independent row phase. No truth comparison is
    performed. Clipped fit readings are excluded per detector, not treated as 255
    linear observations. Predictions include the same quantization and ROI mean.
    """
    cfg = BenchConfig.model_validate(manifest["config"])
    regions = recorded_regions(manifest, cfg.frame_side)
    detector_count = len(regions)
    fit_count = cfg.fit_patterns
    X = np.exp(1j * data["phases"])
    measured = data["detector_readings"]
    clipped = data["detector_saturation"] > 0
    names = ("ggs", "intensity_ls") if compare else ("ggs",)
    estimates = {name: np.zeros((detector_count, cfg.input_regions), dtype=complex) for name in names}
    identifiable = []
    fit_usable = []
    details = {name: [] for name in names}
    for j in range(detector_count):
        mask = ~clipped[:fit_count, j]
        usable = int(mask.sum())
        fit_usable.append(usable)
        valid = usable >= 8 * cfg.input_regions and np.linalg.matrix_rank(X[:fit_count][mask]) == cfg.input_regions
        identifiable.append(bool(valid))
        if valid:
            # Known synthetic camera offset is removed before intensity fitting.
            intensity = np.maximum(measured[:fit_count, j:j+1][mask] - manifest["background_counts"], 0)
            methods = fit_methods(X[:fit_count][mask], intensity, settings, deadline=deadline)
            for name in names:
                selected = methods[name]  # Both share this exact fit-only call and mask.
                estimates[name][j] = selected["T"][0]
                details[name].append({k: v for k, v in selected.items() if k != "T"})
        else:
            for name in names:
                details[name].append({"failure": "Insufficient unclipped independent fit measurements"})
    if not compare:
        return _evaluate_fit(manifest, data, directory, estimates["ggs"], identifiable, fit_usable, details["ggs"], deadline=deadline)
    per_method, outputs, reports = {}, [], {}
    for name in names:
        method_dir = directory / name
        method_dir.mkdir()
        metrics, refs = _evaluate_fit(manifest, data, method_dir, estimates[name], identifiable, fit_usable, details[name],
            deadline=deadline, method=name, matrix_name=name + ".npz")
        per_method[name] = metrics
        reports[name] = json.loads((method_dir / "bench-evaluation.json").read_text(encoding="utf-8"))
        outputs.extend(refs)
    common_keys = ("fit_pattern_count", "holdout_pattern_count", "fit_usable_readings", "fit_clipped_readings",
                   "holdout_usable_readings", "holdout_clipped_readings", "holdout_scored_readings", "measurement_passed")
    metrics = {key: per_method["ggs"][key] for key in common_keys}
    metrics.update({f"{name}.{key}": value for name, result in per_method.items() for key, value in result.items()})
    metrics["prediction_passed"] = int(all(per_method[name]["prediction_passed"] for name in names))
    metrics["shared_eligible_masks"] = 1
    for key in ("mean_baseline.holdout_rmse", "mean_baseline.holdout_nmse", "shared_initialization_seconds"):
        if key in per_method["ggs"]:
            metrics[key] = per_method["ggs"][key]
    if "holdout_rmse" in per_method["ggs"]:
        first, second = (per_method[name]["holdout_rmse"] for name in names)
        metrics["intensity_ls_minus_ggs.holdout_rmse"] = second - first
        metrics["intensity_ls_minus_ggs.holdout_nmse"] = per_method["intensity_ls"]["holdout_nmse"] - per_method["ggs"]["holdout_nmse"]
        metrics["comparison.brightness_tie"] = int(np.isclose(first, second, rtol=1e-9, atol=1e-12))
    report = {"method": "Matched GGS versus nonlinear intensity least squares; no method selected using held-out results",
        "methods": reports, "metrics": metrics, "data_origin": "simulated",
        "measurement_variation": reports["ggs"]["measurement_variation"],
        "comparison": {"selection": "Identical fit/holdout patterns and per-detector eligibility masks. Shared seeded starts; best restart/iterate selected by training intensity MSE only.",
            "result": "tie" if metrics.get("comparison.brightness_tie") else
                ("ggs has lower RMSE on this held-out set" if metrics["intensity_ls_minus_ggs.holdout_rmse"] > 0 else "intensity_ls has lower RMSE on this held-out set")
                if "comparison.brightness_tie" in metrics else "inconclusive",
            "limitations": "One dataset/run, not a significance test. Low prediction error is not unique TM recovery. Truth is unavailable to fitting and agents.",
            "timing": "Per-method runtime excludes shared initialization, which is reported separately; plotting/export time excluded."}}
    write_json(directory / "bench-evaluation.json", report)
    outputs.append(saved_artifact(directory / "bench-evaluation.json", "Matched benchmark of GGS and nonlinear intensity least squares"))
    return metrics, outputs


def _evaluate_fit(manifest, data, directory, estimates, identifiable, fit_usable, details, *, deadline,
                  method="ggs", matrix_name="estimated-tm.npz"):
    from bench.plots import prediction_plot
    cfg = BenchConfig.model_validate(manifest["config"])
    regions = recorded_regions(manifest, cfg.frame_side)
    detector_count, fit_count = len(regions), cfg.fit_patterns
    X, measured = np.exp(1j * data["phases"]), data["detector_readings"]
    clipped = data["detector_saturation"] > 0
    scale = manifest["camera_gain"] * manifest["shutter_seconds"] * manifest["laser_power_mw"]
    predicted_matrix = np.zeros((cfg.frame_side, cfg.frame_side, cfg.input_regions), dtype=complex)
    for j, (y0, y1, x0, x1) in enumerate(regions):
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
    for j in range(detector_count):
        valid = ~hold_clipped[:, j]
        metrics[f"detector_{j}.fit_usable"] = fit_usable[j]
        metrics[f"detector_{j}.holdout_usable"] = int(valid.sum())
        metrics[f"detector_{j}.identifiable"] = int(identifiable[j])
        metrics[f"detector_{j}.fit_eligible"] = int(identifiable[j])
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
    baseline = mean_baseline(measured, np.arange(fit_count), np.arange(fit_count, len(measured)), predicted, identifiable, clipped)
    metrics["holdout_scored_readings"] = baseline["usable"]
    for key, value in baseline.items():
        target = {"fit_rmse": "holdout_rmse", "fit_nmse": "holdout_nmse",
                  "mean_baseline_rmse": "mean_baseline.holdout_rmse", "mean_baseline_nmse": "mean_baseline.holdout_nmse"}.get(key)
        if target and value is not None:
            metrics[target] = value
    metrics["runtime_seconds"] = sum(detail.get("runtime_seconds", 0.) for detail in details)
    metrics["shared_initialization_seconds"] = sum(detail.get("shared_initialization_seconds", 0.) for detail in details)
    rows = []
    for i, pattern_id in enumerate(data["pattern_ids"][fit_count:]):
        for j in range(detector_count):
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
    np.savez(directory / matrix_name, T=estimates / np.sqrt(scale),
             identifiable=np.asarray(identifiable), fit_eligible=np.asarray(identifiable), holdout_prediction=predicted)
    inspection, inspection_files = inspection_outputs(manifest, data, predicted, predicted_sat, identifiable, directory)
    report = {"method": "GGS with fit-only restart selection; intensity LS also calculated by shared fitter but not selected" if matrix_name == "estimated-tm.npz" else method + " with fit-only restart selection",
        **inspection,
        "regions": regions, "camera_crop": manifest.get("camera_crop"),
        "detector_order": manifest.get("detector_order", list(range(detector_count))),
        "data_origin": "simulated", "metrics": metrics, "fit_usable_per_detector": fit_usable,
        "measurement_quality": "pass" if metrics["measurement_passed"] else "inconclusive",
        "prediction_quality": "pass" if passed else "fail_or_inconclusive",
        "thresholds": {"rmse_counts_max": 2, "nmse_max": .02, "usable_holdout_per_detector_min": 8},
        "fit_eligibility": {"legacy_field": "identifiable", "meaning": "At least 8*input_count unclipped fit samples and full complex input-pattern rank; NOT unique recovery"},
        "ambiguity": "Independent global phase per detector row; other mapping/amplitude ambiguities may remain. True TM and internal fields never used for fitting/evaluation",
        "limitations": ["Uniform coherent response in each ROI", "Known synthetic offset subtracted", "Clipped holdout excluded from scores but retained in CSV", "No laboratory realism established"],
        "settings": {k: manifest[k] for k in ("shutter_seconds", "laser_power_mw", "background_counts", "read_noise_std")},
        "fitting": details, "mean_baseline": baseline}
    write_json(directory / "bench-evaluation.json", report)
    outputs = [saved_artifact(directory / name, description) for name, description in [
        ("heldout-predictions.csv", "Per-pattern/per-detector held-out readings, errors and clipping flags"),
        ("heldout-predictions.png", "Measured versus predicted held-out detector counts"),
        (matrix_name, "Estimated matrix only, arbitrary independent row phases; no hidden truth"),
        ("bench-evaluation.json", "Separate measurement and prediction quality with usable measurement counts")]]
    outputs.extend(saved_artifact(directory / name, "Recorded held-out inspection: " + name) for name in inspection_files)
    if time.monotonic() > deadline:
        raise TimeoutError("Bench evaluation exceeded numerical deadline")
    return metrics, outputs
