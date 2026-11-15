"""Truth-free measurement diagnostics, not a certificate of unique TM recovery."""

import numpy as np


def measurement_diagnostics(inputs, readings, clipped=None, *, noise_std=None,
                            quantization_counts=None, background=None):
    inputs, readings = np.asarray(inputs), np.asarray(readings)
    clipping_known = clipped is not None
    clipped = np.zeros(readings.shape, dtype=bool) if clipped is None else np.asarray(clipped, dtype=bool)
    detectors = []
    for j in range(readings.shape[1]):
        valid = ~clipped[:, j]
        values = readings[valid, j]
        noise = float(np.broadcast_to(noise_std, (readings.shape[1],))[j]) if noise_std is not None else None
        floor = max(noise or 0., (quantization_counts or 0.) / np.sqrt(12))
        mean = float(np.mean(values)) if len(values) else None
        std = float(np.std(values)) if len(values) else None
        detectors.append({"detector": j, "total": len(valid), "unclipped": int(valid.sum()) if clipping_known else None,
            "clipped": int((~valid).sum()) if clipping_known else None, "mean": mean,
            "span": float(np.ptp(values)) if len(values) else None, "std": std,
            "noise_std": noise, "quantization_counts": quantization_counts,
            "low_variation": bool(std <= 2 * floor) if len(values) >= 2 and (noise_std is not None or quantization_counts is not None) else None,
            "weak_signal": bool(mean - background <= 3 * floor) if mean is not None and background is not None else None,
            "pattern_rank_on_available_readings": int(np.linalg.matrix_rank(inputs[valid])) if valid.any() else 0})
    equal_amplitude = bool(np.allclose(np.abs(inputs), np.abs(inputs[0, 0])))
    return {"input_count": inputs.shape[1], "pattern_count": len(inputs),
        "input_pattern_rank": int(np.linalg.matrix_rank(inputs)), "equal_amplitude_inputs": equal_amplitude,
        "clipping_known": clipping_known, "scope": "All recorded patterns; not a fit-only uniqueness test",
        "per_detector": detectors,
        "interpretation": "Variation, rank and sample counts are diagnostics, not proof of unique mapping or complex matrix recovery. "
            "Clipped observations censor brightness; weak signals and unresolved variation must be judged against noise and quantization. "
            "Equal-amplitude intensity measurements mix diagonal connection energies; phase diversity alone does not always separate them. "
            "Even varying readings can retain amplitude/mapping ambiguities. Independent row phase is unobservable without a reference.",
        "uncertainty": "Noise or quantization not supplied remains unknown; low variation is a diagnostic to investigate, not a non-identifiability verdict."}


def review_evidence(state, analysis_ids, config):
    """Use registered acquisition inputs only. Never load T/ground-truth archives."""
    from pathlib import Path
    from analysis.datasets import registered_artifacts, load_dataset
    from analysis.bench_analysis import load_bench_dataset

    results = []
    for analysis in state.analyses:
        if analysis.id not in analysis_ids:
            continue
        result = {"analysis_id": str(analysis.id), "operation": analysis.method_settings.get("operation"),
                  "numerical_settings": analysis.method_settings, "metrics": analysis.metrics,
                  "acquisitions": [{"id": str(a.id), "settings": {k: v.model_dump(mode="json") for k, v in a.actual_settings.items()}}
                                   for a in state.acquisitions if a.id in analysis.acquisition_ids]}
        errors = []
        try:
            refs, _ = registered_artifacts(state, [analysis.id])
            for ref in analysis.input_data:
                if Path(ref.path).suffix.lower() != ".json":
                    continue
                try:
                    if analysis.method_settings.get("operation") == "bench.retrieve":
                        manifest, data, _ = load_bench_dataset(ref, refs, config)
                        regions = manifest["regions"]
                        noise = [manifest["read_noise_std"] / np.sqrt((r[1]-r[0])*(r[3]-r[2])) for r in regions]
                        result["measurements"] = measurement_diagnostics(np.exp(1j * data["phases"]), data["detector_readings"],
                            data["detector_saturation"] > 0, noise_std=noise, quantization_counts=1., background=manifest["background_counts"])
                        split = manifest["fit_patterns"]
                        for j, detector in enumerate(result["measurements"]["per_detector"]):
                            saturated = data["detector_saturation"][:, j] > 0
                            detector.update({"fit_unclipped": int((~saturated[:split]).sum()),
                                "fit_clipped": int(saturated[:split].sum()),
                                "holdout_unclipped": int((~saturated[split:]).sum()),
                                "holdout_clipped": int(saturated[split:].sum())})
                        result["camera_settings"] = {k: manifest[k] for k in ("shutter_seconds", "laser_power_mw", "background_counts", "read_noise_std")}
                    else:
                        dataset = load_dataset(ref, refs, config)
                        result["measurements"] = measurement_diagnostics(dataset.X, dataset.intensities)
                    break
                except (ValueError, OSError, KeyError) as exc:
                    errors.append(str(exc)[:300])
        except ValueError as exc:
            errors.append(str(exc)[:300])
        if "measurements" not in result:
            result["diagnostics_unavailable"] = errors or ["No supported registered measurement manifest; do not infer recovery from metrics alone"]
        results.append(result)
    return results
