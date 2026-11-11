"""Synthetic calibration and frame-acquisition procedure; no hidden optical model."""

import csv
import json
import time

import numpy as np

from workflow.state import Artifact, Setting
from tools.hardware import ActionReport
from tools.python_execution import fingerprint


def extract_detectors(frames, regions):
    """Identical extraction for acquisition and prediction: mean raw uint8 ROI."""
    values = np.stack([frames[..., y0:y1, x0:x1].mean(axis=(-2, -1))
                       for y0, y1, x0, x1 in regions], axis=-1)
    saturation = np.stack([(frames[..., y0:y1, x0:x1] == 255).mean(axis=(-2, -1))
                           for y0, y1, x0, x1 in regions], axis=-1)
    return values, saturation


def saved_artifact(path, description):
    return Artifact(path=str(path.resolve()), description=description, sha256=fingerprint(path))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def run_acquisition(bench, config, action, *, output_dir, deadline, cancel_event, camera_gain):
    """Calibrate and capture through the bench's public frame/regions interface.

    The Runner adapter validates the action and capability fingerprint first.
    This procedure owns the acquisition order, locked settings and artifacts.
    """
    cfg = config
    shutter = float(action.requested_settings["shutter_seconds"].value)
    power = float(action.requested_settings["laser_power_mw"].value)
    attempts, frames = [], []
    artifacts = []
    error = None
    def check():
        if cancel_event.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("Synthetic acquisition cancelled or deadline reached")
    def capture(s, p):
        check()
        frame = bench.frame(np.zeros(cfg.input_regions), s, p, capture_index=len(attempts))
        saturation = float(np.mean(frame == 255))  # WHOLE raw frame, never ROI-only
        attempts.append({"attempt": len(attempts) + 1, "shutter_seconds": s,
                         "laser_power_mw": p, "whole_frame_saturation": saturation})
        frames.append(frame)
        return saturation
    def settings_for(product):
        s = float(np.clip(product / power, cfg.shutter_min, cfg.shutter_max))
        p = float(np.clip(product / s, cfg.power_min, cfg.power_max))
        return s, p
    try:
        saturation = capture(shutter, power)
        if not .03 <= saturation <= .05:
            low, high = cfg.shutter_min * cfg.power_min, cfg.shutter_max * cfg.power_max
            low_sat = capture(*settings_for(low))
            if .03 <= low_sat <= .05:
                saturation = low_sat
            else:
                high_sat = capture(*settings_for(high))
                if .03 <= high_sat <= .05:
                    saturation = high_sat
                elif low_sat > .05 or high_sat < .03:
                    raise ValueError(f"Saturation target unreachable within configured limits: minimum {low_sat:.3%}, maximum {high_sat:.3%}; target 3–5%")
                else:
                    while len(attempts) < cfg.calibration_attempts:
                        product = float(np.sqrt(low * high))
                        saturation = capture(*settings_for(product))
                        if .03 <= saturation <= .05:
                            break
                        if saturation < .03:
                            low = product
                        else:
                            high = product
                    else:
                        raise ValueError("Saturation target unreachable at available pixel/setting resolution within calibration attempt budget")
        shutter, power = attempts[-1]["shutter_seconds"], attempts[-1]["laser_power_mw"]
    except (ValueError, TimeoutError) as exc:
        error = str(exc)
    # Preserve all attempted calibration diagnostics even if no sequence follows.
    csv_path = output_dir / "calibration.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["attempt", "shutter_seconds", "laser_power_mw", "whole_frame_saturation"])
        writer.writeheader()
        writer.writerows(attempts)
    artifacts.append(saved_artifact(csv_path, "Every calibration adjustment; whole-frame saturation fraction"))
    from bench.plots import calibration_plot, frame_preview
    calibration_plot(attempts, output_dir / "calibration.png")
    artifacts.append(saved_artifact(output_dir / "calibration.png", "Calibration with shaded 3–5% target band"))
    diagnostics = {"target": [.03, .05], "success": error is None, "error": error, "attempts": attempts}
    for name, frame in zip(("before", "after"), (frames[0], frames[-1])) if frames else []:
        np.save(output_dir / f"raw_{name}.npy", frame)
        frame_preview(frame, bench.regions, output_dir / f"preview_{name}.png", name)
        _, sat = extract_detectors(frame, bench.regions)
        diagnostics[name] = {"whole_frame_saturation": float(np.mean(frame == 255)),
                             "detector_saturation": sat.tolist()}
        for filename in (f"raw_{name}.npy", f"preview_{name}.png"):
            artifacts.append(saved_artifact(output_dir / filename, f"Calibration {name}; same raw 0–255 scale"))
    write_json(output_dir / "calibration.json", diagnostics)
    artifacts.append(saved_artifact(output_dir / "calibration.json", "Whole-frame/per-detector saturation and stopping reason"))
    actual = {k: v.model_copy(deep=True) for k, v in action.requested_settings.items()}
    if attempts:
        actual["shutter_seconds"] = Setting(value=attempts[-1]["shutter_seconds"], unit="s")
        actual["laser_power_mw"] = Setting(value=attempts[-1]["laser_power_mw"], unit="mW")
    if error:
        return ActionReport(actual_settings=actual, raw_data=artifacts, errors=[error],
                            aborted=cancel_event.is_set())
    # Settings are frozen here for fit AND holdout. No adaptation per pattern.
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 502]))
    phases = rng.uniform(-np.pi, np.pi, (cfg.fit_patterns + cfg.holdout_patterns, cfg.input_regions))
    sequence = []
    try:
        for i, pattern in enumerate(phases):
            check()
            sequence.append(bench.frame(pattern, shutter, power, capture_index=1000 + i))
    except TimeoutError as exc:
        return ActionReport(raw_data=artifacts, errors=[str(exc)], aborted=True)
    raw = np.asarray(sequence)
    readings, saturation = extract_detectors(raw, bench.regions)
    ids = np.asarray([f"fit-{i:04}" if i < cfg.fit_patterns else f"holdout-{i-cfg.fit_patterns:04}" for i in range(len(raw))])
    arrays_path = output_dir / "measurements.npz"
    np.savez_compressed(arrays_path, phases=phases, frames=raw, pattern_ids=ids,
                        detector_readings=readings, detector_saturation=saturation,
                        whole_frame_saturation=(raw == 255).mean(axis=(1, 2)))
    arrays = saved_artifact(arrays_path, "Known phase-only patterns and uint8 camera measurements; no hidden fields/T")
    artifacts.append(arrays)
    manifest = {"kind": "synthetic-bench-v1", "arrays": arrays.model_dump(mode="json"),
        "data_origin": "simulated", "regions": bench.regions,
        "fit_patterns": cfg.fit_patterns, "holdout_patterns": cfg.holdout_patterns,
        "shutter_seconds": shutter, "laser_power_mw": power,
        "background_counts": cfg.background_counts, "read_noise_std": cfg.read_noise_std,
        "camera_gain": camera_gain, "config": cfg.model_dump(),
        "extraction": "Mean uint8 pixels in each uniform-response detector ROI; any saturated ROI pixel flags reading"}
    write_json(output_dir / "bench-dataset.json", manifest)
    artifacts.append(saved_artifact(output_dir / "bench-dataset.json", "Synthetic bench dataset manifest for bench.retrieve"))
    return ActionReport(actual_settings=actual, raw_data=artifacts)
