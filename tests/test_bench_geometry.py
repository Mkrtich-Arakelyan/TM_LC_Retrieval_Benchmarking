"""Fixed new speckle-grid geometry and geometry-driven archive reanalysis."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from analysis.bench_analysis import load_bench_dataset, retrieve
from bench.acquisition import extract_detectors
from bench.config import BenchConfig
from bench.geometry import detector_regions, geometry_metadata, recorded_regions, slm_phase_pattern
from bench.optical_bench import OpticalBench
from tools.python_execution import fingerprint
from workflow.config import AppConfig
from workflow.state import Artifact, TMFitSettings


def test_empty_config_tiles_all_64_crop_pixels_once_with_sixteen_controls():
    cfg = BenchConfig.model_validate_json("{}")
    bench = OpticalBench(cfg)
    geometry = geometry_metadata()
    assert cfg.input_regions == 16 and cfg.frame_side == 20
    assert geometry["camera_crop"] == [6, 14, 6, 14]
    assert geometry["slm_shape"] == [8, 8]
    assert geometry["detector_order"] == list(range(16))
    expected = [(y, y+2, x, x+2) for y in (6, 8, 10, 12) for x in (6, 8, 10, 12)]
    assert bench.regions == expected
    coverage = np.zeros((20, 20), dtype=int)
    for y0, y1, x0, x1 in bench.regions:
        coverage[y0:y1, x0:x1] += 1
    assert coverage.sum() == 64
    assert (coverage[6:14, 6:14] == 1).all()
    assert np.count_nonzero(coverage) == 64
    controls = np.arange(16) * .15
    slm = slm_phase_pattern(controls)
    assert slm.shape == (8, 8)
    for phase, (y0, y1, x0, x1) in zip(controls, geometry["slm_input_regions"]):
        assert (slm[y0:y1, x0:x1] == phase).all()
    frames = np.stack([bench.frame(phases, .01, 1.) for phases in (np.zeros(16), controls, -controls)])
    readings, saturation = extract_detectors(frames, bench.regions)
    assert frames.shape == (3, 20, 20)
    assert readings.shape == saturation.shape == (3, 16)
    for j, (y0, y1, x0, x1) in enumerate(bench.regions):
        np.testing.assert_array_equal(frames[:, y0:y1, x0:x1],
                                      np.broadcast_to(readings[:, j, None, None], (3, 2, 2)))


@pytest.mark.parametrize("changes", [{"input_regions": 4}, {"input_regions": 32},
                                   {"frame_side": 12}, {"frame_side": 40}, {"layout_version": 1}])
def test_new_simulations_reject_other_layouts(changes):
    with pytest.raises(ValueError, match="New simulated runs require"):
        OpticalBench(BenchConfig(**changes))


def test_recorded_geometry_rejects_overlap_and_mismatched_new_order():
    manifest = geometry_metadata()
    assert len(recorded_regions(manifest, 20)) == 16
    manifest["detector_order"] = list(reversed(range(16)))
    with pytest.raises(ValueError, match="detector_order"):
        recorded_regions(manifest, 20)
    with pytest.raises(ValueError, match="overlap"):
        recorded_regions({"regions": [[1, 3, 1, 3], [2, 4, 2, 4]]}, 20)


def test_legacy_measurements_use_recorded_geometry_and_input_count(tmp_path, monkeypatch):
    # Archival fixture only: deliberately non-current detector positions/order.
    regions = [[15, 17, 2, 4], [2, 4, 15, 17], [2, 4, 2, 4], [15, 17, 15, 17]]
    cfg = BenchConfig(input_regions=4, fit_patterns=128, holdout_patterns=8)
    saved_config = cfg.model_dump()
    saved_config.pop("layout_version")
    rng = np.random.default_rng(81)
    phases = rng.uniform(-np.pi, np.pi, (136, 4))
    matrix = (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))) * .2
    counts = np.clip(np.rint(40 * np.abs(np.exp(1j * phases) @ matrix.T)**2), 0, 255).astype(np.uint8)
    frames = np.zeros((136, 20, 20), dtype=np.uint8)
    for j, (y0, y1, x0, x1) in enumerate(regions):
        frames[:, y0:y1, x0:x1] = counts[:, j, None, None]
    readings, saturation = extract_detectors(frames, regions)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    arrays_path = inputs / "measurements.npz"
    np.savez(arrays_path, phases=phases, frames=frames, detector_readings=readings,
             detector_saturation=saturation, whole_frame_saturation=(frames == 255).mean(axis=(1, 2)),
             pattern_ids=np.asarray([f"fit-{i:04}" if i < 128 else f"holdout-{i-128:04}" for i in range(136)]))
    arrays = Artifact(path=str(arrays_path), description="Recorded legacy measurement fixture", sha256=fingerprint(arrays_path))
    manifest = {"kind": "synthetic-bench-v1", "data_origin": "simulated", "arrays": arrays.model_dump(mode="json"),
                "config": saved_config, "regions": regions, "fit_patterns": 128, "holdout_patterns": 8,
                "camera_gain": 4000., "background_counts": 0., "read_noise_std": 0.,
                "shutter_seconds": .01, "laser_power_mw": 1.}
    manifest_path = inputs / "bench-dataset.json"
    manifest_path.write_text(json.dumps(manifest))
    ref = Artifact(path=str(manifest_path), description="Recorded geometry", sha256=fingerprint(manifest_path))
    app = AppConfig(numerical_input_path=inputs, run_output_path=tmp_path / "runs")
    loaded, data, _ = load_bench_dataset(ref, {arrays.id: arrays}, app)
    assert loaded["regions"] == regions
    np.testing.assert_array_equal(data["detector_readings"], counts)
    from bench import matrices
    monkeypatch.setattr(matrices, "load_detector_matrix", lambda *_: pytest.fail("Reanalysis read truth"))
    output = tmp_path / "reanalysis"
    output.mkdir()
    metrics, _ = retrieve(loaded, data, TMFitSettings(iterations=50, restarts=1), output,
                          deadline=time.monotonic()+120)
    with np.load(output / "estimated-tm.npz") as result:
        assert result["T"].shape == (4, 4)
        assert result["holdout_prediction"].shape == (8, 4)
    report = json.loads((output / "bench-evaluation.json").read_text())
    assert report["regions"] == regions and metrics["holdout_usable_readings"] == 32
