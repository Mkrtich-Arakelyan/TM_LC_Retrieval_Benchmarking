"""Real optical/numerical calculations with explicitly labeled local agent doubles."""

import json
import time
from threading import Event
from uuid import uuid4

import numpy as np
import pytest

from workflow.config import AppConfig
from workflow.graph import WorkflowApp
from workflow.state import AcquisitionAction, ExperimentProcedure, Setting
from bench.config import BenchConfig
from bench.workflow import BENCH_QUESTION, build_integrations, setup_context
from workflow.report import export_decisions
from tools.hardware import canonical_hash, preflight
from fixtures.hardware import RecordingFakeBackend
from checks.interference import run_check
from bench.optical_bench import OpticalBench, SimulatedBackend
from bench.acquisition import extract_detectors


def start_workflow(tmp_path, bench):
    architecture = tmp_path / "synthetic.txt"
    architecture.write_text(setup_context(bench), encoding="utf-8")
    cfg = AppConfig(optical_bench=bench, bench_test_agents=True, hardware_backend="simulated",
        llm_router_enabled=True, reviewed_numerics_enabled=True, max_workflow_iterations=40,
        numerical_timeout_seconds=180, hardware_action_timeout_seconds=180,
        numerical_input_path=tmp_path / "inputs", hardware_input_path=tmp_path / "inputs",
        run_output_path=tmp_path / "runs", workflow_database_path=tmp_path / "workflow.sqlite",
        max_acquisition_measurements=bench.frame_budget)
    app = WorkflowApp(cfg, build_integrations(cfg))
    from bench.workflow import PAPER_PASSAGES as passages
    view = app.start(BENCH_QUESTION, architecture_path=architecture, literature_context=passages)
    return app, view


def approve(app, view):
    return app.submit(view["workflow_id"], {"action": "approve", "reviewer": "Automated synthetic verification",
        "review_fingerprint": view["pending"]["review"]["review_fingerprint"]})


def test_interference_independent_checkpoints_and_shared_camera_kernel(tmp_path):
    result = run_check(tmp_path)
    assert result["simulated"] == [160, 80, 0, 80, 160]
    assert result["one_input_alone"] == 40
    assert (tmp_path / "interference-check.png").is_file()


@pytest.mark.parametrize("shutter", [.01, .001, .1])
def test_end_to_end_calibration_approval_and_heldout_prediction(tmp_path, shutter):
    bench = BenchConfig(shutter_seconds=shutter)
    app, waiting = start_workflow(tmp_path, bench)
    with app:
        assert waiting["status"] == "awaiting_approval", waiting.get("reason")
        assert waiting["state"]["acquisitions"] == []
        rejected = app.submit(waiting["workflow_id"], {"action": "approve", "review_fingerprint": "wrong"})
        assert rejected.get("input_error") and not rejected["state"]["acquisitions"]
        view = approve(app, waiting)
        assert view["status"] == "completed", view.get("reason")
    state = view["state"]
    assert len(state["acquisitions"]) == 1
    acquisition = state["acquisitions"][0]
    assert acquisition["status"] == "completed" and acquisition["data_origin"] == "simulated"
    calibration_ref = next(a for a in acquisition["raw_data"] if a["path"].endswith("calibration.json"))
    from pathlib import Path
    calibration = json.loads(Path(calibration_ref["path"]).read_text())
    assert .03 <= calibration["after"]["whole_frame_saturation"] <= .05
    assert calibration["success"]
    if shutter == .001:
        assert calibration["before"]["whole_frame_saturation"] < .03
        assert len(calibration["attempts"]) > 1
    elif shutter == .01:
        assert len(calibration["attempts"]) == 1
    else:
        assert calibration["before"]["whole_frame_saturation"] > .05
        assert len(calibration["attempts"]) > 1
    analysis = state["analyses"][-1]
    assert analysis["metrics"]["prediction_passed"] == 1, analysis["metrics"]
    assert analysis["metrics"]["measurement_passed"] == 1
    assert analysis["metrics"]["holdout_usable_readings"] == 4 * bench.holdout_patterns
    assert state["analyst_reviews"][-1]["status"] == "reviewed"
    assert {e["role"] for e in state["conversation"]} >= {"scientist", "experimentor", "runner", "analyst"}
    manifest_ref = next(a for a in acquisition["raw_data"] if a["path"].endswith("bench-dataset.json"))
    manifest = json.loads(Path(manifest_ref["path"]).read_text())
    arrays = np.load(manifest["arrays"]["path"], allow_pickle=False)
    assert arrays["frames"].dtype == np.uint8
    assert "T" not in arrays.files and "field" not in arrays.files
    readings, saturation = extract_detectors(arrays["frames"], manifest["regions"])
    np.testing.assert_array_equal(readings, arrays["detector_readings"])
    np.testing.assert_array_equal(saturation, arrays["detector_saturation"])
    # Acquisition and held-out observations use the same frozen calibrated controls.
    independent = OpticalBench(bench)
    for i in (0, bench.fit_patterns-1, bench.fit_patterns, len(arrays["phases"])-1):
        np.testing.assert_array_equal(arrays["frames"][i], independent.frame(arrays["phases"][i],
            manifest["shutter_seconds"], manifest["laser_power_mw"], capture_index=1000+i))
    assert state["control"]["literature_context"]
    export_decisions(view)
    quality = json.loads((Path(state["control"]["configuration"]["run_output_path"]) / "quality.json").read_text())
    assert quality["workflow_completed"] and quality["prediction_quality"] == "pass"


def test_unreachable_band_preserves_diagnostics_and_never_measures(tmp_path):
    app, waiting = start_workflow(tmp_path, BenchConfig(shutter_seconds=.0005, shutter_max=.001, power_max=1.))
    with app:
        view = approve(app, waiting)
    assert view["status"] == "awaiting_information"
    assert not view["state"]["analyses"]
    acquisition = view["state"]["acquisitions"][0]
    assert acquisition["status"] == "partial"
    assert "unreachable" in acquisition["errors"][0]
    assert any(a["path"].endswith("calibration.csv") for a in acquisition["raw_data"])
    assert any(a["path"].endswith("raw_after.npy") for a in acquisition["raw_data"])
    assert not any(a["path"].endswith("measurements.npz") for a in acquisition["raw_data"])
    export_decisions(view)


def test_seeded_phase_only_frames_and_control_product():
    cfg = BenchConfig()
    a, b = OpticalBench(cfg), OpticalBench(cfg)
    phases = np.linspace(0, np.pi, cfg.input_regions)
    np.testing.assert_array_equal(a.frame(phases, .01, 1), b.frame(phases, .01, 1))
    np.testing.assert_array_equal(a.frame(phases, .01, 1), a.frame(phases, .005, 2))
    assert np.mean(a.frame(np.zeros(cfg.input_regions), cfg.shutter_min, cfg.power_min) == 255) < .03
    assert np.mean(a.frame(np.zeros(cfg.input_regions), cfg.shutter_max, cfg.power_max) == 255) > .05
    with pytest.raises(ValueError, match="phase"):
        a.frame(np.zeros(cfg.input_regions-1), .01, 1)
    with pytest.raises(ValueError, match="phase"):
        a.frame(np.ones(cfg.input_regions) * 1j, .01, 1)


def test_completed_workflow_can_have_failed_prediction_quality(tmp_path):
    app, waiting = start_workflow(tmp_path, BenchConfig(read_noise_std=20))
    with app:
        view = approve(app, waiting)
    assert view["status"] == "completed", view.get("reason")
    assert view["state"]["analyses"][-1]["status"] == "completed"
    assert view["state"]["analyses"][-1]["metrics"]["prediction_passed"] == 0
    export_decisions(view)


def test_clipped_fit_exclusion_and_holdout_never_change_fit(tmp_path):
    from pathlib import Path
    from analysis.bench_analysis import retrieve
    from workflow.state import TMFitSettings
    app, waiting = start_workflow(tmp_path, BenchConfig())
    with app:
        view = approve(app, waiting)
    acquisition = view["state"]["acquisitions"][0]
    ref = next(a for a in acquisition["raw_data"] if a["path"].endswith("bench-dataset.json"))
    manifest = json.loads(Path(ref["path"]).read_text())
    with np.load(manifest["arrays"]["path"], allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    for y0, y1, x0, x1 in manifest["regions"]:
        data["frames"][:4, y0:y1, x0:x1] = 255
        data["frames"][manifest["fit_patterns"], y0:y1, x0:x1] = 255
    data["detector_readings"], data["detector_saturation"] = extract_detectors(data["frames"], manifest["regions"])
    settings = TMFitSettings(iterations=100, restarts=1)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    metrics, _ = retrieve(manifest, data, settings, first, deadline=time.monotonic()+120)
    assert metrics["fit_clipped_readings"] == 16
    assert metrics["fit_usable_readings"] == 4 * (manifest["fit_patterns"]-4)
    assert metrics["holdout_clipped_readings"] == 4
    assert metrics["holdout_usable_readings"] == 4 * (manifest["holdout_patterns"]-1)
    # Alter held-out outcomes drastically; recovered matrix must stay identical.
    data["detector_readings"][manifest["fit_patterns"]:] = 0
    retrieve(manifest, data, settings, second, deadline=time.monotonic()+120)
    with np.load(first / "estimated-tm.npz") as a, np.load(second / "estimated-tm.npz") as b:
        np.testing.assert_array_equal(a["T"], b["T"])


def test_cancelled_backend_does_not_acquire_sequence(tmp_path):
    cfg = BenchConfig()
    backend = SimulatedBackend(cfg)
    action = AcquisitionAction(action="bench.acquire", instruction="Calibrate then measure",
        requested_settings={"shutter_seconds": Setting(value=.01, unit="s"),
            "laser_power_mw": Setting(value=1., unit="mW"), "frame_budget": Setting(value=cfg.frame_budget, unit="frame")})
    cancelled = Event()
    cancelled.set()
    result = backend.execute(action, output_dir=tmp_path, input_artifacts=[],
        capability_fingerprint=canonical_hash(backend.describe().model_dump(mode="json")),
        deadline=time.monotonic()+60, cancel_event=cancelled)
    assert result.aborted and result.errors
    assert not (tmp_path / "measurements.npz").exists()


def test_simulated_configuration_rejects_other_backend_origin(tmp_path):
    from workflow.state import Artifact
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}")
    from tools.python_execution import fingerprint
    fake = RecordingFakeBackend(Artifact(path=str(calibration), description="test", sha256=fingerprint(calibration)))
    procedure = ExperimentProcedure(proposal_id=uuid4(), version=1,
        actions=[AcquisitionAction(action="fixture.capture", instruction="test", requested_settings={"count": Setting(value=1, unit="frame")})],
        prerequisites=["fixture setup verified"], expected_outputs=["test"], calibration_artifacts=fake.describe().verified_calibrations)
    cfg = AppConfig(hardware_backend="simulated", hardware_input_path=tmp_path, run_output_path=tmp_path / "runs")
    review = preflight(procedure, cfg, fake)
    assert any("physical fallback is forbidden" in issue for issue in review.issues)
