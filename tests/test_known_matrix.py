"""Matrix selection, independent camera checks, and immutable workflow inputs."""

import cmath
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import main as cli
from bench.acquisition import extract_detectors
from bench.config import BenchConfig
from bench.matrices import pin_detector_matrix
from bench.optical_bench import OpticalBench, SimulatedBackend
from bench.workflow import build_integrations, setup_context
from workflow.config import AppConfig
from workflow.graph import Integrations, WorkflowApp

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def known_config():
    cfg = BenchConfig.model_validate_json((CONFIGS / "bench-mixing.json").read_text())
    return pin_detector_matrix(cfg, base_directory=CONFIGS)


@pytest.mark.parametrize("seed,noise", [(17, 0.), (29, .75)])
def test_random_grid_camera_is_seeded_and_repeatable(seed, noise):
    cfg = BenchConfig(seed=seed, read_noise_std=noise)
    assert "detector_matrix_path" not in cfg.model_dump()
    assert "detector_matrix_sha256" not in cfg.model_dump(mode="json")
    a, b = OpticalBench(cfg), OpticalBench(cfg)
    phases = np.linspace(-np.pi, np.pi, cfg.input_regions)
    frames = np.stack([a.frame(p, .01, 1., capture_index=i)
                       for p, i in [(np.zeros(16), 0), (phases, 1000), (phases, 1001)]])
    expected = np.stack([b.frame(p, .01, 1., capture_index=i)
                       for p, i in [(np.zeros(16), 0), (phases, 1000), (phases, 1001)]])
    np.testing.assert_array_equal(frames, expected)
    assert a.metadata["matrix_mode"] == "seeded_random"
    assert a.metadata["matrix_fingerprint"] == b.metadata["matrix_fingerprint"]


@pytest.mark.parametrize("phases", [np.zeros(16), np.linspace(-1.,2.,16),
                                   np.tile([0.,np.pi/2,np.pi,3*np.pi/2],4)])
def test_known_coefficients_follow_independent_coherent_camera_calculation(phases):
    cfg = known_config()
    values = json.loads(Path(cfg.detector_matrix_path).read_text())
    expected = []
    # Independent scalar complex arithmetic, not the simulator camera function.
    for real, imag in zip(values["real"], values["imag"]):
        field = sum(complex(a, b) * cmath.exp(1j * phase) for a, b, phase in zip(real, imag, phases))
        expected.append(min(255, max(0, round(40 * abs(field)**2))))
    bench = OpticalBench(cfg)
    frame = bench.frame(phases, .01, 1.)
    readings, _ = extract_detectors(frame, bench.regions)
    np.testing.assert_array_equal(readings, expected)
    random = OpticalBench(BenchConfig()).frame(phases, .01, 1.)
    outside = np.ones(frame.shape, dtype=bool)
    for counts, (y0, y1, x0, x1) in zip(expected, bench.regions):
        assert (frame[y0:y1, x0:x1] == counts).all()
        outside[y0:y1, x0:x1] = False
    np.testing.assert_array_equal(frame[outside], random[outside])


def test_identity_phase_only_inputs_have_constant_detector_readings():
    cfg = BenchConfig.model_validate_json((CONFIGS / "bench-identity.json").read_text())
    cfg = pin_detector_matrix(cfg, base_directory=CONFIGS)
    bench = OpticalBench(cfg)
    for phases in np.random.default_rng(81).uniform(-np.pi, np.pi, (9, 16)):
        readings, _ = extract_detectors(bench.frame(phases, .01, 1.), bench.regions)
        np.testing.assert_array_equal(readings, [40] * 16)


@pytest.mark.parametrize("content,match", [
    ("not-json", "Invalid detector matrix JSON"),
    ('{"real": [], "real": [], "imag": []}', "Invalid detector matrix JSON"),
    ('{"real": []}', "exactly 'real' and 'imag'"),
    (json.dumps({"real": [[1]*4]*3, "imag": [[0]*16]*16}), "shape"),
    (json.dumps({"real": [[1]*16]*16, "imag": [[0]*3]*4}), "shape"),
    (json.dumps({"real": [[True]*16]*16, "imag": [[0]*16]*16}), "no strings or booleans"),
    (json.dumps({"real": [["1"]*16]*16, "imag": [[0]*16]*16}), "no strings or booleans"),
    (json.dumps({"real": [[float('nan')]*16]*16, "imag": [[0]*16]*16}), "finite"),
    (json.dumps({"real": [[1]*16]*16, "imag": [[float('inf')]*16]*16}), "finite"),
])
def test_malformed_matrix_rejected_before_backend_acquisition(tmp_path, content, match):
    path = tmp_path / "matrix.json"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        SimulatedBackend(BenchConfig(detector_matrix_path=str(path)))


def test_missing_and_wrong_input_count_fail_before_backend_acquisition(tmp_path):
    with pytest.raises(ValueError, match="Cannot read detector matrix file"):
        SimulatedBackend(BenchConfig(detector_matrix_path=str(tmp_path / "missing.json")))
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"real": [[1]*4]*4, "imag": [[0]*4]*4}))
    with pytest.raises(ValueError, match=r"shape \[16, 16\]"):
        SimulatedBackend(BenchConfig(detector_matrix_path=str(wrong)))


@pytest.mark.parametrize("test_agents", [True, False])
def test_known_backend_is_shared_by_local_and_live_agent_modes(monkeypatch, test_agents):
    from workflow import llm
    calls = []
    def live_models(config):
        calls.append(config)
        return Integrations()
    monkeypatch.setattr(llm, "build_integrations", live_models)
    cfg = AppConfig(hardware_backend="simulated", optical_bench=known_config(), bench_test_agents=test_agents)
    services = build_integrations(cfg)
    assert services.hardware_backend.config == cfg.optical_bench
    assert len(calls) == (0 if test_agents else 1)
    assert '"real"' not in setup_context(cfg.optical_bench)
    assert '"imag"' not in setup_context(cfg.optical_bench)


@pytest.mark.parametrize("mutation", ["change", "delete"])
def test_matrix_mutation_before_approval_leaves_workflow_unexecuted(tmp_path, monkeypatch, capsys, mutation):
    matrix = tmp_path / "matrix.json"
    matrix.write_bytes((CONFIGS / "matrices/complex-mixing-16x16.json").read_bytes())
    config_file = tmp_path / "bench.json"
    config_file.write_text(json.dumps({"detector_matrix_path": "matrix.json"}))
    cfg = AppConfig(workflow_database_path=tmp_path / "workflow.sqlite", run_output_path=tmp_path / "runs")
    monkeypatch.setattr(AppConfig, "from_env", lambda: cfg.model_copy(deep=True))
    assert cli.main(["start", "--simulate", "--test-agents", "--bench-config", str(config_file)]) == 0
    path = next((tmp_path / "runs").glob("*/workflow.json"))
    waiting = json.loads(path.read_text())
    saved = waiting["state"]["control"]["configuration"]["optical_bench"]
    assert saved["detector_matrix_path"] == str(matrix.resolve())
    assert saved["detector_matrix_sha256"] == hashlib.sha256(matrix.read_bytes()).hexdigest()
    if mutation == "change":
        matrix.write_text(matrix.read_text() + "\n")
    else:
        matrix.unlink()
    args = ["review", waiting["workflow_id"], "approve", "--simulate", "--fingerprint",
            waiting["pending"]["review"]["review_fingerprint"], "--reviewer", "Local verification"]
    assert cli.main(args) == 2
    error = "changed since workflow setup" if mutation == "change" else "Cannot read detector matrix file"
    assert error in capsys.readouterr().out
    with WorkflowApp(cfg) as app:
        unchanged = app.show(waiting["workflow_id"])
        assert unchanged["status"] == "awaiting_approval"
        assert unchanged["state"]["acquisitions"] == []
        assert unchanged["state"]["approvals"] == []
        with pytest.raises(ValueError, match=error):
            app.submit(waiting["workflow_id"], {"action": "approve",
                "review_fingerprint": waiting["pending"]["review"]["review_fingerprint"]})
        with pytest.raises(ValueError, match=error):
            app.resume(waiting["workflow_id"])
    assert cli.main(["cancel", waiting["workflow_id"]]) == 0
    assert json.loads(path.read_text())["status"] == "cancelled"


def test_cached_camera_and_backend_reject_a_changed_source(tmp_path):
    matrix = tmp_path / "matrix.json"
    matrix.write_bytes((CONFIGS / "matrices/complex-mixing-16x16.json").read_bytes())
    cfg = BenchConfig(detector_matrix_path=str(matrix))
    camera = OpticalBench(cfg)
    backend = SimulatedBackend(cfg)
    original_fingerprint = camera.metadata["matrix_fingerprint"]
    matrix.write_bytes((CONFIGS / "matrices/identity-16x16.json").read_bytes())
    with pytest.raises(ValueError, match="changed since workflow setup"):
        camera.frame(np.zeros(16), .01, 1.)
    with pytest.raises(ValueError, match="changed since workflow setup"):
        backend.describe()
    assert camera.metadata["matrix_fingerprint"] == original_fingerprint


def test_known_unreachable_calibration_keeps_diagnostics_without_measurements(tmp_path, monkeypatch):
    config_file = tmp_path / "bench.json"
    config_file.write_text(json.dumps({        "detector_matrix_path": str(CONFIGS / "matrices/identity-16x16.json"),
        "shutter_seconds": .0005, "shutter_max": .001, "power_max": 1.}))
    cfg = AppConfig(workflow_database_path=tmp_path / "workflow.sqlite", run_output_path=tmp_path / "runs")
    monkeypatch.setattr(AppConfig, "from_env", lambda: cfg.model_copy(deep=True))
    assert cli.main(["start", "--simulate", "--test-agents", "--bench-config", str(config_file)]) == 0
    path = next((tmp_path / "runs").glob("*/workflow.json"))
    waiting = json.loads(path.read_text())
    assert cli.main(["review", waiting["workflow_id"], "approve", "--simulate", "--fingerprint",
                     waiting["pending"]["review"]["review_fingerprint"]]) == 2
    stopped = json.loads(path.read_text())
    acquisition = stopped["state"]["acquisitions"][0]
    assert "unreachable" in acquisition["errors"][0]
    assert not stopped["state"]["analyses"]
    assert not list(path.parent.rglob("measurements.npz"))
    diagnostics = json.loads(next(path.parent.rglob("calibration.json")).read_text())
    assert not diagnostics["success"] and diagnostics["attempts"]
    assert (path.parent / "bench-metadata.json").is_file()


def test_known_workflow_retrieves_only_saved_measurements_and_records_identity(tmp_path, monkeypatch):
    matrix = tmp_path / "matrix.json"
    matrix.write_bytes((CONFIGS / "matrices/complex-mixing-16x16.json").read_bytes())
    config_file = tmp_path / "bench.json"
    config_file.write_text(json.dumps({"detector_matrix_path": "matrix.json"}))
    cfg = AppConfig(workflow_database_path=tmp_path / "workflow.sqlite", run_output_path=tmp_path / "runs")
    monkeypatch.setattr(AppConfig, "from_env", lambda: cfg.model_copy(deep=True))
    assert cli.main(["start", "--simulate", "--test-agents", "--bench-config", str(config_file)]) == 0
    path = next((tmp_path / "runs").glob("*/workflow.json"))
    waiting = json.loads(path.read_text())
    metadata = json.loads((path.parent / "bench-metadata.json").read_text())
    assert metadata["matrix_mode"] == "known" and len(metadata["matrix_fingerprint"]) == 64
    assert cli.main(["review", waiting["workflow_id"], "approve", "--simulate", "--fingerprint",
                     waiting["pending"]["review"]["review_fingerprint"]]) == 0
    final = json.loads(path.read_text())
    assert final["status"] == "completed"
    assert final["state"]["analyses"][-1]["metrics"]["prediction_passed"] == 1
    assert json.loads((path.parent / "bench-metadata.json").read_text()) == metadata
    acquisition = final["state"]["acquisitions"][0]
    manifest_path = next(a["path"] for a in acquisition["raw_data"] if a["path"].endswith("bench-dataset.json"))
    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest["matrix_fingerprint"] == metadata["matrix_fingerprint"]
    with np.load(manifest["arrays"]["path"], allow_pickle=False) as arrays:
        assert set(arrays.files) == {"phases", "frames", "pattern_ids", "detector_readings",
                                    "detector_saturation", "whole_frame_saturation"}
    matrix.unlink()  # Analysis-only reuse must not consult the truth source.
    assert cli.main(["start", "--reuse-data-from", waiting["workflow_id"]]) == 0
