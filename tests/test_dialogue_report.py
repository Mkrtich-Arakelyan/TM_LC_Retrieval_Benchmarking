"""Verbatim model handoffs and human-only numerical inspection exports."""

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

import main
from analysis.run_inspection import matrix_comparison, mean_baseline, verified_truth
from bench.config import BenchConfig
from bench.matrices import pin_detector_matrix
from bench.optical_bench import OpticalBench
from fixtures.offline import FixtureAnalystModel, FixtureExperimentorModel, FixtureScientistModel, offline_integrations
from workflow.config import AppConfig
from workflow.graph import Integrations, WorkflowApp
from workflow.report import export_decisions, export_workflow_view, regenerate_report
from workflow.state import WorkflowState

SCIENCE = "  Experimentalist—why this split?!\r\nKeep 17% held out; don't resample.\n  Answer the Analyst's recommendation, please…  "
PROCEDURE = "Analyst: here is my answer—fixed settings!\nCheck clipping first; then assess mapping.\nIs that distinction supported?"
REVIEW = "Scientist, prediction ≠ recovery.\nThe mapping remains unresolved: A → B?!\nPlease retain this limitation."


def workflow(tmp_path):
    cfg = AppConfig(workflow_database_path=tmp_path / "flow.sqlite", run_output_path=tmp_path / "runs",
        hardware_input_path=tmp_path / "raw", numerical_input_path=tmp_path / "inputs",
        hardware_backend="test", numerical_backend="test", max_workflow_iterations=40)
    services = Integrations(**offline_integrations(cfg, uuid4(), include_analyst=True))
    calls = {"scientist": [], "experimentor": [], "analyst": []}
    class Scientist(FixtureScientistModel):
        def invoke(self, messages):
            payload = json.loads(messages[1][1]); calls["scientist"].append(payload)
            result = super().invoke(messages)
            if result["kind"] == "proposal":
                result["handoff"] = {"recipient": "experimentor", "message": SCIENCE, "reply_to": None}
            return result
    class Experimentalist(FixtureExperimentorModel):
        def invoke(self, messages):
            payload = json.loads(messages[1][1]); calls["experimentor"].append(payload)
            result = super().invoke(messages)
            if result.get("kind") == "procedure":
                incoming = next(m for m in payload["handoffs"] if m["message"] == SCIENCE)
                result["handoff"] = {"recipient": "analyst", "message": PROCEDURE, "reply_to": incoming["id"]}
            return result
    class Analyst(FixtureAnalystModel):
        def invoke(self, messages):
            payload = json.loads(messages[1][1]); calls["analyst"].append(payload)
            result = super().invoke(messages)
            if payload["stage"] == "after_analysis":
                incoming = next(m for m in payload["handoffs"] if m["message"] == PROCEDURE)
                result["handoff"] = {"recipient": "scientist", "message": REVIEW, "reply_to": incoming["id"]}
            return result
    services.scientist_model = Scientist()
    services.experimentor_model = Experimentalist(services.hardware_backend)
    services.analyst_model = Analyst()
    return cfg, services, calls


def test_handoffs_saved_prompted_and_exported_verbatim_with_replies_and_approval(tmp_path):
    cfg, services, calls = workflow(tmp_path)
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Check exact coordination text")
        assert waiting["status"] == "awaiting_approval" and not services.hardware_backend.execute_calls
        invalid = app.submit(waiting["workflow_id"], {"action": "approve", "review_fingerprint": "wrong"})
        assert invalid["input_error"] and not services.hardware_backend.execute_calls
    with WorkflowApp(cfg, services) as app:
        final = app.submit(waiting["workflow_id"], {"action": "approve", "review_fingerprint": waiting["pending"]["review"]["review_fingerprint"]})
    assert final["status"] == "completed"
    count_before = {key: len(value) for key, value in calls.items()}
    entries = {entry["message"]: entry for entry in final["state"]["conversation"] if entry["kind"] == "handoff"}
    assert entries[PROCEDURE]["reply_to"] == entries[SCIENCE]["id"]
    assert entries[REVIEW]["reply_to"] == entries[PROCEDURE]["id"]
    assert calls["experimentor"][0]["handoffs"][0]["message"] == SCIENCE
    assert any(m["message"] == PROCEDURE for m in calls["analyst"][-1]["handoffs"])
    assert any(m["message"] == REVIEW for m in calls["scientist"][-1]["handoffs"])
    root = tmp_path / "separate-report"
    export_workflow_view(final, root); export_decisions(final, root)
    text = (root / "conversation.md").read_bytes().decode("utf-8")
    assert all(message in text for message in (SCIENCE, PROCEDURE, REVIEW))
    assert text.index(SCIENCE) < text.index(PROCEDURE) < text.index(REVIEW)
    assert "Scientist → Experimentalist" in text and "Replies to:" in text
    assert "execution event (program-generated)" in text
    assert "Workflow step " not in text and '"stage":' not in text
    assert '"stage":' in (root / "agent-messages.md").read_text(encoding="utf-8")
    assert count_before == {key: len(value) for key, value in calls.items()}  # No export model calls.
    assert "Matrix recovery: mapping **unresolved**" in (root / "run-summary.md").read_text(encoding="utf-8")


def test_legacy_run_loads_without_inventing_handoffs(tmp_path):
    cfg, services, _ = workflow(tmp_path)
    with WorkflowApp(cfg, services) as app:
        view = app.start("Legacy review")
    view["state"]["conversation"] = [entry for entry in view["state"]["conversation"] if entry["kind"] != "handoff"]
    for entry in view["state"]["conversation"]:
        for key in ("kind", "recipient", "reply_to"):
            entry.pop(key, None)
    for review in view["state"]["analyst_reviews"]:
        review.pop("handoff", None)
    restored = WorkflowState.model_validate(view["state"])
    assert all(entry.kind == "legacy" and entry.recipient is None for entry in restored.conversation)
    source = tmp_path / "old-workflow.json"
    source.write_text(json.dumps(view), encoding="utf-8")
    before = source.read_bytes()
    root = regenerate_report(source, tmp_path / "legacy-report")
    assert "legacy recorded message" in (root / "conversation.md").read_text(encoding="utf-8")
    assert SCIENCE not in (root / "conversation.md").read_text(encoding="utf-8")
    assert source.read_bytes() == before
    assert "Matrix recovery: **not assessed**" in (root / "run-summary.md").read_text(encoding="utf-8")


def test_fitting_mean_baseline_uses_fit_only_and_identical_heldout_mask():
    readings = np.array([[10., 20.], [20., 30.], [255., 40.], [15., 31.], [255., 33.], [17., 255.]])
    prediction = np.array([[15., 30.], [255., 32.], [16., 254.]])
    result = mean_baseline(readings, [0, 1, 2], [3, 4, 5], prediction, [True, True], readings == 255)
    assert result["per_detector"][0]["fitting_mean"] == 15
    assert result["usable"] == 4 and result["clipped"] == 2
    assert result["fit_rmse"] == np.sqrt(.75) and result["mean_baseline_rmse"] == np.sqrt(3.5)
    tied = mean_baseline(np.full((6, 2), 40.), [0, 1, 2], [3, 4, 5], np.full((3, 2), 40.), [True, True])
    assert tied["fit_rmse"] == tied["mean_baseline_rmse"] == 0 and tied["result"] == "tie"


def test_truth_requires_the_recorded_fingerprint_and_source_integrity(tmp_path):
    source = tmp_path / "identity.json"
    source.write_text(json.dumps({"real": np.eye(16).tolist(), "imag": np.zeros((16, 16)).tolist()}))
    config = pin_detector_matrix(BenchConfig(detector_matrix_path=str(source)))
    manifest = {"data_origin": "simulated", "config": config.model_dump(), **OpticalBench(config).metadata}
    np.testing.assert_array_equal(verified_truth(manifest), np.eye(16))
    with pytest.raises(ValueError, match="fingerprint"):
        verified_truth({**manifest, "matrix_fingerprint": "0"*64})
    source.write_text(source.read_text() + " ")
    with pytest.raises(ValueError, match="changed since"):
        verified_truth(manifest)
    random = BenchConfig(seed=81)
    random_manifest = {"data_origin": "simulated", "config": random.model_dump(), **OpticalBench(random).metadata}
    assert verified_truth(random_manifest).shape == (16, 16)


@pytest.mark.parametrize("candidates", [False, True])
def test_varying_data_no_truth_and_candidate_labels(tmp_path, candidates, monkeypatch):
    from test_recovery import varying_case
    _, readings, cfg, _, state, _ = varying_case(tmp_path)
    state.control.configuration = cfg.model_dump(mode="json")
    directory = tmp_path / "derived"
    directory.mkdir()
    matrices = {"ggs": np.array([[2., 1.]])}
    if candidates:
        matrices["intensity_ls"] = np.array([[1., 2.]])
    for name, matrix in matrices.items():
        path = directory / f"{name}.npz"
        np.savez(path, T=matrix.astype(complex), holdout_prediction=readings[48:])
        from workflow.state import Artifact
        state.analyses[0].outputs.append(Artifact(path=str(path), description="Candidate estimate", sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    view = {"workflow_id": str(state.id), "status": "completed", "state": state.model_dump(mode="json"), "pending": None, "reason": None}
    source = tmp_path / "view.json"; source.write_text(json.dumps(view), encoding="utf-8")
    root = tmp_path / "report"
    monkeypatch.setattr(AppConfig, "from_env", lambda: pytest.fail("Report generation must not load API credentials"))
    assert main.main(["report", str(source), "--output", str(root)]) == 0
    inspection = json.loads((root / "matrix-inspection.json").read_text())
    for estimate in inspection["estimates"]:
        assert estimate["truth"] == "unavailable" and estimate["baseline"]["result"] == "fit better"
        assert Path(estimate["heatmap"]).is_file() and Path(estimate["coefficients_csv"]).is_file()
        assert estimate["label"] == ("Candidate" if candidates else "Estimate")
    summary = (root / "run-summary.md").read_text(encoding="utf-8")
    assert "Matrix recovery: **not assessed**" in summary
    assert "Truth comparison: unavailable" in summary
    assert "<summary>Estimated TM (as saved)" in summary
    assert "<summary>True TM" not in summary
    assert "row-phase-aligned relative matrix error" not in summary
    assert_embedded_images(root, summary)


def test_stopped_report_has_last_stage_and_recorded_reason(tmp_path):
    cfg, services, _ = workflow(tmp_path)
    with WorkflowApp(cfg, services) as app:
        view = app.start("Do not execute yet")
    view["status"] = "stopped"; view["pending"] = None; view["reason"] = "Stopped by recorded limit: no acquisition permitted."
    root = tmp_path / "stopped-report"
    export_workflow_view(view, root); export_decisions(view, root)
    text = (root / "run-summary.md").read_text(encoding="utf-8")
    assert "Workflow: **stopped**" in text and view["reason"] in text
    assert "Runner preflight/review recorded; no acquisition executed" in text
    assert "No available saved matrix estimate" in text


def assert_embedded_images(root, summary):
    import re
    from urllib.parse import unquote, urlparse
    from PIL import Image
    paths = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", summary)
    assert paths
    for path in paths:
        assert not urlparse(path).scheme and not Path(unquote(path)).is_absolute()
        with Image.open(root / unquote(path)) as image:
            image.verify()  # Resolve/decode the same relative URL a Markdown viewer uses.


def identity_report_view(tmp_path, estimate):
    """A registered identity measurement fixture; report inputs contain no truth T."""
    from test_recovery import varying_case
    from bench.geometry import geometry_metadata
    from tools.python_execution import fingerprint
    from workflow.state import Artifact
    _, _, cfg, _, state, _ = varying_case(tmp_path)
    config = pin_detector_matrix(BenchConfig(fit_patterns=128, holdout_patterns=8,
        detector_matrix_path=str(Path(__file__).resolve().parents[1] / "configs/matrices/identity-16x16.json")))
    geometry = geometry_metadata()
    phases = np.random.default_rng(9).uniform(-np.pi, np.pi, (136, 16))
    frames = np.zeros((136, 20, 20), dtype=np.uint8)
    for y0, y1, x0, x1 in geometry["regions"]:
        frames[:, y0:y1, x0:x1] = 40
    path = cfg.numerical_input_path / "identity-measurements.npz"
    ids = [f"fit-{i:04}" for i in range(128)] + [f"holdout-{i:04}" for i in range(8)]
    np.savez(path, phases=phases, frames=frames, pattern_ids=np.array(ids),
        detector_readings=np.full((136, 16), 40.), detector_saturation=np.zeros((136, 16)),
        whole_frame_saturation=np.zeros(136))
    arrays = Artifact(path=str(path), description="Saved measured identity frames", sha256=fingerprint(path))
    manifest = {"kind": "synthetic-bench-v1", "data_origin": "simulated", **OpticalBench(config).metadata,
        "config": config.model_dump(), "arrays": arrays.model_dump(mode="json"),
        "fit_patterns": 128, "holdout_patterns": 8, "shutter_seconds": .01, "laser_power_mw": 1.,
        "background_counts": 0., "read_noise_std": 0., "camera_gain": 4000.}
    path = cfg.numerical_input_path / "identity-dataset.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    recorded = Artifact(path=str(path), description="Saved identity manifest", sha256=fingerprint(path))
    state.acquisitions[0].raw_data = [recorded, arrays]
    state.acquisitions[0].data_origin = "simulated"
    analysis = state.analyses[0]
    analysis.input_data = [recorded, arrays]
    analysis.method_settings = {"operation": "bench.retrieve"}
    analysis.metrics = {"prediction_passed": 1, "measurement_passed": 1}
    path = tmp_path / "estimated-tm.npz"
    np.savez(path, T=estimate, holdout_prediction=np.full((8, 16), 40.), fit_eligible=np.ones(16, bool))
    analysis.outputs = [Artifact(path=str(path), description="Saved estimate", sha256=fingerprint(path))]
    state.control.configuration = cfg.model_dump(mode="json")
    return {"workflow_id": str(state.id), "status": "completed", "pending": None, "reason": None,
            "state": state.model_dump(mode="json")}


@pytest.mark.parametrize("phase_equivalent", [False, True])
def test_identity_truth_comparison_embedded_next_to_zero_prediction_error(tmp_path, phase_equivalent):
    if phase_equivalent:
        estimate = np.eye(16, dtype=complex) * np.exp(1j*np.linspace(-2., 2., 16))[:, None]
    else:
        estimate = np.zeros((16, 16), complex)
        estimate[:, 5] = 1  # Every detector assigned to input 6.
    view = identity_report_view(tmp_path, estimate)
    original = json.dumps(view, sort_keys=True)
    source = tmp_path / "saved-view.json"
    source.write_text(json.dumps(view), encoding="utf-8")
    root = regenerate_report(source, tmp_path / "regenerated")
    text = (root / "run-summary.md").read_text(encoding="utf-8")
    result = json.loads((root / "matrix-inspection.json").read_text())["estimates"][0]
    comparison = result["matrix_comparison"]
    assert comparison["strongest_input_agreements"] == (16 if phase_equivalent else 1)
    assert comparison["ambiguous_rows"] == []
    if phase_equivalent:
        assert comparison["row_phase_aligned_relative_error"] < 1e-14
        np.testing.assert_array_equal(result["estimated_coefficients"]["imag"], estimate.imag)
    else:
        assert comparison["row_phase_aligned_relative_error"] == pytest.approx(np.sqrt(30)/4)
        assert comparison["strongest_input_mismatches"] == [j for j in range(16) if j != 5]
        assert "**1/16 detector rows**" in text
        assert "D0 (true input 1, estimated input 6)" in text
    assert "Held-out brightness prediction status (recorded): **pass**" in text
    assert "| Matrix vs verified simulator truth (human-only) | Held-out brightness prediction |" in text
    assert "Held-out fit RMSE **0.0**, fitting-mean baseline RMSE **0.0**; **tie**" in text
    assert "identity-16x16.json" in text
    assert "<summary>True TM (fingerprint verified)" in text and "<summary>Estimated TM (as saved)" in text
    assert "| Camera detector / SLM input | Input 1 |" in text
    assert "separate recorded Analyst recovery assessment" in text
    assert "Matrix recovery: **not assessed**" in text  # Truth diagnostics don't invent an agent conclusion.
    assert_embedded_images(root, text)
    assert json.dumps(view, sort_keys=True) == original
    assert "true_coefficients" not in source.read_text()  # Human export only.


def test_strongest_input_ties_near_ties_and_zero_rows_are_ambiguous():
    truth = np.eye(3, dtype=complex)
    estimate = np.array([[1., 1.-5e-7, 0.], [0., 1., 0.], [0., 0., 0.]], dtype=complex)
    result = matrix_comparison(estimate, truth, [0, 1, 2])
    assert result["strongest_input_agreements"] == 1
    assert result["strongest_input_mismatches"] == []
    assert result["ambiguous_rows"] == [0, 2]
    assert result["strongest_inputs"][0]["estimated_inputs"] == [1, 2]
    assert result["strongest_inputs"][2]["estimated_inputs"] == []
    swapped = matrix_comparison(truth, estimate, [0, 1, 2])
    assert swapped["ambiguous_rows"] == [0, 2]  # Ambiguity in the true matrix is also flagged.
