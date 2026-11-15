"""Recovery evidence is not a flat-signal classifier; all models here are doubles."""

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from analysis.recovery import measurement_diagnostics
from bench.config import BenchConfig
from bench.workflow import build_integrations, setup_context
from fixtures.offline import LocalAnalyst, LocalRouter, LocalScientist, offline_integrations
from tools.hardware import ParameterSpec
from tools.python_execution import fingerprint
from workflow.config import AppConfig
from workflow.graph import Integrations, WorkflowApp
from workflow.report import export_decisions
from workflow.state import (AcquisitionResult, Artifact, ExperimentProcedure, ExperimentProposal,
                            NumericalAnalysisResult, Setting, WorkflowState)


class ObservingAnalyst(LocalAnalyst):
    def __init__(self, goal="input_mapping", supported=False):
        self.task_goal, self.supported, self.calls = goal, supported, []

    def invoke(self, messages):
        self.calls.append(json.loads(messages[1][1]))
        result = super().invoke(messages)
        if self.supported and result.get("recovery"):
            result["recovery"]["input_mapping"]["reason"] = "Two unequal input strengths can be exchanged while retaining the same phase-only intensity curve; varying brightness alone does not distinguish them"
            result["recovery"]["follow_up"]["required_actions"] = ["fixture.capture"]
            result["recovery"]["follow_up"]["required_parameters"] = {"fixture.capture": ["input_amplitudes"]}
        return result


@pytest.mark.parametrize("goal", ["input_mapping", "brightness_prediction"])
def test_identity_predictions_pass_but_mapping_unresolved_without_amplitude_control(tmp_path, monkeypatch, goal):
    # Actual model/camera/acquisition/retrieval/export path; only the fit is a
    # deliberate counterexample: every detector assigned to input 6 (index 5) predicts 40.
    def monomial_fit(X, intensities, settings, **kwargs):
        T = np.zeros((1, X.shape[1]), complex)
        T[0, 5] = np.sqrt(40)
        return {"ggs": {"T": T, "fit_mse": 0., "runtime_seconds": 0.}}
    monkeypatch.setattr("analysis.bench_analysis.fit_methods", monomial_fit)
    matrix = Path(__file__).resolve().parents[1] / "configs/matrices/identity-16x16.json"
    bench = BenchConfig(detector_matrix_path=str(matrix), fit_patterns=128, holdout_patterns=16)
    cfg = AppConfig(hardware_backend="simulated", optical_bench=bench, bench_test_agents=True,
        reviewed_numerics_enabled=True, llm_router_enabled=True, max_workflow_iterations=40,
        numerical_input_path=tmp_path / "raw", hardware_input_path=tmp_path / "raw",
        run_output_path=tmp_path / "runs", workflow_database_path=tmp_path / "flow.sqlite",
        max_acquisition_measurements=bench.frame_budget)
    services = build_integrations(cfg)
    model = ObservingAnalyst(goal)
    services.analyst_model = model
    architecture = tmp_path / "setup.txt"
    architecture.write_text(setup_context(bench), encoding="utf-8")
    question = "Recover the input-to-detector mapping" if goal == "input_mapping" else "Only predict unseen brightness"
    with WorkflowApp(cfg, services) as app:
        waiting = app.start(question, architecture_path=architecture)
        assert waiting["status"] == "awaiting_approval" and not waiting["state"]["acquisitions"]
        final = app.submit(waiting["workflow_id"], {"action": "approve",
            "review_fingerprint": waiting["pending"]["review"]["review_fingerprint"]})
    assert final["status"] == "completed", final.get("reason")
    assert len(final["state"]["acquisitions"]) == 1
    analysis = final["state"]["analyses"][-1]
    assert analysis["metrics"]["prediction_passed"] == 1
    assert all(analysis["metrics"][f"detector_{j}.holdout_rmse"] == 0 for j in range(16))
    assert all(analysis["metrics"][f"detector_{j}.fit_eligible"] == analysis["metrics"][f"detector_{j}.identifiable"] == 1 for j in range(16))
    payload = model.calls[-1]
    diagnostics = payload["recovery_evidence"][-1]["measurements"]
    assert len(diagnostics["per_detector"]) == 16
    assert all(d["mean"] == 40 and d["std"] == 0 and d["clipped"] == 0 for d in diagnostics["per_detector"])
    review = final["state"]["analyst_reviews"][-1]
    assert review["recovery"]["brightness_prediction"]["status"] == "supported"
    assert review["recovery"]["input_mapping"]["status"] == "unresolved"
    assert review["recovery"]["input_mapping"]["affected_detectors"] == list(range(16))
    if goal == "input_mapping":
        assert any("input_amplitudes" in note for note in review["limitations"])
        assert "global exposure/power" in review["recovery"]["follow_up"]["distinguishing_result"]
    else:
        assert review["recovery"]["follow_up"] is None
    assert not any(d["action"] == "plan_follow_up" for d in final["state"]["router_decisions"])
    assert not any("Waiting for human input:" in c["message"] for c in final["state"]["conversation"])
    export_decisions(final)
    quality = json.loads((Path(final["state"]["control"]["configuration"]["run_output_path"]) / "quality.json").read_text())
    assert quality["recovery"]["input_mapping"]["status"] == "unresolved"
    assert "real" not in json.dumps(payload["recovery_evidence"]) and '"T"' not in json.dumps(payload)
    estimate = next(a["path"] for a in analysis["outputs"] if a["path"].endswith("estimated-tm.npz"))
    with np.load(estimate) as saved:
        assert np.array_equal(saved["fit_eligible"], saved["identifiable"])
        assert np.all(np.argmax(np.abs(saved["T"]), axis=1) == 5)


def varying_case(tmp_path):
    X = np.exp(1j * np.random.default_rng(81).uniform(-np.pi, np.pi, (64, 2)))
    # Independently computed ambiguity, only in test code: [2,1] and [1,2]
    # both give 5 + 4*cos(delta), despite nonconstant readings and full rank.
    values = (5 + 4 * np.cos(np.angle(X[:, 0]) - np.angle(X[:, 1])))[:, None]
    assert np.allclose(values, np.abs(X @ np.array([[2., 1.]]).T)**2)
    assert np.allclose(values, np.abs(X @ np.array([[1., 2.]]).T)**2)
    cfg = AppConfig(hardware_backend="test", llm_router_enabled=True, max_workflow_iterations=40,
        numerical_input_path=tmp_path / "raw", hardware_input_path=tmp_path / "hardware",
        run_output_path=tmp_path / "runs", workflow_database_path=tmp_path / "flow.sqlite")
    cfg.numerical_input_path.mkdir()
    def artifact(path):
        return Artifact(path=str(path), description="Measured intensity fixture, not hidden coefficients", sha256=fingerprint(path))
    ids = np.array([f"pattern-{i}" for i in range(len(X))])
    array_path = cfg.numerical_input_path / "data.npz"
    np.savez(array_path, X=X, intensities=values, pattern_ids=ids, measurement_pattern_ids=ids, frame_ids=ids)
    arrays = artifact(array_path)
    manifest_path = cfg.numerical_input_path / "dataset.json"
    manifest_path.write_text(json.dumps({"version": 1, "arrays": arrays.model_dump(mode="json"),
        "data_origin": "test_only", "provenance": "Analytic two-input varying-intensity fixture",
        "fit_indices": list(range(48)), "holdout_indices": list(range(48, 64))}))
    manifest = artifact(manifest_path)
    proposal = ExperimentProposal(scientific_question="Recover the input mapping", hypothesis="A discriminating measurement separates two assignments",
        rationale="Prediction alone cannot distinguish alternatives", variables={"inputs": "Equal-amplitude phases"},
        controls=["Fixed exposure"], required_measurements=["Intensities"], evaluation_criteria=["Resolve mapping"])
    proc = ExperimentProcedure(proposal_id=proposal.id, version=1, actions=[{"action": "fixture.capture", "instruction": "Prior phase-only measurements"}],
                               prerequisites=["fixture setup verified"], expected_outputs=["Saved data"])
    acquisition = AcquisitionResult(procedure_id=proc.id, procedure_version=1, procedure_fingerprint=proc.fingerprint(),
        status="completed", data_origin="test_only", actual_settings={"exposure": Setting(value=1)}, raw_data=[manifest, arrays])
    analysis = NumericalAnalysisResult(acquisition_ids=[acquisition.id], status="completed", input_data=[manifest, arrays],
        code=Artifact(path="fixture.py", description="Not executed"), code_version="fixture", metrics={"prediction_passed": 1},
        method_settings={"operation": "tm.compare_reference_free"})
    state = WorkflowState(user_request="Recover the input mapping", proposals=[proposal], procedures=[proc], acquisitions=[acquisition], analyses=[analysis])
    state.control.active_proposal_ids = [proposal.id]
    state.control.acquisition_attempts = 1
    services = Integrations(**offline_integrations(cfg, state.id, include_analyst=True), router_model=LocalRouter())
    services.scientist_model = LocalScientist()
    architecture = tmp_path / "setup.txt"
    architecture.write_text("TEST ONLY intensity acquisition. No hidden matrix supplied.")
    return X, values, cfg, services, state, architecture


@pytest.mark.parametrize("supported", [False, True])
def test_varying_readings_leave_mapping_open_and_supported_follow_up_reaches_planning(tmp_path, supported):
    X, values, cfg, services, state, architecture = varying_case(tmp_path)
    assert np.ptp(values) > 7 and np.linalg.matrix_rank(X) == 2
    model = ObservingAnalyst(supported=supported)
    services.analyst_model = model
    if supported:
        # Only this test backend gains a capability. No simulator control added.
        services.hardware_backend.capabilities.actions["fixture.capture"].parameters["input_amplitudes"] = ParameterSpec(
            kind="string", required=False, verified=True, choices=["one-input-at-a-time"])
        original = services.experimentor_model
        class Planner:
            def with_structured_output(self, schema):
                original.with_structured_output(schema)
                return self
            def invoke(self, messages):
                result = original.invoke(messages)
                result["procedure"]["actions"][0]["requested_settings"]["input_amplitudes"] = {"value": "one-input-at-a-time"}
                return result
        services.experimentor_model = Planner()
    with WorkflowApp(cfg, services) as app:
        final = app.start(state.user_request, initial_state=state, architecture_path=architecture)
        if supported:
            assert final["status"] == "awaiting_approval", final.get("reason")
            assert any(d["action"] == "plan_follow_up" for d in final["state"]["router_decisions"])
            assert final["state"]["procedures"][-1]["version"] == 2
            assert final["state"]["procedures"][-1]["actions"][0]["requested_settings"]["input_amplitudes"]["value"] == "one-input-at-a-time"
            assert final["state"]["acquisition_reviews"][-1]["status"] == "ready"
            rejected = app.submit(final["workflow_id"], {"action": "approve", "review_fingerprint": "wrong"})
            assert rejected["input_error"]
        else:
            assert final["status"] == "completed", final.get("reason")
            assert final["state"]["analyst_reviews"][-1]["recovery"]["input_mapping"]["status"] == "unresolved"
    assert not services.hardware_backend.execute_calls
    diag = model.calls[-1]["recovery_evidence"][0]["measurements"]
    assert diag["per_detector"][0]["std"] > 2 and diag["input_pattern_rank"] == 2
    assert diag["per_detector"][0]["clipped"] is None  # generic manifest has no clipping metadata


def test_noise_quantization_clipping_and_weak_signal_are_diagnostics():
    X = np.exp(1j * np.random.default_rng(4).uniform(size=(8, 2)))
    values = np.array([[40., .01, 255.], [41., .02, 20.]] * 4)
    clipped = values == 255
    result = measurement_diagnostics(X, values, clipped, noise_std=[2., 1., 0.], quantization_counts=1., background=0.)
    first, weak, censored = result["per_detector"]
    assert first["low_variation"] and first["noise_std"] == 2
    assert weak["weak_signal"] and censored["clipped"] == 4 and censored["unclipped"] == 4
    assert "not proof" in result["interpretation"]
