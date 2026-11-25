"""Private simulator sources never become external acquisition artifacts."""

import json
import time
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

import pytest

from agents.common import planning_context, planning_state, read_context
from bench.optical_bench import OpticalBench
from bench.workflow import BENCH_QUESTION, setup_context
from fixtures.offline import LocalExperimentor
from tools.hardware import canonical_hash, preflight
from tools.python_execution import fingerprint
from workflow.graph import WorkflowApp
from workflow.state import Artifact, WorkflowState
from test_procedure_contract import integrations


class MatrixArtifactPlanner(LocalExperimentor):
    def __init__(self, artifact, field="input_artifacts", *, repeat=False, ask_on_repair=False):
        self.artifact, self.field, self.repeat = artifact, field, repeat
        self.ask_on_repair, self.calls = ask_on_repair, []

    def invoke(self, messages):
        payload = json.loads(messages[1][1])
        self.calls.append(payload)
        if len(self.calls) > 1:
            feedback = payload["procedure_validation"]
            assert f"procedure.{self.field}" in " ".join(feedback["errors"])
            assert feedback["artifact_contract"]["input_artifacts"] == []
            assert feedback["artifact_contract"]["calibration_artifacts"] == {}
            if self.ask_on_repair:
                return {"kind": "request_information", "explanation": "Please relocate the matrix file",
                    "questions": ["Can you copy the matrix to data/raw?"], "recommended_next_step": "Ask user",
                    "handoff": {"recipient": "human", "message": "Please supply the matrix path.", "reply_to": None}}
        result = super().invoke(messages)
        if len(self.calls) == 1 or self.repeat:
            result["procedure"]["id"] = str(uuid4())
            result["procedure"][self.field] = ([self.artifact.model_dump(mode="json")] if self.field == "input_artifacts" else
                                             {"matrix-calibration": self.artifact.model_dump(mode="json")})
        else:
            result["procedure"]["input_artifacts"] = []
            result["procedure"]["calibration_artifacts"] = {}
        return result


def bad_artifact(cfg, backend, *, in_root=False):
    from pathlib import Path
    source = Path(backend.config.detector_matrix_path) if backend.config.detector_matrix_path else None
    if in_root or source is None:
        cfg.hardware_input_path.mkdir(parents=True, exist_ok=True)
        path = cfg.hardware_input_path / "matrix-copy.json"
        path.write_bytes(source.read_bytes() if source else b'{"not":"an acquisition input"}')
    else:
        path = source
    return Artifact(path=str(path), description="Pinned simulator matrix source, incorrectly classified as experiment input", sha256=fingerprint(path))


@pytest.mark.parametrize("router", [True, False])
@pytest.mark.parametrize("matrix,field,in_root", [
    ("identity-16x16.json", "input_artifacts", False),
    ("identity-16x16.json", "calibration_artifacts", False),
    ("identity-16x16.json", "input_artifacts", True),
    ("complex-mixing-16x16.json", "input_artifacts", False),
    (None, "input_artifacts", True),
])
def test_bad_matrix_artifact_repairs_to_normal_approval_without_capture(tmp_path, monkeypatch, router, matrix, field, in_root):
    cfg, services, _, backend, architecture = integrations(tmp_path, router=router, matrix_name=matrix)
    model = MatrixArtifactPlanner(bad_artifact(cfg, backend, in_root=in_root), field)
    services.experimentor_model = model
    monkeypatch.setattr(OpticalBench, "frame", lambda *a, **kw: pytest.fail("Camera called before approval"))
    root_before = cfg.hardware_input_path
    with WorkflowApp(cfg, services) as app:
        view = app.start(BENCH_QUESTION, architecture_path=architecture)
        assert view["status"] == "awaiting_approval", view.get("reason")
        assert view["pending"]["kind"] == "review" and len(model.calls) == 2
        state = WorkflowState.model_validate(view["state"])
        assert len(state.procedures) == 1 and state.control.procedure_revisions == 1
        assert state.procedures[0].input_artifacts == [] and state.procedures[0].calibration_artifacts == {}
        assert not state.acquisitions and not state.approvals and not backend.execute_calls
        assert state.acquisition_reviews[-1].procedure_fingerprint == state.procedures[0].fingerprint()
        assert model.calls[0]["procedure_contract"]["artifacts"]["input_artifacts"] == []
        assert model.calls[0]["procedure_contract"]["artifacts"]["calibration_artifacts"] == {}
        assert "detector_matrix_path" not in model.calls[0]["workflow_state"]["control"]["configuration"]["optical_bench"]
        if matrix:
            assert state.control.configuration["optical_bench"]["detector_matrix_path"] == backend.config.detector_matrix_path
        rejected = app.submit(view["workflow_id"], {"action": "approve", "review_fingerprint": "wrong"})
        assert rejected["input_error"] and not backend.execute_calls
    assert cfg.hardware_input_path == root_before  # No widening of file access.


@pytest.mark.parametrize("ask_on_repair", [False, True])
def test_unrepairable_artifact_blocks_internally_not_user_clarification(tmp_path, monkeypatch, ask_on_repair):
    cfg, services, _, backend, architecture = integrations(tmp_path, repairs=1)
    model = MatrixArtifactPlanner(bad_artifact(cfg, backend, in_root=True), repeat=True, ask_on_repair=ask_on_repair)
    services.experimentor_model = model
    monkeypatch.setattr(OpticalBench, "frame", lambda *a, **kw: pytest.fail("Camera must not run"))
    with WorkflowApp(cfg, services) as app:
        view = app.start(BENCH_QUESTION, architecture_path=architecture)
    assert view["status"] == "blocked" and view["pending"] is None
    assert "Procedure validation error" in view["reason"] and "procedure.input_artifacts" in view["reason"]
    assert "budget exhausted" in view["reason"] and len(model.calls) == 2
    assert not view["state"]["procedures"] and not view["state"]["acquisitions"] and not backend.execute_calls


@pytest.mark.parametrize("field", ["input_artifacts", "calibration_artifacts"])
def test_preflight_defends_against_bypassing_planning_and_requests_repair(tmp_path, field):
    cfg, services, _, backend, architecture = integrations(tmp_path, mode="claims_verified")
    with WorkflowApp(cfg, services) as app:
        waiting = app.start(BENCH_QUESTION, architecture_path=architecture)
        state = WorkflowState.model_validate(waiting["state"])
        procedure = state.procedures[-1].model_copy(deep=True)
        artifact = bad_artifact(cfg, backend, in_root=True)  # A valid hash inside the allowed root still cannot pass.
        if field == "input_artifacts":
            procedure.input_artifacts = [artifact]
        else:
            procedure.calibration_artifacts = {"private-matrix": artifact}
        review = preflight(procedure, cfg, backend)
        assert review.status == "blocked"
        assert any("Procedure validation error: procedure." + field in error for error in review.issues)
        app._preflight_issues(state, SimpleNamespace(technical_issues=review.issues))
        assert state.control.next_node == "experimentor" and state.control.task == "technical_review"
        assert state.status != "awaiting_information" and not backend.execute_calls


def test_backend_rejects_direct_external_artifact_before_camera_or_output_creation(tmp_path, monkeypatch):
    cfg, services, _, backend, architecture = integrations(tmp_path, mode="claims_verified")
    with WorkflowApp(cfg, services) as app:
        waiting = app.start(BENCH_QUESTION, architecture_path=architecture)
    state = WorkflowState.model_validate(waiting["state"])
    monkeypatch.setattr(OpticalBench, "frame", lambda *a, **kw: pytest.fail("Camera must not run"))
    output = tmp_path / "must-not-exist"
    result = backend.execute(state.procedures[-1].actions[0], output_dir=output,
        input_artifacts=[bad_artifact(cfg, backend, in_root=True)], capability_fingerprint=canonical_hash(backend.describe().model_dump(mode="json")),
        deadline=time.monotonic()+10, cancel_event=Event())
    assert result.errors and "no external" in result.errors[0] and not output.exists()


def test_public_planning_views_hide_sources_but_keep_saved_configuration_and_legacy_files(tmp_path):
    cfg, services, _, backend, _ = integrations(tmp_path, mode="claims_verified")
    private = backend.config.model_dump(mode="json")
    legacy = tmp_path / "legacy-setup.txt"
    text = "Legacy setup\n" + json.dumps(private, indent=2)
    legacy.write_text(text, encoding="utf-8")
    state = WorkflowState(user_request="Run identity")
    cfg.optical_bench = backend.config
    state.control.configuration = cfg.model_dump(mode="json")
    source_before = state.model_dump_json()
    public = planning_state(state)
    context = planning_context(read_context(legacy, []), state)
    for key in ("detector_matrix_path", "detector_matrix_sha256"):
        assert key not in public["control"]["configuration"]["optical_bench"]
        assert private[key] not in context.architecture_text
        assert json.dumps(private[key])[1:-1] not in context.architecture_text
    assert "matrix mode: known" in context.architecture_text and "backend-owned" in context.architecture_text
    assert state.model_dump_json() == source_before and legacy.read_text(encoding="utf-8") == text
    assert private["detector_matrix_path"] not in setup_context(backend.config)
    assert private["detector_matrix_sha256"] not in setup_context(backend.config)
