"""Analyst contracts and workflow handoffs, using metadata and canned models only."""

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from agents.analyst import DEFAULT_SYSTEM_PROMPT, analyst
from config import AppConfig
from graph import Integrations, WorkflowApp
from state import Artifact, HardwareExecution, NumericalAnalysisResult, WorkflowState
from tools.offline_fixtures import FixtureAnalystModel, offline_integrations


class FakeAnalyst(FixtureAnalystModel):
    def __init__(self, change=lambda payload, result: result):
        self.change = change
        self.calls = []

    def invoke(self, messages):
        payload = json.loads(messages[1][1])
        self.calls.append(payload)
        return self.change(payload, super().invoke(messages))


@pytest.fixture
def evidence(experiment_setup):
    state, _, architecture = experiment_setup
    acquisition = state.acquisitions[0]
    state.analyses.append(NumericalAnalysisResult(acquisition_ids=[acquisition.id],
        input_data=acquisition.raw_data, code=Artifact(path="fixture.py", description="Not executed"),
        code_version="fixture-v1", status="completed", metrics={"fixture_metric": 1.0}))
    return state, architecture


def test_advice_is_copied_serializable_context_and_reaches_data_requirements(evidence):
    state, architecture = evidence
    before = state.model_dump_json()
    model = FakeAnalyst()
    result = analyst(state, model, stage="before_acquisition", architecture_path=architecture)
    assert state.model_dump_json() == before
    assert result.outcome.status == "advised"
    assert "coherent-reference" in result.outcome.recommendations[0]
    assert len(result.state.conversation) == 1
    assert result.state.conversation[-1].role == "analyst"
    assert result.state.conversation[-1].scientific_context.architecture_text
    assert WorkflowState.model_validate_json(result.state.model_dump_json()) == result.state
    assert result.state.procedures == state.procedures and result.state.approvals == state.approvals
    assert model.calls[0]["stage"] == "before_acquisition"


@pytest.mark.parametrize("origin", ["physical", "simulated", "test_only", "unknown"])
def test_result_provenance_is_derived_not_invented(evidence, origin):
    state, architecture = evidence
    if origin != "unknown":
        now = datetime.now(timezone.utc)
        state.acquisitions[0].hardware = HardwareExecution(approval_id=uuid4(), review_id=uuid4(),
            review_fingerprint="fixture", backend_id="metadata-fixture", data_origin=origin,
            run_directory="fixture-run", started_at=now, ended_at=now, actions=[])
    result = analyst(state, FakeAnalyst(lambda p, r: {**r, "provenance": {"invented": ["physical"]}}),
                     stage="after_analysis", architecture_path=architecture)
    assert result.outcome.provenance[str(state.analyses[0].id)] == [origin]
    assert "invented" not in result.outcome.provenance
    if origin == "unknown":
        assert any("missing/unknown" in note for note in result.outcome.limitations)


@pytest.mark.parametrize("problem", ["missing_id", "missing_path", "wrong_stage", "wrong_target", "failed_analysis", "partial_acquisition", "missing_upstream", "analysis_as_upstream"])
def test_invalid_or_ineligible_evidence_is_not_success(evidence, problem):
    state, architecture = evidence
    def change(payload, result):
        if problem == "missing_id":
            result["evidence_ids"].append(str(uuid4()))
        elif problem == "missing_path":
            result["artifact_paths"] = ["invented.npy"]
        elif problem == "wrong_stage":
            result["stage"] = "before_acquisition"
        elif problem == "wrong_target":
            result["analysis_ids"] = []
        return result
    if problem == "failed_analysis":
        state.analyses[0] = NumericalAnalysisResult.model_validate({**state.analyses[0].model_dump(), "status": "failed", "errors": ["fixture failed"]})
    elif problem == "partial_acquisition":
        from state import AcquisitionResult
        state.acquisitions[0] = AcquisitionResult.model_validate({**state.acquisitions[0].model_dump(), "status": "partial", "errors": ["partial fixture"]})
    elif problem == "missing_upstream":
        state.analyses[0].acquisition_ids = [uuid4()]
    elif problem == "analysis_as_upstream":
        state.analyses[0].acquisition_ids = [state.analyses[0].id]
    before = state.model_dump_json()
    with pytest.raises(ValueError):
        analyst(state, FakeAnalyst(change), stage="after_analysis", architecture_path=architecture)
    assert state.model_dump_json() == before and not state.analyst_reviews


def test_missing_evidence_can_be_reported_as_information_not_review(evidence):
    state, architecture = evidence
    state.analyses.clear()
    def missing(payload, result):
        return {**result, "status": "needs_information", "questions": ["Supply saved numerical results and validation split metadata"]}
    result = analyst(state, FakeAnalyst(missing), stage="after_analysis", architecture_path=architecture)
    assert result.outcome.status == "needs_information"
    with pytest.raises(ValueError, match="completed analysis"):
        analyst(state, FakeAnalyst(), stage="after_analysis", architecture_path=architecture)


def request_more(payload, result):
    key = payload["analysis_ids"][0]
    return {**result, "status": "more_analysis", "analysis_request": {
        "calculation": "Report held-out prediction-error uncertainty and disclose split independence",
        "input_record_ids": [key], "input_artifact_paths": []}}


def test_structured_request_uses_existing_inputs_and_validates_artifact_ownership(evidence):
    state, architecture = evidence
    result = analyst(state, FakeAnalyst(request_more), stage="after_analysis", architecture_path=architecture)
    assert result.outcome.analysis_request.input_record_ids == [state.analyses[0].id]
    assert result.state.acquisitions == state.acquisitions
    def bad(payload, result):
        result = request_more(payload, result)
        result["analysis_request"]["input_artifact_paths"] = ["missing.npy"]
        return result
    with pytest.raises(ValueError, match="input artifacts"):
        analyst(state, FakeAnalyst(bad), stage="after_analysis", architecture_path=architecture)


@pytest.fixture
def workflow(tmp_path):
    cfg = AppConfig(workflow_database_path=tmp_path / "analyst.sqlite", run_output_path=tmp_path / "runs",
        hardware_input_path=tmp_path / "input", hardware_backend="test", numerical_backend="test",
        max_workflow_iterations=40)
    services = Integrations(**offline_integrations(cfg, uuid4(), include_analyst=True))
    return cfg, services


def approve(view):
    return {"action": "approve", "review_fingerprint": view["pending"]["review"]["review_fingerprint"]}


def test_both_workflow_handoffs_and_restart(workflow):
    cfg, services = workflow
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Analyst workflow fixture")
        assert waiting["state"]["analyst_reviews"][0]["stage"] == "before_acquisition"
        assert not services.hardware_backend.execute_calls
    with WorkflowApp(cfg, services) as app:
        final = app.submit(waiting["workflow_id"], approve(waiting))
        assert final["status"] == "completed"
        assert [r["stage"] for r in final["state"]["analyst_reviews"]] == ["before_acquisition", "after_analysis"]
        roles = [c["role"] for c in final["state"]["conversation"] if c["role"] != "orchestrator"]
        assert roles == ["scientist", "analyst", "experimentor", "runner", "human", "runner", "experimentor", "analyst", "scientist"]
        ids = [c["id"] for c in final["state"]["conversation"]]
        assert len(ids) == len(set(ids))


def test_workflow_without_analyst_remains_usable_and_skip_visible(workflow):
    cfg, services = workflow
    services.analyst_model = None
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Optional Analyst")
        final = app.submit(waiting["workflow_id"], approve(waiting))
        assert final["status"] == "completed" and not final["state"]["analyst_reviews"]
        assert any("Analyst" in c["message"] and "skipped" in c["message"] for c in final["state"]["conversation"])


def test_failed_analyst_blocks_and_never_records_success(workflow):
    cfg, services = workflow
    services.analyst_model = FakeAnalyst(lambda p, r: {})
    with WorkflowApp(cfg, services) as app:
        blocked = app.start("Invalid Analyst")
        assert blocked["status"] == "blocked" and "Analyst" in blocked["reason"]
        assert not blocked["state"]["analyst_reviews"]
        assert not services.hardware_backend.execute_calls


def test_analyst_more_analysis_uses_existing_data_without_acquisition(workflow):
    cfg, services = workflow
    services.analyst_model = FakeAnalyst(lambda p, r: request_more(p, r) if p["stage"] == "after_analysis"
        and len(p["workflow_state"]["analyses"]) == 1 else r)
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Analyst calculation request")
        final = app.submit(waiting["workflow_id"], approve(waiting))
        assert final["status"] == "completed"
        assert len(services.hardware_backend.execute_calls) == 1
        assert len(services.execution_backend.calls) == 2
        assert any(r["status"] == "more_analysis" for r in final["state"]["analyst_reviews"])


def test_prompt_has_feasibility_validation_and_authority_rules():
    for phrase in ("intensity", "coherent reference", "held-out", "calibration", "Scientist owns",
                   "Experimentor chooses", "unknown", "metadata", "never schedule measurements"):
        assert phrase in DEFAULT_SYSTEM_PROMPT


def test_fixture_selection_survives_restart(workflow):
    cfg, _ = workflow
    with WorkflowApp(cfg) as app:
        waiting = app.start("Explicit Analyst fixture", fixture=True, fixture_analyst=True)
        assert waiting["state"]["control"]["fixture_analyst"]
    with WorkflowApp(cfg) as app:
        final = app.submit(waiting["workflow_id"], approve(waiting))
        assert final["status"] == "completed"
        assert len(final["state"]["analyst_reviews"]) == 2


def test_post_analysis_failure_preserves_results_without_scientist_interpretation(workflow):
    cfg, services = workflow
    services.analyst_model = FakeAnalyst(lambda p, r: {} if p["stage"] == "after_analysis" else r)
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Fail post-analysis review")
        final = app.submit(waiting["workflow_id"], approve(waiting))
        assert final["status"] == "blocked" and "after_analysis failed" in final["reason"]
        assert final["state"]["analyses"][0]["status"] == "completed"
        assert [r["stage"] for r in final["state"]["analyst_reviews"]] == ["before_acquisition"]
        assert not final["state"]["findings"]
