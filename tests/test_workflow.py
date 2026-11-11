"""Persistent software handoffs only: fake models, fake execution, fake hardware."""

import json
from pathlib import Path
from uuid import uuid4

import pytest

from agents.experimentor import ExperimentorOutcome
from workflow.config import AppConfig
from workflow.graph import Integrations, WorkflowApp
from workflow.state import WorkflowState
from fixtures.offline import offline_integrations
from tools.python_execution import BackendReport


class ModelProxy:
    def __init__(self, model):
        self.model = model
        self.transform = lambda payload, schema, result: result
        self.calls = []

    def with_structured_output(self, schema):
        self.schema = schema
        self.model.with_structured_output(schema)
        return self

    def invoke(self, messages):
        payload = json.loads(messages[1][1])
        self.calls.append(payload)
        return self.transform(payload, self.schema, self.model.invoke(messages))


@pytest.fixture
def workflow_setup(tmp_path):
    cfg = AppConfig(workflow_database_path=tmp_path / "workflow.sqlite",
        run_output_path=tmp_path / "runs", hardware_input_path=tmp_path / "inputs",
        numerical_input_path=tmp_path / "runs" / "acquisitions",
        hardware_backend="test", numerical_backend="test", max_workflow_iterations=40)
    services = Integrations(**offline_integrations(cfg, uuid4()))
    services.scientist_model = ModelProxy(services.scientist_model)
    services.experimentor_model = ModelProxy(services.experimentor_model)
    return cfg, services


def approve(view):
    return {"action": "approve", "review_fingerprint": view["pending"]["review"]["review_fingerprint"],
            "reviewer": "Explicit fake-workflow human decision"}


def assert_unique_history(view):
    ids = [r["id"] for r in view["state"]["conversation"]]
    assert len(ids) == len(set(ids))


def test_full_workflow_and_no_acquisition_before_approval(workflow_setup):
    cfg, services = workflow_setup
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Test only software handoffs")
        assert waiting["status"] == "awaiting_approval"
        assert not services.hardware_backend.execute_calls
        planning = services.experimentor_model.calls[0]["workflow_state"]["control"]["hardware_context"]
        assert planning["actions"]["fixture.capture"]["parameters"]["count"]["unit"] == "frame"
        assert planning["verified_calibrations"]
        review = waiting["pending"]
        assert review["scientific_purpose"]
        assert review["review"]["procedure"]["actions"]
        assert review["review"]["capabilities_snapshot"]["data_origin"] == "test_only"
        assert review["review"]["procedure"]["calibration_artifacts"]
        final = app.submit(waiting["workflow_id"], approve(waiting))
        assert final["status"] == "completed"
        assert len(services.hardware_backend.execute_calls) == 1
        assert len(services.execution_backend.calls) == 1
        state = WorkflowState.model_validate(final["state"])
        assert len(state.proposals) == len(state.procedures) == len(state.approvals) == 1
        assert state.acquisitions[0].status == state.analyses[0].status == "completed"
        assert state.acquisitions[0].hardware.data_origin == "test_only"
        assert state.findings[0].evidence_ids == [state.analyses[0].id]
        assert state.user_request == "Test only software handoffs"
        assert state.control.acquisition_attempts == state.control.analysis_attempts == 1
        assert app.resume(state.id) == final
        assert len(services.hardware_backend.execute_calls) == 1
        assert_unique_history(final)


def test_restart_awaiting_review_and_ambiguous_input(workflow_setup):
    cfg, services = workflow_setup
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Restart fixture")
    calls = len(services.scientist_model.calls) + len(services.experimentor_model.calls)
    with WorkflowApp(cfg, services) as app:
        assert app.show(waiting["workflow_id"]) == waiting
        assert app.resume(waiting["workflow_id"]) == waiting
        ambiguous = app.submit(waiting["workflow_id"], {"action": "probably"})
        assert ambiguous["input_error"] and ambiguous["status"] == "awaiting_approval"
        assert app.show(waiting["workflow_id"]) == waiting
        assert calls == len(services.scientist_model.calls) + len(services.experimentor_model.calls)
        assert app.submit(waiting["workflow_id"], approve(waiting))["status"] == "completed"


def test_rejection_revises_and_old_approval_does_not_cover_revision(workflow_setup):
    cfg, services = workflow_setup
    with WorkflowApp(cfg, services) as app:
        first = app.start("Reject and revise")
        second = app.submit(first["workflow_id"], {"action": "reject", "feedback": "Explain output naming more clearly", "reviewer": "Test reviewer"})
        assert second["status"] == "awaiting_approval"
        assert [p["version"] for p in second["state"]["procedures"]] == [1, 2]
        assert second["state"]["approvals"][0]["status"] == "rejected"
        stale = app.submit(first["workflow_id"], approve(first))
        assert stale["input_error"] and not services.hardware_backend.execute_calls
        final = app.submit(first["workflow_id"], approve(second))
        assert final["status"] == "completed" and len(final["state"]["approvals"]) == 2
        assert len(services.hardware_backend.execute_calls) == 1
        assert_unique_history(final)


def test_analysis_only_follow_up_does_not_acquire_again(workflow_setup):
    cfg, services = workflow_setup
    def followup(payload, schema, result):
        state = payload["workflow_state"]
        if any(e["role"] == "user" and "more analysis" in e["message"] for e in state["conversation"]) and len(state["analyses"]) == 1:
            return {"kind": "analysis_request", "explanation": "More diagnostics on saved data only",
                    "recommended_next_step": "Experimentor analyze existing data"}
        return result
    services.scientist_model.transform = followup
    with WorkflowApp(cfg, services) as app:
        first = app.start("Original request")
        final = app.submit(first["workflow_id"], approve(first))
        again = app.follow_up(final["workflow_id"], "Please do more analysis of the saved data")
        assert again["status"] == "completed"
        assert len(again["state"]["analyses"]) == 2
        assert len(again["state"]["procedures"]) == 1
        assert len(services.hardware_backend.execute_calls) == 1
        assert len(services.execution_backend.calls) == 2
        assert again["state"]["user_request"] == "Original request"
        assert_unique_history(again)


def test_technical_problem_routes_to_experimentor_then_new_review(workflow_setup):
    cfg, services = workflow_setup
    def unsupported_first(payload, schema, result):
        if schema is ExperimentorOutcome and not payload["workflow_state"]["procedures"] and result["kind"] == "procedure":
            result["procedure"]["actions"][0]["action"] = "unknown.action"
        return result
    services.experimentor_model.transform = unsupported_first
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Technical revision")
        assert waiting["status"] == "awaiting_approval"
        assert [p["version"] for p in waiting["state"]["procedures"]] == [1, 2]
        assert any(c["workflow_state"]["control"]["task"] == "technical_review" for c in services.experimentor_model.calls)
        assert len(services.scientist_model.calls) == 1
        assert not services.hardware_backend.execute_calls


def test_scientific_change_is_returned_to_scientist(workflow_setup):
    cfg, services = workflow_setup
    def decision(payload, schema, result):
        state = payload["workflow_state"]
        if schema is ExperimentorOutcome and state["control"]["task"] == "technical_review":
            return {"kind": "request_information", "explanation": "Control change needs a scientific decision",
                    "scientific_decisions": ["May the scientific control change?"], "recommended_next_step": "Scientist decision"}
        if result["kind"] == "procedure" and not state["procedures"]:
            result["procedure"]["actions"][0]["action"] = "unsupported"
        return result
    services.experimentor_model.transform = decision
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Scientific decision routing")
        assert waiting["status"] == "awaiting_approval"
        assert len(waiting["state"]["proposals"]) == 2
        assert len(services.scientist_model.calls) == 2
        assert "scientific decision" in " ".join(e["message"] for e in waiting["state"]["conversation"])


def test_missing_information_pauses_without_model_busy_loop(workflow_setup):
    cfg, services = workflow_setup
    def need_info(payload, schema, result):
        if not any(e["role"] == "user" for e in payload["workflow_state"]["conversation"]):
            return {"kind": "request_information", "explanation": "Essential format missing",
                    "recommended_next_step": "Ask user", "questions": ["What is the saved-data format?"]}
        return result
    services.scientist_model.transform = need_info
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Clarification fixture")
        assert waiting["status"] == "awaiting_information"
        assert waiting["pending"]["questions"] == ["What is the saved-data format?"]
        assert app.resume(waiting["workflow_id"]) == waiting
        assert len(services.scientist_model.calls) == 1
        resumed = app.submit(waiting["workflow_id"], {"action": "clarify", "message": "Use the test fixture JSON"})
        assert resumed["status"] == "awaiting_approval"
        assert len([e for e in resumed["state"]["conversation"] if e["role"] == "user"]) == 1


@pytest.mark.parametrize("missing", ["scientist", "experimentor", "hardware", "numerical"])
def test_disabled_integrations_are_actionably_blocked(workflow_setup, missing):
    cfg, services = workflow_setup
    if missing == "scientist":
        services.scientist_model = None
    elif missing == "experimentor":
        services.experimentor_model = None
    elif missing == "hardware":
        cfg.hardware_backend = "disabled"
    else:
        cfg.numerical_backend = "disabled"
    with WorkflowApp(cfg, services) as app:
        view = app.start("Disabled integration")
        if missing == "numerical":
            view = app.submit(view["workflow_id"], approve(view))
            assert view["state"]["analysis_code"]
            assert view["state"]["analyses"][-1]["execution"]["status"] == "unavailable"
        assert view["status"] == "blocked" and view["reason"]
        assert "limit reached" not in view["reason"]
        assert len(services.execution_backend.calls) == 0


def test_partial_failure_needs_explicit_partial_analysis_request(workflow_setup):
    cfg, services = workflow_setup
    services.hardware_backend.fail_at = 1
    def two_actions(payload, schema, result):
        if result.get("kind") == "procedure":
            result["procedure"]["actions"] *= 2
        return result
    services.experimentor_model.transform = two_actions
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Partial-data fixture")
        paused = app.submit(waiting["workflow_id"], approve(waiting))
        assert paused["status"] == "awaiting_information"
        assert paused["state"]["acquisitions"][0]["status"] == "partial"
        assert len(paused["state"]["acquisitions"][0]["raw_data"]) == 1
        assert not services.execution_backend.calls and len(services.hardware_backend.cleanup_calls) == 1
        completed = app.submit(paused["workflow_id"], {"action": "clarify", "message": "Explicitly analyze only the saved partial fixture data", "allow_partial_analysis": True})
        assert completed["status"] == "completed"
        assert len(services.execution_backend.calls) == 1
        assert completed["state"]["acquisitions"][0]["status"] == "partial"
        assert not completed["state"]["findings"]


def test_analysis_failure_and_attempt_limit(workflow_setup):
    cfg, services = workflow_setup
    cfg.max_workflow_analysis_attempts = 1
    class FailExecution:
        def run(self, **kwargs):
            return BackendReport(exit_code=1, stderr="TEST ONLY injected failure", elapsed_seconds=0.0)
    services.execution_backend = FailExecution()
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Failed analysis")
        stopped = app.submit(waiting["workflow_id"], approve(waiting))
        assert stopped["status"] == "stopped" and "analysis attempt limit" in stopped["reason"]
        assert stopped["state"]["analyses"][0]["status"] == "failed"
        assert stopped["state"]["analyses"][0]["execution"]["stderr"] == "TEST ONLY injected failure"
        assert len(services.hardware_backend.execute_calls) == 1
    with WorkflowApp(cfg, services) as restarted:
        assert restarted.resume(stopped["workflow_id"]) == stopped
        assert stopped["state"]["control"]["analysis_attempts"] == 1


def test_acquisition_limit_survives_restart_and_new_approval(workflow_setup):
    cfg, services = workflow_setup
    cfg.max_workflow_acquisition_attempts = 1
    services.hardware_backend.fail_at = 0
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Acquisition limit")
        paused = app.submit(waiting["workflow_id"], approve(waiting))
        assert paused["status"] == "awaiting_information"
    def revise(payload, schema, result):
        if schema is ExperimentorOutcome and result["kind"] == "request_information":
            copied = json.loads(json.dumps(payload))
            copied["workflow_state"]["control"]["task"] = "procedure"
            return services.experimentor_model.model.invoke([("system", "fixture"), ("human", json.dumps(copied))])
        return result
    services.experimentor_model.transform = revise
    with WorkflowApp(cfg, services) as app:
        second = app.submit(paused["workflow_id"], {"action": "clarify", "message": "Prepare a revised procedure for another review"})
        assert second["status"] == "awaiting_approval"
        stopped = app.submit(second["workflow_id"], approve(second))
        assert stopped["status"] == "stopped" and "Acquisition attempt limit" in stopped["reason"]
        assert len(services.hardware_backend.execute_calls) == 1
        assert len(stopped["state"]["approvals"]) == 2


def test_cancellation_and_revision_step_limits(workflow_setup):
    cfg, services = workflow_setup
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Cancel")
        cancelled = app.cancel(waiting["workflow_id"])
        assert cancelled["status"] == "cancelled"
        assert app.resume(waiting["workflow_id"]) == cancelled
        assert not services.hardware_backend.execute_calls
    cfg.max_workflow_iterations = 1
    with WorkflowApp(cfg, services) as app:
        stopped = app.start("Step limit")
        assert stopped["status"] == "stopped" and stopped["state"]["proposals"]
        assert "step limit" in stopped["reason"]


def test_uncertain_acquisition_restarts_into_reconciliation(workflow_setup):
    cfg, services = workflow_setup
    original_execute = services.hardware_backend.execute
    class PowerLoss(BaseException):
        pass
    def crash(*args, **kwargs):
        raise PowerLoss()
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Uncertain hardware attempt")
        services.hardware_backend.execute = crash
        with pytest.raises(PowerLoss):
            app.submit(waiting["workflow_id"], approve(waiting))
    services.hardware_backend.execute = original_execute
    with WorkflowApp(cfg, services) as app:
        paused = app.resume(waiting["workflow_id"])
        assert paused["status"] == "awaiting_information"
        assert paused["pending"]["kind"] == "reconciliation"
        assert paused["state"]["acquisitions"][0]["status"] == "started"
        assert not services.hardware_backend.execute_calls
        rejected = app.submit(paused["workflow_id"], {"action": "approve"})
        assert rejected["input_error"]
        assert app.cancel(paused["workflow_id"])["status"] == "cancelled"


def test_completed_journal_recovered_without_duplicate_acquisition(workflow_setup, monkeypatch):
    from workflow import graph as graph_module
    cfg, services = workflow_setup
    original_runner = graph_module.runner
    class LostCheckpoint(BaseException):
        pass
    def crash_after_record(*args, **kwargs):
        result = original_runner(*args, **kwargs)
        if kwargs.get("execute") and result.acquisition:
            raise LostCheckpoint()
        return result
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Recover completed journal")
        monkeypatch.setattr(graph_module, "runner", crash_after_record)
        with pytest.raises(LostCheckpoint):
            app.submit(waiting["workflow_id"], approve(waiting))
    monkeypatch.setattr(graph_module, "runner", original_runner)
    with WorkflowApp(cfg, services) as app:
        final = app.resume(waiting["workflow_id"])
        assert final["status"] == "completed"
        assert len(services.hardware_backend.execute_calls) == 1
        assert len(final["state"]["acquisitions"]) == 1
        assert_unique_history(final)


def test_changed_constraints_after_human_review_need_new_approval(workflow_setup):
    cfg, services = workflow_setup
    with WorkflowApp(cfg, services) as app:
        first = app.start("Changed constraints")
        services.hardware_backend.capabilities.revision = "updated fixture revision"
        renewed = app.submit(first["workflow_id"], approve(first))
        assert renewed["status"] == "awaiting_approval"
        assert not services.hardware_backend.execute_calls
        assert renewed["pending"]["review"]["review_fingerprint"] != first["pending"]["review"]["review_fingerprint"]
        assert app.submit(renewed["workflow_id"], approve(renewed))["status"] == "completed"


def test_cli_explicit_fixture_and_normal_missing_configuration(tmp_path, monkeypatch, capsys):
    from main import main
    # A developer's local .env must never turn this offline test into API calls.
    for name in ("TM_LLM_PROVIDER", "TM_LLM_MODEL", "TM_API_KEY"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("TM_RUN_OUTPUT_PATH", str(tmp_path / "runs"))
    db = str(tmp_path / "cli.sqlite")
    assert main(["--db", db, "start", "Interface fixture", "--fixture"]) == 0
    output = capsys.readouterr().out
    workflow_id = output.splitlines()[0].split()[1].rstrip(":")
    assert "TEST ONLY" in output and "awaiting_approval" in output
    assert main(["--db", db, "show", workflow_id]) == 0
    capsys.readouterr()
    with WorkflowApp(AppConfig(workflow_database_path=Path(db))) as app:
        fingerprint = app.show(workflow_id)["pending"]["review"]["review_fingerprint"]
    assert main(["--db", db, "review", workflow_id, "approve", "--fingerprint", fingerprint, "--reviewer", "Test reviewer"]) == 0
    assert "completed" in capsys.readouterr().out.splitlines()[0]
    assert main(["--db", db, "start", "Normal mode must not fake a model"]) == 2
    assert "unconfigured" in capsys.readouterr().out


def test_procedure_revision_limit_preserves_original_procedure(workflow_setup):
    cfg, services = workflow_setup
    cfg.max_procedure_revisions = 0
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Revision cap")
        stopped = app.submit(waiting["workflow_id"], {"action": "reject", "feedback": "Needs revision"})
        assert stopped["status"] == "stopped" and "revision limit" in stopped["reason"]
        assert len(stopped["state"]["procedures"]) == 1
        assert not services.hardware_backend.execute_calls


def test_more_measurements_require_another_explicit_approval(workflow_setup):
    cfg, services = workflow_setup
    def additional(payload, schema, result):
        state = payload["workflow_state"]
        if any(e["role"] == "user" and "additional measurement" in e["message"] for e in state["conversation"]) and len(state["proposals"]) == 1:
            rewritten = json.loads(json.dumps(payload))
            rewritten["workflow_state"]["control"]["task"] = "proposal"
            return services.scientist_model.model.invoke([("system", "fixture"), ("human", json.dumps(rewritten))])
        return result
    services.scientist_model.transform = additional
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Additional measurements fixture")
        final = app.submit(waiting["workflow_id"], approve(waiting))
        second = app.follow_up(final["workflow_id"], "Propose an additional measurement for separate review")
        assert second["status"] == "awaiting_approval"
        assert len(second["state"]["proposals"]) == 2
        assert len(services.hardware_backend.execute_calls) == 1
        finished = app.submit(second["workflow_id"], approve(second))
        assert finished["status"] == "completed"
        assert len(services.hardware_backend.execute_calls) == 2
        assert len(finished["state"]["approvals"]) == 2
        assert finished["state"]["control"]["procedure_revisions"] == 0


def test_lab_information_does_not_become_verified_limits_from_prose(workflow_setup):
    cfg, services = workflow_setup
    services.hardware_backend.capabilities.prerequisites["fixture setup verified"].verified = False
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Unverified laboratory fact")
        assert waiting["status"] == "awaiting_information"
        still_waiting = app.submit(waiting["workflow_id"], {"action": "clarify", "message": "Please just assume it is calibrated"})
        assert still_waiting["status"] == "awaiting_information"
        assert not services.hardware_backend.execute_calls
        # Only an explicitly verified backend snapshot can resolve this condition.
        services.hardware_backend.capabilities.prerequisites["fixture setup verified"].verified = True
        ready = app.submit(waiting["workflow_id"], {"action": "clarify", "message": "Test-only backend verification has been updated"})
        assert ready["status"] == "awaiting_approval"


def test_malformed_execution_review_keeps_the_recorded_analysis(workflow_setup):
    from agents.experimentor import ExecutionReview
    cfg, services = workflow_setup
    def malformed_review(payload, schema, result):
        return {} if schema is ExecutionReview else result
    services.experimentor_model.transform = malformed_review
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Execution before malformed review")
        blocked = app.submit(waiting["workflow_id"], approve(waiting))
        assert blocked["status"] == "blocked"
        assert len(blocked["state"]["analyses"]) == 1
        assert blocked["state"]["analyses"][0]["status"] == "completed"
        assert blocked["state"]["analysis_code"]
        assert not blocked["state"]["findings"]
        assert len(services.execution_backend.calls) == 1


def test_configuration_and_clients_are_not_serialized_as_secrets(workflow_setup):
    from pydantic import SecretStr
    cfg, services = workflow_setup
    cfg.api_key = SecretStr("fixture-secret-never-checkpoint")
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Secret handling")
        assert "fixture-secret-never-checkpoint" not in json.dumps(waiting)
        assert "api_key" not in waiting["state"]["control"]["configuration"]
