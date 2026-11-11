"""Same routing interface in saved-data/no-data states; no live API calls."""

import json
from uuid import uuid4

import pytest

from agents.orchestrator import choose_route
from workflow.config import AppConfig
from workflow.graph import Integrations, WorkflowApp
from workflow.state import WorkflowState
from fixtures.router import TestRouter as ScriptedRouter, double_integrations
from checks.router import make_case, safety_checks
from workflow.routing_catalog import build_catalog


@pytest.fixture
def setup(tmp_path):
    cfg = AppConfig(llm_router_enabled=True, hardware_backend="test", reviewed_numerics_enabled=True,
        numerical_input_path=tmp_path / "inputs", hardware_input_path=tmp_path / "hardware-inputs",
        run_output_path=tmp_path / "runs", workflow_database_path=tmp_path / "workflow.sqlite",
        max_workflow_iterations=40, max_workflow_analysis_attempts=6)
    return cfg, double_integrations(cfg)


def test_same_router_reuses_data_or_prepares_acquisition_or_reports_missing(setup):
    cfg, services = setup
    existing = make_case(cfg, with_data=True)
    empty = make_case(cfg, with_data=False)
    saved, reuse = choose_route(existing, cfg, services)
    planned, prepare = choose_route(empty, cfg, services)
    unavailable = Integrations(scientist_model=services.scientist_model, analyst_model=services.analyst_model,
        experimentor_model=services.experimentor_model, router_model=services.router_model)
    missing, ask = choose_route(make_case(cfg, with_data=False), cfg, unavailable)
    assert saved.status == planned.status == missing.status == "accepted"
    assert reuse.kind == "run_operation" and reuse.request.operation == "tm.compare_reference_free"
    assert prepare.kind == "prepare_procedure"
    assert ask.kind == "request_information" and "Missing" in missing.reason
    assert existing.user_request == empty.user_request
    calls = services.router_model.calls
    assert len(calls) == 3 and len({c["research_question"] for c in calls}) == 1
    assert len(calls[0]["datasets"]) == 1 and calls[1]["datasets"] == calls[2]["datasets"] == []
    assert "workflow_state" not in calls[0]  # compact context, no full transcript/credentials
    assert len({d["pattern_content_sha256"] for d in calls[0]["datasets"]}) == 1
    assert len({d["split_sha256"] for d in calls[0]["datasets"]}) == 1
    assert saved.record_ids == reuse.record_ids
    assert existing.router_decisions[-1].reason == saved.reason


def test_wrong_record_id_is_rejected_before_dispatch(setup):
    cfg, services = setup
    state = make_case(cfg, with_data=True)
    class WrongID(ScriptedRouter):
        def invoke(self, messages):
            return {**super().invoke(messages), "record_ids": [str(uuid4())]}
    services.router_model = WrongID()
    before = [a.model_dump() for a in state.acquisitions]
    decision, action = choose_route(state, cfg, services)
    assert action is None and decision.status == "rejected" and "not bound" in decision.error
    assert before == [a.model_dump() for a in state.acquisitions]
    assert services.hardware_backend.execute_calls == [] and state.analyses == []


def test_invalid_action_and_unapproved_acquisition_are_blocked(tmp_path):
    checks = safety_checks(tmp_path)
    assert checks["invalid_action_blocked"] and checks["unapproved_acquisition_calls"] == 0


def test_invalid_choice_is_checkpointed_and_can_resume_without_losing_data(setup):
    cfg, services = setup
    state = make_case(cfg, with_data=True)
    services.router_model = ScriptedRouter(invalid=True)
    with WorkflowApp(cfg, services) as app:
        first = app.start(state.user_request, initial_state=state)
        assert first["status"] == "blocked"
        assert first["state"]["control"]["next_node"] == "router"
        assert len(first["state"]["router_decisions"]) == 1
    with WorkflowApp(cfg, services) as app:
        again = app.resume(first["workflow_id"])
        assert again["status"] == "blocked" and len(again["state"]["router_decisions"]) == 2
        assert again["state"]["acquisitions"] == first["state"]["acquisitions"]
    assert services.hardware_backend.execute_calls == []


def test_router_checks_live_availability_again_after_model_response(setup):
    cfg, services = setup
    state = make_case(cfg, with_data=False)
    class StaleAction(ScriptedRouter):
        def invoke(self, messages):
            choice = super().invoke(messages)
            cfg.hardware_backend = "disabled"
            return choice
    services.router_model = StaleAction()
    decision, action = choose_route(state, cfg, services)
    assert action is None and decision.status == "rejected" and "became unavailable" in decision.error
    assert services.hardware_backend.execute_calls == []


def test_fixture_mode_never_calls_router_even_if_enabled(setup):
    cfg, services = setup
    services.router_model = ScriptedRouter(invalid=True)
    with WorkflowApp(cfg, services) as app:
        waiting = app.start("Explicit deterministic fixture", fixture=True)
    assert waiting["status"] == "awaiting_approval"
    assert services.router_model.calls == []
    assert waiting["state"]["control"]["configuration"]["llm_router_enabled"] is False
    assert waiting["state"]["router_decisions"] == []


def test_attempt_limit_removes_numerical_actions(setup):
    cfg, services = setup
    state = make_case(cfg, with_data=True)
    state.control.analysis_attempts = cfg.max_workflow_analysis_attempts
    actions, *_ = build_catalog(state, cfg, services)
    assert not any(a.kind == "run_operation" for a in actions)
