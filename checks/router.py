"""Routing scenarios. Default: test doubles. --live: three real routing LLM calls.

Uses existing algebraic and recording-backend fixtures. Safety checks
always use test doubles; the live option invokes only the router, never hardware.
"""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from agents.orchestrator import choose_route
from workflow.config import AppConfig
from workflow.graph import Integrations, WorkflowApp
from workflow.state import ExperimentProposal, WorkflowState
from fixtures.router import TestRouter, double_integrations
from fixtures.numerical import create_fixture


def make_case(config, *, with_data):
    from bench.workflow import PAPER_QUESTION as question
    proposal = ExperimentProposal(scientific_question=question, hypothesis="Prediction accuracy may depend on the noise condition",
        rationale="ROUTER CHECK imported scientific-plan fixture; not a live Scientist outcome",
        variables={"method": "GGS/intensity LS", "noise": "clean/shot noise"},
        controls=["Same phase-only patterns, TM, splits and fit settings"],
        required_measurements=["Paired camera intensities"], evaluation_criteria=["Held-out prediction errors"])
    state = WorkflowState(user_request=question, proposals=[proposal])
    state.control.active_proposal_ids = [proposal.id]
    state.control.router_event = {"source": "scientist", "outcome": {"kind": "proposal", "explanation": proposal.rationale}}
    state.control.configuration = config.model_dump(mode="json")
    if with_data:
        fixture, _ = create_fixture(config.numerical_input_path / str(uuid4()))
        state.acquisitions.extend(fixture.acquisitions)
    return state


def safety_checks(directory):
    cfg = AppConfig(llm_router_enabled=True, hardware_backend="test", reviewed_numerics_enabled=True,
        workflow_database_path=directory / "safety.sqlite", run_output_path=directory / "runs",
        numerical_input_path=directory / "inputs", hardware_input_path=directory / "hardware-inputs",
        max_workflow_iterations=40, max_workflow_analysis_attempts=6)
    services = double_integrations(cfg, invalid=True)
    invalid_state = make_case(cfg, with_data=False)
    with WorkflowApp(cfg, services) as app:
        rejected = app.start(invalid_state.user_request, initial_state=invalid_state)
    assert rejected["status"] == "blocked" and rejected["state"]["router_decisions"][-1]["status"] == "rejected"
    assert services.hardware_backend.execute_calls == []
    assert rejected["state"]["acquisitions"] == []
    services = double_integrations(cfg)
    state = make_case(cfg, with_data=False)
    with WorkflowApp(cfg, services) as app:
        waiting = app.start(state.user_request, initial_state=state)
        assert waiting["status"] == "awaiting_approval", waiting.get("reason")
        unchanged = app.submit(waiting["workflow_id"], {"action": "approve", "review_fingerprint": "not-the-reviewed-fingerprint"})
        assert unchanged.get("input_error") and services.hardware_backend.execute_calls == []
        assert unchanged["state"]["acquisitions"] == []
    return {"invalid_action_blocked": True, "unapproved_acquisition_calls": len(services.hardware_backend.execute_calls),
            "model": "test_double"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly invoke the configured routing LLM")
    parser.add_argument("--output", type=Path, default=Path("outputs/router-check"))
    args = parser.parse_args()
    root = args.output.resolve() / str(uuid4())
    cfg = AppConfig.from_env() if args.live else AppConfig()
    cfg.llm_router_enabled = True
    cfg.hardware_backend = "test"
    cfg.numerical_backend = "disabled"
    cfg.reviewed_numerics_enabled = True
    cfg.numerical_input_path = root / "inputs"
    cfg.hardware_input_path = root / "hardware-inputs"
    cfg.run_output_path = root / "runs"
    if args.live:
        from workflow.llm import build_integrations
        services = double_integrations(cfg)
        services.router_model = build_integrations(cfg).router_model
    else:
        services = double_integrations(cfg)
    records = []
    scenarios = [("compatible_saved_data", True, True), ("no_data_backend_available", False, True),
                 ("no_data_no_backend", False, False)]
    expected = {"compatible_saved_data": {"run_operation"}, "no_data_backend_available": {"prepare_procedure", "advise"},
                "no_data_no_backend": {"request_information"}}
    for name, with_data, backend_available in scenarios:
        state = make_case(cfg, with_data=with_data)
        selected_services = services if backend_available else Integrations(
            scientist_model=services.scientist_model, experimentor_model=services.experimentor_model,
            analyst_model=services.analyst_model, router_model=services.router_model)
        decision, action = choose_route(state, cfg, selected_services)
        item = {"scenario": name, "model": "live" if args.live else "test_double", **decision.model_dump(mode="json")}
        records.append(item)
        root.mkdir(parents=True, exist_ok=True)
        (root / "router-check.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        assert action is not None and action.kind in expected[name], item
        if with_data:
            assert action.request.operation == "tm.compare_reference_free", item
    safety = safety_checks(root / "safety")
    (root / "safety-checks.json").write_text(json.dumps(safety, indent=2), encoding="utf-8")
    print(f"Routing scenarios passed ({'live router' if args.live else 'test double'}); safety checks: test doubles.")
    print(root / "router-check.json")


if __name__ == "__main__":
    main()
