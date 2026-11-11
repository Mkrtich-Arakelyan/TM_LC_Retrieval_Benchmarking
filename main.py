"""Primary CLI for persistent workflows and explicitly selected synthetic benches."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from workflow.config import AppConfig, DEFAULT_ARCHITECTURE_PATH
from workflow.graph import WorkflowApp
from workflow.state import ConversationEntry, WorkflowState
from workflow.report import export_decisions, export_workflow_view
from workflow.persistence import canonical_factory_path

BENCH_FACTORY = "bench.workflow:build_integrations"
MODEL_FACTORY = "bench.workflow:build_model_integrations"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, help="SQLite checkpoint database; reuse on later commands")
    parser.add_argument("--factory", help="Trusted module:function returning workflow.graph.Integrations(config)")
    parser.add_argument("--simulate", action="store_true", help="Explicit synthetic selection; never physical hardware")
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("question", nargs="?", help="Research question; simulation has a default")
    start.add_argument("--fixture", action="store_true", help="Explicit TEST ONLY offline handoffs")
    start.add_argument("--analyst", action="store_true", help="Include canned Analyst in fixture mode")
    start.add_argument("--architecture", type=Path)
    start.add_argument("--test-agents", action="store_true", help="Local simulation agent doubles; actual optics/numerics")
    start.add_argument("--bench-config", type=Path, help="JSON synthetic settings; requires --simulate")
    start.add_argument("--output", type=Path, help="Output parent for a new workflow")
    start.add_argument("--reuse-data-from", metavar="WORKFLOW_ID", help="New analysis-only run from completed acquisitions")
    commands = [start]
    for name in ("show", "resume", "cancel", "review", "clarify", "reconcile", "follow-up"):
        command = sub.add_parser(name)
        commands.append(command)
        command.add_argument("workflow_id")
        if name == "resume":
            command.add_argument("--refresh-config", action="store_true", help="Environment refresh for blocked general workflows")
        if name == "cancel":
            command.add_argument("--reason", default="Cancelled by user")
        if name == "review":
            command.add_argument("decision", choices=["approve", "reject"])
            command.add_argument("--fingerprint", help="Exact review fingerprint; mandatory for approval")
            command.add_argument("--reviewer")
            command.add_argument("--feedback")
        if name in ("clarify", "reconcile", "follow-up"):
            command.add_argument("message")
        if name in ("clarify", "follow-up"):
            command.add_argument("--allow-partial-analysis", action="store_true")
        if name == "reconcile":
            command.add_argument("--confirmed-stopped", action="store_true")
    for command in commands:
        # Allow selection before or after the subcommand without overriding it.
        command.add_argument("--simulate", action="store_true", default=argparse.SUPPRESS,
                             help="Explicitly select/continue a saved simulated workflow")
    return parser


def _validate(args, parser):
    if args.simulate and args.factory and args.factory != BENCH_FACTORY:
        parser.error("--simulate selects its own synthetic integration factory")
    if args.command == "start":
        if args.simulate and (args.fixture or args.reuse_data_from):
            parser.error("--simulate cannot be combined with --fixture or --reuse-data-from")
        if (args.test_agents or args.bench_config) and not args.simulate:
            parser.error("--test-agents and --bench-config require explicit --simulate")
        if args.simulate and (args.architecture or args.analyst):
            parser.error("Simulation uses its synthetic setup context and configured Analyst")
        if not args.question and not args.simulate and not args.reuse_data_from:
            parser.error("start requires a question, --simulate, or --reuse-data-from")
        if args.reuse_data_from and args.fixture:
            parser.error("Saved-data reuse cannot start a fixture acquisition")
    if args.command == "reconcile" and not args.confirmed_stopped:
        parser.error("Reconciliation requires --confirmed-stopped after inspecting the acquisition journal")


def _open(app_factory, config, factory=None):
    return app_factory(config, **({"integration_factory": factory} if factory else {}))


def _load(app_factory, config, workflow_id):
    # No clients/backend are invoked: load checkpoint configuration before continuing.
    with _open(app_factory, config) as app:
        return app.show(workflow_id)


def _saved_config(view, environment):
    config = AppConfig.model_validate(view["state"]["control"]["configuration"])
    config.workflow_database_path = environment.workflow_database_path
    config.api_key = environment.api_key  # current secret only; never checkpoint it
    return config


def _factory(config, explicit=None):
    if config.hardware_backend == "simulated":
        return BENCH_FACTORY
    if config.hardware_backend == "disabled" and config.optical_bench is not None:
        return MODEL_FACTORY
    return explicit


def _simulation(config, args, parser):
    from bench.config import BenchConfig
    from bench.workflow import BENCH_QUESTION, PAPER_PASSAGES, setup_context
    try:
        config.optical_bench = BenchConfig.model_validate_json(args.bench_config.read_text(encoding="utf-8-sig")) if args.bench_config else BenchConfig()
    except (ValueError, OSError) as exc:
        parser.error(f"Invalid synthetic bench configuration: {exc}")
    config.hardware_backend = "simulated"
    config.numerical_backend = "disabled"
    config.bench_test_agents = args.test_agents
    config.llm_router_enabled = True
    config.reviewed_numerics_enabled = True
    config.max_workflow_iterations = max(config.max_workflow_iterations, 40)
    config.max_workflow_analysis_attempts = max(config.max_workflow_analysis_attempts, 6)
    config.hardware_action_timeout_seconds = max(config.hardware_action_timeout_seconds, 180)
    config.numerical_timeout_seconds = max(config.numerical_timeout_seconds, 180)
    config.max_acquisition_actions = 1
    config.max_acquisition_measurements = config.optical_bench.frame_budget
    if not args.test_agents and not (config.llm_provider and config.llm_model and config.api_key):
        parser.error("Live simulation agents require configured LLM credentials; use --test-agents explicitly for local verification")
    context_root = config.run_output_path / "setup-context" / str(uuid4())
    context_root.mkdir(parents=True, exist_ok=False)
    architecture = context_root / "synthetic-architecture.txt"
    architecture.write_text(setup_context(config.optical_bench), encoding="utf-8")
    return args.question or BENCH_QUESTION, {"architecture_path": architecture, "literature_context": PAPER_PASSAGES}


def _reuse(source, config, args):
    from bench.workflow import PAPER_PASSAGES
    saved = WorkflowState.model_validate(source["state"])
    completed = [a for a in saved.acquisitions if a.status == "completed"]
    if not completed:
        raise ValueError("Source workflow has no completed saved acquisitions to reuse")
    # Retain bench model mode, but do not construct an acquisition backend.
    if saved.control.configuration.get("optical_bench") is not None:
        parent_output = config.run_output_path
        config = _saved_config(source, config)
        config.run_output_path = parent_output
    config.hardware_backend = "disabled"
    config.numerical_backend = "disabled"
    config.llm_router_enabled = True
    config.reviewed_numerics_enabled = True
    config.max_workflow_iterations = max(config.max_workflow_iterations, 30)
    config.max_workflow_analysis_attempts = max(config.max_workflow_analysis_attempts, 6)
    config.numerical_input_path = Path(saved.control.configuration["numerical_input_path"])
    request = args.question or saved.user_request
    initial = WorkflowState(user_request=request, proposals=saved.proposals, procedures=saved.procedures,
        acquisitions=completed, acquisition_reviews=saved.acquisition_reviews, approvals=saved.approvals,
        conversation=[ConversationEntry(role="orchestrator", record_ids=[a.id for a in completed],
            message=f"Imported saved acquisitions from workflow {saved.id}; source analyses/decisions are not imported as results of this new run.")])
    return config, request, {"initial_state": initial, "literature_context": PAPER_PASSAGES,
        "architecture_path": args.architecture or DEFAULT_ARCHITECTURE_PATH}


def _continue(app, args, environment):
    if args.command == "resume":
        return app.resume(args.workflow_id, configuration=environment if args.refresh_config else None)
    if args.command == "cancel":
        return app.cancel(args.workflow_id, args.reason)
    if args.command == "review":
        return app.submit(args.workflow_id, {"action": args.decision, "review_fingerprint": args.fingerprint,
            "reviewer": args.reviewer, "feedback": args.feedback})
    if args.command == "clarify":
        response = {"action": "clarify", "message": args.message}
        response["allow_partial_analysis"] = args.allow_partial_analysis
        return app.submit(args.workflow_id, response)
    if args.command == "reconcile":
        return app.submit(args.workflow_id, {"action": "reconcile", "message": args.message,
            "confirmed_stopped": args.confirmed_stopped})
    return app.follow_up(args.workflow_id, args.message, allow_partial_analysis=args.allow_partial_analysis)


def _render(view, config, old_count=0):
    root = export_workflow_view(view)
    simulated = view["state"]["control"]["configuration"].get("hardware_backend") == "simulated"
    if simulated:
        export_decisions(view)
    print(f"Workflow {view['workflow_id']}: {view['status']}")
    if view.get("reason"):
        print(view["reason"])
    if view.get("input_error"):
        print("Input rejected; workflow unchanged: " + view["input_error"])
    for entry in view["state"]["conversation"][old_count:]:
        print(f"\n[{entry['role']}] {entry['message']}")
    print(f"\nSaved workflow: {root}")
    command = f'python main.py --db "{config.workflow_database_path}"'
    selection = " --simulate" if simulated else ""
    pending = view.get("pending")
    if pending:
        print("\nPending human input:\n" + json.dumps(pending, indent=2, ensure_ascii=False))
        if pending["kind"] == "review":
            print(f'Inspect: {command} show {view["workflow_id"]}')
            print(f'{command} review {view["workflow_id"]} approve{selection} --fingerprint '
                  f'{pending["review"]["review_fingerprint"]} --reviewer "Your name"')
        elif pending["kind"] == "reconciliation":
            print(f'{command} reconcile {view["workflow_id"]} "Describe the journal inspection"{selection} --confirmed-stopped')
        else:
            print(f'{command} clarify {view["workflow_id"]} "Your answer"{selection}')
    elif view["status"] == "blocked":
        print(f'{command} resume {view["workflow_id"]}{selection}')
    failed_acquisition = simulated and any(a["status"] in ("failed", "partial", "aborted") for a in view["state"]["acquisitions"])
    return 2 if view["status"] in ("blocked", "failed", "stopped") or view.get("input_error") or failed_acquisition else 0


def main(argv=None, *, app_factory=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.factory = canonical_factory_path(args.factory)
    _validate(args, parser)
    environment = AppConfig.from_env().model_copy(deep=True)
    if args.db:
        environment.workflow_database_path = args.db
    config = environment.model_copy(deep=True)
    application = app_factory or WorkflowApp
    try:
        if args.command == "start":
            if args.output:
                config.run_output_path = args.output
            if args.reuse_data_from:
                config, request, kwargs = _reuse(_load(application, environment, args.reuse_data_from), config, args)
            elif args.simulate:
                request, kwargs = _simulation(config, args, parser)
            else:
                if config.hardware_backend == "simulated" or args.factory == BENCH_FACTORY:
                    parser.error("Use --simulate to explicitly select the synthetic backend")
                request = args.question
                kwargs = {"architecture_path": args.architecture} if args.architecture else {}
            with _open(application, config, _factory(config, args.factory)) as app:
                result = app.start(request, fixture=args.fixture, fixture_analyst=args.analyst, **kwargs)
            return _render(result, config)
        previous = _load(application, environment, args.workflow_id)
        if args.command == "show":
            return _render(previous, environment)
        saved = previous["state"]["control"]["configuration"]
        backend = saved.get("hardware_backend", "disabled")
        if args.command != "cancel":
            if "simulation" in saved or (args.simulate and backend != "simulated") or (backend == "simulated" and not args.simulate):
                parser.error("Saved acquisition backend does not match the explicit selection. Use --simulate to continue a synthetic workflow, or show to inspect. Hardware is never selected automatically.")
        if args.command == "resume" and args.refresh_config and saved.get("optical_bench") is not None:
            parser.error("Bench workflows retain saved settings and agent mode; start a new run to change configuration")
        config = _saved_config(previous, environment)
        with _open(application, config, _factory(config, args.factory)) as app:
            result = _continue(app, args, environment)
        return _render(result, config)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Workflow error: {exc}")
        return 2


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
