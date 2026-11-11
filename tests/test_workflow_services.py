"""Factory isolation through actual graph resolution; injected lifetime unchanged."""

from workflow.config import AppConfig
from workflow.graph import WorkflowApp
from workflow.state import WorkflowState
from bench import workflow as bench_workflow
from bench.config import BenchConfig


def test_factory_services_are_scoped_to_saved_workflow_a_b_a(tmp_path, monkeypatch):
    bench_a = BenchConfig(seed=17)
    bench_b = BenchConfig(seed=29, shutter_seconds=.004, shutter_min=.002, shutter_max=.04)
    cfg_a = AppConfig(workflow_database_path=tmp_path / "workflow.sqlite", run_output_path=tmp_path / "runs",
        hardware_input_path=tmp_path / "inputs", numerical_input_path=tmp_path / "inputs",
        hardware_backend="simulated", optical_bench=bench_a, bench_test_agents=True,
        llm_router_enabled=True, reviewed_numerics_enabled=True, max_workflow_iterations=40)
    cfg_b = cfg_a.model_copy(deep=True, update={"optical_bench": bench_b})
    calls = []
    build = bench_workflow.build_integrations
    def factory(config):
        calls.append(config.optical_bench)
        return build(config)
    monkeypatch.setattr(bench_workflow, "build_integrations", factory)
    for label, bench in (("A", bench_a), ("B", bench_b)):
        (tmp_path / f"{label}.txt").write_text(bench_workflow.setup_context(bench), encoding="utf-8")
    with WorkflowApp(cfg_a, integration_factory="bench.workflow:build_integrations") as app:
        a = app.start(bench_workflow.BENCH_QUESTION, architecture_path=tmp_path / "A.txt")
        app.config = cfg_b
        b = app.start(bench_workflow.BENCH_QUESTION, architecture_path=tmp_path / "B.txt")
        assert a["status"] == b["status"] == "awaiting_approval"
        a_state = WorkflowState.model_validate(app.show(a["workflow_id"])["state"])
        b_state = WorkflowState.model_validate(app.show(b["workflow_id"])["state"])
        a_first, b_services, a_again = [app._services(s) for s in (a_state, b_state, a_state)]
        assert a_first.hardware_backend.config == bench_a
        assert b_services.hardware_backend.config == bench_b
        assert a_first is a_again and a_first is not b_services
        assert calls == [bench_a, bench_b]
        assert not a["state"]["acquisitions"] and not b["state"]["acquisitions"]


def test_explicit_injected_integrations_keep_caller_owned_lifetime(tmp_path):
    cfg = AppConfig(workflow_database_path=tmp_path / "workflow.sqlite",
                    hardware_backend="simulated", optical_bench=BenchConfig(), bench_test_agents=True)
    injected = bench_workflow.build_integrations(cfg)
    a = WorkflowState(user_request="A")
    b = WorkflowState(user_request="B")
    a.control.configuration = cfg.model_dump(mode="json")
    b.control.configuration = cfg.model_copy(update={"optical_bench": BenchConfig(seed=29)}).model_dump(mode="json")
    with WorkflowApp(cfg, injected) as app:
        assert app.factory is None
        assert all(app._services(state) is injected for state in (a, b, a))
