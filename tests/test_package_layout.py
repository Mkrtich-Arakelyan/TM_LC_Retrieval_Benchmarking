"""Direct package imports, saved factory migration, and reviewed source audits."""

import importlib
from pathlib import Path

from bench.config import BenchConfig
from workflow.config import AppConfig
from fixtures.numerical import create_fixture
from workflow.graph import WorkflowApp
from workflow.state import WorkflowState
from tools.python_execution import fingerprint
from tools.tm_numerics import execute_operation
from workflow.persistence import canonical_factory_path


def test_tools_contains_only_execution_interfaces():
    root = Path(__file__).resolve().parents[1] / "tools"
    assert {path.name for path in root.glob("*.py")} == {
        "__init__.py", "hardware.py", "tm_numerics.py", "python_execution.py",
    }


def test_factory_migration_leaves_custom_factories_and_missing_selection_unchanged():
    assert canonical_factory_path(None) is None
    assert canonical_factory_path("custom.integrations:build") == "custom.integrations:build"
    assert canonical_factory_path("bench.workflow:build_integrations") == "bench.workflow:build_integrations"


def test_saved_simulation_and_model_only_factories_resolve_through_graph(tmp_path):
    for module in ("tools.bench_workflow", "bench.workflow"):
        for factory, backend in (("build_integrations", "simulated"),
                                 ("build_model_integrations", "disabled")):
            cfg = AppConfig(workflow_database_path=tmp_path / "workflow.sqlite",
                            hardware_backend=backend, optical_bench=BenchConfig(), bench_test_agents=True)
            state = WorkflowState(user_request="Resolve a saved factory with local agents")
            state.control.configuration = cfg.model_dump(mode="json")
            with WorkflowApp(cfg, integration_factory=f"{module}:{factory}") as app:
                assert app.factory == f"bench.workflow:{factory}"
                services = app._services(state)
                assert services.scientist_model is not None
                assert (services.hardware_backend is not None) == (backend == "simulated")


def test_reviewed_audit_covers_relocated_support_without_changing_output_schema(tmp_path):
    state, request = create_fixture(tmp_path / "inputs")
    cfg = AppConfig(numerical_input_path=tmp_path / "inputs", run_output_path=tmp_path / "runs",
                    reviewed_numerics_enabled=True)
    request.operation = "tm.validate_dataset"
    result = execute_operation(state, request, cfg)
    assert result.status == "completed", result.errors
    assert len(result.outputs) == 1 and Path(result.outputs[0].path).name == "report.json"
    audit = Path(result.code.path)
    text = audit.read_text(encoding="utf-8")
    for module in ("analysis.datasets", "analysis.metrics", "analysis.tm_analysis", "analysis.tm_solvers"):
        implementation = importlib.import_module(module)
        snapshot = audit.parent / "reviewed_sources" / f"{module}.py"
        assert snapshot.read_bytes() == Path(implementation.__file__).read_bytes()
        assert f"{snapshot.name} sha256={fingerprint(snapshot)}" in text
    assert result.code.sha256 == fingerprint(audit)
