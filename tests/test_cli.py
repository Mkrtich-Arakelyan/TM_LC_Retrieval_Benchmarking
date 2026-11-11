"""Shared CLI behavior; real local bench calculations, never live model calls."""

import json
from pathlib import Path

import pytest

import main as cli
from workflow.config import AppConfig
from workflow.graph import WorkflowApp
from bench import workflow as bench_workflow


@pytest.mark.parametrize("argv", [["--simulate", "start"], ["start", "--simulate"]])
def test_explicit_simulation_flag_positions(argv):
    assert cli.build_parser().parse_args(argv).simulate


@pytest.mark.parametrize("argv", [
    ["start", "--test-agents"],
    ["start", "--bench-config", "configs/bench-normal.json"],
    ["start", "--simulate", "--fixture"],
    ["start", "--simulate", "--reuse-data-from", "some-id"],
])
def test_invalid_start_combinations_stop_before_integrations(argv, monkeypatch):
    monkeypatch.setattr(AppConfig, "from_env", lambda: pytest.fail("Invalid command loaded configuration"))
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2


def test_environment_does_not_select_simulation_implicitly(tmp_path, monkeypatch):
    cfg = AppConfig(hardware_backend="simulated", workflow_database_path=tmp_path / "workflow.sqlite")
    monkeypatch.setattr(AppConfig, "from_env", lambda: cfg)
    with pytest.raises(SystemExit) as error:
        cli.main(["start", "A question"])
    assert error.value.code == 2
    assert not cfg.workflow_database_path.exists()


def test_primary_review_uses_saved_mode_and_reuse_never_acquires(tmp_path, monkeypatch):
    cfg = AppConfig(workflow_database_path=tmp_path / "workflow.sqlite", run_output_path=tmp_path / "runs",
                    hardware_input_path=tmp_path / "inputs", numerical_input_path=tmp_path / "inputs")
    monkeypatch.setattr(AppConfig, "from_env", lambda: cfg.model_copy(deep=True))
    assert cli.main(["start", "--simulate", "--test-agents", "--bench-config", "configs/bench-normal.json"]) == 0
    exported = next((tmp_path / "runs").glob("*/workflow.json"))
    waiting = json.loads(exported.read_text(encoding="utf-8"))
    workflow_id = waiting["workflow_id"]
    assert waiting["status"] == "awaiting_approval" and waiting["state"]["acquisitions"] == []
    assert cli.main(["show", workflow_id]) == 0
    fingerprint = waiting["pending"]["review"]["review_fingerprint"]
    with pytest.raises(SystemExit):
        cli.main(["review", workflow_id, "approve", "--fingerprint", fingerprint])
    assert cli.main(["review", workflow_id, "approve", "--simulate", "--fingerprint", "wrong"]) == 2
    with WorkflowApp(cfg) as app:
        assert app.show(workflow_id)["state"]["acquisitions"] == []
    # Contradictory current defaults must not replace saved local mode/controls.
    cfg.hardware_backend = "external"
    cfg.bench_test_agents = False
    cfg.max_workflow_iterations = 1
    assert cli.main(["review", workflow_id, "approve", "--simulate", "--fingerprint", fingerprint,
                     "--reviewer", "CLI regression"]) == 0
    final = json.loads(exported.read_text(encoding="utf-8"))
    assert final["workflow_id"] == workflow_id and final["status"] == "completed"
    assert final["state"]["control"]["configuration"] == waiting["state"]["control"]["configuration"] | {
        "numerical_input_path": final["state"]["control"]["configuration"]["numerical_input_path"]}
    assert final["state"]["analyses"][-1]["metrics"]["prediction_passed"] == 1
    assert cli.main(["show", workflow_id]) == 0
    with pytest.raises(SystemExit):
        cli.main(["resume", workflow_id, "--simulate", "--refresh-config"])
    monkeypatch.setattr(bench_workflow, "SimulatedBackend", lambda *_: pytest.fail("Reuse constructed an acquisition backend"))
    assert cli.main(["start", "--reuse-data-from", workflow_id]) == 0
    views = [json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / "runs").glob("*/workflow.json")]
    reused = next(v for v in views if v["workflow_id"] != workflow_id)
    assert reused["status"] == "completed"
    assert reused["state"]["control"]["configuration"]["hardware_backend"] == "disabled"
    assert reused["state"]["control"]["configuration"]["bench_test_agents"] is True
    assert reused["state"]["acquisitions"] == final["state"]["acquisitions"]
    assert not (Path(reused["state"]["control"]["configuration"]["run_output_path"]) / "acquisitions").exists()
    assert cli.main(["follow-up", reused["workflow_id"], "Interpret the existing results"]) == 0
    with WorkflowApp(cfg) as app:
        assert app.show(reused["workflow_id"])["state"]["acquisitions"] == final["state"]["acquisitions"]


@pytest.mark.parametrize("argv,expected", [
    (["resume", "id"], ("resume", "id", {"configuration": None})),
    (["clarify", "id", "answer"], ("submit", "id",
        {"action": "clarify", "message": "answer", "allow_partial_analysis": False})),
    (["reconcile", "id", "inspected", "--confirmed-stopped"], ("submit", "id",
        {"action": "reconcile", "message": "inspected", "confirmed_stopped": True})),
    (["cancel", "id", "--reason", "stop"], ("cancel", "id", "stop")),
])
def test_commands_dispatch_through_primary_cli(argv, expected, monkeypatch):
    cfg = AppConfig()
    view = {"state": {"control": {"configuration": cfg.model_dump(mode="json")}}}
    calls = []
    class Application:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def resume(self, workflow_id, **kwargs):
            calls.append(("resume", workflow_id, kwargs))
            return view
        def submit(self, workflow_id, response):
            calls.append(("submit", workflow_id, response))
            return view
        def cancel(self, workflow_id, reason):
            calls.append(("cancel", workflow_id, reason))
            return view
    monkeypatch.setattr(AppConfig, "from_env", lambda: cfg)
    monkeypatch.setattr(cli, "_load", lambda *args: view)
    monkeypatch.setattr(cli, "_render", lambda *args: 7)
    assert cli.main(argv, app_factory=lambda *args, **kwargs: Application()) == 7
    assert calls == [expected]
