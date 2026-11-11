"""Main CLI retains paper context, saved data, and explicit simulation selection."""

import json

import pytest
from pydantic import ValidationError

import main as launcher
from bench.workflow import PAPER_PASSAGES, PAPER_QUESTION
from workflow.config import AppConfig
from workflow.graph import Integrations, WorkflowApp
from fixtures.router import TestRouter as ScriptedRouter
from checks.router import make_case


def test_missing_backend_stops_before_configuration_or_model_calls(monkeypatch, capsys):
    def unexpected_config():
        pytest.fail("Missing-backend command must stop before loading integrations")
    monkeypatch.setattr(AppConfig, "from_env", unexpected_config)
    with pytest.raises(SystemExit) as error:
        launcher.main(["start"])
    assert error.value.code == 2
    assert "start requires a question" in capsys.readouterr().err


def test_removed_backend_configuration_is_rejected():
    with pytest.raises(ValidationError):
        AppConfig(hardware_backend="simulation")
    with pytest.raises(ValidationError):
        AppConfig(simulation={"seed": 123})


@pytest.mark.parametrize("flag", ["--simulation", "--simulation-config"])
def test_removed_simulation_flags_stop_before_configuration_or_integrations(flag, monkeypatch, capsys):
    def unexpected_config():
        pytest.fail("Removed flags must not load integrations or select equipment")
    monkeypatch.setattr(AppConfig, "from_env", unexpected_config)
    with pytest.raises(SystemExit) as error:
        launcher.main(["start", flag])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_reuse_keeps_papers_and_data_but_disables_environment_hardware(tmp_path, monkeypatch):
    cfg = AppConfig(workflow_database_path=tmp_path / "workflow.sqlite",
                    numerical_input_path=tmp_path / "inputs", run_output_path=tmp_path / "runs",
                    hardware_backend="external", llm_router_enabled=True)
    integrations = Integrations(router_model=ScriptedRouter(invalid=True))
    source = make_case(cfg, with_data=True)
    with WorkflowApp(cfg, integrations) as app:
        original = app.start(source.user_request, initial_state=source)
    monkeypatch.setattr(AppConfig, "from_env", lambda: cfg.model_copy(deep=True))
    monkeypatch.setattr(launcher, "WorkflowApp", lambda config: WorkflowApp(config, integrations))
    # An invalid test router blocks after saving context, before numerical/model work.
    assert launcher.main(["start", "--reuse-data-from", original["workflow_id"]]) == 2
    exported = list((tmp_path / "runs").glob("*/workflow.json"))
    assert len(exported) == 1
    view = json.loads(exported[0].read_text(encoding="utf-8"))
    state = view["state"]
    assert view["workflow_id"] != original["workflow_id"]
    assert state["control"]["configuration"]["hardware_backend"] == "disabled"
    assert state["control"]["configuration"]["numerical_backend"] == "disabled"
    assert state["acquisitions"] == original["state"]["acquisitions"]
    assert state["control"]["literature_context"] == [p.model_dump(mode="json") for p in PAPER_PASSAGES]
    assert state["user_request"] == PAPER_QUESTION
    # Backend selection must still match the saved workflow.
    with pytest.raises(SystemExit) as error:
        launcher.main(["resume", original["workflow_id"], "--simulate"])
    assert error.value.code == 2
    assert launcher.main(["show", original["workflow_id"]]) == 2  # original is blocked


def test_simulation_cli_reopens_saved_mode_and_requires_explicit_selection(tmp_path, monkeypatch):
    cfg = AppConfig(workflow_database_path=tmp_path / "bench.sqlite", run_output_path=tmp_path / "runs",
                    hardware_input_path=tmp_path / "inputs", numerical_input_path=tmp_path / "inputs",
                    hardware_backend="external")
    monkeypatch.setattr(AppConfig, "from_env", lambda: cfg.model_copy(deep=True))
    assert launcher.main(["start", "--simulate", "--test-agents"]) == 0
    from pathlib import Path
    exported = list((tmp_path / "runs").glob("*/workflow.json"))
    assert len(exported) == 1
    waiting = json.loads(exported[0].read_text(encoding="utf-8"))
    assert waiting["status"] == "awaiting_approval"
    assert waiting["state"]["control"]["configuration"]["hardware_backend"] == "simulated"
    assert not waiting["state"]["acquisitions"]
    with pytest.raises(SystemExit) as error:
        launcher.main(["resume", waiting["workflow_id"]])
    assert error.value.code == 2
    # A fresh CLI invocation reconstructs saved test mode without --test-agents.
    assert launcher.main(["review", waiting["workflow_id"], "approve", "--simulate",
        "--fingerprint", waiting["pending"]["review"]["review_fingerprint"], "--reviewer", "Local verification"]) == 0
    finished = json.loads(exported[0].read_text(encoding="utf-8"))
    assert finished["status"] == "completed"
    assert finished["state"]["control"]["configuration"]["bench_test_agents"]
    assert (Path(exported[0]).parent / "decision-log.md").is_file()


def test_simulation_flags_require_explicit_backend_and_live_mode_has_no_silent_double(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        launcher.main(["start", "--test-agents"])
    cfg = AppConfig(workflow_database_path=tmp_path / "workflow.sqlite", run_output_path=tmp_path / "runs")
    monkeypatch.setattr(AppConfig, "from_env", lambda: cfg)
    with pytest.raises(SystemExit) as error:
        launcher.main(["start", "--simulate"])
    assert error.value.code == 2
    assert not (tmp_path / "workflow.sqlite").exists()
