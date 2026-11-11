"""Root-file moves preserve configuration, context selection, and saved factories."""

import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import main as cli
from agents.common import read_context
from fixtures.numerical import create_fixture
from workflow import config as settings, llm
from workflow.config import AppConfig, DEFAULT_ARCHITECTURE_PATH, DEFAULT_DOTENV_PATH
from workflow.graph import Integrations, WorkflowApp
from workflow.state import WorkflowState


def test_default_dotenv_stays_at_repository_root_and_explicit_override_wins(tmp_path, monkeypatch):
    import dotenv

    root = Path(__file__).resolve().parents[1]
    assert DEFAULT_DOTENV_PATH == root / ".env"
    assert DEFAULT_ARCHITECTURE_PATH == root / "configs" / "architecture.txt"
    assert inspect.signature(WorkflowApp.start).parameters["architecture_path"].default == DEFAULT_ARCHITECTURE_PATH
    override = tmp_path / "custom.env"
    seen = []
    def values(path, **kwargs):
        seen.append(path)
        return {"TM_LLM_MODEL": "test-default" if path == DEFAULT_DOTENV_PATH else "test-override"}
    monkeypatch.setattr(dotenv, "dotenv_values", values)
    monkeypatch.setattr(Path, "is_file", lambda path: path in (DEFAULT_DOTENV_PATH, override))
    monkeypatch.delenv("TM_LLM_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)
    before = dict(os.environ)
    assert AppConfig.from_env().llm_model == "test-default"
    assert AppConfig.from_env(override).llm_model == "test-override"
    assert seen == [DEFAULT_DOTENV_PATH, override]
    assert dict(os.environ) == before


def test_exact_saved_legacy_architecture_default_still_reads_same_contents():
    legacy = settings.REPOSITORY_ROOT / "architecture.txt"
    context = read_context(legacy, [])
    expected = read_context(DEFAULT_ARCHITECTURE_PATH, [])
    assert context == expected
    assert not legacy.exists()


@pytest.mark.parametrize("name", ["custom/architecture.txt", "setup-context/synthetic-architecture.txt"])
def test_custom_and_generated_architecture_paths_are_not_redirected(tmp_path, name):
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"Custom context\r\nKeep exact newlines.\r\n")
    context = read_context(path, [])
    assert context.architecture_path == str(path.resolve())
    assert context.architecture_text == "Custom context\r\nKeep exact newlines.\r\n"


def test_missing_custom_architecture_does_not_fall_back_to_default(tmp_path):
    with pytest.raises(ValueError, match="Cannot read architecture file"):
        read_context(tmp_path / "architecture.txt", [])


def test_saved_data_reuse_shares_default_and_preserves_explicit_override(tmp_path):
    saved, _ = create_fixture(tmp_path / "inputs")
    config = AppConfig(numerical_input_path=tmp_path / "inputs", run_output_path=tmp_path / "runs")
    saved.control.configuration = config.model_dump(mode="json")
    source = {"state": saved.model_dump(mode="json")}
    args = SimpleNamespace(question=None, architecture=None)
    _, _, context = cli._reuse(source, config.model_copy(deep=True), args)
    assert context["architecture_path"] == DEFAULT_ARCHITECTURE_PATH
    args.architecture = tmp_path / "custom.txt"
    _, _, context = cli._reuse(source, config.model_copy(deep=True), args)
    assert context["architecture_path"] == args.architecture


def test_historical_live_factory_resolves_without_constructing_live_clients(tmp_path, monkeypatch):
    services = Integrations()
    seen = []
    def build(config):
        seen.append(config)
        return services
    monkeypatch.setattr(llm, "build_integrations", build)
    config = AppConfig(workflow_database_path=tmp_path / "workflow.sqlite")
    state = WorkflowState(user_request="Offline factory migration check")
    state.control.configuration = config.model_dump(mode="json")
    with WorkflowApp(config, integration_factory="llm:build_integrations") as app:
        assert app.factory == "workflow.llm:build_integrations"
        assert app._services(state) is app._services(state) is services
    assert len(seen) == 1
