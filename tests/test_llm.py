"""Credential loading and provider wiring; no API calls or real keys."""

import json
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from workflow.config import AppConfig
from workflow.graph import Integrations, WorkflowApp
from workflow.llm import JsonChatModel, build_integrations


def test_dotenv_precedence_literal_secret_and_no_environment_mutation(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("TM_"):
            monkeypatch.delenv(name)
    path = tmp_path / ".env"
    path.write_text('TM_LLM_PROVIDER=gemini\nTM_LLM_MODEL=file-model\n'
                    'TM_API_KEY="fixture-${UNEXPANDED}-key"\n', encoding="utf-8-sig")
    monkeypatch.setenv("TM_LLM_MODEL", "environment-model")
    before = dict(os.environ)
    cfg = AppConfig.from_env(path)
    assert cfg.llm_provider == "gemini" and cfg.llm_model == "environment-model"
    assert cfg.api_key.get_secret_value() == "fixture-${UNEXPANDED}-key"
    assert "fixture-" not in cfg.model_dump_json() and "fixture-" not in repr(cfg)
    assert dict(os.environ) == before
    # An explicitly empty environment setting overrides the file too.
    monkeypatch.setenv("TM_API_KEY", "")
    assert AppConfig.from_env(path).api_key is None


@pytest.mark.parametrize("provider,module_name,class_name", [
    ("openai", "langchain_openai", "ChatOpenAI"),
    ("gemini", "langchain_google_genai", "ChatGoogleGenerativeAI"),
])
def test_provider_construction_and_agent_schema_roundtrip(monkeypatch, provider, module_name, class_name):
    from agents.scientist import ScientistOutcome
    from agents.experimentor import ExperimentorOutcome
    from workflow.state import AnalystReview

    calls = []

    class Client:
        def __init__(self, **kwargs):
            calls.append(kwargs)
        def bind(self, **kwargs):
            self.options = kwargs
            return self
        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(content=self.response)

    module = ModuleType(module_name)
    setattr(module, class_name, Client)
    monkeypatch.setitem(sys.modules, module_name, module)
    services = build_integrations(AppConfig(llm_provider=provider, llm_model="fixture-model",
                                           api_key="fixture-key", llm_router_enabled=True))
    assert services.scientist_model is services.analyst_model is services.experimentor_model is services.router_model
    assert calls == [{"model": "fixture-model", "api_key": "fixture-key", "timeout": 120,
                      "max_retries": 2, **({"vertexai": False} if provider == "gemini" else {})}]
    client = services.scientist_model.client
    assert client.options == ({"response_format": {"type": "json_object"}} if provider == "openai"
                              else {"response_mime_type": "application/json"})
    # Every agent's actual schema can be sent without provider schema restrictions.
    for schema in (ScientistOutcome, ExperimentorOutcome, AnalystReview):
        invoker = services.scientist_model.with_structured_output(schema)
        client.response = "{}"
        with pytest.raises(ValidationError):
            invoker.invoke([("user", "fixture request")])
        assert json.loads(client.messages[-1][1].split("\n", 1)[1]) == schema.model_json_schema()

    class Result(BaseModel):
        settings: dict[str, float]

    client.response = [{"type": "thinking", "text": "excluded"},
                       {"type": "text", "text": '{"settings":{"gain":2}}'}]
    assert services.analyst_model.with_structured_output(Result).invoke([]).settings == {"gain": 2}
    client.response = "```json\n{}\n```"
    with pytest.raises(ValidationError):
        services.analyst_model.with_structured_output(Result).invoke([])
    client.response = []
    with pytest.raises(ValueError, match="no JSON"):
        services.analyst_model.with_structured_output(Result).invoke([])


@pytest.mark.parametrize("settings", [
    {}, {"llm_provider": "unsupported"}, {"llm_provider": "openai", "llm_model": "fixture"},
    {"llm_provider": "gemini", "api_key": "fixture"},
])
def test_incomplete_provider_settings_fail_before_client_creation(settings):
    with pytest.raises(ValueError, match="TM_"):
        build_integrations(AppConfig(**settings))


def test_workflow_automatically_uses_config_but_preserves_explicit_injection(tmp_path, monkeypatch):
    from workflow import llm
    from workflow.state import WorkflowState

    cfg = AppConfig(workflow_database_path=tmp_path / "workflow.sqlite",
                    hardware_input_path=tmp_path / "fixture-inputs",
                    llm_provider="openai", llm_model="fixture-model", api_key="fixture-key")
    seen = []
    services = Integrations(scientist_model=JsonChatModel(object()))
    def build(runtime_config):
        seen.append(runtime_config)
        return services
    monkeypatch.setattr(llm, "build_integrations", build)
    state = WorkflowState(user_request="Offline wiring check")
    state.control.configuration = cfg.model_dump(mode="json")
    assert "fixture-key" not in state.model_dump_json()
    with WorkflowApp(cfg) as app:
        assert app.factory == "workflow.llm:build_integrations"
        assert app._services(state) is services
        assert app._services(state) is services
    assert len(seen) == 1 and seen[0].api_key.get_secret_value() == "fixture-key"
    with WorkflowApp(cfg, services) as app:
        assert app.factory is None and app._services(state) is services
    state.control.fixture_mode = True
    with WorkflowApp(cfg) as app:
        assert app._services(state).scientist_model is not services.scientist_model
    assert len(seen) == 1
