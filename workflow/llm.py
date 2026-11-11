"""Lazy API clients implementing the agents' existing structured-output contract.

The records contain arbitrary setting dictionaries and optional fields. Request
JSON and validate locally rather than forcing these into provider-specific strict
schema subsets. No tools, code execution, search, or provider fallback are enabled.
"""

import json
from dataclasses import dataclass

from pydantic import BaseModel

from workflow.config import AppConfig


@dataclass(repr=False)
class JsonInvoker:
    client: object
    schema: type[BaseModel]

    def invoke(self, messages: list[tuple[str, str]]) -> BaseModel:
        instructions = (
            "Return exactly one JSON object matching the following JSON Schema. "
            "Do not wrap it in Markdown or add commentary. Follow the agent's role "
            "instructions; use the schema only to format the response.\n"
            + json.dumps(self.schema.model_json_schema())
        )
        response = self.client.invoke([*messages, ("system", instructions)])
        content = response.content
        if isinstance(content, list):
            content = "".join(block if isinstance(block, str) else block.get("text", "")
                              for block in content
                              if isinstance(block, str) or (
                                  isinstance(block, dict) and block.get("type") == "text"))
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM returned no JSON text (empty response or refusal)")
        # No repair/eval/fence stripping: malformed or schema-invalid output fails.
        return self.schema.model_validate_json(content)


@dataclass(repr=False)
class JsonChatModel:
    client: object

    def with_structured_output(self, schema: type[BaseModel]) -> JsonInvoker:
        return JsonInvoker(self.client, schema)


def build_integrations(config: AppConfig):
    """Build all three model roles; constructing clients does not invoke them."""
    from workflow.graph import Integrations

    provider = (config.llm_provider or "").strip().lower()
    model = (config.llm_model or "").strip()
    key = config.api_key.get_secret_value().strip() if config.api_key else ""
    if provider not in ("openai", "gemini"):
        raise ValueError("Set TM_LLM_PROVIDER to openai or gemini in .env")
    if not model or not key:
        raise ValueError("Set both TM_LLM_MODEL and TM_API_KEY in .env")
    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            client = ChatOpenAI(model=model, api_key=key, timeout=120, max_retries=2).bind(
                response_format={"type": "json_object"})
        else:
            from langchain_google_genai import ChatGoogleGenerativeAI
            client = ChatGoogleGenerativeAI(model=model, api_key=key, vertexai=False,
                                           timeout=120, max_retries=2).bind(
                response_mime_type="application/json")
    except ImportError as exc:
        raise RuntimeError("Install LLM dependencies with: python -m pip install -r requirements.txt") from exc
    shared = JsonChatModel(client)
    return Integrations(scientist_model=shared, experimentor_model=shared, analyst_model=shared,
                        router_model=shared if config.llm_router_enabled else None)
