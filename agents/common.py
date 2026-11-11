"""Small shared chat-model interface and editable context reader."""

from pathlib import Path
from typing import Protocol, Sequence

from pydantic import BaseModel

from workflow.state import LiteraturePassage, ScientificContext
from workflow.config import DEFAULT_ARCHITECTURE_PATH, resolve_architecture_path

NO_LITERATURE_NOTICE = "Recommendations have not been checked against the project's papers."


class StructuredInvoker(Protocol):
    def invoke(self, messages: list[tuple[str, str]]) -> object: ...


class ChatModel(Protocol):
    def with_structured_output(self, schema: type[BaseModel]) -> StructuredInvoker: ...


def read_context(
    architecture_path: str | Path,
    literature_context: Sequence[LiteraturePassage] | None,
) -> ScientificContext:
    path = resolve_architecture_path(architecture_path)
    try:
        with path.open(encoding="utf-8", newline="") as file:
            architecture = file.read()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read architecture file {path}: {exc}") from exc
    if not architecture.strip():
        raise ValueError(f"Architecture file is empty: {path}")
    return ScientificContext(
        architecture_path=str(path.resolve()), architecture_text=architecture,
        literature_context=[LiteraturePassage.model_validate(p.model_dump())
                            for p in (literature_context or [])],
    )
