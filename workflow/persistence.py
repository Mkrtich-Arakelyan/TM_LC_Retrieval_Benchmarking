"""Workflow filesystem persistence primitives, independent of graph execution."""

from contextlib import contextmanager
import os
from pathlib import Path

from workflow.state import WorkflowState


def canonical_factory_path(path: str | None) -> str | None:
    """Resolve historical saved factory references without old import modules."""
    return {
        "llm:build_integrations": "workflow.llm:build_integrations",
        "tools.bench_workflow:build_integrations": "bench.workflow:build_integrations",
        "tools.bench_workflow:build_model_integrations": "bench.workflow:build_model_integrations",
    }.get(path, path)


@contextmanager
def workflow_writer_lock(database: Path):
    # Kernel locks release on process exit, unlike a persistent sentinel file.
    with database.with_suffix(database.suffix + ".lock").open("a+b") as file:
        file.seek(0, os.SEEK_END)
        if not file.tell():
            file.write(b"0")
            file.flush()
        file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another workflow command is active for this database; do not run concurrent acquisition commands.") from exc
        try:
            yield
        finally:
            file.seek(0)
            if os.name == "nt":
                msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file, fcntl.LOCK_UN)


def save_analysis_progress(operation: Path, state: WorkflowState):
    path = operation / "progress.tmp"
    with path.open("w", encoding="utf-8") as file:
        file.write(state.model_dump_json())
        file.flush()
        os.fsync(file.fileno())
    os.replace(path, operation / "progress.json")
