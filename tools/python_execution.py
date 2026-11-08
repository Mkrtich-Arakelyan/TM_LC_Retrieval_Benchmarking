"""Artifact policy and numerical execution interface. No host execution backend.

The only active backend mode is explicitly for trusted fake backends in tests.
Path checks are not a sandbox. A real backend requires a separately reviewed
isolation implementation; none is available in this step.
"""

from __future__ import annotations

import hashlib
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from config import AppConfig
from state import AnalysisCode, Artifact, ExecutionDetails, NumericalAnalysisResult


class BackendReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    elapsed_seconds: float = Field(ge=0)
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    outputs: list[Artifact] = Field(default_factory=list)


class ExecutionBackend(Protocol):
    def run(self, *, script: Path, inputs: list[Path], output_dir: Path,
            method_settings: dict[str, JsonValue], random_seed: int | None,
            timeout_seconds: int) -> BackendReport: ...


def contained_file(path: str | Path, root: Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"Artifact must be an existing file inside {root}: {path}")
    return resolved


def roots(config: AppConfig) -> tuple[Path, Path]:
    inputs = config.numerical_input_path.resolve()
    outputs = (config.run_output_path / "analysis").resolve()
    if inputs.is_relative_to(outputs) or outputs.is_relative_to(inputs):
        raise ValueError("Input and analysis output directories must be disjoint")
    return inputs, outputs


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_inputs(artifacts: list[Artifact], config: AppConfig) -> list[Artifact]:
    input_root, _ = roots(config)
    checked = []
    for artifact in artifacts:
        path = contained_file(artifact.path, input_root)
        sha = fingerprint(path)
        if artifact.sha256 is not None and artifact.sha256 != sha:
            raise ValueError(f"Input artifact hash mismatch: {path}")
        checked.append(Artifact(**{**artifact.model_dump(), "path": str(path), "sha256": sha}))
    return checked


def save_script(source: str, config: AppConfig) -> Artifact:
    if not source.strip():
        raise ValueError("Analysis script must not be empty")
    _, output_root = roots(config)
    directory = output_root / str(uuid4())
    directory.mkdir(parents=True, exist_ok=False)
    # Recheck after directory creation to reject pre-existing symlink escapes.
    if not directory.resolve().is_relative_to(output_root):
        raise ValueError("Analysis script directory escaped output root")
    path = directory / "analysis.py"
    path.write_bytes(source.encode("utf-8"))
    return Artifact(path=str(path), description="Generated Python analysis; not yet executed",
                    media_type="text/x-python", sha256=fingerprint(path))


def validated_metrics(metrics: dict) -> dict[str, float]:
    clean = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.strip() or type(value) not in (int, float):
            raise ValueError(f"Invalid or non-finite metric: {key!r}")
        try:
            number = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid or non-finite metric: {key!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"Invalid or non-finite metric: {key!r}")
        clean[key] = number
    return clean


def execute_python(
    code: AnalysisCode, config: AppConfig, *, attempt_number: int,
    backend: ExecutionBackend | None = None,
) -> NumericalAnalysisResult:
    """Validate policy and record one attempt; never run a host subprocess.

    Timeout is a contract for an injected backend, not a security guarantee.
    Attempt budgets are also checked by the agent before each call.
    """
    config = AppConfig.model_validate(config.model_dump())
    if not 1 <= attempt_number <= config.max_execution_attempts:
        raise ValueError("Execution-attempt limit exceeded")
    _, output_root = roots(config)
    inputs = checked_inputs(code.input_data, config)
    script = contained_file(code.code.path, output_root)
    if not code.code.sha256 or fingerprint(script) != code.code.sha256:
        raise ValueError("Script hash mismatch")
    common = dict(acquisition_ids=code.acquisition_ids, input_data=inputs,
                  code=code.code, code_version=code.code.sha256,
                  code_record_id=code.id, method_settings=code.method_settings,
                  random_seed=code.random_seed)
    if config.numerical_backend == "disabled" or backend is None:
        reason = "Execution unavailable: no isolated backend is configured; generated code was not executed."
        return NumericalAnalysisResult(**common, execution=ExecutionDetails(
            status="unavailable", backend=config.numerical_backend,
            attempt_number=attempt_number, reason=reason))

    directory = script.parent / f"attempt-{attempt_number}-{uuid4()}"
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    report = None
    try:
        raw = backend.run(script=script, inputs=[Path(a.path) for a in inputs], output_dir=directory,
                          method_settings=deepcopy(code.method_settings), random_seed=code.random_seed,
                          timeout_seconds=config.numerical_timeout_seconds)
        report = BackendReport.model_validate(raw.model_dump() if isinstance(raw, BaseModel) else raw)
        elapsed = max(report.elapsed_seconds, time.monotonic() - started)
        if elapsed > config.numerical_timeout_seconds:
            raise ValueError("Execution exceeded the configured time limit")
        if report.exit_code != 0:
            raise ValueError(f"Execution failed with exit code {report.exit_code}")
        metrics = validated_metrics(report.metrics)
        outputs = []
        for artifact in report.outputs:
            path = contained_file(artifact.path, directory)
            sha = fingerprint(path)
            if artifact.sha256 is not None and sha != artifact.sha256:
                raise ValueError(f"Output hash mismatch: {path}")
            outputs.append(Artifact(**{**artifact.model_dump(), "path": str(path), "sha256": sha}))
        if not metrics and not outputs:
            raise ValueError("Execution produced no metrics or output artifacts")
        # Detect accidental corruption by a trusted test backend; not prevention.
        checked_inputs(inputs, config)
        if fingerprint(script) != code.code.sha256:
            raise ValueError("Script changed during execution")
        return NumericalAnalysisResult(**common, status="completed", metrics=metrics, outputs=outputs,
                                       execution=ExecutionDetails(status="completed", backend="test",
                                       attempt_number=attempt_number, exit_code=0,
                                       stdout=report.stdout, stderr=report.stderr, elapsed_seconds=elapsed))
    except Exception as exc:
        return NumericalAnalysisResult(**common, status="failed", errors=[str(exc) or type(exc).__name__],
                                       execution=ExecutionDetails(status="failed", backend="test",
                                       attempt_number=attempt_number,
                                       exit_code=report.exit_code if report else None,
                                       stdout=report.stdout if report else "",
                                       stderr=report.stderr if report else "",
                                       elapsed_seconds=max(time.monotonic() - started,
                                                           report.elapsed_seconds if report else 0),
                                       reason=str(exc) or type(exc).__name__))
