"""Deterministic hardware boundary. No connections, discovery, or drivers here."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Literal, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from workflow.config import AppConfig
from workflow.state import (AcquisitionAction, AcquisitionResult, AcquisitionReview, ActionProgress,
                   Artifact, ExperimentProcedure, HardwareExecution, HumanApproval,
                   Setting, Text, approval_matches)
from tools.python_execution import contained_file, fingerprint


class ParameterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    kind: Literal["integer", "number", "boolean", "string"]
    unit: Text | None = None
    required: bool = True
    verified: bool = False
    minimum: float | None = None
    maximum: float | None = None
    choices: list[Text] | None = None


class ActionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parameters: dict[Text, ParameterSpec]
    required_calibrations: list[Text] = Field(default_factory=list)
    required_prerequisites: list[Text] = Field(default_factory=list)
    measurement_count_setting: Text | None = None
    measurement_count_multiplier: int = Field(default=1, ge=1, le=1000)


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verified: bool = False
    evidence: Text


class HardwareCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend_id: Text
    revision: Text
    available: bool = False
    data_origin: Literal["physical", "simulated", "test_only"]
    actions: dict[Text, ActionSpec] = Field(default_factory=dict)
    prerequisites: dict[Text, Verification] = Field(default_factory=dict)
    verified_calibrations: dict[Text, Artifact] = Field(default_factory=dict)
    supports_deadlines: bool = False
    supports_cancellation: bool = False
    supports_cleanup: bool = False


class ActionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actual_settings: dict[Text, Setting] = Field(default_factory=dict)
    raw_data: list[Artifact] = Field(default_factory=list)
    errors: list[Text] = Field(default_factory=list)
    aborted: bool = False


class HardwareBackend(Protocol):
    """Trusted harness contract; describe/validate must never alter equipment state.

    execute must enforce current limits atomically, honor the reviewed capability
    fingerprint, use only declared inputs/output_dir, and cooperate with deadlines
    and cancellation. A backend must not claim these guarantees unless verified.
    cleanup is backend-defined, not a universal safe device setting.
    """
    def describe(self) -> HardwareCapabilities: ...
    def validate(self, procedure: ExperimentProcedure) -> list[str]: ...
    def execute(self, action: AcquisitionAction, *, output_dir: Path,
                input_artifacts: list[Artifact], capability_fingerprint: str,
                deadline: float, cancel_event: Event) -> ActionReport: ...
    def cleanup(self, *, reason: str, deadline: float) -> str: ...


def canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class DisabledHardwareBackend:
    def describe(self) -> HardwareCapabilities:
        return HardwareCapabilities(backend_id="disabled", revision="1", data_origin="test_only")

    def validate(self, procedure: ExperimentProcedure) -> list[str]:
        return ["Hardware unavailable: no laboratory backend is configured."]

    def execute(self, *args, **kwargs) -> ActionReport:
        raise RuntimeError("Hardware unavailable")

    def cleanup(self, **kwargs) -> str:
        raise RuntimeError("No hardware cleanup is available")


def parameter_issues(settings: dict[str, Setting], spec: ActionSpec) -> list[str]:
    issues = []
    for name in settings.keys() - spec.parameters.keys():
        issues.append(f"Unsupported parameter: {name}")
    for name, rule in spec.parameters.items():
        setting = settings.get(name)
        if setting is None:
            if rule.required:
                issues.append(f"Missing required parameter: {name}")
            continue
        if not rule.verified:
            issues.append(f"Unverified equipment constraint for {name}")
        if setting.unit != rule.unit:
            issues.append(f"Incorrect unit for {name}: expected {rule.unit!r}, got {setting.unit!r}")
        value = setting.value
        valid = {"integer": type(value) is int, "number": type(value) in (int, float),
                 "boolean": type(value) is bool, "string": type(value) is str}[rule.kind]
        if not valid:
            issues.append(f"Incorrect parameter type for {name}: expected {rule.kind}")
            continue
        if rule.kind in ("integer", "number"):
            if rule.minimum is None or rule.maximum is None or rule.minimum > rule.maximum:
                issues.append(f"Unknown required limits for {name}")
            else:
                try:
                    finite = math.isfinite(value)
                except OverflowError:
                    finite = False
                if not finite or not rule.minimum <= value <= rule.maximum:
                    issues.append(f"Exceeded verified limits for {name}: [{rule.minimum}, {rule.maximum}]")
        if rule.choices is not None and value not in rule.choices:
            issues.append(f"Unsupported value for {name}: {value!r}")
    return issues


def hardware_configuration(config: AppConfig) -> dict:
    settings = {"backend": config.hardware_backend,
            "input_root": str(config.hardware_input_path.resolve()),
            "run_root": str((config.run_output_path / "acquisitions").resolve()),
            "max_actions": config.max_acquisition_actions,
            "max_measurements": config.max_acquisition_measurements,
            "action_timeout_seconds": config.hardware_action_timeout_seconds,
            "run_timeout_seconds": config.hardware_run_timeout_seconds}
    if config.hardware_backend == "simulated" and config.optical_bench is not None:
        settings["optical_bench"] = config.optical_bench.model_dump(mode="json")
    return settings


def preflight(procedure: ExperimentProcedure, config: AppConfig,
              backend: HardwareBackend | None = None) -> AcquisitionReview:
    """Read-only checks; architecture text is deliberately not a source of limits."""
    cfg = AppConfig.model_validate(config.model_dump())
    proc = ExperimentProcedure.model_validate(procedure.model_dump())
    hardware = backend or DisabledHardwareBackend()
    settings = hardware_configuration(cfg)
    issues = []
    try:
        caps = HardwareCapabilities.model_validate(hardware.describe().model_dump())
        backend_issues = hardware.validate(proc.model_copy(deep=True))
        if not isinstance(backend_issues, list) or any(not isinstance(s, str) or not s.strip() for s in backend_issues):
            raise ValueError("Backend validation must return a list of readable issues")
        issues.extend(backend_issues)
    except Exception as exc:
        caps = DisabledHardwareBackend().describe()
        issues.append(f"Backend preflight failed: {exc}")
    if cfg.hardware_backend == "disabled":
        issues.append("Hardware backend is disabled in application configuration")
    if not caps.available:
        issues.append("Hardware backend is unavailable")
    if cfg.hardware_backend == "test" and caps.data_origin != "test_only":
        issues.append("Test configuration cannot operate a physical backend")
    if cfg.hardware_backend == "simulated" and caps.data_origin != "simulated":
        issues.append("Simulated configuration requires an explicitly simulated backend; physical fallback is forbidden")
    if not caps.supports_deadlines or not caps.supports_cancellation:
        issues.append("Backend has not verified bounded execution and cancellation support")
    if cfg.max_acquisition_actions is not None and len(proc.actions) > cfg.max_acquisition_actions:
        issues.append("Configured acquisition action budget exceeded")
    required = set(proc.prerequisites)
    calibrations = set()
    measurements = 0
    for index, action in enumerate(proc.actions):
        spec = caps.actions.get(action.action or "")
        if spec is None:
            issues.append(f"Action {index}: unsupported action identifier {action.action!r}; instructions are not executable")
            continue
        issues.extend(f"Action {index}: {issue}" for issue in parameter_issues(action.requested_settings, spec))
        required.update(spec.required_prerequisites)
        calibrations.update(spec.required_calibrations)
        if spec.measurement_count_setting is not None:
            count = action.requested_settings.get(spec.measurement_count_setting)
            if count is None or type(count.value) is not int or count.value < 0:
                issues.append(f"Action {index}: unknown measurement budget")
            else:
                measurements += count.value * spec.measurement_count_multiplier
        elif cfg.max_acquisition_measurements is not None:
            issues.append(f"Action {index}: no verified measurement budget definition")
    if cfg.max_acquisition_measurements is not None and measurements > cfg.max_acquisition_measurements:
        issues.append("Configured acquisition measurement budget exceeded")
    for prerequisite in sorted(required):
        verification = caps.prerequisites.get(prerequisite)
        if verification is None or not verification.verified:
            issues.append(f"Unresolved prerequisite: {prerequisite}")
    for name in sorted(calibrations | proc.calibration_artifacts.keys()):
        declared = proc.calibration_artifacts.get(name)
        verified = caps.verified_calibrations.get(name)
        if declared is None or verified is None or declared.model_dump() != verified.model_dump():
            issues.append(f"Missing or mismatched verified calibration: {name}")
    input_root = Path(settings["input_root"])
    run_root = Path(settings["run_root"])
    if input_root.is_relative_to(run_root) or run_root.is_relative_to(input_root):
        issues.append("Hardware input and run-output directories must be disjoint")
    relative_output = Path(proc.output_subdirectory)
    if (relative_output.is_absolute() or relative_output.drive or ".." in relative_output.parts
            or str(relative_output) in (".", "")):
        issues.append("Output subdirectory must be a relative path without traversal")
    # No destination is created by preflight. Check existing ancestors only.
    parent = run_root
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        issues.append("Acquisition output destination is not writable")
    for artifact in [*proc.input_artifacts, *proc.calibration_artifacts.values()]:
        try:
            path = contained_file(artifact.path, input_root)
            if artifact.sha256 is None or fingerprint(path) != artifact.sha256:
                issues.append(f"Missing or stale input/calibration content hash: {artifact.path}")
        except (ValueError, OSError) as exc:
            issues.append(f"Invalid input/calibration artifact: {exc}")
    snapshot = caps.model_dump(mode="json")
    content = {"procedure": proc.model_dump(mode="json"), "capabilities": snapshot,
               "configuration": settings, "issues": issues}
    return AcquisitionReview(procedure=proc, procedure_fingerprint=proc.fingerprint(),
        review_fingerprint=canonical_hash(content), status="blocked" if issues else "ready",
        issues=issues, capabilities_snapshot=snapshot, configuration_snapshot=settings)


class HardwareRunResult(BaseModel):
    review: AcquisitionReview
    acquisition: AcquisitionResult | None = None
    issues: list[str] = Field(default_factory=list)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _save_record(path: Path, record: AcquisitionResult) -> None:
    """Atomically replace metadata, never raw measurements. Flush before hardware."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        file.write(record.model_dump_json(indent=2))
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def execute_acquisition(
    procedure: ExperimentProcedure, reviewed: AcquisitionReview, approval: HumanApproval | None,
    config: AppConfig, backend: HardwareBackend | None = None, *,
    attempt_id: UUID | None = None, cancel_event: Event | None = None,
) -> HardwareRunResult:
    """Enforce current review + trusted-caller approval, claim once, and journal.

    No retries or resume. A pre-existing attempt directory is a permanent claim,
    even if an interruption left no metadata. A harness must also serialize access
    to physical equipment across distinct attempt IDs/processes.
    """
    cfg = AppConfig.model_validate(config.model_dump())
    proc = ExperimentProcedure.model_validate(procedure.model_dump())
    approval = HumanApproval.model_validate(approval.model_dump()) if approval else None
    hardware = backend or DisabledHardwareBackend()
    current = preflight(proc, cfg, hardware)
    issues = list(current.issues)
    if reviewed.status != "ready" or current.review_fingerprint != reviewed.review_fingerprint:
        issues.append("Execution-relevant review changed or was blocked; renewed review and approval are required")
    if (approval is None or not approval_matches(approval, proc)
            or approval.review_fingerprint != current.review_fingerprint):
        issues.append("Missing, rejected, stale, or mismatched exact-procedure/review approval")
    if issues:
        return HardwareRunResult(review=current, issues=issues)
    caps = HardwareCapabilities.model_validate(current.capabilities_snapshot)
    attempt_id = attempt_id or uuid4()
    run_root = Path(current.configuration_snapshot["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    directory = run_root / str(attempt_id)
    try:
        directory.mkdir(exist_ok=False)
    except FileExistsError:
        return HardwareRunResult(review=current, issues=[
            "Acquisition attempt already claimed; do not replay started, completed, or incomplete attempts. Inspect its durable record and review recovery first."])
    if not directory.resolve().is_relative_to(run_root):
        raise ValueError("Acquisition directory escaped designated root")
    # Preserve review and caller-supplied approval, not model-created approval.
    (directory / "review.json").write_text(current.model_dump_json(indent=2), encoding="utf-8")
    (directory / "approval.json").write_text(approval.model_dump_json(indent=2), encoding="utf-8")
    hardware_record = HardwareExecution(approval_id=approval.id, review_id=current.id,
        review_fingerprint=current.review_fingerprint, backend_id=caps.backend_id,
        data_origin=caps.data_origin, run_directory=str(directory), started_at=_utcnow(),
        actions=[ActionProgress(index=i, requested=a.model_copy(deep=True)) for i, a in enumerate(proc.actions)])
    record = AcquisitionResult(id=attempt_id, procedure_id=proc.id, procedure_version=proc.version,
        procedure_fingerprint=proc.fingerprint(), status="started", data_origin=caps.data_origin,
        hardware=hardware_record)
    record_path = directory / "record.json"
    _save_record(record_path, record)
    cancellation = cancel_event if cancel_event is not None else Event()
    run_deadline = time.monotonic() + cfg.hardware_run_timeout_seconds
    active = None
    aborted = False
    try:
        for index, action in enumerate(proc.actions):
            if cancellation.is_set() or time.monotonic() >= run_deadline:
                aborted = True
                raise RuntimeError("Acquisition cancelled or run deadline reached; equipment stop must be confirmed")
            # Recheck before every action, including inputs and live backend limits.
            live = preflight(proc, cfg, hardware)
            if live.status != "ready" or live.review_fingerprint != current.review_fingerprint:
                raise RuntimeError("Constraints, calibration, or prerequisites changed during acquisition; renewed review required")
            if not approval_matches(approval, proc) or approval.review_fingerprint != live.review_fingerprint:
                raise RuntimeError("Approval no longer matches execution")
            active = record.hardware.actions[index]
            active.status, active.started_at = "started", _utcnow()
            _save_record(record_path, record)
            output = directory / proc.output_subdirectory / str(index)
            output.mkdir(parents=True, exist_ok=False)
            if not output.resolve().is_relative_to(directory):
                raise ValueError("Raw-data destination escaped run directory")
            deadline = min(run_deadline, time.monotonic() + cfg.hardware_action_timeout_seconds)
            raw = hardware.execute(action.model_copy(deep=True), output_dir=output,
                input_artifacts=[a.model_copy(deep=True) for a in
                                 [*proc.input_artifacts, *proc.calibration_artifacts.values()]],
                capability_fingerprint=canonical_hash(caps.model_dump(mode="json")),
                deadline=deadline, cancel_event=cancellation)
            report = ActionReport.model_validate(raw.model_dump())
            active.actual_settings = report.actual_settings
            for name, setting in report.actual_settings.items():
                record.actual_settings[f"{index}.{name}"] = setting
            for artifact in report.raw_data:
                path = contained_file(artifact.path, output)
                sha = fingerprint(path)
                if artifact.sha256 is not None and artifact.sha256 != sha:
                    raise ValueError(f"Raw-data content hash mismatch: {path}")
                verified_artifact = Artifact(**{**artifact.model_dump(), "path": str(path), "sha256": sha})
                active.artifacts.append(verified_artifact)
                record.raw_data.append(verified_artifact)
            active.errors.extend(report.errors)
            _save_record(record_path, record)
            if report.aborted or cancellation.is_set() or time.monotonic() >= deadline:
                aborted = True
                raise RuntimeError("Action aborted/cancelled or deadline exceeded; physical stop is not guaranteed")
            if report.errors:
                raise RuntimeError("; ".join(report.errors))
            actual_issues = parameter_issues(report.actual_settings, caps.actions[action.action])
            if actual_issues or not report.raw_data:
                raise RuntimeError("Invalid actual settings or missing measurements: " + "; ".join(actual_issues))
            active.status, active.ended_at = "completed", _utcnow()
            _save_record(record_path, record)
            active = None
        record = AcquisitionResult.model_validate({**record.model_dump(), "status": "completed",
            "hardware": {**record.hardware.model_dump(), "ended_at": _utcnow()}})
    except (Exception, KeyboardInterrupt) as exc:
        aborted = aborted or isinstance(exc, KeyboardInterrupt)
        reason = str(exc) or type(exc).__name__
        if active is not None:
            active.status = "aborted" if aborted else "failed"
            active.ended_at = _utcnow()
            active.errors.append(reason)
        record.errors.append(reason)
        record.hardware.cleanup_status = "pending" if caps.supports_cleanup else "unsupported"
        try:
            _save_record(record_path, record)
        except OSError as journal_error:
            record.errors.append(f"Progress persistence failed: {journal_error}")
        if caps.supports_cleanup:
            try:
                cleanup_deadline = time.monotonic() + cfg.hardware_action_timeout_seconds
                record.hardware.cleanup_message = hardware.cleanup(
                    reason=reason, deadline=cleanup_deadline)
                if time.monotonic() >= cleanup_deadline:
                    raise RuntimeError("Cleanup deadline exceeded; physical equipment state remains uncertain")
                record.hardware.cleanup_status = "completed"
            except (Exception, KeyboardInterrupt) as cleanup_error:
                record.hardware.cleanup_status = "failed"
                record.hardware.cleanup_message = str(cleanup_error) or type(cleanup_error).__name__
                record.errors.append(f"Backend cleanup failed: {record.hardware.cleanup_message}")
        record = AcquisitionResult.model_validate({**record.model_dump(),
            "status": "aborted" if aborted else ("partial" if record.raw_data else "failed"),
            "hardware": {**record.hardware.model_dump(), "ended_at": _utcnow()}})
    _save_record(record_path, record)
    return HardwareRunResult(review=current, acquisition=record,
                             issues=list(record.errors))
