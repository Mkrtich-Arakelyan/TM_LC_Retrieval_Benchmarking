"""Runner boundary tests: no LLM, hardware, network, or optical simulation."""

import json
from pathlib import Path
from threading import Event
from uuid import uuid4

import pytest

from agents.runner import runner
from fixtures.hardware import RecordingFakeBackend
from workflow.state import (AcquisitionAction, AcquisitionResult, Artifact, ExperimentProcedure,
                   HumanApproval, Setting, WorkflowState)
from tools.hardware import (ActionReport, DisabledHardwareBackend,
                            execute_acquisition, preflight)
from tools.python_execution import fingerprint


@pytest.fixture
def hardware_setup(experiment_setup):
    state, cfg, _ = experiment_setup
    cfg.hardware_backend = "test"
    cfg.hardware_input_path = cfg.numerical_input_path
    path = cfg.hardware_input_path / "calibration.json"
    path.write_text('{"calibration": "TEST ONLY"}')
    artifact = Artifact(path=str(path), description="Test-only calibration fixture", sha256=fingerprint(path))
    backend = RecordingFakeBackend(artifact)
    procedure = ExperimentProcedure(proposal_id=state.proposals[0].id, version=1,
        actions=[AcquisitionAction(action="fixture.capture", instruction="Produce test-only fixture data",
                                   requested_settings={"count": Setting(value=2, unit="frame")})],
        prerequisites=["fixture setup verified"], expected_outputs=["TEST ONLY fixture.json"],
        calibration_artifacts={"fixture-calibration": artifact})
    state.procedures.append(procedure)
    return state, cfg, backend, procedure


def approved(review):
    return HumanApproval(procedure_id=review.procedure.id, procedure_version=review.procedure.version,
        procedure_fingerprint=review.procedure_fingerprint, review_fingerprint=review.review_fingerprint,
        status="approved", reviewer="Trusted test caller")


def test_ready_preflight_is_read_only_and_records_review(hardware_setup):
    state, cfg, backend, procedure = hardware_setup
    before = state.model_dump_json()
    result = runner(state, procedure.id, config=cfg, backend=backend)
    assert result.status == "review" and result.review.status == "ready"
    assert not backend.execute_calls and not backend.cleanup_calls
    assert not (cfg.run_output_path / "acquisitions").exists()
    assert result.review.procedure == procedure
    assert result.review.capabilities_snapshot["data_origin"] == "test_only"
    assert "Requested settings" in result.explanation and "Expected outputs" in result.explanation
    assert "verified limits" in result.explanation
    assert state.model_dump_json() == before
    assert len(result.state.conversation) == 1
    assert result.state.conversation[0].record_ids == [result.review.id]
    assert WorkflowState.model_validate_json(result.state.model_dump_json()) == result.state


@pytest.mark.parametrize("problem,fragment", [
    ("missing_action", "unsupported action"), ("unknown_action", "unsupported action"),
    ("unknown_parameter", "Unsupported parameter"), ("missing_parameter", "Missing required parameter"),
    ("wrong_type", "Incorrect parameter type"), ("bool_as_int", "Incorrect parameter type"),
    ("wrong_unit", "Incorrect unit"), ("exceeded_limit", "Exceeded verified limits"),
    ("unverified_constraint", "Unverified equipment constraint"), ("unknown_limits", "Unknown required limits"),
    ("missing_calibration", "Missing or mismatched verified calibration"),
    ("unverified_prerequisite", "Unresolved prerequisite"),
    ("arbitrary_prerequisite", "Unresolved prerequisite"), ("action_budget", "action budget exceeded"),
    ("measurement_budget", "measurement budget exceeded"), ("path_escape", "without traversal"),
    ("input_hash", "stale input/calibration"), ("no_bounds_support", "bounded execution"),
])
def test_preflight_blocks_specific_problems(hardware_setup, problem, fragment):
    state, cfg, backend, procedure = hardware_setup
    setting = procedure.actions[0].requested_settings["count"]
    rule = backend.capabilities.actions["fixture.capture"].parameters["count"]
    if problem == "missing_action":
        procedure.actions[0].action = None
    elif problem == "unknown_action":
        procedure.actions[0].action = "arbitrary.python()"
    elif problem == "unknown_parameter":
        procedure.actions[0].requested_settings["voltage"] = Setting(value=1)
    elif problem == "missing_parameter":
        procedure.actions[0].requested_settings.clear()
    elif problem == "wrong_type":
        setting.value = "2"
    elif problem == "bool_as_int":
        setting.value = True
    elif problem == "wrong_unit":
        setting.unit = "ms"
    elif problem == "exceeded_limit":
        setting.value = 5
    elif problem == "unverified_constraint":
        rule.verified = False
    elif problem == "unknown_limits":
        rule.maximum = None
    elif problem == "missing_calibration":
        procedure.calibration_artifacts.clear()
    elif problem == "unverified_prerequisite":
        backend.capabilities.prerequisites["fixture setup verified"].verified = False
    elif problem == "arbitrary_prerequisite":
        procedure.prerequisites.append("Unresolved laboratory calibration")
    elif problem == "action_budget":
        cfg.max_acquisition_actions = 1
        procedure.actions.append(procedure.actions[0].model_copy(deep=True))
    elif problem == "measurement_budget":
        cfg.max_acquisition_measurements = 1
    elif problem == "path_escape":
        procedure.output_subdirectory = "../outside"
    elif problem == "input_hash":
        Path(procedure.calibration_artifacts["fixture-calibration"].path).write_text("changed")
    elif problem == "no_bounds_support":
        backend.capabilities.supports_deadlines = False
    result = runner(state, procedure.id, config=cfg, backend=backend)
    assert result.status == "blocked"
    assert any(fragment in issue for issue in result.technical_issues)
    assert not backend.execute_calls
    # Approval cannot override deterministic failure.
    run = execute_acquisition(procedure, result.review, approved(result.review), cfg, backend)
    assert run.acquisition is None and not backend.execute_calls


def test_default_disabled_backend(hardware_setup):
    state, cfg, _, procedure = hardware_setup
    cfg.hardware_backend = "disabled"
    result = runner(state, procedure.id, config=cfg)
    assert result.status == "blocked"
    assert any("unavailable" in issue.lower() for issue in result.technical_issues)
    assert any("disabled" in issue.lower() for issue in result.technical_issues)
    with pytest.raises(RuntimeError, match="unavailable"):
        DisabledHardwareBackend().execute(None)


@pytest.mark.parametrize("problem", ["missing", "pending", "rejected", "revoked", "wrong_id",
    "wrong_version", "wrong_fingerprint", "no_review", "wrong_review"])
def test_approval_gate_has_zero_acquisition_calls(hardware_setup, problem):
    state, cfg, backend, procedure = hardware_setup
    review = preflight(procedure, cfg, backend)
    approval = approved(review)
    if problem == "missing":
        approval = None
    elif problem in ("pending", "rejected", "revoked"):
        approval.status = problem
    elif problem == "wrong_id":
        approval.procedure_id = uuid4()
    elif problem == "wrong_version":
        approval.procedure_version = 2
    elif problem == "wrong_fingerprint":
        approval.procedure_fingerprint = "0" * 64
    elif problem == "no_review":
        approval.review_fingerprint = None
    elif problem == "wrong_review":
        approval.review_fingerprint = "0" * 64
    result = runner(state, procedure.id, config=cfg, backend=backend, execute=True,
                    reviewed=review, approval=approval)
    assert result.status == "blocked" and result.acquisition is None
    assert not backend.execute_calls and not backend.cleanup_calls
    assert not (cfg.run_output_path / "acquisitions").exists()


@pytest.mark.parametrize("change", ["setting", "action", "calibration", "input", "output"])
def test_same_version_content_change_invalidates_approval(hardware_setup, change):
    state, cfg, backend, procedure = hardware_setup
    review = preflight(procedure, cfg, backend)
    approval = approved(review)
    if change == "setting":
        procedure.actions[0].requested_settings["count"].value = 3
    elif change == "action":
        procedure.actions[0].instruction = "Changed instruction"
    elif change == "calibration":
        artifact = procedure.calibration_artifacts["fixture-calibration"]
        artifact.description = "Different calibration identity"
        backend.capabilities.verified_calibrations["fixture-calibration"] = artifact.model_copy(deep=True)
    elif change == "input":
        procedure.input_artifacts.append(procedure.calibration_artifacts["fixture-calibration"].model_copy(deep=True))
    else:
        procedure.output_subdirectory = "other-raw"
    assert procedure.version == 1
    assert procedure.fingerprint() != approval.procedure_fingerprint
    result = runner(state, procedure.id, config=cfg, backend=backend, execute=True,
                    reviewed=review, approval=approval)
    assert result.status == "blocked" and not backend.execute_calls


@pytest.mark.parametrize("change", ["limit", "prerequisite_evidence", "output_root", "budget"])
def test_changed_review_requires_new_review_approval(hardware_setup, tmp_path, change):
    state, cfg, backend, procedure = hardware_setup
    review = preflight(procedure, cfg, backend)
    approval = approved(review)
    if change == "limit":
        backend.capabilities.actions["fixture.capture"].parameters["count"].maximum = 3
    elif change == "prerequisite_evidence":
        backend.capabilities.prerequisites["fixture setup verified"].evidence = "New fixture verification"
    elif change == "output_root":
        cfg.run_output_path = tmp_path / "changed-root"
    else:
        cfg.max_acquisition_measurements = 3
    assert preflight(procedure, cfg, backend).status == "ready"
    result = runner(state, procedure.id, config=cfg, backend=backend, execute=True,
                    reviewed=review, approval=approval)
    assert result.status == "blocked" and not backend.execute_calls
    assert any("review changed" in issue for issue in result.technical_issues)


def test_constraints_change_between_initial_and_execution_checks(hardware_setup):
    state, cfg, backend, procedure = hardware_setup
    reviewed = preflight(procedure, cfg, backend)
    original_describe = backend.describe
    calls = 0

    def changing_describe():
        nonlocal calls
        calls += 1
        if calls >= 2:
            backend.capabilities.revision = "changed-before-execution"
        return original_describe()

    backend.describe = changing_describe
    result = runner(state, procedure.id, config=cfg, backend=backend, execute=True,
                    reviewed=reviewed, approval=approved(reviewed))
    assert result.status == "blocked" and not backend.execute_calls


def test_approved_recording_fake_preserves_actual_and_provenance(hardware_setup):
    state, cfg, backend, procedure = hardware_setup
    initial = runner(state, procedure.id, config=cfg, backend=backend)
    before = initial.state.model_dump_json()
    result = runner(initial.state, procedure.id, config=cfg, backend=backend, execute=True,
                    reviewed=initial.review, approval=approved(initial.review))
    assert initial.state.model_dump_json() == before
    assert result.status == "completed"
    record = result.acquisition
    assert record.hardware.data_origin == "test_only"
    assert record.hardware.backend_id == "recording-fake-test-only"
    assert record.actual_settings["0.count"].value == 2
    assert record.hardware.actions[0].requested.requested_settings["count"].value == 2
    assert record.hardware.actions[0].status == "completed"
    assert record.hardware.started_at <= record.hardware.ended_at
    assert record.raw_data[0].sha256 == fingerprint(Path(record.raw_data[0].path))
    assert json.loads(Path(record.raw_data[0].path).read_text())["data_origin"] == "test_only"
    directory = Path(record.hardware.run_directory)
    assert AcquisitionResult.model_validate_json((directory / "record.json").read_text()) == record
    assert json.loads((directory / "approval.json").read_text())["id"] == str(record.hardware.approval_id)
    assert json.loads((directory / "review.json").read_text())["review_fingerprint"] == initial.review.review_fingerprint
    assert len(result.state.conversation) == 2
    assert len(result.state.acquisitions) == len(state.acquisitions) + 1
    assert result.state.status == state.status
    assert not backend.cleanup_calls
    assert WorkflowState.model_validate_json(result.state.model_dump_json()) == result.state


def test_partial_failure_preserves_data_and_cleanup(hardware_setup):
    state, cfg, backend, procedure = hardware_setup
    procedure.actions.append(procedure.actions[0].model_copy(deep=True))
    backend.fail_at = 1
    review = preflight(procedure, cfg, backend)
    result = runner(state, procedure.id, config=cfg, backend=backend, execute=True,
                    reviewed=review, approval=approved(review))
    record = result.acquisition
    assert record.status == "partial"
    assert [a.status for a in record.hardware.actions] == ["completed", "failed"]
    assert len(record.raw_data) == 1 and Path(record.raw_data[0].path).exists()
    assert len(backend.cleanup_calls) == 1
    assert record.hardware.cleanup_status == "completed"
    assert "Experimentor" in result.recommended_next_step and "Scientist" in result.recommended_next_step
    saved = AcquisitionResult.model_validate_json((Path(record.hardware.run_directory) / "record.json").read_text())
    assert saved.status == "partial" and saved.errors


def test_cancellation_has_cleanup_and_no_automatic_retry(hardware_setup):
    state, cfg, backend, procedure = hardware_setup
    review = preflight(procedure, cfg, backend)
    event = Event()
    event.set()
    result = runner(state, procedure.id, config=cfg, backend=backend, execute=True,
                    reviewed=review, approval=approved(review), cancel_event=event)
    assert result.acquisition.status == "aborted"
    assert not backend.execute_calls and len(backend.cleanup_calls) == 1
    assert result.acquisition.hardware.actions[0].status == "pending"


def test_duplicate_attempts_rejected_on_disk_and_in_state(hardware_setup):
    state, cfg, backend, procedure = hardware_setup
    review = preflight(procedure, cfg, backend)
    attempt_id = uuid4()
    first = runner(state, procedure.id, config=cfg, backend=backend, execute=True,
                   reviewed=review, approval=approved(review), attempt_id=attempt_id)
    calls = len(backend.execute_calls)
    for supplied_state in (state, first.state):
        duplicate = runner(supplied_state, procedure.id, config=cfg, backend=backend,
            execute=True, reviewed=review, approval=approved(review), attempt_id=attempt_id)
        assert duplicate.status == "blocked" and duplicate.acquisition is None
    assert len(backend.execute_calls) == calls


def test_process_loss_leaves_started_record_and_blocks_replay(hardware_setup):
    state, cfg, backend, procedure = hardware_setup
    review = preflight(procedure, cfg, backend)
    attempt_id = uuid4()

    class ProcessLoss(BaseException):
        pass

    def crash(*args, **kwargs):
        path = cfg.run_output_path / "acquisitions" / str(attempt_id) / "record.json"
        record = AcquisitionResult.model_validate_json(path.read_text())
        assert record.status == "started" and record.hardware.actions[0].status == "started"
        raise ProcessLoss()

    backend.execute = crash
    with pytest.raises(ProcessLoss):
        runner(state, procedure.id, config=cfg, backend=backend, execute=True,
               reviewed=review, approval=approved(review), attempt_id=attempt_id)
    duplicate = runner(state, procedure.id, config=cfg, backend=backend, execute=True,
                       reviewed=review, approval=approved(review), attempt_id=attempt_id)
    assert duplicate.status == "blocked"
    assert any("already claimed" in issue for issue in duplicate.technical_issues)


def test_cleanup_failure_is_recorded(hardware_setup):
    state, cfg, backend, procedure = hardware_setup
    backend.fail_at = 0

    def fail_cleanup(**kwargs):
        raise RuntimeError("Injected cleanup failure")

    backend.cleanup = fail_cleanup
    review = preflight(procedure, cfg, backend)
    result = runner(state, procedure.id, config=cfg, backend=backend, execute=True,
                    reviewed=review, approval=approved(review))
    assert result.acquisition.status == "failed"
    assert result.acquisition.hardware.cleanup_status == "failed"
    assert any("cleanup failed" in issue for issue in result.acquisition.errors)


def test_approval_from_state_is_not_implicitly_trusted(hardware_setup):
    state, cfg, backend, procedure = hardware_setup
    review = preflight(procedure, cfg, backend)
    state.approvals.append(approved(review))
    result = runner(state, procedure.id, config=cfg, backend=backend, execute=True, reviewed=review)
    assert result.status == "blocked" and not backend.execute_calls


def test_actual_settings_are_not_filled_from_requests(hardware_setup):
    state, cfg, backend, procedure = hardware_setup
    original = backend.execute

    def different_actual(*args, **kwargs):
        report = original(*args, **kwargs)
        report.actual_settings["count"] = Setting(value=1, unit="frame")
        return report

    backend.execute = different_actual
    review = preflight(procedure, cfg, backend)
    result = runner(state, procedure.id, config=cfg, backend=backend, execute=True,
                    reviewed=review, approval=approved(review))
    assert result.acquisition.status == "completed"
    assert result.acquisition.actual_settings["0.count"].value == 1
    assert result.acquisition.hardware.actions[0].requested.requested_settings["count"].value == 2


def test_changed_constraints_after_first_action_stop_without_retry(hardware_setup):
    state, cfg, backend, procedure = hardware_setup
    procedure.actions.append(procedure.actions[0].model_copy(deep=True))
    original = backend.execute

    def change_after_capture(*args, **kwargs):
        report = original(*args, **kwargs)
        backend.capabilities.revision = "modified after first action"
        return report

    backend.execute = change_after_capture
    review = preflight(procedure, cfg, backend)
    result = runner(state, procedure.id, config=cfg, backend=backend, execute=True,
                    reviewed=review, approval=approved(review))
    assert result.acquisition.status == "partial"
    assert len(backend.execute_calls) == 1
    assert [a.status for a in result.acquisition.hardware.actions] == ["completed", "pending"]
    assert len(backend.cleanup_calls) == 1


def test_cancellation_after_backend_returns_preserves_measurement(hardware_setup):
    state, cfg, backend, procedure = hardware_setup
    original = backend.execute
    event = Event()

    def cancel_after_capture(*args, **kwargs):
        report = original(*args, **kwargs)
        event.set()
        return report

    backend.execute = cancel_after_capture
    review = preflight(procedure, cfg, backend)
    result = runner(state, procedure.id, config=cfg, backend=backend, execute=True,
                    reviewed=review, approval=approved(review), cancel_event=event)
    assert result.acquisition.status == "aborted"
    assert result.acquisition.raw_data
    assert result.acquisition.hardware.actions[0].status == "aborted"
    assert len(backend.cleanup_calls) == 1


def test_backend_raw_path_escape_is_not_accepted(hardware_setup, tmp_path):
    state, cfg, backend, procedure = hardware_setup
    outside = tmp_path / "unrelated.json"
    outside.write_text("Unrelated data")

    def malicious_report(*args, **kwargs):
        return ActionReport(actual_settings={"count": Setting(value=2, unit="frame")},
            raw_data=[Artifact(path=str(outside), description="Outside attempt")])

    backend.execute = malicious_report
    review = preflight(procedure, cfg, backend)
    result = runner(state, procedure.id, config=cfg, backend=backend, execute=True,
                    reviewed=review, approval=approved(review))
    assert result.acquisition.status == "failed" and not result.acquisition.raw_data
    assert "inside" in result.acquisition.errors[0]
    assert len(backend.cleanup_calls) == 1


def test_backend_deadline_overrun_cannot_be_completed(hardware_setup, monkeypatch):
    import tools.hardware as hardware_module

    state, cfg, backend, procedure = hardware_setup
    cfg.hardware_action_timeout_seconds = 1
    clock_value = [0.0]

    class FakeClock:
        @staticmethod
        def monotonic():
            return clock_value[0]

    monkeypatch.setattr(hardware_module, "time", FakeClock)

    def late_capture(*args, **kwargs):
        clock_value[0] = kwargs["deadline"] + 0.1
        return ActionReport(actual_settings={"count": Setting(value=2, unit="frame")})

    backend.execute = late_capture
    review = preflight(procedure, cfg, backend)
    result = runner(state, procedure.id, config=cfg, backend=backend, execute=True,
                    reviewed=review, approval=approved(review))
    assert result.acquisition.status == "aborted"
    assert "deadline exceeded" in result.acquisition.errors[0]
    assert len(backend.cleanup_calls) == 1
