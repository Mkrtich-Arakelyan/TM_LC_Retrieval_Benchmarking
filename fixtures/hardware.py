"""Recording acquisition backend double; never physical hardware."""

import json
import time
from pathlib import Path
from threading import Event
from workflow.state import AcquisitionAction, Artifact, ExperimentProcedure
from tools.hardware import (ActionReport, ActionSpec, HardwareCapabilities, ParameterSpec,
                            Verification, canonical_hash, parameter_issues)


class RecordingFakeBackend:
    """TEST ONLY: deterministic fixture files, never physical observations."""
    def __init__(self, calibration: Artifact, *, fail_at: int | None = None):
        self.capabilities = HardwareCapabilities(
            backend_id="recording-fake-test-only", revision="1", available=True,
            data_origin="test_only", supports_deadlines=True, supports_cancellation=True,
            supports_cleanup=True,
            prerequisites={"fixture setup verified": Verification(verified=True, evidence="test fixture only")},
            verified_calibrations={"fixture-calibration": calibration.model_copy(deep=True)},
            actions={"fixture.capture": ActionSpec(parameters={"count": ParameterSpec(
                kind="integer", unit="frame", verified=True, minimum=1, maximum=4)},
                required_calibrations=["fixture-calibration"],
                required_prerequisites=["fixture setup verified"], measurement_count_setting="count")})
        self.fail_at = fail_at
        self.execute_calls: list[AcquisitionAction] = []
        self.cleanup_calls: list[str] = []

    def describe(self) -> HardwareCapabilities:
        return self.capabilities.model_copy(deep=True)

    def validate(self, procedure: ExperimentProcedure) -> list[str]:
        return []

    def execute(self, action: AcquisitionAction, *, output_dir: Path,
                input_artifacts: list[Artifact], capability_fingerprint: str,
                deadline: float, cancel_event: Event) -> ActionReport:
        if capability_fingerprint != canonical_hash(self.describe().model_dump(mode="json")):
            return ActionReport(errors=["Fake capabilities changed"])
        spec = self.capabilities.actions.get(action.action or "")
        if spec is None:
            return ActionReport(errors=["Unsupported fake action"])
        if invalid := parameter_issues(action.requested_settings, spec):
            return ActionReport(errors=invalid)
        index = len(self.execute_calls)
        self.execute_calls.append(action.model_copy(deep=True))
        if cancel_event.is_set() or time.monotonic() >= deadline:
            return ActionReport(aborted=True, errors=["Fake cancelled"])
        if index == self.fail_at:
            return ActionReport(errors=["Injected test-only acquisition failure"])
        path = output_dir / "fixture.json"
        with path.open("x", encoding="utf-8") as file:
            json.dump({"data_origin": "test_only", "fixture_measurements": [0] * action.requested_settings["count"].value}, file)
        return ActionReport(actual_settings={k: v.model_copy(deep=True) for k, v in action.requested_settings.items()},
            raw_data=[Artifact(path=str(path), description="TEST ONLY deterministic fixture; not physical data",
                               media_type="application/json")])

    def cleanup(self, *, reason: str, deadline: float) -> str:
        self.cleanup_calls.append(reason)
        return "Test-only cleanup recorded; no physical equipment exists"
