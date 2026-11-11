"""Offline fixtures. Fake backends never interpret or execute the saved script."""

import pytest

from workflow.config import AppConfig
from workflow.state import AcquisitionResult, Artifact, ExperimentProposal, Setting, WorkflowState


@pytest.fixture
def experiment_setup(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    data = raw / "measurements.txt"
    data.write_text("fixture data only")
    architecture = tmp_path / "architecture.txt"
    architecture.write_bytes(b"Provisional setup\r\nHardware and calibration unverified.\r\n")
    config = AppConfig(numerical_input_path=raw, run_output_path=tmp_path / "runs")
    proposal = ExperimentProposal(scientific_question="Compare sample efficiency",
        hypothesis="Additional samples may improve validation", rationale="Control the budget",
        variables={"budget": "Number of observations"}, controls=["Same calibration"],
        required_measurements=["Held-out intensities"], evaluation_criteria=["Prediction error"])
    acquisition = AcquisitionResult(procedure_id=proposal.id, procedure_version=1,
        procedure_fingerprint="a" * 64, status="completed",
        actual_settings={"count": Setting(value=1)},
        raw_data=[Artifact(path=str(data), description="Synthetic test artifact")])
    state = WorkflowState(user_request="Implement the proposal or analyze saved data",
                          proposals=[proposal], acquisitions=[acquisition])
    return state, config, architecture


@pytest.fixture
def fake_model():
    class FakeModel:
        def __init__(self, *responses):
            self.responses = iter(responses)
            self.calls = []

        def with_structured_output(self, schema):
            self.schema = schema
            return self

        def invoke(self, messages):
            self.calls.append((self.schema, messages))
            return next(self.responses)
    return FakeModel


@pytest.fixture
def fake_backend():
    from tools.python_execution import BackendReport

    class FakeBackend:
        def __init__(self, exit_codes=(0,)):
            self.exit_codes = iter(exit_codes)
            self.calls = []

        def run(self, **kwargs):
            self.calls.append(kwargs)
            output = kwargs["output_dir"] / "plot.txt"
            output.write_text("Synthetic output, not a scientific plot")
            return BackendReport(exit_code=next(self.exit_codes), elapsed_seconds=0.01,
                stdout="fake stdout", stderr="fake stderr", metrics={"fixture_metric": 0.25},
                outputs=[Artifact(path=str(output), description="Fake derived output")])
    return FakeBackend
