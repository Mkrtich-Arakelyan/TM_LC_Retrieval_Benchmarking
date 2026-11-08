import pytest
from pydantic import ValidationError

from config import AppConfig
from state import (
    AcquisitionAction, AcquisitionResult, Artifact, ConversationEntry,
    ExperimentProcedure, ExperimentProposal, HumanApproval, NumericalAnalysisResult,
    ScientificFinding, Setting, WorkflowState, approval_matches,
)


def chain():
    """Synthetic schema examples, not measurements or scientific claims."""
    proposal = ExperimentProposal(
        scientific_question="How does sample count affect retrieval error?",
        hypothesis="More samples reduce error", rationale="Test sample efficiency",
        variables={"sample_count": "Number of input patterns"},
        controls=["Use the same calibration"], required_measurements=["Output intensities"],
        evaluation_criteria=["Compare held-out prediction error"],
        assumptions=["Calibration is stable"],
    )
    procedure = ExperimentProcedure(
        proposal_id=proposal.id, version=1,
        actions=[AcquisitionAction(instruction="Acquire a validated pattern sequence",
                                   requested_settings={"sample_count": Setting(value=100)})],
        prerequisites=["Verify equipment limits and calibration"],
        expected_outputs=["Intensity frames and actual settings"],
    )
    acquisition = AcquisitionResult(
        procedure_id=procedure.id, procedure_version=procedure.version,
        procedure_fingerprint=procedure.fingerprint(), status="completed",
        actual_settings={"sample_count": Setting(value=100)},
        raw_data=[Artifact(path="synthetic/frames.npy", description="Test fixture reference")],
    )
    analysis = NumericalAnalysisResult(
        acquisition_ids=[acquisition.id], input_data=acquisition.raw_data,
        code=Artifact(path="synthetic/analysis.py", description="Fixture code reference"),
        code_version="fixture-v1", method_settings={"solver": "example"}, random_seed=7,
        status="completed", metrics={"fixture_error": 0.25},
    )
    return proposal, procedure, acquisition, analysis


def test_chain_and_json_round_trip():
    proposal, procedure, acquisition, analysis = chain()
    finding = ScientificFinding(evidence_ids=[analysis.id], interpretation="Fixture interpretation",
                                limitations=["Synthetic records only"], recommended_next_step="Review")
    state = WorkflowState(
        user_request="Benchmark retrieval", proposals=[proposal], procedures=[procedure],
        acquisitions=[acquisition], analyses=[analysis], findings=[finding],
        conversation=[ConversationEntry(role="scientist", message="Proposed comparison",
                                        record_ids=[proposal.id])],
    )
    restored = WorkflowState.model_validate_json(state.model_dump_json())
    assert restored == state
    assert restored.procedures[0].proposal_id == restored.proposals[0].id
    assert restored.analyses[0].acquisition_ids == [restored.acquisitions[0].id]
    assert restored.findings[0].evidence_ids == [restored.analyses[0].id]


@pytest.mark.parametrize("change", [
    {"status": "success"}, {"procedure_version": 0}, {"procedure_id": "not-a-uuid"},
    {"procedure_fingerprint": "bad"}, {"raw_data": [{"path": ""}]},
    {"status": "completed", "raw_data": []},
    {"status": "completed", "actual_settings": {}},
    {"status": "failed", "errors": []},
])
def test_invalid_acquisition(change):
    data = chain()[2].model_dump()
    data.update(change)
    with pytest.raises(ValidationError):
        AcquisitionResult.model_validate(data)


def test_pending_and_failed_are_not_completed():
    _, procedure, acquisition, analysis = chain()
    pending = AcquisitionResult(procedure_id=procedure.id, procedure_version=1,
                                procedure_fingerprint=procedure.fingerprint())
    failed = AcquisitionResult(**{**pending.model_dump(), "status": "failed", "errors": ["Camera unavailable"]})
    assert [pending.status, failed.status, acquisition.status] == ["pending", "failed", "completed"]
    assert pending.raw_data == []
    for status, errors in [("pending", []), ("failed", ["Execution timed out"])]:
        other = NumericalAnalysisResult.model_validate({**analysis.model_dump(), "status": status,
                                                        "metrics": {}, "outputs": [], "errors": errors})
        assert other.status != "completed"


@pytest.mark.parametrize("change", [
    {"status": "success"}, {"status": "failed", "errors": []},
    {"metrics": {}, "outputs": []}, {"input_data": []},
    {"code_version": " "}, {"errors": ["Execution failed"]},
])
def test_invalid_analysis(change):
    with pytest.raises(ValidationError):
        NumericalAnalysisResult.model_validate({**chain()[3].model_dump(), **change})


def test_required_proposal_and_procedure_data():
    proposal, procedure, _, _ = chain()
    for model, data, change in [
        (ExperimentProposal, proposal.model_dump(), {"hypothesis": " "}),
        (ExperimentProposal, proposal.model_dump(), {"variables": {}}),
        (ExperimentProcedure, procedure.model_dump(), {"actions": []}),
        (ExperimentProcedure, procedure.model_dump(), {"unexpected": True}),
    ]:
        with pytest.raises(ValidationError):
            model.model_validate({**data, **change})


def test_approval_covers_exact_content_and_version():
    procedure = chain()[1]
    approval = HumanApproval(procedure_id=procedure.id, procedure_version=1,
                             procedure_fingerprint=procedure.fingerprint(), status="approved",
                             reviewer="Test reviewer")
    assert approval_matches(approval, procedure)
    revised = procedure.model_copy(deep=True)
    revised.version = 2
    assert not approval_matches(approval, revised)
    revised = procedure.model_copy(deep=True)
    revised.actions[0].requested_settings["sample_count"] = Setting(value=200)
    assert not approval_matches(approval, revised)
    revised = procedure.model_copy(deep=True)
    revised.actions[0].instruction = "Different acquisition"
    assert not approval_matches(approval, revised)
    for status in ["pending", "rejected", "revoked"]:
        assert not approval_matches(approval.model_copy(update={"status": status}), procedure)


def test_config_is_optional_secret_redacted_and_limits_unknown(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("TM_"):
            monkeypatch.delenv(name)
    config = AppConfig.from_env()
    assert config.llm_model is None and config.api_key is None
    assert config.verified_hardware_limits is None
    monkeypatch.setenv("TM_API_KEY", "fixture-secret")
    monkeypatch.setenv("TM_NUMERICAL_TIMEOUT_SECONDS", "120")
    config = AppConfig.from_env()
    assert config.numerical_timeout_seconds == 120
    assert "fixture-secret" not in config.model_dump_json()
    assert "fixture-secret" not in repr(config)
    monkeypatch.setenv("TM_MAX_WORKFLOW_ITERATIONS", "0")
    with pytest.raises(ValidationError):
        AppConfig.from_env()
