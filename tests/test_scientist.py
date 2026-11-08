"""Contract tests using canned responses, not scientific reasoning evaluations."""

import json
from uuid import uuid4

import pytest

from agents.scientist import DEFAULT_SYSTEM_PROMPT, ScientistOutcome, scientist
from state import (
    AcquisitionResult, Artifact, ConversationEntry, ExperimentProposal,
    LiteraturePassage, NumericalAnalysisResult, Setting, WorkflowState,
)


class FakeChatModel:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def with_structured_output(self, schema):
        assert schema is ScientistOutcome
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.response


@pytest.fixture
def architecture(tmp_path):
    path = tmp_path / "architecture.txt"
    path.write_bytes(b"Provisional setup\r\nCalibration unverified.\r\n")
    return path


def proposal_response():
    proposal = ExperimentProposal(
        scientific_question="How does measurement budget affect prediction error?",
        hypothesis="Error decreases with additional measurements",
        rationale="Compare sample efficiency using general domain knowledge",
        variables={"budget": "Acquisition count"}, controls=["Same calibration"],
        required_measurements=["Intensity observations and held-out validation data"],
        evaluation_criteria=["Held-out prediction error at equal budgets"],
        assumptions=["Calibration must be verified before acquisition"],
    )
    return dict(kind="proposal", explanation="Compare budgets conditional on calibration.",
                recommended_next_step="Confirm calibration", proposals=[proposal.model_dump()])


def information_response():
    return dict(kind="request_information", explanation="Calibration information is essential.",
                recommended_next_step="Supply calibration status",
                questions=["Has command-to-phase calibration been verified?"])


def test_proposal_inputs_state_copy_and_single_history_append(architecture):
    original = WorkflowState(user_request="Compare retrieval measurement budgets",
                             conversation=[ConversationEntry(role="user", message="Please compare")])
    before = original.model_dump_json()
    model = FakeChatModel(proposal_response())
    result = scientist(original, model, architecture_path=architecture)
    assert original.model_dump_json() == before
    assert result.outcome.kind == "proposal"
    assert len(result.state.proposals) == 1
    assert len(result.state.conversation) == 2
    assert result.state.conversation[0] == original.conversation[0]
    assert result.state.conversation[-1].record_ids == [result.state.proposals[0].id]
    assert result.state.status == original.status
    assert not result.state.acquisitions and not result.state.approvals
    assert "not been checked against the project's papers" in result.outcome.explanation
    payload = json.loads(model.calls[0][1][1])
    assert payload["workflow_state"]["user_request"] == original.user_request
    assert payload["architecture_text"] == "Provisional setup\r\nCalibration unverified.\r\n"
    restored = WorkflowState.model_validate_json(result.state.model_dump_json())
    assert restored == result.state
    assert restored.conversation[-1].scientific_context.architecture_text == payload["architecture_text"]
    second = scientist(result.state, FakeChatModel(information_response()), architecture_path=architecture)
    assert len(second.state.conversation) == 3
    assert len(result.state.conversation) == 2


def test_request_missing_information(architecture):
    result = scientist(WorkflowState(user_request="Plan an experiment"),
                       FakeChatModel(information_response()), architecture_path=architecture)
    assert result.outcome.kind == "request_information"
    assert result.outcome.questions
    assert not result.state.proposals and not result.state.findings
    assert result.outcome.questions[0] in result.state.conversation[-1].message


def test_architecture_is_reread(architecture):
    state = WorkflowState(user_request="Plan")
    model = FakeChatModel(information_response())
    first = scientist(state, model, architecture_path=architecture)
    architecture.write_text("Updated setup: coherent reference unconfirmed.", encoding="utf-8")
    second = scientist(state, model, architecture_path=architecture)
    assert first.state.conversation[-1].scientific_context.architecture_text != second.state.conversation[-1].scientific_context.architecture_text
    assert json.loads(model.calls[-1][1][1])["architecture_text"].startswith("Updated setup")


def test_literature_metadata_sent_and_preserved(architecture):
    passage = LiteraturePassage(source_id="paper-a:p3", filename="paper-a.pdf", page_number=3,
                                passage="  Example supplied passage.\n")
    response = proposal_response()
    response["cited_source_ids"] = [passage.source_id]
    response["proposals"][0]["literature_references"] = [passage.source_id]
    model = FakeChatModel(response)
    result = scientist(WorkflowState(user_request="Compare"), model, [passage], architecture_path=architecture)
    payload = json.loads(model.calls[0][1][1])
    assert payload["literature_context"] == [passage.model_dump()]
    context = result.state.conversation[-1].scientific_context
    assert context.literature_context == [passage]
    assert "not been checked" not in result.outcome.explanation


@pytest.mark.parametrize("location", ["cited_source_ids", "proposal"])
def test_invented_citation_rejected(architecture, location):
    response = proposal_response()
    if location == "proposal":
        response["proposals"][0]["literature_references"] = ["invented-paper"]
    else:
        response[location] = ["invented-paper"]
    state = WorkflowState(user_request="Compare")
    before = state.model_dump_json()
    with pytest.raises(ValueError, match="Unknown literature source identifiers.*invented-paper"):
        scientist(state, FakeChatModel(response), architecture_path=architecture)
    assert state.model_dump_json() == before


def evidence_state():
    acquisition = AcquisitionResult(
        procedure_id=uuid4(), procedure_version=1, procedure_fingerprint="a" * 64,
        status="completed", actual_settings={"count": Setting(value=1)},
        raw_data=[Artifact(path="fixtures/data.npy", description="Synthetic metadata only")],
    )
    analysis = NumericalAnalysisResult(
        acquisition_ids=[acquisition.id], input_data=acquisition.raw_data,
        code=Artifact(path="fixtures/code.py", description="Not executed"), code_version="test",
        status="completed", metrics={"synthetic_error": 0.25},
    )
    state = WorkflowState(user_request="Interpret saved analysis", acquisitions=[acquisition], analyses=[analysis])
    response = dict(kind="interpretation", explanation="Interpretation of synthetic evidence only.",
                    recommended_next_step="Collect independent validation",
                    findings=[dict(evidence_ids=[str(acquisition.id), str(analysis.id)],
                                   interpretation="Recorded synthetic error is 0.25; generalization is inference.",
                                   limitations=["No independent validation"], recommended_next_step="Validate")])
    return state, response


def test_interpretation_uses_existing_records(architecture):
    state, response = evidence_state()
    before = state.model_dump_json()
    model = FakeChatModel(response)
    result = scientist(state, model, architecture_path=architecture)
    assert len(result.state.findings) == 1
    assert result.state.findings[0].evidence_ids == [state.acquisitions[0].id, state.analyses[0].id]
    assert state.model_dump_json() == before
    sent = json.loads(model.calls[0][1][1])["workflow_state"]
    assert sent["analyses"][0]["metrics"] == {"synthetic_error": 0.25}
    assert sent["acquisitions"][0]["status"] == "completed"


@pytest.mark.parametrize("failure", ["missing", "pending_analysis", "failed_analysis", "pending_acquisition", "missing_upstream"])
def test_ineligible_findings_rejected(architecture, failure):
    state, response = evidence_state()
    if failure == "missing":
        response["findings"][0]["evidence_ids"] = [str(uuid4())]
    elif failure == "missing_upstream":
        state.analyses[0].acquisition_ids = [uuid4()]
    elif failure == "pending_acquisition":
        state.acquisitions[0].status = "pending"
        response["findings"][0]["evidence_ids"] = [str(state.analyses[0].id)]
    else:
        state.analyses[0] = NumericalAnalysisResult.model_validate({
            **state.analyses[0].model_dump(),
            "status": "pending" if failure == "pending_analysis" else "failed",
            "errors": [] if failure == "pending_analysis" else ["Execution failed"],
        })
    with pytest.raises(ValueError, match="missing or incomplete evidence"):
        scientist(state, FakeChatModel(response), architecture_path=architecture)


@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_missing_empty_architecture(tmp_path, content):
    path = tmp_path / "setup.txt"
    if content is not None:
        path.write_text(content)
    model = FakeChatModel(information_response())
    with pytest.raises(ValueError, match="Cannot read architecture|Architecture file is empty"):
        scientist(WorkflowState(user_request="Plan"), model, architecture_path=path)
    assert not model.calls


@pytest.mark.parametrize("response", [None, "not structured", {},
    {"kind": "proposal", "explanation": "Missing proposals", "recommended_next_step": "Review"},
    {"kind": "request_information", "explanation": "No questions", "recommended_next_step": "Review"},
])
def test_invalid_model_output(architecture, response):
    state = WorkflowState(user_request="Plan")
    before = state.model_dump_json()
    with pytest.raises(ValueError, match="Invalid Scientist model response"):
        scientist(state, FakeChatModel(response), architecture_path=architecture)
    assert state.model_dump_json() == before


def test_prompt_contains_scientific_and_evidence_boundaries():
    for phrase in ["provisional paper specifications", "verified laboratory facts",
                   "grayscale commands, applied voltage, and optical phase response",
                   "command-to-phase mapping", "Intensity-only", "coherent reference",
                   "measurement budgets", "validation data", "noise", "drift", "phase ambiguities",
                   "measured evidence, numerical results", "completed upstream acquisitions",
                   "Only cite supplied", "not been checked", "Never invent equipment limits",
                   "Do not execute Python", "grant approval"]:
        assert phrase in DEFAULT_SYSTEM_PROMPT
    assert "HSP192" not in DEFAULT_SYSTEM_PROMPT


def test_model_errors_are_not_replaced_with_fallbacks(architecture):
    class BrokenModel(FakeChatModel):
        def invoke(self, messages):
            raise RuntimeError("Injected provider failed")

    with pytest.raises(RuntimeError, match="Injected provider failed"):
        scientist(WorkflowState(user_request="Plan"), BrokenModel(None), architecture_path=architecture)


def test_default_architecture_and_pydantic_response():
    model = FakeChatModel(ScientistOutcome.model_validate(information_response()))
    result = scientist(WorkflowState(user_request="Plan"), model)
    assert "TM RETRIEVAL BENCHMARKING" in result.state.conversation[-1].scientific_context.architecture_text
    assert model.calls[0][0] == ("system", DEFAULT_SYSTEM_PROMPT)


def test_duplicate_record_ids_are_rejected(architecture):
    response = proposal_response()
    response["proposals"].append(response["proposals"][0])
    with pytest.raises(ValueError, match="duplicate record ID"):
        scientist(WorkflowState(user_request="Plan"), FakeChatModel(response), architecture_path=architecture)
