import json
from pathlib import Path

import pytest

from agents.experimentor import DEFAULT_SYSTEM_PROMPT, experimentor
from workflow.state import HumanApproval, LiteraturePassage, WorkflowState, approval_matches


def procedure_response(state, version=1):
    return dict(kind="procedure", explanation="Implements the requested measurement budget without changing controls.",
        recommended_next_step="Runner preflight and human review",
        procedure=dict(proposal_id=str(state.proposals[0].id), version=version,
            actions=[dict(instruction="Record the approved input sequence and associated camera frames",
                          requested_settings={"pattern_count": {"value": 1}})],
            prerequisites=["Unresolved: confirm camera frame format"],
            expected_outputs=["Raw frames and actual settings manifest"]))


def analysis_response(state, source="raise RuntimeError('not a host script')\n"):
    return dict(kind="analysis", explanation="Calculates fixture diagnostics; preprocessing is specified by normalize.",
        recommended_next_step="Review numerical diagnostics with Scientist",
        analysis=dict(acquisition_ids=[str(state.acquisitions[0].id)],
                      input_data=[a.model_dump() for a in state.acquisitions[0].raw_data],
                      source=source, method_settings={"normalize": False}, random_seed=42))


def review(retry=False):
    return dict(technical_explanation="Technical review of the recorded fake execution; no scientific inference.",
                recommended_next_step="Repair code" if retry else "Send technical record to Scientist", retry=retry)


def test_prepare_procedure_and_context_copy(experiment_setup, fake_model):
    state, config, architecture = experiment_setup
    before = state.model_dump_json()
    model = fake_model(procedure_response(state))
    result = experimentor(state, model, config=config, architecture_path=architecture)
    assert state.model_dump_json() == before
    assert len(result.state.procedures) == 1
    procedure = result.state.procedures[0]
    assert procedure.proposal_id == state.proposals[0].id and procedure.version == 1
    assert procedure.actions[0].requested_settings["pattern_count"].value == 1
    assert any("calibration" in p for p in procedure.prerequisites)
    assert any("operating limits" in p for p in procedure.prerequisites)
    assert any("capabilities" in p for p in procedure.prerequisites)
    assert len(result.state.conversation) == 1
    assert result.state.conversation[0].record_ids == [procedure.id]
    assert result.state.conversation[0].scientific_context.architecture_text.endswith("\r\n")
    assert result.state.approvals == state.approvals
    assert result.state.acquisitions == state.acquisitions
    assert WorkflowState.model_validate_json(result.state.model_dump_json()) == result.state


def test_revision_requires_new_version_and_approval(experiment_setup, fake_model):
    state, config, architecture = experiment_setup
    first = experimentor(state, fake_model(procedure_response(state)), config=config, architecture_path=architecture)
    old = first.state.procedures[0]
    approval = HumanApproval(procedure_id=old.id, procedure_version=1,
        procedure_fingerprint=old.fingerprint(), status="approved")
    first.state.approvals.append(approval)
    before = first.state.model_dump_json()
    revised = experimentor(first.state, fake_model(procedure_response(state, 2)), config=config, architecture_path=architecture)
    assert revised.state.procedures[-1].version == 2
    assert not approval_matches(approval, revised.state.procedures[-1])
    assert revised.state.approvals == [approval]
    assert first.state.model_dump_json() == before
    assert len(revised.state.conversation) == 2
    with pytest.raises(ValueError, match="requires version 2"):
        experimentor(first.state, fake_model(procedure_response(state)), config=config, architecture_path=architecture)


def test_code_generation_without_execution(experiment_setup, fake_model, fake_backend):
    state, config, architecture = experiment_setup
    backend = fake_backend()
    result = experimentor(state, fake_model(analysis_response(state)), config=config,
                          architecture_path=architecture, backend=backend)
    assert len(result.state.analysis_code) == 1 and not result.state.analyses
    code = result.state.analysis_code[0]
    assert Path(code.code.path).read_text() == analysis_response(state)["analysis"]["source"]
    assert code.code.sha256 and code.input_data[0].sha256
    assert code.random_seed == 42 and code.method_settings == {"normalize": False}
    assert code.scientific_context.architecture_text.startswith("Provisional setup")
    assert not backend.calls
    assert "not requested" in result.outcome.explanation


def test_execution_disabled_is_not_success(experiment_setup, fake_model, fake_backend):
    state, config, architecture = experiment_setup
    backend = fake_backend()
    result = experimentor(state, fake_model(analysis_response(state)), config=config,
                          architecture_path=architecture, execute=True, backend=backend)
    attempt = result.state.analyses[0]
    assert attempt.status == "pending" and attempt.execution.status == "unavailable"
    assert "Execution unavailable" in result.outcome.explanation
    assert not backend.calls


def test_repair_then_success_separate_records_and_one_conversation(experiment_setup, fake_model, fake_backend):
    state, config, architecture = experiment_setup
    config.numerical_backend = "test"
    before = state.model_dump_json()
    model = fake_model(analysis_response(state), review(True), analysis_response(state, "# repaired fixture\npass\n"), review())
    backend = fake_backend((1, 0))
    result = experimentor(state, model, config=config, architecture_path=architecture, execute=True, backend=backend)
    assert state.model_dump_json() == before
    assert [r.status for r in result.state.analyses] == ["failed", "completed"]
    assert [r.execution.attempt_number for r in result.state.analyses] == [1, 2]
    assert len(result.state.analysis_code) == 2
    assert result.state.analysis_code[0].code.sha256 != result.state.analysis_code[1].code.sha256
    assert result.state.analyses[1].outputs
    assert len(result.state.conversation) == 1
    assert len(result.state.conversation[0].record_ids) == 4
    assert json.loads(model.calls[1][1][1][1])["latest_execution"]["status"] == "failed"
    assert json.loads(model.calls[2][1][1][1])["repair_request"]["retry"] is True


def test_attempt_limit_and_model_call_limit(experiment_setup, fake_model, fake_backend):
    state, config, architecture = experiment_setup
    config.numerical_backend = "test"
    config.max_execution_attempts = 1
    model = fake_model(analysis_response(state), review(True))
    backend = fake_backend((1,))
    result = experimentor(state, model, config=config, architecture_path=architecture, execute=True, backend=backend)
    assert len(backend.calls) == 1 and len(result.state.analyses) == 1
    assert "Execution-attempt limit reached" in result.outcome.explanation
    config.max_execution_attempts = 2
    config.max_workflow_iterations = 1
    model = fake_model(analysis_response(state))
    result = experimentor(state, model, config=config, architecture_path=architecture, execute=True, backend=fake_backend())
    assert len(model.calls) == 1
    assert "without model review" in result.outcome.explanation


def test_clarification_and_scientific_decisions(experiment_setup, fake_model):
    state, config, architecture = experiment_setup
    response = dict(kind="request_information", explanation="Budget changes require a Scientist decision.",
        recommended_next_step="Ask Scientist", scientific_decisions=["May the measurement budget change?"])
    result = experimentor(state, fake_model(response), config=config, architecture_path=architecture)
    assert result.outcome.scientific_decisions
    assert not result.state.procedures and not result.state.analysis_code
    assert len(result.state.conversation) == 1
    response = procedure_response(state)
    response["scientific_decisions"] = ["Change the budget"]
    with pytest.raises(ValueError, match="Scientific decisions"):
        experimentor(state, fake_model(response), config=config, architecture_path=architecture)


def test_literature_context_and_architecture_refresh(experiment_setup, fake_model):
    state, config, architecture = experiment_setup
    source = LiteraturePassage(source_id="paper:p1", filename="paper.pdf", page_number=1, passage="Supplied evidence")
    response = procedure_response(state)
    response["cited_source_ids"] = [source.source_id]
    model = fake_model(response, response)
    first = experimentor(state, model, [source], config=config, architecture_path=architecture)
    architecture.write_text("Updated setup; limits unknown")
    second = experimentor(state, model, [source], config=config, architecture_path=architecture)
    assert first.state.conversation[-1].scientific_context != second.state.conversation[-1].scientific_context
    sent = json.loads(model.calls[0][1][1][1])
    assert sent["literature_context"] == [source.model_dump()]
    response["cited_source_ids"] = ["invented"]
    with pytest.raises(ValueError, match="Unknown literature"):
        experimentor(state, fake_model(response), config=config, architecture_path=architecture)


def test_invalid_input_selection_and_path(experiment_setup, fake_model, tmp_path):
    state, config, architecture = experiment_setup
    outside = tmp_path / "outside.txt"
    outside.write_text("Not designated input")
    state.acquisitions[0].raw_data[0].path = str(outside)
    with pytest.raises(ValueError, match="inside"):
        experimentor(state, fake_model(analysis_response(state)), config=config, architecture_path=architecture)
    response = analysis_response(state)
    response["analysis"]["input_data"][0]["path"] = str(tmp_path / "invented.txt")
    with pytest.raises(ValueError, match="saved artifact"):
        experimentor(state, fake_model(response), config=config, architecture_path=architecture)


def test_model_cannot_omit_recorded_input_hash(experiment_setup, fake_model):
    state, config, architecture = experiment_setup
    state.acquisitions[0].raw_data[0].sha256 = "0" * 64
    response = analysis_response(state)
    response["analysis"]["input_data"][0]["sha256"] = None
    with pytest.raises(ValueError, match="Input artifact hash mismatch"):
        experimentor(state, fake_model(response), config=config, architecture_path=architecture)


def test_partial_failed_data_is_labeled_and_cannot_become_acquired(experiment_setup, fake_model, fake_backend):
    from workflow.state import AcquisitionResult

    state, config, architecture = experiment_setup
    state.acquisitions[0] = AcquisitionResult.model_validate({
        **state.acquisitions[0].model_dump(), "status": "failed", "errors": ["Acquisition interrupted"],
    })
    config.numerical_backend = "test"
    result = experimentor(state, fake_model(analysis_response(state), review()), config=config,
                          architecture_path=architecture, execute=True, backend=fake_backend())
    assert result.state.analyses[0].status == "completed"
    assert result.state.acquisitions[0].status == "failed"
    assert "partial data from a failed acquisition" in result.outcome.explanation


def test_repair_cannot_switch_input_selection(experiment_setup, fake_model, fake_backend):
    state, config, architecture = experiment_setup
    extra = config.numerical_input_path / "other.txt"
    extra.write_text("Different fixture")
    from workflow.state import Artifact

    state.acquisitions[0].raw_data.append(Artifact(path=str(extra), description="Second input"))
    first = analysis_response(state)
    first["analysis"]["input_data"] = first["analysis"]["input_data"][:1]
    second = analysis_response(state)
    second["analysis"]["input_data"] = second["analysis"]["input_data"][1:]
    config.numerical_backend = "test"
    with pytest.raises(ValueError, match="cannot silently change"):
        experimentor(state, fake_model(first, review(True), second), config=config,
                     architecture_path=architecture, execute=True, backend=fake_backend((1,)))


def test_invalid_output_and_missing_architecture(experiment_setup, fake_model, tmp_path):
    state, config, architecture = experiment_setup
    with pytest.raises(ValueError, match="Invalid Experimentor model response"):
        experimentor(state, fake_model({}), config=config, architecture_path=architecture)
    with pytest.raises(ValueError, match="Cannot read architecture"):
        experimentor(state, fake_model({}), config=config, architecture_path=tmp_path / "missing")


def test_prompt_boundaries():
    for rule in ["No arbitrary hardware scripts", "measurement budgets", "scientific assumptions",
                 "Do not assume command-to-phase calibration", "Inten", "Only cite supplied",
                 "Never turn failed execution", "No networking", "No algorithm"]:
        assert rule in DEFAULT_SYSTEM_PROMPT
