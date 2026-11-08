"""Explicit offline handoff fixtures. No generated script execution or optical model."""

import json
from uuid import UUID, uuid5

from agents.experimentor import ExecutionReview, UNRESOLVED_PREREQUISITES
from state import Artifact, ExperimentProposal
from tools.hardware import RecordingFakeBackend, Verification
from tools.python_execution import BackendReport, fingerprint


class FixtureScientistModel:
    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        state = json.loads(messages[1][1])["workflow_state"]
        if state["control"]["task"] == "interpretation" and state["analyses"]:
            latest = state["analyses"][-1]
            completed = {a["id"] for a in state["acquisitions"] if a["status"] == "completed"}
            if any(key not in completed for key in latest["acquisition_ids"]):
                return {"kind": "conclusion", "explanation": "TEST ONLY partial-data diagnostics recorded; no complete-dataset scientific finding can be made.",
                        "recommended_next_step": "Review partial-data limitations"}
            return {"kind": "interpretation", "explanation": "TEST ONLY: software handoff records received; no scientific conclusion is validated.",
                "recommended_next_step": "Review the interface fixture",
                "findings": [{"evidence_ids": [latest["id"]],
                    "interpretation": "TEST ONLY fixture records, not physical observations or a TM-retrieval finding.",
                    "limitations": ["Canned response; no scientific reasoning or optical experiment"],
                    "recommended_next_step": "Configure and validate real integrations separately"}]}
        proposal = ExperimentProposal(scientific_question="TEST ONLY: can software records traverse the workflow?",
            hypothesis="TEST ONLY handoff assertion", rationale="Interface fixture, not scientific inference",
            variables={"fixture": "Software record"}, controls=["No physical device"],
            required_measurements=["Test-only fixture file"], evaluation_criteria=["Record links and statuses survive restart"],
            assumptions=["This is not a scientific benchmark"])
        return {"kind": "proposal", "explanation": "TEST ONLY canned proposal for software handoffs.",
                "recommended_next_step": "Prepare fixture procedure", "proposals": [proposal.model_dump(mode="json")]}


class FixtureExperimentorModel:
    def __init__(self, hardware: RecordingFakeBackend):
        self.hardware = hardware

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        payload = json.loads(messages[1][1])
        state = payload["workflow_state"]
        if self.schema is ExecutionReview:
            return {"technical_explanation": "TEST ONLY: fake execution report inspected; proposed Python was never run.",
                    "recommended_next_step": "Send fixture records to Scientist", "retry": False}
        if state["control"]["task"] == "analysis" or (state["control"]["task"] == "technical_review" and state["control"]["allow_partial_analysis"]):
            eligible = [a for a in state["acquisitions"] if a["raw_data"] and
                        (a["status"] == "completed" or state["control"]["allow_partial_analysis"])]
            if not eligible:
                return {"kind": "request_information", "explanation": "TEST ONLY: saved data is missing.",
                        "recommended_next_step": "Provide saved data", "questions": ["Which saved data should be analyzed?"]}
            acquisition = eligible[-1]
            return {"kind": "analysis", "explanation": "TEST ONLY generates a script artifact; fake backend supplies canned diagnostics without executing it.",
                "recommended_next_step": "Inspect fake report", "analysis": {
                    "acquisition_ids": [acquisition["id"]], "input_data": acquisition["raw_data"],
                    "source": "# TEST ONLY proposed code; must never be run on the host.\nraise RuntimeError('Fixture script is not executable analysis')\n",
                    "method_settings": {"fixture_only": True}, "random_seed": 0}}
        if state["control"]["task"] == "technical_review" and any(a["status"] in ("partial", "failed", "aborted", "started") for a in state["acquisitions"]):
            return {"kind": "request_information", "explanation": "TEST ONLY: acquisition requires technical review, not automatic replay.",
                    "recommended_next_step": "Inspect the saved partial/uncertain records",
                    "questions": ["Review the acquisition failure and specify how to proceed. Partial-data analysis needs explicit authorization."]}
        active = state["control"]["active_proposal_ids"]
        proposal_id = active[0] if active else state["proposals"][-1]["id"]
        return {"kind": "procedure", "explanation": "TEST ONLY fixture procedure, no device APIs or scientific benchmark.",
            "recommended_next_step": "Runner preflight and explicit human review", "procedure": {
                "proposal_id": proposal_id, "version": payload["next_procedure_versions"][proposal_id],
                "actions": [{"action": "fixture.capture", "instruction": "Create labelled test-only fixture file",
                             "requested_settings": {"count": {"value": 1, "unit": "frame"}}}],
                "prerequisites": ["fixture setup verified"], "expected_outputs": ["TEST ONLY fixture.json"],
                "calibration_artifacts": {k: v.model_dump(mode="json") for k, v in self.hardware.describe().verified_calibrations.items()}}}


class FixtureExecutionBackend:
    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        path = kwargs["output_dir"] / "fixture-report.txt"
        path.write_text("TEST ONLY canned artifact; generated Python was not executed.", encoding="utf-8")
        return BackendReport(exit_code=0, stdout="TEST ONLY fake backend: no Python executed",
            elapsed_seconds=0.0, metrics={"fixture_record_check": 1},
            outputs=[Artifact(path=str(path), description="TEST ONLY handoff artifact, not numerical evidence from a physical experiment")])


class FixtureAnalystModel:
    """Canned metadata advice/review only; no numerical calculation."""
    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        payload = json.loads(messages[1][1])
        before = payload["stage"] == "before_acquisition"
        return {"stage": payload["stage"], "status": "advised" if before else "reviewed",
                "proposal_ids": payload["proposal_ids"], "analysis_ids": payload["analysis_ids"],
                "evidence_ids": payload["proposal_ids"] if before else payload["analysis_ids"],
                "explanation": "TEST ONLY canned Analyst metadata handoff, not a validated scientific review.",
                "recommendations": ["Confirm coherent-reference requirements and command-to-phase calibration; specify controls and held-out data." if before else
                    "Check whether metrics answer the comparison; independent validation is not established by fixture metadata."],
                "limitations": ["TEST ONLY fixtures; no files inspected, calculations performed, or physical conclusions established."]}


def offline_integrations(config, workflow_id: UUID, *, include_analyst: bool = False) -> dict:
    config.hardware_input_path.mkdir(parents=True, exist_ok=True)
    path = config.hardware_input_path / "calibration-fixture.json"
    if not path.exists():
        path.write_text('{"kind":"TEST ONLY; not laboratory calibration"}', encoding="utf-8")
    artifact = Artifact(id=uuid5(workflow_id, "calibration-fixture-v1"), path=str(path),
                        description="TEST ONLY calibration fixture", sha256=fingerprint(path))
    hardware = RecordingFakeBackend(artifact)
    for prerequisite in UNRESOLVED_PREREQUISITES:
        hardware.capabilities.prerequisites[prerequisite] = Verification(
            verified=True, evidence="TEST ONLY fixture assertion of interface requirements; not laboratory verification")
    return {"scientist_model": FixtureScientistModel(), "experimentor_model": FixtureExperimentorModel(hardware),
            "hardware_backend": hardware, "execution_backend": FixtureExecutionBackend(),
            "analyst_model": FixtureAnalystModel() if include_analyst else None}
