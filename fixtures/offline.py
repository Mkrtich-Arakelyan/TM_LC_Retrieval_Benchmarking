"""Explicit offline handoff fixtures. No generated script execution or optical model."""

import json
from uuid import UUID, uuid5

from agents.experimentor import ExecutionReview, UNRESOLVED_PREREQUISITES
from workflow.state import Artifact, ExperimentProposal
from fixtures.hardware import RecordingFakeBackend
from tools.hardware import Verification
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


# Synthetic bench agent doubles. These remain separate from the interface-only
# Fixture* agents above; the synthetic backend and numerical tools execute normally.
class LocalModel:
    """Explicit local test doubles; optical acquisition and fitting remain real code."""
    def with_structured_output(self, schema):
        return self


class LocalScientist(LocalModel):
    def invoke(self, messages):
        payload = json.loads(messages[1][1])
        state = payload["workflow_state"]
        if state["control"]["task"] == "interpretation":
            latest = state["analyses"][-1]
            metrics = latest["metrics"]
            passed = bool(metrics.get("prediction_passed"))
            return {"kind": "interpretation", "explanation": f"LOCAL TEST DOUBLE: held-out prediction {'passed' if passed else 'failed or is inconclusive'}; this is synthetic evidence, not laboratory validation.",
                "recommended_next_step": "Inspect individual calibration and prediction artifacts",
                "findings": [{"evidence_ids": [latest["id"]],
                    "interpretation": f"Synthetic held-out prediction_passed={int(passed)}; measurement_passed={metrics.get('measurement_passed', 0)}.",
                    "limitations": ["Local canned agent reasoning", "Uniform detector ROIs and ideal stable phase-only model", "Independent detector-row phase ambiguity"],
                    "recommended_next_step": "Review evidence before improving the physical model"}]}
        proposal = ExperimentProposal(scientific_question=state["user_request"],
            hypothesis="A reference-free intensity fit can predict unseen phase-only detector readings",
            rationale="Candidate approaches: GGS amplitude replacement or intensity least squares; both require diverse known phases, not measured complex fields",
            variables={"approaches": "GGS or intensity least squares", "inputs": "Random phase-only SLM patterns"},
            controls=["Same camera settings for fit and holdout", "Separate deterministic fit and held-out patterns"],
            required_measurements=["Whole raw camera frames and mean ROI intensities"],
            evaluation_criteria=["Held-out detector RMSE <=2 counts and NMSE <=0.02 with at least 8 usable readings per detector"],
            assumptions=["Synthetic model only", "Uniform coherent detector ROIs"],
            literature_references=[p["source_id"] for p in payload["literature_context"]])
        return {"kind": "proposal", "explanation": "LOCAL TEST DOUBLE: propose GGS and intensity least squares as intensity-only candidates.",
                "recommended_next_step": "Experimentor selects the reference GGS acquisition/evaluation procedure",
                "proposals": [proposal.model_dump(mode="json")]}


class LocalExperimentor(LocalModel):
    def invoke(self, messages):
        payload = json.loads(messages[1][1])
        state = payload["workflow_state"]
        if any(a["status"] in ("failed", "partial", "aborted") for a in state["acquisitions"]):
            return {"kind": "request_information", "explanation": "LOCAL TEST DOUBLE: calibration/acquisition stopped; preserve diagnostics and do not automatically repeat.",
                "recommended_next_step": "Inspect the saved calibration failure", "questions": ["Review model bounds and saved stopping reason before requesting another run."]}
        cfg = state["control"]["configuration"]["optical_bench"]
        proposal_id = state["control"]["active_proposal_ids"][0]
        return {"kind": "procedure", "explanation": "LOCAL TEST DOUBLE: select reference-free GGS, acquire random phase-only patterns after calibration; no coherent reference or amplitude masking is used.",
            "recommended_next_step": "Runner preflight, then exact human review/approval", "procedure": {
                "proposal_id": proposal_id, "version": payload["next_procedure_versions"][proposal_id],
                "actions": [{"action": "bench.acquire", "instruction": "Calibrate fixed pattern to whole-frame 3–5% saturation, freeze settings, acquire fit and held-out phase patterns",
                    "requested_settings": {"shutter_seconds": {"value": cfg["shutter_seconds"], "unit": "s"},
                        "laser_power_mw": {"value": cfg["laser_power_mw"], "unit": "mW"},
                        "frame_budget": {"value": cfg["fit_patterns"] + cfg["holdout_patterns"] + cfg["calibration_attempts"], "unit": "frame"}}}],
                "prerequisites": ["synthetic phase mapping verified"], "expected_outputs": ["Calibration diagnostics, raw frames, independent fit/holdout detector dataset"],
                "output_subdirectory": "bench"}}


class LocalAnalyst(LocalModel):
    def invoke(self, messages):
        payload = json.loads(messages[1][1])
        before = payload["stage"] == "before_acquisition"
        state = payload["workflow_state"]
        metrics = state["analyses"][-1]["metrics"] if not before else {}
        return {"stage": payload["stage"], "status": "advised" if before else "reviewed",
            "proposal_ids": payload["proposal_ids"], "analysis_ids": payload["analysis_ids"],
            "evidence_ids": payload["proposal_ids"] if before else payload["analysis_ids"],
            "explanation": "LOCAL TEST DOUBLE: require separate holdout, clipped-fit exclusion and row-phase ambiguity." if before else
                f"LOCAL TEST DOUBLE: measurement_passed={metrics.get('measurement_passed')}, prediction_passed={metrics.get('prediction_passed')}; inspect per-detector usable counts and held-out errors.",
            "recommendations": ["Use bench.retrieve on saved raw frames; do not use hidden fields or truth" if before else "Keep workflow completion distinct from measurement and prediction quality"],
            "limitations": ["Local canned agent reasoning; synthetic model only"]}


class LocalRouter(LocalModel):
    def invoke(self, messages):
        context = json.loads(messages[1][1])
        actions = context["available_actions"]
        latest = context["latest_agent_outcome"]
        if latest.get("kind") in ("acquisition_failed", "request_information"):
            order = ["request_information"]
        else:
            order = ["conclude", "run_operation", "review_results", "interpret", "advise", "preflight", "prepare_procedure", "propose", "request_information"]
        selected = next(a for kind in order for a in actions if a["kind"] == kind)
        return {"action_id": selected["id"], "record_ids": selected["record_ids"],
            "reason": "LOCAL TEST DOUBLE: " + (latest.get("explanation", "Stopped acquisition; inspect diagnostics") if selected["kind"] == "request_information" else selected["description"])}
