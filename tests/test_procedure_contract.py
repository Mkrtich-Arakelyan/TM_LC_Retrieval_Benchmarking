"""Live-style drafts use capability IDs; invalid plans repair without user questions."""

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from agents.experimentor import UNRESOLVED_PREREQUISITES
from bench.config import BenchConfig
from bench.optical_bench import OpticalBench, SimulatedBackend
from bench.workflow import BENCH_QUESTION, PAPER_PASSAGES, setup_context
from fixtures.offline import LocalAnalyst, LocalExperimentor, LocalRouter, LocalScientist
from tools.hardware import preflight
from workflow.config import AppConfig
from workflow.graph import Integrations, WorkflowApp
from workflow.routing_catalog import build_catalog
from workflow.state import AcquisitionResult, Artifact, Setting, WorkflowState

FAILED_PREREQUISITES = [
    "Human review and approval must match this exact procedure version and fingerprint before acquisition.",
    "Runner preflight must verify actual backend capabilities and operating limits; the supplied capability description is a synthetic model contract, not laboratory verification.",
    "The acquisition protocol must use the fixed all-zero phase pattern for calibration, stop with diagnostics if the 3-5% whole-frame clipping target is not attained within 32 calibration frames, and freeze shutter and power after calibration.",
    "The synthetic phase-in-radians, equal-amplitude input and uniform coherent response within each detector ROI are model assumptions; no physical grayscale-command, applied-voltage, or command-to-optical-phase calibration is established.",
    "bench.retrieve is not among the verified actions supplied for this procedure. Its capability, input/output contract, and review must be established separately before retrieval or held-out prediction.",
]

MALFORMED_CITATION = "HUANG-S2: supplied passage describes phase-varied equal-amplitude inputs and a GGS-style iterative procedure that replaces predicted field amplitude with the square root of measured intensity."


class CitationScientist(LocalScientist):
    def __init__(self, location, *, repeat=False, validate_schema=False):
        self.location = location
        self.repeat = repeat
        self.validate_schema = validate_schema
        self.calls = []
        self.schemas = []
        self.bad_id = None

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return self

    def invoke(self, messages):
        payload = json.loads(messages[1][1])
        self.calls.append(payload)
        response = super().invoke(messages)
        if len(self.calls) == 1 or self.repeat:
            self.bad_id = response["proposals"][0]["id"]
            if self.location in ("top", "both"):
                response["cited_source_ids"] = [MALFORMED_CITATION]
            if self.location in ("proposal", "both"):
                response["proposals"][0]["literature_references"] = [MALFORMED_CITATION]
        else:
            feedback = payload["scientist_validation"]
            assert feedback["allowed_source_ids"] == ["HUANG-S2", "RAFAYELYAN-APP-A"]
            assert all(error["invalid_value"] == MALFORMED_CITATION for error in feedback["errors"])
            assert all(error["field"] for error in feedback["errors"])
            response["cited_source_ids"] = ["HUANG-S2"]
            response["explanation"] = "The supplied HUANG-S2 passage describes phase-varied equal-amplitude inputs."
        return self.schemas[-1].model_validate(response) if self.validate_schema else response


class LiveStylePlanner(LocalExperimentor):
    def __init__(self, mode="repair"):
        self.mode = mode
        self.calls = []
        self.bad_id = str(uuid4())

    def invoke(self, messages):
        payload = json.loads(messages[1][1])
        self.calls.append(payload)
        if self.mode == "missing_input":
            return {"kind": "request_information", "explanation": "The requested external reference dataset is missing; please provide its path.",
                    "handoff": {"recipient": "human", "message": "Please provide the external reference dataset path.", "reply_to": None},
                    "questions": ["Which external reference dataset should be used?"], "recommended_next_step": "Ask for the missing dataset"}
        response = super().invoke(messages)
        if len(self.calls) == 1 or self.mode == "never_repair":
            response["procedure"]["id"] = self.bad_id
            response["procedure"]["prerequisites"] = list(FAILED_PREREQUISITES)
        if self.mode == "malformed" and len(self.calls) == 1:
            response["procedure"]["prerequisites"] = [{"id": "synthetic phase mapping verified", "verified": True}]
        if self.mode == "claims_verified":
            response["procedure"]["prerequisites"] = payload["procedure_contract"]["required_prerequisite_ids"]
            response["explanation"] = "I verified all prerequisites; calibration succeeded and human approval is satisfied."
        if len(self.calls) > 1 and self.mode != "never_repair":
            assert payload["procedure_validation"]["errors"]
            assert payload["procedure_validation"]["allowed_prerequisite_ids"] == payload["procedure_contract"]["allowed_prerequisite_ids"]
            response["procedure"]["id"] = str(uuid4())
            response["procedure"]["prerequisites"] = payload["procedure_contract"]["required_prerequisite_ids"]
            response["explanation"] = "Revised prerequisite selectors to the supported IDs; workflow checks remain at their owning stages."
        return response


class TrackingBackend(SimulatedBackend):
    def __init__(self, config, *, unverified=False):
        super().__init__(config)
        self.execute_calls = []
        self.unverified = unverified

    def describe(self):
        caps = super().describe()
        if self.unverified:
            caps.prerequisites["synthetic phase mapping verified"].verified = False
            caps.prerequisites["synthetic phase mapping verified"].evidence = "No backend verification available"
        return caps

    def execute(self, *args, **kwargs):
        self.execute_calls.append((args, kwargs))
        return super().execute(*args, **kwargs)


def integrations(tmp_path, mode="repair", *, router=True, repairs=3, unverified=False,
                 matrix_name="identity-16x16.json"):
    matrix = Path(__file__).resolve().parents[1] / "configs/matrices" / matrix_name if matrix_name else None
    bench = BenchConfig(detector_matrix_path=str(matrix) if matrix else None)
    cfg = AppConfig(optical_bench=bench, hardware_backend="simulated", bench_test_agents=False,
                    reviewed_numerics_enabled=True, llm_router_enabled=router, max_workflow_iterations=40,
                    max_procedure_revisions=repairs, workflow_database_path=tmp_path / "workflow.sqlite",
                    run_output_path=tmp_path / "runs", hardware_input_path=tmp_path / "inputs",
                    numerical_input_path=tmp_path / "inputs", max_acquisition_measurements=bench.frame_budget)
    model = LiveStylePlanner(mode)
    backend = TrackingBackend(bench, unverified=unverified)
    services = Integrations(scientist_model=LocalScientist(), analyst_model=LocalAnalyst(),
                            experimentor_model=model, hardware_backend=backend,
                            router_model=LocalRouter() if router else None)
    architecture = tmp_path / "synthetic.txt"
    architecture.write_text(setup_context(bench), encoding="utf-8")
    return cfg, services, model, backend, architecture


@pytest.mark.parametrize("router", [True, False])
@pytest.mark.parametrize("mode,matrix_name", [("repair", None), ("repair", "identity-16x16.json"),
                                           ("repair", "complex-mixing-16x16.json"), ("malformed", "identity-16x16.json")])
def test_bad_live_style_prerequisites_repair_to_exact_approval_without_acquiring(tmp_path, monkeypatch, router, mode, matrix_name):
    cfg, services, model, backend, architecture = integrations(tmp_path, mode, router=router, matrix_name=matrix_name)
    frames = []
    monkeypatch.setattr(OpticalBench, "frame", lambda *args, **kwargs: frames.append(args) or pytest.fail("Camera called before approval"))
    with WorkflowApp(cfg, services) as app:
        view = app.start(BENCH_QUESTION, architecture_path=architecture)
        assert view["status"] == "awaiting_approval", view.get("reason")
        assert view["pending"]["kind"] == "review"
        state = WorkflowState.model_validate(view["state"])
        assert state.control.procedure_revisions == 1 and len(state.procedures) == 1
        assert str(state.procedures[0].id) != model.bad_id
        assert set(state.procedures[0].prerequisites) == set(backend.describe().prerequisites)
        assert not any(p in state.procedures[0].prerequisites for p in UNRESOLVED_PREREQUISITES)
        assert state.acquisition_reviews[-1].procedure_fingerprint == state.procedures[0].fingerprint()
        assert len(model.calls) == 2
        assert not state.acquisitions and not state.approvals and not frames and not backend.execute_calls
        contract = model.calls[0]["procedure_contract"]
        assert all({"id", "verified", "evidence"} <= check.keys() for check in contract["supported_prerequisites"])
        assert "bench.retrieve" in model.calls[0]["numerical_capabilities"]["operations"]
        assert "bench.retrieve" not in state.control.hardware_context["actions"]
        assert "procedure.prerequisites" in str(model.calls[1]["procedure_validation"]["errors"])
        rejected = app.submit(view["workflow_id"], {"action": "approve", "review_fingerprint": "wrong"})
        assert rejected["input_error"] and not backend.execute_calls and not frames
        assert app.show(view["workflow_id"])["pending"]["review"]["review_fingerprint"] == view["pending"]["review"]["review_fingerprint"]


def test_exhausted_repair_budget_is_field_error_not_a_clarification(tmp_path):
    cfg, services, model, backend, architecture = integrations(tmp_path, "never_repair", repairs=1)
    with WorkflowApp(cfg, services) as app:
        view = app.start(BENCH_QUESTION, architecture_path=architecture)
    assert view["status"] == "blocked" and view["pending"] is None
    assert "Procedure validation error" in view["reason"]
    assert "procedure.prerequisites[0]" in view["reason"] and "allowed choices" in view["reason"]
    assert "budget exhausted" in view["reason"] and len(model.calls) == 2
    assert view["state"]["control"]["procedure_revisions"] == 1
    assert not view["state"]["procedures"] and not backend.execute_calls


@pytest.mark.parametrize("router", [True, False])
def test_genuine_missing_user_input_still_asks_for_clarification(tmp_path, router):
    cfg, services, model, backend, architecture = integrations(tmp_path, "missing_input", router=router)
    with WorkflowApp(cfg, services) as app:
        view = app.start("Compare the identity simulation with my external reference dataset", architecture_path=architecture)
    assert view["status"] == "awaiting_information" and view["pending"]["kind"] == "clarification"
    assert "reference dataset" in " ".join(view["pending"]["questions"])
    assert len(model.calls) == 1 and not backend.execute_calls


def test_model_claim_cannot_verify_a_supported_but_unverified_requirement(tmp_path):
    cfg, services, model, backend, architecture = integrations(tmp_path, "claims_verified", unverified=True)
    with WorkflowApp(cfg, services) as app:
        view = app.start(BENCH_QUESTION, architecture_path=architecture)
    assert view["status"] == "blocked" and view["pending"] is None
    assert "Unresolved prerequisite: synthetic phase mapping verified" in view["reason"]
    assert not view["state"]["acquisitions"] and not view["state"]["approvals"] and not backend.execute_calls
    proc = WorkflowState.model_validate(view["state"]).procedures[-1]
    proc.prerequisites = ["synthetic bench configuration valid"]
    assert any("Unresolved prerequisite: synthetic phase mapping verified" in error for error in preflight(proc, cfg, backend).issues)
    proc.prerequisites = ["synthetic phase mapping verified = satisfied by the agent"]
    result = preflight(proc, cfg, backend)
    assert result.status == "blocked" and any("unsupported prerequisite ID" in error for error in result.issues)
    assert not backend.describe().prerequisites["synthetic phase mapping verified"].verified


def test_changed_capabilities_trigger_internal_technical_revision_not_user_question(tmp_path):
    cfg, services, model, backend, architecture = integrations(tmp_path)
    with WorkflowApp(cfg, services) as app:
        waiting = app.start(BENCH_QUESTION, architecture_path=architecture)
        state = WorkflowState.model_validate(waiting["state"])
        state.procedures[-1].prerequisites.append("stale capability ID")
        review = preflight(state.procedures[-1], cfg, backend)
        app._preflight_issues(state, SimpleNamespace(technical_issues=review.issues))
        assert state.control.next_node == "experimentor" and state.control.task == "technical_review"
        assert state.status != "awaiting_information"


def test_missing_retrieval_capability_is_reported_at_analysis_stage(tmp_path):
    cfg, services, _, backend, _ = integrations(tmp_path)
    cfg.reviewed_numerics_enabled = False
    state = WorkflowState(user_request="Retrieve from saved simulated measurements")
    state.control.configuration = cfg.model_dump(mode="json")
    recorded = tmp_path / "completed-acquisition.json"
    recorded.write_text('{"data_origin":"simulated"}')
    acquisition = AcquisitionResult(procedure_id=uuid4(), procedure_version=1, procedure_fingerprint="0"*64,
        status="completed", data_origin="simulated", actual_settings={"frame_budget": Setting(value=592, unit="frame")},
        raw_data=[Artifact(path=str(recorded), description="Recorded simulated acquisition")])
    state.acquisitions.append(acquisition)
    with WorkflowApp(cfg, services) as app:
        app._saved_acquisition_analysis(state, acquisition)
    assert state.status == "blocked" and state.control.pending is None
    assert "Analysis capability unavailable: bench.retrieve" in state.control.reason
    assert not backend.execute_calls


@pytest.mark.parametrize("router", [True, False])
@pytest.mark.parametrize("location,validate_schema", [("top", False), ("proposal", False), ("both", True)])
def test_scientist_citations_repair_before_procedure_and_exact_approval(tmp_path, monkeypatch, router, location, validate_schema):
    cfg, services, planner, backend, architecture = integrations(tmp_path, router=router)
    model = CitationScientist(location, validate_schema=validate_schema)
    services.scientist_model = model
    monkeypatch.setattr(OpticalBench, "frame", lambda *args, **kwargs: pytest.fail("Camera called before approval"))
    with WorkflowApp(cfg, services) as app:
        view = app.start(BENCH_QUESTION, architecture_path=architecture, literature_context=PAPER_PASSAGES)
        state = WorkflowState.model_validate(view["state"])
        assert view["status"] == "awaiting_approval" and view["pending"]["kind"] == "review"
        assert len(model.calls) == 2 and len(state.proposals) == 1 and len(state.procedures) == 1
        assert str(state.proposals[0].id) != model.bad_id
        assert state.proposals[0].literature_references == [p.source_id for p in PAPER_PASSAGES]
        assert planner.calls
        assert not state.acquisitions and not state.approvals and not backend.execute_calls
        assert any(
            "Rejected Scientist output" in entry.message for entry in state.conversation)
        review = view["pending"]["review"]
        assert state.acquisition_reviews[-1].procedure_fingerprint == state.procedures[-1].fingerprint()
        rejected = app.submit(view["workflow_id"], {"action": "approve", "review_fingerprint": "wrong"})
        assert rejected["input_error"] and not backend.execute_calls
        assert app.show(view["workflow_id"])["pending"]["review"]["review_fingerprint"] == review["review_fingerprint"]


@pytest.mark.parametrize("location", ["top", "proposal"])
def test_repeated_invalid_scientist_citations_block_without_user_clarification(tmp_path, location):
    cfg, services, planner, backend, architecture = integrations(tmp_path)
    model = CitationScientist(location, repeat=True, validate_schema=True)
    services.scientist_model = model
    with WorkflowApp(cfg, services) as app:
        view = app.start(BENCH_QUESTION, architecture_path=architecture, literature_context=PAPER_PASSAGES)
    assert view["status"] == "blocked" and view["pending"] is None
    assert "Scientist output validation error" in view["reason"] and "budget exhausted" in view["reason"]
    assert MALFORMED_CITATION in view["reason"] and "allowed_source_ids" in view["reason"]
    assert ("cited_source_ids.0" if location == "top" else "proposals.0.literature_references.0") in view["reason"]
    assert len(model.calls) == 2 and not planner.calls and not backend.execute_calls
    assert not view["state"]["proposals"] and not view["state"]["procedures"]
    assert not view["state"]["acquisitions"] and not view["state"]["approvals"]
    assert any("Rejected Scientist output" in entry["message"] for entry in view["state"]["conversation"])
