"""Read-only action discovery. Availability is Python policy, never model text.

Tool adapters live here; no research-question matching or optical-task classifier.
Add reviewed tool adapters to the catalog as tools grow, not branches for questions.
"""

import json
import hashlib
from pathlib import Path
from typing import Literal
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from workflow.config import AppConfig
from workflow.state import AnalysisRequest, TMFitSettings, WorkflowState
from tools.python_execution import checked_inputs, contained_file
from tools.tm_numerics import OPERATIONS
from analysis.datasets import load_dataset, registered_artifacts


class AvailableAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = ""
    kind: Literal["propose", "advise", "prepare_procedure", "revise_procedure", "preflight",
                  "run_operation", "review_results", "interpret", "request_information", "conclude"]
    description: str
    record_ids: list[UUID] = Field(default_factory=list)
    proposal_id: UUID | None = None
    procedure_id: UUID | None = None
    analysis_ids: list[UUID] = Field(default_factory=list)
    request: AnalysisRequest | None = None


def _checked_json(ref, config):
    if ref.sha256 is None:
        raise ValueError("Registered dataset/index must have a recorded hash")
    checked = checked_inputs([ref], config)[0]
    path = Path(checked.path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Dataset/index JSON exceeds 1 MiB")
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_inventory(state: WorkflowState, config: AppConfig):
    datasets, limitations = {}, []
    inventory = []
    for acquisition in state.acquisitions:
        try:
            refs, _ = registered_artifacts(state, [acquisition.id])
        except ValueError:
            continue
        for ref in acquisition.raw_data:
            if Path(ref.path).suffix.lower() != ".json":
                continue
            try:
                payload = _checked_json(ref, config)
                if not isinstance(payload, dict):
                    continue
                if payload.get("kind") == "synthetic-bench-v1":
                    from analysis.bench_analysis import load_bench_dataset
                    manifest, data, _ = load_bench_dataset(ref, refs, config)
                    if acquisition.data_origin != "simulated" or (acquisition.hardware and acquisition.hardware.data_origin != "simulated"):
                        raise ValueError("Bench manifest conflicts with acquisition provenance")
                    inventory.append({"validated": True, "kind": "synthetic-bench-v1",
                        "dataset_artifact_id": str(ref.id), "input_record_id": str(acquisition.id),
                        "description": ref.description, "data_origin": "simulated",
                        "patterns": len(data["phases"]), "input_modes": data["phases"].shape[1],
                        "output_channels": 4, "fit_count": manifest["fit_patterns"],
                        "holdout_count": manifest["holdout_patterns"], "equal_input_amplitudes": True,
                        "clipped_readings": int((data["detector_saturation"] > 0).sum()),
                        "ground_truth_artifact_id": None})
                    continue
                if not {"arrays", "fit_indices", "holdout_indices"} <= payload.keys():
                    continue
                dataset = load_dataset(ref, refs, config)
                origins = [acquisition.data_origin, acquisition.hardware.data_origin if acquisition.hardware else None]
                if any(origin and origin != dataset.manifest.data_origin for origin in origins):
                    raise ValueError("Dataset conflicts with acquisition provenance")
                datasets[ref.id] = (ref, dataset, acquisition.id)
            except Exception as exc:
                limitations.append(f"Artifact {ref.id} unavailable as a dataset/index: {str(exc)[:300]}")
    for dataset_id, (ref, dataset, acquisition_id) in datasets.items():
        inventory.append({"validated": True, "dataset_artifact_id": str(dataset_id), "input_record_id": str(acquisition_id),
            "description": ref.description, "data_origin": dataset.manifest.data_origin,
            "provenance": dataset.manifest.provenance, "patterns": len(dataset.X),
            "input_modes": dataset.X.shape[1], "output_channels": dataset.intensities.shape[1],
            "fit_count": len(dataset.manifest.fit_indices), "holdout_count": len(dataset.manifest.holdout_indices),
            "pattern_content_sha256": hashlib.sha256(dataset.X.tobytes()).hexdigest(),
            "split_sha256": hashlib.sha256(json.dumps([dataset.manifest.fit_indices, dataset.manifest.holdout_indices]).encode()).hexdigest(),
            "equal_input_amplitudes": bool(np.allclose(np.abs(dataset.X), np.abs(dataset.X[0, 0]))),
            "ground_truth_artifact_id": None})
    return inventory, limitations


def build_catalog(state: WorkflowState, config: AppConfig, services):
    """Return only concrete, bounded operations with validated existing inputs."""
    datasets, limitations = dataset_inventory(state, config)
    completed = []
    for analysis in state.analyses:
        if analysis.status != "completed":
            continue
        try:
            registered_artifacts(state, [analysis.id])
        except ValueError:
            continue
        completed.append(analysis)
    capabilities = {"available": False, "backend_id": "disabled", "data_origin": None, "actions": []}
    if services.hardware_backend is not None and config.hardware_backend != "disabled":
        try:
            caps = services.hardware_backend.describe()
            capabilities = {"available": bool(caps.available), "backend_id": caps.backend_id,
                            "data_origin": caps.data_origin, "actions": list(caps.actions)}
        except Exception as exc:
            limitations.append(f"Acquisition capability unavailable: {exc}")
    actions = []
    def add(kind, description, **kwargs):
        actions.append(AvailableAction(id=f"a{len(actions) + 1}", kind=kind, description=description, **kwargs))
    def operation(request, description):
        ids = [*request.input_record_ids, request.dataset_artifact_id]
        if request.ground_truth_artifact_id:
            ids.append(request.ground_truth_artifact_id)
        add("run_operation", description, record_ids=list(dict.fromkeys(ids)), request=request)
    if services.scientist_model is not None:
        add("propose", "Ask Scientist to propose or revise the scientific plan")
    active = [p for p in state.proposals if p.id in state.control.active_proposal_ids]
    if active and services.analyst_model is not None and not any(
        r.stage == "before_acquisition" and r.status == "advised" and set(p.id for p in active) <= set(r.proposal_ids)
        for r in state.analyst_reviews):
        add("advise", "Ask Analyst for acquisition/data-design advice on active proposals", record_ids=[p.id for p in active])
    can_acquire = capabilities["available"] and state.control.acquisition_attempts < config.max_workflow_acquisition_attempts
    if can_acquire and services.experimentor_model is not None:
        for proposal in active:
            procedures = [p for p in state.procedures if p.proposal_id == proposal.id]
            if not procedures:
                add("prepare_procedure", "Prepare an acquisition procedure; execution still needs preflight and human approval",
                    record_ids=[proposal.id], proposal_id=proposal.id)
            elif state.control.procedure_revisions < config.max_procedure_revisions:
                proc = max(procedures, key=lambda p: p.version)
                add("revise_procedure", "Request a technical procedure revision; a revision requires fresh review/approval",
                    record_ids=[proposal.id, proc.id], proposal_id=proposal.id, procedure_id=proc.id)
    if can_acquire:
        for proc in state.procedures:
            if proc.proposal_id not in {p.id for p in active} or any(
                a.procedure_id == proc.id and a.status in ("completed", "started") for a in state.acquisitions):
                continue
            if any(p.proposal_id == proc.proposal_id and p.version > proc.version for p in state.procedures):
                continue
            if any(a.status == "rejected" and a.procedure_id == proc.id and a.procedure_fingerprint == proc.fingerprint()
                   for a in state.approvals):
                continue
            add("preflight", "Runner checks this exact procedure, then pauses for human review; this action cannot acquire",
                record_ids=[proc.id, proc.proposal_id], procedure_id=proc.id, proposal_id=proc.proposal_id)
    if not capabilities["available"]:
        limitations.append("No acquisition backend is available; missing saved data cannot be acquired in this configuration")
    if config.reviewed_numerics_enabled and state.control.analysis_attempts < config.max_workflow_analysis_attempts:
        for dataset in datasets:
            dataset_id = UUID(dataset["dataset_artifact_id"])
            sources = [a for a in completed if any(ref.id == dataset_id for ref in a.input_data)]
            if dataset.get("kind") == "synthetic-bench-v1":
                if not any(a.method_settings.get("operation") == "bench.retrieve" for a in sources):
                    request = AnalysisRequest(calculation="Fit phase-only intensity measurements; evaluate held-out camera detector readings with clipping diagnostics",
                        operation="bench.retrieve", input_record_ids=[UUID(dataset["input_record_id"])],
                        dataset_artifact_id=dataset_id, parameters=TMFitSettings(iterations=800, restarts=4, seed=7))
                    operation(request, "bench.retrieve: reference-free GGS, fit-only selection, held-out detector predictions, clipping flags and usable counts")
                continue
            for name in ("tm.validate_dataset", "tm.compare_reference_free"):
                if any(a.method_settings.get("operation") == name for a in sources):
                    continue
                request = AnalysisRequest(calculation=f"{name} on existing registered dataset", operation=name,
                    input_record_ids=[UUID(dataset["input_record_id"])], dataset_artifact_id=dataset_id,
                    ground_truth_artifact_id=UUID(dataset["ground_truth_artifact_id"]) if dataset["ground_truth_artifact_id"] and name != "tm.validate_dataset" else None,
                    parameters=TMFitSettings(iterations=800, restarts=4, seed=7))
                operation(request, f"{name}: {dataset['description']}; use saved fit/holdout data, never acquire")
            for source in sources:
                if source.method_settings.get("operation") != "tm.compare_reference_free":
                    continue
                required = {f"{method}.holdout_{metric}" for method in ("ggs", "intensity_ls") for metric in ("rmse", "mae")}
                if required <= source.metrics.keys() or any(a.method_settings.get("operation") == "tm.evaluate_saved_predictions" for a in sources):
                    continue
                try:
                    for ref in source.outputs:
                        contained_file(ref.path, config.run_output_path / "analysis")
                    params = TMFitSettings.model_validate({k: v for k, v in source.method_settings.items() if k != "operation"})
                except (ValueError, OSError):
                    continue
                truth_id = dataset["ground_truth_artifact_id"]
                request = AnalysisRequest(calculation="Evaluate existing saved predictions without refitting",
                    operation="tm.evaluate_saved_predictions", input_record_ids=[source.id], dataset_artifact_id=dataset_id,
                    ground_truth_artifact_id=UUID(truth_id) if truth_id and any(str(r.id) == truth_id for r in source.input_data) else None,
                    parameters=params)
                operation(request, "Evaluate saved predictions, including RMSE/MAE and optional noiseless-target scores")
        if state.analyst_reviews and state.analyst_reviews[-1].analysis_request:
            request = state.analyst_reviews[-1].analysis_request
            if request.operation in OPERATIONS and any(str(request.dataset_artifact_id) == d["dataset_artifact_id"] for d in datasets):
                refs, _ = registered_artifacts(state, request.input_record_ids)
                if request.dataset_artifact_id in refs and (not request.ground_truth_artifact_id or request.ground_truth_artifact_id in refs):
                    settings = {"operation": request.operation, **request.parameters.model_dump()}
                    if not any(a.method_settings == settings and any(r.id == request.dataset_artifact_id for r in a.input_data) for a in completed):
                        operation(request, "Run the Analyst's validated named operation request on its existing inputs")
    elif not config.reviewed_numerics_enabled:
        limitations.append("Reviewed numerical execution is disabled")
    else:
        limitations.append("Numerical attempt limit reached")
    result_ids = [a.id for a in completed]
    if result_ids:
        if active and services.analyst_model is not None and not any(
            r.status == "reviewed" and set(result_ids) <= set(r.analysis_ids) for r in state.analyst_reviews):
            add("review_results", "Have Analyst review completed results and missing evidence",
                record_ids=[*result_ids, *[p.id for p in active]], analysis_ids=result_ids)
        if services.scientist_model is not None:
            add("interpret", "Ask Scientist to interpret existing completed evidence; no new acquisition",
                record_ids=result_ids, analysis_ids=result_ids)
    event = state.control.router_event
    outcome = event.get("outcome", {})
    if event.get("source") == "scientist" and (outcome.get("kind") == "conclusion" or (
        outcome.get("kind") == "interpretation" and outcome.get("follow_up", "conclude") == "conclude")):
        add("conclude", "Finish using the Scientist's recorded conclusion", record_ids=[f.id for f in state.findings[-1:]])
    add("request_information", "Ask the user for missing data, unavailable capabilities or a necessary clarification; explain exactly what is missing")
    return actions, datasets, completed, capabilities, limitations
