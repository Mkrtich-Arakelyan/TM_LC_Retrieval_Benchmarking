"""Deterministic Runner: review and approved acquisition, with no LLM dependency."""

import json
from threading import Event
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from workflow.config import AppConfig
from workflow.state import (AcquisitionResult, AcquisitionReview, ConversationEntry,
                   HumanApproval, WorkflowState)
from tools.hardware import HardwareBackend, execute_acquisition, preflight


class RunnerResult(BaseModel):
    status: Literal["review", "blocked", "completed", "failed", "aborted", "partial"]
    explanation: str
    review: AcquisitionReview
    acquisition: AcquisitionResult | None = None
    technical_issues: list[str] = Field(default_factory=list)
    recommended_next_step: str
    state: WorkflowState


def review_summary(review: AcquisitionReview) -> str:
    proc = review.procedure
    lines = [f"Runner preflight: {review.status}. Procedure {proc.id}, version {proc.version}.",
             f"Procedure fingerprint: {review.procedure_fingerprint}",
             f"Review fingerprint: {review.review_fingerprint}"]
    for index, action in enumerate(proc.actions):
        lines.append(f"{index + 1}. {action.action or '[missing action identifier]'}: {action.instruction}. "
                     f"Requested settings: {json.dumps({k: v.model_dump() for k, v in action.requested_settings.items()})}")
    lines.extend([f"Expected outputs: {', '.join(proc.expected_outputs)}",
                  f"Raw-data subdirectory: {proc.output_subdirectory}",
                  "Reviewed capabilities, units, verified limits, and prerequisites: "
                  + json.dumps(review.capabilities_snapshot, sort_keys=True),
                  "Reviewed output/budget/timeout configuration: "
                  + json.dumps(review.configuration_snapshot, sort_keys=True)])
    lines.extend(f"Unresolved: {issue}" for issue in review.issues)
    return "\n".join(lines)


def runner(
    state: WorkflowState, procedure_id: UUID, *, config: AppConfig | None = None,
    backend: HardwareBackend | None = None, execute: bool = False,
    reviewed: AcquisitionReview | None = None, approval: HumanApproval | None = None,
    attempt_id: UUID | None = None, cancel_event: Event | None = None,
) -> RunnerResult:
    """Trusted application caller supplies approval explicitly; never from a model.

    Preflight alone is read-only with respect to equipment and disk. A successful
    invocation returns copied state with one conversation entry. The hardware tool
    alone owns validation, durable acquisition records, and backend invocation.
    """
    cfg = AppConfig.model_validate((config or AppConfig()).model_dump())
    snapshot = WorkflowState.model_validate(state.model_dump())
    candidates = [p for p in snapshot.procedures if p.id == procedure_id]
    if len(candidates) != 1:
        raise ValueError("Runner requires one unambiguous procedure identity in workflow state")
    procedure = candidates[0]
    review = preflight(procedure, cfg, backend)
    issues = list(review.issues)
    if procedure.proposal_id not in {p.id for p in snapshot.proposals}:
        issues.append("Procedure does not reference a Scientist proposal in workflow state")
    acquisition = None
    status = "blocked" if issues else "review"
    if execute:
        if reviewed is None:
            issues.append("Acquisition requires the exact prior reviewed snapshot and trusted-caller approval")
        if attempt_id is not None and any(a.id == attempt_id for a in snapshot.acquisitions):
            issues.append("Acquisition attempt already exists in state; recovery review is required, never automatic replay")
        if not issues:
            run = execute_acquisition(procedure, reviewed, approval, cfg, backend,
                                      attempt_id=attempt_id, cancel_event=cancel_event)
            review, acquisition = run.review, run.acquisition
            issues.extend(run.issues)
        status = acquisition.status if acquisition else "blocked"
    explanation = review_summary(review)
    if acquisition is not None:
        snapshot.acquisitions.append(acquisition.model_copy(deep=True))
        completed = [a.index + 1 for a in acquisition.hardware.actions if a.status == "completed"]
        uncertain = [a.index + 1 for a in acquisition.hardware.actions if a.status in ("started", "failed", "aborted")]
        pending = [a.index + 1 for a in acquisition.hardware.actions if a.status == "pending"]
        explanation += (f"\nAcquisition {acquisition.id}: {acquisition.status}; "
                        f"data origin: {acquisition.hardware.data_origin}. "
                        f"Completed actions: {completed}; uncertain/failed actions: {uncertain}; "
                        f"not attempted: {pending}. Cleanup: {acquisition.hardware.cleanup_status}.")
    if issues:
        explanation += "\nTechnical issues for Experimentor:\n" + "\n".join(issues)
        next_step = ("Experimentor must resolve technical issues and review incomplete measurements. "
                     "If controls, budgets, or scientific assumptions change, request Scientist review. "
                     "Another acquisition requires fresh preflight and matching approval; no automatic retry.")
    elif acquisition is not None:
        next_step = "Experimentor may analyze the saved records; test-only data must never be represented as physical observations."
    else:
        next_step = "Trusted application caller must obtain human approval of this exact procedure and review before requesting acquisition."
    explanation += "\nNext step: " + next_step
    snapshot.acquisition_reviews.append(review.model_copy(deep=True))
    record_ids = [review.id] + ([acquisition.id] if acquisition else [])
    snapshot.conversation.append(ConversationEntry(role="runner", message=explanation, record_ids=record_ids))
    return RunnerResult(status=status, explanation=explanation, review=review, acquisition=acquisition,
                        technical_issues=issues, recommended_next_step=next_step, state=snapshot)
