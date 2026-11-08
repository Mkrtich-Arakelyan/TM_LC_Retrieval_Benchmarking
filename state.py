"""Shared records only; no acquisition, execution, or agent behavior."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Status = Literal["pending", "completed", "failed"]
Role = Literal["user", "scientist", "analyst", "experimentor", "runner", "orchestrator", "human"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    id: UUID = Field(default_factory=uuid4)


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: UUID = Field(default_factory=uuid4)
    path: Text
    description: Text
    media_type: Text | None = None
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None


class Setting(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    value: str | int | float | bool
    unit: Text | None = None


class ExperimentProposal(Record):
    scientific_question: Text
    hypothesis: Text
    rationale: Text
    variables: dict[Text, Text] = Field(min_length=1)
    controls: list[Text] = Field(min_length=1)
    required_measurements: list[Text] = Field(min_length=1)
    evaluation_criteria: list[Text] = Field(min_length=1)
    assumptions: list[Text] = Field(default_factory=list)
    literature_references: list[Text] = Field(default_factory=list)


class AcquisitionAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instruction: Text
    action: Text | None = None
    requested_settings: dict[Text, Setting] = Field(default_factory=dict)


class ExperimentProcedure(Record):
    proposal_id: UUID
    version: int = Field(ge=1)
    actions: list[AcquisitionAction] = Field(min_length=1)
    prerequisites: list[Text] = Field(min_length=1)
    expected_outputs: list[Text] = Field(min_length=1)
    input_artifacts: list[Artifact] = Field(default_factory=list)
    calibration_artifacts: dict[Text, Artifact] = Field(default_factory=dict)
    output_subdirectory: Text = "raw"

    def fingerprint(self) -> str:
        """Hash every field, including ordered actions and requested settings."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True,
                             separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ActionProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=0)
    requested: AcquisitionAction
    status: Literal["pending", "started", "completed", "failed", "aborted"] = "pending"
    started_at: datetime | None = None
    ended_at: datetime | None = None
    actual_settings: dict[Text, Setting] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    errors: list[Text] = Field(default_factory=list)


class HardwareExecution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approval_id: UUID
    review_id: UUID
    review_fingerprint: Text
    backend_id: Text
    data_origin: Literal["physical", "simulated", "test_only"]
    run_directory: Text
    started_at: datetime
    ended_at: datetime | None = None
    actions: list[ActionProgress]
    cleanup_status: Literal["not_requested", "pending", "completed", "failed", "unsupported"] = "not_requested"
    cleanup_message: str | None = None


class AcquisitionResult(Record):
    data_origin: Literal["physical", "simulated", "test_only"] | None = None
    procedure_id: UUID
    procedure_version: int = Field(ge=1)
    procedure_fingerprint: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    status: Literal["pending", "started", "completed", "failed", "aborted", "partial"] = "pending"
    actual_settings: dict[Text, Setting] = Field(default_factory=dict)
    raw_data: list[Artifact] = Field(default_factory=list)
    errors: list[Text] = Field(default_factory=list)
    hardware: HardwareExecution | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> AcquisitionResult:
        if self.status == "completed" and (not self.raw_data or not self.actual_settings or self.errors):
            raise ValueError("Completed acquisition requires raw data and actual settings, without errors")
        if self.status in ("failed", "aborted", "partial") and not self.errors:
            raise ValueError("Failed, aborted, or partial acquisition requires an error")
        if self.status == "completed" and self.hardware is not None:
            if (self.hardware.ended_at is None
                    or any(a.status != "completed" for a in self.hardware.actions)
                    or self.hardware.cleanup_status == "failed"):
                raise ValueError("Completed hardware acquisition requires all actions to finish")
        return self


class ExecutionDetails(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    status: Literal["unavailable", "completed", "failed"]
    backend: Text
    attempt_number: int = Field(ge=1)
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    elapsed_seconds: float = Field(default=0, ge=0)
    reason: str | None = None


class NumericalAnalysisResult(Record):
    data_origin: Literal["physical", "simulated", "test_only"] | None = None
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)
    acquisition_ids: list[UUID] = Field(min_length=1)
    input_data: list[Artifact] = Field(min_length=1)
    code: Artifact
    code_version: Text
    method_settings: dict[Text, JsonValue] = Field(default_factory=dict)
    random_seed: int | None = None
    status: Status = "pending"
    metrics: dict[Text, float] = Field(default_factory=dict)
    outputs: list[Artifact] = Field(default_factory=list)
    errors: list[Text] = Field(default_factory=list)
    code_record_id: UUID | None = None
    execution: ExecutionDetails | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> NumericalAnalysisResult:
        if self.execution is not None:
            expected = {"unavailable": "pending", "completed": "completed", "failed": "failed"}
            if self.status != expected[self.execution.status]:
                raise ValueError("Analysis and execution statuses disagree")
            if self.status == "completed" and self.execution.exit_code != 0:
                raise ValueError("Completed execution requires exit code zero")
        if self.status == "completed" and (not (self.metrics or self.outputs) or self.errors):
            raise ValueError("Completed analysis requires metrics or outputs, without errors")
        if self.status == "failed" and not self.errors:
            raise ValueError("Failed analysis requires an error")
        return self


class ScientificFinding(Record):
    evidence_ids: list[UUID] = Field(min_length=1)
    interpretation: Text
    limitations: list[Text] = Field(min_length=1)
    recommended_next_step: Text


class HumanApproval(Record):
    procedure_id: UUID
    procedure_version: int = Field(ge=1)
    procedure_fingerprint: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    status: Literal["pending", "approved", "rejected", "revoked"] = "pending"
    reviewer: Text | None = None
    comment: Text | None = None
    review_fingerprint: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None


def approval_matches(approval: HumanApproval, procedure: ExperimentProcedure) -> bool:
    """Pure comparison only; this does not authorize or perform hardware operations."""
    return (approval.status == "approved" and approval.procedure_id == procedure.id
            and approval.procedure_version == procedure.version
            and approval.procedure_fingerprint == procedure.fingerprint())


class LiteraturePassage(BaseModel):
    """Caller-supplied passage; no retrieval or claim that a paper was read."""
    model_config = ConfigDict(extra="forbid")
    source_id: Text
    filename: Text
    page_number: int | None = Field(default=None, ge=1)
    passage: Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]


class ScientificContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    architecture_path: str
    architecture_text: str
    literature_context: list[LiteraturePassage] = Field(default_factory=list)



class AnalysisCode(Record):
    """Saved proposed script, distinct from any execution of that script."""
    acquisition_ids: list[UUID] = Field(min_length=1)
    input_data: list[Artifact] = Field(min_length=1)
    code: Artifact
    method_settings: dict[Text, JsonValue] = Field(default_factory=dict)
    random_seed: int | None = None
    technical_explanation: Text
    scientific_context: ScientificContext


class ConversationEntry(Record):
    role: Role
    message: Text
    record_ids: list[UUID] = Field(default_factory=list)
    scientific_context: ScientificContext | None = None


class TMFitSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    iterations: int = Field(default=400, ge=1, le=2000)
    restarts: int = Field(default=2, ge=1, le=8)
    seed: int = Field(default=7, ge=0, le=4294967295)
    tolerance: float = Field(default=1e-10, ge=0, le=0.01)
    step_size: float = Field(default=1.0, gt=0, le=10)


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    calculation: Text
    input_record_ids: list[UUID] = Field(min_length=1)
    input_artifact_paths: list[Text] = Field(default_factory=list)
    operation: Literal["tm.validate_dataset", "tm.compare_reference_free"] | None = None
    dataset_artifact_id: UUID | None = None
    ground_truth_artifact_id: UUID | None = None
    parameters: TMFitSettings = Field(default_factory=TMFitSettings)

    @model_validator(mode="after")
    def named_inputs(self) -> AnalysisRequest:
        if self.operation is not None and self.dataset_artifact_id is None:
            raise ValueError("Named operations require an existing dataset_artifact_id")
        if self.operation is None and (self.dataset_artifact_id or self.ground_truth_artifact_id):
            raise ValueError("Artifact IDs require an explicit named operation")
        return self


class AnalystReview(Record):
    stage: Literal["before_acquisition", "after_analysis"]
    status: Literal["advised", "reviewed", "needs_information", "more_analysis"]
    proposal_ids: list[UUID] = Field(default_factory=list)
    analysis_ids: list[UUID] = Field(default_factory=list)
    evidence_ids: list[UUID] = Field(default_factory=list)
    artifact_paths: list[Text] = Field(default_factory=list)
    explanation: Text
    recommendations: list[Text] = Field(default_factory=list)
    limitations: list[Text] = Field(default_factory=list)
    questions: list[Text] = Field(default_factory=list)
    analysis_request: AnalysisRequest | None = None
    cited_source_ids: list[Text] = Field(default_factory=list)
    # Derived by code from source records, never trusted from the model.
    provenance: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def coherent_review(self) -> AnalystReview:
        if self.status == "advised" and self.stage != "before_acquisition":
            raise ValueError("Advice requires before_acquisition stage")
        if self.status in ("reviewed", "more_analysis") and self.stage != "after_analysis":
            raise ValueError("Result review/request requires after_analysis stage")
        if self.status in ("advised", "reviewed") and not self.recommendations:
            raise ValueError("Successful advice/review requires actionable recommendations")
        if self.status == "needs_information" and not self.questions:
            raise ValueError("Missing information requires explicit questions")
        if (self.status == "more_analysis") != (self.analysis_request is not None):
            raise ValueError("more_analysis requires exactly one structured analysis request")
        return self


class AcquisitionReview(Record):
    """Exact executable procedure and trusted, execution-relevant review context."""
    procedure: ExperimentProcedure
    procedure_fingerprint: Text
    review_fingerprint: Text
    status: Literal["ready", "blocked"]
    issues: list[Text] = Field(default_factory=list)
    capabilities_snapshot: dict[str, JsonValue]
    configuration_snapshot: dict[str, JsonValue]


class WorkflowControl(BaseModel):
    """Persisted routing metadata; agent records remain the source of evidence."""
    model_config = ConfigDict(extra="forbid")
    next_node: Literal["scientist", "analyst", "experimentor", "preflight", "human", "acquire", "analyze", "end"] = "scientist"
    task: Literal["proposal", "procedure", "technical_review", "analysis", "interpretation"] = "proposal"
    steps: int = Field(default=0, ge=0)
    procedure_revisions: int = Field(default=0, ge=0)
    analysis_attempts: int = Field(default=0, ge=0)
    acquisition_attempts: int = Field(default=0, ge=0)
    active_proposal_ids: list[UUID] = Field(default_factory=list)
    active_procedure_id: UUID | None = None
    active_review_id: UUID | None = None
    active_approval_id: UUID | None = None
    acquisition_attempt_id: UUID | None = None
    operation_id: UUID | None = None
    pending: dict[str, JsonValue] | None = None
    return_node: Literal["scientist", "analyst", "experimentor", "preflight", "analyze"] = "scientist"
    analyst_stage: Literal["before_acquisition", "after_analysis"] = "before_acquisition"
    reason: str | None = None
    failures: list[str] = Field(default_factory=list)
    allow_partial_analysis: bool = False
    fixture_mode: bool = False
    fixture_analyst: bool = False
    configuration: dict[str, JsonValue] = Field(default_factory=dict)
    architecture_path: str | None = None
    literature_context: list[LiteraturePassage] = Field(default_factory=list)
    hardware_context: dict[str, JsonValue] | None = None
    numerical_request: AnalysisRequest | None = None


class WorkflowState(Record):
    user_request: Text
    status: Literal["pending", "proposing", "planning", "awaiting_approval", "acquiring",
                    "analyzing", "interpreting", "completed", "failed", "blocked",
                    "awaiting_information", "cancelled", "stopped"] = "pending"
    control: WorkflowControl = Field(default_factory=WorkflowControl)
    proposals: list[ExperimentProposal] = Field(default_factory=list)
    procedures: list[ExperimentProcedure] = Field(default_factory=list)
    acquisitions: list[AcquisitionResult] = Field(default_factory=list)
    acquisition_reviews: list[AcquisitionReview] = Field(default_factory=list)
    analyses: list[NumericalAnalysisResult] = Field(default_factory=list)
    analysis_code: list[AnalysisCode] = Field(default_factory=list)
    findings: list[ScientificFinding] = Field(default_factory=list)
    analyst_reviews: list[AnalystReview] = Field(default_factory=list)
    approvals: list[HumanApproval] = Field(default_factory=list)
    conversation: list[ConversationEntry] = Field(default_factory=list)
