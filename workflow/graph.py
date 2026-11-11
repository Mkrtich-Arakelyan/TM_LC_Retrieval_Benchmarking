"""Local, sequential LangGraph workflow with SQLite checkpoints and human interrupts."""

from __future__ import annotations

import importlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import TypedDict
from uuid import UUID, uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.common import ChatModel, DEFAULT_ARCHITECTURE_PATH
from agents.experimentor import experimentor
from agents.orchestrator import (TERMINAL, ask, block, experimentor_route, explain,
                                 route, scientist_route, stop)
from agents.runner import runner
from agents.scientist import scientist
from agents.analyst import analyst
from workflow.config import AppConfig
from workflow.state import (AcquisitionResult, ConversationEntry, HumanApproval,
                   LiteraturePassage, WorkflowState)
from tools.hardware import DisabledHardwareBackend, HardwareBackend
from tools.python_execution import ExecutionBackend
from workflow.persistence import canonical_factory_path, save_analysis_progress, workflow_writer_lock


@dataclass
class Integrations:
    scientist_model: ChatModel | None = None
    experimentor_model: ChatModel | None = None
    hardware_backend: HardwareBackend | None = None
    execution_backend: ExecutionBackend | None = None
    cancellation_event: Event | None = None
    analyst_model: ChatModel | None = None
    router_model: ChatModel | None = None


class GraphState(TypedDict):
    # Last-value replacement, deliberately no concatenating history reducer.
    workflow: dict


def _state(data: GraphState) -> WorkflowState:
    return WorkflowState.model_validate(data["workflow"])


def _update(state: WorkflowState) -> GraphState:
    return {"workflow": state.model_dump(mode="json")}


def _input_error(pending: dict, response: object) -> str | None:
    if not isinstance(response, dict):
        return "Supply an explicit structured decision; ambiguous input leaves the workflow paused."
    action = response.get("action")
    if action == "cancel":
        return None
    for field in ("reviewer", "feedback"):
        value = response.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            return f"{field} must be a nonempty string when supplied."
    if pending["kind"] == "review":
        if action == "approve" and response.get("review_fingerprint") == pending["review"]["review_fingerprint"]:
            return None
        if action == "reject" and isinstance(response.get("feedback"), str) and response["feedback"].strip():
            return None
        return "Approve the displayed review fingerprint, reject with feedback, or cancel."
    if pending["kind"] == "reconciliation":
        if action == "reconcile" and response.get("confirmed_stopped") is True and isinstance(response.get("message"), str) and response["message"].strip():
            return None
        return "Reconciliation requires an operator message and confirmed_stopped=true; no automatic replay is permitted."
    if action == "clarify" and isinstance(response.get("message"), str) and response["message"].strip():
        if "allow_partial_analysis" in response and type(response["allow_partial_analysis"]) is not bool:
            return "allow_partial_analysis must be a boolean."
        return None
    return "Supply a nonempty clarification response or cancel."


class WorkflowApp:
    """Use as a context manager. Inject integrations; none are instantiated by default.

    Checkpoints hold JSON-shaped copies of shared WorkflowState, not live clients.
    One OS-released writer lock serializes mutations of this local database.
    """
    def __init__(self, config: AppConfig | None = None, integrations: Integrations | None = None,
                 *, integration_factory: str | None = None):
        self.config = config or AppConfig()
        self.integrations = integrations or Integrations()
        # Explicit injected integrations/factories retain priority. Built-in clients
        # are constructed lazily, after the fixture branch, and never checkpointed.
        configured_llm = any((self.config.llm_provider, self.config.llm_model, self.config.api_key))
        self.factory = canonical_factory_path(integration_factory) or (
            "workflow.llm:build_integrations" if integrations is None and configured_llm else None)
        self._resolved = {}
        self.path = self.config.workflow_database_path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self.saver = SqliteSaver(self.connection)
        builder = StateGraph(GraphState)
        builder.add_node("ingress", lambda data: data)
        builder.add_node("dispatch", self._dispatch)
        for name, node in {"scientist": self._scientist, "analyst": self._analyst, "experimentor": self._experimentor,
                           "router": self._router,
                           "preflight": self._preflight, "human": self._human,
                           "acquire": self._acquire, "analyze": self._analyze}.items():
            builder.add_node(name, node)
            builder.add_edge(name, "dispatch")
        builder.add_edge(START, "ingress")
        builder.add_edge("ingress", "dispatch")
        builder.add_conditional_edges("dispatch", self._destination,
            {name: name for name in ("router", "scientist", "analyst", "experimentor", "preflight", "human", "acquire", "analyze")} | {"end": END})
        self.graph = builder.compile(checkpointer=self.saver)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.connection.close()

    def _writer(self):
        return workflow_writer_lock(self.path)

    def _thread(self, workflow_id) -> dict:
        return {"configurable": {"thread_id": str(UUID(str(workflow_id)))},
                "recursion_limit": 10000}

    def _cfg(self, state: WorkflowState) -> AppConfig:
        return AppConfig.model_validate(state.control.configuration)

    def _services(self, state: WorkflowState) -> Integrations:
        if state.control.fixture_mode:
            if state.id not in self._resolved:
                from fixtures.offline import offline_integrations
                self._resolved[state.id] = Integrations(**offline_integrations(self._cfg(state), state.id,
                    include_analyst=state.control.fixture_analyst))
            return self._resolved[state.id]
        if self.factory:
            # Factories consume saved per-workflow configuration. Share an instance
            # within that workflow, never with a different workflow's backend.
            key = (state.id, self.factory)
            if key not in self._resolved:
                module, name = self.factory.split(":", 1)
                runtime_config = self._cfg(state)
                runtime_config.api_key = self.config.api_key  # memory only, never checkpoint secrets
                value = getattr(importlib.import_module(module), name)(runtime_config)
                if not isinstance(value, Integrations):
                    raise TypeError("Integration factory must return workflow.graph.Integrations")
                self._resolved[key] = value
            return self._resolved[key]
        return self.integrations

    def show(self, workflow_id) -> dict:
        snapshot = self.graph.get_state(self._thread(workflow_id))
        if not snapshot.values:
            raise ValueError(f"Unknown workflow: {workflow_id}")
        state = _state(snapshot.values)
        pending = [item.value for task in snapshot.tasks for item in task.interrupts]
        return {"workflow_id": str(state.id), "status": state.status,
                "reason": state.control.reason, "pending": pending[0] if pending else state.control.pending,
                "state": state.model_dump(mode="json")}

    def _drive(self, workflow_id, value=None) -> dict:
        # Sync durability commits reservations before advancing to side effects.
        self.graph.invoke(value, self._thread(workflow_id), durability="sync")
        return self.show(workflow_id)

    def start(self, question: str, *, fixture: bool = False, fixture_analyst: bool = False,
              initial_state: WorkflowState | None = None,
              architecture_path: str | Path = DEFAULT_ARCHITECTURE_PATH,
              literature_context: list[LiteraturePassage] | None = None) -> dict:
        with self._writer():
            if fixture_analyst and not fixture:
                raise ValueError("Fixture Analyst requires explicit fixture mode")
            state = WorkflowState.model_validate(initial_state.model_dump()) if initial_state else WorkflowState(user_request=question)
            if initial_state and state.user_request != question:
                raise ValueError("Initial state's original request must match question")
            if self.graph.get_state(self._thread(state.id)).values:
                raise ValueError("Workflow already exists; resume it instead")
            cfg = self.config.model_copy(deep=True)
            for field in ("workflow_database_path", "papers_path", "index_path", "hardware_input_path", "numerical_input_path"):
                setattr(cfg, field, getattr(cfg, field).resolve())
            cfg.run_output_path = (cfg.run_output_path / str(state.id)).resolve()
            if fixture:
                cfg.llm_router_enabled = False
                cfg.hardware_backend = "test"
                cfg.numerical_backend = "test"
                cfg.hardware_input_path = cfg.run_output_path / "fixture-input"
                cfg.numerical_input_path = cfg.run_output_path / "acquisitions"
            state.control.configuration = cfg.model_dump(mode="json")
            state.control.fixture_mode = fixture
            state.control.fixture_analyst = fixture_analyst
            state.control.architecture_path = str(Path(architecture_path).resolve())
            state.control.literature_context = literature_context or []
            if cfg.llm_router_enabled:
                self._queue_router(state, "start", {"kind": "research_request"})
            explain(state, "TEST ONLY offline fixture workflow; no physical measurements or scientific reasoning are validated." if fixture else "Start workflow with explicit integrations; no fake responses are substituted.")
            return self._drive(state.id, _update(state))

    def submit(self, workflow_id, response: object) -> dict:
        with self._writer():
            current = self.show(workflow_id)
            pending = current["pending"]
            if current["status"] in TERMINAL or not pending:
                return {**current, "input_error": "Workflow is not awaiting human input."}
            if error := _input_error(pending, response):
                return {**current, "input_error": error}
            return self._drive(workflow_id, Command(resume=response))

    def resume(self, workflow_id, *, configuration: AppConfig | None = None) -> dict:
        with self._writer():
            snapshot = self.graph.get_state(self._thread(workflow_id))
            current = self.show(workflow_id)
            state = WorkflowState.model_validate(current["state"])
            if state.status in TERMINAL or current["pending"]:
                return current
            if state.status == "blocked":
                if configuration is not None:
                    cfg = configuration.model_copy(deep=True)
                    cfg.run_output_path = (cfg.run_output_path / str(state.id)).resolve()
                    for field in ("workflow_database_path", "hardware_input_path", "numerical_input_path", "papers_path", "index_path"):
                        setattr(cfg, field, getattr(cfg, field).resolve())
                    state.control.configuration = cfg.model_dump(mode="json")
                    explain(state, "Trusted caller refreshed configuration; counters are retained and changed hardware context requires renewed review.")
                state.status = "pending"
                state.control.reason = None
                self.graph.update_state(self._thread(workflow_id), _update(state), as_node="ingress")
            elif not snapshot.next:
                return current
            return self._drive(workflow_id)

    def cancel(self, workflow_id, reason: str = "Cancelled by user") -> dict:
        with self._writer():
            current = self.show(workflow_id)
            state = WorkflowState.model_validate(current["state"])
            if state.status in TERMINAL:
                return current
            if current["pending"]:
                return self._drive(workflow_id, Command(resume={"action": "cancel", "feedback": reason}))
            stop(state, reason, "cancelled")
            state.control.pending = None
            self.graph.update_state(self._thread(workflow_id), _update(state), as_node="ingress")
            return self._drive(workflow_id)

    def follow_up(self, workflow_id, request: str, *, allow_partial_analysis: bool = False) -> dict:
        with self._writer():
            state = WorkflowState.model_validate(self.show(workflow_id)["state"])
            if state.status != "completed" or not request.strip():
                raise ValueError("Follow-up requires a completed workflow and a nonempty request")
            state.conversation.append(ConversationEntry(role="user", message=request))
            state.control.allow_partial_analysis = allow_partial_analysis
            state.control.active_proposal_ids = []
            state.control.pending = None
            state.control.reason = None
            state.status = "pending"
            if self._router_enabled(state):
                self._queue_router(state, "user", {"kind": "follow_up", "message": request})
            else:
                route(state, "scientist", "interpretation", "User follow-up: use existing saved data where appropriate; do not automatically acquire again.")
            self.graph.update_state(self._thread(workflow_id), _update(state), as_node="ingress")
            return self._drive(workflow_id)

    def _destination(self, data):
        state = _state(data)
        return "end" if state.status in TERMINAL | {"blocked"} else state.control.next_node

    def _router_enabled(self, state):
        return self._cfg(state).llm_router_enabled and not state.control.fixture_mode

    def _queue_router(self, state, source, outcome):
        state.control.router_event = {"source": source, "outcome": outcome}
        route(state, "router", state.control.task, f"{source} outcome saved → scientific routing LLM")

    def _router(self, data):
        state = _state(data)
        try:
            if not self._router_enabled(state):
                raise ValueError("LLM routing is disabled for this run")
            from agents.orchestrator import choose_route
            decision, action = choose_route(state, self._cfg(state), self._services(state))
            if action is None or decision.status != "accepted":
                block(state, "Routing LLM decision rejected: " + (decision.error or decision.reason) + ". Saved records are preserved; correct the response/integration and resume.")
                return _update(state)
            if action.kind in ("prepare_procedure", "revise_procedure", "preflight"):
                state.control.active_proposal_ids = [action.proposal_id, *[
                    key for key in state.control.active_proposal_ids if key != action.proposal_id]]
                state.control.active_approval_id = None
                state.control.active_procedure_id = action.procedure_id
            if action.kind == "propose":
                route(state, "scientist", "proposal", decision.reason)
            elif action.kind == "advise":
                state.control.analyst_stage = "before_acquisition"
                route(state, "analyst", "procedure", decision.reason)
            elif action.kind in ("prepare_procedure", "revise_procedure"):
                route(state, "experimentor", "procedure" if action.kind == "prepare_procedure" else "technical_review", decision.reason)
            elif action.kind == "preflight":
                route(state, "preflight", "procedure", decision.reason)
            elif action.kind == "run_operation":
                from workflow.state import AnalysisRequest
                state.control.numerical_request = AnalysisRequest.model_validate(action.request.model_dump())
                route(state, "analyze", "analysis", decision.reason)
            elif action.kind == "review_results":
                state.control.analyst_stage = "after_analysis"
                state.control.router_analysis_ids = action.analysis_ids
                route(state, "analyst", "interpretation", decision.reason)
            elif action.kind == "interpret":
                route(state, "scientist", "interpretation", decision.reason)
            elif action.kind == "request_information":
                ask(state, [decision.reason], "router")
            elif action.kind == "conclude":
                stop(state, "Scientist conclusion accepted by routing LLM: " + decision.reason, "completed")
            else:
                raise ValueError("No Python dispatcher exists for the selected router action")
        except Exception as exc:
            block(state, f"Routing LLM failed: {exc}. No selected action was executed; correct integration and resume.")
        return _update(state)

    def _dispatch(self, data):
        state = _state(data)
        c = state.control
        cfg = self._cfg(state)
        if state.status in TERMINAL | {"blocked"} or c.next_node == "human":
            return _update(state)
        limit = None
        revising = (c.next_node == "experimentor" and c.task == "procedure"
                    and bool(c.active_proposal_ids) and any(
                        p.proposal_id == c.active_proposal_ids[0] for p in state.procedures))
        if c.steps >= cfg.max_workflow_iterations:
            limit = "Workflow step limit reached; progress and counters are preserved."
        elif c.next_node == "analyze" and c.analysis_attempts >= cfg.max_workflow_analysis_attempts:
            limit = "Numerical analysis attempt limit reached; no further execution was requested."
        elif c.next_node == "acquire" and c.acquisition_attempts >= cfg.max_workflow_acquisition_attempts:
            limit = "Acquisition attempt limit reached; no further hardware call was made."
        elif revising and c.procedure_revisions >= cfg.max_procedure_revisions:
            limit = "Procedure revision limit reached; review the preserved revisions."
        if limit:
            stop(state, limit)
            return _update(state)
        c.steps += 1
        c.operation_id = uuid4()
        if c.next_node == "analyze":
            c.analysis_attempts += 1
        if c.next_node == "acquire":
            c.acquisition_attempts += 1
            c.acquisition_attempt_id = uuid4()
        if revising:
            c.procedure_revisions += 1
        state.status = {"router": "planning", "scientist": "interpreting" if c.task == "interpretation" else "proposing",
                        "analyst": "planning" if c.analyst_stage == "before_acquisition" else "interpreting",
                        "experimentor": "planning", "preflight": "planning",
                        "acquire": "acquiring", "analyze": "analyzing"}[c.next_node]
        explain(state, f"Workflow step {c.steps}: {c.next_node}, task={c.task}.")
        return _update(state)

    def _scientist(self, data):
        state = _state(data)
        try:
            services = self._services(state)
            if services.scientist_model is None:
                block(state, "Scientist model is unconfigured. Inject workflow.graph.Integrations(scientist_model=...) or explicitly select offline fixture mode.")
                return _update(state)
            result = scientist(state, services.scientist_model, state.control.literature_context,
                               architecture_path=state.control.architecture_path)
            state = result.state
            if self._router_enabled(state):
                if result.outcome.kind == "proposal":
                    state.control.active_proposal_ids = [p.id for p in result.outcome.proposals]
                self._queue_router(state, "scientist", result.outcome.model_dump(mode="json"))
                return _update(state)
            if result.outcome.kind == "interpretation" and result.outcome.follow_up == "conclude" and len(state.control.active_proposal_ids) > 1:
                state.control.active_proposal_ids.pop(0)
                route(state, "experimentor", "procedure", "Continue with the next Scientist proposal; each requires a separate procedure and review.")
            else:
                scientist_route(state, result.outcome)
            if state.control.next_node == "experimentor" and state.control.task == "procedure":
                self._analyst_handoff(state, "before_acquisition")
        except Exception as exc:
            block(state, f"Scientist failed: {exc}. Correct the integration or supplied context, then resume.")
        return _update(state)

    def _analyst_handoff(self, state, stage):
        if self._services(state).analyst_model is None:
            explain(state, f"Analyst {stage} skipped: no Analyst model configured; no successful review recorded.")
        else:
            state.control.analyst_stage = stage
            state.control.next_node = "analyst"
            explain(state, f"Route to Analyst {stage}; review does not authorize acquisition or scientific conclusions.")

    def _analyst(self, data):
        state = _state(data)
        stage = state.control.analyst_stage
        target = "experimentor" if stage == "before_acquisition" else "scientist"
        task = "procedure" if stage == "before_acquisition" else "interpretation"
        try:
            model = self._services(state).analyst_model
            if model is None:
                explain(state, f"Analyst {stage} skipped: model unavailable; no review recorded.")
                if self._router_enabled(state):
                    self._queue_router(state, "analyst", {"kind": "unavailable", "explanation": "No Analyst model is configured"})
                else:
                    route(state, target, task, "Continue through optional Analyst handoff without a successful review.")
                return _update(state)
            analysis_ids = None
            if self._router_enabled(state) and stage == "after_analysis":
                analysis_ids = state.control.router_analysis_ids
            result = analyst(state, model, stage=stage, analysis_ids=analysis_ids,
                             literature_context=state.control.literature_context,
                             architecture_path=state.control.architecture_path)
            state = result.state
            if self._router_enabled(state):
                self._queue_router(state, "analyst", result.outcome.model_dump(mode="json"))
                return _update(state)
            if result.outcome.status == "needs_information":
                ask(state, result.outcome.questions, "analyst")
            elif result.outcome.status == "more_analysis":
                request = result.outcome.analysis_request
                if request.operation:
                    state.control.numerical_request = request.model_copy(deep=True)
                route(state, "analyze" if request.operation else "experimentor", "analysis", "Analyst requests calculation on saved inputs: "
                      + result.outcome.analysis_request.model_dump_json() + "; no additional measurements scheduled.")
            else:
                route(state, target, task, "Analyst recommendations are available in analyst_reviews; "
                      + ("Experimentor owns acquisition choices." if stage == "before_acquisition" else "Scientist owns the conclusion."))
        except Exception as exc:
            block(state, f"Analyst {stage} failed: {exc}. No successful review recorded; correct integration/response and resume.")
        return _update(state)

    def _experimentor(self, data):
        state = _state(data)
        try:
            if self._router_enabled(state) and state.control.task == "analysis":
                self._queue_router(state, "experimentor", {"kind": "analysis_requested", "explanation": "Select a reviewed saved-data operation; generated-code execution is not a router action"})
                return _update(state)
            services = self._services(state)
            if services.experimentor_model is None:
                block(state, "Experimentor model is unconfigured; inject an externally configured chat model, then resume.")
                return _update(state)
            if state.control.task in ("procedure", "technical_review"):
                hardware = services.hardware_backend or DisabledHardwareBackend()
                state.control.hardware_context = hardware.describe().model_dump(mode="json")
                if state.control.task == "procedure" and (
                    self._cfg(state).hardware_backend == "disabled" or not state.control.hardware_context["available"]
                ):
                    block(state, "Hardware backend is disabled or unavailable. Configure a verified backend before procedure preparation; standalone code/procedure generation remains available separately.")
                    return _update(state)
            result = experimentor(state, services.experimentor_model, state.control.literature_context,
                config=self._cfg(state), architecture_path=state.control.architecture_path,
                allow_partial_data=state.control.allow_partial_analysis)
            state = result.state
            if self._router_enabled(state):
                if result.outcome.kind == "procedure":
                    if state.control.task == "technical_review" and result.outcome.procedure.version > 1:
                        state.control.procedure_revisions += 1
                    state.control.active_procedure_id = result.outcome.procedure.id
                    state.control.active_approval_id = None
                self._queue_router(state, "experimentor", result.outcome.model_dump(mode="json"))
                return _update(state)
            if result.outcome.kind == "procedure" and state.control.task == "technical_review" and result.outcome.procedure.version > 1:
                state.control.procedure_revisions += 1
            experimentor_route(state, result.outcome)
        except Exception as exc:
            if "Procedure revision limit" in str(exc):
                stop(state, str(exc))
            elif "requires an explicit user request" in str(exc):
                ask(state, ["Explicitly authorize analysis of partial saved data, or request a technical/scientific revision."], "analyze")
            else:
                block(state, f"Experimentor failed: {exc}. Correct the technical issue, then resume.")
        return _update(state)

    def _prepare_review(self, state, review):
        purpose = next((p.model_dump(mode="json") for p in state.proposals if p.id == review.procedure.proposal_id), "Unknown proposal")
        state.status = "awaiting_approval"
        state.control.active_review_id = review.id
        state.control.active_approval_id = None
        state.control.pending = {"kind": "review", "scientific_purpose": purpose,
                                 "review": review.model_dump(mode="json"),
                                 "decisions": ["approve", "reject with feedback", "cancel"]}
        state.control.next_node = "human"
        explain(state, "Runner preflight succeeded; waiting for explicit human approval of the exact procedure and execution context.")

    def _preflight_issues(self, state, result):
        issues = " ".join(result.technical_issues)
        if any(term in issues for term in ("disabled", "unavailable", "Backend preflight failed", "bounded execution")):
            block(state, "Hardware integration is blocked: " + issues + " Configure a verified backend; architecture specifications do not supply safe limits.")
        elif any(term in issues for term in ("Unresolved prerequisite", "verified calibration", "Unknown required limits", "Unverified equipment", "Invalid input/calibration")):
            ask(state, result.technical_issues, "preflight")
        else:
            route(state, "experimentor", "technical_review", "Runner technical preflight failed: " + issues)

    def _preflight(self, data):
        state = _state(data)
        try:
            result = runner(state, state.control.active_procedure_id, config=self._cfg(state),
                            backend=self._services(state).hardware_backend)
            state = result.state
            if result.status == "review":
                self._prepare_review(state, result.review)
            else:
                self._preflight_issues(state, result)
        except Exception as exc:
            block(state, f"Runner preflight failed: {exc}")
        return _update(state)

    def _human(self, data):
        state = _state(data)
        pending = state.control.pending
        # No writes, UUID generation, model calls, or hardware before interrupt.
        while True:
            response = interrupt(pending)
            error = _input_error(pending, response)
            if error is None:
                break
            pending = {**pending, "input_error": error}
        action = response["action"]
        if action == "cancel":
            message = str(response.get("feedback") or "Cancelled by user")
            if pending["kind"] == "reconciliation":
                message += "; acquisition remains uncertain: cancelling the workflow does not confirm physical equipment stopped."
            stop(state, message, "cancelled")
        elif pending["kind"] == "review":
            review = next(r for r in state.acquisition_reviews if r.id == state.control.active_review_id)
            approval = HumanApproval(procedure_id=review.procedure.id,
                procedure_version=review.procedure.version, procedure_fingerprint=review.procedure_fingerprint,
                review_fingerprint=review.review_fingerprint,
                status="approved" if action == "approve" else "rejected",
                reviewer=response.get("reviewer"), comment=response.get("feedback"))
            state.approvals.append(approval)
            state.control.active_approval_id = approval.id if action == "approve" else None
            state.conversation.append(ConversationEntry(role="human", message=f"{action}: {response.get('feedback') or 'exact displayed review'}", record_ids=[approval.id]))
            if action == "reject" and self._router_enabled(state):
                self._queue_router(state, "human", {"kind": "procedure_rejected", "explanation": response.get("feedback") or "Procedure rejected"})
            else:
                route(state, "acquire" if action == "approve" else "experimentor", "procedure",
                      "Human approved exact review; Runner must revalidate before acquisition." if action == "approve" else "Human rejected procedure with feedback; Experimentor must revise and obtain new approval.")
        elif action == "reconcile":
            state.conversation.append(ConversationEntry(role="human", message="Operator reconciliation (not a successful-acquisition claim): " + response["message"]))
            state.control.acquisition_attempt_id = None
            state.control.active_approval_id = None
            if self._router_enabled(state):
                self._queue_router(state, "human", {"kind": "acquisition_reconciled", "explanation": "Operator reports stopped; incomplete evidence requires technical review or a new reviewed procedure, never replay"})
            else:
                route(state, "experimentor", "technical_review", "Operator reports equipment stopped; retain uncertain records and request technical review. Never replay the interrupted attempt.")
        else:
            state.conversation.append(ConversationEntry(role="user", message=response["message"]))
            state.control.allow_partial_analysis = response.get("allow_partial_analysis", state.control.allow_partial_analysis)
            if self._router_enabled(state):
                self._queue_router(state, "human", {"kind": "clarification", "message": response["message"]})
            else:
                route(state, state.control.return_node, state.control.task, "Human clarification received; resume the requesting task without changing verified limits or approvals.")
        state.control.pending = None
        if state.status not in TERMINAL:
            state.status = "pending"
        return _update(state)

    def _acquire(self, data):
        state = _state(data)
        cfg = self._cfg(state)
        attempt = state.control.acquisition_attempt_id
        directory = cfg.run_output_path / "acquisitions" / str(attempt)
        try:
            if directory.exists():
                path = directory / "record.json"
                record = AcquisitionResult.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None
                if record is not None and record.id != attempt:
                    raise ValueError("Acquisition journal identity mismatch")
                reviewed = next(r for r in state.acquisition_reviews if r.id == state.control.active_review_id)
                if record is not None and (
                    record.procedure_id != reviewed.procedure.id
                    or record.procedure_version != reviewed.procedure.version
                    or record.procedure_fingerprint != reviewed.procedure_fingerprint
                    or record.hardware is None
                    or record.hardware.approval_id != state.control.active_approval_id
                    or record.hardware.review_fingerprint != reviewed.review_fingerprint
                ):
                    raise ValueError("Acquisition journal does not match checkpointed procedure/approval")
                if record is not None and not any(r.id == record.id for r in state.acquisitions):
                    state.acquisitions.append(record)
                if record is not None and record.status == "completed":
                    state.control.configuration["numerical_input_path"] = str(cfg.run_output_path / "acquisitions")
                    explain(state, "Recovered completed Runner journal after interruption; hardware was not called again.")
                    self._saved_acquisition_analysis(state, record)
                    return _update(state)
                ask(state, ["Acquisition outcome is uncertain. Inspect the durable journal and raw files, confirm equipment stopped, then reconcile for technical review or cancel. No automatic retry."], "experimentor", kind="reconciliation")
                return _update(state)
            reviewed = next(r for r in state.acquisition_reviews if r.id == state.control.active_review_id)
            approval = next(a for a in state.approvals if a.id == state.control.active_approval_id)
            services = self._services(state)
            result = runner(state, state.control.active_procedure_id, config=cfg,
                backend=services.hardware_backend, execute=True, reviewed=reviewed, approval=approval,
                attempt_id=attempt, cancel_event=services.cancellation_event)
            state = result.state
            if result.acquisition is not None and result.acquisition.raw_data:
                state.control.configuration["numerical_input_path"] = str(cfg.run_output_path / "acquisitions")
            if result.acquisition is None:
                if result.review.status == "ready":
                    self._prepare_review(state, result.review)
                else:
                    self._preflight_issues(state, result)
            elif result.acquisition.status == "completed":
                state.control.configuration["numerical_input_path"] = str(cfg.run_output_path / "acquisitions")
                self._saved_acquisition_analysis(state, result.acquisition)
            else:
                if self._router_enabled(state):
                    self._queue_router(state, "runner", {"kind": "acquisition_failed", "record_id": str(result.acquisition.id),
                        "status": result.acquisition.status, "explanation": "; ".join(result.acquisition.errors)})
                else:
                    route(state, "experimentor", "technical_review", "Acquisition failed/partial/aborted. Review completed and uncertain actions; never automatically rerun hardware or analyze partial data without an explicit request.")
        except Exception as exc:
            if directory.exists():
                ask(state, [f"Acquisition interrupted or recording failed: {exc}. Reconcile hardware and journal before further work."], "experimentor", kind="reconciliation")
            else:
                block(state, f"Acquisition could not start: {exc}")
        return _update(state)

    def _saved_acquisition_analysis(self, state, acquisition):
        if self._router_enabled(state):
            self._queue_router(state, "runner", {"kind": "acquisition_completed", "record_id": str(acquisition.id),
                "data_origin": acquisition.data_origin, "status": acquisition.status})
            return
        route(state, "analyze", "analysis", "Completed acquisition → Experimentor numerical analysis of saved data.")

    def _analyze(self, data):
        state = _state(data)
        cfg = self._cfg(state)
        operation = cfg.run_output_path / "analysis-operations" / str(state.control.operation_id)
        try:
            # Preserve a complete node receipt; uncertain replay never re-executes code.
            if operation.exists():
                receipt = operation / "state.json"
                if receipt.exists():
                    recovered = WorkflowState.model_validate_json(receipt.read_text(encoding="utf-8"))
                    if recovered.id != state.id or recovered.control.operation_id != state.control.operation_id:
                        raise ValueError("Analysis receipt identity mismatch")
                    return _update(recovered)
                progress = operation / "progress.json"
                if progress.exists():
                    state = WorkflowState.model_validate_json(progress.read_text(encoding="utf-8"))
                block(state, "Interrupted numerical operation: inspect saved scripts/output artifacts. No automatic re-execution; explicitly resume a new bounded attempt after review.")
                return _update(state)
            if state.control.numerical_request is not None:
                return self._reviewed_analysis(state, cfg, operation)
            if self._router_enabled(state):
                self._queue_router(state, "workflow", {"kind": "analysis_requested"})
                return _update(state)
            services = self._services(state)
            if services.experimentor_model is None:
                block(state, "Experimentor model is unconfigured; inject it before numerical analysis.")
                return _update(state)
            operation.mkdir(parents=True, exist_ok=False)
            cfg.max_execution_attempts = 1  # graph reserves and checkpoints every attempt
            result = experimentor(state, services.experimentor_model, state.control.literature_context,
                config=cfg, architecture_path=state.control.architecture_path, execute=True,
                backend=services.execution_backend, allow_partial_data=state.control.allow_partial_analysis,
                on_progress=lambda progress: self._save_analysis_progress(operation, progress))
            state = result.state
            experimentor_route(state, result.outcome, analysis=True)
            if state.control.next_node == "scientist" and state.control.task == "interpretation":
                self._analyst_handoff(state, "after_analysis")
            (operation / "state.json").write_text(state.model_dump_json(), encoding="utf-8")
        except Exception as exc:
            progress_path = operation / "progress.json"
            if progress_path.exists():
                state = WorkflowState.model_validate_json(progress_path.read_text(encoding="utf-8"))
            if "requires an explicit user request" in str(exc):
                ask(state, ["Explicitly authorize analysis of partial saved data or request technical revision."], "analyze")
            else:
                block(state, f"Numerical analysis failed: {exc}. Inspect saved artifacts before resuming.")
        return _update(state)

    def _reviewed_analysis(self, state, cfg, operation):
        from tools.tm_numerics import execute_operation
        operation.mkdir(parents=True, exist_ok=False)
        numerical = execute_operation(state, state.control.numerical_request, cfg,
            attempt_number=state.control.analysis_attempts, run_id=state.control.operation_id)
        state.analyses.append(numerical)
        self._save_analysis_progress(operation, state)
        explain(state, "Reviewed numerical operation: " + numerical.model_dump_json())
        if numerical.status == "completed":
            state.control.numerical_request = None
            if self._router_enabled(state):
                self._queue_router(state, "numerical_tool", {"kind": "analysis_completed", "record_id": str(numerical.id),
                    "operation": numerical.method_settings.get("operation"), "status": numerical.status})
            elif state.control.pending_numerical_requests:
                state.control.numerical_request = state.control.pending_numerical_requests.pop(0)
                route(state, "analyze", "analysis", "Saved first condition; compare the paired condition with the same reviewed settings before interpretation.")
            else:
                route(state, "scientist", "interpretation", "Reviewed numerical results → Analyst review → Scientist interpretation.")
                self._analyst_handoff(state, "after_analysis")
        else:
            block(state, "Reviewed numerical operation " + numerical.status + ": "
                  + "; ".join(numerical.errors or [numerical.execution.reason or "Unavailable"]))
        (operation / "state.json").write_text(state.model_dump_json(), encoding="utf-8")
        return _update(state)

    @staticmethod
    def _save_analysis_progress(operation: Path, state: WorkflowState):
        save_analysis_progress(operation, state)
