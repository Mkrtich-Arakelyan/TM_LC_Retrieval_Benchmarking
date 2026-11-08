"""Explicit routing rules and readable explanations; no model or side effects."""

from state import ConversationEntry, WorkflowState

TERMINAL = {"completed", "cancelled", "stopped", "failed"}


def explain(state: WorkflowState, message: str) -> None:
    state.conversation.append(ConversationEntry(role="orchestrator", message=message))


def route(state: WorkflowState, node: str, task: str, reason: str) -> None:
    state.control.next_node = node
    state.control.task = task
    explain(state, reason)


def block(state: WorkflowState, reason: str) -> None:
    state.status = "blocked"
    state.control.reason = reason
    state.control.failures.append(reason)
    explain(state, "Blocked: " + reason)


def stop(state: WorkflowState, reason: str, status: str = "stopped") -> None:
    state.status = status
    state.control.next_node = "end"
    state.control.reason = reason
    explain(state, reason)


def ask(state: WorkflowState, questions: list[str], return_node: str,
        *, kind: str = "clarification") -> None:
    state.status = "awaiting_information"
    state.control.return_node = return_node
    state.control.pending = {"kind": kind, "questions": questions}
    state.control.next_node = "human"
    explain(state, "Waiting for human input: " + " ".join(questions))


def scientist_route(state: WorkflowState, outcome) -> None:
    if outcome.kind == "proposal":
        state.control.active_proposal_ids = [p.id for p in outcome.proposals]
        route(state, "experimentor", "procedure", "Scientist proposal → Experimentor procedure preparation; a new acquisition always requires review.")
    elif outcome.kind == "request_information":
        ask(state, outcome.questions, "scientist")
    elif outcome.kind == "analysis_request" or (outcome.kind == "interpretation" and outcome.follow_up == "analysis"):
        available = [a for a in state.acquisitions if a.raw_data and
                     (a.status == "completed" or (state.control.allow_partial_analysis and a.status in ("partial", "failed", "aborted")))]
        if available:
            route(state, "analyze", "analysis", "Scientist requests analysis of existing saved data; no new hardware acquisition is needed.")
        else:
            ask(state, ["Which saved data should be analyzed? Partial data requires explicit authorization."], "scientist")
    else:
        stop(state, "Scientist concluded: " + outcome.explanation, "completed")


def experimentor_route(state: WorkflowState, outcome, *, analysis: bool = False) -> None:
    if outcome.scientific_decisions:
        route(state, "scientist", "proposal", "Experimentor requests a scientific decision: " + " ".join(outcome.scientific_decisions))
    elif outcome.kind == "request_information":
        ask(state, outcome.questions, "analyze" if analysis else "experimentor")
    elif outcome.kind == "procedure":
        state.control.active_procedure_id = outcome.procedure.id
        state.control.active_approval_id = None
        route(state, "preflight", "procedure", "Experimentor procedure → Runner preflight; previous approvals cannot authorize a revision.")
    elif analysis:
        latest = state.analyses[-1] if state.analyses else None
        if latest is not None and latest.status == "completed":
            route(state, "scientist", "interpretation", "Recorded numerical analysis → Scientist interpretation; numerical output is distinct from scientific inference.")
        elif latest is not None and latest.execution and latest.execution.status == "unavailable":
            block(state, "Numerical execution is unavailable. Generated code is saved; configure a verified execution backend (or explicit fixture mode), then resume.")
        else:
            route(state, "experimentor", "analysis", "Analysis failed; Experimentor must review diagnostics and repair code using saved data. This technical review cannot switch to acquisition.")
    else:
        route(state, "analyze", "analysis", "Experimentor requests numerical work on saved data; acquisition is not implied.")
