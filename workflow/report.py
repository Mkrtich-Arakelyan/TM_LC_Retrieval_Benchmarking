"""Export saved workflow records without invoking agents or acquisition backends."""

import json
from pathlib import Path


def export_workflow_view(view: dict) -> Path:
    state = view["state"]
    root = Path(state["control"]["configuration"]["run_output_path"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "workflow.json").write_text(json.dumps(view, indent=2, ensure_ascii=False), encoding="utf-8")
    messages = [f"# Workflow {view['workflow_id']}", f"Status: {view['status']}"]
    for entry in state["conversation"]:
        messages.extend([f"\n## {entry['role']}", entry["message"]])
    (root / "agent-messages.md").write_text("\n\n".join(messages) + "\n", encoding="utf-8")
    decisions = state.get("router_decisions", [])
    (root / "router-decisions.json").write_text(json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Routing decisions"]
    for decision in decisions:
        lines.extend([f"\n## {decision['status']}: {decision.get('action') or 'unselected'}",
                      decision["reason"]])
        if decision.get("error"):
            lines.append(decision["error"])
    (root / "router-decisions.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return root


def export_decisions(view):
    state = view["state"]
    root = Path(state["control"]["configuration"]["run_output_path"])
    analyses = [a for a in state["analyses"] if a["method_settings"].get("operation") == "bench.retrieve"]
    metrics = analyses[-1]["metrics"] if analyses else {}
    failed = any(a["status"] in ("partial", "failed", "aborted") for a in state["acquisitions"])
    quality = {"workflow_status": view["status"], "workflow_completed": view["status"] == "completed",
        "agent_mode": "local_test_double" if state["control"]["configuration"].get("bench_test_agents") else "live",
        "measurement_quality": "pass" if metrics.get("measurement_passed") else "failed" if failed else "inconclusive",
        "prediction_quality": "pass" if metrics.get("prediction_passed") else "failed_or_inconclusive" if analyses else "not_evaluated",
        "metrics": metrics, "acquisition_errors": [e for a in state["acquisitions"] for e in a["errors"]]}
    (root / "quality.json").write_text(json.dumps(quality, indent=2, allow_nan=False), encoding="utf-8")
    lines = [f"# Synthetic workflow {view['workflow_id']}",
        f"Agent mode: {quality['agent_mode']}", f"Workflow: {view['status']}",
        f"Measurement quality: {quality['measurement_quality']}", f"Prediction quality: {quality['prediction_quality']}"]
    lines.append("\n## Scientist proposed")
    for proposal in state["proposals"]:
        lines.append(f"{proposal['id']}: {proposal['hypothesis']}\n{proposal['rationale']}")
    lines.append("\n## Experimentor selected")
    for procedure in state["procedures"]:
        lines.append(f"Procedure {procedure['id']}, version {procedure['version']}, proposal {procedure['proposal_id']}.")
        for action in procedure["actions"]:
            lines.append(f"{action['action']}: {action['instruction']}\nInitial requested settings: {json.dumps(action['requested_settings'])}")
    lines.append("\n## Runner executed")
    if not state["acquisitions"]:
        lines.append("No acquisition executed; exact human review/approval is still required.")
    for acquisition in state["acquisitions"]:
        lines.append(f"Acquisition {acquisition['id']}: {acquisition['status']}; origin {acquisition['data_origin']}. "
                     f"Actual settings: {json.dumps(acquisition['actual_settings'])}")
    lines.append("\n## Analyst evaluated")
    for review in state["analyst_reviews"]:
        lines.append(f"{review['stage']}: {review['status']}. {review['explanation']}")
    for analysis in analyses:
        lines.append(f"Reviewed operation {analysis['method_settings']['operation']}: {analysis['status']}, evidence {analysis['id']}.")
    lines.append("\n## Scientist concluded")
    for finding in state["findings"]:
        lines.append(f"{finding['interpretation']}\nLimitations: {'; '.join(finding['limitations'])}")
    if not state["findings"]:
        lines.append("No evidence-backed Scientist finding recorded.")
    if quality["acquisition_errors"]:
        lines.append("\nStopping reason: " + "; ".join(quality["acquisition_errors"]))
    (root / "decision-log.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")
