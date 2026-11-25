"""Export saved workflow records without invoking agents or acquisition backends."""

import json
import hashlib
import math
import os
import re
from pathlib import Path
from urllib.parse import quote


def conversation_entries(state):
    """Select recorded dialogue/events, never extract or rewrite agent prose."""
    for entry in state["conversation"]:
        kind, text, role = entry.get("kind", "legacy"), entry["message"], entry["role"]
        if kind == "trace":
            continue
        if role == "orchestrator" and ("outcome saved" in text and "routing LLM" in text or text.startswith("Start workflow with explicit integrations")):
            continue
        if kind == "legacy" and (text.lstrip().startswith(("{", "[")) or any(
            marker in text for marker in ("Workflow step ", "Routing LLM decision:", "Rejected draft", "Rejected Scientist output", "Procedure validation:"))):
            continue
        yield entry


def export_conversation(state, root):
    labels = {"experimentor": "Experimentalist", "scientist": "Scientist", "analyst": "Analyst",
              "orchestrator": "Orchestrator", "runner": "Runner", "user": "User", "human": "Human"}
    chunks = ["# Recorded conversation\n\nVerbatim saved messages in workflow order. Structured records, validation retries and routine updates are in the detailed trace.\n"]
    for entry in conversation_entries(state):
        label = labels.get(entry["role"], entry["role"])
        if entry.get("recipient"):
            label += " → " + labels.get(entry["recipient"], entry["recipient"])
        if entry["role"] == "runner":
            label += " — execution event (program-generated)"
        elif entry["role"] == "orchestrator":
            label += " — orchestration decision"
        elif entry.get("kind", "legacy") == "legacy":
            label += " — legacy recorded message"
        metadata = f"Message ID: {entry.get('id', 'not recorded')}."
        if entry.get("reply_to"):
            metadata += f" Replies to: {entry['reply_to']}."
        chunks.append(f"\n## {label}\n\n{metadata}\n\n" + entry["message"] + "\n")
    if len(chunks) == 1:
        chunks.append("\nNo conversational messages were recorded; see the detailed trace.\n")
    # Bytes avoid newline translation: the saved text is not normalized or stripped.
    (root / "conversation.md").write_bytes("\n".join(chunks).encode("utf-8"))


def export_workflow_view(view: dict, output_dir=None) -> Path:
    state = view["state"]
    root = Path(output_dir or state["control"]["configuration"]["run_output_path"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "workflow.json").write_text(json.dumps(view, indent=2, ensure_ascii=False), encoding="utf-8")
    messages = [f"# Detailed trace: workflow {view['workflow_id']}", f"Status: {view['status']}",
                "All saved messages, including validation/retry and routine updates. Structured records and raw state: [workflow.json](workflow.json)."]
    for entry in state["conversation"]:
        messages.extend([f"\n## {entry['role']}", entry["message"]])
    (root / "agent-messages.md").write_bytes(("\n\n".join(messages) + "\n").encode("utf-8"))
    export_conversation(state, root)
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


def bench_analyses(state):
    return [a for a in state["analyses"] if a["method_settings"].get("operation") == "bench.retrieve" or
            (a["method_settings"].get("operation") == "tm.compare_reference_free" and
             any(Path(ref["path"]).name == "bench-evaluation.json" for ref in a["outputs"]))]


def export_decisions(view, output_dir=None):
    state = view["state"]
    root = Path(output_dir or state["control"]["configuration"]["run_output_path"])
    root.mkdir(parents=True, exist_ok=True)
    analyses = bench_analyses(state)
    metrics = analyses[-1]["metrics"] if analyses else {}
    failed = any(a["status"] in ("partial", "failed", "aborted") for a in state["acquisitions"])
    quality = {"workflow_status": view["status"], "workflow_completed": view["status"] == "completed",
        "agent_mode": "local_test_double" if state["control"]["configuration"].get("bench_test_agents") else "live",
        "measurement_quality": "pass" if metrics.get("measurement_passed") else "failed" if failed else "inconclusive",
        "prediction_quality": "pass" if metrics.get("prediction_passed") else "failed_or_inconclusive" if analyses else "not_evaluated",
        "metrics": metrics, "acquisition_errors": [e for a in state["acquisitions"] for e in a["errors"]]}
    assessment = next((r for r in reversed(state["analyst_reviews"]) if r.get("recovery")), None)
    quality["recovery"] = assessment["recovery"] if assessment else None
    quality["recovery_limitations"] = assessment["limitations"] if assessment else ["Recovery not assessed in this saved run"]
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
        if review.get("recovery"):
            lines.append("Recovery assessment: " + json.dumps(review["recovery"], ensure_ascii=False))
            lines.append("Limitations: " + "; ".join(review["limitations"]))
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
    export_run_summary(view, quality, output_dir=root)


def regenerate_report(workflow_file, output_dir):
    """Read a saved view only: no database, backend, model, fitting or routing."""
    from workflow.state import WorkflowState
    view = json.loads(Path(workflow_file).read_text(encoding="utf-8"))
    WorkflowState.model_validate(view["state"])  # Includes legacy optional fields.
    root = Path(output_dir).resolve()
    if root == Path(view["state"]["control"]["configuration"]["run_output_path"]).resolve():
        raise ValueError("Choose a separate report output directory to preserve the original artifacts")
    export_workflow_view(view, root)
    export_decisions(view, root)
    return root


def _artifact(records, name):
    return next((ref for ref in reversed(records) if Path(ref["path"]).name == name), None)


def _recorded_json(ref, issues):
    if ref is None:
        return None
    path = Path(ref["path"])
    try:
        content = path.read_bytes()
        if ref.get("sha256") and hashlib.sha256(content).hexdigest() != ref["sha256"]:
            raise ValueError("recorded SHA-256 does not match")
        return json.loads(content)
    except (OSError, ValueError) as exc:
        issues.append(f"{path.name} unavailable: {exc}")
        return None


def _link(root, path, label):
    path = Path(path)
    if not path.is_file():
        return f"{label} (recorded file unavailable)"
    try:
        target = Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()
    except ValueError:
        target = path.resolve().as_posix()
    return f"[{label}]({quote(target, safe='/.:')})"


def _brief(value):
    text = " ".join(str(value or "Reason not recorded.").split())
    return text[:600] + ("... (see recorded messages)" if len(text) > 600 else "")


def follow_up_support(state, follow):
    caps = state["control"].get("hardware_context")
    if not caps and state.get("acquisition_reviews"):
        caps = state["acquisition_reviews"][-1].get("capabilities_snapshot")
    if not caps:
        return "not assessed — no acquisition capability snapshot was recorded"
    missing = list(follow.get("missing_capabilities", []))
    for name in set(follow["required_actions"]) | set(follow["required_parameters"]):
        action = caps.get("actions", {}).get(name) if caps.get("available") else None
        if action is None:
            missing.append("Unavailable action: " + name)
            continue
        for parameter in follow["required_parameters"].get(name, []):
            if not action.get("parameters", {}).get(parameter, {}).get("verified"):
                missing.append("Missing verified control: " + name + "." + parameter)
    return "unavailable — " + "; ".join(missing) if missing else "supported by recorded capabilities; a new procedure still requires preflight and human approval"


def coefficient_table(parts, detector_ids, title):
    """Round-trip precision for each saved component; no phase alignment of tables."""
    width = len(parts["real"][0])
    lines = ["<details>", f"<summary>{title} — exact complex coefficients</summary>", "",
             "Rows are camera detectors; columns are SLM inputs (1-based). Entries are real ± imaginary i, "
             "shown with 17 significant digits so each binary64 component round-trips. Values are not phase-aligned.", "",
             "| Camera detector / SLM input | " + " | ".join(f"Input {i+1}" for i in range(width)) + " |",
             "| --- | " + " | ".join(["---"]*width) + " |"]
    for detector, real, imag in zip(detector_ids, parts["real"], parts["imag"]):
        values = [f"`{r:.17g} {'-' if math.copysign(1, i) < 0 else '+'} {abs(i):.17g}i`" for r, i in zip(real, imag)]
        lines.append(f"| D{detector} | " + " | ".join(values) + " |")
    return "\n".join([*lines, "", "</details>"])


def export_run_summary(view, quality, output_dir=None):
    """Assemble recorded decisions and numerical exports; never call agents or refit."""
    state = view["state"]
    root = Path(output_dir or state["control"]["configuration"]["run_output_path"])
    root.mkdir(parents=True, exist_ok=True)
    acquisition = state["acquisitions"][-1] if state["acquisitions"] else None
    analyses = bench_analyses(state)
    analysis = analyses[-1] if analyses else None
    raw_refs = acquisition["raw_data"] if acquisition else []
    outputs = analysis["outputs"] if analysis else []
    issues = []
    calibration = _recorded_json(_artifact(raw_refs, "calibration.json"), issues)
    manifest = _recorded_json(_artifact(raw_refs, "bench-dataset.json"), issues)
    evaluation = _recorded_json(_artifact(outputs, "bench-evaluation.json"), issues)
    example = _recorded_json(_artifact(outputs, "heldout-example.json"), issues)
    from analysis.run_inspection import export_estimates
    estimates, matrix_issues = export_estimates(view, root)
    issues.extend(matrix_issues)
    if acquisition and acquisition["status"] != "completed":
        stage = ("Calibration completed; measurement sequence not completed" if calibration and calibration.get("success")
                 else "Calibration diagnostics saved; target not accepted" if calibration else "Acquisition started; no completed measurement stage recorded")
    elif analysis and analysis["status"] == "completed":
        stage = "Retrieval and held-out evaluation completed"
        if any(r["stage"] == "after_analysis" for r in state["analyst_reviews"]):
            stage = "Post-analysis Analyst review recorded"
        if state["findings"]:
            stage = "Scientist interpretation recorded"
    elif acquisition and acquisition["status"] == "completed":
        stage = "Acquisition completed; retrieval not completed"
    elif state["acquisition_reviews"]:
        stage = "Runner preflight/review recorded; no acquisition executed"
    elif state["procedures"]:
        stage = "Experimentor procedure recorded"
    elif state["proposals"]:
        stage = "Scientist proposal recorded"
    else:
        stage = "Workflow initialized; no experimental stage completed"
    lines = [f"# Run summary: {view['workflow_id']}",
             f"Agent mode: **{quality['agent_mode']}**. Results describe a synthetic workflow, not a laboratory validation.",
             f"Workflow: **{view['status']}** (completed: {quality['workflow_completed']}). "
             f"Measurement quality: **{quality['measurement_quality']}**. Prediction quality: **{quality['prediction_quality']}**.",
             f"Held-out brightness prediction: **{quality['prediction_quality']}**. This does not establish matrix recovery.",
             f"Last completed stage: {stage}."]
    if not quality.get("recovery"):
        lines.insert(3, "Matrix recovery: **not assessed** — no structured recovery assessment was recorded in this run.")
    reasons = [view.get("reason"), *(acquisition["errors"] if acquisition else []),
               *(analysis.get("errors", []) if analysis else []), calibration.get("error") if calibration else None]
    reasons = list(dict.fromkeys(reason for reason in reasons if reason))
    if reasons:
        lines.append("Recorded stopping/status reason: " + " | ".join(_brief(r) for r in reasons))
    elif view["status"] != "completed":
        pending = view.get("pending") or {}
        lines.append("Pending: " + ("Exact human review approval" if pending.get("kind") == "review" else _brief(pending.get("questions") or "No stopping reason recorded.")))
    if quality.get("recovery"):
        recovery = quality["recovery"]
        lines.insert(3, "Matrix recovery: mapping **" + recovery["input_mapping"]["status"] + "**, complex matrix **" +
                     recovery["complex_matrix"]["status"] + "**. " + recovery["complex_matrix"]["reason"])
        lines.append("Recovery task: " + recovery["task_goal"] + ". " + " ".join(
            f"{name}: **{recovery[name]['status']}** — {recovery[name]['reason']}"
            for name in ("brightness_prediction", "input_mapping", "complex_matrix")))
        lines.append("Recovery limitations: " + "; ".join(quality["recovery_limitations"]))
        follow = recovery.get("follow_up")
        if follow:
            lines.append("Recommended follow-up: " + follow["measurement"] + "\n\nUnresolved question: " +
                follow["unresolved_question"] + "\n\nDistinguishing result: " + follow["distinguishing_result"])
            lines.append("Bench support: " + follow_up_support(state, follow))
    lines.append("Conversation: " + _link(root, root / "conversation.md", "verbatim conversation") + ". Detailed trace: " +
                 _link(root, root / "agent-messages.md", "all recorded messages and validation details") + ".")
    lines.append("\n## Returned matrix and prediction baseline")
    if evaluation and evaluation.get("methods"):
        lines.append("**Matched algorithm comparison on one acquisition**. " + evaluation["comparison"]["selection"])
        lines.extend(["| Method | Held-out RMSE | Held-out NMSE | Fit time (s) | Shared initialization (s) | Scored readings | Fit clipped | Held-out clipped | Mean-baseline RMSE |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for name, result in evaluation["methods"].items():
            numbers = result["metrics"]
            keys = ("holdout_rmse", "holdout_nmse", "runtime_seconds", "shared_initialization_seconds", "holdout_scored_readings",
                    "fit_clipped_readings", "holdout_clipped_readings", "mean_baseline.holdout_rmse")
            values = [f"{numbers[key]:.9g}" if key in numbers else "not scored" for key in keys]
            lines.append("| " + name + " | " + " | ".join(values) + " |")
        lines.append("Brightness comparison: **" + evaluation["comparison"]["result"] + "**. " + evaluation["comparison"]["limitations"] +
                     " " + evaluation["comparison"]["timing"])
        for name in evaluation["methods"]:
            refs = [ref for ref in outputs if Path(ref["path"]).parent.name == name and Path(ref["path"]).name in
                    ("heldout-predictions.csv", "heldout-predictions.png", "heldout-example-grids.png", "heldout-detector-errors.csv")]
            lines.append(name + " outputs: " + ", ".join(_link(root, ref["path"], Path(ref["path"]).name) for ref in refs))
    if len(estimates) > 1:
        lines.append("Several estimates were saved; no cross-estimate workflow selection is recorded. These are candidates.")
    for estimate in estimates:
        lines.append(f"{estimate['label']}: **{estimate['method']}**, analysis `{estimate['analysis_id']}`, shape {estimate['shape']} [detector, SLM input].")
        source = estimate.get("truth_source")
        if source:
            lines.append("Simulator true TM source (saved configuration): " +
                         (f"JSON `{Path(source['json_path']).name}` — `{source['json_path']}`." if source["json_path"] else
                          f"Seeded random matrix, seed **{source['seed']}**.") +
                         f" Recorded matrix fingerprint: `{source['matrix_fingerprint'] or 'not recorded'}`.")
        lines.append(", ".join(_link(root, estimate[key], label) for key, label in (
            ("estimate_path", "saved estimated TM NPZ"), ("heatmap", "magnitude heatmap"), ("coefficients_csv", "real/imaginary coefficient CSV"))))
        lines.append("Truth comparison: " + estimate["truth"] + (". " + estimate["truth_issue"] if "truth_issue" in estimate else
                     ". True and estimated magnitudes use the same axes and color scale; this is evaluation-only truth, never an agent/fitting input."))
        comparison, baseline = estimate.get("matrix_comparison"), estimate["baseline"]
        if baseline:
            error = comparison["row_phase_aligned_relative_error"] if comparison else None
            matrix_result = ((f"Aligned relative error **{error:.12g}**" if error is not None else "Relative error undefined (zero truth norm)") +
                             f"; strongest inputs **{comparison['strongest_input_agreements']}/{comparison['total_rows']}** agree; "
                             f"{len(comparison['ambiguous_rows'])} ambiguous rows") if comparison else "Comparison unavailable; see the recorded reason above"
            lines.append("| Matrix vs verified simulator truth (human-only) | Held-out brightness prediction |\n"
                         "| --- | --- |\n"
                         f"| {matrix_result} | **{estimate['prediction_status']}**; fit RMSE **{baseline['fit_rmse']}**, NMSE **{baseline['fit_nmse']}**; "
                         f"mean-baseline RMSE **{baseline['mean_baseline_rmse']}**, NMSE **{baseline['mean_baseline_nmse']}** — **{baseline['result']}** |")
        if Path(estimate["heatmap"]).is_file():
            lines.append("!" + _link(root, estimate["heatmap"], "True and estimated TM magnitudes" if "true_coefficients" in estimate else "Estimated TM magnitude"))
        if comparison:
            error = comparison["row_phase_aligned_relative_error"]
            lines.append("**Comparison against simulator truth (human report only):** row-phase-aligned relative matrix error " +
                         (f"**{error:.12g}**" if error is not None else "**undefined (true matrix has zero norm)**") +
                         ". Defined as `||T_est,aligned - T_true||_F / ||T_true||_F`. " + comparison["alignment"] + ".")
            lines.append(f"Strongest-input agreement: **{comparison['strongest_input_agreements']}/{comparison['total_rows']} detector rows**. "
                         f"Ambiguous rows: {', '.join('D'+str(j) for j in comparison['ambiguous_rows']) or 'none'}. " + comparison["diagnostic_limit"] + ".")
            mismatches = [row for row in comparison["strongest_inputs"] if row["status"] == "mismatch"]
            lines.append("Strongest-input mismatches: " + ("; ".join(
                f"D{row['detector']} (true input {row['true_inputs'][0]}, estimated input {row['estimated_inputs'][0]})" for row in mismatches) or "none") + ".")
            ambiguities = [row for row in comparison["strongest_inputs"] if row["status"] == "ambiguous"]
            if ambiguities:
                lines.append("No arbitrary winner is selected for tied/near-tied or zero rows: " + "; ".join(
                    f"D{row['detector']}: true candidates {row['true_inputs']}, estimated candidates {row['estimated_inputs']} (empty means no distinguished input)"
                    for row in ambiguities) + ".")
            lines.append(f"Strongest-input tie tolerance on magnitudes: rtol={comparison['tie_rtol']:g}, atol={comparison['tie_atol']:g}. "
                         "This numerical truth comparison does not replace the separate recorded Analyst recovery assessment.")
        lines.append("Held-out brightness prediction status (recorded): **" + estimate["prediction_status"] + "**.")
        if baseline:
            lines.append(f"Held-out fit RMSE **{baseline['fit_rmse']}**, fitting-mean baseline RMSE **{baseline['mean_baseline_rmse']}**; **{baseline['result']}**. "
                f"NMSE: fit {baseline['fit_nmse']}, mean baseline {baseline['mean_baseline_nmse']}. "
                f"Same eligible readings: {baseline['usable']}/{baseline['total']}; clipped: {baseline['clipped'] if baseline['clipped'] is not None else 'not recorded'}; "
                f"unclipped but unscorable: {baseline['unscorable_unclipped'] if baseline['unscorable_unclipped'] is not None else 'not recorded'}. " + baseline["selection"])
            lines.append(_link(root, estimate["baseline_csv"], "per-detector fit versus mean-baseline scores and counts"))
        else:
            lines.append("Mean-baseline comparison unavailable: " + estimate.get("baseline_issue", "No saved held-out predictions"))
        if "true_coefficients" in estimate:
            lines.append(coefficient_table(estimate["true_coefficients"], estimate["detector_ids"], "True TM (fingerprint verified)"))
        lines.append(coefficient_table(estimate["estimated_coefficients"], estimate["detector_ids"], "Estimated TM (as saved)"))
    if not estimates:
        lines.append("No available saved matrix estimate; no matrix has been invented or fitted by this report.")
    lines.append("\n## Recorded decisions")
    procedure = next((p for p in state["procedures"] if acquisition and p["id"] == acquisition["procedure_id"]),
                     state["procedures"][-1] if state["procedures"] else None)
    proposal = next((p for p in state["proposals"] if procedure and p["id"] == procedure["proposal_id"]),
                    state["proposals"][-1] if state["proposals"] else None)
    lines.append("- Scientist proposed: " + (_brief(proposal["hypothesis"]) + " Reason: " + _brief(proposal.get("rationale")) if proposal else "No proposal/reason recorded."))
    selected_reason = next((e["message"] for e in reversed(state["conversation"])
                            if e["role"] == "experimentor" and procedure and procedure["id"] in e["record_ids"]), None)
    lines.append("- Experimentor selected: " + ("; ".join(_brief(a["instruction"]) for a in procedure["actions"]) +
                 " Reason: " + _brief(selected_reason) if procedure else "No procedure/reason recorded."))
    runner_reason = next((e["message"] for e in reversed(state["conversation"])
                          if e["role"] == "runner" and acquisition and acquisition["id"] in e["record_ids"]), None)
    imported = next((e["message"] for e in state["conversation"] if acquisition and acquisition["id"] in e["record_ids"]
                     and e["role"] == "orchestrator" and e["message"].startswith("Imported saved acquisitions")), None)
    if imported:
        lines.append("- Runner: saved acquisition reused; no new Runner acquisition recorded. Import reason: " + _brief(imported))
    else:
        lines.append("- Runner: " + (f"acquisition {acquisition['id']} {acquisition['status']}. Actual settings: " +
                     json.dumps(acquisition["actual_settings"], sort_keys=True) + ". Recorded explanation: " + _brief(runner_reason)
                     if acquisition else "No acquisition executed; execution reason not recorded."))
    if manifest:
        lines.append(f"  Saved data contains {manifest['fit_patterns']} fit and {manifest['holdout_patterns']} held-out frames, "
                     f"{len(manifest['regions'])} detector readings per frame, with fixed shutter/power.")
    reviews = [r for r in state["analyst_reviews"] if r["stage"] == "after_analysis"]
    lines.append("- Analyst concluded: " + (_brief(reviews[-1]["explanation"]) if reviews else "No post-analysis conclusion/reason recorded."))
    if state["findings"]:
        lines.append("- Scientist interpretation: " + _brief(state["findings"][-1]["interpretation"]))
    lines.append("Full reasons and routing: " + ", ".join(_link(root, root / name, label) for name, label in
                 [("agent-messages.md", "agent messages"), ("decision-log.md", "decision log"),
                  ("router-decisions.json", "routing decisions"), ("workflow.json", "saved workflow")]))
    lines.append("\n## Camera and calibration")
    if calibration:
        if calibration.get("camera_crop"):
            lines.append(f"Camera crop [y0,y1,x0,x1]: `{calibration['camera_crop']}`; half-open bounds. "
                         "Detector IDs match recorded region indices and TM rows; left to right, top to bottom for the 16-block bench.")
        lines += ["", "| Attempt | Shutter (s) | Laser (mW) | Whole-frame saturation | Result |",
                  "| --- | ---: | ---: | ---: | --- |"]
        accepted = calibration.get("accepted_attempt")
        if accepted is None and calibration.get("success") and calibration["attempts"]:
            accepted = calibration["attempts"][-1]["attempt"]  # Older recorded diagnostics.
        for attempt in calibration["attempts"]:
            mark = "Accepted; settings locked" if attempt["attempt"] == accepted else "Not accepted"
            lines.append(f"| {attempt['attempt']} | {attempt['shutter_seconds']:.9g} | {attempt['laser_power_mw']:.9g} | {100*attempt['whole_frame_saturation']:.3f}% | {mark} |")
        if not calibration.get("success"):
            lines.append("Calibration stopped: " + _brief(calibration.get("error")))
        after = calibration.get("after", {})
        saturation = after.get("detector_saturation", [])
        lines.append("Detector clipping in the final calibration attempt (separate from the whole-frame target): " +
                     (", ".join(f"D{j}={100*value:.1f}%" for j, value in enumerate(saturation)) if saturation else "not recorded.") + ".")
    else:
        lines.append("No calibration results recorded.")
    def links(records, specifications):
        return [ _link(root, ref["path"], label) for name, label in specifications
                 if (ref := _artifact(records, name)) is not None ]
    for name in ("before", "after"):
        files = links(raw_refs, [(f"raw_{name}.png", "raw PNG (no overlay)"), (f"raw_{name}.npy", "raw pixel array"),
                                 (f"preview_{name}.png", "camera overlay")])
        if files:
            lines.append(f"Calibration {name}: " + ", ".join(files) + ".")
    files = links(raw_refs, [("calibration.csv", "all calibration attempts CSV"), ("calibration.png", "calibration plot"),
                             ("calibration.json", "calibration diagnostics"), ("bench-dataset.json", "geometry/settings manifest"),
                             ("measurements.npz", "all measured frames and input phases")])
    if files:
        lines.append("Data: " + ", ".join(files) + ".")
    if (root / "bench-metadata.json").is_file():
        lines.append("Run geometry and matrix identity: " + _link(root, root / "bench-metadata.json", "bench metadata") + ".")
    lines.append("\n## Retrieval and what it establishes")
    if evaluation:
        lines.append("Executed method: " + _brief(evaluation.get("method")))
        if example:
            lines.append(f"Representative pattern: `{example['pattern_id']}`. {example['selection']}. "
                         "Measured and predicted grids share the 0-255 count scale; difference is predicted minus measured. Stars mark clipping.")
        else:
            lines.append("A representative held-out inspection was not recorded by this analysis.")
        files = links(outputs, [("heldout-slm-pattern.png", "saved SLM pattern"), ("heldout-camera-raw.png", "raw held-out camera PNG"),
                                ("heldout-camera-raw.npy", "raw held-out array"), ("heldout-camera-preview.png", "held-out overlay"),
                                ("heldout-example-grids.png", "measured/predicted/difference grids"), ("heldout-example.json", "example data and IDs"),
                                ("heldout-detector-errors.png", "all-held-out error/count plot"), ("heldout-detector-errors.csv", "per-detector statistics CSV"),
                                ("heldout-predictions.png", "all-held-out scatter plots"), ("heldout-predictions.csv", "all predictions and clipping flags"),
                                ("estimated-tm.npz", "estimated matrix"), ("bench-evaluation.json", "numerical evaluation")])
        lines.append("Individual outputs: " + ", ".join(files) + ".")
        statistics = evaluation.get("heldout_detector_errors", [])
        if statistics:
            lines += ["", "All held-out patterns (error scores exclude clipped readings and unavailable fits):", "",
                      "| TM row / detector | RMSE | MAE | NMSE | Usable for score | Clipped | Unclipped but unscorable |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for row in statistics:
                values = [f"{row[key]:.6g}" if row[key] is not None else "not scored" for key in ("rmse_counts", "mae_counts", "nmse")]
                lines.append(f"| D{row['detector']} | {' | '.join(values)} | {row['usable_for_score']}/{row['total_readings']} | {row['clipped_readings']} | {row['unscorable_unclipped']} |")
        else:
            lines.append("An all-held-out per-detector summary was not recorded; inspect the saved prediction CSV if available.")
        variation = evaluation.get("measurement_variation", {})
        low = variation.get("low_variation_detectors", [])
        if low:
            lines.append(f"**Low measurement variation:** detectors {', '.join('D'+str(j) for j in low)} have unclipped brightness span <= "
                         f"{variation['span_threshold_counts']:g} count across fit and held-out patterns. "
                         "Low prediction error alone does not establish unique matrix recovery, as in the phase-only identity case.")
        elif variation:
            lines.append("No detector met the recorded low-variation threshold (unclipped span <= 1 count across all patterns).")
        else:
            lines.append("Measurement variation was not recorded; unique matrix recovery is not established.")
        lines.append("This establishes only the recorded workflow execution and held-out detector prediction quality under synthetic camera settings. "
                     "Completion alone does not establish measurement or prediction success. Independent TM-row phase ambiguity remains; no hidden truth was used for retrieval.")
    else:
        lines.append("No completed numerical evaluation available. Prediction quality and matrix recovery are not established.")
    if issues:
        lines.append("Artifact issues: " + " | ".join(issues))
    lines.append("Status export: " + _link(root, root / "quality.json", "quality.json") + ".")
    markdown = re.sub(r"\|\n\n(?=\|)", "|\n", "\n\n".join(lines))
    (root / "run-summary.md").write_text(markdown + "\n", encoding="utf-8")
