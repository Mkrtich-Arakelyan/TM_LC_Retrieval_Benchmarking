"""Small local CLI. Normal mode never silently selects fake integrations."""

import argparse
import json
from pathlib import Path

from config import AppConfig
from graph import WorkflowApp


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Persistent TM experiment workflow; no real drivers included")
    parser.add_argument("--db", type=Path, help="SQLite checkpoint database")
    parser.add_argument("--factory", help="Trusted Python module:function returning graph.Integrations(config)")
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start")
    start.add_argument("question")
    start.add_argument("--fixture", action="store_true", help="Explicit TEST ONLY offline handoffs")
    start.add_argument("--analyst", action="store_true", help="Include canned Analyst in fixture mode")
    start.add_argument("--architecture", type=Path)
    for name in ("show", "resume", "cancel", "review", "clarify", "reconcile", "follow-up"):
        command = sub.add_parser(name)
        command.add_argument("workflow_id")
        if name == "resume":
            command.add_argument("--refresh-config", action="store_true", help="Use current environment config for a blocked workflow; retain counters")
        if name == "cancel":
            command.add_argument("--reason", default="Cancelled by user")
        if name == "review":
            command.add_argument("decision", choices=["approve", "reject"])
            command.add_argument("--fingerprint", help="Exact review fingerprint displayed by show; mandatory for approval")
            command.add_argument("--reviewer")
            command.add_argument("--feedback")
        if name in ("clarify", "reconcile", "follow-up"):
            command.add_argument("message")
        if name in ("clarify", "follow-up"):
            command.add_argument("--allow-partial-analysis", action="store_true")
        if name == "reconcile":
            command.add_argument("--confirmed-stopped", action="store_true", help="Operator confirmation, not an automated safety check")
    args = parser.parse_args(argv)
    cfg = AppConfig.from_env()
    if args.db:
        cfg.workflow_database_path = args.db
    try:
        with WorkflowApp(cfg, integration_factory=args.factory) as app:
            if args.command == "start":
                kwargs = {"architecture_path": args.architecture} if args.architecture else {}
                result = app.start(args.question, fixture=args.fixture, fixture_analyst=args.analyst, **kwargs)
            elif args.command == "show":
                result = app.show(args.workflow_id)
            elif args.command == "resume":
                result = app.resume(args.workflow_id, configuration=cfg if args.refresh_config else None)
            elif args.command == "cancel":
                result = app.cancel(args.workflow_id, args.reason)
            elif args.command == "review":
                result = app.submit(args.workflow_id, {"action": args.decision,
                    "review_fingerprint": args.fingerprint, "reviewer": args.reviewer, "feedback": args.feedback})
            elif args.command == "clarify":
                result = app.submit(args.workflow_id, {"action": "clarify", "message": args.message,
                                                       "allow_partial_analysis": args.allow_partial_analysis})
            elif args.command == "reconcile":
                result = app.submit(args.workflow_id, {"action": "reconcile", "message": args.message,
                                                       "confirmed_stopped": args.confirmed_stopped})
            else:
                result = app.follow_up(args.workflow_id, args.message, allow_partial_analysis=args.allow_partial_analysis)
        print(f"Workflow {result['workflow_id']}: {result['status']}")
        if result.get("reason"):
            print(result["reason"])
        if result.get("input_error"):
            print("Input rejected; workflow unchanged: " + result["input_error"])
        for entry in result["state"]["conversation"]:
            print(f"\n[{entry['role']}] {entry['message']}")
        if result["pending"]:
            print("\nPending human input:\n" + json.dumps(result["pending"], indent=2, ensure_ascii=False))
        return 2 if result["status"] in ("blocked", "failed", "stopped") or result.get("input_error") else 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Workflow error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
