"""Read device cleanup evidence without importing the workflow or SDKs."""

import hashlib
import json
from pathlib import Path


CAPTURE_RECORD = "capture-result.json"


def object_or_empty(value):
    return value if isinstance(value, dict) else {}


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path.name}")
    return value


def terminal_capture(record):
    return record.get("capture_status") in ("completed", "failed")


def capture_records(run_dir):
    """Read a final capture or the physical Runner's linked per-device records."""
    runner = run_dir / "record.json"
    if not runner.exists():
        entry = read_json(run_dir / CAPTURE_RECORD)
        if not terminal_capture(entry):
            raise ValueError("capture is still running or has no final record")
        return [entry]

    record = read_json(runner)
    hardware = object_or_empty(record.get("hardware"))
    if (record.get("status") not in ("completed", "failed", "aborted", "partial")
            or not hardware.get("ended_at") or hardware.get("cleanup_status") == "pending"):
        raise ValueError("Runner acquisition or cleanup is still running or has no final record")
    if hardware.get("data_origin") != "physical":
        raise ValueError("Runner record does not describe a physical acquisition")

    # Older records may have only a top-level artifact list. Current Runner
    # records associate evidence with each attempted action.
    actions = hardware.get("actions")
    if actions is None:
        return _capture_artifacts(run_dir, record.get("raw_data"))
    captures = []
    for action in actions if isinstance(actions, list) else []:
        action = object_or_empty(action)
        if action.get("status") != "pending":
            captures.extend(_capture_artifacts(run_dir, action.get("artifacts")) or [{}])
    return captures


def _capture_artifacts(run_dir, artifacts):
    captures = []
    for artifact in artifacts if isinstance(artifacts, list) else []:
        artifact = object_or_empty(artifact)
        name = artifact.get("path")
        if not isinstance(name, str) or Path(name).name != CAPTURE_RECORD:
            continue
        try:
            path = (run_dir / name).resolve()
            path.relative_to(run_dir)
            content = path.read_bytes()
            expected_hash = artifact.get("sha256")
            if expected_hash is not None and hashlib.sha256(content).hexdigest() != expected_hash:
                raise ValueError("Runner cleanup artifact hash mismatch")
            entry = object_or_empty(json.loads(content.decode("utf-8")))
        except (OSError, ValueError):
            entry = {}
        captures.append(entry if terminal_capture(entry) else {})
    return captures


def recorded_release(records, device):
    entries = [object_or_empty(object_or_empty(record.get("cleanup")).get(device))
               for record in records]
    for entry in entries:
        if entry.get("status") == "failed":
            return "failed", entry.get("error") or "run record reports release failed"
    statuses = [entry.get("status") for entry in entries]
    if statuses and all(status in ("disconnect_returned", "not_initialized") for status in statuses):
        if "disconnect_returned" in statuses:
            return "released", "release verified from run record"
        return "not_initialized", "run record says device was not initialized"
    return "unverified", "run record has no conclusive device release"
