"""Verify an optical bench run's shutdown using live queries and run records."""

import argparse
from dataclasses import replace
from pathlib import Path

from hardware.bench_access import BenchBusyError, bench_access
from hardware.generator import GeneratorSettings, RigolDG812
from hardware.laser import LabLaser, LaserSettings, SHUTDOWN_EXPECTED
from hardware.run_records import (capture_records, object_or_empty as _object,
                                  read_json, recorded_release, terminal_capture)


DEVICES = ("Laser", "Generator", "Camera", "SLM")
def _recorded_release(records, device):
    state, reason = recorded_release(records, device)
    if state == "failed":
        return "FAIL", "run record reports release failed"
    return {"released": "PASS", "not_initialized": "NOT USED", "unverified": "UNVERIFIED"}[state], reason


def _check_laser(settings):
    states = LabLaser(LaserSettings(**settings)).read_status_once()
    values = {query: state["value"] for query, state in states.items() if state.get("valid") is True}
    shutter = {0: "closed", 1: "open"}.get(values.get("?S"), "unknown")
    laser = {0: "off (standby)", 1: "on", 2: "off due to fault"}.get(values.get("?L"), "unknown")
    setpoint = f'{values["?SP"]:g} W' if "?SP" in values else "unverified"
    measured = f'{values["?P"]:g} W' if "?P" in values else "unverified"
    reason = (f"shutter {shutter}, laser {laser}, shutdown setpoint {setpoint} "
              f'(expected {SHUTDOWN_EXPECTED["?SP"]:g} W), measured power {measured}')
    if any(query in values and values[query] != expected
           for query, expected in SHUTDOWN_EXPECTED.items()):
        return "FAIL", reason
    if not SHUTDOWN_EXPECTED.keys() <= values.keys() or "?P" not in values:
        return "UNVERIFIED", reason
    return "PASS", reason


def _check_generator(records, settings):
    usage = []
    for record in records:
        cleanup = _object(record.get("cleanup"))
        status = _object(cleanup.get("generator")).get("status")
        if isinstance(record.get("lcc"), dict) or status in ("confirmed", "failed", "UNCONFIRMED"):
            usage.append(True)
        elif status == "not_initialized" or (terminal_capture(record) and cleanup
                                              and "lcc" not in record and "generator" not in cleanup):
            usage.append(False)  # One-frame capture records omit LCC entirely when it is skipped.
        else:
            usage.append(None)
    if True not in usage:
        if usage and all(used is False for used in usage):
            return "NOT USED", "run record shows no LCC drive"
        return "UNVERIFIED", "run record does not establish whether CH1 was used"

    # The run establishes usage. The current config supplies the address, even
    # if the operator has since disabled the generator for future captures.
    generator = RigolDG812(replace(GeneratorSettings(**settings), enabled=True))
    generator.connect()
    try:
        enabled = generator.read_output_enabled()
    finally:
        close_errors = generator.disconnect()
    if enabled:
        return "FAIL", "CH1 output is ON"
    if close_errors:
        return "UNVERIFIED", "CH1 reported OFF, but VISA close failed"
    return "PASS", "CH1 output is OFF (live query)"


def _query(check, *args):
    try:
        return check(*args)
    except Exception as exc:
        return "UNVERIFIED", str(exc) or type(exc).__name__


def _check_run(run_dir, config_path):
    records = capture_records(run_dir)
    try:
        settings = read_json(config_path)
    except (OSError, ValueError):
        settings = {}
    laser = _query(_check_laser, _object(settings.get("laser")))
    generator = _query(_check_generator, records, _object(settings.get("generator")))
    return (laser, generator, _recorded_release(records, "camera"), _recorded_release(records, "slm"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.local.json"),
                        help="Lab configuration JSON (default: config.local.json)")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Exact capture output or Runner acquisition ID directory")
    args = parser.parse_args(argv)
    try:
        with bench_access():
            results = _check_run(args.run_dir.resolve(), args.config)
    except (BenchBusyError, OSError, ValueError, KeyboardInterrupt) as exc:
        results = [("UNVERIFIED", str(exc) or type(exc).__name__)] * len(DEVICES)
    for device, (status, reason) in zip(DEVICES, results):
        print(f'{device}: {status} - {" ".join(reason.split())}')
    return int(any(status not in ("PASS", "NOT USED") for status, _ in results))


if __name__ == "__main__":
    raise SystemExit(main())
