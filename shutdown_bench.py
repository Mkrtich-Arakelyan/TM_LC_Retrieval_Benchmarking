"""Request bench cancellation, shut down independent outputs, and verify release."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import time

from hardware.bench_access import (BenchBusyError, bench_access, latest_bench_run,
                                   rearm_bench, request_shutdown, shutdown_access)
from hardware.generator import GeneratorSettings, RigolDG812
from hardware.laser import LabLaser, LaserSettings, SHUTDOWN_EXPECTED
from hardware.run_records import capture_records, read_json, recorded_release


_DEVICES = ("Laser", "Generator", "Camera", "SLM")
_OWNER_WAIT_SECONDS = 3.0
_DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.local.json"


def _laser(settings):
    if settings is None:
        return "NOT CONFIGURED", "no laser in bench configuration"
    report = LabLaser(LaserSettings(**settings)).shutdown_once()
    checks = {item["query"]: item for item in report["checks"]}
    values = {query: item["value"] for query, item in checks.items() if item["valid"]}
    evidence = ", ".join(f"{query}={values.get(query, 'unverified')}"
                         for query in ("?S", "?L", "?SP", "?P"))
    evidence += " (?SP setpoint W, ?P measured W)"
    errors = [f'{item["command"]}: {item.get("error", "failed")}'
              for item in report["actions"] if item["status"] != "sent"]
    if report["serial_close"] != "closed":
        errors.append(f'serial close {report["serial_close"]}')
    if errors:
        return "FAILED", f'{evidence}. {". ".join(errors)}'
    if any(query in values and values[query] != expected
           for query, expected in SHUTDOWN_EXPECTED.items()):
        return "FAILED", f"shutdown readback differs from closed shutter, standby, or 0.01 W setpoint. {evidence}"
    if not report["confirmed"]:
        read_errors = [f'{query}: {item["error"]}' for query, item in checks.items()
                       if not item["valid"] and item.get("error")]
        detail = ". ".join([evidence, *read_errors])
        return "UNVERIFIED", f"shutdown procedure lacks a required confirmed readback. {detail}"
    return "VERIFIED", f"closed shutter and standby confirmed, serial closed. {evidence}"


def _generator(settings):
    if settings is None:
        return "NOT CONFIGURED", "no generator in bench configuration"
    if isinstance(settings, dict) and settings.get("enabled", False) is False:
        return "NOT CONFIGURED", "generator is disabled in bench configuration"
    result = RigolDG812(GeneratorSettings(**settings)).shutdown_once()
    errors = ". ".join(result["errors"])
    if result["output_enabled"] is True:
        return "FAILED", f"CH1 reads ON. {errors}"
    if result["output_enabled"] is None:
        return "UNVERIFIED", f"CH1 OFF readback unavailable. {errors}"
    if errors:
        return "FAILED", f"CH1 reads OFF, but shutdown reported an error. {errors}"
    return "VERIFIED", "CH1 OFF command sent, CH1 readback OFF, VISA closed"


def _attempt(operation, settings):
    try:
        return operation(settings)
    except (Exception, KeyboardInterrupt) as exc:
        return "UNVERIFIED", str(exc) or type(exc).__name__


def _wait_for_bench(stack):
    deadline = time.monotonic() + _OWNER_WAIT_SECONDS
    while True:
        try:
            stack.enter_context(bench_access())
            return None
        except BenchBusyError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return f"bench owner did not release within {_OWNER_WAIT_SECONDS:g} s of cancellation"
            time.sleep(min(0.05, remaining))
        except OSError as exc:
            return f"could not verify release of bench ownership: {exc}"


def _cleanup_evidence(fallback):
    try:
        run_dir = latest_bench_run()
    except FileNotFoundError:
        if fallback is None:
            return [], "no tracked run record. Use --run-dir for a run predating bench tracking"
        run_dir = fallback
    return capture_records(run_dir.resolve()), str(run_dir)


def _release_result(device, settings, records, evidence, owner_problem):
    if settings.get(device) is None and not any(
            isinstance(record.get("cleanup"), dict) and device in record["cleanup"] for record in records):
        return "NOT CONFIGURED", f"no {device.upper() if device == 'slm' else device} in bench configuration"
    if owner_problem:
        return "UNVERIFIED", owner_problem
    state, reason = recorded_release(records, device)
    status = {"released": "VERIFIED", "not_initialized": "VERIFIED",
              "failed": "FAILED", "unverified": "UNVERIFIED"}[state]
    return status, f"{reason}. Evidence: {evidence}"


def _shutdown(config_path, fallback):
    with shutdown_access(), ExitStack() as stack:
        request_shutdown()
        owner_problem = _wait_for_bench(stack)
        try:
            settings = read_json(config_path)
        except (OSError, ValueError) as exc:
            return [("UNVERIFIED", f"bench configuration unavailable: {exc}")] * len(_DEVICES), 1

        laser = _attempt(_laser, settings.get("laser"))
        generator = _attempt(_generator, settings.get("generator"))
        if owner_problem:
            try:
                stack.enter_context(bench_access())
            except (BenchBusyError, OSError):
                pass
            else:
                owner_problem = None
        records, evidence = [], ""
        if owner_problem is None:
            try:
                records, evidence = _cleanup_evidence(fallback)
            except (OSError, ValueError) as exc:
                evidence = f"run record unavailable: {exc}"
        results = (laser, generator,
                   _release_result("camera", settings, records, evidence, owner_problem),
                   _release_result("slm", settings, records, evidence, owner_problem))
        if owner_problem:
            results = tuple((status, f"{reason}. {owner_problem}") if status == "UNVERIFIED"
                            and owner_problem not in reason else (status, reason)
                            for status, reason in results)
            if not any(owner_problem in reason for _, reason in results):
                results = ((laser[0], f"{laser[1]}. {owner_problem}"), *results[1:])
        return results, int(bool(owner_problem) or
                            any(status not in ("VERIFIED", "NOT CONFIGURED") for status, _ in results))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                        help="Lab configuration JSON (default: config.local.json beside this script)")
    parser.add_argument("--run-dir", type=Path, help="Run directory to use if there is no tracked physical run")
    parser.add_argument("--rearm", action="store_true",
                        help="Clear the shutdown latch after the operator confirms the bench is safe")
    args = parser.parse_args(argv)
    if args.rearm:
        try:
            rearm_bench()
        except (BenchBusyError, OSError) as exc:
            print(f"Bench remains latched off - {exc}")
            return 1
        print("Bench re-armed. No hardware connections opened.")
        return 0
    try:
        results, code = _shutdown(args.config, args.run_dir)
    except (BenchBusyError, OSError, KeyboardInterrupt) as exc:
        results = [("UNVERIFIED", str(exc) or type(exc).__name__)] * len(_DEVICES)
        code = 1
    for device, (status, reason) in zip(_DEVICES, results):
        print(f'{device}: {status} - {" ".join(reason.split())}')
    return code


if __name__ == "__main__":
    raise SystemExit(main())
