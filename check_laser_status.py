"""Query the configured laser controller without changing its operating state."""

import argparse
import json
from pathlib import Path

from hardware.bench_access import bench_access
from hardware.laser import LabLaser, LaserSettings


STATUS_FIELDS = (("?S", "Shutter"), ("?L", "Laser"),
                 ("?P", "Measured power"), ("?SP", "Setpoint"))


def _describe(query, state):
    if state.get("valid") is not True:
        return "UNVERIFIED - " + str(state.get("error", "missing reply"))
    value = state["value"]
    if query == "?S":
        return {0: "closed", 1: "open"}[value]
    if query == "?L":
        return {0: "standby", 1: "on", 2: "fault (off due to fault)"}[value]
    return f"{value:g} W"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.local.json"),
                        help="Lab configuration JSON (default: config.local.json)")
    args = parser.parse_args(argv)
    try:
        with bench_access():
            data = json.loads(args.config.read_text(encoding="utf-8"))
            settings = LaserSettings(**data["laser"])
            states = LabLaser(settings).read_status_once()
    except (Exception, KeyboardInterrupt) as exc:
        # Includes configuration, ownership, serial-open and serial-close errors.
        reason = " ".join((str(exc) or type(exc).__name__).split())
        for _, label in STATUS_FIELDS:
            print(f"{label}: UNVERIFIED - {reason}")
        return 1

    for query, label in STATUS_FIELDS:
        print(f'{label}: {" ".join(_describe(query, states.get(query, {})).split())}')
    print("Controller responses only; no startup or shutdown commands sent.")
    return int(any(states.get(query, {}).get("valid") is not True
                   for query, _ in STATUS_FIELDS))


if __name__ == "__main__":
    raise SystemExit(main())
