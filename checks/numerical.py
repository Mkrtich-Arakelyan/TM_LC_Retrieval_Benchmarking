"""Deterministic algebraic test dataset and real reviewed-operation workflow.

This is a tiny I=|Tx|² fixture, not an optical simulator or physical acquisition.
Run: python -m checks.numerical --output outputs/numerical-check
"""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from workflow.config import AppConfig
from workflow.graph import Integrations, WorkflowApp
from fixtures.offline import FixtureScientistModel
from fixtures.numerical import create_fixture, NumericalFixtureAnalyst


def run_fixture(output: Path) -> dict:
    case = output.resolve() / str(uuid4())
    state, request = create_fixture(case / "inputs")
    cfg = AppConfig(workflow_database_path=case / "workflow.sqlite", run_output_path=case / "runs",
        numerical_input_path=case / "inputs", reviewed_numerics_enabled=True, max_workflow_iterations=12)
    services = Integrations(analyst_model=NumericalFixtureAnalyst(request), scientist_model=FixtureScientistModel())
    with WorkflowApp(cfg, services) as app:
        return app.start(state.user_request, initial_state=state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/numerical-check"))
    view = run_fixture(parser.parse_args().output)
    print(f"Workflow {view['workflow_id']}: {view['status']}")
    if view["state"]["analyses"]:
        result = view["state"]["analyses"][-1]
        print(json.dumps({"status": result["status"], "metrics": result["metrics"],
                          "outputs": result["outputs"], "errors": result["errors"]}, indent=2))
    if view["reason"]:
        print(view["reason"])
    return 0 if view["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
