"""Deterministic algebraic test dataset and real reviewed-operation workflow.

This is a tiny I=|Tx|² fixture, not an optical simulator or physical acquisition.
Run: python -m tools.numerical_fixture --output outputs/numerical-check
"""

import argparse
import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from config import AppConfig
from graph import Integrations, WorkflowApp
from state import (AcquisitionResult, AnalysisRequest, Artifact, ExperimentProposal,
                   Setting, TMFitSettings, WorkflowState)
from tools.offline_fixtures import FixtureAnalystModel, FixtureScientistModel
from tools.python_execution import fingerprint


def create_fixture(directory: Path) -> tuple[WorkflowState, AnalysisRequest]:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    X = (rng.normal(size=(130, 3)) + 1j * rng.normal(size=(130, 3))) / np.sqrt(6)
    T = (rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))) / np.sqrt(2)
    I = np.abs(X @ T.T) ** 2
    ids = np.asarray([f"pattern-{i}" for i in range(len(X))])
    path = directory / "patterns-and-intensities.npz"
    if path.exists():
        raise ValueError("Use a fresh fixture directory; saved input data is never overwritten")
    np.savez(path, X=X, intensities=I, pattern_ids=ids,
             measurement_pattern_ids=ids, frame_ids=np.asarray([f"frame-{i}" for i in range(len(X))]))
    arrays = Artifact(path=str(path.resolve()), description="TEST ONLY algebraic complex fields and intensities",
                      media_type="application/x-npz", sha256=fingerprint(path))
    truth_path = directory / "evaluation-only-truth.npz"
    np.savez(truth_path, T=T)
    truth = Artifact(path=str(truth_path.resolve()), description="TEST ONLY evaluation-only ground truth; excluded from fitting",
                     sha256=fingerprint(truth_path))
    manifest_path = directory / "dataset.json"
    manifest_path.write_text(json.dumps({"version": 1, "arrays": arrays.model_dump(mode="json"),
        "data_origin": "test_only", "provenance": "Deterministic algebraic fixture; no hardware, optical simulation or calibration",
        "fit_indices": list(range(100)), "holdout_indices": list(range(100, 130))}, indent=2), encoding="utf-8")
    manifest = Artifact(path=str(manifest_path.resolve()), description="TEST ONLY dataset manifest with independent row splits",
                        media_type="application/json", sha256=fingerprint(manifest_path))
    proposal = ExperimentProposal(scientific_question="Verify the offline numerical comparison path on a known algebraic fixture",
        hypothesis="Reviewed functions produce finite predictions from fit-only estimates",
        rationale="Software/numerical verification only, not a physical TM benchmark",
        variables={"method": "Generalized GS versus intensity least squares"},
        controls=["Identical fit patterns, intensities, random-phase LS initializations and restart count"],
        required_measurements=["Known saved complex fields and their paired intensities"],
        evaluation_criteria=["Final held-out prediction errors; ground truth used only for row-phase-aligned evaluation"])
    acquisition = AcquisitionResult(procedure_id=uuid4(), procedure_version=1,
        procedure_fingerprint="0" * 64, status="completed", data_origin="test_only",
        actual_settings={"fixture_rows": Setting(value=len(X))}, raw_data=[manifest, arrays, truth])
    state = WorkflowState(user_request="Run reviewed offline numerical operations on saved test-only data",
                          proposals=[proposal], acquisitions=[acquisition])
    state.control.next_node = "analyst"
    state.control.analyst_stage = "after_analysis"
    state.control.task = "analysis"
    state.control.active_proposal_ids = [proposal.id]
    request = AnalysisRequest(calculation="Compare reference-free fits with final holdout evaluation",
        operation="tm.compare_reference_free", input_record_ids=[acquisition.id],
        dataset_artifact_id=manifest.id, ground_truth_artifact_id=truth.id,
        parameters=TMFitSettings(iterations=400, restarts=2, seed=7))
    return state, request


class NumericalFixtureAnalyst(FixtureAnalystModel):
    def __init__(self, request):
        self.request = request

    def invoke(self, messages):
        payload = json.loads(messages[1][1])
        if not payload["workflow_state"]["analyses"]:
            return {"stage": "after_analysis", "status": "more_analysis",
                "proposal_ids": payload["proposal_ids"], "analysis_ids": [],
                "evidence_ids": list(map(str, self.request.input_record_ids)),
                "explanation": "TEST ONLY model requests real reviewed numerical functions on saved fixture data",
                "recommendations": ["Use only fit data for estimation and selection"],
                "analysis_request": self.request.model_dump(mode="json")}
        result = super().invoke(messages)
        result["limitations"] = ["Analyst response is canned; numerical outputs are real, but physical/scientific validity is not established"]
        return result


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
