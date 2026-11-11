"""Reviewed workflow-facing numerical operations; no model-supplied code execution."""

from __future__ import annotations

import time
from pathlib import Path
from uuid import UUID, uuid4

from workflow.config import AppConfig
from workflow.state import AnalysisRequest, Artifact, ExecutionDetails, NumericalAnalysisResult, WorkflowState
from tools.python_execution import fingerprint, validated_metrics
from analysis.datasets import registered_artifacts
from analysis.tm_solvers import fit_methods
from analysis.tm_analysis import evaluate

OPERATIONS = ("tm.validate_dataset", "tm.compare_reference_free", "tm.evaluate_saved_predictions", "bench.retrieve")


def execute_operation(state: WorkflowState, request: AnalysisRequest, config: AppConfig,
                      *, attempt_number=1, run_id: UUID | None = None) -> NumericalAnalysisResult:
    """Dispatch only an enumerated reviewed function; no imports/calls from model strings."""
    if request.operation not in OPERATIONS:
        raise ValueError("An explicit supported numerical operation is required")
    request = AnalysisRequest.model_validate(request.model_dump())
    config = AppConfig.model_validate(config.model_dump())
    artifacts, acquisition_ids = registered_artifacts(state, request.input_record_ids)
    requested_ids = [request.dataset_artifact_id] + ([request.ground_truth_artifact_id] if request.ground_truth_artifact_id else [])
    if any(key not in artifacts for key in requested_ids):
        raise ValueError("Named operation references an unregistered input artifact ID")
    if not 1 <= attempt_number <= config.max_workflow_analysis_attempts:
        raise ValueError("Numerical attempt budget exceeded")
    root = (config.run_output_path / "analysis").resolve()
    if root.is_relative_to(config.numerical_input_path.resolve()) or config.numerical_input_path.resolve().is_relative_to(root):
        raise ValueError("Numerical input and output roots must be disjoint")
    directory = root / str(run_id or uuid4())
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    # Record relocated reviewed code as well as this dispatch interface. Hashes in
    # the primary audit copy bind its code_version to these support snapshots.
    from analysis import datasets, metrics, tm_analysis, tm_solvers
    implementations = [datasets, metrics, tm_analysis, tm_solvers]
    if request.operation == "bench.retrieve":
        from analysis import bench_analysis
        from bench import config as bench_config, optical_bench, acquisition, plots
        implementations.extend([bench_analysis, bench_config, optical_bench, acquisition, plots])
    snapshot_root = directory / "reviewed_sources"
    snapshot_root.mkdir()
    for implementation in implementations:
        snapshot = snapshot_root / (implementation.__name__ + ".py")
        snapshot.write_bytes(Path(implementation.__file__).read_bytes())
        source += f"\n# Reviewed dependency: {snapshot.name} sha256={fingerprint(snapshot)}\n".encode("utf-8")
    code_path = directory / "reviewed_implementation.py"
    code_path.write_bytes(source)  # audit copy only; never imported/executed
    code = Artifact(path=str(code_path), description="Reviewed tm_numerics implementation audit copy", sha256=fingerprint(code_path))
    common = dict(acquisition_ids=acquisition_ids, input_data=[artifacts[k] for k in requested_ids],
                  code=code, code_version=code.sha256, method_settings={"operation": request.operation,
                  **request.parameters.model_dump()}, random_seed=request.parameters.seed)
    started = time.monotonic()
    if not config.reviewed_numerics_enabled:
        return NumericalAnalysisResult(**common, execution=ExecutionDetails(status="unavailable",
            backend="reviewed_numpy", attempt_number=attempt_number, reason="Enable TM_REVIEWED_NUMERICS_ENABLED to run reviewed named operations; arbitrary code remains disabled"))
    try:
        if request.operation == "bench.retrieve":
            from analysis.bench_analysis import load_bench_dataset, retrieve
            if request.ground_truth_artifact_id is not None:
                raise ValueError("Bench retrieval does not accept hidden ground-truth inputs")
            manifest, data, arrays_ref = load_bench_dataset(artifacts[request.dataset_artifact_id], artifacts, config)
            if any(next(a for a in state.acquisitions if a.id == key).data_origin != "simulated" for key in acquisition_ids):
                raise ValueError("Bench analysis requires explicitly simulated source acquisitions")
            common["input_data"].append(arrays_ref)
            common["data_origin"] = "simulated"
            # Include an audit copy of the specialized reviewed implementation.
            import analysis.bench_analysis as implementation
            audit_path = directory / "reviewed_bench_analysis.py"
            audit_path.write_bytes(Path(implementation.__file__).read_bytes())
            metrics, outputs = retrieve(manifest, data, request.parameters, directory,
                                       deadline=started + config.numerical_timeout_seconds)
            outputs.append(Artifact(path=str(audit_path), description="Reviewed bench analysis audit copy", sha256=fingerprint(audit_path)))
            return NumericalAnalysisResult(**common, status="completed", metrics=validated_metrics(metrics), outputs=outputs,
                execution=ExecutionDetails(status="completed", backend="reviewed_numpy", attempt_number=attempt_number,
                    exit_code=0, elapsed_seconds=time.monotonic() - started,
                    stdout="Reviewed intensity-only bench retrieval; no generated code or hidden truth used"))
        return evaluate(state, request, config, artifacts, acquisition_ids, common, started,
                        directory, fit_methods=fit_methods, attempt_number=attempt_number)
    except Exception as exc:
        return NumericalAnalysisResult(**common, status="failed", errors=[str(exc) or type(exc).__name__],
            execution=ExecutionDetails(status="failed", backend="reviewed_numpy", attempt_number=attempt_number,
                reason=str(exc), elapsed_seconds=time.monotonic() - started))
