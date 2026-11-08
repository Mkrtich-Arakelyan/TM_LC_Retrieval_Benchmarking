"""Real NumPy fitting on a deterministic algebraic dataset; no optics/hardware."""

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from config import AppConfig
from state import AnalysisRequest, TMFitSettings
from tools.numerical_fixture import create_fixture, run_fixture
from tools.python_execution import fingerprint
from tools.tm_numerics import (aligned_matrix_errors, execute_operation,
                               intensity_loss_gradient, load_dataset, registered_artifacts)


@pytest.fixture
def dataset(tmp_path):
    state, request = create_fixture(tmp_path / "inputs")
    cfg = AppConfig(numerical_input_path=tmp_path / "inputs", run_output_path=tmp_path / "outputs",
                    reviewed_numerics_enabled=True)
    return state, request, cfg


def artifacts(state, request):
    return registered_artifacts(state, request.input_record_ids)[0]


def rewrite(dataset, *, arrays_change=None, manifest_change=None):
    state, request, _ = dataset
    refs = artifacts(state, request)
    manifest_ref = refs[request.dataset_artifact_id]
    manifest = json.loads(Path(manifest_ref.path).read_text())
    arrays_ref = refs[next(key for key in refs if str(key) == manifest["arrays"]["id"])]
    if arrays_change:
        with np.load(arrays_ref.path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays_change(arrays)
        np.savez(arrays_ref.path, **arrays)
        arrays_ref.sha256 = fingerprint(Path(arrays_ref.path))
        # Update the registered artifact, not merely the manifest's assertion.
        for record in state.acquisitions:
            for ref in record.raw_data:
                if ref.id == arrays_ref.id:
                    ref.sha256 = arrays_ref.sha256
        manifest["arrays"] = arrays_ref.model_dump(mode="json")
    if manifest_change:
        manifest_change(manifest)
    Path(manifest_ref.path).write_text(json.dumps(manifest), encoding="utf-8")
    for ref in state.acquisitions[0].raw_data:
        if ref.id == manifest_ref.id:
            ref.sha256 = fingerprint(Path(ref.path))


def output(result, method):
    path = next(a.path for a in result.outputs if Path(a.path).name == method + ".npz")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def test_real_operation_outputs_and_audit_metadata(dataset):
    state, request, cfg = dataset
    before = state.model_dump_json()
    input_hashes = {a.path: fingerprint(Path(a.path)) for a in state.acquisitions[0].raw_data}
    result = execute_operation(state, request, cfg)
    assert result.status == "completed" and result.execution.backend == "reviewed_numpy"
    assert result.data_origin == "test_only" and result.random_seed == request.parameters.seed
    assert result.code_version == fingerprint(Path(result.code.path))
    for method in ("ggs", "intensity_ls"):
        arrays = output(result, method)
        assert arrays["T"].shape == (2, 3) and np.iscomplexobj(arrays["T"])
        assert arrays["holdout_prediction"].shape == (30, 2)
        assert np.isfinite(arrays["holdout_squared_error"]).all()
        assert result.metrics[method + ".holdout_nmse"] < 1e-8
        assert result.metrics[method + ".matrix_aligned_relative_error"] < 1e-4
        assert result.metrics[method + ".runtime_seconds"] >= 0
    report = json.loads(Path(result.outputs[-1].path).read_text())
    assert report["selection"].startswith("fit intensity MSE only")
    assert report["settings"]["restarts"] == 2
    assert report["methods"]["ggs"]["fit_histories"]
    assert all(a.sha256 == fingerprint(Path(a.path)) for a in result.outputs)
    assert state.model_dump_json() == before
    assert input_hashes == {p: fingerprint(Path(p)) for p in input_hashes}


@pytest.mark.parametrize("change", ["holdout_intensity", "holdout_input", "ground_truth"])
def test_holdout_and_ground_truth_do_not_change_fitting_or_selection(dataset, change):
    state, request, cfg = dataset
    original = execute_operation(state, request, cfg)
    if change == "ground_truth":
        ref = artifacts(state, request)[request.ground_truth_artifact_id]
        np.savez(ref.path, T=np.ones((2, 3), dtype=complex) * (50 + 2j))
        for saved in state.acquisitions[0].raw_data:
            if saved.id == ref.id:
                saved.sha256 = fingerprint(Path(saved.path))
    else:
        def alter(arrays):
            if change == "holdout_intensity":
                arrays["intensities"][100:] *= 9
            else:
                arrays["X"][100:] *= 2 + 1j
        rewrite(dataset, arrays_change=alter)
    modified = execute_operation(state, request, cfg)
    assert modified.status == "completed"
    old_report = json.loads(Path(original.outputs[-1].path).read_text())
    new_report = json.loads(Path(modified.outputs[-1].path).read_text())
    for method in ("ggs", "intensity_ls"):
        np.testing.assert_array_equal(output(original, method)["T"], output(modified, method)["T"])
        for field in ("selected_restart", "selected_iteration", "fit_histories"):
            assert old_report["methods"][method][field] == new_report["methods"][method][field]
    if change != "ground_truth":
        assert modified.metrics["ggs.holdout_mse"] > original.metrics["ggs.holdout_mse"]


def test_gradient_matches_complex_directional_finite_difference():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(8, 3)) + 1j * rng.normal(size=(8, 3))
    W = rng.normal(size=(3, 2)) + 1j * rng.normal(size=(3, 2))
    direction = rng.normal(size=W.shape) + 1j * rng.normal(size=W.shape)
    I = rng.uniform(size=(8, 2))
    _, gradient = intensity_loss_gradient(X, I, W)
    epsilon = 1e-6
    finite = (intensity_loss_gradient(X, I, W + epsilon * direction)[0]
              - intensity_loss_gradient(X, I, W - epsilon * direction)[0]) / (2 * epsilon)
    assert finite == pytest.approx(np.real(np.vdot(gradient, direction)), rel=1e-7)


def test_row_phase_alignment_is_independent_for_each_output():
    rng = np.random.default_rng(2)
    truth = rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))
    estimate = truth * np.exp(1j * np.array([0.2, -1.7, 2.5]))[:, None]
    rows, absolute, relative = aligned_matrix_errors(estimate, truth)
    assert max(rows) < 1e-14 and absolute < 1e-14 and relative < 1e-14


@pytest.mark.parametrize("case", ["dimension", "pairing", "nan_input", "inf_intensity", "negative_intensity",
    "duplicate_frame", "duplicate_pattern", "identical_patterns", "rank", "overlap", "duplicate_split",
    "out_of_range", "unassigned", "embedded_truth", "provenance"])
def test_invalid_dataset_rejected_without_success(dataset, case):
    state, request, cfg = dataset
    def bad_arrays(a):
        if case == "dimension": a["intensities"] = a["intensities"][:-1]
        if case == "pairing": a["measurement_pattern_ids"] = a["measurement_pattern_ids"][::-1]
        if case == "nan_input": a["X"][0, 0] = np.nan
        if case == "inf_intensity": a["intensities"][0, 0] = np.inf
        if case == "negative_intensity": a["intensities"][0, 0] = -1
        if case == "duplicate_frame": a["frame_ids"][-1] = a["frame_ids"][0]
        if case == "duplicate_pattern": a["pattern_ids"][-1] = a["pattern_ids"][0]
        if case == "identical_patterns": a["X"][-1] = a["X"][0]
        if case == "rank": a["X"][:100, 1] = a["X"][:100, 0]
        if case == "embedded_truth": a["T"] = np.eye(3, dtype=complex)
    def bad_manifest(m):
        if case == "overlap": m["holdout_indices"][0] = 0
        if case == "duplicate_split": m["fit_indices"][1] = 0
        if case == "out_of_range": m["holdout_indices"][-1] = 999
        if case == "unassigned": m["holdout_indices"].pop()
        if case == "provenance": m["data_origin"] = "physical"
    rewrite(dataset, arrays_change=bad_arrays, manifest_change=bad_manifest)
    result = execute_operation(state, request, cfg)
    assert result.status == "failed" and result.errors and not result.metrics


def test_registry_parameter_policy_and_disabled_execution(dataset):
    state, request, cfg = dataset
    with pytest.raises(ValidationError):
        TMFitSettings(iterations=1000000)
    with pytest.raises(ValidationError):
        TMFitSettings(seed=True)
    with pytest.raises(ValidationError):
        AnalysisRequest.model_validate({**request.model_dump(), "operation": "exec"})
    with pytest.raises(ValueError, match="unregistered"):
        execute_operation(state, request.model_copy(update={"dataset_artifact_id": uuid4()}), cfg)
    cfg.reviewed_numerics_enabled = False
    result = execute_operation(state, request, cfg)
    assert result.status == "pending" and result.execution.status == "unavailable"
    assert not result.metrics


def test_validation_operation_and_no_truth_mode(dataset):
    state, request, cfg = dataset
    validation = execute_operation(state, request.model_copy(update={"operation": "tm.validate_dataset", "ground_truth_artifact_id": None}), cfg)
    assert validation.status == "completed" and len(validation.outputs) == 1
    result = execute_operation(state, request.model_copy(update={"ground_truth_artifact_id": None}), cfg)
    assert result.status == "completed" and not any("matrix_aligned" in k for k in result.metrics)


def test_real_workflow_request_returns_to_analyst_without_hardware_or_generated_code(tmp_path):
    view = run_fixture(tmp_path / "reproduce")
    assert view["status"] == "completed"
    state = view["state"]
    assert state["approvals"] == []
    assert [r["status"] for r in state["analyst_reviews"]] == ["more_analysis", "reviewed"]
    assert state["analyses"][0]["execution"]["backend"] == "reviewed_numpy"
    assert state["analyses"][0]["metrics"]["intensity_ls.holdout_nmse"] < 1e-8
    assert state["analysis_code"] == []  # no model-generated script was dispatched
    assert state["acquisitions"][0]["data_origin"] == "test_only"
    assert state["control"]["analysis_attempts"] == 1 and state["control"]["acquisition_attempts"] == 0
