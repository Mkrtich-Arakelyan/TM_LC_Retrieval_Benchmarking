"""One acquisition, matched reviewed fits, normal approval and truthful interpretation."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

import main
from analysis import bench_analysis
from bench.acquisition import extract_detectors
from bench.config import BenchConfig
from bench.geometry import geometry_metadata
from workflow.config import AppConfig
from workflow.state import TMFitSettings

COMPARISON_QUESTION = "Compare GGS and nonlinear intensity least squares (intensity_ls) on this simulated phase-only bench in one comparison run."


@pytest.mark.parametrize("identity,compare", [(True, True), (False, True), (False, False)])
def test_initial_task_selects_one_bench_operation_and_reaches_interpretation(tmp_path, monkeypatch, capsys, identity, compare):
    config_file = tmp_path / "bench.json"
    fit_count, holdout_count = (512, 48) if identity else (128, 16)
    config = {"fit_patterns": fit_count, "holdout_patterns": holdout_count}
    if identity:
        config["detector_matrix_path"] = str(Path(__file__).resolve().parents[1] / "configs/matrices/identity-16x16.json")
    config_file.write_text(json.dumps(config))
    cfg = AppConfig(workflow_database_path=tmp_path / "workflow.sqlite", run_output_path=tmp_path / "runs")
    monkeypatch.setattr(AppConfig, "from_env", lambda: cfg.model_copy(deep=True))
    calls = []
    original = bench_analysis.fit_methods
    def fit(X, I, settings, **kwargs):
        calls.append((X.copy(), I.copy(), settings.model_dump()))
        return original(X, I, settings, **kwargs)
    monkeypatch.setattr(bench_analysis, "fit_methods", fit)
    question = COMPARISON_QUESTION if compare else "Run GGS only on this simulated phase-only bench."
    assert main.main(["start", question, "--simulate", "--test-agents", "--bench-config", str(config_file)]) == 0
    source = next((tmp_path / "runs").glob("*/workflow.json"))
    waiting = json.loads(source.read_text(encoding="utf-8"))
    assert waiting["state"]["user_request"] == question
    assert waiting["status"] == "awaiting_approval" and not waiting["state"]["acquisitions"] and not calls
    assert waiting["state"]["proposals"][0]["retrieval_methods"] == (["ggs", "intensity_ls"] if compare else ["ggs"])
    assert main.main(["review", waiting["workflow_id"], "approve", "--simulate", "--fingerprint", "wrong"]) == 2
    assert not calls
    assert main.main(["review", waiting["workflow_id"], "approve", "--simulate", "--fingerprint", waiting["pending"]["review"]["review_fingerprint"]]) == 0
    final = json.loads(source.read_text(encoding="utf-8"))
    assert final["workflow_id"] == waiting["workflow_id"] and final["status"] == "completed", final.get("reason")
    assert len(final["state"]["acquisitions"]) == len(final["state"]["analyses"]) == 1
    result = final["state"]["analyses"][0]
    assert result["method_settings"]["operation"] == ("tm.compare_reference_free" if compare else "bench.retrieve")
    acquisition = final["state"]["acquisitions"][0]
    manifest = json.loads(Path(next(a["path"] for a in acquisition["raw_data"] if a["path"].endswith("bench-dataset.json"))).read_text())
    with np.load(manifest["arrays"]["path"]) as data:
        for j, (X, I, settings) in enumerate(calls):
            np.testing.assert_array_equal(X, np.exp(1j*data["phases"][:fit_count]))
            np.testing.assert_array_equal(I[:, 0], data["detector_readings"][:fit_count, j])
        assert len(calls) == 16  # One shared fitter call per detector, not one per method.
        if identity:
            assert np.all(data["detector_readings"] == 40)
        else:
            assert np.max(np.std(data["detector_readings"], axis=0)) > 1
    metrics = result["metrics"]
    names = {Path(ref["path"]).name for ref in result["outputs"]}
    text = (source.parent / "run-summary.md").read_text(encoding="utf-8")
    if compare:
        assert {"ggs.npz", "intensity_ls.npz"} <= names
        assert metrics["shared_eligible_masks"] == 1
        for name in ("ggs", "intensity_ls"):
            assert metrics[f"{name}.holdout_scored_readings"] == 16*holdout_count
            assert metrics[f"{name}.holdout_clipped_readings"] == 0
            assert metrics[f"{name}.runtime_seconds"] > 0
            assert metrics[f"{name}.holdout_rmse"] >= 0 and metrics[f"{name}.holdout_nmse"] >= 0
            assert f"{name}.mean_baseline.holdout_rmse" in metrics
        assert "Matched algorithm comparison on one acquisition" in text and "| intensity_ls |" in text
        assert text.count("row-phase-aligned relative matrix error") == 2  # Human-only truth comparison for both.
        review = final["state"]["analyst_reviews"][-1]
        assert "GGS: RMSE=" in review["explanation"] and "nonlinear intensity_ls:" in review["explanation"]
        if identity:
            assert metrics["comparison.brightness_tie"] == 1
            assert metrics["ggs.holdout_rmse"] == metrics["intensity_ls.holdout_rmse"] == 0
            assert review["recovery"]["input_mapping"]["status"] == "unresolved"
            assert "brightness tie" in review["explanation"] and "**tie**" in text
    else:
        assert "estimated-tm.npz" in names and not {"ggs.npz", "intensity_ls.npz"} & names
        assert "Matched algorithm comparison" not in text
    assert not any(d["action"] == "plan_follow_up" for d in final["state"]["router_decisions"])
    capsys.readouterr()


def test_matched_clipping_masks_and_holdout_never_affect_either_fit(tmp_path, monkeypatch):
    cfg = BenchConfig(fit_patterns=144, holdout_patterns=8)
    geometry = geometry_metadata()
    manifest = {**geometry, "config": cfg.model_dump(), "fit_patterns": 144,
        "camera_gain": 4000., "shutter_seconds": .01, "laser_power_mw": 1., "background_counts": 0., "read_noise_std": 0.}
    frames = np.zeros((152, 20, 20), np.uint8)
    for y0, y1, x0, x1 in geometry["regions"]:
        frames[:, y0:y1, x0:x1] = 40
        frames[:4, y0:y1, x0:x1] = 255
        frames[144, y0:y1, x0:x1] = 255
    readings, saturation = extract_detectors(frames, geometry["regions"])
    data = {"frames": frames, "phases": np.random.default_rng(8).uniform(-np.pi, np.pi, (152, 16)),
        "detector_readings": readings, "detector_saturation": saturation,
        "pattern_ids": np.array([f"fit-{i}" for i in range(144)] + [f"held-{i}" for i in range(8)])}
    calls = []
    def fit(X, I, settings, **kwargs):
        calls.append((X.copy(), I.copy()))
        T = np.zeros((1, 16), complex); T[0, 5] = np.sqrt(40)
        return {name: {"T": T.copy() * phase, "fit_mse": 0., "runtime_seconds": 0.}
                for name, phase in (("ggs", 1), ("intensity_ls", 1j))}
    monkeypatch.setattr(bench_analysis, "fit_methods", fit)
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir(); second.mkdir()
    settings = TMFitSettings(iterations=2, restarts=1)
    metrics, _ = bench_analysis.retrieve(manifest, data, settings, first, deadline=time.monotonic()+90, compare=True)
    data["detector_readings"][144:] = 10  # Changes evaluation only, not fit input or selection.
    bench_analysis.retrieve(manifest, data, settings, second, deadline=time.monotonic()+90, compare=True)
    assert len(calls) == 32 and all(X.shape == (140, 16) and np.all(I == 40) for X, I in calls)
    for j in range(16):
        np.testing.assert_array_equal(calls[j][0], calls[16+j][0])
    for name in ("ggs", "intensity_ls"):
        assert metrics[f"{name}.fit_clipped_readings"] == 64 and metrics[f"{name}.holdout_clipped_readings"] == 16
        assert metrics[f"{name}.holdout_scored_readings"] == 16*7
        with np.load(first / name / (name+".npz")) as a, np.load(second / name / (name+".npz")) as b:
            np.testing.assert_array_equal(a["T"], b["T"])
