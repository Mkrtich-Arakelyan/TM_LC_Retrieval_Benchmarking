"""Recorded-only summaries, consistent detector IDs, and honest error/count views."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from analysis.bench_analysis import inspection_outputs
from bench.acquisition import extract_detectors
from bench.geometry import geometry_metadata
from workflow.report import export_decisions, export_workflow_view


def artifact(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def inspection(tmp_path):
    geometry = geometry_metadata()
    counts = np.arange(16)[None, :] + 2*np.arange(5)[:, None] + 20
    counts[2, 3] = 255
    frames = np.zeros((5, 20, 20), dtype=np.uint8)
    for j, (y0, y1, x0, x1) in enumerate(geometry["regions"]):
        frames[:, y0:y1, x0:x1] = counts[:, j, None, None]
    readings, saturated = extract_detectors(frames, geometry["regions"])
    data = {"frames": frames, "detector_readings": readings, "detector_saturation": saturated,
            "phases": np.random.default_rng(8).uniform(-np.pi, np.pi, (5, 16)),
            "pattern_ids": np.array(["fit-0000", "fit-0001", "holdout-0000", "holdout-0001", "holdout-0002"])}
    manifest = {**geometry, "fit_patterns": 2, "holdout_patterns": 3}
    predicted = np.minimum(readings[2:] + .25, 255)
    predicted_sat = np.zeros((3, 16))
    predicted_sat[0, 3] = 1
    identifiable = [True]*16
    identifiable[4] = False
    output = tmp_path / "analysis"
    output.mkdir()
    report, files = inspection_outputs(manifest, data, predicted, predicted_sat, identifiable, output)
    return output, manifest, data, predicted, report, files


def test_representative_grids_and_raw_frame_share_saved_detector_ids(inspection):
    output, manifest, data, predicted, report, files = inspection
    example = json.loads((output / "heldout-example.json").read_text())
    assert example["pattern_id"] == "holdout-0000"
    assert example["detector_ids"] == list(range(16))
    assert example["grid_positions"] == [[j//4, j % 4] for j in range(16)]
    assert example["regions"] == manifest["regions"]
    assert example["brightness_scale"] == [0, 255]
    np.testing.assert_array_equal(example["measured_counts"], data["detector_readings"][2])
    np.testing.assert_array_equal(example["predicted_counts"], predicted[0])
    np.testing.assert_array_equal(example["difference_counts"], predicted[0]-data["detector_readings"][2])
    with Image.open(output / "heldout-camera-raw.png") as image:
        assert image.mode == "L" and image.size == (20, 20)
        np.testing.assert_array_equal(np.asarray(image), data["frames"][2])
    assert all((output / name).is_file() for name in files)
    rows = report["heldout_detector_errors"]
    assert rows[3]["clipped_readings"] == 1 and rows[3]["usable_for_score"] == 2
    assert rows[4]["usable_for_score"] == 0 and rows[4]["unscorable_unclipped"] == 3
    assert rows[4]["rmse_counts"] is None
    for row in rows:
        assert row["usable_for_score"] + row["clipped_readings"] + row["unscorable_unclipped"] == 3
    with (output / "heldout-detector-errors.csv").open() as file:
        assert [int(row["detector"]) for row in csv.DictReader(file)] == list(range(16))


def make_view(tmp_path, inspection=None, *, stopped=False, reasons=True):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    calibration = {**geometry_metadata(), "success": not stopped,
                   "error": "Saturation target unreachable within configured limits" if stopped else None,
                   "accepted_attempt": None if stopped else 2,
                   "attempts": [{"attempt": 1, "shutter_seconds": .001, "laser_power_mw": 1., "whole_frame_saturation": .01},
                                {"attempt": 2, "shutter_seconds": .01, "laser_power_mw": 1., "whole_frame_saturation": .02 if stopped else .04}],
                   "after": {"detector_saturation": [0., 0., 0., .25] + [0.]*12}}
    path = raw / "calibration.json"
    path.write_text(json.dumps(calibration))
    state = {"control": {"configuration": {"run_output_path": str(tmp_path), "bench_test_agents": True}},
             "proposals": [{"id": "proposal", "hypothesis": "Recorded Scientist method: GGS", "rationale": "Recorded fit-only rationale"}],
             "procedures": [{"id": "procedure", "proposal_id": "proposal", "version": 1,
                             "actions": [{"action": "bench.acquire", "instruction": "Recorded phase-only acquisition", "requested_settings": {}}]}],
             "acquisition_reviews": [], "router_decisions": [], "findings": [],
             "acquisitions": [{"id": "acquisition", "procedure_id": "procedure", "status": "partial" if stopped else "completed",
                               "data_origin": "simulated", "actual_settings": {}, "raw_data": [artifact(path)],
                               "errors": [calibration["error"]] if stopped else []}],
             "analyses": [], "analyst_reviews": [], "conversation": []}
    if reasons:
        state["conversation"] = [{"role": "experimentor", "record_ids": ["procedure"], "message": "Recorded selection reason"},
                                 {"role": "runner", "record_ids": ["acquisition"], "message": "Recorded execution explanation"}]
    if inspection:
        output, _, _, _, report, files = inspection
        report = {**report, "method": "Recorded reviewed GGS operation"}
        path = output / "bench-evaluation.json"
        path.write_text(json.dumps(report))
        state["analyses"] = [{"id": "analysis", "status": "completed", "method_settings": {"operation": "bench.retrieve"},
                              "metrics": {"measurement_passed": 1, "prediction_passed": 0},
                              "outputs": [artifact(output / name) for name in files] + [artifact(path)], "errors": []}]
        state["analyst_reviews"] = [{"stage": "after_analysis", "status": "reviewed", "explanation": "Recorded Analyst says prediction failed"}]
    return {"workflow_id": "summary-test", "status": "awaiting_information" if stopped else "completed",
            "reason": None, "state": state}


def test_summary_separates_quality_uses_recorded_reasons_and_links_all_outputs(tmp_path, inspection):
    view = make_view(tmp_path, inspection)
    export_workflow_view(view)
    export_decisions(view)
    text = (tmp_path / "run-summary.md").read_text()
    assert "Workflow: **completed**" in text
    assert "Measurement quality: **pass**" in text and "Prediction quality: **failed_or_inconclusive**" in text
    for reason in ("Recorded fit-only rationale", "Recorded selection reason", "Recorded execution explanation", "Recorded Analyst says prediction failed"):
        assert reason in text
    assert "Accepted; settings locked" in text and "D3=25.0%" in text
    assert "heldout-example-grids.png" in text and "heldout-detector-errors.csv" in text
    assert "heldout-camera-raw.png" in text and "heldout-camera-preview.png" in text
    assert "| D0 |" in text and "| D15 |" in text
    assert "|\n\n|" not in text  # Markdown table rows must stay contiguous.


def test_stopped_summary_names_calibration_stage_and_does_not_invent_reasons(tmp_path):
    view = make_view(tmp_path, stopped=True, reasons=False)
    export_workflow_view(view)
    export_decisions(view)
    text = (tmp_path / "run-summary.md").read_text()
    assert "Calibration diagnostics saved; target not accepted" in text
    assert "Saturation target unreachable" in text
    assert "Measurement quality: **failed**" in text and "Prediction quality: **not_evaluated**" in text
    assert "Accepted; settings locked" not in text
    assert "Reason not recorded" in text and "No post-analysis conclusion/reason recorded" in text
    assert "Prediction quality and matrix recovery are not established" in text


def test_reused_measurements_are_not_described_as_a_new_runner_acquisition(tmp_path, inspection):
    view = make_view(tmp_path, inspection)
    view["state"]["conversation"] = [{"role": "orchestrator", "record_ids": ["acquisition"],
        "message": "Imported saved acquisitions from workflow source; acquisition is disabled"}]
    export_workflow_view(view)
    export_decisions(view)
    text = (tmp_path / "run-summary.md").read_text()
    assert "saved acquisition reused; no new Runner acquisition recorded" in text
    assert "Imported saved acquisitions from workflow source" in text


def test_identity_like_constant_measurements_flag_ambiguity(tmp_path, monkeypatch):
    # Exercise reporting on measured values only, without a known matrix input.
    import bench.plots as plots
    for name in ("heldout_error_plot", "representative_grids", "frame_preview", "slm_pattern_plot"):
        monkeypatch.setattr(plots, name, lambda *args, **kwargs: None)
    values = np.full((5, 16), 40.)
    data = {"detector_readings": values, "detector_saturation": np.zeros_like(values),
            "phases": np.zeros((5, 16)), "pattern_ids": np.array(["fit-0", "fit-1", "held-0", "held-1", "held-2"]),
            "frames": np.full((5, 20, 20), 40, dtype=np.uint8)}
    manifest = {**geometry_metadata(), "fit_patterns": 2}
    report, files = inspection_outputs(manifest, data, values[2:], np.zeros((3, 16)), [True]*16, tmp_path)
    assert report["measurement_variation"]["low_variation_detectors"] == list(range(16))
    assert all(row["rmse_counts"] == 0 for row in report["heldout_detector_errors"])
    view = make_view(tmp_path)
    path = tmp_path / "bench-evaluation.json"
    path.write_text(json.dumps({**report, "method": "Measured-only fixture"}))
    view["state"]["analyses"] = [{"id": "analysis", "status": "completed", "method_settings": {"operation": "bench.retrieve"},
        "metrics": {"measurement_passed": 1, "prediction_passed": 1}, "outputs": [artifact(path)], "errors": []}]
    export_workflow_view(view)
    export_decisions(view)
    text = (tmp_path / "run-summary.md").read_text()
    assert "**Low measurement variation:**" in text
    assert "Low prediction error alone does not establish unique matrix recovery" in text


def test_changed_numerical_artifact_is_not_silently_summarized(tmp_path, inspection):
    view = make_view(tmp_path, inspection)
    ref = view["state"]["analyses"][0]["outputs"][-1]
    Path(ref["path"]).write_text('{"method":"changed after registration"}')
    export_workflow_view(view)
    export_decisions(view)
    text = (tmp_path / "run-summary.md").read_text()
    assert "SHA-256 does not match" in text
    assert "changed after registration" not in text
