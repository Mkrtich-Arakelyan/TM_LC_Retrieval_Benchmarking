"""Human-only matrix/mean-baseline inspection. Never included in agent inputs."""

import csv
import hashlib
import json
from pathlib import Path
from uuid import UUID

import numpy as np
from analysis.metrics import mean_baseline


def checked_bytes(ref):
    content = Path(ref["path"]).read_bytes()
    if not ref.get("sha256") or hashlib.sha256(content).hexdigest() != ref["sha256"]:
        raise ValueError("Missing/mismatched recorded artifact SHA-256: " + ref["path"])
    return content


def load_estimate(ref):
    import io
    with np.load(io.BytesIO(checked_bytes(ref)), allow_pickle=False) as saved:
        matrix = saved["T"].copy()
        if matrix.ndim != 2 or matrix.size > 128*256 or not np.isfinite(matrix).all():
            raise ValueError("Expected a finite small estimated matrix [detector,input]")
        prediction = saved["holdout_prediction"].copy() if "holdout_prediction" in saved else None
        eligible = saved["fit_eligible"].copy() if "fit_eligible" in saved else saved["identifiable"].copy() if "identifiable" in saved else np.ones(len(matrix), bool)
    return matrix, prediction, eligible



def _inputs(view, analysis):
    from workflow.state import WorkflowState
    from workflow.config import AppConfig
    from analysis.datasets import registered_artifacts, load_dataset
    from analysis.bench_analysis import is_bench_manifest, load_bench_dataset
    state = WorkflowState.model_validate(view["state"])
    config = AppConfig.model_validate(state.control.configuration)
    refs, _ = registered_artifacts(state, [UUID(analysis["id"])])
    for ref in analysis["input_data"]:
        if Path(ref["path"]).suffix.lower() != ".json":
            continue
        reference = refs[UUID(ref["id"])]
        if is_bench_manifest(reference, config):
            manifest, data, _ = load_bench_dataset(reference, refs, config)
            split = manifest["fit_patterns"]
            return manifest, data["detector_readings"], np.arange(split), np.arange(split, len(data["phases"])), data["detector_saturation"] > 0
        dataset = load_dataset(reference, refs, config)
        return None, dataset.intensities, dataset.manifest.fit_indices, dataset.manifest.holdout_indices, None
    raise ValueError("No registered measurement manifest available for the mean baseline")


def verified_truth(manifest):
    if not manifest or manifest.get("data_origin") != "simulated":
        raise ValueError("Verified simulation truth not available")
    from bench.config import BenchConfig
    from bench.optical_bench import OpticalBench
    bench = OpticalBench(BenchConfig.model_validate(manifest["config"]))
    return bench._evaluation_detector_rows(manifest.get("matrix_fingerprint"))


def matrix_comparison(estimate, truth, detector_ids, *, rtol=1e-6, atol=1e-12):
    """Human-only diagnostics; align row phase, never permute or rescale inputs."""
    from analysis.metrics import aligned_matrix_errors
    row_errors, absolute, relative = aligned_matrix_errors(estimate, truth)
    def strongest(row):
        magnitudes = np.abs(row)
        maximum = float(magnitudes.max())
        if maximum <= atol:
            return []  # No distinguished input for a zero/numerically tiny row.
        return (np.flatnonzero(np.isclose(magnitudes, maximum, rtol=rtol, atol=atol)) + 1).tolist()
    rows = []
    for detector, true_row, fitted_row in zip(detector_ids, truth, estimate):
        expected, recovered = strongest(true_row), strongest(fitted_row)
        ambiguous = len(expected) != 1 or len(recovered) != 1
        rows.append({"detector": detector, "true_inputs": expected, "estimated_inputs": recovered,
                     "status": "ambiguous" if ambiguous else "agree" if expected == recovered else "mismatch"})
    return {"row_phase_aligned_relative_error": relative, "row_phase_aligned_absolute_error": absolute,
            "row_phase_aligned_errors": row_errors.tolist(), "total_rows": len(rows),
            "strongest_input_agreements": sum(row["status"] == "agree" for row in rows),
            "strongest_input_mismatches": [row["detector"] for row in rows if row["status"] == "mismatch"],
            "ambiguous_rows": [row["detector"] for row in rows if row["status"] == "ambiguous"],
            "strongest_inputs": rows, "tie_rtol": rtol, "tie_atol": atol,
            "alignment": "One unit-modulus global phase per detector row only; no amplitude rescaling or row/column permutation",
            "diagnostic_limit": "Strongest-input agreement is a diagnostic, not a full test of matrix recovery"}


def coefficient_parts(matrix):
    return {"real": matrix.real.tolist(), "imag": matrix.imag.tolist()}


def matrix_plot(estimate, truth, detector_ids, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [np.abs(estimate)] if truth is None else [np.abs(truth), np.abs(estimate)]
    fig, axes = plt.subplots(1, len(panels), figsize=(7*len(panels), 6), squeeze=False)
    maximum = max(float(p.max()) for p in panels) or 1.
    for axis, panel, title in zip(axes[0], panels, ["Estimated magnitude"] if truth is None else ["Verified true magnitude", "Estimated magnitude"]):
        im = axis.imshow(panel, vmin=0, vmax=maximum, cmap="viridis", aspect="auto", interpolation="nearest")
        axis.set(title=title, xlabel="SLM input (1-based; column index = input − 1)", ylabel="Detector / matrix row")
        axis.set_xticks(range(panel.shape[1]), [str(i+1) for i in range(panel.shape[1])])
        axis.set_yticks(range(panel.shape[0]), [f"D{i}" for i in detector_ids])
    fig.colorbar(im, ax=list(axes[0]), label="Complex coefficient magnitude", shrink=.8)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def export_estimates(view, root):
    """Derived inspection files stay outside workflow artifacts and agent state."""
    candidates = [(analysis, ref) for analysis in view["state"]["analyses"] if analysis["status"] == "completed"
        for ref in analysis["outputs"] if Path(ref["path"]).name in ("estimated-tm.npz", "ggs.npz", "intensity_ls.npz")]
    results, issues = [], []
    for analysis, ref in candidates:
        try:
            estimate, prediction, eligible = load_estimate(ref)
            directory = root / "matrix-inspection" / (analysis["id"] + "-" + Path(ref["path"]).stem)
            directory.mkdir(parents=True, exist_ok=True)
            result = {"analysis_id": analysis["id"], "estimate_path": ref["path"],
                "label": "Candidate" if len(candidates) > 1 else "Estimate",
                "method": "GGS (bench.retrieve fixes this method before held-out evaluation)" if Path(ref["path"]).name == "estimated-tm.npz" else Path(ref["path"]).stem,
                "shape": list(estimate.shape), "truth": "unavailable", "baseline": None}
            method = Path(ref["path"]).stem
            passed = analysis.get("metrics", {}).get(f"{method}.prediction_passed", analysis.get("metrics", {}).get("prediction_passed"))
            result["prediction_status"] = "pass" if passed == 1 else "failed_or_inconclusive" if passed == 0 else "not recorded"
            manifest = None
            try:
                manifest, readings, fit, held, clipped = _inputs(view, analysis)
                if prediction is not None:
                    result["baseline"] = mean_baseline(readings, fit, held, prediction, eligible, clipped)
            except (ValueError, OSError, KeyError) as exc:
                result["baseline_issue"] = str(exc)
            truth = None
            try:
                truth = verified_truth(manifest)
                if truth.shape != estimate.shape:
                    raise ValueError("Verified truth and estimate shapes differ")
                result["truth"] = "verified against recorded matrix fingerprint"
            except (ValueError, OSError, KeyError) as exc:
                truth = None
                result["truth_issue"] = str(exc)
            ids = manifest.get("detector_order", list(range(len(estimate)))) if manifest else list(range(len(estimate)))
            result["detector_ids"] = ids
            result["estimated_coefficients"] = coefficient_parts(estimate)
            result["truth_source"] = None
            config = manifest["config"] if manifest else view["state"]["control"]["configuration"].get("optical_bench")
            if config:
                result["truth_source"] = {"json_path": config.get("detector_matrix_path"), "seed": config.get("seed"),
                                          "matrix_fingerprint": manifest.get("matrix_fingerprint") if manifest else None}
            if truth is not None:
                result["true_coefficients"] = coefficient_parts(truth)
                result["matrix_comparison"] = matrix_comparison(estimate, truth, ids)
            heatmap = directory / "matrix-magnitudes.png"
            matrix_plot(estimate, truth, ids, heatmap)
            coefficients = directory / "estimated-coefficients.csv"
            with coefficients.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["detector", "matrix_row", "input_index", "SLM_input_1_based", "real", "imag", "magnitude"])
                for j, row in enumerate(estimate):
                    for i, value in enumerate(row):
                        writer.writerow([f"D{ids[j]}", j, i, i+1, value.real, value.imag, abs(value)])
            result.update(heatmap=str(heatmap), coefficients_csv=str(coefficients))
            if result["baseline"]:
                path = directory / "prediction-baseline.csv"
                rows = result["baseline"]["per_detector"]
                with path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                result["baseline_csv"] = str(path)
            results.append(result)
        except (ValueError, OSError, KeyError) as exc:
            issues.append(f"Estimate {ref['path']} unavailable: {exc}")
    (root / "matrix-inspection.json").write_text(json.dumps({"estimates": results, "issues": issues}, indent=2, allow_nan=False), encoding="utf-8")
    return results, issues
