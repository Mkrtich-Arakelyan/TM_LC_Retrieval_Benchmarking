"""Human-only matrix/mean-baseline inspection. Never included in agent inputs."""

import csv
import hashlib
import json
from pathlib import Path
from uuid import UUID

import numpy as np


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


def mean_baseline(readings, fit_indices, held_indices, prediction, eligible, clipped=None):
    """Use fit data only; compare predictors on one identical held-out mask."""
    known_clipping = clipped is not None
    clipped = np.zeros_like(readings, bool) if clipped is None else np.asarray(clipped, bool)
    fit, held = readings[fit_indices], readings[held_indices]
    fit_ok, held_ok = ~clipped[fit_indices], ~clipped[held_indices]
    if prediction.shape != held.shape or np.shape(eligible) != (held.shape[1],):
        raise ValueError("Saved predictions/eligibility do not match recorded detector readings")
    means = np.array([np.mean(fit[fit_ok[:, j], j]) if fit_ok[:, j].any() else np.nan for j in range(held.shape[1])])
    baseline = np.broadcast_to(means, held.shape)
    common = held_ok & np.asarray(eligible, bool)[None, :] & np.isfinite(prediction) & np.isfinite(baseline)
    def scores(mask):
        if not mask.any():
            return {"fit_rmse": None, "mean_baseline_rmse": None, "fit_nmse": None, "mean_baseline_nmse": None, "result": "not scored"}
        squared = [(value[mask] - held[mask])**2 for value in (prediction, baseline)]
        mse = [float(np.mean(value)) for value in squared]
        denominator = max(float(np.mean(held[mask]**2)), 1e-12)
        return {"fit_rmse": float(np.sqrt(mse[0])), "mean_baseline_rmse": float(np.sqrt(mse[1])),
                "fit_nmse": mse[0]/denominator, "mean_baseline_nmse": mse[1]/denominator,
                "result": "tie" if mse[0] == mse[1] else "fit better" if mse[0] < mse[1] else "mean baseline better"}
    rows = []
    for j in range(held.shape[1]):
        mask = np.zeros_like(common)
        mask[:, j] = common[:, j]
        rows.append({"detector": j, "fitting_mean": float(means[j]) if np.isfinite(means[j]) else None,
            "eligible_fit": int(fit_ok[:, j].sum()), "heldout_usable": int(mask.sum()),
            "heldout_clipped": int((~held_ok[:, j]).sum()) if known_clipping else None, **scores(mask)})
    return {**scores(common), "usable": int(common.sum()), "total": int(common.size),
        "clipped": int((~held_ok).sum()) if known_clipping else None,
        "unscorable_unclipped": int((held_ok & ~common).sum()) if known_clipping else None, "per_detector": rows,
        "selection": "Per-detector mean of eligible fitting readings only; both predictors scored on the same eligible held-out readings. No refitting or holdout selection."
            + (" Clipping flags were not recorded; eligibility uses the available saved data." if not known_clipping else "")}


def _inputs(view, analysis):
    from workflow.state import WorkflowState
    from workflow.config import AppConfig
    from analysis.datasets import registered_artifacts, load_dataset
    from analysis.bench_analysis import load_bench_dataset
    state = WorkflowState.model_validate(view["state"])
    config = AppConfig.model_validate(state.control.configuration)
    refs, _ = registered_artifacts(state, [UUID(analysis["id"])])
    for ref in analysis["input_data"]:
        if Path(ref["path"]).suffix.lower() != ".json":
            continue
        reference = refs[UUID(ref["id"])]
        if analysis["method_settings"].get("operation") == "bench.retrieve":
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
