"""Standalone inspection plots, deliberately separate from optical calculations."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def calibration_plot(attempts, path, *, accepted_attempt=None):
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.axhspan(3, 5, color="green", alpha=.18, label="Target 3–5%")
    ax.plot([a["attempt"] for a in attempts], [100*a["whole_frame_saturation"] for a in attempts], "o-", label="Whole raw frame")
    accepted = next((a for a in attempts if a["attempt"] == accepted_attempt), None)
    if accepted is not None:
        ax.scatter([accepted["attempt"]], [100*accepted["whole_frame_saturation"]],
                   marker="*", s=180, color="green", zorder=5, label="Accepted; settings locked")
    ax.set(xlabel="Calibration attempt (settings in calibration.csv)", ylabel="Pixels at 255 (%)", title="Synthetic camera saturation calibration")
    ax.legend()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def raw_frame_image(frame, path):
    """Lossless grayscale camera pixels, with no annotations or rescaling."""
    from PIL import Image
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path)


def frame_preview(frame, regions, path, title, *, crop=None):
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    image = ax.imshow(frame, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    y, x = np.where(frame == 255)
    ax.scatter(x, y, marker="x", s=22, color="red", label="Saturated (255)")
    for i, (y0, y1, x0, x1) in enumerate(regions):
        ax.add_patch(Rectangle((x0-.5, y0-.5), x1-x0, y1-y0, fill=False, edgecolor="cyan", linewidth=1.5))
        ax.text((x0+x1-1)/2, (y0+y1-1)/2, f"D{i}", color="cyan", fontsize=6, ha="center", va="center")
    if crop is not None:
        y0, y1, x0, x1 = crop
        ax.add_patch(Rectangle((x0-.5, y0-.5), x1-x0, y1-y0, fill=False,
                               edgecolor="yellow", linestyle="--", linewidth=1.2,
                               label="8 x 8 camera crop"))
    ax.set(title=f"Raw frame: {title}", xlabel="Camera x", ylabel="Camera y")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(image, ax=ax, label="Raw camera counts (fixed 0–255)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def prediction_plot(measured, predicted, clipped, path):
    count = measured.shape[1]
    columns = min(4, int(np.ceil(np.sqrt(count))))
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                             figsize=(4 * columns, 3.4 * rows), layout="constrained")
    for j, ax in enumerate(axes.flat):
        if j >= count:
            ax.set_visible(False)
            continue
        valid = ~clipped[:, j]
        ax.scatter(measured[valid, j], predicted[valid, j], s=18, label="Usable")
        ax.scatter(measured[~valid, j], predicted[~valid, j], marker="x", color="red", label="Clipped measured")
        top = max(float(measured[:, j].max()), float(predicted[:, j].max()), 1)
        ax.plot([0, top], [0, top], "k--", alpha=.5)
        ax.set(title=f"Detector {j}", xlabel="Measured held-out counts", ylabel="Predicted held-out counts")
        ax.legend(fontsize=8)
    fig.suptitle("Predictions from fit-only TM estimates; same camera/ROI extraction")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def representative_grids(example, path):
    """Same saved detector-to-pixel mapping for measured/predicted/difference grids."""
    positions = example["grid_positions"]
    shape = (max(p[0] for p in positions)+1, max(p[1] for p in positions)+1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    difference_limit = max(1., max(abs(v) for v in example["difference_counts"]))
    images = []
    for ax, key, title in zip(axes, ("measured_counts", "predicted_counts", "difference_counts"),
                              ("Measured", "Predicted", "Predicted - measured")):
        values = np.full(shape, np.nan)
        for value, position in zip(example[key], positions):
            values[tuple(position)] = value
        difference = key == "difference_counts"
        images.append(ax.imshow(values, cmap="RdBu_r" if difference else "gray",
                                vmin=-difference_limit if difference else 0,
                                vmax=difference_limit if difference else 255,
                                interpolation="nearest"))
        for j, (row, column) in enumerate(positions):
            flagged = (example["measured_clipped"][j] or example["predicted_clipped"][j]) if difference else example["measured_clipped" if key == "measured_counts" else "predicted_clipped"][j]
            color = ("white" if abs(example[key][j]) > .6*difference_limit else "black") if difference else ("black" if example[key][j] > 127 else "white")
            ax.text(column, row, f"D{example['detector_ids'][j]}{'*' if flagged else ''}\n{example[key][j]:.2f}",
                    ha="center", va="center", color=color, fontsize=9)
        ax.set(title=title, xlabel="Detector block column", ylabel="Detector block row")
        ax.set_xticks(range(shape[1]))
        ax.set_yticks(range(shape[0]))
    fig.colorbar(images[0], ax=axes[:2], label="Camera counts (shared 0-255 scale)", shrink=.75)
    fig.colorbar(images[2], ax=axes[2], label="Signed error (counts)", shrink=.75)
    fig.suptitle(f"{example['pattern_id']}: first saved held-out pattern; * marks clipping")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def heldout_error_plot(rows, path):
    ids = [r["detector"] for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained", sharex=True)
    axes[0].bar(ids, [r["rmse_counts"] if r["rmse_counts"] is not None else np.nan for r in rows], label="RMSE")
    axes[0].plot(ids, [r["mae_counts"] if r["mae_counts"] is not None else np.nan for r in rows], "o-", color="orange", label="MAE")
    axes[0].axhline(2, linestyle="--", color="red", label="RMSE pass threshold")
    axes[0].set(ylabel="Error (counts)", title="All held-out patterns; clipped observations excluded from error scores")
    axes[0].legend()
    usable = np.array([r["usable_for_score"] for r in rows])
    clipped = np.array([r["clipped_readings"] for r in rows])
    axes[1].bar(ids, usable, label="Usable for score", color="green")
    axes[1].bar(ids, clipped, bottom=usable, label="Clipped", color="red")
    axes[1].bar(ids, [r["unscorable_unclipped"] for r in rows], bottom=usable+clipped,
                label="Unclipped; fit unavailable", color="gray")
    axes[1].set(xlabel="Detector ID = TM row", ylabel="Held-out readings")
    axes[1].set_xticks(ids)
    axes[1].legend()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def slm_pattern_plot(pattern, path, pattern_id):
    from bench.geometry import slm_phase_pattern
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    image = ax.imshow(slm_phase_pattern(pattern), cmap="twilight", vmin=-np.pi, vmax=np.pi,
                      interpolation="nearest")
    for j in range(16):
        ax.text(2*(j % 4)+.5, 2*(j // 4)+.5, f"I{j}", ha="center", va="center", fontsize=8,
                bbox=dict(facecolor="white", alpha=.7, edgecolor="none"))
    ax.set(title=f"{pattern_id}: saved phase-only SLM input", xlabel="SLM pixel x", ylabel="SLM pixel y")
    fig.colorbar(image, ax=ax, label="Phase (radians)")
    fig.savefig(path, dpi=140)
    plt.close(fig)
