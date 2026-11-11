"""Standalone inspection plots, deliberately separate from optical calculations."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def calibration_plot(attempts, path):
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.axhspan(3, 5, color="green", alpha=.18, label="Target 3–5%")
    ax.plot([a["attempt"] for a in attempts], [100*a["whole_frame_saturation"] for a in attempts], "o-", label="Whole raw frame")
    ax.set(xlabel="Calibration attempt (settings in calibration.csv)", ylabel="Pixels at 255 (%)", title="Synthetic camera saturation calibration")
    ax.legend()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def frame_preview(frame, regions, path, title):
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    image = ax.imshow(frame, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    y, x = np.where(frame == 255)
    ax.scatter(x, y, marker="x", s=22, color="red", label="Saturated (255)")
    for i, (y0, y1, x0, x1) in enumerate(regions):
        ax.add_patch(Rectangle((x0-.5, y0-.5), x1-x0, y1-y0, fill=False, edgecolor="cyan", linewidth=1.5))
        ax.text(x0-.5, y0-1, f"D{i}", color="cyan", fontsize=9)
    ax.set(title=f"Raw frame {title} calibration", xlabel="Camera x", ylabel="Camera y")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(image, ax=ax, label="Raw camera counts (fixed 0–255)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def prediction_plot(measured, predicted, clipped, path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    for j, ax in enumerate(axes.flat):
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
