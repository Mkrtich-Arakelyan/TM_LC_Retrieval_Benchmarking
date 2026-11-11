"""Deterministic coherent interference check, without agents or LLM calls."""

import argparse
import csv
from pathlib import Path

import numpy as np

from bench.optical_bench import pattern_to_camera
from bench.acquisition import write_json


def run_check(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    phases = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    # Independent analytic identity |1 + exp(i phi)|² = 2 + 2 cos(phi).
    # These literal checkpoints are also independent of simulator helpers.
    expected = np.array([160, 80, 0, 80, 160])
    matrix = np.array([[[1+0j, 1+0j]]])
    settings = dict(shutter_seconds=.01, laser_power_mw=1., gain=4000., background=0., noise_std=0.)
    one_alone = int(pattern_to_camera(matrix, np.array([1, 0]), **settings)[0, 0])
    measured = np.array([int(pattern_to_camera(matrix, np.array([1, np.exp(1j*phase)]), **settings)[0, 0]) for phase in phases])
    assert one_alone == 40, one_alone
    assert np.array_equal(measured, expected), (measured, expected)
    assert np.max(measured) < 255
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curve = np.linspace(0, 2*np.pi, 401)
    analytic = 80 * (1 + np.cos(curve))
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(curve/np.pi, analytic, label="Independent analytic: 80(1 + cos φ)")
    ax.scatter(phases/np.pi, measured, marker="x", color="red", s=65, label="Shared camera kernel")
    ax.set(xlabel="Relative phase / π", ylabel="Raw camera counts", title="Two equal coherent contributions; one alone = 40")
    ax.legend()
    fig.savefig(directory / "interference-check.png", dpi=150)
    plt.close(fig)
    with (directory / "interference-check.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["relative_phase_radians", "expected_counts", "simulated_counts"])
        writer.writerows(zip(phases.tolist(), expected.tolist(), measured.tolist()))
    result = {"passed": True, "one_input_alone": one_alone, "expected": expected.tolist(),
              "simulated": measured.tolist(), "settings": settings, "llm_calls": 0}
    write_json(directory / "interference-check.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/interference-check"))
    args = parser.parse_args()
    result = run_check(args.output.resolve())
    print(f"Interference check passed: {result['simulated']}; {args.output.resolve()}")


if __name__ == "__main__":
    main()
