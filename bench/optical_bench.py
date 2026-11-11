"""Small synthetic coherent bench. Only uint8 frames leave the optical model.

Private attributes are a workflow boundary, not a security sandbox. Never serialize
the bench object, its matrix, or its intermediate complex fields into agent state.
"""


import numpy as np

from agents.experimentor import UNRESOLVED_PREREQUISITES
from bench.config import BenchConfig
from bench.acquisition import run_acquisition
from tools.hardware import (ActionReport, ActionSpec, HardwareCapabilities,
                            ParameterSpec, Verification, canonical_hash, parameter_issues)

CAMERA_GAIN = 4000.0  # counts / (synthetic field intensity * seconds * mW)


def pattern_to_camera(matrix, inputs, *, shutter_seconds, laser_power_mw,
                      gain=CAMERA_GAIN, background=0.0, noise_std=0.0, rng=None):
    """Shared Runner/physics-check kernel: coherent sum BEFORE square, then camera.

    The generic kernel accepts masked inputs for the isolated interference fixture;
    the acquisition API below accepts only equal-amplitude phase controls.
    """
    field = np.einsum("...d,d->...", matrix, inputs)
    response = gain * shutter_seconds * laser_power_mw * np.abs(field) ** 2 + background
    if noise_std:
        if rng is None:
            raise ValueError("Noise requires an explicit seeded random generator")
        response = response + rng.normal(0, noise_std, response.shape)
    return np.clip(np.rint(response), 0, 255).astype(np.uint8)


def detector_regions(side):
    return [(2, 4, 2, 4), (2, 4, side - 4, side - 2),
            (side - 4, side - 2, 2, 4), (side - 4, side - 2, side - 4, side - 2)]


class OpticalBench:
    def __init__(self, config: BenchConfig):
        self.config = config
        self.regions = detector_regions(config.frame_side)
        rng = np.random.default_rng(config.seed)
        shape = (config.frame_side, config.frame_side, config.input_regions)
        matrix = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2 * config.input_regions)
        matrix *= np.exp(rng.normal(0, 1.0, shape[:2]))[..., None]
        # Fixed scene normalization defines synthetic brightness, independently of
        # configured operating limits. At nominal 0.01 s * 1 mW ~4% saturates.
        calibration_intensity = np.abs(matrix.sum(axis=-1)) ** 2
        matrix *= np.sqrt(254.5 / (CAMERA_GAIN * 0.01 * np.quantile(calibration_intensity, .96)))
        for y0, y1, x0, x1 in self.regions:
            row = rng.normal(size=config.input_regions) + 1j * rng.normal(size=config.input_regions)
            row *= np.sqrt(220 / (CAMERA_GAIN * .01)) / np.abs(row).sum()
            # Uniform coherent response within each ROI: its mean is a rank-one
            # intensity channel. Arbitrary pixel mixtures need a different model.
            matrix[y0:y1, x0:x1] = row
        self.__matrix = matrix

    def frame(self, phases, shutter, power, *, capture_index=0):
        phases = np.asarray(phases)
        if phases.shape != (self.config.input_regions,) or phases.dtype.kind not in "fiu" or not np.isfinite(phases).all():
            raise ValueError("One finite phase in radians per input region is required")
        if not self.config.shutter_min <= shutter <= self.config.shutter_max:
            raise ValueError("Shutter outside synthetic limits")
        if not self.config.power_min <= power <= self.config.power_max:
            raise ValueError("Laser power outside synthetic limits")
        return pattern_to_camera(self.__matrix, np.exp(1j * phases), shutter_seconds=shutter,
            laser_power_mw=power, background=self.config.background_counts,
            noise_std=self.config.read_noise_std,
            rng=np.random.default_rng(np.random.SeedSequence([self.config.seed, capture_index, 901])))


class SimulatedBackend:
    """One bounded calibration + intensity-only sequence; no physical integration."""
    def __init__(self, config: BenchConfig):
        self.config = config
        self.__bench = OpticalBench(config)

    def describe(self):
        cfg = self.config
        def number(unit, low, high):
            return ParameterSpec(kind="number", unit=unit, verified=True, minimum=low, maximum=high)
        exact = ParameterSpec(kind="integer", unit="frame", verified=True,
                              minimum=cfg.frame_budget, maximum=cfg.frame_budget)
        prerequisites = {s: Verification(verified=True, evidence="Synthetic model contract only; no laboratory verification")
                         for s in [*UNRESOLVED_PREREQUISITES, "synthetic phase mapping verified"]}
        return HardwareCapabilities(backend_id="synthetic-optical-bench", revision=canonical_hash(cfg.model_dump()),
            available=True, data_origin="simulated", supports_deadlines=True, supports_cancellation=True,
            supports_cleanup=True, prerequisites=prerequisites,
            actions={"bench.acquire": ActionSpec(parameters={
                "shutter_seconds": number("s", cfg.shutter_min, cfg.shutter_max),
                "laser_power_mw": number("mW", cfg.power_min, cfg.power_max),
                "frame_budget": exact}, measurement_count_setting="frame_budget")})

    def validate(self, procedure):
        if len(procedure.actions) != 1 or procedure.actions[0].action != "bench.acquire":
            return ["Reference bench procedure requires exactly one bench.acquire action"]
        return []

    def cleanup(self, *, reason, deadline):
        return "Synthetic sequence stopped; no physical equipment was selected"

    def execute(self, action, *, output_dir, input_artifacts, capability_fingerprint, deadline, cancel_event):
        cfg = self.config
        caps = self.describe()
        if capability_fingerprint != canonical_hash(caps.model_dump(mode="json")):
            return ActionReport(errors=["Synthetic configuration changed after review"])
        if action.action != "bench.acquire":
            return ActionReport(errors=["Unsupported synthetic action"])
        issues = parameter_issues(action.requested_settings, caps.actions[action.action])
        if issues:
            return ActionReport(errors=issues)
        return run_acquisition(self.__bench, cfg, action, output_dir=output_dir,
                               deadline=deadline, cancel_event=cancel_event, camera_gain=CAMERA_GAIN)
