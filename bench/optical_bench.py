"""Small synthetic coherent bench. Only uint8 frames leave the optical model.

Private attributes are a workflow boundary, not a security sandbox. Never serialize
the bench object, its matrix, or its intermediate complex fields into agent state.
"""


import hashlib

import numpy as np

from bench.config import BenchConfig
from bench.acquisition import run_acquisition
from bench.matrices import load_detector_matrix, pin_detector_matrix, verify_matrix_source
from bench.geometry import detector_regions, geometry_metadata, slm_phase_pattern
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


class OpticalBench:
    def __init__(self, config: BenchConfig):
        config.require_current_layout()
        config = pin_detector_matrix(config)
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
        if config.detector_matrix_path is not None:
            rows, _ = load_detector_matrix(config)
            for row, (y0, y1, x0, x1) in zip(rows, self.regions):
                matrix[y0:y1, x0:x1] = row  # Exact supplied coefficients; no normalization.
        detector_rows = np.stack([matrix[y0, x0] for y0, _, x0, _ in self.regions])
        self.__metadata = {
            **geometry_metadata(),
            "matrix_mode": "known" if config.detector_matrix_path is not None else "seeded_random",
            "matrix_fingerprint": hashlib.sha256(detector_rows.astype("<c16").tobytes()).hexdigest(),
        }
        if config.detector_matrix_sha256 is not None:
            self.__metadata["matrix_source_sha256"] = config.detector_matrix_sha256
        self.__matrix = matrix

    @property
    def metadata(self):
        return dict(self.__metadata)

    def _evaluation_detector_rows(self, expected_fingerprint):
        """Evaluation-only access; never called by agents, acquisition or fitting."""
        if not expected_fingerprint or self.__metadata["matrix_fingerprint"] != expected_fingerprint:
            raise ValueError("Reconstructed truth does not match the run's recorded matrix fingerprint")
        return np.stack([self.__matrix[y0, x0].copy() for y0, _, x0, _ in self.regions])

    def frame(self, phases, shutter, power, *, capture_index=0):
        verify_matrix_source(self.config)
        phases = np.asarray(phases)
        if phases.shape != (self.config.input_regions,) or phases.dtype.kind not in "fiu" or not np.isfinite(phases).all():
            raise ValueError("One finite phase in radians per input region is required")
        if not self.config.shutter_min <= shutter <= self.config.shutter_max:
            raise ValueError("Shutter outside synthetic limits")
        if not self.config.power_min <= power <= self.config.power_max:
            raise ValueError("Laser power outside synthetic limits")
        # Each 2 x 2 SLM region has one phase and one effective unit-amplitude
        # input. TM coefficients integrate the four SLM pixels' contribution.
        active_slm = slm_phase_pattern(phases)
        inputs = np.exp(1j * active_slm[::2, ::2].ravel())
        return pattern_to_camera(self.__matrix, inputs, shutter_seconds=shutter,
            laser_power_mw=power, background=self.config.background_counts,
            noise_std=self.config.read_noise_std,
            rng=np.random.default_rng(np.random.SeedSequence([self.config.seed, capture_index, 901])))


class SimulatedBackend:
    """One bounded calibration + intensity-only sequence; no physical integration."""
    def __init__(self, config: BenchConfig):
        self.__bench = OpticalBench(config)
        self.config = self.__bench.config

    def describe(self):
        cfg = self.config
        cfg.require_current_layout()
        verify_matrix_source(cfg)
        def number(unit, low, high):
            return ParameterSpec(kind="number", unit=unit, verified=True, minimum=low, maximum=high)
        exact = ParameterSpec(kind="integer", unit="frame", verified=True,
                              minimum=cfg.frame_budget, maximum=cfg.frame_budget)
        prerequisites = {
            "synthetic bench configuration valid": Verification(verified=True,
                evidence="Bench settings, fixed geometry, and any pinned matrix source validated by the simulator; not laboratory verification"),
            "synthetic phase mapping verified": Verification(verified=True,
                evidence="Simulator accepts sixteen phase-in-radians controls with equal input amplitudes; no physical command calibration claimed"),
        }
        return HardwareCapabilities(backend_id="synthetic-optical-bench", revision=canonical_hash(cfg.model_dump()),
            available=True, data_origin="simulated", supports_deadlines=True, supports_cancellation=True,
            supports_cleanup=True, prerequisites=prerequisites,
            actions={"bench.acquire": ActionSpec(parameters={
                "shutter_seconds": number("s", cfg.shutter_min, cfg.shutter_max),
                "laser_power_mw": number("mW", cfg.power_min, cfg.power_max),
                "frame_budget": exact}, measurement_count_setting="frame_budget",
                required_prerequisites=list(prerequisites))})

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
