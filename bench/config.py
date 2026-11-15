"""Synthetic settings, independent of laboratory specifications."""

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator
from bench.geometry import FRAME_SIDE, INPUT_COUNT, GEOMETRY_VERSION


class BenchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    seed: int = Field(default=17, ge=0, le=4294967295)
    input_regions: int = Field(default=16, ge=2, le=32)
    frame_side: int = Field(default=20, ge=12, le=40)
    layout_version: int = Field(default=GEOMETRY_VERSION)
    shutter_seconds: float = Field(default=0.01, gt=0)
    shutter_min: float = Field(default=0.0001, gt=0)
    shutter_max: float = Field(default=0.1, gt=0)
    laser_power_mw: float = Field(default=1.0, gt=0)
    power_min: float = Field(default=0.1, gt=0)
    power_max: float = Field(default=2.0, gt=0)
    background_counts: float = Field(default=0.0, ge=0, le=100)
    read_noise_std: float = Field(default=0.0, ge=0, le=20)
    fit_patterns: int = Field(default=512, ge=128, le=2048)
    holdout_patterns: int = Field(default=48, ge=8, le=256)
    calibration_attempts: int = Field(default=32, ge=4, le=64)
    detector_matrix_path: str | None = Field(default=None, min_length=1)
    detector_matrix_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_serializer(mode="wrap")
    def serialize_settings(self, handler):
        # Omit optional source identity in random mode.
        settings = handler(self)
        for key in ("detector_matrix_path", "detector_matrix_sha256"):
            if settings.get(key) is None:
                settings.pop(key, None)
        return settings

    @model_validator(mode="after")
    def ordered_limits(self):
        if self.detector_matrix_sha256 and not self.detector_matrix_path:
            raise ValueError("A detector matrix fingerprint requires detector_matrix_path")
        if self.detector_matrix_path is not None and not self.detector_matrix_path.strip():
            raise ValueError("detector_matrix_path must not be blank")
        if not self.shutter_min <= self.shutter_seconds <= self.shutter_max:
            raise ValueError("Initial shutter must be inside ordered shutter limits")
        if not self.power_min <= self.laser_power_mw <= self.power_max:
            raise ValueError("Initial power must be inside ordered power limits")
        if self.fit_patterns < 8 * self.input_regions:
            raise ValueError("Reference procedure needs at least 8 fit patterns per input region")
        return self

    @property
    def frame_budget(self):
        return self.fit_patterns + self.holdout_patterns + self.calibration_attempts

    def require_current_layout(self):
        # Older dimensions can still be parsed for measurement-only reanalysis.
        if self.input_regions != INPUT_COUNT or self.frame_side != FRAME_SIDE or self.layout_version != GEOMETRY_VERSION:
            raise ValueError("New simulated runs require layout version 2: 16 phase inputs, a 20 x 20 frame, and sixteen 2 x 2 detectors in the centered 8 x 8 crop")
