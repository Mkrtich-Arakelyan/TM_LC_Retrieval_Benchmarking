"""Synthetic settings, independent of laboratory specifications."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BenchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    seed: int = Field(default=17, ge=0, le=4294967295)
    input_regions: int = Field(default=16, ge=2, le=32)
    frame_side: int = Field(default=20, ge=12, le=40)
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

    @model_validator(mode="after")
    def ordered_limits(self):
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
