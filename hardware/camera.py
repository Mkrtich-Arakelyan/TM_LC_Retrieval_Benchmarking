"""Raw Mono8 ALKERIA Celera adapter using the lab's alkusb3 property API.

Import and construction do not import alkusb3 or connect a camera. No SLM
synchronization, pixel processing or guessed SDK release/timeout API is used.
See docs/alkeria-camera.md for the assumptions requiring bench verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import math
from threading import RLock
from typing import Callable

import numpy as np

from hardware.bench_access import require_acquisition_allowed


class CameraError(RuntimeError):
    """Camera configuration, frame transfer or acquisition stop failed."""


def _integer(name, value, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class CameraTrigger:
    """Acquisition-start trigger, using the external source shown by the lab.

    Disabled means free running. Delay and debounce remain in native SDK units.
    Other trigger sources and polarity controls are not established by the lab
    Python examples and are not implemented here.
    """

    enabled: bool
    external_input: int
    detect_external_input_edge: bool
    delay: int
    debounce_time: int

    def __post_init__(self):
        for name in ("enabled", "detect_external_input_edge"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")
        for name in ("external_input", "delay", "debounce_time"):
            _integer(name, getattr(self, name), 0)


@dataclass(frozen=True)
class CameraSettings:
    """Caller settings; ROI values map to SDK image_size/start_x/y before binning.

    Actual frame dimensions are read back after configuration, not inferred
    by dividing the ROI by binning factors. No lab defaults are hardcoded.
    """

    camera_index: int
    roi_width: int
    roi_height: int
    roi_x: int
    roi_y: int
    horizontal_binning: int
    vertical_binning: int
    video_mode: int
    pixel_format: str
    shutter: int
    frame_rate: float
    trigger: CameraTrigger

    def __post_init__(self):
        for name in ("camera_index", "roi_x", "roi_y", "video_mode"):
            _integer(name, getattr(self, name), 0)
        for name in ("roi_width", "roi_height", "horizontal_binning", "vertical_binning"):
            _integer(name, getattr(self, name), 1)
        if self.pixel_format != "mono8":
            raise ValueError("Only pixel_format='mono8' is supported")
        if type(self.shutter) is not int or not 2000 <= self.shutter <= 15000:
            raise ValueError("shutter must be an integer in [2000, 15000] native SDK units")
        if (type(self.frame_rate) not in (int, float)
                or not math.isfinite(self.frame_rate) or self.frame_rate <= 0):
            raise ValueError("frame_rate must be positive and finite")
        if not isinstance(self.trigger, CameraTrigger):
            raise ValueError("trigger must be CameraTrigger settings")


class CeleraCamera:
    """Explicit connect / capture_frame / disconnect lifecycle.

    ``sdk_loader`` returns the alkusb3 module or a fake with the same interface.
    Without injection it is imported only by connect(). Property setter and
    get_raw_data exceptions are propagated as CameraError with their cause.

    capture_frame() calls get_raw_data() without arguments, as in the lab code.
    It can block waiting for a frame/trigger. Timeout support in the installed
    Python binding is unverified; there is no promised timeout or cancellation.
    A simultaneous disconnect cannot interrupt a blocked call under this lock.
    """

    def __init__(self, settings: CameraSettings, *, sdk_loader: Callable | None = None):
        if not isinstance(settings, CameraSettings):
            raise ValueError("settings must be CameraSettings")
        self.settings = settings
        self._loader = sdk_loader
        self._camera = None
        self._shape = None
        self._mutex = RLock()

    @property
    def image_shape(self) -> tuple[int, int]:
        """Post-configuration SDK dimensions as (height, width)."""
        with self._mutex:
            if self._shape is None:
                raise CameraError("Camera is not connected")
            return self._shape

    def connect(self) -> CeleraCamera:
        """Select camera, configure while stopped, then start acquisition."""
        with self._mutex:
            if self._camera is not None:
                raise CameraError("Camera is already open; disconnect before reconnecting")
            stage = "loading alkusb3"
            try:
                sdk = self._loader() if self._loader else import_module("alkusb3")
                stage = "constructing celera_camera"
                self._camera = sdk.celera_camera()
                cam, cfg = self._camera, self.settings
                stage = "selecting camera"
                cam.camera = cfg.camera_index
                stage = "stopping acquisition before setup"
                cam.acquire = False
                stage = "configuring video mode, ROI, binning and Mono8"
                cam.video_mode = cfg.video_mode
                cam.image_size_x = cfg.roi_width
                cam.image_size_y = cfg.roi_height
                cam.image_start_x = cfg.roi_x
                cam.image_start_y = cfg.roi_y
                cam.horizontal_binning = cfg.horizontal_binning
                cam.vertical_binning = cfg.vertical_binning
                cam.color_coding = sdk.color_coding.mono8
                stage = "configuring shutter and frame rate"
                cam.shutter = cfg.shutter
                cam.frame_rate = cfg.frame_rate
                stage = "configuring acquisition-start trigger"
                trigger = cam.acquisition_start_trigger
                trigger.enabled = False
                if cfg.trigger.enabled:
                    trigger.source = sdk.trigger_source.external
                    trigger.external_input = cfg.trigger.external_input
                    trigger.detect_external_input_edge = cfg.trigger.detect_external_input_edge
                    trigger.delay = cfg.trigger.delay
                    cam.pio_ports[cfg.trigger.external_input].debounce_time = cfg.trigger.debounce_time
                    trigger.enabled = True
                stage = "checking configured format and dimensions"
                if cam.color_coding != sdk.color_coding.mono8:
                    raise CameraError("Camera did not accept Mono8 pixel format")
                if cam.shutter != cfg.shutter:
                    raise CameraError(f"Camera shutter readback {cam.shutter} differs from requested {cfg.shutter}")
                width, height = cam.image_size_x, cam.image_size_y
                if (type(width) is not int or type(height) is not int
                        or width <= 0 or height <= 0):
                    raise CameraError(f"Invalid camera dimensions: {width!r} x {height!r}")
                stage = "starting acquisition"
                require_acquisition_allowed()
                cam.acquire = True
                self._shape = (height, width)
            except BaseException as exc:
                self._cleanup_error(exc)
                if isinstance(exc, (KeyboardInterrupt, SystemExit, CameraError)):
                    raise
                raise CameraError(f"Camera setup failed while {stage}: {exc}") from exc
            return self

    def capture_frame(self) -> np.ndarray:
        """Read one tightly packed Mono8 frame, preserving values and byte order.

        Require exactly height * width bytes. Return an independent uint8 array;
        do not average, normalize, take square roots or transpose the pixels.
        Capture/decoding errors stop acquisition and require explicit reconnect.
        """
        with self._mutex:
            height, width = self.image_shape
            try:
                raw = self._camera.get_raw_data()
                view = memoryview(raw)
                if not view.c_contiguous or view.itemsize != 1:
                    raise CameraError("Mono8 raw data must be a contiguous byte buffer")
                expected = height * width
                if view.nbytes != expected:
                    raise CameraError(f"Mono8 frame has {view.nbytes} bytes; expected {expected} "
                                      f"for (height, width)=({height}, {width})")
                return np.frombuffer(view, dtype=np.uint8).reshape(height, width).copy()
            except BaseException as exc:
                self._cleanup_error(exc)
                if isinstance(exc, (KeyboardInterrupt, SystemExit, CameraError)):
                    raise
                raise CameraError(f"Camera frame capture failed: {exc}") from exc

    def _cleanup_error(self, original):
        try:
            self.disconnect()
        except Exception as cleanup_error:
            if hasattr(original, "add_note"):
                original.add_note(f"Stopping camera acquisition also failed: {cleanup_error}")

    def disconnect(self) -> None:
        """Stop acquisition, including partially configured cameras; repeatable.

        No additional release method is established for this Python API. If
        stopping fails, retain the camera reference so disconnect can be retried.
        """
        with self._mutex:
            self._shape = None
            if self._camera is not None:
                try:
                    self._camera.acquire = False
                except Exception as exc:
                    raise CameraError(f"Could not stop camera acquisition: {exc}") from exc
                self._camera = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, traceback):
        if exc is None:
            self.disconnect()
        else:
            self._cleanup_error(exc)
