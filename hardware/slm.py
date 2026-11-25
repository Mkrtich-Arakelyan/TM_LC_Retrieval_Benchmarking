"""Single-image adapter for the supplied Meadowlark PCIe Blink C header.

No SDK loading occurs until ``connect()``. This is the eight-argument
Create_SDK / seven-argument Write_image ABI, not the HDMI SDK. Images are
already LUT-indexed uint8 gray levels, not floating-point phases.
See docs/meadowlark-slm.md for the interface evidence and lab checks.
"""

from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
import os
from pathlib import Path
from threading import Lock, RLock
from typing import Callable, Literal

import numpy as np

from hardware.bench_access import require_acquisition_allowed


class SlmError(RuntimeError):
    """SDK initialization or image transfer failed."""


@dataclass(frozen=True)
class SlmReturnCodes:
    """Exact success codes verified by the caller for the installed SDK.

    The supplied header declares int results but does not define their values;
    the matching public PCIe manual does not resolve that omission either.
    Deliberately no guessed default (in particular, no generic zero-is-success).
    """

    load_lut: int
    write_image: int
    image_complete: int

    def __post_init__(self):
        for name in ("load_lut", "write_image", "image_complete"):
            _integer(name, getattr(self, name), -(2**31), 2**31 - 1)


@dataclass(frozen=True)
class SlmSettings:
    dll_path: Path
    lut_path: Path
    board_number: int
    bit_depth: int
    is_nematic: bool
    ram_write_enabled: bool
    use_gpu: bool
    max_transient_frames: int
    wait_for_trigger: bool
    pulse_on_image_flip: bool
    pulse_on_refresh: bool
    timeout_ms: int
    image_size_convention: Literal["pixel_count", "width"]
    return_codes: SlmReturnCodes

    def __post_init__(self):
        object.__setattr__(self, "dll_path", Path(self.dll_path).resolve())
        object.__setattr__(self, "lut_path", Path(self.lut_path).resolve())
        _integer("board_number", self.board_number, 1, 2**31 - 1)
        _integer("bit_depth", self.bit_depth, 1, 2**32 - 1)
        _integer("max_transient_frames", self.max_transient_frames, 0, 2**31 - 1)
        _integer("timeout_ms", self.timeout_ms, 1, 2**32 - 1)
        for name in ("is_nematic", "ram_write_enabled", "use_gpu", "wait_for_trigger",
                     "pulse_on_image_flip", "pulse_on_refresh"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")
        if self.pulse_on_image_flip and self.pulse_on_refresh:
            raise ValueError("Only one output pulse option may be enabled")
        if self.image_size_convention not in ("pixel_count", "width"):
            raise ValueError("image_size_convention must be 'pixel_count' or 'width'")
        if not isinstance(self.return_codes, SlmReturnCodes):
            raise ValueError("return_codes must specify verified SDK success codes")


def _integer(name, value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


def _bind(sdk):
    """Use C int, including flags and constructed_ok, exactly as in the header."""
    signatures = {
        "Create_SDK": ([ct.c_uint, ct.POINTER(ct.c_uint), ct.POINTER(ct.c_int),
                        ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_char_p], None),
        "Delete_SDK": ([], None),
        "Get_last_error_message": ([], ct.c_char_p),
        "Load_LUT_file": ([ct.c_int, ct.c_char_p], ct.c_int),
        "Get_image_width": ([ct.c_int], ct.c_int),
        "Get_image_height": ([ct.c_int], ct.c_int),
        "Write_image": ([ct.c_int, ct.POINTER(ct.c_ubyte), ct.c_uint,
                         ct.c_int, ct.c_int, ct.c_int, ct.c_uint], ct.c_int),
        "ImageWriteComplete": ([ct.c_int, ct.c_uint], ct.c_int),
    }
    for name, (args, result) in signatures.items():
        function = getattr(sdk, name)
        function.argtypes = args
        function.restype = result


# The wrapper has process-global state, not one handle per adapter instance.
_sdk_owner = Lock()


class MeadowlarkSlm:
    """Explicit, synchronous, single-owner SDK lifecycle.

    ``sdk_loader`` is injectable for laptop tests and defaults to ctypes.CDLL
    (cdecl). ``display_image`` waits for ImageWriteComplete even with external
    triggering; another device must supply that trigger before the timeout.
    Completion means transfer readiness, not verified optical settling.
    """

    def __init__(self, settings: SlmSettings, *, sdk_loader: Callable | None = None):
        self.settings = settings
        self._loader = sdk_loader
        self._sdk = None
        self._initialized = False
        self._owns_sdk = False
        self._dll_directory = None
        self._release_error = None
        self._shape = None
        self._mutex = RLock()

    @property
    def image_shape(self) -> tuple[int, int]:
        """Queried (height, width); available only after successful connection."""
        with self._mutex:
            if self._shape is None:
                raise SlmError("SLM is not connected")
            return self._shape

    def connect(self) -> MeadowlarkSlm:
        """Load SDK, initialize, validate selected board, query size and load LUT."""
        with self._mutex:
            if self._release_error is not None:
                raise SlmError(self._release_error)
            if self._shape is not None:
                raise SlmError("SLM is already connected")
            cfg = self.settings
            for path in (cfg.dll_path, cfg.lut_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            # char* encoding is unspecified. Accept only ASCII until verified
            # on the lab PC, rather than silently sending a wrong path encoding.
            try:
                lut_name = str(cfg.lut_path).encode("ascii")
            except UnicodeEncodeError as exc:
                raise ValueError("Use an ASCII LUT path; SDK path encoding needs lab verification") from exc
            if not _sdk_owner.acquire(blocking=False):
                raise SlmError("Another SLM adapter owns the process-global Blink SDK")
            self._owns_sdk = True
            try:
                if self._loader is None:
                    if os.name != "nt":
                        raise SlmError("The physical Blink SDK requires Windows")
                    self._dll_directory = os.add_dll_directory(str(cfg.dll_path.parent))
                self._sdk = (self._loader or ct.CDLL)(str(cfg.dll_path))
                _bind(self._sdk)
                boards, constructed = ct.c_uint(0), ct.c_int(0)
                self._initialized = True  # release even a partially constructed SDK
                self._sdk.Create_SDK(cfg.bit_depth, ct.byref(boards), ct.byref(constructed),
                                     int(cfg.is_nematic), int(cfg.ram_write_enabled),
                                     int(cfg.use_gpu), cfg.max_transient_frames, None)
                if constructed.value == 0:
                    raise self._error("Create_SDK reported constructed_ok=0")
                if cfg.board_number > boards.value:
                    raise SlmError(f"Board {cfg.board_number} unavailable; SDK found {boards.value} board(s)")
                width = self._sdk.Get_image_width(cfg.board_number)
                height = self._sdk.Get_image_height(cfg.board_number)
                if width <= 0 or height <= 0 or width * height > 2**32 - 1:
                    raise self._error(f"Invalid SDK dimensions: {width} x {height}")
                if cfg.image_size_convention == "width" and (width, height) != (512, 512):
                    raise SlmError("The documented width image-size convention requires a 512 x 512 board")
                self._check("Load_LUT_file", self._sdk.Load_LUT_file(cfg.board_number, lut_name),
                            cfg.return_codes.load_lut)
                self._shape = (height, width)
            except BaseException as exc:
                try:
                    self.disconnect()
                except Exception as cleanup_error:
                    if hasattr(exc, "add_note"):
                        exc.add_note(f"SDK cleanup also failed: {cleanup_error}")
                raise
            return self

    def display_image(self, image: np.ndarray) -> None:
        """Display a C-contiguous (height, width) uint8 array without conversion.

        No flattening/casting, phase conversion, resizing, orientation correction,
        or wavefront correction is performed. A private copy keeps DMA storage
        stable for both SDK calls. Transfer errors close the SDK before retry.
        """
        with self._mutex:
            shape = self.image_shape
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
                raise ValueError("SLM image must be a NumPy uint8 array")
            if image.shape != shape:
                raise ValueError(f"SLM image shape must be {shape}; got {image.shape}")
            if not image.flags.c_contiguous:
                raise ValueError("SLM image must be C-contiguous in row-major order")
            buffer = image.copy(order="C")
            cfg = self.settings
            count = buffer.size if cfg.image_size_convention == "pixel_count" else shape[1]
            try:
                require_acquisition_allowed()
                self._check("Write_image", self._sdk.Write_image(
                    cfg.board_number, buffer.ctypes.data_as(ct.POINTER(ct.c_ubyte)), count,
                    int(cfg.wait_for_trigger), int(cfg.pulse_on_image_flip),
                    int(cfg.pulse_on_refresh), cfg.timeout_ms), cfg.return_codes.write_image)
                self._check("ImageWriteComplete", self._sdk.ImageWriteComplete(
                    cfg.board_number, cfg.timeout_ms), cfg.return_codes.image_complete)
            except BaseException as exc:
                try:
                    self.disconnect()
                except Exception as cleanup_error:
                    if hasattr(exc, "add_note"):
                        exc.add_note(f"SDK cleanup also failed: {cleanup_error}")
                raise

    def _check(self, name, result, expected):
        if result != expected:
            raise self._error(f"{name} returned {result}; expected verified success code {expected}")

    def _error(self, message):
        try:
            detail = self._sdk.Get_last_error_message()
            if detail:
                message += ": " + detail.decode("ascii", errors="backslashreplace")
        except Exception:
            pass  # diagnostics must not obscure the original failure
        return SlmError(message)

    def disconnect(self) -> None:
        """Release SDK once; safe before connection and after setup failure."""
        with self._mutex:
            if self._release_error is not None:
                raise SlmError(self._release_error)
            try:
                try:
                    if self._initialized:
                        self._initialized = False
                        self._sdk.Delete_SDK()
                finally:
                    self._shape = None
                    self._sdk = None
                    try:
                        if self._dll_directory is not None:
                            self._dll_directory.close()
                    finally:
                        self._dll_directory = None
                        if self._owns_sdk:
                            self._owns_sdk = False
                            _sdk_owner.release()
            except BaseException as exc:
                self._release_error = f"SLM release failed: {exc}"
                raise

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.disconnect()
