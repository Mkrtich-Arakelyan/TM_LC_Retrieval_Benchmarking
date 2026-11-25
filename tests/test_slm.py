"""Laptop-only ABI, buffer and lifecycle tests; no vendor DLL is loaded."""

import ctypes as ct
from dataclasses import replace
import subprocess
import sys

import numpy as np
import pytest

from hardware.slm import MeadowlarkSlm, SlmError, SlmReturnCodes, SlmSettings


class Function:
    def __init__(self, implementation):
        self.implementation = implementation

    def __call__(self, *args):
        return self.implementation(*args)


class FakeSdk:
    def __init__(self, *, shape=(3, 4), boards=2, constructed=1):
        self.calls = []
        # Deliberately distinct test-only codes: never assume zero/one uniformly.
        self.results = {"Load_LUT_file": 11, "Write_image": 22, "ImageWriteComplete": 33}

        def create(*args):
            self.calls.append(("Create_SDK", args))
            ct.cast(args[1], ct.POINTER(ct.c_uint))[0] = boards
            ct.cast(args[2], ct.POINTER(ct.c_int))[0] = constructed

        def invoke(name, args):
            self.calls.append((name, args))
            if name == "Write_image":
                self.frame = np.ctypeslib.as_array(args[1], shape=(shape[0] * shape[1],)).copy()
            return self.results.get(name)

        self.Create_SDK = Function(create)
        self.Get_image_height = Function(lambda board: shape[0])
        self.Get_image_width = Function(lambda board: shape[1])
        self.Get_last_error_message = Function(lambda: b"fake SDK diagnostic")
        for name in ("Load_LUT_file", "Write_image", "ImageWriteComplete", "Delete_SDK"):
            setattr(self, name, Function(lambda *args, name=name: invoke(name, args)))


@pytest.fixture
def settings(tmp_path):
    dll, lut = tmp_path / "Blink_C_wrapper.dll", tmp_path / "device.LUT"
    dll.touch()
    lut.touch()
    return SlmSettings(dll, lut, board_number=2, bit_depth=12, is_nematic=True,
                       ram_write_enabled=True, use_gpu=False, max_transient_frames=20,
                       wait_for_trigger=False, pulse_on_image_flip=True, pulse_on_refresh=False,
                       timeout_ms=5000, image_size_convention="pixel_count",
                       return_codes=SlmReturnCodes(11, 22, 33))


def adapter(settings, fake):
    return MeadowlarkSlm(settings, sdk_loader=lambda path: fake)


def names(fake):
    return [name for name, _ in fake.calls]


def test_release_failure_cannot_become_a_successful_noop(settings):
    fake = FakeSdk()
    slm = adapter(settings, fake)
    slm.connect()
    calls = []

    def fail_release():
        calls.append("delete")
        raise OSError("fake Delete_SDK error")

    fake.Delete_SDK = Function(fail_release)
    with pytest.raises(OSError, match="Delete_SDK"):
        slm.disconnect()
    with pytest.raises(SlmError, match="release failed"):
        slm.disconnect()
    assert calls == ["delete"]


def test_import_and_construction_do_not_load_sdk(settings):
    # A fresh interpreter, rather than importing an already cached test module.
    script = """
import ctypes
import numpy
def forbidden(*args, **kwargs):
    raise AssertionError('Import or construction loaded a DLL')
ctypes.CDLL = forbidden
from hardware.slm import MeadowlarkSlm
MeadowlarkSlm(None)
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("trigger", [False, True])
def test_exact_abi_board_lut_frame_and_completion(settings, trigger):
    fake = FakeSdk()
    settings = replace(settings, wait_for_trigger=trigger)
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    with adapter(settings, fake) as slm:
        assert slm.image_shape == (3, 4)
        assert fake.Create_SDK.restype is None
        assert fake.Create_SDK.argtypes == [ct.c_uint, ct.POINTER(ct.c_uint),
            ct.POINTER(ct.c_int), ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_char_p]
        assert fake.Write_image.argtypes == [ct.c_int, ct.POINTER(ct.c_ubyte), ct.c_uint,
                                            ct.c_int, ct.c_int, ct.c_int, ct.c_uint]
        assert fake.Load_LUT_file.restype is ct.c_int
        create = fake.calls[0][1]
        assert (create[0], *create[3:]) == (12, 1, 1, 0, 20, None)
        assert fake.calls[1] == ("Load_LUT_file", (2, str(settings.lut_path).encode("ascii")))
        slm.display_image(image)
        np.testing.assert_array_equal(fake.frame, image.ravel())
        write = next(args for name, args in fake.calls if name == "Write_image")
        assert (write[0], *write[2:]) == (2, 12, int(trigger), 1, 0, 5000)
        assert fake.calls[-1] == ("ImageWriteComplete", (2, 5000))
    slm.disconnect()
    assert names(fake) == ["Create_SDK", "Load_LUT_file", "Write_image", "ImageWriteComplete", "Delete_SDK"]
    with pytest.raises(SlmError, match="not connected"):
        slm.display_image(image)


@pytest.mark.parametrize("image", [np.zeros((3, 4)), np.zeros((3, 4), dtype=np.uint16),
    np.zeros((4, 3), dtype=np.uint8), np.zeros(12, dtype=np.uint8), [[0] * 4] * 3,
    np.zeros((3, 8), dtype=np.uint8)[:, ::2], np.zeros((4, 3), dtype=np.uint8).T])
def test_rejects_invalid_buffer_before_write(settings, image):
    fake = FakeSdk()
    with adapter(settings, fake) as slm:
        with pytest.raises(ValueError):
            slm.display_image(image)
        assert "Write_image" not in names(fake)


@pytest.mark.parametrize("stage", ["Load_LUT_file", "Write_image", "ImageWriteComplete"])
@pytest.mark.parametrize("code", [0, -1, 1])
def test_checks_each_result_and_cleans_up(settings, stage, code):
    fake = FakeSdk()
    fake.results[stage] = code
    slm = adapter(settings, fake)
    with pytest.raises(SlmError, match=rf"{stage} returned {code}.*fake SDK diagnostic"):
        with slm:
            slm.display_image(np.zeros((3, 4), dtype=np.uint8))
    assert names(fake).count("Delete_SDK") == 1
    if stage == "Write_image":
        assert "ImageWriteComplete" not in names(fake)


@pytest.mark.parametrize("kwargs, message", [({"constructed": 0}, "constructed_ok=0"),
    ({"boards": 1}, "Board 2 unavailable"), ({"boards": 0}, "found 0"),
    ({"shape": (0, 4)}, "Invalid SDK dimensions"), ({"shape": (3, -1)}, "Invalid SDK dimensions")])
def test_setup_failures_release_sdk(settings, kwargs, message):
    fake = FakeSdk(**kwargs)
    with pytest.raises(SlmError, match=message):
        adapter(settings, fake).connect()
    assert names(fake).count("Delete_SDK") == 1
    assert "Write_image" not in names(fake)


def test_single_global_sdk_owner(settings):
    first, second = FakeSdk(), FakeSdk()
    with adapter(settings, first):
        with pytest.raises(SlmError, match="owns"):
            adapter(settings, second).connect()
        assert not second.calls
    with adapter(settings, second):
        pass


def test_missing_file_does_not_initialize(settings):
    settings.lut_path.unlink()
    fake = FakeSdk()
    with pytest.raises(FileNotFoundError):
        adapter(settings, fake).connect()
    assert not fake.calls


def test_missing_symbol_does_not_construct_and_releases_owner(settings):
    fake = FakeSdk()
    del fake.ImageWriteComplete
    with pytest.raises(AttributeError):
        adapter(settings, fake).connect()
    assert not fake.calls
    with adapter(settings, FakeSdk()):
        pass


@pytest.mark.parametrize("stage", ["Create_SDK", "Get_image_height", "Write_image"])
def test_python_exception_after_initialization_releases_sdk(settings, stage):
    fake = FakeSdk()
    def fail(*args):
        raise OSError("test SDK exception")
    setattr(fake, stage, Function(fail))
    with pytest.raises(OSError, match="test SDK exception"):
        with adapter(settings, fake) as slm:
            slm.display_image(np.zeros((3, 4), dtype=np.uint8))
    assert names(fake).count("Delete_SDK") == 1


def test_ctypes_call_boundary_with_c_callbacks(settings):
    fake = FakeSdk()
    # Use actual ctypes conversions as well as Python spies. The C callbacks
    # retain their lifetime on fake and never load an external/native DLL.
    fake.Create_SDK = ct.CFUNCTYPE(None, ct.c_uint, ct.POINTER(ct.c_uint),
        ct.POINTER(ct.c_int), ct.c_int, ct.c_int, ct.c_int, ct.c_int,
        ct.c_char_p)(fake.Create_SDK.implementation)
    fake.Write_image = ct.CFUNCTYPE(ct.c_int, ct.c_int, ct.POINTER(ct.c_ubyte),
        ct.c_uint, ct.c_int, ct.c_int, ct.c_int, ct.c_uint)(fake.Write_image.implementation)
    with adapter(settings, fake) as slm:
        image = np.arange(12, dtype=np.uint8).reshape(3, 4)
        slm.display_image(image)
        np.testing.assert_array_equal(fake.frame, image.ravel())


def test_default_loader_is_lazy_and_retains_dependency_directory(monkeypatch, settings):
    import hardware.slm as module
    fake = FakeSdk()
    events = []
    class Directory:
        def close(self):
            events.append("close directory")
    def directory(path):
        events.append(("directory", path))
        return Directory()
    def load(path):
        events.append(("load", path))
        return fake
    monkeypatch.setattr(module.os, "add_dll_directory", directory)
    monkeypatch.setattr(module.ct, "CDLL", load)
    slm = MeadowlarkSlm(settings)
    assert not events
    with slm:
        assert events == [("directory", str(settings.dll_path.parent)), ("load", str(settings.dll_path))]
    assert events[-1] == "close directory"


def test_loader_failure_releases_owner(settings):
    def fail(path):
        raise OSError("missing SDK dependency")
    with pytest.raises(OSError, match="missing SDK dependency"):
        MeadowlarkSlm(settings, sdk_loader=fail).connect()
    with adapter(settings, FakeSdk()):
        pass


def test_width_convention_is_explicit_and_limited_to_documented_board(settings):
    settings = replace(settings, image_size_convention="width")
    fake = FakeSdk(shape=(512, 512))
    with adapter(settings, fake) as slm:
        slm.display_image(np.zeros((512, 512), dtype=np.uint8))
        assert next(args[2] for name, args in fake.calls if name == "Write_image") == 512
    with pytest.raises(SlmError, match="512 x 512"):
        adapter(settings, FakeSdk()).connect()


@pytest.mark.parametrize("changes", [{"board_number": 0}, {"timeout_ms": -1},
    {"bit_depth": True}, {"max_transient_frames": 2**31}, {"wait_for_trigger": 1},
    {"pulse_on_refresh": True}, {"image_size_convention": "guess"}, {"return_codes": None}])
def test_invalid_settings(settings, changes):
    with pytest.raises(ValueError):
        replace(settings, **changes)
