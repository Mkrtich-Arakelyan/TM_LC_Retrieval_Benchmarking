"""Camera-only laptop tests. No alkusb3 installation or device is needed."""

from dataclasses import replace
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hardware.camera import CameraError, CameraSettings, CameraTrigger, CeleraCamera


class Properties:
    def __init__(self, events, name, **values):
        self.__dict__.update(events=events, name=name, values=values)

    def __getattr__(self, key):
        return self.values[key]

    def __setattr__(self, key, value):
        assert key in self.values, f"Unknown SDK property: {key}"
        self.events.append((f"{self.name}.{key}", value))
        self.values[key] = value


class FakeCamera:
    def __init__(self):
        self.__dict__.update(events=[], values={
            "camera": 0, "acquire": False, "video_mode": 0,
            "image_size_x": 4, "image_size_y": 3, "image_start_x": 0, "image_start_y": 0,
            "horizontal_binning": 1, "vertical_binning": 1, "color_coding": None,
            "shutter": 0, "frame_rate": 0,
        }, fail_property=None, fail_stop=False, raw=bytearray(range(12)), reads=0,
            capture_error=None, shutter_override=None)
        self.__dict__["acquisition_start_trigger"] = Properties(
            self.events, "trigger", enabled=True, source=None, external_input=0,
            detect_external_input_edge=False, delay=0)
        self.__dict__["pio_ports"] = [Properties(self.events, f"pio{i}", debounce_time=0) for i in range(3)]

    def __getattr__(self, key):
        value = self.values[key]
        if key == "image_size_x":
            return value // self.values["horizontal_binning"]
        if key == "image_size_y":
            return value // self.values["vertical_binning"]
        if key == "shutter" and self.shutter_override is not None:
            return self.shutter_override
        return value

    def __setattr__(self, key, value):
        if key in self.values:
            self.events.append((key, value))
            self.values[key] = value
            if key == self.fail_property or (key == "acquire" and value is False and self.fail_stop):
                raise OSError(f"fake failure setting {key}")
        else:
            assert key in {"raw", "fail_property", "capture_error", "reads", "fail_stop", "shutter_override"}
            self.__dict__[key] = value

    def get_raw_data(self):
        self.reads += 1
        if self.capture_error:
            raise self.capture_error
        return self.raw

    def close(self):
        pytest.fail("Undocumented close() call")

    def release(self):
        pytest.fail("Undocumented release() call")


@pytest.fixture
def settings():
    return CameraSettings(camera_index=2, roi_width=4, roi_height=3, roi_x=8, roi_y=6,
        horizontal_binning=1, vertical_binning=1, video_mode=0, pixel_format="mono8",
        shutter=6000, frame_rate=60, trigger=CameraTrigger(True, 1, True, 3000, 25))


def adapter(settings, fake):
    sdk = SimpleNamespace(celera_camera=lambda: fake,
        color_coding=SimpleNamespace(mono8="Mono8 SDK enum"),
        trigger_source=SimpleNamespace(external="external SDK enum"))
    return CeleraCamera(settings, sdk_loader=lambda: sdk)


def test_import_and_construction_are_lazy_in_fresh_process():
    script = """
import importlib.abc
import sys
class NoCameraSdk(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'alkusb3':
            raise AssertionError('Importing or constructing adapter loaded camera SDK')
sys.meta_path.insert(0, NoCameraSdk())
from hardware.camera import CameraSettings, CameraTrigger, CeleraCamera
cfg = CameraSettings(0, 4, 3, 0, 0, 1, 1, 0, 'mono8', 6000, 60,
                     CameraTrigger(False, 0, False, 0, 0))
CeleraCamera(cfg)
assert 'alkusb3' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_exact_raw_values_shape_independent_storage_and_settings(settings):
    fake = FakeCamera()
    fake.raw = bytearray([0, 1, 255, 127, 2, 3, 4, 5, 6, 7, 8, 9])
    camera = adapter(settings, fake)
    assert not fake.events
    with camera:
        assert camera.image_shape == (3, 4)
        frame = camera.capture_frame()
        assert frame.dtype == np.uint8
        np.testing.assert_array_equal(frame, [[0, 1, 255, 127], [2, 3, 4, 5], [6, 7, 8, 9]])
        fake.raw[:] = b"\xff" * 12
        assert frame[0, 0] == 0
        assert fake.reads == 1
        assert fake.values["camera"] == 2
        assert fake.values["image_start_x"] == 8
        assert fake.values["image_start_y"] == 6
        assert fake.values["shutter"] == 6000  # never the lab test helper's 1000
        assert fake.values["frame_rate"] == 60
        assert fake.events[-1] == ("acquire", True)
        assert fake.acquisition_start_trigger.values == dict(enabled=True,
            source="external SDK enum", external_input=1,
            detect_external_input_edge=True, delay=3000)
        assert fake.pio_ports[1].debounce_time == 25
    assert fake.events[-1] == ("acquire", False)
    count = len(fake.events)
    camera.disconnect()
    assert len(fake.events) == count


def test_disabled_trigger_overrides_existing_enabled_state(settings):
    fake = FakeCamera()
    cfg = replace(settings, trigger=replace(settings.trigger, enabled=False))
    with adapter(cfg, fake) as camera:
        assert fake.acquisition_start_trigger.enabled is False
        assert not any(name.startswith("pio") for name, _ in fake.events)
        assert camera.capture_frame().shape == (3, 4)


def test_binning_uses_sdk_dimension_readback(settings):
    fake = FakeCamera()
    cfg = replace(settings, roi_width=8, roi_height=6, horizontal_binning=2, vertical_binning=2)
    with adapter(cfg, fake) as camera:
        assert fake.values["image_size_x"] == 8
        assert fake.values["horizontal_binning"] == 2
        assert camera.capture_frame().shape == (3, 4)


@pytest.mark.parametrize("shutter", [1000, 1999, 15001, -1, 2000.0, True, float("nan")])
def test_invalid_shutter_rejected_before_open(settings, shutter):
    with pytest.raises(ValueError, match=r"\[2000, 15000\].*native SDK units"):
        replace(settings, shutter=shutter)


@pytest.mark.parametrize("shutter", [2000, 15000])
def test_shutter_range_is_inclusive(settings, shutter):
    fake = FakeCamera()
    with adapter(replace(settings, shutter=shutter), fake):
        assert fake.shutter == shutter


@pytest.mark.parametrize("raw, message", [
    (b"abc", "3 bytes; expected 12"), (bytes(13), "13 bytes; expected 12"),
    (None, "frame capture failed"), ([0] * 12, "frame capture failed"),
    (np.zeros(6, dtype=np.uint16), "contiguous byte buffer"),
    (memoryview(bytes(24))[::2], "contiguous byte buffer"),
])
def test_bad_frames_stop_acquisition(settings, raw, message):
    fake = FakeCamera()
    fake.raw = raw
    camera = adapter(settings, fake).connect()
    with pytest.raises(CameraError, match=message):
        camera.capture_frame()
    assert fake.acquire is False
    assert fake.events[-1] == ("acquire", False)
    with pytest.raises(CameraError, match="not connected"):
        camera.capture_frame()


def test_capture_sdk_exception_stops_camera_and_preserves_cause(settings):
    fake = FakeCamera()
    fake.capture_error = OSError("lost USB connection")
    camera = adapter(settings, fake).connect()
    with pytest.raises(CameraError, match="lost USB connection") as error:
        camera.capture_frame()
    assert isinstance(error.value.__cause__, OSError)
    assert fake.acquire is False


@pytest.mark.parametrize("stage", ["camera", "image_start_x", "shutter", "frame_rate", "acquire"])
def test_partial_setup_failure_attempts_stop(settings, stage):
    fake = FakeCamera()
    fake.fail_property = stage
    camera = adapter(settings, fake)
    with pytest.raises(CameraError, match="Camera setup failed"):
        camera.connect()
    assert fake.events[-1] == ("acquire", False)
    assert fake.reads == 0
    # Retrying stop is supported if its setter itself failed during setup.
    fake.fail_property = None
    camera.disconnect()


def test_shutter_readback_mismatch_fails_closed(settings):
    fake = FakeCamera()
    fake.shutter_override = 1000
    with pytest.raises(CameraError, match="shutter readback 1000"):
        adapter(settings, fake).connect()
    assert fake.acquire is False
    assert ("acquire", True) not in fake.events


def test_cleanup_failure_keeps_original_error_and_can_retry_stop(settings):
    fake = FakeCamera()
    camera = adapter(settings, fake).connect()
    fake.raw = b""
    fake.fail_stop = True
    with pytest.raises(CameraError, match="0 bytes; expected 12") as error:
        camera.capture_frame()
    assert any("Stopping camera acquisition also failed" in note for note in error.value.__notes__)
    with pytest.raises(CameraError, match="already open"):
        camera.connect()
    fake.fail_stop = False
    camera.disconnect()


def test_start_failure_after_acquire_set_is_stopped(settings):
    fake = FakeCamera()
    original = FakeCamera.__setattr__
    class FailsOnStart(FakeCamera):
        def __setattr__(self, key, value):
            original(self, key, value)
            if key == "acquire" and value is True:
                raise OSError("start failed after partially starting")
    fake = FailsOnStart()
    with pytest.raises(CameraError, match="partially starting"):
        adapter(settings, fake).connect()
    assert ("acquire", True) in fake.events
    assert fake.events[-1] == ("acquire", False)


def test_closed_capture_double_connect_and_reconnect(settings):
    fake = FakeCamera()
    camera = adapter(settings, fake)
    camera.disconnect()
    with pytest.raises(CameraError, match="not connected"):
        camera.capture_frame()
    with camera:
        with pytest.raises(CameraError, match="already open"):
            camera.connect()
    with camera:
        assert camera.capture_frame().shape == (3, 4)


@pytest.mark.parametrize("changes", [{"pixel_format": "mono16"}, {"roi_width": 0},
    {"roi_y": -1}, {"horizontal_binning": 0}, {"camera_index": -1},
    {"frame_rate": float("inf")}, {"frame_rate": 0}, {"trigger": None}])
def test_invalid_settings(settings, changes):
    with pytest.raises(ValueError):
        replace(settings, **changes)
