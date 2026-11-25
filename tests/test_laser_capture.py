"""One-frame milestone verified only against fake devices and fake serial IO."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from hardware.capture_once import capture_once, load_settings, main
from hardware.bench_access import BenchBusyError, request_shutdown
from hardware.laser import LabLaser, LaserError, LaserSettings, parse_verdi_reply


@pytest.fixture
def settings():
    return LaserSettings("COM-FAKE", 2.25, ("set_power", "enable", "open_shutter"), True,
        "\r\n", "\r\n", None)


class SerialFake:
    def __init__(self, events, *, fail_command=None):
        self.events = events
        self.is_open = False
        self.reply = b""
        self.fail_command = fail_command
        self.responses = {"?S": b"0\r\n", "?L": b"0\r\n", "?P": b"0.000\r\n", "?SP": b"0.0100\r\n"}
        self.echo = False
        self.short_write = False
        self.fail_close = False

    def open(self):
        self.events.append("serial.open")
        self.is_open = True

    def write(self, data):
        command = data.decode().strip()
        self.events.append(command)
        if command == self.fail_command:
            raise OSError("fake serial write error")
        response = self.responses.get(command, b"\r\n")
        if self.echo:
            response = command.encode() + b"\r\n" + (response if command.startswith("?") else b"")
        self.reply += response  # retain unread bytes across commands like a real port
        return len(data) - int(self.short_write)

    def read_until(self, **kwargs):
        line, separator, rest = self.reply.partition(b"\n")
        self.reply = rest
        return line + separator

    def close(self):
        self.events.append("serial.close")
        if self.fail_close:
            raise OSError("fake close failure")
        self.is_open = False


def laser_for(settings, events, **kwargs):
    port = SerialFake(events, **kwargs)
    def factory(**arguments):
        assert arguments == dict(port=None, baudrate=19200, bytesize=8, parity="N", stopbits=1,
            xonxoff=False, rtscts=False, dsrdtr=False, timeout=2.0, write_timeout=2.0)
        return port
    return LabLaser(settings, serial_factory=factory), port


@pytest.mark.parametrize("power", [0.01, 0.99, 2.51, float("inf"), float("nan"), True])
def test_reject_operating_power_outside_lab_range(settings, power):
    with pytest.raises(ValueError, match="1.0, 2.5"):
        replace(settings, operating_power=power)


@pytest.mark.parametrize("power", [1.0, 2.5])
def test_operating_range_endpoints(settings, power):
    assert replace(settings, operating_power=power).operating_power == power


def test_startup_is_explicit_and_shutdown_order_is_fixed(settings):
    events = []
    laser, port = laser_for(settings, events)
    assert events == []
    laser.connect()
    laser.start_emission()
    assert events == ["serial.open", "P=2.25", "L=1", "S=1"]
    result = laser.disconnect()
    assert events[-8:] == ["S=0", "L=0", "P=0.01", "?S", "?L", "?P", "?SP", "serial.close"]
    assert result["confirmed"] and result["serial_close"] == "closed"
    assert laser.disconnect() == result


def test_unconfirmed_startup_never_sends_emission_commands(settings):
    events = []
    laser, _ = laser_for(replace(settings, startup_confirmed=False), events)
    laser.connect()
    with pytest.raises(LaserError, match="not lab-confirmed"):
        laser.start_emission()
    assert events == ["serial.open"]
    laser.disconnect()


def test_shutdown_latch_blocks_remaining_laser_startup_commands(settings):
    events = []
    laser, port = laser_for(settings, events)
    write = port.write

    def stop_after_enable(data):
        length = write(data)
        if data == b"L=1\r\n":
            request_shutdown()
        return length

    port.write = stop_after_enable
    laser.connect()
    with pytest.raises(BenchBusyError, match="latched off"):
        laser.start_emission()
    assert "S=1" not in events
    assert "S=0" in events and "L=0" in events
    assert laser.last_cleanup["confirmed"]


@pytest.mark.parametrize("failed", ["S=0", "L=0", "P=0.01", "?S", "?L", "?P", "?SP"])
def test_all_shutdown_steps_attempted_despite_error(settings, failed):
    events = []
    laser, _ = laser_for(settings, events, fail_command=failed)
    laser.connect()
    result = laser.disconnect()
    assert not result["confirmed"]
    assert events[-8:] == ["S=0", "L=0", "P=0.01", "?S", "?L", "?P", "?SP", "serial.close"]


@pytest.mark.parametrize("reply", [b"", b"FAULT\r\n", b"0", b"0\r\n"])
def test_only_verified_status_replies_establish_shutdown(settings, reply):
    events = []
    laser, port = laser_for(settings, events)
    port.responses["?S"] = reply
    laser.connect()
    assert laser.disconnect()["confirmed"] == (reply == b"0\r\n")


def test_documented_semantics_cannot_be_overridden(settings):
    with pytest.raises(ValueError, match="documented"):
        replace(settings, shutdown_replies={"?P": "0.01"})


def test_echo_is_checked_not_guessed(settings):
    laser, port = laser_for(replace(settings, echo_commands=True), [])
    port.echo = True
    laser.connect()
    laser.start_emission()
    assert laser.disconnect()["confirmed"]


def test_short_write_triggers_shutdown_attempts(settings):
    events = []
    laser, port = laser_for(settings, events)
    laser.connect()
    port.short_write = True
    with pytest.raises(LaserError, match="Incomplete serial"):
        laser.start_emission()
    assert "S=0" in events and "L=0" in events and "P=0.01" in events
    assert events[-1] == "serial.close"


def test_partial_serial_open_failure_attempts_shutdown_and_close(settings):
    events = []
    laser, port = laser_for(settings, events)
    original = port.open
    def partial_open():
        original()
        raise OSError("failed after serial port opened")
    port.open = partial_open
    with pytest.raises(OSError, match="after serial port opened"):
        laser.connect()
    assert events[-8:] == ["S=0", "L=0", "P=0.01", "?S", "?L", "?P", "?SP", "serial.close"]


def test_start_ack_failure_does_not_become_startup_success(settings):
    events = []
    laser, port = laser_for(settings, events)
    port.responses["P=2.25"] = b"RANGE ERROR: P=2.25\r\n"
    laser.connect()
    with pytest.raises(LaserError, match="error reply"):
        laser.start_emission()
    assert "L=1" not in events
    assert "S=0" in events and "L=0" in events and "P=0.01" in events


class DeviceFake:
    image_shape = (3, 4)
    def __init__(self, name, events, fail=None):
        self.name, self.events, self.fail = name, events, fail
    def _event(self, action):
        event = f"{self.name}.{action}"
        self.events.append(event)
        if self.fail == action:
            raise OSError(f"fake {event} failure")
    def connect(self):
        self._event("connect")
    def display_image(self, image):
        self._event("display")
        np.testing.assert_array_equal(image, np.full((3, 4), 17, dtype=np.uint8))
    def capture_frame(self):
        self._event("capture")
        return np.arange(12, dtype=np.uint8).reshape(3, 4)
    def disconnect(self):
        self._event("disconnect")


def run_once(tmp_path, settings, *, slm_fail=None, camera_fail=None, laser_fail=None):
    events = []
    laser, _ = laser_for(settings, events, fail_command=laser_fail)
    report = capture_once(slm=DeviceFake("slm", events, slm_fail),
        camera=DeviceFake("camera", events, camera_fail), laser=laser,
        output_dir=tmp_path / "capture", gray_level=17, laser_settle_seconds=0)
    return report, events


def test_single_capture_arms_before_flip_and_saves_exact_raw_frame(tmp_path, settings):
    report, events = run_once(tmp_path, settings)
    assert report["capture_status"] == "completed" and report["cleanup_ok"]
    assert events.index("camera.connect") < events.index("slm.display") < events.index("camera.capture")
    assert events.index("S=0") < events.index("camera.disconnect") < events.index("slm.disconnect")
    raw = np.arange(12, dtype=np.uint8).reshape(3, 4)
    np.testing.assert_array_equal(np.load(tmp_path / "capture/camera-frame.npy"), raw)
    assert (tmp_path / "capture/raw-frame.bin").read_bytes() == raw.tobytes()
    saved = json.loads((tmp_path / "capture/capture-result.json").read_text())
    assert saved["frame"]["shape"] == [3, 4]
    assert set(saved["cleanup"]) == {"laser", "camera", "slm"}


@pytest.mark.parametrize("slm_fail,camera_fail", [("display", None), (None, "capture"), (None, "connect")])
def test_write_capture_and_partial_camera_errors_cleanup_every_device(tmp_path, settings, slm_fail, camera_fail):
    report, events = run_once(tmp_path, settings, slm_fail=slm_fail, camera_fail=camera_fail)
    assert report["capture_status"] == "failed"
    assert "camera.disconnect" in events and "slm.disconnect" in events
    assert "S=0" in events and "L=0" in events and "P=0.01" in events
    assert not (tmp_path / "capture/camera-frame.npy").exists()


def test_cleanup_failure_does_not_skip_other_devices(tmp_path, settings):
    report, events = run_once(tmp_path, settings, camera_fail="disconnect", laser_fail="S=0")
    assert report["capture_status"] == "completed"
    assert not report["cleanup_ok"]
    assert report["cleanup"]["laser"]["status"] == "UNCONFIRMED"
    assert report["cleanup"]["camera"]["status"] == "failed"
    assert report["cleanup"]["slm"]["status"] == "disconnect_returned"


def test_unconfirmed_shutdown_surfaces_even_when_capture_passes(tmp_path, settings):
    report, _ = run_once(tmp_path, settings, laser_fail="?SP")
    assert report["capture_status"] == "completed" and not report["cleanup_ok"]
    assert report["cleanup"]["laser"]["status"] == "UNCONFIRMED"


def test_serial_close_failure_is_not_successful_cleanup(tmp_path, settings):
    events = []
    laser, port = laser_for(settings, events)
    port.fail_close = True
    report = capture_once(slm=DeviceFake("slm", events), camera=DeviceFake("camera", events),
        laser=laser, output_dir=tmp_path / "capture", gray_level=17, laser_settle_seconds=0)
    assert report["capture_status"] == "completed"
    assert report["cleanup"]["laser"]["details"]["confirmed"] is True
    assert report["cleanup"]["laser"]["status"] == "failed"
    assert not report["cleanup_ok"]
    assert "camera.disconnect" in events and "slm.disconnect" in events


def test_frame_save_error_still_cleans_up(tmp_path, settings, monkeypatch):
    save = np.save
    def fail_frame_save(path, *args, **kwargs):
        if Path(path).name == "camera-frame.npy":
            raise OSError("fake disk write failure")
        return save(path, *args, **kwargs)
    monkeypatch.setattr(np, "save", fail_frame_save)
    report, events = run_once(tmp_path, settings)
    assert report["capture_status"] == "failed"
    assert report["last_stage"] == "camera_frame_received"
    assert "fake disk write failure" in report["error"]
    assert report["cleanup_ok"]
    assert "camera.disconnect" in events and "slm.disconnect" in events


def test_cli_requires_operator_and_verified_settings(tmp_path):
    template = Path(__file__).resolve().parents[1] / "configs/capture-once.example.json"
    with pytest.raises(SystemExit) as error:
        main(["--config", str(template), "--output", str(tmp_path / "live")])
    assert error.value.code == 2
    with pytest.raises(ValueError, match="trigger configuration"):
        load_settings(template)


def test_settings_preserve_reference_trigger_and_require_confirmed_startup(tmp_path):
    template = Path(__file__).resolve().parents[1] / "configs/capture-once.example.json"
    data = json.loads(template.read_text())
    data["camera_trigger_confirmed"] = True
    data["slm"]["return_codes"] = {"load_lut": 11, "write_image": 22, "image_complete": 33}
    local = tmp_path / "machine.json"
    local.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="startup order must be lab-confirmed"):
        load_settings(local)
    data["laser"]["startup_confirmed"] = True
    local.write_text(json.dumps(data))
    _, camera, laser, gray, settle = load_settings(local)
    assert (camera.trigger.external_input, camera.trigger.detect_external_input_edge,
            camera.trigger.delay, camera.shutter) == (1, True, 3000, 6000)
    assert laser.startup_order == ("enable", "open_shutter", "set_power")


def test_imports_and_help_do_not_touch_hardware():
    script = """
import sys
sys.modules['serial'] = None
sys.modules['alkusb3'] = None
import hardware.laser
from hardware.capture_once import main
main(['--help'])
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("query,value,expected", [
    ("?L", "0", 0), ("?L", "1", 1), ("?L", "2", 2),
    ("?S", "0", 0), ("?S", "1", 1),
    ("?P", "1.234", 1.234), ("?SP", "0.0100", 0.01),
])
@pytest.mark.parametrize("form", ["{v}\r\n", "Verdi> {v}\r\n",
    "{q}{v}\r\n", "{q}\r\n{v}\r\n", "Verdi> {q}{v}\r\n",
    "Verdi>\r\n{q}\r\n{v}\r\n", "{q}\r\nVerdi> {v}\r\n",
    "{v}\r\nVerdi>\r\n"])
def test_verdi_documented_values_with_prompt_and_echo(query, value, expected, form):
    assert parse_verdi_reply(query, form.format(q=query, v=value).encode("ascii")) == expected


@pytest.mark.parametrize("query,raw", [
    ("?L", b""), ("?L", b"\r\n"), ("?L", b"Verdi>\r\n"),
    ("?L", b"?L\r\n"), ("?L", b"0\n"), ("?L", b"3\r\n"),
    ("?L", b"0.0\r\n"), ("?S", b"2\r\n"), ("?S", b"?L0\r\n"),
    ("?L", b"0\r\n1\r\n"), ("?L", b"\xff\r\n"),
    ("?SP", b"NaN\r\n"), ("?SP", b"inf\r\n"), ("?SP", b"-0.01\r\n"),
    ("?SP", b"0.01 W\r\n"), ("?SP", b"?P0.01\r\n"),
    ("?SP", b"Verdi> Query Error: ?SP\r\n"),
    ("?L", b"Verdi> ?L Command Error: 0\r\n"),
    ("?SP", b"RANGE ERROR: 0.01\r\n"), ("?SP", b"9" * 400 + b"\r\n"),
])
def test_invalid_verdi_replies_never_become_confirmed_values(query, raw):
    with pytest.raises(LaserError, match="[Uu]nconfirmed"):
        parse_verdi_reply(query, raw)


@pytest.mark.parametrize("form", ["{v}\r\n", "Verdi>{v}\r\n",
    "{q}{v}\r\n", "Verdi> {q}\r\n{v}\r\n",
    "Verdi>\r\n{q}\r\n{v}\r\n"])
def test_serial_status_path_accepts_optional_echo_and_prompt(settings, form):
    laser, port = laser_for(settings, [])
    for q, v in (("?S", "0"), ("?L", "0"), ("?P", "0.000"), ("?SP", "0.0100")):
        port.responses[q] = form.format(q=q, v=v).encode("ascii")
    laser.connect()
    assert laser.disconnect()["confirmed"]


def test_setpoint_cannot_be_verified_by_measured_power(settings):
    laser, port = laser_for(settings, [])
    port.responses["?P"] = b"0.010\r\n"
    port.responses["?SP"] = b"2.2500\r\n"
    laser.connect()
    result = laser.disconnect()
    assert not result["confirmed"]
    checks = {check["query"]: check for check in result["checks"]}
    assert checks["?P"]["value"] == 0.01 and checks["?P"]["meaning"] == "measured_output"
    assert checks["?SP"]["value"] == 2.25 and not checks["?SP"]["confirmed"]


def test_faulted_off_is_recorded_and_not_normal_standby(settings):
    laser, port = laser_for(settings, [])
    port.responses["?L"] = b"2\r\n"
    laser.connect()
    result = laser.disconnect()
    assert not result["confirmed"]
    check = next(c for c in result["checks"] if c["query"] == "?L")
    assert check["valid"] and check["state"] == "off_due_to_fault" and not check["confirmed"]


@pytest.mark.parametrize("reply", [b"", b"ERROR\r\n", b"0.01", b"NaN\r\n"])
def test_bad_setpoint_response_leaves_shutdown_unconfirmed(settings, reply):
    laser, port = laser_for(settings, [])
    port.responses["?SP"] = reply
    laser.connect()
    assert not laser.disconnect()["confirmed"]


def test_read_status_is_read_only_and_reports_typed_states(settings):
    events = []
    laser, port = laser_for(settings, events)
    port.responses["?L"] = b"1\r\n"
    port.responses["?S"] = b"1\r\n"
    port.responses["?P"] = b"1.234\r\n"
    port.responses["?SP"] = b"2.2500\r\n"
    laser.connect()
    status = laser.read_status()
    assert events == ["serial.open", "?S", "?L", "?P", "?SP"]
    assert status["?L"]["state"] == "on" and status["?S"]["state"] == "open"
    assert status["?P"]["value"] == 1.234 and status["?SP"]["value"] == 2.25
    laser.disconnect()


def test_pending_prompt_between_queries_is_consumed_without_losing_status(settings):
    laser, port = laser_for(settings, [])
    for query, value in (("?S", "0"), ("?L", "0"), ("?P", "0.000"), ("?SP", "0.0100")):
        port.responses[query] = f"{value}\r\nVerdi>\r\n".encode("ascii")
    laser.connect()
    assert laser.disconnect()["confirmed"]


@pytest.mark.parametrize("format", ["\r\n", "Verdi>\r\n", "{command}\r\n", "Verdi> {command}\r\n"])
def test_documented_command_ack_forms(settings, format):
    laser, port = laser_for(settings, [])
    for command in ("S=0", "L=0", "P=0.01"):
        port.responses[command] = format.format(command=command).encode("ascii")
    laser.connect()
    assert laser.disconnect()["confirmed"]


def test_command_errors_with_numeric_operands_are_not_successful_off_actions(settings):
    laser, port = laser_for(settings, [])
    port.responses["S=0"] = b"Verdi> S=0 Command Error: S=0\r\n"
    laser.connect()
    result = laser.disconnect()
    assert not result["confirmed"]
    assert result["actions"][0]["status"] == "failed"


@pytest.mark.parametrize("query,raw,expected", [
    ("?P", b"Viper V-5>0.005\r\n", 0.005),
    ("?SP", b"Viper V-5>0.0100\r\n", 0.01),
    ("?S", b"Viper V-5>0\r\n", 0),
    ("?L", b"Viper V-5>?L\r\n1\r\n", 1),
    ("?P", b"?P\r\nViper V-5>0.005\r\n", 0.005),
    ("?SP", b"0.0100\r\nViper V-5>\r\n", 0.01),
])
def test_observed_viper_prompt_is_known_framing(query, raw, expected):
    assert parse_verdi_reply(query, raw) == expected


@pytest.mark.parametrize("query,raw", [
    ("?L", b"Viper V-5> Command Error:L\r\n"),
    ("?P", b"Viper V-5>0.005"),  # A literal missing CRLF is still invalid.
    ("?SP", b"Viper V-5>0.0100"),
    ("?SP", b"Viper V-5>?P0.005\r\n"),  # Wrong echoed query.
    ("?P", b"Other controller>0.005\r\n"),
    ("?P", b"Viper V-5>0.005\r\n0.0100\r\n"),
])
def test_viper_prompt_does_not_relax_errors_framing_or_query_matching(query, raw):
    with pytest.raises(LaserError):
        parse_verdi_reply(query, raw)


def test_viper_pending_prompt_and_error_do_not_shift_next_query(settings):
    events = []
    laser, port = laser_for(settings, events)
    port.responses.update({"?S": b"0\r\nViper V-5>\r\n",
        "?L": b"Viper V-5> Command Error:L\r\n",
        "?P": b"Viper V-5>?P\r\n0.005\r\nViper V-5>\r\n",
        "?SP": b"Viper V-5>?SP\r\n0.0100\r\n"})
    states = laser.read_status_once()
    assert states["?S"]["value"] == 0
    assert not states["?L"]["valid"]
    assert "Command Error:L" in states["?L"]["error"]
    assert states["?P"]["value"] == 0.005
    assert states["?SP"]["value"] == 0.01
    assert events == ["serial.open", "?S", "?L", "?P", "?SP", "serial.close"]
    assert laser.transcript[1]["raw_replies_hex"] == [
        b"Viper V-5>\r\n".hex(), b"Viper V-5> Command Error:L\r\n".hex()]
    assert not port.is_open
