"""The standalone checker uses fake serial only, including error paths."""

import json
from pathlib import Path

import pytest

import check_laser_status
from hardware.bench_access import bench_access
from hardware.laser import LabLaser
from tests.test_laser_capture import SerialFake


@pytest.fixture
def checker(tmp_path, monkeypatch):
    template = Path(__file__).resolve().parents[1] / "configs/capture-once.example.json"
    laser_settings = json.loads(template.read_text(encoding="utf-8"))["laser"]
    laser_settings["port"] = "COM-FAKE"
    assert laser_settings["startup_confirmed"] is False
    config = tmp_path / "config.json"
    # Deliberately omit every non-laser configuration section.
    config.write_text(json.dumps({"laser": laser_settings}), encoding="utf-8")
    events, writes = [], []
    port = SerialFake(events)
    write = port.write

    def record_write(payload):
        writes.append(payload)
        return write(payload)

    port.write = record_write

    def serial_factory(**kwargs):
        assert kwargs == dict(port=None, baudrate=19200, bytesize=8, parity="N",
            stopbits=1, xonxoff=False, rtscts=False, dsrdtr=False,
            timeout=2.0, write_timeout=2.0)
        return port

    def forbidden(*args, **kwargs):
        pytest.fail("Query-only checker called startup/shutdown connection path")

    for name in ("connect", "disconnect", "shutdown", "start_emission"):
        monkeypatch.setattr(LabLaser, name, forbidden)
    monkeypatch.setattr(check_laser_status, "LabLaser",
                        lambda settings: LabLaser(settings, serial_factory=serial_factory))
    return ["--config", str(config)], port, events, writes


QUERIES = [b"?S\r\n", b"?L\r\n", b"?P\r\n", b"?SP\r\n"]
SESSION = ["serial.open", "?S", "?L", "?P", "?SP", "serial.close"]


@pytest.mark.parametrize("laser,description", [(0, "standby"), (1, "on"),
                                               (2, "fault (off due to fault)")])
def test_queries_only_and_reports_actual_state(checker, capsys, laser, description):
    args, port, events, writes = checker
    port.responses.update({"?S": b"1\r\n", "?L": f"{laser}\r\n".encode(),
                           "?P": b"1.234\r\n", "?SP": b"2.25\r\n"})
    assert check_laser_status.main(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == ["Shutter: open", f"Laser: {description}",
                         "Measured power: 1.234 W", "Setpoint: 2.25 W"]
    assert events == SESSION and writes == QUERIES
    assert not port.is_open


@pytest.mark.parametrize("reply", [b"", b"ERROR\r\n", b"3\r\n"])
def test_bad_reply_is_unverified_and_other_queries_continue(checker, capsys, reply):
    args, port, events, writes = checker
    port.responses["?L"] = reply
    assert check_laser_status.main(args) == 1
    output = capsys.readouterr().out
    assert "Laser: UNVERIFIED" in output and "Setpoint: 0.01 W" in output
    assert events == SESSION and writes == QUERIES
    assert not port.is_open


def test_query_write_failure_still_closes(checker, capsys):
    args, port, events, writes = checker
    port.fail_command = "?S"
    assert check_laser_status.main(args) == 1
    assert "Shutter: UNVERIFIED" in capsys.readouterr().out
    assert events == SESSION and writes == QUERIES
    assert not port.is_open


def test_partial_open_failure_closes_without_queries(checker, capsys):
    args, port, events, writes = checker
    open_port = port.open

    def fail_open():
        open_port()
        raise OSError("fake partial open failure")

    port.open = fail_open
    assert check_laser_status.main(args) == 1
    assert "UNVERIFIED - fake partial open failure" in capsys.readouterr().out
    assert events == ["serial.open", "serial.close"] and writes == []
    assert not port.is_open


def test_close_failure_is_nonzero_without_shutdown_writes(checker, capsys):
    args, port, events, writes = checker
    port.fail_close = True
    assert check_laser_status.main(args) == 1
    assert "UNVERIFIED - fake close failure" in capsys.readouterr().out
    assert events == SESSION and writes == QUERIES


def test_busy_bench_never_opens_serial(checker, capsys):
    args, _, events, writes = checker
    with bench_access():
        assert check_laser_status.main(args) == 1
    assert "UNVERIFIED" in capsys.readouterr().out
    assert events == [] and writes == []


def test_captured_viper_replies_keep_laser_error_unverified(checker, capsys):
    args, port, events, writes = checker
    port.responses.update({"?S": b"0\r\n",
        "?L": b"Viper V-5> Command Error:L\r\n",
        "?P": b"Viper V-5>0.005\r\n",
        "?SP": b"Viper V-5>0.0100\r\n"})
    assert check_laser_status.main(args) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Shutter: closed"
    assert lines[1].startswith("Laser: UNVERIFIED") and "Command Error:L" in lines[1]
    assert lines[2:4] == ["Measured power: 0.005 W", "Setpoint: 0.01 W"]
    assert events == SESSION and writes == QUERIES
    assert not port.is_open
