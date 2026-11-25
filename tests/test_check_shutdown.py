import json
from pathlib import Path
import subprocess
import sys

import pytest

import check_shutdown
from hardware import bench_access
from hardware.generator import RigolDG812
from tests.test_generator import FakeVisa
from tests.test_laser_capture import laser_for


def save(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def bench(tmp_path, monkeypatch):
    template = Path(__file__).resolve().parents[1] / "configs/capture-once.example.json"
    config = tmp_path / "config.json"
    save(config, json.loads(template.read_text(encoding="utf-8")))
    run = tmp_path / "run-001"
    run.mkdir()
    record = {"capture_status": "completed", "cleanup": {
        "laser": {"status": "confirmed"},
        "camera": {"status": "disconnect_returned"},
        "slm": {"status": "disconnect_returned"},
    }}
    args = ["--config", str(config), "--run-dir", str(run)]
    events, visa = [], FakeVisa()
    monkeypatch.setattr(bench_access, "_LOCK_PATH", tmp_path / "bench.lock")

    class FakeLaserFactory:
        responses = {}

        def __call__(self, settings):
            laser, port = laser_for(settings, events)
            port.responses.update(self.responses)
            return laser

    laser_factory = FakeLaserFactory()
    monkeypatch.setattr(check_shutdown, "LabLaser", laser_factory)
    monkeypatch.setattr(check_shutdown, "RigolDG812",
                        lambda settings: RigolDG812(settings, resource_manager_factory=visa.resource_manager))
    return args, run, record, events, visa, laser_factory


def test_capture_all_pass_without_disabled_generator(bench, capsys):
    args, run, record, events, visa, _ = bench
    save(run / "capture-result.json", record)
    assert check_shutdown.main(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("Laser: PASS")
    assert "shutdown setpoint 0.01 W" in lines[0] and "measured power 0 W" in lines[0]
    assert lines[1].startswith("Generator: NOT USED")
    assert lines[2:] == ["Camera: PASS - release verified from run record",
                         "SLM: PASS - release verified from run record"]
    assert events == ["serial.open", "?S", "?L", "?P", "?SP", "serial.close"]
    assert not visa.calls


@pytest.mark.parametrize("query,reply,status", [
    ("?S", b"1\r\n", "FAIL"),
    ("?L", b"1\r\n", "FAIL"),
    ("?L", b"2\r\n", "FAIL"),
    ("?L", b"", "UNVERIFIED"),
    ("?SP", b"0.02\r\n", "FAIL"),
    ("?SP", b"", "UNVERIFIED"),
])
def test_laser_evaluates_independent_replies(bench, capsys, query, reply, status):
    args, run, record, events, _, laser = bench
    save(run / "capture-result.json", record)
    laser.responses = {query: reply}
    assert check_shutdown.main(args) == 1
    assert capsys.readouterr().out.startswith(f"Laser: {status}")
    assert events == ["serial.open", "?S", "?L", "?P", "?SP", "serial.close"]


def test_laser_open_error_closes_without_shutdown_writes(bench, monkeypatch, capsys):
    args, run, record, events, _, _ = bench
    save(run / "capture-result.json", record)

    def factory(settings):
        laser, port = laser_for(settings, events)
        open_port = port.open

        def fail_after_open():
            open_port()
            raise OSError("fake partial open error")

        port.open = fail_after_open
        return laser

    monkeypatch.setattr(check_shutdown, "LabLaser", factory)
    assert check_shutdown.main(args) == 1
    assert capsys.readouterr().out.startswith("Laser: UNVERIFIED")
    assert events == ["serial.open", "serial.close"]


@pytest.mark.parametrize("state,status", [("off", "PASS"), ("on", "FAIL"), ("lost", "UNVERIFIED")])
def test_generator_queries_only_output_when_run_used_it(bench, monkeypatch, capsys, state, status):
    args, run, record, _, visa, _ = bench
    # Current config is disabled, but the selected run attempted an LCC acquisition.
    record["lcc"] = {"requested": {"amplitude_vpp": 2}}
    record["cleanup"]["generator"] = {"status": "not_initialized" if state == "lost" else "confirmed"}
    save(run / "capture-result.json", record)
    visa.enabled = state == "on"
    if state == "lost":
        def unreachable(address, *, open_timeout):
            raise OSError("fake VISA address unreachable")
        monkeypatch.setattr(visa, "open_resource", unreachable)
    assert check_shutdown.main(args) == int(state != "off")
    assert capsys.readouterr().out.splitlines()[1].startswith(f"Generator: {status}")
    assert visa.queries == ([] if state == "lost" else [":OUTP1?"])
    assert not visa.writes
    assert visa.calls[-1] == "close"


def test_inconclusive_and_failed_camera_slm_cleanup(bench, capsys):
    args, run, record, _, _, _ = bench
    record["cleanup"]["camera"]["status"] = "failed"
    del record["cleanup"]["slm"]
    save(run / "capture-result.json", record)
    assert check_shutdown.main(args) == 1
    assert capsys.readouterr().out.splitlines()[2:] == [
        "Camera: FAIL - run record reports release failed",
        "SLM: UNVERIFIED - run record has no conclusive device release",
    ]


def runner_record(*, status="completed", origin="physical", paths=()):
    return {"status": status, "raw_data": [{"path": str(path)} for path in paths],
            "hardware": {"data_origin": origin, "cleanup_status": "completed",
                         "ended_at": "2026-11-24T10:00:00Z"}}


def test_runner_aggregate_cannot_verify_releases(bench, capsys):
    args, run, _, events, visa, _ = bench
    save(run / "record.json", runner_record())
    assert check_shutdown.main(args) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("Laser: PASS") and events
    assert all(line.split(": ")[1].startswith("UNVERIFIED") for line in lines[1:])
    assert not visa.calls


def test_runner_linked_device_record_can_verify_releases(bench, capsys):
    args, run, record, _, visa, _ = bench
    artifact = run / "raw" / "0" / "capture-result.json"
    artifact.parent.mkdir(parents=True)
    save(artifact, record)
    save(run / "record.json", runner_record(paths=[artifact]))
    assert check_shutdown.main(args) == 0
    assert capsys.readouterr().out.count("verified from run record") == 2
    assert not visa.calls


@pytest.mark.parametrize("kind", ["missing", "unfinished", "simulated", "busy"])
def test_no_hardware_without_idle_physical_run(bench, capsys, kind):
    args, run, record, events, visa, _ = bench
    if kind == "unfinished":
        save(run / "record.json", runner_record(status="started"))
    elif kind == "simulated":
        save(run / "record.json", runner_record(origin="simulated"))
    if kind == "busy":
        save(run / "capture-result.json", record)
        # A different process must be unable to enter the shared bench while
        # an acquisition holds it, even when given an older completed record.
        script = ("import sys; from pathlib import Path; from hardware import bench_access; "
                  "import check_shutdown; bench_access._LOCK_PATH = Path(sys.argv[1]); "
                  "raise SystemExit(check_shutdown.main(sys.argv[2:]))")
        with bench_access.bench_access(acquisition=True):
            with bench_access.bench_access(acquisition=True):
                with pytest.raises(bench_access.BenchBusyError):
                    with bench_access.bench_access():
                        pytest.fail("status check entered an active acquisition")
            result = subprocess.run([sys.executable, "-c", script, str(bench_access._LOCK_PATH), *args],
                                    capture_output=True, text=True, check=False)
        assert result.returncode == 1 and result.stdout.count("bench is in use") == 4
    else:
        assert check_shutdown.main(args) == 1
        assert capsys.readouterr().out.count(": UNVERIFIED") == 4
    assert not events and not visa.calls
