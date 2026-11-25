import hashlib
import json
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread

import pytest

import shutdown_bench
from hardware import bench_access
from hardware.generator import GeneratorSettings, LccAcquisitionPlan, RigolDG812
from hardware.laser import LaserSettings
from tests.test_generator import FakeVisa
from tests.test_laser_capture import DeviceFake, laser_for
from tests.test_runner import approved, hardware_setup


@pytest.fixture
def bench(tmp_path, monkeypatch):
    template = Path(__file__).resolve().parents[1] / "configs/capture-once.example.json"
    data = json.loads(template.read_text(encoding="utf-8"))
    config = tmp_path / "config.json"
    events, visa = [], FakeVisa()
    reply_updates = {}

    def make_laser(settings):
        laser, port = laser_for(settings, events)
        port.responses.update(reply_updates)
        return laser

    monkeypatch.setattr(shutdown_bench, "LabLaser", make_laser)
    monkeypatch.setattr(shutdown_bench, "RigolDG812",
                        lambda settings: RigolDG812(settings, resource_manager_factory=visa.resource_manager))
    monkeypatch.setattr(shutdown_bench, "_DEFAULT_CONFIG", config)

    def write_config(*, generator=False):
        data["generator"]["enabled"] = generator
        config.write_text(json.dumps(data), encoding="utf-8")

    write_config()
    return write_config, events, visa, reply_updates


def save_release(directory, *, slm="disconnect_returned"):
    record = {"capture_status": "failed", "cleanup": {
        "camera": {"status": "disconnect_returned"},
        "slm": {"status": slm, "error": "SDK cleanup failed" if slm == "failed" else None},
    }}
    (directory / "capture-result.json").write_text(json.dumps(record), encoding="utf-8")


@pytest.mark.parametrize("generator_enabled", [False, True])
def test_exited_owner_repeated_shutdown_and_explicit_rearm(bench, tmp_path, capsys, generator_enabled):
    write_config, events, visa, _ = bench
    write_config(generator=generator_enabled)
    visa.enabled = generator_enabled
    with bench_access.bench_access(acquisition=True):
        bench_access.record_bench_run(tmp_path)
        save_release(tmp_path)

    for _ in range(2):
        assert shutdown_bench.main([]) == 0
        lines = capsys.readouterr().out.splitlines()
        generator_status = "VERIFIED" if generator_enabled else "NOT CONFIGURED"
        assert [line.split(" - ")[0] for line in lines] == [
            "Laser: VERIFIED", f"Generator: {generator_status}", "Camera: VERIFIED", "SLM: VERIFIED"]
        assert "?S=0, ?L=0, ?SP=0.01, ?P=0.0" in lines[0]
        assert str(tmp_path) in lines[2]
        with pytest.raises(bench_access.BenchBusyError, match="latched off"):
            with bench_access.bench_access(acquisition=True):
                pytest.fail("shutdown allowed a new acquisition")
    assert events == 2 * ["serial.open", "S=0", "L=0", "P=0.01", "?S", "?L", "?P", "?SP", "serial.close"]
    if generator_enabled:
        assert visa.writes == [":OUTP1 OFF"] * 2 and not visa.enabled
        assert visa.queries == [":OUTP1?"] * 2
    else:
        assert not visa.calls
    assert shutdown_bench.main(["--rearm"]) == 0
    with bench_access.bench_access(acquisition=True):
        assert not bench_access.BenchCancellation().is_set()


def test_generator_shutdown_uses_config_without_run_record(bench, capsys):
    write_config, _, visa, _ = bench
    write_config(generator=True)
    visa.enabled = True
    assert shutdown_bench.main([]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert "Generator: VERIFIED" in lines[1]
    assert all("UNVERIFIED" in line for line in lines[2:])
    assert visa.queries == [":OUTP1?"] and visa.writes == [":OUTP1 OFF"]
    assert not visa.enabled and visa.calls[-2:] == ["close", "close"]


@pytest.mark.parametrize("timeout_device", ["laser", "generator"])
def test_readback_timeout_does_not_skip_independent_shutdown(bench, tmp_path, monkeypatch, capsys,
                                                             timeout_device):
    write_config, events, visa, replies = bench
    write_config(generator=True)
    visa.enabled = True
    save_release(tmp_path)
    if timeout_device == "laser":
        replies["?S"] = b""  # pySerial returns an empty buffer on read timeout.
    else:
        def timeout(command):
            visa.queries.append(command)
            raise TimeoutError("fake VISA read timeout")
        monkeypatch.setattr(visa, "query", timeout)

    assert shutdown_bench.main(["--run-dir", str(tmp_path)]) == 1
    laser, generator, camera, slm = capsys.readouterr().out.splitlines()
    affected, independent = (laser, generator) if timeout_device == "laser" else (generator, laser)
    assert "UNVERIFIED" in affected and "timeout" in affected
    assert ": VERIFIED" in independent
    assert ": VERIFIED" in camera and ": VERIFIED" in slm
    assert events == ["serial.open", "S=0", "L=0", "P=0.01", "?S", "?L", "?P", "?SP", "serial.close"]
    assert visa.writes == [":OUTP1 OFF"] and visa.queries == [":OUTP1?"]
    assert visa.calls[-2:] == ["close", "close"]


def test_independent_generator_continues_when_laser_owned(bench, monkeypatch, capsys):
    write_config, events, visa, _ = bench
    write_config(generator=True)
    visa.enabled = True
    monkeypatch.setattr(shutdown_bench, "_OWNER_WAIT_SECONDS", 0.01)
    with bench_access.bench_access(acquisition=True), bench_access.device_access("laser"):
        assert shutdown_bench.main([]) == 1
        lines = capsys.readouterr().out.splitlines()
        assert "still owns the laser session" in lines[0]
        assert "Generator: VERIFIED" in lines[1]
        assert all("did not release within" in line for line in lines[2:])
        assert not events
        assert visa.writes == [":OUTP1 OFF"]
        with pytest.raises(bench_access.BenchBusyError):
            bench_access.rearm_bench()


@pytest.mark.parametrize("reply,status", [(b"1\r\n", "FAILED"), (b"", "UNVERIFIED")])
def test_setpoint_alone_cannot_verify_laser_and_sdk_failure_is_preserved(bench, tmp_path, capsys, reply, status):
    _, _, _, replies = bench
    replies["?S"] = reply
    save_release(tmp_path, slm="failed")
    assert shutdown_bench.main(["--run-dir", str(tmp_path)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith(f"Laser: {status}") and "?SP=0.01" in lines[0]
    assert "Camera: VERIFIED" in lines[2]
    assert "SLM: FAILED" in lines[3] and "SDK cleanup failed" in lines[3]


@pytest.mark.parametrize("evidence", ["complete", "missing_action", "edited_artifact"])
def test_exited_runner_release_evidence_must_cover_attempted_actions(bench, tmp_path, capsys, evidence):
    capture = tmp_path / "capture-result.json"
    save_release(tmp_path)
    artifact = {"path": str(capture), "sha256": hashlib.sha256(capture.read_bytes()).hexdigest()}
    actions = [{"status": "completed", "artifacts": [artifact]}]
    if evidence == "missing_action":
        actions.append({"status": "failed", "artifacts": []})
    else:
        actions.append({"status": "pending", "artifacts": []})
    if evidence == "edited_artifact":
        capture.write_bytes(capture.read_bytes() + b" ")
    record = {"status": "completed" if evidence != "missing_action" else "partial",
              "raw_data": [artifact],
              "hardware": {"data_origin": "physical", "ended_at": "2026-11-25T12:00:00Z",
                           "cleanup_status": "completed", "actions": actions}}
    (tmp_path / "record.json").write_text(json.dumps(record), encoding="utf-8")

    script = """import sys
from pathlib import Path
from hardware import bench_access
bench_access._LOCK_PATH = Path(sys.argv[1])
with bench_access.bench_access(acquisition=True):
    bench_access.record_bench_run(Path(sys.argv[2]))
"""
    subprocess.run([sys.executable, "-c", script, str(bench_access._LOCK_PATH), str(tmp_path)],
                   check=True, timeout=10)
    complete = evidence == "complete"
    assert shutdown_bench.main([]) == int(not complete)
    laser, _, camera, slm = capsys.readouterr().out.splitlines()
    assert "Laser: VERIFIED" in laser
    assert all(f": {'VERIFIED' if complete else 'UNVERIFIED'}" in line for line in (camera, slm))


def test_runner_owner_can_finish_and_publish_cleanup_during_shutdown(bench, tmp_path, capsys):
    ready, observed = Event(), Event()

    def owner():
        with bench_access.bench_access(acquisition=True):
            bench_access.record_bench_run(tmp_path)
            ready.set()
            if bench_access.BenchCancellation().wait(2):
                save_release(tmp_path)
                observed.set()

    worker = Thread(target=owner)
    worker.start()
    try:
        assert ready.wait(2)
        assert shutdown_bench.main([]) == 0
    finally:
        worker.join(3)
    assert observed.is_set() and not worker.is_alive()
    assert "Camera: VERIFIED" in capsys.readouterr().out


def test_active_one_frame_capture_cancels_before_camera_and_releases_slm(bench, tmp_path, capsys):
    from hardware.capture_once import capture_once

    ready = Event()
    events = []
    laser, _ = laser_for(LaserSettings("COM-FAKE", 2.25, ("set_power", "enable", "open_shutter"),
                                        True, "\r\n", "\r\n", None), events)
    start = laser.start_emission

    def start_and_signal():
        start()
        ready.set()

    laser.start_emission = start_and_signal
    worker = Thread(target=capture_once, kwargs={
        "slm": DeviceFake("slm", events), "camera": DeviceFake("camera", events),
        "laser": laser, "gray_level": 17, "laser_settle_seconds": 5,
        "output_dir": tmp_path / "capture",
    })
    worker.start()
    try:
        assert ready.wait(2)
        assert shutdown_bench.main([]) == 0
    finally:
        worker.join(6)
    assert not worker.is_alive()
    assert "camera.connect" not in events and "slm.disconnect" in events
    assert "Camera: VERIFIED - run record says device was not initialized" in capsys.readouterr().out


def test_active_runner_cannot_enable_outputs_after_direct_shutdown(bench, hardware_setup, monkeypatch, capsys):
    from agents.runner import runner
    from tools.hardware import ActionReport, preflight

    state, cfg, backend, procedure = hardware_setup
    cfg.hardware_backend = "external"
    backend.capabilities.data_origin = "physical"
    write_config, shutdown_laser_events, shutdown_visa, _ = bench
    write_config(generator=True)
    monkeypatch.setattr(shutdown_bench, "_OWNER_WAIT_SECONDS", 0)
    ready, direct_shutdown_finished = Event(), Event()
    laser_events, runner_visa = [], FakeVisa()
    laser, _ = laser_for(LaserSettings("COM-FAKE", 2.25, ("set_power", "enable", "open_shutter"),
                                        True, "\r\n", "\r\n", None), laser_events)
    generator = RigolDG812(GeneratorSettings("USB0::FAKE::INSTR", enabled=True),
                          resource_manager_factory=runner_visa.resource_manager)

    def execute(action, *, cancel_event, **kwargs):
        ready.set()
        assert direct_shutdown_finished.wait(5)
        assert cancel_event.is_set()
        laser.connect()
        with pytest.raises(bench_access.BenchBusyError, match="latched off"):
            laser.start_emission()
        with pytest.raises(bench_access.BenchBusyError, match="latched off"):
            with generator.drive_for_acquisition(LccAcquisitionPlan(2), {}):
                pytest.fail("cancelled Runner reached LCC acquisition")
        return ActionReport(aborted=True, errors=["bench shutdown requested"])

    backend.execute = execute
    review = preflight(procedure, cfg, backend)
    results = []
    worker = Thread(target=lambda: results.append(
        runner(state, procedure.id, config=cfg, backend=backend, execute=True,
               reviewed=review, approval=approved(review))))
    worker.start()
    try:
        assert ready.wait(5)
        assert shutdown_bench.main([]) == 1
        laser_result, generator_result, camera, slm = capsys.readouterr().out.splitlines()
        assert "Laser: VERIFIED" in laser_result and "Generator: VERIFIED" in generator_result
        assert all("did not release" in line for line in (camera, slm))
        assert "L=0" in shutdown_laser_events and shutdown_visa.writes == [":OUTP1 OFF"]
    finally:
        direct_shutdown_finished.set()
        worker.join(5)
    assert not worker.is_alive()
    assert results[0].status == "aborted" and len(backend.cleanup_calls) == 1
    assert "L=1" not in laser_events and "S=1" not in laser_events and not runner_visa.calls


def test_latch_observed_in_separate_process():
    script = """import sys
from pathlib import Path
from hardware import bench_access
bench_access._LOCK_PATH = Path(sys.argv[1])
assert bench_access.BenchCancellation().wait(0.1)
try:
    with bench_access.bench_access(acquisition=True):
        raise AssertionError("latched bench accepted acquisition")
except bench_access.BenchBusyError:
    pass
"""
    bench_access.request_shutdown()
    result = subprocess.run([sys.executable, "-c", script, str(bench_access._LOCK_PATH)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
