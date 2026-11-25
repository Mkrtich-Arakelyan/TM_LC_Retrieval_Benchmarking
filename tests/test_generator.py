import json
from pathlib import Path

import numpy as np
import pytest

from hardware import capture_once
from hardware.bench_access import BenchBusyError, request_shutdown
from hardware.generator import GeneratorError, GeneratorSettings, LccAcquisitionPlan, RigolDG812


class FakeVisa:
    def __init__(self):
        self.queries = []
        self.writes = []
        self.calls = []
        self.waveform = "SIN"
        self.frequency = 1000
        self.amplitude = 5
        self.offset = 1
        self.phase = 39.5
        self.load = "0"
        self.unit = "VPP"
        self.enabled = False
        self.clamp_amplitude = False
        self.fail_off = False

    def open_resource(self, address, *, open_timeout):
        self.calls.append(("open", address, open_timeout))
        return self

    def resource_manager(self):
        self.calls.append("manager")
        return self

    def close(self):
        self.calls.append("close")

    def query(self, command):
        self.queries.append(command)
        return {
            "*IDN?": "RIGOL TECHNOLOGIES,DG812,FAKE,00.02.06.00.01\n",
            ":OUTP1?": "ON" if self.enabled else "OFF",
            ":OUTP1:LOAD?": self.load,
            ":SOUR1:APPL?": f'"{self.waveform},{self.frequency:.6E},{self.amplitude:.6E},'
                            f'{self.offset:.6E},{self.phase:.6E}"',
            ":SOUR1:VOLT:UNIT?": self.unit,
            ":SOUR1:VOLT:HIGH?": "3.5",
            ":SOUR1:VOLT:LOW?": "-1.5",
        }[command]

    def write(self, command):
        self.writes.append(command)
        name, value = command.split()
        if name == ":OUTP1":
            if self.fail_off and value == "OFF":
                raise OSError("fake OFF write failure")
            self.enabled = value == "ON"
        elif name == ":SOUR1:VOLT:UNIT":
            self.unit = value
        elif name == ":SOUR1:FUNC":
            self.waveform = value
        else:
            parameter = {":SOUR1:FREQ": "frequency", ":SOUR1:VOLT:OFFS": "offset",
                         ":SOUR1:VOLT": "amplitude"}[name]
            if not (parameter == "amplitude" and self.clamp_amplitude):
                setattr(self, parameter, float(value))


@pytest.fixture
def fake_generator():
    visa = FakeVisa()
    settings = GeneratorSettings(visa_resource="USB0::FAKE::INSTR", enabled=True)
    return RigolDG812(settings, resource_manager_factory=visa.resource_manager), visa


@pytest.mark.parametrize("load", ["0", "9.900000E+37", "INF", "50"])
def test_identification_and_status_are_read_only(fake_generator, load):
    generator, visa = fake_generator
    visa.load = load
    visa.enabled = True
    generator.connect()
    assert generator.identify().split(",")[1] == "DG812"
    status = generator.read_status()
    assert (status.waveform, status.frequency_hz, status.amplitude_vpp,
            status.offset_v, status.phase_deg) == ("SIN", 1000, 5, 1, 39.5)
    assert status.configured_load == (50 if load == "50" else "High-Z")
    assert status.output_enabled
    assert generator.disconnect() == []
    assert not visa.writes and visa.enabled
    assert visa.calls == ["manager", ("open", "USB0::FAKE::INSTR", 2000), "close", "close"]


@pytest.mark.parametrize("unit", ["VRMS", "DBM"])
def test_reports_vpp_in_other_unit_modes(fake_generator, unit):
    generator, visa = fake_generator
    visa.unit, visa.amplitude, visa.load = unit, 1.5, "50"
    generator.connect()
    assert generator.read_status().amplitude_vpp == 5
    assert visa.unit == unit and not visa.writes
    generator.disconnect()


@pytest.mark.parametrize("amplitude", [0.1, 12])
def test_prepare_accepts_boundaries_and_leaves_ch1_off(fake_generator, amplitude):
    generator, visa = fake_generator
    visa.waveform = "SQU"
    generator.connect()
    actual = generator.prepare_ch1(amplitude)
    assert actual.amplitude_vpp == amplitude
    assert (actual.waveform, actual.frequency_hz, actual.offset_v) == ("SIN", 20_000, 0)
    assert actual.phase_deg == 39.5 and actual.configured_load == "High-Z"
    assert actual.output_enabled is False
    assert visa.writes == [
        ":SOUR1:FUNC SIN", ":SOUR1:FREQ 20000", ":SOUR1:VOLT:OFFS 0",
        ":SOUR1:VOLT:UNIT VPP", f":SOUR1:VOLT {float(amplitude)!r}",
    ]
    generator.disconnect()


@pytest.mark.parametrize("amplitude", [None, True, "1", float("nan"), float("inf"),
                                       -float("inf"), 0, 0.099, 12.01, 10**400])
def test_invalid_amplitude_fails_before_visa_io(fake_generator, amplitude):
    generator, visa = fake_generator
    generator.connect()
    with pytest.raises(ValueError, match="amplitude"):
        generator.prepare_ch1(amplitude)
    with pytest.raises(ValueError, match="amplitude"):
        LccAcquisitionPlan(amplitude)
    assert not visa.queries and not visa.writes
    generator.disconnect()


def test_prepare_refuses_to_configure_live_output(fake_generator):
    generator, visa = fake_generator
    visa.enabled = True
    generator.connect()
    with pytest.raises(GeneratorError, match="verified OFF"):
        generator.prepare_ch1(2)
    assert not visa.writes
    generator.disconnect()


def test_prepare_refuses_unknown_output_state(fake_generator, monkeypatch):
    generator, visa = fake_generator
    query = visa.query
    monkeypatch.setattr(visa, "query", lambda cmd: "?" if cmd == ":OUTP1?" else query(cmd))
    generator.connect()
    with pytest.raises(GeneratorError, match="Unrecognized DG812 CH1 output state"):
        generator.prepare_ch1(2)
    assert not visa.writes
    generator.disconnect()


def test_readback_mismatch_blocks_enable_and_attempts_shutdown(fake_generator):
    generator, visa = fake_generator
    visa.clamp_amplitude = True
    record = {}
    with pytest.raises(GeneratorError, match="readback mismatch"):
        with generator.drive_for_acquisition(LccAcquisitionPlan(12), record):
            pytest.fail("mismatched CH1 reached capture")
    assert ":OUTP1 ON" not in visa.writes
    assert visa.writes[-1] == ":OUTP1 OFF"
    assert record["shutdown"] == {"confirmed": True, "errors": []}
    assert visa.calls[-2:] == ["close", "close"]


def test_stop_latch_blocks_output_after_preparation(fake_generator, monkeypatch):
    generator, visa = fake_generator
    prepare = generator.prepare_ch1

    def prepare_then_cancel(amplitude):
        status = prepare(amplitude)
        request_shutdown()
        return status

    monkeypatch.setattr(generator, "prepare_ch1", prepare_then_cancel)
    record = {}
    with pytest.raises(BenchBusyError, match="latched off"):
        with generator.drive_for_acquisition(LccAcquisitionPlan(2), record):
            pytest.fail("cancelled drive reached capture")
    assert ":OUTP1 ON" not in visa.writes
    assert record["shutdown"]["confirmed"]


@pytest.mark.parametrize("failure", [None, OSError("capture failure"), KeyboardInterrupt()])
def test_acquisition_records_settings_and_cleans_up(fake_generator, failure):
    generator, visa = fake_generator
    record = {}

    def acquire():
        with generator.drive_for_acquisition(LccAcquisitionPlan(2), record):
            assert visa.enabled and visa.amplitude == 2
            with pytest.raises(GeneratorError, match="verified OFF"):
                generator.prepare_ch1(3)
            if failure is not None:
                raise failure

    if failure is None:
        acquire()
    else:
        with pytest.raises(type(failure)):
            acquire()
    assert record["requested"]["amplitude_vpp"] == record["readback"]["amplitude_vpp"] == 2
    assert record["prepared"]["output_enabled"] is False
    assert record["readback"]["output_enabled"] is True
    assert record["shutdown"] == {"confirmed": True, "errors": []}
    assert visa.writes[-2:] == [":OUTP1 ON", ":OUTP1 OFF"]
    assert not visa.enabled and visa.calls[-2:] == ["close", "close"]


def test_unconfirmed_shutdown_is_explicit_even_when_handles_close(fake_generator):
    generator, visa = fake_generator
    visa.fail_off = True
    record = {}
    with pytest.raises(GeneratorError, match="capture failure.*shutdown unconfirmed"):
        with generator.drive_for_acquisition(LccAcquisitionPlan(2), record):
            raise OSError("capture failure")
    assert record["shutdown"]["confirmed"] is False
    assert visa.enabled and visa.calls[-2:] == ["close", "close"]


def test_disabled_generator_never_constructs_a_visa_manager(fake_generator):
    _, visa = fake_generator
    generator = RigolDG812(GeneratorSettings("USB0::FAKE::INSTR"),
                          resource_manager_factory=visa.resource_manager)
    with pytest.raises(GeneratorError, match="disabled"):
        generator.connect()
    assert not visa.calls


@pytest.mark.parametrize("with_lcc,fail_off", [(False, False), (True, False), (True, True)])
def test_approved_capture_cli_optional_generator(tmp_path, monkeypatch, fake_generator,
                                                 with_lcc, fail_off):
    generator, visa = fake_generator
    visa.fail_off = fail_off
    template = Path(__file__).resolve().parents[1] / "configs/capture-once.example.json"
    data = json.loads(template.read_text(encoding="utf-8"))
    assert data["generator"]["enabled"] is False and data["lcc"] is None
    data["camera_trigger_confirmed"] = data["laser"]["startup_confirmed"] = True
    data["slm"]["return_codes"] = {"load_lut": 0, "write_image": 0, "image_complete": 0}
    if with_lcc:
        data["generator"]["enabled"] = True
        data["lcc"] = {"amplitude_vpp": 0.1}
    config = tmp_path / "lab.json"
    config.write_text(json.dumps(data), encoding="utf-8")

    class FakeCaptureDevices:
        image_shape = (2, 2)
        transcript = []

        def __init__(self, settings):
            pass

        def connect(self):
            pass

        def start_emission(self):
            pass

        def display_image(self, image):
            pass

        def capture_frame(self):
            assert visa.enabled == with_lcc
            return np.zeros(self.image_shape, dtype=np.uint8)

        def disconnect(self):
            return {"confirmed": True, "serial_close": "closed"}

    def make_generator(settings):
        assert with_lcc, "disabled capture constructed a generator adapter"
        return generator

    for name in ("MeadowlarkSlm", "CeleraCamera", "LabLaser"):
        monkeypatch.setattr(capture_once, name, FakeCaptureDevices)
    monkeypatch.setattr(capture_once, "RigolDG812", make_generator)
    output = tmp_path / "output"
    assert capture_once.main(["--operator-approved", "--config", str(config),
                              "--output", str(output)]) == int(fail_off)
    result = json.loads((output / "capture-result.json").read_text(encoding="utf-8"))
    assert (output / "camera-frame.npy").is_file()
    if with_lcc:
        assert result["lcc"]["requested"]["amplitude_vpp"] == 0.1
        assert result["lcc"]["readback"]["amplitude_vpp"] == 0.1
        assert result["cleanup"]["generator"]["status"] == ("UNCONFIRMED" if fail_off else "confirmed")
    else:
        assert "lcc" not in result and not visa.calls
