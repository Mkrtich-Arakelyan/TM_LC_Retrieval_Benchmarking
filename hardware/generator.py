"""RIGOL DG812 CH1 adapter for the LCC. PyVISA is loaded only on connect."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from importlib import import_module
import math
from threading import RLock
from typing import Optional, Union

from hardware.bench_access import device_access, require_acquisition_allowed


_WAVEFORM = "SIN"
_FREQUENCY_HZ = 20_000
_OFFSET_V = 0


class GeneratorError(RuntimeError):
    """The generator could not establish or verify the requested state."""


def _amplitude(value):
    if type(value) not in (int, float) or not 0.1 <= value <= 12:
        raise ValueError("LCC amplitude must be a finite number in [0.1, 12] Vpp")
    return float(value)


@dataclass(frozen=True)
class LccAcquisitionPlan:
    """One amplitude for the entire acquisition, including all frames in a TM dataset."""

    amplitude_vpp: float

    def __post_init__(self):
        object.__setattr__(self, "amplitude_vpp", _amplitude(self.amplitude_vpp))


@dataclass(frozen=True)
class GeneratorSettings:
    visa_resource: str
    enabled: bool = False
    timeout_ms: int = 2000

    def __post_init__(self):
        if not isinstance(self.visa_resource, str) or not self.visa_resource.strip():
            raise ValueError("A VISA resource address is required")
        if type(self.enabled) is not bool:
            raise ValueError("Generator enabled must be a boolean")
        if type(self.timeout_ms) is not int or self.timeout_ms <= 0:
            raise ValueError("Generator timeout_ms must be a positive integer")


@dataclass(frozen=True)
class ChannelStatus:
    waveform: str
    frequency_hz: Optional[float]
    amplitude_vpp: Optional[float]
    offset_v: Optional[float]
    phase_deg: Optional[float]
    configured_load: Union[float, str]  # ohms for a finite load, otherwise High-Z
    output_enabled: bool


def _number(reply):
    try:
        value = float(reply)
    except ValueError as exc:
        raise GeneratorError(f"Invalid DG812 numeric reply {reply!r}") from exc
    if not math.isfinite(value):
        raise GeneratorError(f"Non-finite DG812 numeric reply {reply!r}")
    return value


def _load(reply):
    if reply in ("INF", "INFINITY", "HIGHZ", "HIGH-Z"):
        return "High-Z"
    value = _number(reply)
    if value in (0, 9.9e37):  # Firmware zero and the DG800 guide's infinity sentinel.
        return "High-Z"
    if not 1 <= value <= 10_000:
        raise GeneratorError(f"Invalid DG812 load reply {reply!r}")
    return value


class RigolDG812:
    def __init__(self, settings: GeneratorSettings, *, resource_manager_factory=None):
        self.settings = settings
        self._factory = resource_manager_factory
        self._manager = None
        self._instrument = None
        self._ownership = None
        self._prepared = None
        self._mutex = RLock()

    def connect(self):
        with self._mutex:
            if not self.settings.enabled:
                raise GeneratorError("DG812 is disabled in the bench configuration")
            if self._manager is not None:
                raise GeneratorError("DG812 VISA session is already open")
            ownership = device_access("generator")
            ownership.__enter__()
            self._ownership = ownership
            try:
                factory = self._factory or import_module("pyvisa").ResourceManager
                self._manager = factory()
                self._instrument = self._manager.open_resource(
                    self.settings.visa_resource,
                    open_timeout=self.settings.timeout_ms,
                )
                self._instrument.timeout = self.settings.timeout_ms
                self._instrument.read_termination = "\n"
                self._instrument.write_termination = "\n"
            except BaseException:
                self.disconnect()
                raise
            return self

    def _session(self):
        if self._instrument is None:
            raise GeneratorError("DG812 VISA session is not open")
        return self._instrument

    def _query(self, command):
        return self._session().query(command).strip()

    def identify(self):
        with self._mutex:
            return self._query("*IDN?")

    def _output_enabled(self):
        reply = self._query(":OUTP1?").upper()
        if reply in ("OFF", "0"):
            return False
        if reply in ("ON", "1"):
            return True
        raise GeneratorError(f"Unrecognized DG812 CH1 output state {reply!r}")

    def read_output_enabled(self):
        """Query CH1 output state independently of the waveform settings."""
        with self._mutex:
            return self._output_enabled()

    def read_status(self):
        """Read CH1 without changing its output state or voltage unit."""
        with self._mutex:
            raw = self._query(":SOUR1:APPL?")
            fields = [field.strip().upper() for field in raw.strip('"').split(",")]
            if len(fields) != 5 or not fields[0]:
                raise GeneratorError(f"Invalid DG812 CH1 waveform reply {raw!r}")
            waveform = fields[0]
            frequency, amplitude, offset, phase = (
                None if field == "DEF" else _number(field) for field in fields[1:]
            )
            if amplitude is not None:
                unit = self._query(":SOUR1:VOLT:UNIT?").upper()
                if unit in ("VRMS", "DBM"):
                    # DG800 guide §2-138 gives Vpp as high minus low. This
                    # also works for non-sinusoidal waveforms.
                    high = _number(self._query(":SOUR1:VOLT:HIGH?"))
                    low = _number(self._query(":SOUR1:VOLT:LOW?"))
                    amplitude = high - low
                elif unit != "VPP":
                    raise GeneratorError(f"Unrecognized DG812 amplitude unit {unit!r}")
                if not math.isfinite(amplitude) or amplitude < 0:
                    raise GeneratorError(f"Invalid DG812 CH1 peak-to-peak amplitude {amplitude!r}")
            return ChannelStatus(waveform, frequency, amplitude, offset, phase,
                                 _load(self._query(":OUTP1:LOAD?").upper()), self._output_enabled())

    def prepare_ch1(self, amplitude_vpp):
        """Configure and verify the fixed LCC drive while CH1 is off."""
        with self._mutex:
            self._prepared = None
            amplitude = _amplitude(amplitude_vpp)
            require_acquisition_allowed()
            before = self.read_status()
            if before.output_enabled:
                raise GeneratorError("CH1 must be verified OFF before configuration")
            session = self._session()
            # Individual setters avoid assigning an APPL phase default.
            session.write(f":SOUR1:FUNC {_WAVEFORM}")
            session.write(f":SOUR1:FREQ {_FREQUENCY_HZ}")
            session.write(f":SOUR1:VOLT:OFFS {_OFFSET_V}")
            session.write(":SOUR1:VOLT:UNIT VPP")
            if self._query(":SOUR1:VOLT:UNIT?").upper() != "VPP":
                raise GeneratorError("DG812 CH1 did not accept VPP amplitude units")
            session.write(f":SOUR1:VOLT {amplitude!r}")
            actual = self.read_status()
            # APPL? returns seven significant digits.
            if (actual.waveform != _WAVEFORM or actual.frequency_hz != _FREQUENCY_HZ
                    or actual.offset_v != _OFFSET_V or actual.output_enabled
                    or actual.amplitude_vpp is None
                    or not 0.1 <= actual.amplitude_vpp <= 12
                    or not math.isclose(actual.amplitude_vpp, amplitude, rel_tol=5e-7)
                    or (before.phase_deg is not None and actual.phase_deg != before.phase_deg)
                    or actual.phase_deg is None or actual.configured_load != before.configured_load):
                raise GeneratorError(f"DG812 CH1 readback mismatch for {amplitude} Vpp: {actual}")
            self._prepared = actual
            return actual

    def _enable_prepared_ch1(self):
        expected, self._prepared = self._prepared, None
        if expected is None or self.read_status() != expected:
            raise GeneratorError("CH1 requires matching verified preparation before enabling output")
        require_acquisition_allowed()
        self._session().write(":OUTP1 ON")
        actual = self.read_status()
        if actual != replace(expected, output_enabled=True):
            raise GeneratorError(f"DG812 CH1 readback mismatch after enabling output: {actual}")
        return actual

    def _shutdown(self):
        enabled = None
        errors = []
        try:
            self._session().write(":OUTP1 OFF")
        except BaseException as exc:
            errors.append(f"CH1 OFF command failed: {exc}")
        try:
            enabled = self._output_enabled()
            if enabled:
                errors.append("CH1 shutdown unconfirmed: output still ON")
        except BaseException as exc:
            errors.append(f"CH1 shutdown unconfirmed: {exc}")
        return enabled, errors

    def disconnect(self):
        """Release a read-only session. This does not confirm output is off."""
        with self._mutex:
            handles = (("instrument", self._instrument), ("resource manager", self._manager))
            self._instrument = self._manager = self._prepared = None
            errors = []
            for name, handle in handles:
                if handle is not None:
                    try:
                        handle.close()
                    except BaseException as exc:
                        errors.append(f"Could not close VISA {name}: {exc}")
            if self._ownership is not None:
                self._ownership.__exit__(None, None, None)
                self._ownership = None
            return errors

    def shutdown_once(self):
        """Take exclusive VISA ownership, turn off CH1, and verify output state."""
        with self._mutex:
            self.connect()
            try:
                enabled, errors = self._shutdown()
            finally:
                close_errors = self.disconnect()
            return {"output_enabled": enabled, "errors": errors + close_errors}

    @contextmanager
    def drive_for_acquisition(self, plan: LccAcquisitionPlan, record: dict):
        """Runner or operator-approved procedure only. Hold CH1 for one dataset."""
        if not isinstance(plan, LccAcquisitionPlan):
            raise TypeError("An LccAcquisitionPlan is required")
        with self._mutex:
            require_acquisition_allowed()
            record["requested"] = dict(asdict(plan), waveform=_WAVEFORM,
                                       frequency_hz=_FREQUENCY_HZ, offset_v=_OFFSET_V)
            self.connect()
            failure = None
            try:
                require_acquisition_allowed()
                record["identification"] = self.identify()
                record["prepared"] = asdict(self.prepare_ch1(plan.amplitude_vpp))
                record["readback"] = asdict(self._enable_prepared_ch1())
                yield
            except BaseException as exc:
                failure = exc
                raise
            finally:
                try:
                    enabled, errors = self._shutdown()
                finally:
                    close_errors = self.disconnect()
                confirmed = enabled is False
                record["shutdown"] = {"confirmed": confirmed, "errors": errors + close_errors}
                if not confirmed or errors or close_errors:
                    message = ". ".join(errors + close_errors)
                    if failure is not None:
                        message = f"{type(failure).__name__}: {failure}. {message}"
                    raise GeneratorError(message) from failure
