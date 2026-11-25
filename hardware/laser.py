"""Coherent Verdi V-5 serial adapter; no port is opened on import.

Reply syntax/meanings: Verdi V-2/V-5/V-6 Operator's Manual, Tables 5-1/5-4.
Lab-approved startup order and actual controller echo/prompt remain unconfirmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import math
import re
from threading import RLock

from hardware.bench_access import device_access, require_acquisition_allowed


SHUTDOWN_EXPECTED = {"?S": 0, "?L": 0, "?SP": 0.01}
# The second spelling was observed in the lab's read-only status output.
# Recognizing framing does not establish controller identity or new commands.
REPLY_PROMPTS = ("Verdi>", "Viper V-5>")


class LaserError(RuntimeError):
    pass


def _reply_payload(instruction: str, raw: bytes) -> str:
    """Remove only known framing, optional prompt and the exact instruction echo."""
    if not isinstance(raw, bytes) or not raw.endswith(b"\r\n") or len(raw) > 4096:
        raise LaserError("Unconfirmed Verdi reply: missing CRLF or invalid byte buffer")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise LaserError("Unconfirmed Verdi reply: non-ASCII data") from exc
    # Table 5-1 error text must never be reduced to an embedded numeric value.
    if "error" in text.lower():
        raise LaserError(f"Unconfirmed Verdi error reply: {text!r}")
    text = text.strip(" \t\r\n")
    def strip_leading_prompts(value):
        while value.startswith(REPLY_PROMPTS):
            prompt = next(p for p in REPLY_PROMPTS if value.startswith(p))
            value = value[len(prompt):].lstrip(" \t\r\n")
        return value

    # A previous reply can leave a prompt-only line ahead of this query's
    # prompt/echo. Strip only recognized prompts, never unknown text or values.
    text = strip_leading_prompts(text)
    if text.startswith(instruction):
        text = text[len(instruction):].lstrip(" \t\r\n")
        text = strip_leading_prompts(text)
    while text.endswith(REPLY_PROMPTS):
        prompt = next(p for p in REPLY_PROMPTS if text.endswith(p))
        text = text[:-len(prompt)].rstrip(" \t\r\n")
    return text


def parse_verdi_reply(query: str, raw: bytes) -> int | float:
    """Strict typed parsing for ?L, ?S, ?P (measured W), ?SP (setpoint W).

    Accept optional prompt/echo, inline or on separate CRLF lines. Empty,
    ambiguous, error and malformed replies raise LaserError, never imply off.
    """
    if query not in ("?L", "?S", "?P", "?SP"):
        raise ValueError(f"Unsupported Verdi status query: {query}")
    payload = _reply_payload(query, raw)
    if query in ("?L", "?S"):
        allowed = ("0", "1", "2") if query == "?L" else ("0", "1")
        if payload in allowed:
            return int(payload)
    elif re.fullmatch(r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)", payload):
        value = float(payload)
        if math.isfinite(value):
            return value
    raise LaserError(f"Unconfirmed {query} reply: {payload!r}")


@dataclass(frozen=True)
class LaserSettings:
    port: str
    operating_power: float
    startup_order: tuple[str, ...]
    startup_confirmed: bool
    command_terminator: str
    response_terminator: str
    echo_commands: bool | None
    timeout_seconds: float = 2.0
    # Accept old null configuration keys, but never override documented meanings.
    shutdown_replies: dict[str, str] | None = None
    command_ack: str | None = None

    def __post_init__(self):
        if not isinstance(self.port, str) or not self.port.strip():
            raise ValueError("A serial port is required")
        if (type(self.operating_power) not in (int, float)
                or not math.isfinite(self.operating_power) or not 1.0 <= self.operating_power <= 2.5):
            raise ValueError("operating_power must be in [1.0, 2.5] watts")
        object.__setattr__(self, "startup_order", tuple(self.startup_order))
        if (len(self.startup_order) != 3 or set(self.startup_order) !=
                {"enable", "open_shutter", "set_power"}):
            raise ValueError("startup_order must contain enable, open_shutter, set_power exactly once")
        if type(self.startup_confirmed) is not bool:
            raise ValueError("startup_confirmed must be bool")
        if self.echo_commands is not None and type(self.echo_commands) is not bool:
            raise ValueError("echo_commands must be bool or None (unconfirmed)")
        if self.command_terminator != "\r\n" or self.response_terminator != "\r\n":
            raise ValueError("Verdi command and response terminators must be CRLF")
        if (type(self.timeout_seconds) not in (int, float)
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be finite and positive")
        if self.shutdown_replies is not None or self.command_ack is not None:
            raise ValueError("Remove custom shutdown_replies/command_ack; Verdi reply semantics are documented")


class LabLaser:
    def __init__(self, settings: LaserSettings, *, serial_factory=None):
        self.settings = settings
        self._factory = serial_factory
        self._serial = None
        self._ownership = None
        self._mutex = RLock()
        self.transcript = []
        self.last_cleanup = None

    def connect(self):
        """Open 19200 baud, eight bits, no parity, one stop bit, no flow control."""
        with self._mutex:
            if self._serial is not None:
                raise LaserError("Laser serial port is already open")
            try:
                self._open_port()
            except BaseException:
                self.disconnect()
                raise
            return self

    def _open_port(self):
        ownership = device_access("laser")
        ownership.__enter__()
        self._ownership = ownership
        try:
            factory = self._factory or import_module("serial").Serial
            self._serial = factory(port=None, baudrate=19200, bytesize=8, parity="N",
                stopbits=1, xonxoff=False, rtscts=False, dsrdtr=False,
                timeout=self.settings.timeout_seconds, write_timeout=self.settings.timeout_seconds)
            self._serial.port = self.settings.port
            self._serial.open()
        except BaseException:
            if self._serial is None:
                self._close_port()
            raise

    def _close_port(self):
        if self._serial is not None:
            self._serial.close()
            self._serial = None
        if self._ownership is not None:
            self._ownership.__exit__(None, None, None)
            self._ownership = None

    def read_status_once(self):
        """Open a query-only serial session and close it, including on errors."""
        with self._mutex:
            if self._serial is not None:
                raise LaserError("Laser serial port is already open")
            try:
                self._open_port()
                return self.read_status()
            finally:
                self._close_port()

    def _exchange(self, command, *, query=False):
        record = {"command": command, "reply": None, "written": False, "raw_replies_hex": []}
        self.transcript.append(record)
        try:
            if self._serial is None or not self._serial.is_open:
                raise LaserError("Laser serial port is not open")
            payload = (command + self.settings.command_terminator).encode("ascii")
            if self._serial.write(payload) != len(payload):
                raise LaserError("Incomplete serial command write")
            record["written"] = True
            response = b""
            # Bounded framing-only lines allow an echoed query and prompt on
            # separate lines. Never skip malformed data to find a later number.
            for _ in range(4):
                raw = self._serial.read_until(expected=b"\r\n", size=4096 - len(response))
                record["raw_replies_hex"].append(raw.hex())
                if not raw or not raw.endswith(b"\r\n") or len(response) + len(raw) > 4096:
                    raise LaserError("Incomplete/overlong laser reply (timeout or framing mismatch)")
                response += raw
                payload = _reply_payload(command, response)
                if not query:
                    if payload:
                        raise LaserError(f"Unexpected Verdi command acknowledgement: {payload!r}")
                    record["reply"] = response.decode("ascii")
                    return record["reply"]
                if payload:
                    value = parse_verdi_reply(command, response)
                    record.update(reply=response.decode("ascii"), value=value)
                    return value
            raise LaserError("Unconfirmed Verdi reply: no data after prompt/echo")
        except Exception as exc:
            record["error"] = str(exc)
            raise

    def start_emission(self):
        """Execute only the caller's confirmed startup order, within 1.0–2.5 W."""
        with self._mutex:
            if not self.settings.startup_confirmed:
                raise LaserError("Laser startup order is not lab-confirmed; emission was not requested")
            commands = {"enable": "L=1", "open_shutter": "S=1",
                        "set_power": f"P={self.settings.operating_power:g}"}
            try:
                for step in self.settings.startup_order:
                    require_acquisition_allowed()
                    self._exchange(commands[step])
            except BaseException:
                self.disconnect()
                raise

    def read_status(self):
        """Query each documented state independently, preserving failed checks."""
        with self._mutex:
            states = {}
            for query in ("?S", "?L", "?P", "?SP"):
                state = {"query": query, "valid": False}
                try:
                    value = self._exchange(query, query=True)
                    state.update(value=value, valid=True)
                    if query == "?L":
                        state["state"] = {0: "standby", 1: "on", 2: "off_due_to_fault"}[value]
                    elif query == "?S":
                        state["state"] = {0: "closed", 1: "open"}[value]
                    else:
                        state.update(units="W", meaning="measured_output" if query == "?P" else "power_setpoint")
                except Exception as exc:
                    state["error"] = str(exc)
                states[query] = state
            return states

    def shutdown(self):
        """Send off commands, require closed/standby/0.01 W setpoint readback.

        Measured ?P is recorded separately, not used as the ?SP setpoint. All
        status queries must be valid. Off-due-to-fault is not normal standby.
        """
        with self._mutex:
            report = {"confirmed": False, "actions": [], "checks": []}
            for command in ("S=0", "L=0", "P=0.01"):
                try:
                    self._exchange(command)
                    report["actions"].append({"command": command, "status": "sent"})
                except Exception as exc:
                    report["actions"].append({"command": command, "status": "failed", "error": str(exc)})
            for query, state in self.read_status().items():
                check = dict(state)
                expected = SHUTDOWN_EXPECTED.get(query)
                check.update(expected=expected, confirmed=state["valid"] and
                             (expected is None or state["value"] == expected))
                if query == "?P":
                    check["role"] = "measured power diagnostic; not setpoint verification"
                report["checks"].append(check)
            report["confirmed"] = (all(a["status"] == "sent" for a in report["actions"])
                                   and all(c["confirmed"] for c in report["checks"]))
            return report

    def disconnect(self):
        """Attempt shutdown and close serial; return each result without hiding failures."""
        with self._mutex:
            if self._serial is None:
                return self.last_cleanup or {"confirmed": False, "reason": "No serial connection was initialized"}
            report = {"confirmed": False}
            try:
                report = self.shutdown()
            finally:
                try:
                    self._close_port()
                    report["serial_close"] = "closed"
                except Exception as exc:
                    report["serial_close"] = f"failed: {exc}"
                self.last_cleanup = report
            return report

    def shutdown_once(self):
        """Take exclusive serial ownership, execute the shutdown sequence, and close."""
        with self._mutex:
            self.connect()
            return self.disconnect()
