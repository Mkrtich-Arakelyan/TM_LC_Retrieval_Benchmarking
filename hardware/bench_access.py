"""Nonblocking process lock shared by bench acquisition and status checks."""

from contextlib import contextmanager
import errno
import os
from pathlib import Path
from threading import Event, local
import time


_LOCK_PATH = Path(__file__).resolve().parents[1] / ".optical-bench.lock"
_owner = local()


class BenchBusyError(RuntimeError):
    pass


@contextmanager
def _lock(path, busy):
    # Keep the file in place. The OS releases the lock when the handle closes.
    with path.open("ab") as handle:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise BenchBusyError(busy) from exc
            raise
        yield


def _sidecar(suffix):
    return _LOCK_PATH.with_suffix(suffix)


def shutdown_access():
    return _lock(_sidecar(".shutdown.lock"), "another shutdown or re-arm is in progress")


def device_access(device):
    return _lock(_sidecar(f".{device}.lock"), f"another process still owns the {device} session")


def request_shutdown():
    _sidecar(".stop").touch(exist_ok=True)


def rearm_bench():
    with shutdown_access(), bench_access(), device_access("laser"), device_access("generator"):
        _sidecar(".stop").unlink(missing_ok=True)


def _shutdown_requested():
    try:
        _sidecar(".stop").stat()
    except FileNotFoundError:
        return False
    return True


def require_acquisition_allowed():
    if _shutdown_requested():
        raise BenchBusyError("bench shutdown requested; acquisitions are latched off")


class BenchCancellation(Event):
    """Combine the caller's cancellation event with the cross-process stop latch."""

    def __init__(self, caller=None):
        super().__init__()
        self._caller = caller

    def is_set(self):
        if super().is_set() or (self._caller is not None and self._caller.is_set()):
            return True
        try:
            return _shutdown_requested()
        except OSError:
            return True  # Acquisition cannot establish that the stop latch is clear.

    def set(self):
        super().set()
        if self._caller is not None:
            self._caller.set()

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            remaining = 0.05 if deadline is None else deadline - time.monotonic()
            if remaining <= 0:
                return False
            super().wait(min(0.05, remaining))
        return True


def record_bench_run(directory):
    """Remember the physical owner's run directory before it touches hardware."""
    if getattr(_owner, "run_recorded", False):
        return  # A delegated capture belongs to the outer Runner acquisition.
    path = _sidecar(".run")
    pending = _sidecar(".run.tmp")
    pending.write_text(str(directory.resolve()), encoding="utf-8")
    os.replace(pending, path)
    _owner.run_recorded = True


def latest_bench_run():
    return Path(_sidecar(".run").read_text(encoding="utf-8"))


@contextmanager
def bench_access(*, acquisition=False):
    busy = "bench is in use by an acquisition or another status check"
    held = getattr(_owner, "acquisition", None)
    if held is not None:
        if not (held and acquisition):
            raise BenchBusyError(busy)
        require_acquisition_allowed()
        yield
        return
    with _lock(_LOCK_PATH, busy):
        if acquisition:
            require_acquisition_allowed()
        _owner.acquisition = acquisition
        _owner.run_recorded = False
        try:
            yield
        finally:
            del _owner.acquisition
            del _owner.run_recorded
