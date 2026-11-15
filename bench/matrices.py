"""Private matrix inputs and source integrity; never used by retrieval solvers."""

import hashlib
import json
from pathlib import Path

import numpy as np

from bench.config import BenchConfig
from bench.geometry import DETECTOR_COUNT, INPUT_COUNT


def _source(config):
    path = Path(config.detector_matrix_path)
    try:
        with path.open("rb") as file:
            content = file.read(1024 * 1024 + 1)
    except OSError as exc:
        raise ValueError(f"Cannot read detector matrix file {path}: {exc.strerror}") from exc
    if len(content) > 1024 * 1024:
        raise ValueError(f"Detector matrix file exceeds 1 MiB: {path}")
    digest = hashlib.sha256(content).hexdigest()
    if config.detector_matrix_sha256 is not None and digest != config.detector_matrix_sha256:
        raise ValueError(f"Detector matrix file changed since workflow setup: {path}; restore the original file or start a new workflow")
    return content, digest


def verify_matrix_source(config):
    """Check a pinned source without exposing or reloading its coefficients."""
    if config.detector_matrix_path is not None:
        if config.detector_matrix_sha256 is None:
            raise ValueError("Known detector matrix has no saved fingerprint; start a new workflow to pin it")
        _source(config)


def load_detector_matrix(config):
    """Return validated complex [16,16] rows and the source byte hash."""
    config.require_current_layout()
    content, digest = _source(config)
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate detector matrix JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(content.decode("utf-8-sig"), object_pairs_hook=unique_keys)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid detector matrix JSON: {config.detector_matrix_path}") from exc
    if not isinstance(value, dict) or set(value) != {"real", "imag"}:
        raise ValueError("Detector matrix JSON must contain exactly 'real' and 'imag' arrays")
    parts = []
    for key in ("real", "imag"):
        rows = value[key]
        if (not isinstance(rows, list) or len(rows) != DETECTOR_COUNT
                or any(not isinstance(row, list) or len(row) != INPUT_COUNT for row in rows)):
            raise ValueError(f"Detector matrix {key} must have shape [16, 16] in detector order")
        if any(type(number) not in (int, float) for row in rows for number in row):
            raise ValueError(f"Detector matrix {key} must contain finite JSON numbers (no strings or booleans)")
        try:
            array = np.asarray(rows, dtype=np.float64)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"Detector matrix {key} must contain finite numbers") from exc
        if not np.isfinite(array).all():
            raise ValueError(f"Detector matrix {key} must contain finite numbers")
        parts.append(array)
    return parts[0] + 1j * parts[1], digest


def pin_detector_matrix(config: BenchConfig, *, base_directory=None) -> BenchConfig:
    """Resolve and pin a new input, or verify an already pinned workflow input."""
    if config.detector_matrix_path is None:
        return config
    path = Path(config.detector_matrix_path)
    if not path.is_absolute():
        path = (Path(base_directory) if base_directory is not None else Path.cwd()) / path
    resolved = config.model_copy(update={"detector_matrix_path": str(path.resolve())})
    _, digest = load_detector_matrix(resolved)
    return resolved.model_copy(update={"detector_matrix_sha256": digest})
