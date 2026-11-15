"""Shared fixed geometry for new benches, and validation of recorded geometry."""

import numpy as np

FRAME_SIDE = 20
SLM_SIDE = 8
BLOCK_SIDE = 2
GRID_SIDE = 4
INPUT_COUNT = DETECTOR_COUNT = 16
GEOMETRY_VERSION = 2


def detector_regions(side=FRAME_SIDE):
    if side != FRAME_SIDE:
        raise ValueError("New simulated runs require a 20 x 20 camera frame")
    origin = (FRAME_SIDE - SLM_SIDE) // 2
    return [(y, y + BLOCK_SIDE, x, x + BLOCK_SIDE)
            for y in range(origin, origin + SLM_SIDE, BLOCK_SIDE)
            for x in range(origin, origin + SLM_SIDE, BLOCK_SIDE)]


def geometry_metadata():
    return {
        "geometry_version": GEOMETRY_VERSION,
        "camera_frame_shape": [FRAME_SIDE, FRAME_SIDE],
        "camera_crop": [6, 14, 6, 14],  # Half-open [y0,y1,x0,x1], zero-based.
        "slm_shape": [SLM_SIDE, SLM_SIDE],
        "slm_input_regions": [[y, y + 2, x, x + 2] for y in range(0, 8, 2) for x in range(0, 8, 2)],
        "regions": [list(region) for region in detector_regions()],
        "detector_order": list(range(DETECTOR_COUNT)),
        "detector_ordering": "left to right, top to bottom",
    }


def slm_phase_pattern(phases):
    """Expand sixteen region controls onto the 8 x 8 phase-only active area."""
    return np.asarray(phases).reshape(GRID_SIDE, GRID_SIDE).repeat(2, axis=0).repeat(2, axis=1)


def recorded_regions(manifest, frame_side):
    """Use saved row-to-region mapping, including older measurement layouts."""
    regions = manifest.get("regions")
    if not isinstance(regions, list) or not regions or len(regions) > frame_side**2 // 4:
        raise ValueError("Missing or invalid recorded detector geometry")
    covered = set()
    for region in regions:
        if not isinstance(region, (list, tuple)) or len(region) != 4 or any(type(v) is not int for v in region):
            raise ValueError("Recorded detector bounds must be four integer coordinates")
        y0, y1, x0, x1 = region
        if not (0 <= y0 < y1 <= frame_side and 0 <= x0 < x1 <= frame_side and y1-y0 == x1-x0 == 2):
            raise ValueError("Recorded detector blocks must be 2 x 2 and inside the camera frame")
        pixels = {(y, x) for y in range(y0, y1) for x in range(x0, x1)}
        if covered & pixels:
            raise ValueError("Recorded detector blocks overlap")
        covered.update(pixels)
    if "geometry_version" in manifest:
        for key, expected in geometry_metadata().items():
            if manifest.get(key) != expected:
                raise ValueError(f"Recorded detector geometry mismatch: {key}")
        if frame_side != FRAME_SIDE:
            raise ValueError("Recorded camera dimensions disagree with geometry")
    return regions
