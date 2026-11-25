# ALKERIA Celera camera adapter

`hardware/camera.py` implements only camera access. It does not configure the SLM,
synchronize devices, run experiments or integrate with the Runner/MCP. Importing
the module and constructing the adapter do not import `alkusb3` or open hardware.

## Public interface

- `CameraSettings`: camera index, ROI width/height and offsets, horizontal/vertical
  binning, video mode, pixel format, shutter, frame rate, and `CameraTrigger`.
- `CameraTrigger`: enabled flag, external input, edge-versus-level detection,
  native-unit delay and input debounce time. The first version supports the
  external acquisition-start trigger shown in the lab files, or free running
  with that trigger explicitly disabled. It does not guess other Python enums.
- `CeleraCamera(settings).connect()`: select camera, stop acquisition, configure,
  check Mono8/shutter readback, read dimensions, then start acquisition.
- `image_shape`: configured SDK readback as `(height, width)`.
- `capture_frame()`: one `get_raw_data()` call; return an independent NumPy
  `uint8` array with original pixel bytes in `(height, width)` row-major order.
- `disconnect()`: set `acquire = False`; safe before opening and to repeat.
  Context-manager entry/exit connect and disconnect.
- `CameraError`: setup/capture/stop failures, preserving the SDK exception cause.

Example (settings are supplied explicitly; these illustrate the lab reference,
not verified current machine defaults):

```python
from hardware.camera import CameraSettings, CameraTrigger, CeleraCamera

settings = CameraSettings(
    camera_index=0,
    roi_width=1664, roi_height=760, roi_x=832, roi_y=792,
    horizontal_binning=1, vertical_binning=1,
    video_mode=0, pixel_format="mono8", shutter=6000, frame_rate=60,
    trigger=CameraTrigger(
        enabled=True, external_input=1, detect_external_input_edge=True,
        delay=3000, debounce_time=0,  # supply the actual lab debounce value
    ),
)
with CeleraCamera(settings) as camera:
    frame = camera.capture_frame()  # may wait for an external trigger
```

The lab shutter bound **2000–15000 inclusive** is enforced in native SDK units
before camera access, without unit conversion. The caller's value is used;
`utils.py`'s test-helper override to 1000 is deliberately excluded. Only Mono8
is supported. No averaging, normalization, square root, transpose, auto gain
configuration, or hidden frame flushing is performed.

ROI width/height and offsets map to `image_size_x/y` and `image_start_x/y`.
After setting mode, ROI and binning, the adapter reads `image_size_x/y` back as
the delivered width/height rather than guessing binning division. Raw storage
must be contiguous, one-byte elements and exactly `height * width` bytes. It
rejects padding, truncation and other buffer formats instead of silently trimming
or converting. The returned copy survives SDK buffer reuse.

## Evidence and limits

Inspected the actual `SDK/open_Camera.py` inside the supplied `SDK.zip`, plus
camera setup, frame reading and cleanup in the separately supplied `utils.py`.
These establish `celera_camera()`, the property names, `color_coding.mono8`,
`trigger_source.external`, `pio_ports[index].debounce_time`, `get_raw_data()`,
and stopping through `acquire = False`. These reference files remain outside
the repository. No instructor implementation is copied into project source.

ALKERIA's [acquisition interface documentation](https://www.alkeria.com/downloads/mvtec/hAcqAlkUSB3-doc.html)
describes ROI, binning, external acquisition-start triggers and mode-dependent
formats. Its `grab_timeout` belongs to **HALCON**, not necessarily the lab Python
binding; it is not used as an invented `alkusb3` property.
The vendor [provides SDK access through its customer area](https://www.alkeria.com/resources/faq-eng/support/driver-software).
Neither the supplied archive nor public documentation found establishes a
timeout argument or additional release method for this particular Python wrapper.
The wrapper is not installed on this laptop. Property setters are used as in the
reference; their exceptions indicate failure, not invented integer result codes.

**Capture may block**, including indefinitely if the Python API has no internal
timeout and no frame/trigger arrives. This adapter calls `get_raw_data()` without
arguments and promises no deadline. It does not launch a background timeout
thread: a timeout in Python would not cancel an unknown SDK call safely.
The adapter serializes calls, so a concurrent `disconnect()` cannot interrupt a
blocked capture. SLM-to-camera trigger sequencing belongs to the later capture
script, not this adapter.

Setup failures after construction and frame/decoding failures attempt to stop
acquisition. Normal close only sets `acquire = False` and drops the Python
reference; no undocumented `release()`, `close()` or native destructor is called.
If stopping raises, the reference is retained for an explicit retry and capture
is disabled. A cleanup error is attached to the original failure rather than
replacing it.

## Verify on the lab PC

- Installed `alkusb3` version, supported Python/driver versions, camera indices
  and exact Python API documentation, including destructor/resource ownership.
  If a separate release is required, first establish its documented interface.
- Whether `get_raw_data()` blocks, has an internal timeout or supports an
  explicit timeout/cancellation API. Do not infer this from HALCON support.
- Pixel packing, row stride, top/bottom and left/right orientation. **The lab
  examples disagree on x/y order**: active `utils.py` uses PIL size `(y, x)`;
  its test helper uses `(x, y)`. Our contract is NumPy `(y, x)` with no transpose.
  Verify a non-square ROI and an asymmetric known target before measurements.
- ROI bounds/alignment, binning factors and whether `image_size_x/y` readback
  describes delivered, binned dimensions. If not, this version rejects a byte
  count mismatch; it cannot infer geometry from byte count alone.
- Shutter's native units, readback, range and exposure/frame-rate compatibility;
  frame-rate acceptance after configuration and before acquisition starts.
- Trigger input numbering, polarity, edge/level semantics, delay/debounce units,
  appropriate debounce value and acquisition-start versus per-frame behavior.
  This version does not issue a software trigger or configure undocumented
  trigger modules. Check inherited camera state, buffering/stale frames, gain,
  gamma and auto-exposure settings separately before quantitative experiments.

Laptop tests: `python -m pytest tests/test_camera.py -q` (fake camera only).
