# Meadowlark SLM adapter (bench verification pending)

`hardware/slm.py` is an independent, single-image PCIe adapter. It does not
integrate with the Runner or change simulation behavior. Importing it or creating
an instance does not load a DLL or open hardware. No instructor scripts are copied
into the project. Image generation, camera control and experiment timing remain
outside this adapter.

## Public interface

- `SlmSettings`: caller-supplied DLL/LUT paths, board, SDK construction settings,
  trigger/pulse options, timeout, image-size convention and return-code contract.
- `SlmReturnCodes(load_lut=..., write_image=..., image_complete=...)`: **required**
  exact success values verified for the installed SDK; no guessed defaults.
- `MeadowlarkSlm(settings).connect()`: initialize, verify board availability,
  query dimensions and load the LUT. Returns the connected adapter.
- `image_shape`: queried `(height, width)`.
- `display_image(image)`: require a two-dimensional C-contiguous NumPy `uint8`
  image of precisely that shape; write a private buffer and wait for completion.
- `disconnect()`: release SDK; safe to repeat. Context-manager entry/exit call
  `connect()`/`disconnect()`.
- `SlmError`: SDK failures, including vendor error text and failed return codes.

Usage after the lab has verified the settings and return codes:

```python
from hardware.slm import MeadowlarkSlm, SlmSettings, SlmReturnCodes

# machine_settings is supplied by the caller; there are no hardcoded lab paths.
# verified_success_codes must come from the installed SDK's documentation/tests.
settings = SlmSettings(
    **machine_settings,
    return_codes=SlmReturnCodes(**verified_success_codes),
)
with MeadowlarkSlm(settings) as slm:
    print(slm.image_shape)
    slm.display_image(lut_index_image)  # uint8, row-major, exact queried shape
```

No phase-to-gray mapping, rotation, reflection, resizing or wavefront correction
is implicit. External triggering still waits synchronously for completion and
requires an independently supplied trigger within the configured timeout. This
does not establish optical settling. Transfer failures close the adapter; reconnect
explicitly before another attempt. Only one adapter can own the process-global
wrapper at a time; other software must not independently initialize that wrapper.

## Interface evidence

Reviewed external inputs: `SDK.zip` (header and three lab scripts), its
`SDK/Blink_C_wrapper.h`, and the separately supplied `SDK/utils.py`. The archive
contains **no DLL, LUT, machine config or manual**. The header is authoritative
for this binding: cdecl, eight-argument `Create_SDK`, seven-argument `Write_image`,
`int` flags and `int* constructed_ok`; `Create_SDK`/`Delete_SDK` return `void`.
The lab scripts' `c_bool` flags are not reused.

Meadowlark's [1920 PCIe manual, Rev. 1.03, sections 6.1–6.2](https://preview-assets-us-01.kc-usercontent.com/210eb585-dc2c-0008-b663-73b9ea5cff38/aab827a2-3e9c-4a2f-990d-70e0b6f328aa/1920%20PCIe%20User%20Manual.pdf)
supports initialize → LUT → image → completion → destructor ordering, one-based
boards, dimension queries, and eight-bit image entries even for a 12-bit panel.
It distinguishes pixel-count image size on 1920×1152 from width on 512×512.
It describes an older six-argument image write: the supplied header, rather than
that older signature or the incompatible HDMI API, defines our argument list.

The header and this manual **do not establish integer success codes** for
`Load_LUT_file`, `Write_image`, or `ImageWriteComplete`. The lab's commented-out
code treats zero as failure for the last two, but does not prove the convention
or explain LUT results. Therefore callers must provide verified codes explicitly.
Every result is checked against its own code; fake-test codes are not lab values.

## Verify on the lab PC before first use

- Confirm the installed DLL and header are a matched PCIe SDK build and have
  compatible Python/DLL bitness and cdecl exports. Required dependencies must be
  installed; the adapter adds the DLL's parent to Windows' DLL search directories
  without changing the working directory or PATH. It does not need ImageGen.
- Establish the **three return-code conventions** above from the installed
  documentation/vendor support and controlled bench tests. Fake tests prove the
  checking logic, not those numerical conventions.
- Confirm panel model, board number and dimensions. The supplied `open_SLM.py`
  suggests bit depth 12, nematic=true, RAM=true, GPU=false, max transients=20,
  board=1, timeout=5000 ms, wait=false, flip pulse=true, refresh pulse=false.
  `utils.py` instead reads these from an absent config. These observations are
  not hardcoded defaults or evidence of current lab settings.
- Confirm the device/wavelength-specific LUT (the lab example names
  `slm5855_at532_PCIe.LUT`) and image-size convention. This adapter passes NULL
  for regional calibration and implements ordinary `Write_image`, not OverDrive
  transient generation. A 512 board may require model-specific power/DC-balancing
  setup beyond this adapter; do not treat selecting `width` as that setup.
- Verify row-major pixel orientation, LUT-index semantics and optical response
  with a known image. Inputs are eight-bit gray levels, not 12-bit packed data.
  LUT path character encoding is unspecified; the adapter requires an ASCII LUT
  path until verified rather than guessing UTF-8 or a Windows code page.
- Verify trigger polarity, timeout/recovery and pulse support with the actual
  firmware. The lab script limits refresh pulses to 1920×1152 firmware 1.8;
  this is not established by the supplied header. Both pulse flags cannot be true.
  Check optical settling separately from DMA completion. SDK release is not an
  optical safety interlock and cannot promise a particular final screen image.

Laptop tests: `python -m pytest tests/test_slm.py -q` (fake SDK only).
