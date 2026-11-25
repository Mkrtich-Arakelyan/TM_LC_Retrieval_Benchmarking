# One-frame SLM / Celera / laser milestone

This operator-run script uses the existing `MeadowlarkSlm` and `CeleraCamera`
adapters plus the independent serial adapter in `hardware/laser.py`. The capture
can also use the DG812 LCC adapter when explicitly configured.
No hardware connection opens at import time. All verification performed
on the laptop uses fakes; a successful fake test is not bench verification.

## Optional LCC drive

The bench example sets `generator.enabled` to `false` and `lcc` to `null`.
That configuration opens no VISA session. For a lab-approved LCC capture, set
`generator.enabled` to `true` and provide an `lcc` object with one field,
`amplitude_vpp`. Its value must be between 0.1 and 12 inclusive. The existing
`--operator-approved` flag authorizes the capture. The adapter uses PyVISA from
the lab environment. It prepares CH1 at 20 kHz sine and 0 V DC offset, confirms
the settings while CH1 is off, then enables output for the acquisition. The
capture result records the request, CH1 readback, and shutdown confirmation.

Verify the waveform and voltage at the connected cell and the settling time
before collecting LCC data. The bench still needs to verify the new SCPI writes
and CH1 shutdown readback on the installed firmware.

## Laser interface and what is actually established

`LaserSettings` supplies the port, operating power, confirmed startup order,
CRLF command/reply framing, optional echo-setting observation, and finite
read/write timeouts. `LabLaser` exposes `connect()`, `start_emission()`,
`read_status()`, `shutdown()` and `disconnect()`. `parse_verdi_reply()` is a pure
parser available without a serial port.

The supplied `SDK.zip`'s `laser_control.py` was inspected directly, not copied.
It establishes COM1 as the example port, 19200 baud / eight bits / no parity /
one stop bit / no flow control, queries `?S`, `?L`, `?P`, and the vocabulary
`S=0/1`, `L=0/1`, `P=<value>`. The adapter uses pySerial directly, not shell
pipelines, plink, process killing or an instructor helper.

The lab nameplate confirms a **Coherent VERDI-5W / Verdi V-5, Class 4, 532 nm**.
Operating power is restricted to **1.0–2.5 W inclusive**.
The off-state value **0.01 W is used only during shutdown**, outside that operating
range. Shutdown independently attempts, in the lab-confirmed order:

1. `S=0` (close shutter).
2. `L=0` (disable laser).
3. `P=0.01` (off-state setpoint).
4. Query `?S`, `?L`, `?P`, `?SP` independently; require closed shutter, standby,
   a valid measured-power diagnostic and the 0.01 W setpoint.
5. Close the serial port, reporting this separately.

A write succeeding does **not** prove shutdown. Wrong/absent/partial/error replies,
failed off commands and failed port close remain visible in the result. Raw reply
bytes are saved as hex alongside decoded replies and errors. Status results are
typed and independently marked valid/invalid; a valid state can still fail the
shutdown check. Faulted-off is recorded explicitly and is not normal standby.

The confirmed reference is Coherent's [Verdi V-2/V-5/V-6 Operator's Manual,
Tables 5-1 and 5-4](https://7id.xray.aps.anl.gov/internal/hardware/laser/Verdi_V2-V5-V6_RevIB.pdf).
The parser uses these meanings:

| Query | Meaning |
|---|---|
| `?L` | `0`: standby; `1`: on; `2`: off due to fault |
| `?S` | `0`: shutter closed; `1`: shutter open |
| `?P` | Calibrated measured output power in W; diagnostic, not setpoint |
| `?SP` | Configured light-regulation setpoint in W; must equal 0.01 for shutdown |

CRLF replies may include an optional `Verdi>` or observed `Viper V-5>` prompt and the exact query echo,
inline or on separate lines. The parser rejects other text, multiple values,
non-finite/negative power and error messages rather than searching for a number.
Normal commands return blank framing and/or prompt/echo; error responses fail.
Legacy null `shutdown_replies`/`command_ack` keys still load; non-null overrides
are rejected so configuration cannot redefine documented status meanings.

The reference's startup order is `L=1 → S=1 → P=2.25`. **It does not establish
that this is the lab-approved procedure** or establishes prerequisites/warm-up.
The model and reply meanings are confirmed; the actual controller echo/prompt
settings still need a raw bench reply. The example records `echo_commands: null`
for this unknown setting; the parser tolerates both echo variants and does not
change controller settings. The script
does not emit unless `startup_confirmed` is true and the caller supplies the
approved order. The tokens `enable`, `open_shutter`, `set_power` each occur once,
in the lab's approved order; no arbitrary command injection is exposed.
If the lab procedure needs additional checks, delays between steps, fault handling
or commands, extend it from that verified procedure **before** live testing;
the current three-command implementation does not invent them.

[pySerial's API](https://pyserial.readthedocs.io/en/latest/pyserial_api.html)
defines the finite read/write timeouts and byte-count results used here. Laser
reply parsing is now grounded in the confirmed Verdi manual. It describes unused
handshake pins for the standard controller; any lab cabling differences still
need verification.
No startup or live emission test was performed on the laptop.

## One-frame order and artifacts

After all settings are validated and a fresh output directory can be created:

1. Open laser serial connection, then connect SLM and load its LUT.
2. Build and save the `uint8` SLM image using the queried SLM dimensions,
   `gray_level`, and the optional stripe rectangle. Validate the rectangle
   before starting laser emission.
3. Prepare and enable CH1 if the plan uses the LCC, then perform the approved
   laser startup and the lab-specified settling wait.
4. Open/configure/arm the camera with the supplied verified external trigger.
5. Write exactly one SLM image, then request exactly one raw Mono8 camera frame.
6. Save the frame unchanged. For an LCC capture, turn off CH1, verify its state,
   and close VISA before laser, camera and SLM cleanup. Every attempted device
   has a separate cleanup result, even on errors.

SLM settings require input-trigger waiting off, image-flip output pulse on and
refresh output pulse off. This supports a **verified SLM-flip-to-camera-input**
connection. The lab camera reference
is external input **1**, edge detection **true**, delay **3000 native units**.
The example preserves those values, shutter **6000**, and the example ROI.
Debounce, signal polarity, wiring, trigger delay versus optical settling, and
whether acquisition-start triggering yields the intended first frame must be
confirmed on the lab PC. Do not treat reference values as proof of wiring.
`camera_trigger_confirmed` must be true only for the supplied verified settings.

### Optional SLM stripe rectangle

The example config sets `slm_active_rectangle` to `null` for a uniform-gray image.
Omitting the field has the same effect. To add alternating column stripes, set
the field to an object like this:

```json
{
  "slm_active_rectangle": {"x": 1, "y": 2, "width": 8, "height": 4}
}
```

`x` and `y` are the zero-based left column and top row in SLM pixels. Use
nonnegative integers for the origin and positive integers for the dimensions.
The rectangle must fit entirely within the SLM image. Camera ROI coordinates
use the camera's own pixel grid.

The pattern starts with `gray_level` everywhere and adds 128 to every other column
inside the rectangle, starting at its left edge. Pixel values wrap modulo 256,
so a base value of 200 becomes 72 in a striped column. Pixels outside the
rectangle keep the base value. The `known_image` section in `capture-result.json`
records the base value, `slm_active_rectangle`, and `stripe_mask_applied`.

### Saved files

The output directory contains individual files:

```text
one-frame-<unique-name>/
  slm-image.npy       # actual known uint8 image written
  camera-frame.npy    # one raw uint8 frame, shape (height, width)
  raw-frame.bin       # identical pixel bytes, C row-major; geometry in JSON
  capture-result.json
```

On an early error, only completed files are present. JSON and console output
record the last stage, original error, each cleanup result and laser transcript.
Capture success and cleanup success are separate. Exit code is zero only when
both pass, including confirmed laser shutdown and serial close.

**Limits of cleanup:** `finally` runs after an ordinary write/capture error, but
cannot run during a hung native call, process termination, or power loss. The
camera's Python timeout/cancellation API remains unverified; `get_raw_data()` can
block waiting for a trigger. PyCharm's Stop button may terminate the process
without cleanup. The operator must retain the lab's independent shutter/disable
controls and verify shutdown physically if the program hangs or says
`UNCONFIRMED`. SDK release is not an optical interlock.

## PyCharm: one operator-approved lab test

1. Use the lab's Python interpreter with the compatible Blink DLLs, Celera driver
   and `alkusb3`. Install pySerial into that same interpreter if missing:
   `python -m pip install "pyserial>=3.5,<4"`. Review the SLM and camera adapter
   bench checklists in [meadowlark-slm.md](meadowlark-slm.md) and
   [alkeria-camera.md](alkeria-camera.md).
2. Copy `configs/capture-once.example.json` to **`config.local.json`** at the
   repository root (already ignored by Git). Set the real DLL/LUT/COM paths and
   the **verified SLM success codes**; the example intentionally has null codes.
   Verify the camera values, binning/readback geometry and trigger wiring. Supply
   approved laser power/order and settling time. Capture a raw controller reply
   to confirm echo/prompt behavior; keep CRLF framing and do not supply custom
   shutdown reply strings. Debounce/zero wait are placeholders, not validated
   settings. The camera trigger source/wiring remains unconfirmed.
3. After the lab approves that procedure, set `laser.startup_confirmed` and
   `camera_trigger_confirmed` to true. If the order/status/wiring questions above
   are still unanswered, stop here; the template deliberately cannot start live.
4. In **Run → Edit Configurations → + → Python**, select **Module name** and enter
   `hardware.capture_once`. Set working directory to the repository root. Set
   parameters to the following, using a new directory name for this one test:

   ```text
   --config config.local.json --output C:\TM-lab-tests\one-frame-001 --operator-approved
   ```

5. With the operator present and the bench in its approved test state, press Run
   **once**. Check the per-device cleanup results in the console before inspecting
   the saved frame. If laser shutdown is unconfirmed, use the lab's manual shutdown
   procedure and confirm shutter/disable/off setpoint independently.

Equivalent terminal invocation from the repository root:

```powershell
python -m hardware.capture_once --config config.local.json --output C:\TM-lab-tests\one-frame-001 --operator-approved
```

After the acquisition finishes, run this from the repository root in Anaconda Prompt:

```bat
conda activate myenv
python check_shutdown.py --config config.local.json --run-dir "C:\TM-lab-tests\one-frame-001"
```

Use the exact capture output directory or the Runner's acquisition ID directory.
The checker queries the laser and any generator used in the selected run. Camera
and SLM release comes from per-device records in the capture result or from
capture results linked by the physical Runner record. An aggregate Runner cleanup
status gives no per-device confirmation. The shared bench lock covers captures,
physical Runner acquisitions, and checks launched from this checkout. Wait for
any experiment launched outside those entry points to finish before checking.

## Query-only laser status before a capture

From Anaconda Prompt at the repository root:

```bat
conda activate myenv
python check_laser_status.py --config config.local.json
```

No run directory is required, and `laser.startup_confirmed` may remain false.
Only the laser section is loaded. Under the shared bench lock, the checker uses
`LabLaser.read_status_once()` to open serial, send `?S`, `?L`, `?P`, `?SP`, and
close serial directly. It sends no startup or shutdown commands. It reports the
actual shutter/laser state, measured power and setpoint; invalid replies are
`UNVERIFIED`. Exit zero means all four queries and serial close succeeded, even
if the laser reports on or fault. Busy ownership, query or close failures return
nonzero. This is a controller status check, not shutdown confirmation.

## Active shutdown

Run from a separate Anaconda Prompt at the repository root:

```bat
conda activate myenv
python shutdown_bench.py
```

The command works after the capture process has exited. It uses the bench
configuration in `config.local.json` and keeps the stop request latched until
the operator runs `python shutdown_bench.py --rearm`. Re-arm requires bench
ownership to be clear. Use it after verifying shutdown and when ready for
another acquisition. Physical Runner backends must use the shared cancellation
event and honor the adapter ownership and output guards. An instrument command
already in progress can delay the owner's cleanup.

The command waits up to three seconds for cooperative cleanup, then attempts
the configured laser and DG812 shutdown steps for sessions it can acquire.
Camera and SLM release comes from the owning process's record. A Runner
record must link per-device evidence for each attempted action. Cleanup
artifacts with a recorded hash must match it. A missing, incomplete, or
aggregate-only record leaves release unverified. A
process stuck in a native SDK call may retain ownership. Repeat the command
after it releases the devices to collect the resulting evidence.

If the PC or instruments cannot be reached, or either output cannot be
confirmed off, use the lab's existing manual physical shutdown procedure and
its independent shutter and laser-disable controls. The full physical
procedure is maintained by the lab. See the cleanup limits above.

Laptop-only checks (no hardware or laser emission):

```powershell
python -m pytest tests/test_laser_capture.py tests/test_camera.py tests/test_slm.py -q
python -m pytest tests/test_shutdown_bench.py tests/test_check_shutdown.py -q
python -m hardware.capture_once --help
```
