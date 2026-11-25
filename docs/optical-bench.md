# Synthetic optical bench for workflow testing

This bench tests the existing agent workflow with real local optical and numerical
calculations. It is an idealized synthetic apparatus, not a model validated against
laboratory data. It retains the supplied `HUANG-S2` and `RAFAYELYAN-APP-A` excerpts;
no PDF parsing or paper retrieval is performed.

## Run it

Install the project requirements, including Matplotlib for the individual plots:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --db data/bench.sqlite start --simulate --output outputs/bench
```

This uses the existing live model configuration (`TM_LLM_PROVIDER`, `TM_LLM_MODEL`,
`TM_API_KEY`) and pauses at the Runner's human review. Inspect the saved procedure
and fingerprint using the printed `show` command. Then use the printed approval
command, filling in the exact workflow ID, fingerprint, and reviewer:

```powershell
.\.venv\Scripts\python.exe main.py --db data/bench.sqlite show WORKFLOW_ID
.\.venv\Scripts\python.exe main.py --db data/bench.sqlite review WORKFLOW_ID approve --simulate --fingerprint REVIEW_FINGERPRINT --reviewer "Your name"
```

For verification without an LLM, add `--test-agents` **only to the initial command**.
These are explicitly labeled local Scientist, Experimentor, Analyst, and router
test doubles. They exercise the existing graph, Runner, real simulated camera,
and real retrieval operation. They do not validate live agent reasoning. Saved
checkpoints retain agent mode and model settings across subsequent commands.
There is no automatic approval or automatic switch to test doubles.

The Scientist proposes intensity-only approaches; the Experimentor selects the
reference procedure and prepares `bench.acquire`. Runner preflight and exact
human approval precede acquisition. The routing catalog then offers the reviewed
`bench.retrieve` operation on saved data, followed by Analyst review and Scientist
interpretation. Live agents may request clarification or block if their responses
are invalid; their decisions are not silently replaced.

The post-analysis Analyst now records a task-based `recovery` assessment: brightness
prediction, input-to-detector mapping, and complex matrix recovery are separate
claims, each with its evidence, unresolved reason and affected detector row indices.
The same path is used for bench retrieval and other registered TM acquisition
methods. It uses measured variation and usable/clipped counts, noise and quantization
when recorded, acquisition settings, and the backend's verified actions/parameters;
missing diagnostics remain unknown. It does not inspect hidden matrices.

The legacy `identifiable` metric and NPZ mask are retained for saved-run compatibility.
They mean **fit eligibility** (enough unclipped fit samples and full input-pattern
rank), not unique recovery; new results also expose `fit_eligible`. Neither low
held-out error nor varying detector readings alone establishes a unique matrix.
Equal-amplitude phase-only data can leave connection ambiguities even for a full-rank
input design. Low variation is a noise/quantization-aware diagnostic to investigate,
not a universal recovery test.

An Analyst follow-up names the unresolved question, proposed measurement and
distinguishing result, plus exact required action and parameter IDs. Python checks
these against current capabilities and existing run limits. Supported follow-ups
are offered to the Orchestrator as `plan_follow_up`, then the Experimentor writes a
new procedure for normal Runner preflight and fresh human approval. Unsupported
follow-ups conclude with recovery unresolved and the missing capability recorded;
prediction-only tasks may conclude on the narrower prediction result.

The current simulator cannot switch individual inputs off or vary their amplitudes.
Such measurements could resolve individual connection strengths, while general
relative phase recovery requires additional interference measurements. Repeating
equal-amplitude phase patterns or scaling global power/exposure does not remove
every mapping ambiguity. No new simulator controls are implemented here.

The Scientist receives `allowed_source_ids` from the supplied literature. Both
`cited_source_ids` and every proposal's `literature_references` use exact-ID enums
in the model-facing schema; citation explanations belong in prose. Invalid output
gets one repair attempt (also bounded by the workflow model-call limit), with the
offending field, value, and allowed IDs. Unknown citations are never guessed or
normalized. Exhaustion blocks with a Scientist output validation error, without
requesting user clarification or committing the invalid proposals.

The Experimentor receives `procedure_contract.supported_prerequisites`, using the
backend's existing capability/verification schema. `procedure.prerequisites` must
select exact IDs; plain-language reasons belong in `explanation`. The simulated
backend advertises `synthetic bench configuration valid` and
`synthetic phase mapping verified`, with evidence limited to the synthetic model.
Both are required by the acquisition action and rechecked in Runner preflight.
Agent statements cannot grant verification or human approval.

`bench.acquire` takes no external acquisition or calibration files: procedures
must use `input_artifacts=[]` and `calibration_artifacts={}`. The simulator owns
its matrix source, fingerprint verification and in-acquisition calibration. A
matrix JSON, source hash, or setup-context file is not measurement input data;
copying it into `data/raw` or widening input roots is not a remedy. The same
artifact contract is checked before a draft is committed, during Runner preflight,
and at backend execution. Invalid drafts receive field-specific feedback within
the existing procedure-repair budget, without user clarification for internal files.

Agent-facing settings omit private matrix source paths/hashes; legacy setup text
is redacted in the planning view. The saved backend configuration and original
setup files retain their values for restart integrity and human-only truth reports.

Unsupported IDs and malformed procedure drafts are returned to the agent with
field-level errors, allowed IDs, and the rejected response. Repairs consume the
existing procedure-revision budget; invalid drafts are never accepted procedures.
A successfully repaired procedure still needs normal preflight and approval of
its exact fingerprint. Exhausted repair or missing backend verification blocks
with a validation error instead of asking the user to certify internal checks.
Genuine missing inputs or scientific choices can still request clarification.

Calibration success is determined during `bench.acquire`, not declared as a
planning prerequisite. Human approval is a separate workflow gate. Retrieval is
advertised separately from acquisition actions and offered only after eligible
saved data exists. A disabled/missing reviewed `bench.retrieve` capability is an
analysis-stage integration error; it does not require another acquisition.

Example starting conditions (each starts a new workflow and pauses for review):

```powershell
.\.venv\Scripts\python.exe main.py --db data/bench.sqlite start --simulate --test-agents --bench-config configs/bench-normal.json --output outputs/bench
.\.venv\Scripts\python.exe main.py --db data/bench.sqlite start --simulate --test-agents --bench-config configs/bench-correcting.json --output outputs/bench
.\.venv\Scripts\python.exe main.py --db data/bench.sqlite start --simulate --test-agents --bench-config configs/bench-unreachable.json --output outputs/bench
```

`--simulate` always overrides environment hardware selection. Its factory accepts
only `hardware_backend="simulated"` and explicitly constructs `SimulatedBackend`;
the Runner rejects backend provenance other than `simulated`. Saved backend and
configuration mismatches fail. No physical driver discovery or fallback exists.
Without `--simulate`, completed saved-data reuse and inspection remain available.
Inspection re-exports saved reports but never runs models or acquisition.
Legacy simulator checkpoints remain non-executable.

### Random or supplied detector matrix

The only layout for new runs is a 20x20 raw camera frame with a centered 8x8
crop, tiled completely by sixteen non-overlapping 2x2 detector blocks. An empty
config selects this layout and sixteen phase-only SLM inputs. The active SLM is
also 8x8 pixels, tiled by sixteen 2x2 uniform-phase input regions. Both input and
detector indices run left to right, top to bottom. Each camera block represents
one speckle grain: its four pixels share one complex response and their recorded
brightness is averaged to obtain one of sixteen outputs, giving a 16x16 TM.

An omitted `detector_matrix_path` uses seeded random mixing. The layout change
intentionally changes the old four-output seeded results; it is not an optional
layout selector. Other frame sizes, input counts, and old layout versions are
rejected for new acquisition. The existing
`configs/bench-normal.json` is `{}` and remains the random-mode example.

With live agents configured in the root `.env`, run one of these **fresh** starts
from the repository root (each invocation creates a new workflow ID):

```powershell
# Seeded random matrix: 16 inputs, 16 detectors; an empty config is valid.
.\.venv\Scripts\python.exe main.py --db data/random.sqlite start --simulate --bench-config configs/bench-normal.json --output outputs/random

# Exact 16x16 identity: sixteen phase-only inputs and sixteen detectors.
.\.venv\Scripts\python.exe main.py --db data/identity.sqlite start --simulate --bench-config configs/bench-identity.json --output outputs/identity

# Exact 16x16 complex mixing: use this case to examine held-out retrieval.
.\.venv\Scripts\python.exe main.py --db data/mixing.sqlite start --simulate --bench-config configs/bench-mixing.json --output outputs/mixing
```

Add `--test-agents` to a start command for the existing local doubles; matrix and
camera behavior are identical in both agent modes. After inspecting the review,
approve the same workflow with the matching database and fingerprint, for example:

```powershell
.\.venv\Scripts\python.exe main.py --db data/mixing.sqlite show WORKFLOW_ID
.\.venv\Scripts\python.exe main.py --db data/mixing.sqlite review WORKFLOW_ID approve --simulate --fingerprint REVIEW_FINGERPRINT --reviewer "Your name"
.\.venv\Scripts\python.exe main.py --db data/mixing.sqlite resume WORKFLOW_ID --simulate
```

The known-mode configs reference JSON files in `configs/matrices/`. Paths are
resolved relative to the **bench config file's directory**, not the working
directory. A matrix file contains exactly `real` and `imag`, equally shaped
rectangular arrays of finite JSON numbers, each `[16, 16]`. Row `j`
supplies the complex coefficients for every pixel in detector region `j`.
For example, a known-matrix config can contain:

```json
{"detector_matrix_path": "matrices/complex-mixing-16x16.json"}
```

Each acquisition records `geometry_version`, `camera_frame_shape`,
`camera_crop`, `slm_shape`, `slm_input_regions`, `regions`, `detector_order`,
and `detector_ordering` in its manifest and calibration diagnostics. Matrix row
`j`, measurement column `j`, plot panel `j`, and `regions[j]` refer to the same
block. The SLM TM columns integrate each input region's contribution with unit
amplitude; expanding a phase onto four SLM pixels does not multiply that input's
amplitude by four. Detector clipping is reported separately from full-frame
saturation. The pre/post previews show the crop and all sixteen block labels;
held-out prediction plots use a matching 4x4 panel grid.

Reanalysis reads and validates the regions recorded in each saved acquisition,
including legacy four-output measurements, and sizes the recovered matrix from
those measurements. It does not replace archived geometry with the new default.
Old four-input source files/layouts are unsupported for new acquisition. A saved
old approval cannot authorize the new layout because the capability revision
includes the new layout version; start a fresh workflow to acquire.

The supplied coefficients are used exactly, without normalization or gain fitting.
Outside the centered crop, the seeded nonuniform synthetic scene is retained. Calibration
still measures the fraction of **all raw frame pixels** at 255 and targets 3–5%.
Large or unusual matrices can change calibration or clip detectors; unreachable
targets still stop with calibration diagnostics and no fit/holdout sequence.

At start, the matrix file's raw-byte SHA-256 is pinned in saved configuration as
`detector_matrix_sha256`. The absolute source path is saved too. Approval/resume
and camera acquisition check it; changed bytes (including formatting changes) or
a missing file stop execution with an error. Restore the original file or start
a fresh workflow. Never edit a matrix file to alter an already reviewed run.
The simulator uses a private loaded matrix throughout each process and never
substitutes a changed file. `bench-metadata.json` in the workflow directory records
`matrix_mode` (`seeded_random` or `known`) and `matrix_fingerprint` (SHA-256 of
detector rows as row-major little-endian complex128 bytes). Known mode also records
`matrix_source_sha256`. These identities are included in the measurement manifest.

True coefficients are absent from setup prompts, checkpoints, normal measurement
exports, and fitting inputs; only paths and fingerprints identify the source.
Retrieval uses the acquired raw intensities, never the supplied matrix. Analysis-only
saved-data reuse also works without the original matrix file.

For the identity matrix, `|exp(i phase)|² = 1`: changing only phases of equal-amplitude
inputs leaves every detector reading constant at fixed shutter/power (40 counts
at the default 0.01 s and 1 mW, with zero offset/noise). This is useful for camera
checks and demonstrating measurement ambiguity. Constant readings cannot uniquely
identify which input a detector observes, or recover arbitrary diagonal powers
from phase-only measurements. A passing prediction score is not proof of unique
matrix recovery. Use the complex mixing case for nonconstant held-out prediction
plots and metrics; independent output-row phase ambiguity still remains.

### Saved-data reuse and continuation

Use the same database and exact fingerprint to continue the existing workflow.
Do not resupply initial settings or agent-mode flags when continuing. Explicit
simulation selection is required for review, resume, clarification, reconciliation
and follow-up; inspection and cancellation do not acquire. Simulation resumes
cannot refresh configuration from the environment.

Saved-data command:

```powershell
.\.venv\Scripts\python.exe main.py --db data/bench.sqlite start --reuse-data-from WORKFLOW_ID --output outputs/reanalysis
```

These create a new workflow using completed acquisitions and fresh analysis, with
acquisition disabled. A bench source's saved local/live model mode is retained;
the model-only integration factory does not construct a simulated backend. The
main CLI also retains `follow-up WORKFLOW_ID "message"` (add `--simulate` when
continuing a simulated acquisition workflow) and all general/fixture commands.

## Recorded inspection outputs and run summary

Every bench CLI export writes `run-summary.md` in the workflow directory,
including at the approval pause and after an early stop. It links the individual
data files and plots; it does not assemble them into a single image or call a
model to write a new explanation. Scientist rationales, Experimentor selection
messages, Runner records, and post-analysis Analyst conclusions come from saved
workflow records. Missing reasons are labeled as not recorded. Workflow,
measurement, and prediction statuses remain separate in both the summary and
`quality.json`.

Calibration keeps `raw_before.npy` / `raw_after.npy` and now also writes matching
lossless grayscale `raw_before.png` / `raw_after.png` files with no overlays or
rescaling. The existing `preview_before.png` / `preview_after.png` images outline
the crop and label every detector. `calibration.csv` retains every attempt's
shutter, power, and whole-frame saturation. `calibration.json` records the
`accepted_attempt` (null on failure), while the calibration plot and summary mark
accepted settings. Detector clipping is reported separately from the whole-frame
3-5% target. Stopped calibration still produces its diagnostics and summary without
claiming fit or held-out measurements were captured.

Retrieval adds the following inspection files alongside the existing prediction
CSV/scatter plots, evaluation JSON, and estimated matrix:

- `heldout-example.json`: first saved held-out pattern, input phases, detector IDs,
  spatial grid positions, measured/predicted readings, signed differences and clipping
  flags. The example is fixed in advance, not selected by error or clipping.
- `heldout-example-grids.png`: measured brightness, predicted brightness and
  predicted-minus-measured difference as matching 4x4 detector grids. Measured
  and predicted share the 0-255 camera-count scale; difference uses a symmetric
  scale. Labels use the same IDs as camera overlays, CSVs and matrix rows.
- `heldout-slm-pattern.png`: that pattern's 8x8 phase-only input, labeled I0-I15.
- `heldout-camera-raw.npy`, `heldout-camera-raw.png`, and
  `heldout-camera-preview.png`: the same pattern's full raw frame and separate overlay.
- `heldout-detector-errors.csv` / `.png`: error and count summaries over **all**
  held-out patterns. Counts distinguish usable scores, clipped observations,
  and unclipped readings whose detector fit was unavailable. The CSV also records
  predicted clipping, per-detector RMSE/MAE/NMSE/max error and measurement variation.

`bench-evaluation.json` records the same aggregate statistics and the observed
brightness span across all unclipped fit and held-out readings. With at least two
readings and span at most one camera count, a detector is flagged as low variation.
The summary warns that low prediction error for such data (for example a phase-only
identity matrix) does not establish unique matrix recovery. This inspection flag
does not change solver selection or existing quality thresholds. Archived
measurements retain their recorded geometry rather than being relabeled as sixteen
outputs.

## One matched GGS versus intensity least-squares comparison

A supplied initial question is retained verbatim. The default synthetic question
uses GGS, but does not replace a user's comparison request. The Scientist records
`retrieval_methods` in the proposal: `['ggs']` for the existing single-method path,
or `['ggs', 'intensity_ls']` for the matched comparison. The routing catalog offers
the corresponding reviewed operation once saved bench data exist.

```powershell
$question = "Compare GGS and nonlinear intensity least squares (intensity_ls) on this simulated phase-only bench. Use one shared acquisition and identical training and held-out readings. Report both methods and the mean-brightness baseline, then conclude this single comparison run without follow-up experiments."
.\.venv\Scripts\python.exe main.py --db data/compare-live.sqlite start $question --simulate --bench-config configs/bench-normal.json --output outputs/compare-live
```

These are live agents from `.env`; add `--test-agents` only for local doubles.
Inspect the procedure and use the exact approval command printed by the CLI.
Acquisition is still `bench.acquire`. Numerical comparison is
`tm.compare_reference_free` on the original `synthetic-bench-v1` manifest; no
second acquisition or dataset conversion is required. `bench.retrieve` remains
the GGS-only operation. The supplied HUANG-S2 excerpt describes GGS amplitude
replacement/refitting. The implemented `intensity_ls` minimizes
`mean((abs(X @ T.T)**2 - I)**2)` over complex coefficients using the real-coordinate
complex gradient and backtracking. This is an implementation description, not a
claim of another paper or an embedding search.

Both methods receive identical eligible training rows, shared seeded initial
guesses, and training-only restart/iterate selection. Clipped training readings
are excluded per detector; held-out scores use the same eligible readings for
both methods and the training-mean brightness baseline. Saved shutter, power,
offset, quantization and detector extraction apply equally to both predictions.
Per-method fitting time excludes shared initialization and plotting; initialization
is disclosed separately and is not charged twice to a total.

The existing NPZ/CSV/plot formats are reused under method subdirectories, with
`ggs.npz`, `intensity_ls.npz` and a common `bench-evaluation.json`. The normal Analyst
review receives both metric sets; the run summary presents their RMSE/NMSE, times,
usable/clipped counts and baseline side by side. A numerical brightness tie is
declared at RMSE rtol=1e-9, atol=1e-12; this is not a significance test. One dataset
does not establish a general algorithm ranking. Matching constant brightness can
leave the TM unresolved. Verified truth comparisons for both estimates remain in
the human-facing report only, never fitting or agent context.

## Optical and camera assumptions

Defaults and bounds are synthetic design choices, not equipment specifications.
JSON configuration files override any `BenchConfig` fields; unknown fields fail.

| Setting | Default | Configured bounds / supported size |
| --- | --- | --- |
| Input regions | 16, equal amplitude, phase in radians | Fixed for new runs; each controls 2x2 active SLM pixels |
| Camera | 20x20 uint8 grayscale | Fixed; saturation denominator is all 400 pixels |
| Active SLM / camera crop | 8x8 / 8x8 pixels | Camera bounds `[6,14,6,14]`, half-open `[y0,y1,x0,x1]` |
| Detectors | Sixteen 2x2 speckle blocks | Entire crop tiled once, row-major IDs 0-15 |
| Shutter | 0.01 seconds | 0.0001–0.1 seconds |
| Laser power | 1 mW | 0.1–2 mW |
| Seed | 17 | Reproducible hidden TM, separate pattern/noise streams |
| Fit / held-out patterns | 512 / 48 | 128–2048 / 8–256; fit ≥ 8×input regions |
| Calibration frame budget | 32 | 4–64 |
| Background / read noise | 0 counts / 0 counts std | 0–100 / 0–20 counts |

The per-pixel field is `sum_j T[pixel,j] * exp(i*phase[j])`. Optical intensity is
its squared magnitude. The camera response is
`clip(round(4000 * shutter_seconds * laser_power_mw * intensity + background + noise), 0, 255)`.
The 4000 conversion factor defines synthetic count units. Read noise, if enabled,
is additive seeded Gaussian camera noise; no shot noise, drift, diffraction,
SLM crosstalk, nonlinear response, or calibration uncertainty is modeled yet.

The seeded complex matrix has a broad spatial brightness distribution. Its scene
normalization puts the fixed zero-phase calibration pattern near 4% saturation at
the nominal shutter/power product, independently of the allowed control bounds.
Thus reducing or increasing the product can move the same scene below or above
the target. Each detector ROI has a uniform coherent response so that its mean
intensity corresponds to one complex TM row. Detector rows are dimmer than the
bright scene tail. Arbitrary mixtures of distinct pixel responses within one ROI
would generally need a higher-rank intensity model; do not assume a single row
can describe such a mixture.

The true matrix and internal complex fields remain inside the optical model.
Agents receive capabilities, known phase patterns, raw uint8 observations, saved
metrics and artifact references. No true TM/field artifact is registered. Seeded
reproducibility is a workflow boundary, not an adversarial secrecy guarantee.

## Calibration and reference acquisition

`bench.acquire` is one bounded structured action. Its reviewed settings include
the initial shutter and power and the maximum total frame budget (calibration +
fit + holdout). Model dimensions, limits, counts, seed and noise settings are in
the checkpointed configuration and the capability revision covered by review.

Calibration always uses the all-zero phase pattern. Saturation is the fraction
of **all raw-frame pixels equal to 255**, after quantization/clipping, including
pixels outside detectors. It first measures the requested settings. If needed it
checks the minimum and maximum feasible shutter×power products, then searches
geometrically between them. It adjusts shutter at the initially requested power
where possible, and adjusts power when a shutter bound requires it. Every
attempt's separate controls and observed fraction go to the CSV. The shaded plot
uses the inclusive 3–5% target band.

The selected shutter and power stay fixed across every subsequent fit and
held-out frame. Independent known random phase-only patterns supply intensity
measurements; there is no amplitude masking or coherent field measurement in
the acquisition procedure. Calibration frames never enter the fit/holdout split.

If the band is outside allowed products, or cannot be reached within the bounded
search at available pixel resolution/noise, calibration stops. Diagnostics and
before/last-attempt raw frames survive; no fit/holdout sequence is taken. The
acquisition journal records partial diagnostic artifacts and the error. The
workflow pauses for information and the simulation command returns exit code 2.
There is no automatic acquisition retry or new approval.

## Retrieval and interpretation

`bench.retrieve` is a reviewed named operation, not model-generated Python. It
validates artifact identities/hashes, raw-frame extraction, known phase inputs,
IDs and split metadata. It reuses the shared `fit_methods` implementation. The
reference method is GGS amplitude replacement plus least-squares field projection;
the shared fitter also computes nonlinear intensity least squares. The GGS-only
operation exports GGS; `tm.compare_reference_free` exports and evaluates both.
Neither method is selected using holdout. Restarts and iterates are selected solely
by fit intensity MSE. The mean-brightness baseline is a separate predictor based
only on each detector's eligible fitting readings.

Each detector excludes fit readings containing any saturated pixel. At least
8×input regions unclipped patterns with full-rank input vectors are required for
this reference procedure's fit; this is a conservative software adequacy check,
not a proof of unique recovery. Known synthetic background is subtracted before
fitting. The estimated matrix predicts independent held-out frames through the
same camera kernel and the same detector extraction, with calibrated controls;
predictions omit a fresh random noise draw. The saved estimated TM is in synthetic
field units. Its rows have independent arbitrary global phases. No entrywise
comparison with hidden truth is performed by fitting or the numerical tool; verified
simulation truth comparisons are confined to the human-facing report. Phase-only intensity data may have
additional identifiability degeneracies; successful held-out prediction is the
primary result, not a claim of uniquely recovering every matrix element.

Measured/predicted clipping flags appear for every held-out detector reading.
Clipped measured holdout readings remain in the CSV/plot and are excluded from
usable-data RMSE and NMSE. Counts are reported per detector and overall. Default
prediction criteria are RMSE ≤2 counts and NMSE ≤0.02 for every detector with at
least eight usable holdout readings and an adequate fit. These are workflow-test
criteria, not experimentally established scientific thresholds.

`quality.json` and `decision-log.md` separately record workflow status,
measurement quality and prediction quality. A completed calculation/workflow can
have failed predictions, and an acquisition with saved diagnostics can be partial.
The process return code reports execution/acquisition failure; inspect the quality
fields to determine whether prediction criteria passed.

## Verbatim dialogue and separate reports

Scientist, Experimentalist and Analyst responses include an addressed `handoff`
in their existing model call. Its exact text, recipient and optional `reply_to`
message ID are saved separately from proposals, procedures and reviews. Incoming
messages appear verbatim in the receiver's dedicated `handoffs` prompt section.
They provide context, never execution authority or approval. Invalid reply IDs and
missing workflow handoffs fail validation; existing role repair budgets apply.

`conversation.md` contains the saved handoffs in chronological order, consequential
Orchestrator decisions, and explicitly program-generated Runner execution events.
It uses no rewriting call. Punctuation, whitespace and line breaks are preserved.
`agent-messages.md` is the full detailed trace; `workflow.json` retains structured
records. Older runs show only messages actually saved, labeled as legacy where
there was no recipient/reply metadata. No new dialogue is invented for them.

`run-summary.md` puts workflow status, held-out prediction and the recorded matrix
recovery assessment first. Unassessed recovery stays unassessed. It links saved
estimates, labeled magnitude heatmaps and real/imaginary coefficient CSVs under
`matrix-inspection/`. Multiple estimates without a recorded selection are candidates.
True magnitudes are shown only for simulations whose reconstructed matrix matches
the acquisition's recorded matrix fingerprint (and pinned source hash for known
matrices). Truth stays in this human-only export, outside fitting and agent inputs.
An absent, changed or mismatching truth source leaves the estimate-only plot.

The prediction comparison uses each detector's mean of eligible fitting readings.
Both predictors are scored on exactly the same eligible held-out observations;
clipped observations and unscorable fits are disclosed separately. Two zero errors
are explicitly a tie. This comparison does not establish matrix recovery.

Regenerate from a saved `workflow.json` into a **separate directory** without
opening its database, loading API credentials, making model calls or acquiring:

```powershell
.\.venv\Scripts\python.exe main.py report "outputs/identity/WORKFLOW_ID/workflow.json" --output "C:\tmp\identity-report"
Invoke-Item "C:\tmp\identity-report\run-summary.md"
Invoke-Item "C:\tmp\identity-report\conversation.md"
```

Use the actual saved workflow path for other runs. Reports link original numerical
and camera artifacts; those files and the checkpoint database are not modified.

## Individual artifacts

The launcher prints `outputs/bench/WORKFLOW_ID` (or the selected output parent).
Existing output directories and historical experiments are retained. Setup
context files use unique `setup-context/…` subdirectories under that parent.

- Run root: `run-summary.md`, `conversation.md`, `decision-log.md`, `quality.json`, full `agent-messages.md`,
  `workflow.json`, and router decisions.
- `acquisitions/ATTEMPT_ID/bench/0/`: `calibration.png`, `calibration.csv`,
  `calibration.json` with whole-frame and per-detector saturation;
  `raw_before.npy`, `raw_after.npy`, `preview_before.png`, `preview_after.png`.
  Both previews use the identical 0–255 scale, cyan ROI boxes, and red saturated
  pixel crosses. On calibration failure, “after” means the last attempt.
- Same acquisition directory after successful calibration: `measurements.npz`
  with all raw frames, phases, pattern IDs, detector readings, per-frame/per-detector
  saturation; and `bench-dataset.json` with fixed camera settings and extraction.
- `analysis/OPERATION_ID/`: `heldout-predictions.png`, `heldout-predictions.csv`
  with pattern IDs, detector, measured/predicted counts, errors and clipping flags;
  `estimated-tm.npz` with the recovered estimate only; `bench-evaluation.json`
  with usable counts, thresholds, quality and limitations; reviewed source copies.

Plots, raw data, conversation and detailed trace remain individual files linked by
the short Markdown run summary.

## Standalone coherent interference check

```powershell
.\.venv\Scripts\python.exe -m checks.interference --output outputs/interference-check
```

This makes no LLM calls. It uses the same `pattern_to_camera` kernel as Runner
acquisition, with `T=[1,1]`, shutter 0.01 s, power 1 mW, gain 4000, zero background
and noise. One input alone reads 40. The independent analytic expression is
`80*(1+cos(relative_phase))`; literal expected checkpoints at
`0, π/2, π, 3π/2, 2π` are `160, 80, 0, 80, 160`. All remain below clipping.
It asserts those values and saves standalone `interference-check.png`, `.csv`,
and `.json` files. The isolated fixture can mask an input; the bench acquisition
API itself exposes only equal-amplitude phase controls.

## Tests and extension points

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_optical_bench.py tests/test_saved_data_cli.py tests/test_runner.py tests/test_llm_router.py
```

Tests execute the complete existing workflow with local agent doubles, exact
approval, calibration from inside/below/above the band, unreachable bounds,
fixed settings, clipping counts, holdout isolation, and a completed workflow with
failed noisy predictions. Live models are an additional verification path and
are never used by pytest.

`bench.optical_bench.OpticalBench` owns hidden scene generation and camera calls.
`SimulatedBackend` implements the shared Runner contract; improve physical
behavior here while retaining capability/review snapshots and explicit selection.
`analysis.bench_analysis` owns the reviewed reference retrieval/evaluation;
`workflow.routing_catalog` discovers its saved datasets and offers `bench.retrieve`.
`bench.workflow` supplies synthetic setup context and live/local integration
construction. Shared agent roles, routing, Runner approval and real hardware
integration hooks remain intact.

Use `python -m checks.interference --output outputs/interference-check` for the
standalone physics check. The old `tools/` forwarding modules have been removed.
Historical saved bench and model-only factory strings are translated to
`bench.workflow` by the CLI and graph when loading integrations.
