"""Synthetic question, paper context, setup, and integration factories."""

import json

from workflow.state import LiteraturePassage

from bench.optical_bench import SimulatedBackend
from bench.matrices import pin_detector_matrix
from fixtures.offline import LocalScientist, LocalExperimentor, LocalAnalyst, LocalRouter

PAPER_PASSAGES = [
    LiteraturePassage(
        source_id="HUANG-S2",
        filename="Generalizing_the_Gerchberg–Saxton_algorithm_for_retrieving_complex.pdf",
        passage=(
            "Section 2, Figure 1 and equations 1–4. The input patterns vary the "
            "phase across SLM segments while keeping their amplitudes equal. "
            "Starting with a guess for the transmission matrix, the algorithm "
            "predicts the output light field, keeps its predicted phase, and "
            "replaces its amplitude with the square root of the measured "
            "intensity. It then refits the matrix and repeats."
        ),
    ),
    LiteraturePassage(
        source_id="RAFAYELYAN-APP-A",
        filename="Large_Scale_Optical_Reservoir_Computing_for_Spatiotemporal_Chaotic(1).pdf",
        passage=(
            "Appendix A, Experimental Setup. A reflective liquid-crystal SLM "
            "imprints the input pattern onto the phase of the reflected beam. "
            "The light passes through a scattering medium, and a camera "
            "records the resulting intensity pattern."
        ),
    ),
]

PAPER_QUESTION = (
    "Using the same phase-only SLM patterns and measurement budget, which method predicts "
    "camera brightness more accurately on unseen patterns, Generalized Gerchberg–Saxton or "
    "intensity least squares? Compare them with and without simulated camera shot noise."
)


BENCH_QUESTION = (
    "Test the agent workflow on a synthetic phase-only transmission-matrix retrieval experiment. "
    "Scientist: propose intensity-only approaches (for example reference-free GGS or intensity least squares). "
    "Experimentor: select a compatible reference procedure using only bench.acquire capability/settings. "
    "Calibrate whole-frame saturation to 3–5% with a fixed all-zero phase pattern, then freeze shutter and "
    "power and collect independent random phase fit and held-out frames. Use reviewed bench.retrieve "
    "to recover a detector TM with GGS and predict unseen detector readings. Analyst: evaluate usable "
    "counts, clipping and prediction metrics; Scientist: conclude from this evidence. Workflow completion "
    "does not establish measurement or prediction success. Hidden truth and fields are unavailable."
)


def setup_context(config):
    public = config.model_dump()
    for key in ("detector_matrix_path", "detector_matrix_sha256"):
        public.pop(key, None)
    return """SYNTHETIC OPTICAL BENCH — workflow testing, not a physical instrument
The supplied paper passages are context, not a claim to reproduce either apparatus.
SLM controls are phases in radians with exactly equal input amplitudes. The hidden
complex TM sums coherent contributions per pixel before squared magnitude. Camera
response = round(4000 * shutter_seconds * laser_power_mw * intensity + known offset
+ optional Gaussian read noise), clipped to uint8. No shot noise model yet.
The scene has nonuniform brightness; whole-frame calibration target is 3–5% pixels
at 255 with the fixed all-zero phase pattern. All adjustments stay inside declared
limits. Target failure stops acquisition and retains diagnostics. Calibration uses
at most calibration_attempts frames. Then settings stay fixed for every fit/holdout
pattern. One bench.acquire action runs this protocol, including calibration. Supply
shutter_seconds in s, laser_power_mw in mW and frame_budget in frame, exactly as the
verified capabilities allow. Default initial settings and counts are below. Budget
includes the maximum calibration attempts plus all fit and holdout frames.
The active SLM area is 8x8 pixels, tiled by sixteen 2x2 phase-only input regions.
The full raw camera frame is 20x20. Its centered 8x8 crop is [6:14,6:14], tiled by
sixteen 2x2 speckle grains, ordered left to right, top to bottom, matching SLM order.
Each grain has uniform coherent response in its four pixels. Extract
the mean uint8 count; any 255 pixel marks that detector reading clipped. A reference
procedure uses independently sampled random phases. The user's initial task chooses
the numerical operation, not this setup description. bench.retrieve is the GGS-only
path. For a comparison of GGS and nonlinear intensity least squares (intensity_ls),
use tm.compare_reference_free on this same synthetic-bench-v1 dataset. No second
acquisition or generic dataset conversion is needed. Both methods use identical
eligible training and held-out readings. GGS replaces predicted field amplitudes by
sqrt(measured intensity), then refits the coefficients (the supplied HUANG-S2 excerpt).
The implemented intensity_ls objective is mean((abs(X @ T.T)**2 - I)**2), optimized
over complex coefficients using a real-coordinate complex gradient and backtracking.
This description is an implementation contract, not an additional paper excerpt.
These operations fit only unclipped training readings per detector; they select restarts
by fit error alone and predict independently held-out raw detector readings with
the identical camera quantization and ROI extraction. Each detector needs at least
8*input_regions unclipped fitting readings and full complex input-pattern rank;
held-out quality criteria require at least 8 usable readings per detector. This is
fit eligibility, not proof of unique recovery. Neither operation can determine each row's
global optical phase. The uniform-ROI assumption is essential for a rank-one TM;
arbitrary mixtures of different pixel responses require a richer model later.
No internal fields or true matrix are provided. Use the supplied model contract as
synthetic prerequisites, never as verified laboratory facts. The Runner still needs
exact human procedure/review approval; agents cannot grant it. Use the existing
routing actions to obtain Analyst review and Scientist interpretation after retrieval.
Procedure prerequisites must select exact IDs from the supplied backend capability
map. Keep explanatory text separate. Calibration is executed by bench.acquire and
can fail; it is not a condition to certify before acquisition. Human approval is
owned by the workflow after preflight. bench.retrieve and tm.compare_reference_free
are reviewed numerical operations offered after an eligible dataset exists, not
acquisition actions. For this comparison use one acquisition and one matched numerical
comparison, then Analyst review and Scientist interpretation. Do not schedule follow-up
experiments or cross-run reuse for a single comparison request.
bench.acquire takes no external files: input_artifacts=[] and calibration_artifacts={}.
Matrix JSON files and their hashes belong to private simulator configuration, not
experiment input data; do not list, copy or move them as acquisition artifacts.
Model configuration (public synthetic settings only):
""" + json.dumps(public, indent=2) + "\nBackend-owned matrix mode: " + ("known" if config.detector_matrix_path else "seeded_random")


def build_integrations(config):
    """Canonical simulated backend factory; retained for saved integration paths."""
    if config.hardware_backend != "simulated" or config.optical_bench is None:
        raise ValueError("Synthetic factory requires explicit simulated configuration; no hardware fallback")
    backend = SimulatedBackend(config.optical_bench)
    services = build_model_integrations(config)
    services.hardware_backend = backend
    return services


def prepare_simulation(config):
    """Pin inputs before saving workflow state; export only public matrix identity."""
    from bench.optical_bench import OpticalBench
    from bench.acquisition import write_json
    config.optical_bench = pin_detector_matrix(config.optical_bench)
    metadata = OpticalBench(config.optical_bench).metadata
    config.run_output_path.mkdir(parents=True, exist_ok=True)
    write_json(config.run_output_path / "bench-metadata.json", metadata)


def build_model_integrations(config):
    """Use the saved agent mode for simulation or acquisition-disabled data reuse."""
    from workflow.graph import Integrations
    if config.bench_test_agents:
        return Integrations(scientist_model=LocalScientist(), experimentor_model=LocalExperimentor(),
                            analyst_model=LocalAnalyst(), router_model=LocalRouter())
    from workflow.llm import build_integrations as live_models
    return live_models(config)
