"""Synthetic question, paper context, setup, and integration factories."""

import json

from workflow.state import LiteraturePassage

from bench.optical_bench import SimulatedBackend
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
Four 2x2 detector regions have uniform coherent response within each ROI. Extract
the mean uint8 count; any 255 pixel marks that detector reading clipped. A reference
procedure uses independently sampled random phases and reference-free GGS fitting.
bench.retrieve fits only unclipped training readings per detector; it selects restarts
by fit error alone and predicts independently held-out raw detector readings with
the identical camera quantization and ROI extraction. It cannot determine each row's
global optical phase. The uniform-ROI assumption is essential for a rank-one TM;
arbitrary mixtures of different pixel responses require a richer model later.
No internal fields or true matrix are provided. Use the supplied model contract as
synthetic prerequisites, never as verified laboratory facts. The Runner still needs
exact human procedure/review approval; agents cannot grant it. Use the existing
routing actions to obtain Analyst review and Scientist interpretation after retrieval.
Model configuration (synthetic values only):
""" + json.dumps(config.model_dump(), indent=2)


def build_integrations(config):
    """Canonical simulated backend factory; retained for saved integration paths."""
    if config.hardware_backend != "simulated" or config.optical_bench is None:
        raise ValueError("Synthetic factory requires explicit simulated configuration; no hardware fallback")
    backend = SimulatedBackend(config.optical_bench)
    services = build_model_integrations(config)
    services.hardware_backend = backend
    return services


def build_model_integrations(config):
    """Use the saved agent mode for simulation or acquisition-disabled data reuse."""
    from workflow.graph import Integrations
    if config.bench_test_agents:
        return Integrations(scientist_model=LocalScientist(), experimentor_model=LocalExperimentor(),
                            analyst_model=LocalAnalyst(), router_model=LocalRouter())
    from workflow.llm import build_integrations as live_models
    return live_models(config)
