# TM retrieval benchmarking — persistent agent workflow

Foundation for benchmarking transmission-matrix retrieval with an LC-SLM.
The project provides typed records, configuration, an editable provisional optical
setup, injected-model Scientist and Experimentor callables, a deterministic Runner,
explicit orchestration, SQLite checkpoints, human review, and disabled-by-default
numerical and hardware interfaces. Physical experiments
are not available: there are no laboratory drivers or connected equipment.

Use Python 3.10 or newer. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

`workflow/state.py` defines records; `workflow/config.py` loads optional configuration;
`configs/architecture.txt` describes the provisional setup outside future agent prompts.
`agents/scientist.py`, `agents/experimentor.py`, and `agents/runner.py` implement the agents;
`agents/common.py` shares the chat-model protocol and architecture reader.
`tools/python_execution.py` validates artifacts and records execution attempts.
`tools/tm_numerics.py` provides reviewed offline dataset validation and reference-free
TM fits through `analysis/`; `checks/numerical.py` reproduces their real saved-data workflow.
`main.py` is the primary CLI for general, fixture, saved-data, and explicitly
selected synthetic optical bench workflows. Supplied paper excerpts live
with the synthetic setup context in `bench/workflow.py`. See
[Optical bench guide](docs/optical-bench.md) for simulation, calibration, approval, artifacts,
and the standalone interference check. Simulation always requires explicit
`--simulate`; it never selects physical hardware automatically.

Primary commands (PowerShell, from the project root):

```powershell
.\.venv\Scripts\python.exe main.py --db data/bench.sqlite start --simulate --test-agents --bench-config configs/bench-normal.json --output outputs/bench
.\.venv\Scripts\python.exe main.py --db data/bench.sqlite show WORKFLOW_ID
.\.venv\Scripts\python.exe main.py --db data/bench.sqlite review WORKFLOW_ID approve --simulate --fingerprint REVIEW_FINGERPRINT --reviewer "Your name"
.\.venv\Scripts\python.exe main.py --db data/bench.sqlite start --reuse-data-from WORKFLOW_ID --output outputs/reanalysis
```

The first command pauses with zero acquisitions. Inspect its exact review before
approval. Continue simulations with `--simulate`; saved backend, settings and agent
mode are retained. Omit `--test-agents` on the initial command for configured live
agents. Saved-data reuse disables acquisition and retains a source bench's model
mode, including local doubles. General `start "Question"`, `--fixture`, trusted
`--factory`, and `follow-up` commands remain available.

## Project layout

```text
main.py                    Primary CLI
README.md, requirements.txt
.env.example, .gitignore    Maintained root configuration
.env                       Local credentials (ignored)
agents/                    Scientist, Experimentor, Runner, Analyst, orchestrator
workflow/
  graph.py, state.py       Workflow graph and shared checkpoint records
  config.py, llm.py        Application settings and live model construction
  routing_catalog.py      Available-action discovery
  persistence.py          Writer locking, saved references and progress persistence
  report.py               Workflow, decision-log and quality exports
tools/
  hardware.py             Runner acquisition execution boundary
  tm_numerics.py          Reviewed bench.retrieve and tm.* operation dispatch
  python_execution.py     Injected Python execution interface
bench/
  optical_bench.py         Optical model and simulated backend adapter
  acquisition.py          Calibration and frame acquisition sequence
  config.py, plots.py      Synthetic settings and inspection plots
  workflow.py             Question, paper context, setup, integration factories
analysis/
  bench_analysis.py        Bench retrieval and held-out evaluation
  datasets.py, metrics.py  Dataset validation/loading and numerical metrics
  tm_analysis.py           Saved-data TM operation implementation
  tm_solvers.py            Shared fitting and bounded archive loading
fixtures/
  offline.py, hardware.py  Agent/execution doubles and recording backend
  numerical.py, router.py  Shared numerical fixtures and routing doubles
checks/
  interference.py          Standalone camera interference check
  numerical.py             Saved-data numerical workflow check
  router.py                Routing and approval safety checks
configs/
  architecture.txt        Provisional general-workflow setup
  bench-*.json            Reusable synthetic starting conditions
docs/
  optical-bench.md         Simulator, calibration, approval and artifact guide
tests/, papers/           Tests and supplied literature
```

`tools/` contains the interfaces through which the workflow executes actions.
Supporting implementations belong in the packages above. All imports use those
packages directly; obsolete forwarding modules have been removed from `tools/`.
Live model construction, local test doubles, and the simulated optical backend
remain distinct.

New workflows use `bench.workflow:build_integrations` (or the acquisition-disabled
`bench.workflow:build_model_integrations` for saved-data reuse). Historical factory
strings are translated to these paths by
`workflow.persistence.canonical_factory_path` when the CLI or graph loads them;
this preserves saved-workflow loading without the old modules.

Preferred standalone commands are `python -m checks.interference`,
`python -m checks.numerical`, and `python -m checks.router`. Their existing
`--output` arguments are unchanged. Use these package commands in place of the
removed `tools.*` check entry points. All workflow commands use `main.py`.

Configuration reads the repository-root `.env` by default, independently of the
working directory; explicit dotenv paths and environment precedence are unchanged.
General startup and saved-data reuse share `configs/architecture.txt` as their
default; `--architecture PATH` remains an explicit override. On context reads,
only the exact former repository-root `architecture.txt` path is redirected.
Custom paths and generated simulation contexts are preserved.

New live workflows select `workflow.llm:build_integrations`. The factory-path
translator also accepts the historical `llm:build_integrations` string and the
historical bench factory strings, without restoring old import modules.

## Responsibilities and handoffs

- Scientist: uses the user request, setup, domain knowledge, and later literature
  to create `ExperimentProposal`; later interprets evidence in `ScientificFinding`.
- Analyst (optional): advises on analysis data requirements before a procedure and
  reviews recorded numerical results before Scientist interpretation. It can request
  calculations on saved data; it cannot execute code, approve, or schedule acquisition.
- Experimentor: converts a proposal to `ExperimentProcedure`, then writes analysis
  code and analyzes saved data into `NumericalAnalysisResult`. No direct hardware access.
- Runner: enforces preflight and exact human approval, owns the hardware backend
  boundary, and records progress, actual settings, raw-data references, cleanup,
  and failures. Provided backends include a test-only interface fixture and an
  explicitly selected synthetic optical bench; laboratory backends own physical connections.
- Orchestrator: optionally uses a routing LLM for scientific next-step choices from
  a Python-owned action catalog. Preflight, human approval, execution, budgets and
  recovery remain Python responsibilities. Legacy and fixture runs retain deterministic routing.
  `WorkflowState` connects all records to the original request and workflow status;
  `ConversationEntry` stores readable messages with record links.

The record chain is proposal UUID -> procedure proposal_id -> acquisition
procedure_id/version/fingerprint -> analysis acquisition_ids and input_data ->
finding evidence_ids. Conversation links can refer to any record UUID.

## Schema choices

UUIDs give records stable identities. Procedure versions start at 1; action list
order is execution order. Requested settings belong to actions; actual settings
belong to acquisitions. Settings carry optional units: supply units for physical
quantities and use explicit names for camera/SLM channels. No units or equipment
limits are inferred. Flexible JSON method settings avoid prescribing an algorithm.

Raw arrays, frames, plots, and scripts stay outside state. `Artifact` stores their
paths, descriptions, optional media types, and optional SHA-256 hashes. Paths are
references, not assertions that files exist. Relative paths use the project root.
The underlying proposal schema accepts citation strings, DOIs, or URLs. Scientist
output restricts references to the supplied passage source IDs; an empty list
allows a proposal before literature retrieval exists.

Acquisition and analysis default to pending. Failure requires an error; completion
requires recorded evidence and no execution errors. Failed work can retain partial
data or metrics. Analysis permits metrics, output artifacts, or both. Findings
require evidence references and limitations; the schema does not verify scientific
truth. Reference existence, artifact integrity, workflow transitions, and hardware
safety are not generally enforced. The Scientist validates references used in its
own findings against completed acquisition/analysis records, including upstream
acquisition status. Typed IDs allow later validation without
embedding full records inside each other. Build and validate a replacement record
when changing outcomes; nested container edits are not revalidated automatically.

Human approval must apply to the exact procedure being executed. `HumanApproval`
stores identity, version, and a SHA-256 fingerprint of all procedure fields.
`approval_matches` requires approved status and all three to match. Changing
actions or settings invalidates approval even without a version bump. Revisions
should increment the version and require renewed approval. The helper alone is
not hardware enforcement, authentication, revocation storage, or a safety check.
The Runner now enforces matching approval and current backend constraints against
its copied execution snapshot. Runner approval also requires `review_fingerprint`,
covering reviewed capabilities, prerequisite evidence, calibration identities,
output configuration, and budgets.

## Configuration

Call `AppConfig.from_env()` explicitly (the CLI does this). It reads `.env` at the
repository root, then applies process environment overrides. Importing does not load
credentials, create directories, or contact a provider. No provider/model is
selected until you configure one.

| Environment variable | Default |
| --- | --- |
| `TM_LLM_PROVIDER` | unset |
| `TM_LLM_MODEL` | unset |
| `TM_API_KEY` | unset; secret excluded from serialization |
| `TM_PAPERS_PATH` | `papers` |
| `TM_INDEX_PATH` | `data/index` |
| `TM_RUN_OUTPUT_PATH` | `outputs/runs` |
| `TM_MAX_WORKFLOW_ITERATIONS` | `10` |
| `TM_WORKFLOW_DATABASE_PATH` | `data/workflows.sqlite` |
| `TM_MAX_PROCEDURE_REVISIONS` | `3`; initial procedures for new proposals are separate |
| `TM_MAX_WORKFLOW_ANALYSIS_ATTEMPTS` | `3`; persisted across restarts/follow-ups |
| `TM_MAX_WORKFLOW_ACQUISITION_ATTEMPTS` | `3`; persisted across restarts/follow-ups |
| `TM_NUMERICAL_TIMEOUT_SECONDS` | `60` |
| `TM_NUMERICAL_INPUT_PATH` | `data/raw` |
| `TM_NUMERICAL_BACKEND` | `disabled`; `test` accepts a trusted fake backend only |
| `TM_MAX_EXECUTION_ATTEMPTS` | `2` per Experimentor invocation |
| `TM_HARDWARE_BACKEND` | `disabled`; `test` or `external` also requires an injected backend |
| `TM_HARDWARE_INPUT_PATH` | `data/raw` |
| `TM_MAX_ACQUISITION_ACTIONS` | unset; optional positive action budget |
| `TM_MAX_ACQUISITION_MEASUREMENTS` | unset; optional positive measurement budget |
| `TM_HARDWARE_ACTION_TIMEOUT_SECONDS` | `60`; cooperative action/cleanup deadline |
| `TM_HARDWARE_RUN_TIMEOUT_SECONDS` | `600`; cooperative total deadline |

Use `.env` or environment variables for credentials; never commit secrets. `.env`
and `.env.*` are ignored; the blank `.env.example` is safe to commit. The Experimentor enforces model-call and
execution-attempt limits; timeout is passed to the test backend and over-limit
reports are rejected. Global optical equipment limits remain explicitly unknown.
Runner limits come only from a trusted backend's verified capability snapshot,
never the provisional architecture file. Fixture constraints apply only to tests.

## Calling the Scientist

### Connecting a live LLM

Install/update dependencies once:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Fill the local `.env` file (or copy `.env.example` if it is missing):

```dotenv
TM_LLM_PROVIDER=openai
TM_LLM_MODEL=your-api-model-id
TM_API_KEY=your-api-key
TM_REVIEWED_NUMERICS_ENABLED=true
```

Set provider to `gemini` for the Gemini Developer API; use its model ID and API key.
Use API credentials, not a ChatGPT/Gemini chat-app login. No particular model family
is required: choose a text chat model supporting JSON object output with enough
context for the setup, record schemas, and workflow history. Scientific reasoning
and reliable instruction following matter; the model does not perform the matrix
fits itself. Review its scientific conclusions as usual. Model access and pricing
depend on your provider account.

Start the normal workflow without `--fixture` or a custom factory:

```powershell
.\.venv\Scripts\python.exe main.py start "Prepare a plan to compare reference-free TM retrieval methods"
```

The built-in adapter configures Scientist, Experimentor, and Analyst with the same
model. It requests JSON and validates each response against the existing Pydantic
schema. Full schemas are supplied as instructions because the records include
flexible setting dictionaries outside strict provider schema subsets. JSON syntax
mode is not a guarantee of schema correctness: malformed, refused, truncated, or
schema-invalid answers fail validation; there is no silent response repair or
provider/model fallback. Clients use a 120-second request timeout and at most two
SDK retries. No temperature is forced. No provider tools are enabled.

The Analyst receives an explicit response contract containing the exact target
UUIDs. A malformed response, altered target ID or missing required evidence ID gets
one model correction attempt with validation feedback. References are never silently
rewritten. A second invalid answer blocks the workflow; successful corrections and
their initial validation error are recorded in the Analyst message. Acquisition and
evidence eligibility checks still apply before a review can be saved.

### Routing LLM

`main.py start --reuse-data-from WORKFLOW_ID` enables `llm_router_enabled`
for paper-context analysis of completed saved acquisitions. Acquisition and generated
code remain disabled. The built-in provider uses the same API key and model for
`Integrations.router_model`; programmatic callers can inject their own router.

`agents/orchestrator.py` supplies one router prompt and validates `RouterChoice`.
`workflow/routing_catalog.py` builds actions from available agents, verified saved-data
manifests, current capabilities and remaining budgets. It has adapters for the three
reviewed TM operations; it does not match phrases in research questions. Additional
reviewed tools can add catalog adapters and an execution handler without adding a
branch for a particular question. Python chooses what can be offered; the routing
LLM chooses among those offered actions using scientific context.

The compact input contains the research question, latest agent outcome, recent human
clarifications, active proposal summaries, validated dataset dimensions/provenance/
splits and fingerprints, eligible completed results, capabilities, limitations and
concrete action IDs. It excludes arrays, secrets and the full conversation. The model
returns an `action_id`, optional relevant `record_ids`, and a short reason. Leaving
`record_ids` empty is allowed: Python supplies the exact immutable bindings from the
chosen action, avoiding model transcription of long UUIDs. Any supplied IDs must
belong to that action. Stale or changed bindings are rejected on a second catalog check.

Permitted scientific choices include proposing/revising a scientific plan, Analyst
advice/review, preparing/revising a procedure, requesting preflight, running an offered
named operation, interpretation, clarification, or accepting a recorded Scientist
conclusion. No approval, direct acquisition, generated source, arbitrary function,
module import or shell command is a routing action. Even a valid preflight selection
only reaches Runner checks and human review. Only the existing exact-review human
approval can reach acquisition, which Runner revalidates as before. The paper-context launcher keeps acquisition disabled.

Each router turn is a LangGraph node with existing step limits. Chosen actions and
record bindings are checkpointed before the executor node; numerical/acquisition
attempt limits, durable execution receipts, no-blind-replay policy and approval
fingerprints remain in force. Invalid/malformed choices are recorded as rejected,
the workflow blocks with a clear error, and saved data/results remain intact. There
is no automatic deterministic fallback after a bad router response. Correct the issue
and use the existing launcher `--resume WORKFLOW_ID`; clarification, approval and
reconciliation use their existing explicit commands.

Decisions are stored in `WorkflowState.router_decisions` in SQLite and exported under
the printed `outputs/runs/WORKFLOW_ID/` directory:

- `router-decisions.md`: ordered accepted/rejected decisions and reasons.
- `router-decisions.json`: decisions, bound IDs and the compact context/action catalog
  shown to the model, so its available options can be inspected.
- `agent-messages.md` and `workflow.json`: decisions alongside agent messages and records.

Programmatic callers opt in with `AppConfig(llm_router_enabled=True)` and
`Integrations(router_model=...)`; the built-in provider factory constructs this slot
when enabled. `TM_LLM_ROUTER_ENABLED=true` also enables it for new CLI configurations.
`--fixture` forces deterministic routing and never invokes a routing LLM, even with
live credentials/configuration present. Older runs without this flag retain their
original routing rather than silently changing a saved experiment.

Reproduce the routing checks:

```powershell
.\.venv\Scripts\python.exe -m checks.router
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

The harness uses the same test-double router with existing algebraic data, missing
data with a recording test backend, and missing data without a backend. It checks
invalid action rejection and zero acquisition calls without matching approval.
`--live` explicitly invokes only the configured router for these scenarios; the
acquisition capability is still a test fixture. Other agent calls and safety checks
use existing canned fixtures. It writes results beneath `outputs/router-check/`.
The tests cover wrong IDs, stale capabilities, checkpoint reopening, budgets and
fixture mode without live API calls.

Explicitly injected integrations and `--factory` still take priority. Clients are
created lazily; offline fixture mode uses canned models even if `.env` has live
settings. Loading configuration does not modify the process environment or expand
`${...}` inside credentials. Environment variables override `.env`, including empty
values. Secrets are excluded from saved configuration; resume reads the key from
the current process configuration. A resumed workflow retains its saved model name
unless `resume --refresh-config` is requested. `show` needs no API call.

The `.env` template enables only reviewed offline numerical operations. Hardware
and arbitrary generated-code execution remain disabled. The fixture command below
remains the deterministic offline numerical demonstration; a new workflow needs
saved registered data before it can analyze measurements.

Supply a chat model supporting `with_structured_output(PydanticModel).invoke(messages)`.
This follows the [LangChain structured-output interface](https://reference.langchain.com/python/langchain-core/language_models/chat_models/BaseChatModel/with_structured_output).
No LangChain imports, provider SDKs, or additional dependencies are needed by this
module: the caller chooses and configures a compatible model separately. No
client is instantiated and no credentials are read by the Scientist. An invocation
uses the supplied model, so its networking and credentials are the caller's concern.

```python
from agents.scientist import scientist
from workflow.state import LiteraturePassage, WorkflowState

def propose(chat_model):  # externally configured model; a fake also works
    state = WorkflowState(user_request="Compare TM retrieval methods at equal measurement budgets")
    passages = [LiteraturePassage(
        source_id="provided-paper:page-3",
        filename="provided-paper.pdf",
        page_number=3,
        passage="Replace this with an actual supplied passage before model use.",
    )]
    result = scientist(state, chat_model, literature_context=passages)
    # Omit literature_context when no passages are available.
    print(result.outcome.kind, result.outcome.explanation)
    print(result.outcome.recommended_next_step)
    return result.state  # full replacement state; do not append its history again
```

The model selects proposal, interpretation, or request_information; there is no
fixed experiment count. It returns existing proposal/finding models inside a small
`ScientistOutcome` envelope, with questions, an explanation, next step, and cited
source IDs. The callable validates structure, citation IDs, evidence eligibility,
and record ID collisions before adding records. Missing/empty/unreadable setup
files and malformed responses raise errors; there are no fallback findings.
Model/provider errors propagate. No retry loop is implemented.

The default setup path is resolved relative to the project module, independent of
the working directory. Override it with `architecture_path=...`. The file is read
on every invocation. The exact decoded text (including line endings) and supplied
passage metadata/text are saved in the new conversation entry's
`scientific_context`, alongside links to created records. The input state is
unchanged; each successful call returns a new state with exactly one additional
Scientist conversation entry. Existing history is preserved once. Workflow status,
procedures, acquisitions, analyses, and approvals are preserved; routing remains
unimplemented. Without literature a notice that recommendations have not been
checked against project papers is added to the explanation and saved message.

All literature citations must be supplied source IDs in `cited_source_ids` or
proposal `literature_references`. Unknown IDs raise a clear error. Prose citation
discipline and scientific accuracy are prompt requirements, not semantic proof:
the validator cannot detect every unsupported natural-language claim. Completed
record checks likewise do not verify data files, calibration, or scientific truth.

`tests/test_scientist.py` uses canned fake-model responses to test inputs, schema
validation, provenance, and state handling. It does not evaluate actual scientific
reasoning quality or prove compatibility with every provider's schema restrictions.
The existing state tests remain offline regressions. No model API call has been
tested or is needed to run the suite.

## Calling the Experimentor

The Experimentor prepares acquisition procedures, generates saved numerical
analysis scripts, reviews execution diagnostics, and repairs technical code issues
within configured limits. It requests essential information or Scientist decisions
instead of changing scientific controls, budgets, assumptions, or hypotheses.
It has no hardware connection, approval, acquisition, or scientific-finding tools.
The injected model uses the same interface as the Scientist. Provider clients are
configured separately by `workflow/llm.py` or a custom integration factory.

To prepare a procedure, supply a state containing the Scientist's proposal and a
request describing the desired technical work:

```python
from agents.experimentor import experimentor
from workflow.config import AppConfig
from workflow.state import WorkflowState

def prepare_procedure(chat_model, proposal):
    state = WorkflowState(
        user_request="Prepare an acquisition procedure implementing this proposal",
        proposals=[proposal],
    )
    result = experimentor(state, chat_model)
    print(result.outcome.explanation)
    return result.state  # replace the state; history is already included once
```

The model returns the existing `ExperimentProcedure` schema, including ordered
actions and typed requested settings. Versions must be one greater than the
highest existing version for that proposal, and each revision has a new record ID.
Old versions and approvals remain in history; approval never transfers to a
revision. Mandatory unresolved preflight prerequisites cover calibration, actual
equipment capabilities/limits, measurement requirements, and exact-procedure
human approval. A prepared procedure is a preflight/review candidate, not evidence
that equipment or settings are safe. Natural-language scientific consistency is
a prompt requirement and still needs review.

For analysis, supply saved acquisition records and put only designated input data
under the configured input root. The model may ask for missing data or formats:

```python
def generate_analysis(chat_model, saved_state):
    state = saved_state.model_copy(deep=True)
    state.user_request = "Write numerical analysis of these saved acquisitions; explain preprocessing and diagnostics"
    config = AppConfig.from_env()  # execution defaults to disabled
    result = experimentor(state, chat_model, config=config, execute=False)
    # result.state.analysis_code contains saved scripts with exact SHA-256 hashes.
    return result.state

def request_analysis_execution(chat_model, saved_state):
    return experimentor(saved_state, chat_model, config=AppConfig.from_env(), execute=True)
    # With the default backend this records execution unavailable, not completion.
```

Both agents accept optional `literature_context` and `architecture_path` as shown
above. Setup text is re-read for every invocation. Procedure context is preserved
in its linked conversation entry; each `AnalysisCode` also carries its exact
architecture/passage context. Literature references must use supplied source IDs.
There is no retrieval. Relevant records and conversation are sent to the model.

Code is saved as UTF-8 under `run_output_path/analysis/<unique-id>/analysis.py`.
`AnalysisCode` records inputs and their hashes, the script artifact/hash, method
settings, seed, and technical explanation. It is a proposal to execute code, not an
execution result. Each actual request produces a separate `NumericalAnalysisResult`
linked through `code_record_id`, with nested `ExecutionDetails` recording attempt
number, backend, status, exit code, stdout, stderr, elapsed time, and failure reason.
Disabled execution has analysis status `pending` and execution status `unavailable`;
generation alone adds no execution result. Failed attempts retain diagnostics,
never publish metrics as successful results, and remain alongside later repairs.
Exact executed code is identified by its immutable saved hash/version reference.

On an available fake backend, the Experimentor asks the model to inspect each
recorded execution and request technical repairs if needed. Each repair is saved
as a new script record. Inputs cannot silently change during repair. The attempt
budget applies per invocation; `max_workflow_iterations` caps all model calls,
including reviews. There is no automatic continuation after either limit. A call
adds exactly one conversation entry, preserves caller state and acquisition,
approval, and finding records, and leaves workflow routing/status unchanged.

## Generated-code numerical execution boundary

No general-purpose generated-code execution backend is implemented or enabled.
Reviewed named NumPy operations are now available separately (see below); they
never dispatch generated Python. Neither Docker nor
Podman was found on the development machine's command path. No container integration
test was run, and no generated script has been executed to validate this feature.

The supported production mode is `TM_NUMERICAL_BACKEND=disabled`. `execute=True`
cannot override it. `test` mode additionally requires an explicitly injected trusted
`ExecutionBackend`; it exists for fake tests, not for running untrusted code.
Without an injected backend execution remains unavailable. No host subprocess,
`exec`, `eval`, dependency installer, or fallback execution path is supplied.

The interface passes a script path, explicit input paths, method settings, seed,
unique output directory, and timeout to a backend. A fake returns `BackendReport`
with exit code, logs, elapsed time, metrics, and output references. Input files
must exist inside `numerical_input_path`; output files must exist inside that
attempt's output directory. Resolved paths reject traversal and symlink escapes.
Input and output roots must be disjoint. Recorded input hashes and script hashes
are checked, and the tool rejects malformed/non-finite metrics, failed exit codes,
missing output files, and out-of-directory output references. Output references
receive verified hashes. Failed acquisitions may provide partial saved data for
technical diagnostics; this does not make the acquisition successful.

Path validation and timeout checks are **not a security sandbox**. A trusted fake
backend is ordinary test code and receives host paths. It must never execute the
generated script. Timeout is a backend contract; the interface cannot interrupt a
misbehaving injected Python callable. Post-run input hash checks detect accidental
changes but cannot prevent them. Code and output artifacts are created on disk;
state-copy semantics do not roll back these files on later errors.

To enable real execution later, install and verify an isolated runtime and add a
reviewed container adapter plus an explicit configuration option. It must provide
no network, devices, credentials, project/home/Docker-socket mounts; mount only
designated input files read-only and a separate output directory writable; run
unprivileged with resource/time limits and a pinned, reproducible dependency image.
Dependencies must be prepared outside generated scripts, with no arbitrary installs.
There is deliberately no environment switch that enables host execution today.

Offline tests use fake chat models and fake backends that write canned artifacts
without interpreting generated code. They verify record plumbing, review/repair
limits, logs, path/metric validation, state immutability, and approval/version
boundaries. They do not demonstrate scientific correctness, code correctness,
container isolation, or successful real numerical execution. A symlink policy test
is skipped when the operating system does not permit creating symlinks.

## Runner preflight and approval

The Runner needs no LLM. It performs deterministic validation, explains technical
issues for the Experimentor, and never writes analyses or scientific findings.
Changes to scientific controls, budgets, or assumptions require Scientist review.
Preflight does not create run directories, connect, discover, or change equipment.

```python
from agents.runner import runner

def review_procedure(state, procedure_id):
    result = runner(state, procedure_id)  # disabled hardware by default
    assert result.review.status == "blocked"
    print(result.explanation)            # ordered actions, settings, outputs, limits, issues
    return result.state                 # one new conversation entry; replace old state

def blocked_acquisition(state, procedure_id, config, backend, reviewed):
    return runner(state, procedure_id, config=config, backend=backend,
                  execute=True, reviewed=reviewed, approval=None)
    # Missing approval causes zero backend acquisition calls, even if preflight passes.
```

`AcquisitionAction.action` must exactly match a backend capability identifier.
Existing records with only `instruction` prose remain valid records but are not
executable. The Runner never interprets arbitrary text or Python as hardware
commands. The Experimentor must use verified action/parameter information to
revise unsupported procedures; it still requires increasing versions and new IDs.

Preflight checks parameter types, exact units (no automatic unit conversion),
verified limits, required calibrations/prerequisites, backend availability,
cooperative deadline/cancellation support, output paths, and configured budgets.
All required prerequisite strings need explicit verified evidence from the trusted
backend. The Experimentor's generic unresolved prerequisites are not silently
accepted: they require technical revision and harness verification. Verifying
that approval enforcement exists never substitutes for actual human approval.

Procedure `input_artifacts`, `calibration_artifacts`, and `output_subdirectory` are
included in the canonical procedure fingerprint. Declared input/calibration files
must exist inside `hardware_input_path` with matching SHA-256 hashes. Required
calibrations must match the backend's verified artifact identities. The architecture
description remains scientific context, not an equipment-limit authority.

`AcquisitionReview` preserves the full procedure, capability snapshot, relevant
configuration, issues, and fingerprints. Reviews are saved in workflow state and
linked from conversation. The review fingerprint includes execution-relevant
prerequisite evidence, constraints, output configuration, and budgets, but excludes
secrets and changing record IDs/timestamps.

Execution requires `execute=True`, the prior `reviewed` snapshot, and explicit
`approval` from the trusted application caller. Approval is never inferred from
conversation, a model response, or `state.approvals`. The application must
authenticate the reviewer and supply current approval/revocation status. Legacy
approvals lacking `review_fingerprint` cannot authorize Runner acquisition.
There is no bypass flag or interactive pause/resume UI in this step.

The tool repeats preflight and approval checks immediately before starting and
before every action. Any execution-relevant change, even a still-valid limit,
requires renewed review and approval. The backend must also enforce current
constraints atomically in `execute`: software cannot eliminate the gap between
a check and a physical action. Fingerprints are content comparisons, not signatures
or protection against a malicious trusted caller or backend.

## Approved fake acquisition

This is an interface fixture, not an optical simulator or scientific benchmark.
The approval creation below belongs only in test code; a real application must
obtain a human decision. No physical observations or calibration are generated.

```python
from hashlib import sha256
from pathlib import Path

from workflow.config import AppConfig
from workflow.state import (AcquisitionAction, Artifact, ExperimentProcedure, HumanApproval,
                   Setting, WorkflowState)
from fixtures.hardware import RecordingFakeBackend

def approved_fake_fixture(scientist_proposal):
    inputs = Path("data/fake-input").resolve()
    inputs.mkdir(parents=True, exist_ok=True)
    calibration_path = inputs / "fixture-calibration.json"
    calibration_path.write_text('{"kind": "test-only fixture"}', encoding="utf-8")
    calibration = Artifact(
        path=str(calibration_path), description="TEST ONLY calibration fixture",
        sha256=sha256(calibration_path.read_bytes()).hexdigest(),
    )
    backend = RecordingFakeBackend(calibration)
    config = AppConfig(hardware_backend="test", hardware_input_path=inputs,
                       run_output_path=Path("outputs/fake-runs"),
                       max_acquisition_actions=1, max_acquisition_measurements=2)
    procedure = ExperimentProcedure(
        proposal_id=scientist_proposal.id, version=1,
        actions=[AcquisitionAction(
            action="fixture.capture", instruction="Create test-only fixture data",
            requested_settings={"count": Setting(value=2, unit="frame")},
        )],
        prerequisites=["fixture setup verified"],
        calibration_artifacts={"fixture-calibration": calibration},
        expected_outputs=["TEST ONLY fixture.json"],
    )
    state = WorkflowState(user_request="Test acquisition interface only",
                          proposals=[scientist_proposal], procedures=[procedure])
    review = runner(state, procedure.id, config=config, backend=backend)
    assert review.review.status == "ready"
    # TEST ONLY trusted-caller approval; not an agent decision or approval UI.
    approval = HumanApproval(
        procedure_id=procedure.id, procedure_version=procedure.version,
        procedure_fingerprint=review.review.procedure_fingerprint,
        review_fingerprint=review.review.review_fingerprint,
        status="approved", reviewer="Test fixture caller",
    )
    return runner(review.state, procedure.id, config=config, backend=backend,
                  execute=True, reviewed=review.review, approval=approval)
```

Records and files explicitly declare `data_origin="test_only"`. The fake's count
bounds concern fixture generation only. A physical harness must declare
`data_origin="physical"`; test mode rejects physical backends. `external` mode
alone connects to nothing: an explicit trusted backend implementation is required.

## Acquisition journal, cancellation, and recovery

Each attempt has a unique UUID. A caller can supply `attempt_id` as an idempotency
key; otherwise one is generated. The tool exclusively creates
`run_output_path/acquisitions/<attempt-id>/`, saves `review.json` and `approval.json`,
and flushes an atomic `record.json` marked `started` before hardware calls. It
flushes progress before/after each action and after cleanup. Raw data uses new
per-action directories under the approved output subdirectory; measurements are
never overwritten by the Runner.

Records include procedure identity/version/fingerprint, approval/review references,
backend identity, physical/test-only provenance, UTC timestamps, requested actions
and settings, actual settings, raw-data hashes, progress, errors, and cleanup.
Per-action actual settings retain parameter names; the top-level summary uses
`<zero-based-action-index>.<parameter>` so later readings cannot overwrite earlier
ones. Actual values are reported by the backend, not filled in from requests.

Statuses distinguish `started`, `completed`, `failed`, `partial`, and `aborted`
(legacy records may also be `pending`). Failures after saved measurements are
partial; aborted runs may retain data too. Pending and started/uncertain actions
remain distinct. Only completed actions with valid reports and no errors yield
completion. The Experimentor can inspect partial data but cannot treat a still-
started acquisition as finished. Scientist evidence checks require completed
acquisition chains; fixture data must remain explicitly nonphysical.

On failure/cancellation, backend-defined cleanup is invoked when supported, and
its outcome is recorded. No universal safe device setting is invented. An optional
`threading.Event` passed as `cancel_event` provides cancellation; action, run, and
cleanup deadlines use cooperative monotonic-time hooks. Preflight blocks backends
that do not declare deadline/cancellation support, and overruns cannot be successful.
**Software timeouts and cleanup reports do not guarantee physical equipment stopped.**
Future harnesses need verified interlocks, abort handling, and serialization of
equipment access across different attempts/processes; Python cannot forcibly and
safely stop arbitrary device calls here.

No acquisition is automatically retried or resumed. Existing attempt directories
permanently block that ID, even if a crash left no record. Attempts in caller state
are also protected. The application must retain IDs for retries: a different ID
means a new acquisition. Never delete an attempt directory to force a replay.
For an incomplete attempt:

1. Inspect its journal, snapshots, and raw files. A process kill may leave a started
   action or files not yet indexed in metadata; its physical outcome is uncertain.
2. Confirm equipment state and cleanup with the operator/harness. A killed process
   cannot execute Python cleanup.
3. Restore metadata with `AcquisitionResult.model_validate_json(...)` if needed,
   resolve technical issues, and use fresh preflight/matching approval before
   explicitly starting another attempt. Revised procedure/context needs renewed approval.

File flushes and atomic metadata replacement are not transactional hardware control
or protection against storage/power failure. Cleanup is still attempted after a
progress-write failure, but persistence then cannot be guaranteed. Treat incomplete
attempt directories conservatively. For numerical analysis of Runner outputs,
configure `numerical_input_path` to the saved acquisition directory/root; keep it
disjoint from the numerical analysis output root.

## Future laboratory harness

Implement `HardwareBackend.describe`, `validate`, `execute`, and `cleanup` in a
trusted adapter. `describe` supplies cached verified actions, parameter kinds,
units/limits, prerequisite evidence, calibration artifacts, and supported deadline,
cancellation, and cleanup behavior. `validate` adds read-only checks and cannot
waive deterministic validation. Neither operation may connect, discover, configure,
illuminate, or acquire. `execute` receives one structured action, declared inputs,
dedicated output directory, reviewed capability fingerprint, monotonic deadline,
and cancellation event; it returns actual settings, artifacts, and errors through
`ActionReport`. Return partial files on device errors and preserve raw measurements.

No real drivers, physical cleanup procedure, multi-user authentication, or remote
approval service is implemented. Local CLI approval and orchestration are available.
Offline Runner tests use only temporary files and
the recording fake. They verify approval gates, review changes, progress recording,
partial failures, cancellation, cleanup, and recovery, not physical safety,
calibration, timing guarantees, or scientific validity.

## Persistent workflow and CLI

The normal path is Scientist proposal → Experimentor procedure → Runner preflight
→ persistent human review → approved Runner acquisition → Experimentor numerical
analysis → Scientist interpretation. Unsupported technical procedures return to the
Experimentor; scientific decisions return to the Scientist. Missing information
pauses with questions. Missing models, unavailable hardware, or disabled numerical
execution return an actionable blocked state, not an automatic repair loop.
Procedure planning receives the cached backend capability description in
`WorkflowState.control.hardware_context`; this never replaces Runner revalidation.
Analysis-only work does not require hardware to be available.

The Scientist can return `analysis_request` or an interpretation with
`follow_up="analysis"` to use saved data, `proposal` for additional measurements,
or `conclusion` to stop without fabricating findings. Ordinary interpretations
conclude by default. Multiple proposals can be handled sequentially, each with its
own procedure and approval. Further analysis never implies new acquisition. Partial
acquisitions first receive technical review; analysis of partial data requires an
explicit user response with `allow_partial_analysis=True` and stays labelled partial.

The installed/tested workflow dependencies are LangGraph **1.2.19** and
`langgraph-checkpoint-sqlite` **3.1.3**, tested with Python 3.14.4. Implementation
follows the official [interrupt documentation](https://docs.langchain.com/oss/python/langgraph/interrupts)
and [checkpoint documentation](https://docs.langchain.com/oss/python/langgraph/checkpointers):
`interrupt()` pauses under a stable `thread_id`, and `Command(resume=...)` resumes
that same thread. SQLite is a local checkpoint backend; no hosted service is used.

An explicitly selected offline fixture mode exercises software handoffs with canned
models, `RecordingFakeBackend`, and a fake numerical backend. Scripts are saved
but **never executed** by that fixture backend; metrics and artifacts are canned
and labelled TEST ONLY. The mode is not validated TM retrieval, scientific
reasoning, an optical simulator, or physical hardware execution. Approval is still
mandatory and must be submitted explicitly.

From the project root, after installing requirements:

```powershell
.\.venv\Scripts\python.exe main.py start "Check the software handoffs" --fixture
# Copy the printed workflow UUID, then inspect the complete pending review:
$workflowId = "UUID printed by start"
.\.venv\Scripts\python.exe main.py show $workflowId
# Copy review_fingerprint from that review, not merely the procedure version:
.\.venv\Scripts\python.exe main.py review $workflowId approve --fingerprint "DISPLAYED_REVIEW_HASH" --reviewer "Your name"
.\.venv\Scripts\python.exe main.py show $workflowId
```

Review decisions and other commands:

```powershell
# Reject a pending review; the revision will pause for another approval:
.\.venv\Scripts\python.exe main.py review $workflowId reject --feedback "Explain the calibration prerequisite" --reviewer "Your name"
# Respond only when a clarification is pending:
.\.venv\Scripts\python.exe main.py clarify $workflowId "The saved-data format is documented in the supplied context"
# Explicitly authorize partial-data analysis when the pending task needs it:
.\.venv\Scripts\python.exe main.py clarify $workflowId "Analyze only the saved partial data" --allow-partial-analysis
# Resume a blocked/interrupted workflow after fixing its integrations:
.\.venv\Scripts\python.exe main.py resume $workflowId
# Stop a workflow; this is not a physical emergency-stop command:
.\.venv\Scripts\python.exe main.py cancel $workflowId --reason "Stop this workflow"
# A new request on a completed workflow retains all records and counters:
.\.venv\Scripts\python.exe main.py follow-up $workflowId "Request additional analysis of the saved data"
# Operator reconciliation, only when an uncertain-acquisition interrupt is pending:
.\.venv\Scripts\python.exe main.py reconcile $workflowId "Operator inspected journal and confirmed equipment stopped" --confirmed-stopped
```

Plain `resume` while waiting returns the pending review/question unchanged, with no
model calls. Ambiguous input, missing rejection feedback, malformed reviewer data,
or an outdated approval fingerprint leaves the workflow paused. Only a matching
explicit approval decision constructs a `HumanApproval`, outside agent control.
The review presents the complete scientific proposal, exact procedure/fingerprints,
actions/settings, expected outputs, calibration identities, verified constraints,
preflight issues, and physical/test-only backend provenance. No model can override
these checks. Runner revalidation still happens after approval.

Use `--db PATH` before the command to choose another SQLite database:

```powershell
.\.venv\Scripts\python.exe main.py --db data/another.sqlite start "Another workflow" --fixture
```

Normal mode never substitutes fixtures:

```powershell
.\.venv\Scripts\python.exe main.py start "A real research question"
# Returns blocked with an unconfigured-model explanation until integrations exist.
```

Programmatic applications inject `workflow.graph.Integrations(scientist_model=...,
experimentor_model=..., hardware_backend=..., execution_backend=...)` into
`WorkflowApp(config, integrations)`. The chat models use the existing structured
output contract. For the CLI, `--factory package.module:build` loads an explicitly
selected trusted function `build(config) -> workflow.graph.Integrations`; provide it on each
process invocation requiring integrations. That function configures external
clients. Without explicit integrations or a factory, configured LLM settings select
the built-in `workflow.llm:build_integrations` factory described below. Secrets can come from
environment variables or in-memory `config.api_key` and are excluded from saved
configuration. Normal numerical execution and physical hardware remain unavailable
until their separately verified backends are implemented.

`WorkflowApp.start(...)` also accepts supplied literature, architecture path, and an
initial `WorkflowState` containing saved records. `show`, `submit`, `resume`, `cancel`,
and `follow_up` operate by workflow UUID. Every invocation reuses existing agent
functions and records. The original research request is retained; clarifications,
feedback, follow-ups, routing explanations, failures, and agent messages are appended
without replacing earlier proposals/results.

## Persistence, limits, and restart behavior

`WorkflowState.control` stores routing, active references, questions, configuration,
literature context, counters, and stopping reasons. LangGraph stores JSON-shaped
copies through a last-value state channel, with **no concatenating history reducer**.
Each new state contains history exactly once; older checkpoints intentionally retain
their historical snapshots. Model clients and backend objects are not serialized
and must be supplied again by an application/factory. Explicit fixture mode is
remembered and can reconstruct its deterministic test integrations on restart.
SQLite databases, journals, and local lock files are ignored by Git.

Factory-created integrations are cached by workflow UUID and factory path for the
lifetime of one `WorkflowApp`. Different workflows therefore receive backends built
from their own saved configuration; returning to a workflow reuses its services.
Explicitly injected `Integrations` retain their caller-owned shared lifetime.
`workflow/persistence.py` owns the writer lock and atomic analysis-progress
file operation; checkpoint-aware recovery and routing remain in `workflow/graph.py`.

Checkpoint durability is synchronous. Step/attempt reservations and the acquisition
UUID are committed before side-effect nodes run. A node containing an interrupt
restarts on resume; the human node performs no writes, model calls, or acquisition
before `interrupt()`. Its accepted decision is committed before acquisition is
scheduled. Operating-system writer locks serialize local database mutations and
release on process exit; different databases do not serialize physical equipment.

After a crash, `resume` inspects the reserved Runner journal. A matching completed
journal is imported without calling hardware again. A started, missing/corrupt,
partial, or otherwise uncertain claimed attempt pauses for operator reconciliation.
Reconciliation records a human statement, not proof of a successful measurement;
it returns for technical review and never replays the old attempt ID. A new revised
procedure requires another approval. There is **no exactly-once physical execution
guarantee** across hardware and SQLite, and no automatic interrupted acquisition retry.

Numerical operations similarly save code/attempt progress before model review and a
completed node receipt after routing. Replayed completed operations reuse that
receipt. An uncertain operation blocks for inspection rather than re-executing code.
If a model review fails after execution, recorded code, inputs, parameters, results,
and logs are retained; no fallback scientific finding is created. LLM requests may
be repeated after a crash before their node checkpoint; model calls are not claimed
to be exactly once. Storage failure can still lose the latest metadata.

Workflow steps, procedure revisions, numerical attempts, and acquisition attempts
have persisted budgets. Initial procedures for different proposals do not consume
the revision budget. In orchestrated analysis each dispatch reserves at most one
execution attempt (`max_execution_attempts=1` for that agent call); repairs return
through routing so global counters apply. Reservations are conservative: blocked,
unavailable, interrupted, or clarification-producing dispatches may consume a step
or attempt. Waiting for a human consumes neither repeated model calls nor repeated
steps. Terminal `completed`, `cancelled`, or `stopped` states do not rerun on resume.
Follow-ups retain counters. These limits are not monetary API caps or token billing.

The default workflow step budget is 10. Set larger budgets before starting when
testing multiple cycles, for example `$env:TM_MAX_WORKFLOW_ITERATIONS = "40"`.
Configuration is snapshotted at start; changing environment variables alone does
not rewrite a saved workflow. For a blocked workflow, `resume --refresh-config`
explicitly reloads current configuration while retaining counters; changed execution
context still requires review. Do not move active acquisition output directories:
journals are essential to conservative crash recovery.

The CLI is a trusted local operator interface, not multi-user authentication,
electronic-signature infrastructure, or an emergency stop. A concurrent command
cannot preempt a running device call; in-process callers can inject a cancellation
event, and future hardware needs independently verified abort/interlock behavior.
Natural-language clarification never turns an unverified limit or calibration into
a verified backend fact.

Integration tests run with temporary SQLite databases and fake models/backends.
They verify full handoffs, explicit approval, rejection/revision, restart, saved-data
follow-up, technical/scientific routing, questions, blocked integrations, partial and
failed work, limits, cancellation, receipt recovery, and history identity. They do
not verify LLM scientific reasoning, script correctness, real execution isolation,
or physical equipment. Current LangSmith transitively emits Python 3.14 deprecation
warnings; no hosted tracing service is needed by these tests.

## Next steps after review

### Reviewed offline TM numerical operations

#### Shared CLI, paper context and backend integration

`main.py` owns workflow command handling. `bench/workflow.py` retains the supplied `HUANG-S2`
and `RAFAYELYAN-APP-A` excerpts, research questions, model setup and factory paths.
It does not parse PDFs. Run `python main.py start --reuse-data-from WORKFLOW_ID`
to analyze completed registered acquisitions with reviewed numerical operations.
Use `--db PATH` consistently for workflows in another checkpoint database.
`show`, `resume`, `clarify` and `cancel` retain their workflow roles.
Use `--simulate` consistently when approving/resuming a synthetic bench workflow;
saved-data execution accepts backend-disabled workflows.

The old optical simulator remains removed. The new bench is selected explicitly
with `--simulate` and uses `optical_bench` settings and the `simulated` backend tag.
General workflows retain their configured integrations; simulation always requires
explicit `--simulate` selection and never falls back to physical acquisition. Old checkpoints with removed backend configuration can be
inspected using `main.py show WORKFLOW_ID`, but cannot be resumed as acquisition workflows.
Past experiment files remain unchanged. Registered dataset manifests remain
usable by the shared numerical analysis tools, including simulated provenance.

The synthetic bench implements `tools.hardware.HardwareBackend` and is
explicitly injected through `workflow.graph.Integrations.hardware_backend` (or the
trusted `main.py --factory module:function` integration factory), with explicit
configuration and setup context. The shared Runner owns preflight, exact-review
approval, budgets, cancellation and acquisition journals. Saved arrays/manifests
connect to `tools.tm_numerics`; tool discovery belongs in
`workflow.routing_catalog`. There is no automatic physical-hardware fallback.

The acquisition call path is `WorkflowApp` preflight/acquire nodes →
`agents.runner.runner` → `tools.hardware.preflight` / `execute_acquisition` →
the explicitly supplied backend. `agents.orchestrator` and the routing catalog
choose workflow actions; they do not construct acquisition backends. Without an
injected backend the shared hardware boundary uses `DisabledHardwareBackend`.
The paper launcher keeps its literature passages and agent setup separate from
this boundary. It starts new synthetic runs only with `--simulate`; saved-data
reuse starts from completed saved acquisitions.

One real saved-data path now implements `I = abs(X @ T.T)**2`, with input patterns
as rows `X[N,D]`, camera output channels as columns `I[N,M]`, and `T[M,D]`. These
shared calculations use NumPy; the bench inspection plots use Matplotlib. These are fixed reviewed Python functions,
not an optical simulator, generated-code sandbox, or hardware adapter.

Reproduce the complete saved-data request → reviewed calculation → Analyst review
workflow with one command after installing requirements:

```powershell
.\.venv\Scripts\python.exe -m checks.numerical --output outputs/numerical-check
```

Each invocation creates a fresh case directory with a deterministic algebraic
test-only dataset, separate evaluation-only ground truth, SQLite checkpoints, real
numerical results and reports. It performs no hardware acquisition and requires no
API credentials. Analyst/Scientist messages are canned and explicitly labelled;
the NumPy fits and their errors are real calculations, not canned metrics. The
fixture uses seed 123 for 130 general complex input vectors (3 input modes), 2
output channels, 100 fit rows and 30 held-out rows. These vectors are mathematical
test fields, not a verified phase-only LC-SLM command sequence or optical model.

#### Dataset contract

A registered JSON manifest identifies a separately registered NPZ artifact:

```json
{
  "version": 1,
  "arrays": {"id": "ARRAY_ARTIFACT_UUID", "path": "ABSOLUTE_INPUT_PATH.npz",
             "description": "Known fields and corresponding intensities",
             "sha256": "RECORDED_64_CHARACTER_SHA256"},
  "data_origin": "test_only",
  "provenance": "Description of how these saved data were obtained",
  "fit_indices": [0, 1, 2],
  "holdout_indices": [3]
}
```

The example illustrates the schema, not a sufficient measurement design. The NPZ
must contain exactly `X` (complex `[N,D]`), `intensities` (real `[N,M]`), and string
arrays `pattern_ids`, `measurement_pattern_ids`, `frame_ids` of length N. Pattern
and measurement pattern IDs must agree row by row. Pattern/frame IDs must be unique;
splits must be nonempty, unique, disjoint, in range and partition all rows. Identical
input vectors cannot be hidden across the split under different IDs. All values
must be finite, intensities nonnegative, and the fit input matrix must have full
column rank. There is no silent clipping, background subtraction, rescaling from
holdout data, pairing repair, split inference or data-format guessing.

The loader uses `allow_pickle=False`, exact NPZ keys, SHA-256 verification and
designated input-root checks. The manifest is capped at 1 MiB, uncompressed archives
at 128 MiB, array elements at 2 million, input modes at 128 and output channels at
256. This small-data implementation is deliberately bounded. Metadata pairing checks
cannot establish that a physical acquisition actually used the asserted patterns.
Full column rank is necessary here but does not prove phase-retrieval identifiability.

Origin is explicit: `physical`, `simulated` or `test_only`; conflicting source-record
and manifest origins are rejected. Imported saved acquisitions and numerical results
now support `data_origin` without requiring fabricated hardware/approval records.
Existing hardware provenance remains authoritative when supplied. Ground truth is
an optional **separate** registered NPZ containing only complex finite `T[M,D]`.
The fitting data archive rejects embedded ground truth.

#### Named requests and workflow

`AnalysisRequest.operation` accepts `tm.validate_dataset`, `tm.compare_reference_free`,
`tm.evaluate_saved_predictions`, or the synthetic bench's `bench.retrieve`.
Validation writes a report; comparison fits both
methods and evaluates them on the same final holdout. Saved-prediction evaluation
scores an existing completed reviewed comparison without fitting. No function name,
module import, shell text, or Python source supplied by a model is executed.

`Artifact` now includes a stable UUID `id`. Requests name `dataset_artifact_id` and
optional `ground_truth_artifact_id`, alongside existing `input_record_ids`. The
manifest's arrays artifact must be registered under the input records with matching
ID/path/hash metadata. Keep artifact IDs when copying/reconstructing records. Older
serialized artifacts without IDs receive defaults when loaded; persist that updated
state before referring to the IDs. Changed procedure artifact identities can require
renewed hardware approval. The offline fixture calibration now has a stable identity
across restarts.

Example Analyst request (UUIDs must come from the actual state):

```python
from workflow.state import AnalysisRequest, TMFitSettings
request = AnalysisRequest(
    calculation="Compare reference-free retrieval with final held-out prediction errors",
    operation="tm.compare_reference_free",
    input_record_ids=[acquisition.id],
    dataset_artifact_id=manifest_artifact.id,
    ground_truth_artifact_id=truth_artifact.id,  # omit when unknown
    parameters=TMFitSettings(iterations=400, restarts=2, seed=7,
                             tolerance=1e-10, step_size=1.0),
)
```

Set `TM_REVIEWED_NUMERICS_ENABLED=true` before starting a normal workflow, or pass
`AppConfig(reviewed_numerics_enabled=True, numerical_input_path=...)`. The reproduction
command enables this only for its case. It does **not** enable arbitrary Python:
the existing generated-code backend remains disabled independently. Disabled named
operations produce an explicit pending/unavailable execution result.

An Analyst `more_analysis` response with a named operation routes directly to the
existing bounded analysis node, through `tools.tm_numerics.execute_operation`. Existing
artifact IDs, upstream eligibility, partial-data authorization, global analysis
attempt limits and operation receipts apply. Results return to the Analyst before
Scientist interpretation. The Experimentor still owns acquisition/procedure design;
descriptive legacy analysis requests retain their earlier route. No model is needed
to dispatch the reviewed implementation itself.

#### Exact solver variants and isolation of evaluation

Both methods use only `X_fit` and `I_fit`, normalized by the mean fit intensity.
They share the same seeded random-phase least-squares initializations and number
of restarts. They select the best restart and iterate by **fit intensity MSE only**.
Iterations, relative fit-objective tolerance, and gradient step size are validated
and bounded (at most 2000 iterations and 8 restarts). The fit-only function has no
holdout or ground-truth argument. A cooperative deadline uses the configured
numerical timeout; it cannot forcibly interrupt a NumPy/LAPACK kernel.

- Generalized Gerchberg–Saxton (`ggs`) alternates measured-amplitude replacement
  `Z = sqrt(I_fit) * exp(i angle(X_fit W))` with range projection
  `W = pinv(X_fit) Z`, where `W=T.T`. This is the general linear-measurement
  alternating-projection variant; it is not a Fourier-only GS implementation.
- Intensity least squares (`intensity_ls`) minimizes
  `L = mean((abs(X_fit W)**2 - I_fit)**2)`. Its real-coordinate complex gradient is
  `4/(N*M) * X_fit.conj().T @ ((abs(X_fit W)**2-I_fit)*(X_fit W))`.
  Backtracking halves the requested step up to 30 times and accepts only finite,
  non-increasing fit loss. It is a simple intensity-LS baseline, not an exact
  reproduction of a named paper's initialization/rate schedule.

The alternating projection principle is consistent with the formulation discussed
in [Netrapalli et al., Phase Retrieval using Alternating Minimization](https://arxiv.org/abs/1306.0160).
Our exact variants, selection rule and gradient convention are specified above and
tested directly. Both are nonconvex procedures: no global convergence or recovery
guarantee is claimed. Some designs (including constrained constant-amplitude fields)
can have ambiguities beyond output-row phase. Neither dataset rank nor the small
noiseless test establishes feasibility on the actual laboratory setup.

After all fitting/selection, the code predicts holdout intensities. It then loads
optional ground truth for evaluation. Changing holdout observations/inputs or ground
truth must leave estimated matrices and selected restart/iterate unchanged; tests
assert exact matrix equality and identical fit trajectories under those changes.
No method selection by holdout score is performed. Reusing these final holdout scores
to tune future settings would invalidate their role as untouched final evaluation.

#### Saved outputs and metrics

Each execution writes new artifacts under the workflow analysis directory:

- `ggs.npz` and `intensity_ls.npz`: complex estimated `T`, final holdout predictions,
  per-element squared intensity errors, holdout indices, and optional per-row aligned
  matrix errors.
- `report.json`: source manifest/hash, origin/provenance, splits, seed/settings,
  fit-loss trajectories, stopping reasons, selected restart/iterate, normalization,
  shared initialization cost, per-method fitting runtime and scalar metrics.
- `reviewed_implementation.py`: hashed audit copy of the dispatch interface, with
  dependency hashes for the relocated implementations in `reviewed_sources/`; these
  snapshots are never imported/executed as generated scripts. The bench-specific
  `reviewed_bench_analysis.py` copy is retained.

The existing `NumericalAnalysisResult` stores input artifacts, reviewed code hash,
settings/seed, execution diagnostics, origin, output artifact IDs/hashes and finite
metrics. Holdout MSE is the mean squared intensity residual; holdout NMSE is the sum
of squared residuals divided by the sum of squared measured holdout intensities.
For zero denominator NMSE is omitted while MSE remains available. Matrix comparison
multiplies each estimated output row by its own optimal unit phase before reporting
Frobenius absolute/relative error; it does not align gain or fit using truth. Relative
matrix error is omitted for all-zero ground truth. Runtime is observational and will
vary across machines, while seeded estimates/settings should be reproducible in the
same numerical environment. Raw input files are not overwritten.

Comparison and saved-prediction evaluation also report measured holdout RMSE (square
root of MSE) and MAE (mean absolute residual). With separate truth, `holdout_noiseless_*`
scores compare against `abs(X_holdout @ T_true.T)**2`, distinguishing latent brightness
error from noisy observation error. `intensity_ls_minus_ggs.*` records paired metric
differences; a negative difference favors intensity least squares for that score.
Evaluation requests reference the original completed comparison ID and its dataset/
truth artifact IDs, retaining its exact fit settings. Saved output hashes, output-root
containment, matrix/prediction consistency and holdout indices are checked. Matrices
and predictions are reused unchanged; retained method runtimes are the original fit
runtimes, and total runtime measures the evaluation operation. New reports record
source analysis IDs and `fit_reused`. Callers can request this reviewed operation
without generating analysis code.

Tests include a complex directional finite-difference gradient check, independent
row-phase alignment, holdout/truth isolation, real workflow dispatch, malformed data,
pairing/splits, nonfinite values, rank, hashes/IDs, provenance and disabled execution.
Full optical realism, a physical adapter and a general-purpose sandbox remain
outside these shared numerical operations; the small synthetic bench is documented separately.

### Optional Analyst handoffs

`agents/analyst.py` uses the existing injected structured-output interface. Configure
`Integrations(analyst_model=your_chat_model, ...)` to insert:

```text
Scientist proposal → Analyst advice → Experimentor procedure
Saved numerical results → Analyst review → Scientist interpretation
```

Without an Analyst model, each handoff logs an explicit skipped-review message and
uses the previous route. No successful Analyst review is recorded. Configured model
or reference-validation failures block the handoff with an error, preserving state.
Analyst advice never changes approval, preflight, evidence eligibility or partial-data
rules. The Experimentor owns acquisition choices; the Scientist owns conclusions.

Run the canned Analyst in fixture mode (human approval is still required):

```powershell
.\.venv\Scripts\python.exe main.py start "Check Analyst handoffs" --fixture --analyst
.\.venv\Scripts\python.exe main.py show "WORKFLOW_UUID"
.\.venv\Scripts\python.exe main.py review "WORKFLOW_UUID" approve --fingerprint "DISPLAYED_REVIEW_HASH" --reviewer "Test operator"
```

Fixture Analyst selection survives restart. Plain `--fixture` retains the previous
workflow without an Analyst. No new dependency was added.

Standalone: `analyst(state, chat_model, stage="before_acquisition")` or
`stage="after_analysis"` returns an outcome and full replacement state. Optional
proposal/analysis UUID lists select explicit targets; architecture and supplied
literature context are read/preserved using the existing context helper.

`AnalystReview` records stage, status, target/evidence UUIDs, artifact paths,
recommendations, limitations, questions, optional `AnalysisRequest`, and code-derived
provenance. Status is `advised`, `reviewed`, `needs_information`, or `more_analysis`;
the latter two are not successful completed reviews. Questions pause for clarification;
structured calculation requests describe calculations and existing inputs and route
to the Experimentor under existing attempt limits. They cannot order measurements,
invoke hardware, grant approval, execute scripts or write scientific findings.

Code rejects invented IDs, mismatched stages/targets, unknown literature citations,
and paths unregistered under referenced records. Artifacts now have stable IDs as
well as paths and hashes; legacy Analyst path citations remain exact metadata references. The Analyst does
not read their contents. Successful result review requires completed numerical
results and eligible completed upstream acquisitions. Partial-input calculation
requests require explicit authorization and do not make partial evidence eligible
for Scientist findings.

Provenance is derived from source acquisitions: physical, simulated, test-only,
mixed or unknown. `HardwareExecution.data_origin` accepts an explicit simulated tag
for imported metadata and synthetic bench acquisitions. Missing provenance
produces an unknown label and limitation. Metadata checks cannot prove every prose
claim, validation independence, or scientific validity. Fixture reviews and tests
verify software contracts, not reasoning quality or numerical correctness.

The Analyst milestone identified the following numerical/data tooling needs. The
reviewed numerical operations above now implement basic manifests, split checks,
comparative metrics and saved outputs; the broader items remain future work:

- Dataset manifests with origin, hashes, calibration, preprocessing, units, shapes,
  measurement budgets and train/validation/test membership.
- Read-only saved-data/artifact inspection and input/output correspondence checks.
- Validation split and leakage checks for genuinely independent held-out data.
- Reproducible metrics, uncertainty estimates and phase-ambiguity-aware comparisons
  tied to proposal evaluation criteria.
- Provenance auditing and structured calculation execution through a verified isolated
  backend, retaining scripts, parameters, seeds, logs and output artifacts.

The small synthetic workflow bench is now available; richer physical simulation,
data/preprocessing tools, uncertainty estimators and physical hardware remain future
work. The reviewed numerical paths execute real local calculations.

Confirm laboratory equipment, limits, calibration, and setup description. Later
steps will add literature retrieval, a verified isolated numerical backend, and a
verified laboratory harness. There is no RAG, PDF ingestion, embeddings, search,
hardware driver, or laboratory scientific benchmark demonstration.
Physical hardware behavior remains untested.
