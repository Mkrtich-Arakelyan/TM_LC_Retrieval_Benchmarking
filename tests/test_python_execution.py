import hashlib
from pathlib import Path

import pytest

from agents.common import read_context
from state import AnalysisCode, Artifact
from tools.python_execution import (BackendReport, checked_inputs, execute_python,
                                    save_script, validated_metrics)


def make_code(setup):
    state, config, architecture = setup
    script = "raise RuntimeError('This script must never execute on the host')\n"
    code = AnalysisCode(acquisition_ids=[state.acquisitions[0].id],
        input_data=checked_inputs(state.acquisitions[0].raw_data, config),
        code=save_script(script, config), method_settings={"normalize": True}, random_seed=7,
        technical_explanation="Fixture code only", scientific_context=read_context(architecture, None))
    return code


def test_disabled_backend_does_not_call_injected_backend(experiment_setup, fake_backend):
    _, config, _ = experiment_setup
    code = make_code(experiment_setup)
    backend = fake_backend()
    result = execute_python(code, config, attempt_number=1, backend=backend)
    assert result.status == "pending" and result.execution.status == "unavailable"
    assert "not executed" in result.execution.reason
    assert result.execution.exit_code is None
    assert not result.metrics and not backend.calls
    assert code.code.sha256 == hashlib.sha256(Path(code.code.path).read_bytes()).hexdigest()


def test_fake_success_and_failure_preserve_attempt_metadata(experiment_setup, fake_backend):
    _, config, _ = experiment_setup
    config.numerical_backend = "test"
    code = make_code(experiment_setup)
    backend = fake_backend((1, 0))
    failed = execute_python(code, config, attempt_number=1, backend=backend)
    completed = execute_python(code, config, attempt_number=2, backend=backend)
    assert failed.status == "failed" and failed.execution.exit_code == 1
    assert failed.errors and not failed.metrics and not failed.outputs
    assert failed.execution.stdout == "fake stdout" and failed.execution.stderr == "fake stderr"
    assert completed.status == "completed" and completed.execution.exit_code == 0
    assert completed.metrics == {"fixture_metric": 0.25}
    assert completed.code_record_id == code.id and completed.random_seed == 7
    assert completed.method_settings == {"normalize": True}
    assert completed.outputs[0].sha256
    assert backend.calls[0]["timeout_seconds"] == config.numerical_timeout_seconds
    assert backend.calls[0]["inputs"] == [Path(code.input_data[0].path)]
    assert backend.calls[0]["output_dir"] != backend.calls[1]["output_dir"]
    assert Path(code.input_data[0].path).read_text() == "fixture data only"


@pytest.mark.parametrize("value", [True, "1", None, [], {}, float("nan"), float("inf"), -float("inf"), 10**1000])
def test_metrics_are_untrusted(value):
    with pytest.raises(ValueError, match="Invalid or non-finite metric"):
        validated_metrics({"error": value})


def test_empty_metric_name_rejected():
    with pytest.raises(ValueError):
        validated_metrics({" ": 1.0})


def test_path_and_hash_policy(experiment_setup, tmp_path):
    _, config, _ = experiment_setup
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    with pytest.raises(ValueError, match="inside"):
        checked_inputs([Artifact(path=str(outside), description="Forbidden")], config)
    with pytest.raises(ValueError, match="hash mismatch"):
        checked_inputs([Artifact(path=str(config.numerical_input_path / "measurements.txt"),
                                description="Wrong hash", sha256="0" * 64)], config)
    code = make_code(experiment_setup)
    Path(code.code.path).write_text("modified")
    with pytest.raises(ValueError, match="Script hash mismatch"):
        execute_python(code, config, attempt_number=1)
    config.run_output_path = config.numerical_input_path
    with pytest.raises(ValueError, match="disjoint"):
        save_script("pass", config)


@pytest.mark.parametrize("case", ["metrics", "outside_output", "timeout", "exception", "empty"])
def test_invalid_backend_results_are_failed_records(experiment_setup, tmp_path, case):
    _, config, _ = experiment_setup
    config.numerical_backend = "test"
    code = make_code(experiment_setup)
    outside = tmp_path / "escape.txt"
    outside.write_text("outside")

    class InvalidBackend:
        def run(self, **kwargs):
            if case == "exception":
                raise TimeoutError("fake timed out")
            return BackendReport(exit_code=0, stdout="inspection log", elapsed_seconds=100 if case == "timeout" else 0.01,
                metrics={"bad": "not a number"} if case == "metrics" else {},
                outputs=[Artifact(path=str(outside), description="Forbidden output")] if case == "outside_output" else [])

    result = execute_python(code, config, attempt_number=1, backend=InvalidBackend())
    assert result.status == "failed" and result.errors
    assert result.execution.status == "failed"
    assert not result.metrics and not result.outputs
    if case != "exception":
        assert result.execution.stdout == "inspection log"


def test_tool_attempt_limit(experiment_setup, fake_backend):
    _, config, _ = experiment_setup
    backend = fake_backend()
    with pytest.raises(ValueError, match="attempt limit"):
        execute_python(make_code(experiment_setup), config,
                       attempt_number=config.max_execution_attempts + 1, backend=backend)
    assert not backend.calls


def test_symlink_input_escape_rejected_when_supported(experiment_setup, tmp_path):
    _, config, _ = experiment_setup
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = config.numerical_input_path / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Creating symlinks requires OS permission")
    with pytest.raises(ValueError, match="inside"):
        checked_inputs([Artifact(path=str(link), description="Symlink escape")], config)
