"""Environment configuration, loaded explicitly; imports have no side effects."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    llm_provider: str | None = None
    llm_model: str | None = None
    api_key: SecretStr | None = Field(default=None, exclude=True)
    papers_path: Path = Path("papers")
    index_path: Path = Path("data/index")
    run_output_path: Path = Path("outputs/runs")
    max_workflow_iterations: int = Field(default=10, gt=0)
    workflow_database_path: Path = Path("data/workflows.sqlite")
    max_procedure_revisions: int = Field(default=3, ge=0)
    max_workflow_analysis_attempts: int = Field(default=3, gt=0)
    max_workflow_acquisition_attempts: int = Field(default=3, gt=0)
    numerical_timeout_seconds: int = Field(default=60, gt=0)
    numerical_backend: Literal["disabled", "test"] = "disabled"
    reviewed_numerics_enabled: bool = False
    numerical_input_path: Path = Path("data/raw")
    max_execution_attempts: int = Field(default=2, gt=0)
    hardware_backend: Literal["disabled", "test", "external"] = "disabled"
    hardware_input_path: Path = Path("data/raw")
    max_acquisition_actions: int | None = Field(default=None, gt=0)
    max_acquisition_measurements: int | None = Field(default=None, gt=0)
    hardware_action_timeout_seconds: int = Field(default=60, gt=0)
    hardware_run_timeout_seconds: int = Field(default=600, gt=0)
    # No operating limits are known or implied by the reference specifications.
    verified_hardware_limits: None = None

    @classmethod
    def from_env(cls) -> "AppConfig":
        names = {
            "llm_provider": "TM_LLM_PROVIDER", "llm_model": "TM_LLM_MODEL",
            "api_key": "TM_API_KEY", "papers_path": "TM_PAPERS_PATH",
            "index_path": "TM_INDEX_PATH", "run_output_path": "TM_RUN_OUTPUT_PATH",
            "max_workflow_iterations": "TM_MAX_WORKFLOW_ITERATIONS",
            "workflow_database_path": "TM_WORKFLOW_DATABASE_PATH",
            "max_procedure_revisions": "TM_MAX_PROCEDURE_REVISIONS",
            "max_workflow_analysis_attempts": "TM_MAX_WORKFLOW_ANALYSIS_ATTEMPTS",
            "max_workflow_acquisition_attempts": "TM_MAX_WORKFLOW_ACQUISITION_ATTEMPTS",
            "numerical_timeout_seconds": "TM_NUMERICAL_TIMEOUT_SECONDS",
            "numerical_backend": "TM_NUMERICAL_BACKEND",
            "reviewed_numerics_enabled": "TM_REVIEWED_NUMERICS_ENABLED",
            "numerical_input_path": "TM_NUMERICAL_INPUT_PATH",
            "max_execution_attempts": "TM_MAX_EXECUTION_ATTEMPTS",
            "hardware_backend": "TM_HARDWARE_BACKEND",
            "hardware_input_path": "TM_HARDWARE_INPUT_PATH",
            "max_acquisition_actions": "TM_MAX_ACQUISITION_ACTIONS",
            "max_acquisition_measurements": "TM_MAX_ACQUISITION_MEASUREMENTS",
            "hardware_action_timeout_seconds": "TM_HARDWARE_ACTION_TIMEOUT_SECONDS",
            "hardware_run_timeout_seconds": "TM_HARDWARE_RUN_TIMEOUT_SECONDS",
        }
        return cls(**{field: os.environ[name] for field, name in names.items() if name in os.environ})
