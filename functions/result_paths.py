from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_DIR_ENV_VAR = "B_AMATRICE_RESULTS_DIR"
RUN_NAME_ENV_VAR = "B_AMATRICE_RUN_NAME"
WORKFLOW_ORDER = (
    "pretraining",
    "legacy_validation",
    "validation",
    "training",
    "test",
)

WORKFLOW_RESULT_FOLDERS = {
    "pretraining": "a_training",
    "legacy_validation": "validation_legacy",
    "validation": "validation",
    "training": "training",
    "test": "test",
}


def _result_key(mapping: dict[str, str], workflow: str) -> str:
    try:
        return mapping[workflow]
    except KeyError as exc:
        valid = ", ".join(sorted(mapping))
        raise ValueError(
            f"Unknown workflow {workflow!r}. Expected one of: {valid}."
        ) from exc


def default_run_name(*, include_time: bool = True) -> str:
    fmt = "%Y%m%d_%H%M%S" if include_time else "%Y%m%d"
    return datetime.now().strftime(fmt)


def workflow_results_root(workflow: str) -> Path:
    return RESULTS_ROOT / _result_key(WORKFLOW_RESULT_FOLDERS, workflow)


def run_results_dir(workflow: str, run_name: str) -> Path:
    return workflow_results_root(workflow) / run_name


def ensure_results_dir(workflow: str, run_name: str) -> Path:
    path = run_results_dir(workflow, run_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_results_dir(
    workflow: str,
    *,
    env: Mapping[str, str] | None = None,
    fallback_run_name: str | None = None,
) -> Path:
    env_map = os.environ if env is None else env
    explicit_dir = env_map.get(RESULTS_DIR_ENV_VAR)
    if explicit_dir:
        path = Path(explicit_dir).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    run_name = env_map.get(RUN_NAME_ENV_VAR, fallback_run_name)
    if not run_name:
        raise ValueError(
            "No results directory configured. Set "
            f"{RESULTS_DIR_ENV_VAR} or {RUN_NAME_ENV_VAR}, "
            "or provide a fallback run name."
        )
    return ensure_results_dir(workflow, run_name)


def list_available_runs(workflow: str) -> list[str]:
    root = workflow_results_root(workflow)
    if not root.exists():
        return []
    return sorted(child.name for child in root.iterdir() if child.is_dir())


def resolve_run_dir(workflow: str, run_name: str) -> Path:
    canonical = run_results_dir(workflow, run_name)
    if canonical.exists():
        return canonical

    available = list_available_runs(workflow)
    available_str = ", ".join(available) if available else "none"
    raise FileNotFoundError(
        f"No results found for workflow {workflow!r} and run {run_name!r}. "
        f"Available runs: {available_str}."
    )
