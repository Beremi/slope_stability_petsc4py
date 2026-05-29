"""Portable readers for concrete run artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Standard artifact paths for one concrete run root."""

    output_dir: Path
    command_json: Path
    problem_json: Path
    native_problem_manifest_json: Path
    resolved_config_toml: Path
    resolved_options_txt: Path
    summary_json: Path
    continuation_curve_csv: Path
    resolved_run_manifest_json: Path
    environment_json: Path
    stdout_txt: Path
    petsc_log_txt: Path
    options_view_txt: Path
    options_left_txt: Path


def run_artifacts(output_dir: str | Path) -> RunArtifacts:
    root = Path(output_dir)
    data_dir = root / "data"
    logs_dir = root / "logs"
    return RunArtifacts(
        output_dir=root,
        command_json=root / "command.json",
        problem_json=data_dir / "problem.json",
        native_problem_manifest_json=data_dir / "native_problem_manifest.json",
        resolved_config_toml=data_dir / "resolved_config.toml",
        resolved_options_txt=data_dir / "resolved_options.txt",
        summary_json=data_dir / "summary.json",
        continuation_curve_csv=data_dir / "continuation_curve.csv",
        resolved_run_manifest_json=data_dir / "resolved_run_manifest.json",
        environment_json=data_dir / "environment.json",
        stdout_txt=logs_dir / "stdout.txt",
        petsc_log_txt=logs_dir / "petsc_log.txt",
        options_view_txt=logs_dir / "options_view.txt",
        options_left_txt=logs_dir / "options_left.txt",
    )


def run_artifact_manifest(output_dir: str | Path) -> dict[str, str]:
    """Return the standard run artifact contract as string paths."""

    return {name: str(path) for name, path in asdict(run_artifacts(output_dir)).items()}


def load_run_summary(output_dir: str | Path) -> dict[str, Any]:
    return _read_json_if_present(run_artifacts(output_dir).summary_json)


def load_command_manifest(output_dir: str | Path) -> dict[str, Any]:
    return _read_json_if_present(run_artifacts(output_dir).command_json)


def load_resolved_run_manifest(output_dir: str | Path) -> dict[str, Any]:
    return _read_json_if_present(run_artifacts(output_dir).resolved_run_manifest_json)


def load_environment_manifest(output_dir: str | Path) -> dict[str, Any]:
    return _read_json_if_present(run_artifacts(output_dir).environment_json)


def _read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload
