from __future__ import annotations

import json
from pathlib import Path

import pytest

from petsc_ssr.runtime.results import (
    load_command_manifest,
    load_environment_manifest,
    load_resolved_run_manifest,
    load_run_summary,
    run_artifact_manifest,
    run_artifacts,
)


def test_runtime_result_readers_use_standard_run_layout(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (tmp_path / "command.json").write_text(json.dumps({"command": ["petsc-ssr"]}), encoding="utf-8")
    (data_dir / "summary.json").write_text(json.dumps({"wall_time": 1.25}), encoding="utf-8")
    (data_dir / "resolved_run_manifest.json").write_text(json.dumps({"case": "tiny"}), encoding="utf-8")
    (data_dir / "environment.json").write_text(json.dumps({"mpi": {"size": 4}}), encoding="utf-8")

    artifacts = run_artifacts(tmp_path)

    assert artifacts.summary_json == data_dir / "summary.json"
    assert artifacts.resolved_config_toml == data_dir / "resolved_config.toml"
    assert artifacts.resolved_options_txt == data_dir / "resolved_options.txt"
    assert artifacts.petsc_log_txt == tmp_path / "logs" / "petsc_log.txt"
    assert artifacts.options_left_txt == tmp_path / "logs" / "options_left.txt"
    manifest = run_artifact_manifest(tmp_path)
    assert manifest["summary_json"] == str(data_dir / "summary.json")
    assert manifest["native_problem_manifest_json"] == str(data_dir / "native_problem_manifest.json")
    assert manifest["options_left_txt"] == str(tmp_path / "logs" / "options_left.txt")
    assert load_command_manifest(tmp_path) == {"command": ["petsc-ssr"]}
    assert load_run_summary(tmp_path) == {"wall_time": 1.25}
    assert load_resolved_run_manifest(tmp_path) == {"case": "tiny"}
    assert load_environment_manifest(tmp_path) == {"mpi": {"size": 4}}


def test_runtime_result_readers_treat_missing_artifacts_as_empty(tmp_path: Path) -> None:
    assert load_command_manifest(tmp_path) == {}
    assert load_run_summary(tmp_path) == {}
    assert load_resolved_run_manifest(tmp_path) == {}
    assert load_environment_manifest(tmp_path) == {}


def test_runtime_result_readers_reject_non_object_json(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "summary.json").write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected JSON object"):
        load_run_summary(tmp_path)
