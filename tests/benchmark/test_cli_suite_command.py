from __future__ import annotations

import json
from pathlib import Path

from petsc_ssr.cli import main as cli_main
from petsc_ssr.cli.commands.suite import compare_suite_targets, expand_suite_manifest, validate_suite_payload, write_suite_report


ROOT = Path(__file__).resolve().parents[2]


def test_suite_command_module_expands_manifest_without_cli_dispatch(tmp_path: Path) -> None:
    manifest = expand_suite_manifest(
        ROOT / "benchmarks" / "suites" / "local-32-smoke.toml",
        output=tmp_path / "manifest.json",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["suite"]["id"] == "local-32-smoke"
    assert [run["ranks"] for run in payload["runs"]] == [1, 2, 4, 8, 16, 32]
    assert payload["runs"][-1]["resolved_profile"]["pmg"]["p1_active_ranks"] == 16
    assert payload["runs"][-1]["resolved_profile"]["linear"]["native_algorithm"] == "pmg-deflated"


def test_suite_command_module_validates_without_writing_manifest() -> None:
    payload = validate_suite_payload(ROOT / "benchmarks" / "suites" / "local-32-smoke.toml")

    assert payload["suite"]["id"] == "local-32-smoke"
    assert payload["suite"]["profiles"] == ["pmg-deflated-baseline"]
    assert payload["suite"]["ranks"] == [1, 2, 4, 8, 16, 32]
    assert payload["run_count"] == 6
    assert "pmg-deflated-baseline@32" in payload["resolved_profile_groups"]


def test_suite_validate_cli_prints_payload(capsys) -> None:
    status = cli_main.main(["suite", "validate", str(ROOT / "benchmarks" / "suites" / "local-32-smoke.toml")])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["suite"]["id"] == "local-32-smoke"
    assert payload["run_count"] == 6


def test_suite_command_module_reports_and_compares_targets(tmp_path: Path) -> None:
    manifest = expand_suite_manifest(
        ROOT / "benchmarks" / "suites" / "local-32-smoke.toml",
        output=tmp_path / "manifest.json",
    )

    report = write_suite_report(tmp_path)
    comparison = compare_suite_targets(tmp_path, ROOT / "benchmarks" / "targets" / "local-32")

    assert manifest.exists()
    assert report.exists()
    assert "Local 32-core smoke sweep" in report.read_text(encoding="utf-8")
    assert json.loads(comparison.read_text(encoding="utf-8"))["rows"]
