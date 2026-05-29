from __future__ import annotations

import json
from pathlib import Path

from petsc_ssr.cli import main as cli_main
from petsc_ssr.cli.commands.targets import validate_targets_payload


def test_validate_targets_payload_reports_first_class_and_legacy_files(tmp_path: Path) -> None:
    targets = tmp_path / "targets"
    targets.mkdir()
    (targets / "demo.json").write_text(
        json.dumps({"case": "demo", "metrics": {"wall_time": {"max": 1.0}}}),
        encoding="utf-8",
    )
    (targets / "legacy.json").write_text(json.dumps({"benchmark_id": "legacy-demo", "runs": []}), encoding="utf-8")

    payload = validate_targets_payload(targets)

    assert payload["count"] == 2
    assert payload["validated"] == 1
    assert payload["legacy_parse_only"] == 1
    assert payload["errors"] == 0
    assert {entry["kind"] for entry in payload["files"]} == {"first_class", "legacy_parse_only"}


def test_validate_targets_payload_reports_schema_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"case": "demo", "metrics": {"wall_time": {"maximum": 1.0}}}),
        encoding="utf-8",
    )

    payload = validate_targets_payload(bad)

    assert payload["count"] == 1
    assert payload["validated"] == 0
    assert payload["errors"] == 1
    assert "maximum" in payload["issues"][0]["error"]


def test_targets_validate_cli_returns_nonzero_for_invalid_targets(tmp_path: Path, capsys) -> None:
    target = tmp_path / "bad.json"
    target.write_text(
        json.dumps({"case": "demo", "metrics": {"wall_time": {"maximum": 1.0}}}),
        encoding="utf-8",
    )

    status = cli_main.main(["targets", "validate", str(target)])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert status == 2
    assert payload["errors"] == 1
    assert "maximum" in payload["issues"][0]["error"]
