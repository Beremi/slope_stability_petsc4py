from __future__ import annotations

import json
import tomllib
from pathlib import Path

from petsc_ssr.cli import main as cli_main
from petsc_ssr.cli.commands.case import case_override, explain_case_payload, validate_all_cases_payload, validate_case_payload


ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = ROOT / "benchmarks" / "cases"


def test_case_command_payloads_are_importable_without_cli_main_dispatch() -> None:
    case_toml = CASE_ROOT / "3d-heterogeneous-ssr-p4" / "case.toml"

    validation = validate_case_payload(case_toml)
    explanation = explain_case_payload(case_toml)

    assert validation["case"] == "3d-heterogeneous-ssr-p4"
    assert validation["linear_profile"] == "pmg-deflated-baseline"
    assert validation["resolved_pmg"]["p2_policy"] == "cap"
    assert explanation["linear"]["profile"] == "pmg-deflated-baseline"
    assert explanation["continuation"]["profile"] == "indirect-classic"


def test_validate_all_cases_payload_reports_committed_case_set() -> None:
    payload = validate_all_cases_payload(CASE_ROOT)

    assert payload["count"] == 19
    assert payload["valid"] == 19
    assert payload["errors"] == 0
    cases = {entry["case"]: entry for entry in payload["cases"]}
    assert cases["3d-heterogeneous-ssr-p4"]["linear_profile"] == "pmg-deflated-baseline"
    assert cases["3d-heterogeneous-ssr-p4-legacy-rollers"]["linear_profile"] == "pmg-deflated-baseline"
    assert cases["2d-sloan2013-seepage"]["seepage_profile"] == "sloan2013-steady"


def test_case_validate_all_cli_prints_summary(capsys) -> None:
    status = cli_main.main(["case", "validate", "--all"])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] == 19
    assert payload["errors"] == 0


def test_case_command_override_keeps_case_toml_mathematical(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("petsc_ssr.cli.commands.case.ENGINE_ROOT", tmp_path)
    source = CASE_ROOT / "3d-heterogeneous-ssr-p4" / "case.toml"

    override = case_override(source, profile="pmg-deflated-baseline", output_preset="performance")
    raw = tomllib.loads(override.read_text(encoding="utf-8"))

    assert override.is_relative_to(tmp_path / ".local" / "tmp" / "case_overrides")
    assert raw["linear"] == {"profile": "pmg-deflated-baseline"}
    assert raw["output"] == {"preset": "performance"}
    assert "execution" not in raw
