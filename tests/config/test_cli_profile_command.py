from __future__ import annotations

import json
from pathlib import Path

from petsc_ssr.cli import main as cli_main
from petsc_ssr.cli.commands.profile import explain_profile_payload, validate_profiles_payload


def test_profile_explain_payload_resolves_rank_adaptive_solver_profile() -> None:
    payload = explain_profile_payload("pmg-deflated-baseline", world_size=32, element="P4")

    assert payload["kind"] == "solver"
    assert payload["profile"] == "pmg-deflated-baseline"
    assert payload["world_size"] == 32
    assert payload["element"] == "P4"
    assert payload["linear_algorithm"] == "ksp_deflated"
    assert payload["native_linear_algorithm"] == "pmg-deflated"
    assert payload["pc"] == {
        "backend": "pmg_shell",
        "variant": "pmg",
        "requested_variant": "pmg",
        "fallback_reason": None,
    }
    assert payload["pmg"]["p2_active_ranks"] == 32
    assert payload["pmg"]["p1_active_ranks"] == 16
    assert payload["pmg"]["apply_backend"] == "shell_vcycle"
    assert payload["pmg"]["coarse_pc_type"] == "gamg"
    assert payload["pmg"]["coarse_telescope_ksp_max_it"] == 5
    assert payload["pmg"]["p2_telescope_active_ranks"] == 0
    assert payload["pmg"]["smoother_max_it"] == 2
    assert payload["resolved"]["pmg_options_file"].endswith("configs/petsc/pmg_shell_baseline.opts")


def test_profile_explain_payload_records_p1_pc_fallback() -> None:
    payload = explain_profile_payload("pmg-deflated-baseline", world_size=8, element="P1")

    assert payload["pc"]["backend"] == "pmg_shell"
    assert payload["pc"]["variant"] == "gamg"
    assert payload["pc"]["requested_variant"] == "pmg"
    assert payload["pc"]["fallback_reason"] == "p1_has_no_p_hierarchy"
    assert payload["native_linear_algorithm"] == "gamg"
    assert payload["pmg"]["p1_active_ranks"] == 4


def test_profile_explain_cli_supports_control_profiles(capsys) -> None:
    status = cli_main.main(["profile", "explain", "indirect-classic", "--kind", "continuation"])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "continuation"
    assert payload["profile"] == "indirect-classic"
    assert payload["resolved"]["algorithm"] == "indirect"
    assert payload["resolved"]["omega_step_controller"] == "classic"


def test_profile_explain_cli_prints_solver_resolution(capsys) -> None:
    status = cli_main.main(["profile", "explain", "pmg-deflated-baseline", "--world-size", "32", "--element", "P4"])

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["profile"] == "pmg-deflated-baseline"
    assert payload["pmg"]["p2_active_ranks"] == 32
    assert payload["pmg"]["p1_active_ranks"] == 16
    assert payload["pmg"]["coarse_pc_type"] == "gamg"


def test_profile_validate_payload_checks_all_profile_families() -> None:
    payload = validate_profiles_payload(world_sizes=[1, 32], elements=["P1", "P4"])

    assert payload["ok"] is True
    assert payload["counts"]["solver"] == 3
    assert payload["counts"]["continuation"] == 2
    assert payload["counts"]["newton"] == 8
    assert payload["counts"]["seepage"] == 2
    solver = {entry["profile"]: entry for entry in payload["profiles"]["solver"]}
    pmg_checks = solver["pmg-deflated-baseline"]["checks"]
    assert any(check["world_size"] == 32 and check["element"] == "P4" and check["pmg_p1_active_ranks"] == 16 for check in pmg_checks)
    assert any(check["element"] == "P1" and check["pc_variant"] == "gamg" and check["native_linear_algorithm"] == "gamg" for check in pmg_checks)
    assert any(check["pmg_apply_backend"] == "shell_vcycle" and check["pmg_smoother_max_it"] == 2 for check in pmg_checks)


def test_profile_validate_cli_returns_nonzero_for_invalid_profile(monkeypatch, tmp_path: Path, capsys) -> None:
    from petsc_ssr.config import profiles as profile_module

    solver_root = tmp_path / "solver"
    solver_root.mkdir()
    (solver_root / "bad.toml").write_text(
        """
description = "bad"

[linear]
algorithm = "python_array_solver"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(profile_module, "SOLVER_PROFILE_DIR", solver_root)

    status = cli_main.main(["profile", "validate", "--kind", "solver"])
    payload = json.loads(capsys.readouterr().out)

    assert status == 2
    assert payload["ok"] is False
    assert payload["counts"]["solver"] == 0
    assert "python_array_solver" in payload["issues"][0]["error"]
