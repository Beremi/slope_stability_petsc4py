from __future__ import annotations

import json
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from petsc_ssr.config.manifest import (
    ENVIRONMENT_MANIFEST_KIND,
    RUN_COMMAND_MANIFEST_KIND,
    RESOLVED_CONFIG_KIND,
    RESOLVED_RUN_MANIFEST_KIND,
    build_environment_manifest,
    build_resolved_config,
    build_resolved_run_manifest,
    build_run_command_manifest,
    dumps_resolved_config_toml,
)
from petsc_ssr.options import SsrOptions
from petsc_ssr.problem import ProblemSpec


def test_resolved_run_manifest_records_profile_ranks_and_artifacts(tmp_path: Path) -> None:
    problem = replace(
        ProblemSpec.tiny_box(),
        metadata={
            "continuation_profile": "indirect-classic",
            "continuation_algorithm": "indirect",
            "newton_profile": "indirect-regularized",
            "newton_algorithm": "indirect-ssr",
            "linear_profile": "pmg-deflated-baseline",
            "linear_algorithm": "ksp_deflated",
            "pc_backend": "pmg_shell",
            "requested_pc_variant": "pmg",
            "pc_variant_fallback_reason": None,
            "pmg_shell_p2_rank_policy": "cap",
            "pmg_shell_p1_rank_policy": "fraction",
            "seepage_coupled": True,
            "seepage_profile": "darcy-tight",
            "seepage_profile_description": "Darcy defaults",
            "seepage_linear_tolerance": 1.0e-10,
            "seepage_linear_max_iter": 500,
            "seepage_nonlinear_max_iter": 50,
            "native_problem_manifest": str(tmp_path / "run" / "data" / "native_problem_manifest.json"),
            "mechanics_bc_labels_csv": str(tmp_path / "run" / "data" / "mechanics_bc_labels.csv"),
            "mechanics_bc_nodes_csv": str(tmp_path / "run" / "data" / "mechanics_bc_nodes.csv"),
            "debug_coordinate_bc_table": True,
            "seepage_boundary_labels_csv": str(tmp_path / "run" / "data" / "seepage_boundary_labels.csv"),
        },
    )
    options = SsrOptions.current_baseline()
    options.profile_name = "pmg-deflated-baseline"

    manifest = build_resolved_run_manifest(problem, options, output_dir=tmp_path / "run", mpi_size=32)

    assert manifest["kind"] == RESOLVED_RUN_MANIFEST_KIND
    assert manifest["schema_version"] == 1
    assert manifest["case"] == "tiny_box"
    assert manifest["mpi"]["size"] == 32
    assert manifest["continuation"]["profile"] == "indirect-classic"
    assert manifest["continuation"]["algorithm"] == "indirect"
    assert manifest["continuation"]["method"] == options.continuation_method
    assert manifest["newton"]["profile"] == "indirect-regularized"
    assert manifest["newton"]["algorithm"] == "indirect-ssr"
    assert manifest["newton"]["stopping_criterion"] == options.newton_stopping_criterion
    assert manifest["linear"]["profile"] == "pmg-deflated-baseline"
    assert manifest["linear"]["algorithm"] == "ksp_deflated"
    assert manifest["linear"]["native_algorithm"] == "pmg-deflated"
    assert manifest["linear"]["pc_backend"] == "pmg_shell"
    assert manifest["linear"]["pc_variant"] == "pmg"
    assert manifest["linear"]["requested_pc_variant"] == "pmg"
    assert manifest["linear"]["pc_variant_fallback_reason"] is None
    assert manifest["pmg"]["p2_active_ranks"] == options.pmg.p2_active_ranks
    assert manifest["pmg"]["p1_rank_policy"] == "fraction"
    assert manifest["pmg"]["apply_backend"] == "shell_vcycle"
    assert manifest["pmg"]["coarse_pc_type"] == "gamg"
    assert manifest["pmg"]["coarse_telescope_ksp_max_it"] == 5
    assert manifest["pmg"]["p2_telescope_active_ranks"] == 0
    assert manifest["pmg"]["smoother_max_it"] == 2
    assert manifest["compatibility"]["mechanics_constraint_source"] == "label_table_preferred_coordinate_debug_available"
    assert manifest["compatibility"]["mechanics_label_table_active"] is True
    assert manifest["compatibility"]["mechanics_coordinate_debug_table_active"] is True
    assert manifest["compatibility"]["seepage_pressure_coordinate_bridge_active"] is False
    assert manifest["seepage"]["profile"] == "darcy-tight"
    assert manifest["seepage"]["coupled"] is True
    assert manifest["seepage"]["linear_max_iter"] == 500
    assert manifest["artifacts"]["native_problem_manifest"].endswith("native_problem_manifest.json")
    assert manifest["artifacts"]["resolved_config_toml"].endswith("data/resolved_config.toml")
    assert manifest["artifacts"]["resolved_options_txt"].endswith("data/resolved_options.txt")
    assert manifest["artifacts"]["mechanics_bc_labels_csv"].endswith("mechanics_bc_labels.csv")
    assert manifest["artifacts"]["stdout"].endswith("logs/stdout.txt")
    assert manifest["artifacts"]["options_left"].endswith("logs/options_left.txt")
    assert manifest["artifacts"]["options_left_txt"].endswith("logs/options_left.txt")


def test_resolved_config_toml_records_concrete_profiles_options_and_artifacts(tmp_path: Path) -> None:
    problem = replace(
        ProblemSpec.tiny_box(),
        metadata={
            "case_config": str(tmp_path / "case.toml"),
            "asset": "tiny_asset",
            "mesh_variant": "default",
            "elem_type": "P4",
            "continuation_profile": "indirect-classic",
            "continuation_profile_description": "continuation defaults",
            "continuation_algorithm": "indirect",
            "newton_profile": "indirect-regularized",
            "newton_profile_description": "newton defaults",
            "newton_algorithm": "indirect-ssr",
            "linear_profile": "pmg-deflated-baseline",
            "profile_description": "linear defaults",
            "linear_algorithm": "ksp_deflated",
            "pc_backend": "pmg_shell",
            "requested_pc_variant": "pmg",
            "pmg_shell_p2_rank_policy": "cap",
            "pmg_shell_p1_rank_policy": "fraction",
            "seepage_coupled": True,
            "seepage_profile": "darcy-tight",
            "seepage_profile_description": "Darcy defaults",
            "seepage_linear_tolerance": 1.0e-10,
            "seepage_linear_max_iter": 500,
            "seepage_nonlinear_max_iter": 50,
            "native_problem_manifest": str(tmp_path / "run" / "data" / "native_problem_manifest.json"),
            "mechanics_bc_labels_csv": str(tmp_path / "run" / "data" / "mechanics_bc_labels.csv"),
        },
    )
    options = SsrOptions.current_baseline()
    options.profile_name = "pmg-deflated-baseline"
    options.omega_max = 123.0
    options.continuation_step_max = 7
    options.linear.rtol = 0.05
    options.pmg.p2_active_ranks = 32
    options.pmg.p1_active_ranks = 16
    options.petsc_options = ["-options_left"]

    text = dumps_resolved_config_toml(build_resolved_config(problem, options, output_dir=tmp_path / "run", mpi_size=32))
    parsed = tomllib.loads(text)

    assert parsed["resolved"]["kind"] == RESOLVED_CONFIG_KIND
    assert parsed["case"]["id"] == "tiny_box"
    assert parsed["case"]["asset"] == "tiny_asset"
    assert parsed["mpi"]["size"] == 32
    assert parsed["continuation"]["profile"] == "indirect-classic"
    assert parsed["continuation"]["algorithm"] == "indirect"
    assert parsed["continuation"]["omega_max"] == 123.0
    assert parsed["continuation"]["step_max"] == 7
    assert parsed["newton"]["profile"] == "indirect-regularized"
    assert parsed["newton"]["algorithm"] == "indirect-ssr"
    assert parsed["linear"]["profile"] == "pmg-deflated-baseline"
    assert parsed["linear"]["algorithm"] == "ksp_deflated"
    assert parsed["linear"]["native_algorithm"] == "pmg-deflated"
    assert parsed["linear"]["rtol"] == 0.05
    assert parsed["linear"]["pc_backend"] == "pmg_shell"
    assert parsed["linear"]["pc_variant"] == "pmg"
    assert parsed["linear"]["requested_pc_variant"] == "pmg"
    assert parsed["pmg"]["p2_active_ranks"] == 32
    assert parsed["pmg"]["p1_active_ranks"] == 16
    assert parsed["pmg"]["p1_rank_policy"] == "fraction"
    assert parsed["pmg"]["apply_backend"] == "shell_vcycle"
    assert parsed["pmg"]["coarse_pc_type"] == "gamg"
    assert parsed["pmg"]["coarse_telescope_ksp_max_it"] == 5
    assert parsed["pmg"]["p2_telescope_ksp_max_it"] == 50
    assert parsed["pmg"]["smoother_max_it"] == 2
    assert parsed["compatibility"]["mechanics_constraint_source"] == "label_table"
    assert parsed["compatibility"]["mechanics_label_table_active"] is True
    assert parsed["compatibility"]["debug_coordinate_bc_table"] is False
    assert parsed["compatibility"]["seepage_pressure_coordinate_bridge_active"] is False
    assert parsed["seepage"]["profile"] == "darcy-tight"
    assert parsed["seepage"]["coupled"] is True
    assert parsed["seepage"]["linear_tolerance"] == 1.0e-10
    assert parsed["petsc"]["extra_options"] == ["-options_left"]
    assert parsed["artifacts"]["native_problem_manifest"].endswith("native_problem_manifest.json")


def test_environment_manifest_is_portable_and_filters_empty_env(tmp_path: Path) -> None:
    manifest = build_environment_manifest(
        mpi_size=4,
        env={
            "PETSC_DIR": "/opt/petsc",
            "PETSC_ARCH": None,
            "OMP_NUM_THREADS": "1",
            "UNRELATED": "ignored",
        },
        repo_root=tmp_path,
    )

    assert manifest["kind"] == ENVIRONMENT_MANIFEST_KIND
    assert manifest["schema_version"] == 1
    assert manifest["mpi"]["size"] == 4
    assert manifest["env"] == {"PETSC_DIR": "/opt/petsc", "OMP_NUM_THREADS": "1"}
    assert set(manifest["git"]) == {"commit", "branch", "dirty"}


def test_run_command_manifest_records_direct_invocation_and_profile(tmp_path: Path) -> None:
    problem = replace(
        ProblemSpec.tiny_box(),
        metadata={
            "linear_profile": "pmg-deflated-baseline",
            "linear_algorithm": "ksp_deflated",
            "pc_backend": "pmg_shell",
            "requested_pc_variant": "pmg",
            "pmg_shell_p2_rank_policy": "cap",
            "pmg_shell_p1_rank_policy": "fraction",
        },
    )
    options = SsrOptions.current_baseline()
    options.profile_name = "pmg-deflated-baseline"
    options.pmg.p2_active_ranks = 8
    options.pmg.p1_active_ranks = 4
    resolved = build_resolved_run_manifest(problem, options, output_dir=tmp_path / "run", mpi_size=8)

    payload = build_run_command_manifest(
        output_dir=tmp_path / "run",
        mpi_size=8,
        argv=["case.toml", "--output-dir", str(tmp_path / "run")],
        mode="run",
        entrypoint="petsc_ssr.runners.run_case_from_config",
        resolved_run_manifest=resolved,
    )

    assert payload["kind"] == RUN_COMMAND_MANIFEST_KIND
    assert payload["schema_version"] == 1
    assert payload["mode"] == "run"
    assert payload["entrypoint"] == "petsc_ssr.runners.run_case_from_config"
    assert payload["argv"][0] == "case.toml"
    assert payload["command"][0] == "petsc_ssr.runners.run_case_from_config"
    assert payload["case"] == "tiny_box"
    assert payload["profile"] == "pmg-deflated-baseline"
    assert payload["ranks"] == 8
    assert payload["resolved_profile"]["linear"]["native_algorithm"] == "pmg-deflated"
    assert payload["resolved_profile"]["pmg"]["p2_active_ranks"] == 8
    assert payload["artifacts"]["command_json"].endswith("command.json")


def test_preflight_writes_direct_command_manifest_without_clobbering_suite_payload(tmp_path: Path) -> None:
    from petsc_ssr.runners.run_case_from_config import _write_preflight_artifacts

    config = tmp_path / "case.toml"
    config.write_text("# copied preflight source\n", encoding="utf-8")
    output = tmp_path / "run"
    problem = replace(
        ProblemSpec.tiny_box(),
        metadata={
            "case_config": str(config),
            "linear_profile": "pmg-deflated-baseline",
            "linear_algorithm": "ksp_deflated",
            "pc_backend": "pmg_shell",
            "requested_pc_variant": "pmg",
        },
    )
    options = SsrOptions.current_baseline()
    options.profile_name = "pmg-deflated-baseline"
    translation = SimpleNamespace(problem=problem, options=options, config=None)

    _write_preflight_artifacts(
        translation,
        output,
        config,
        4,
        runner_argv=[str(config), "--dry-run"],
        mode="dry-run",
    )

    command_path = output / "command.json"
    payload = json.loads(command_path.read_text(encoding="utf-8"))
    assert payload["kind"] == RUN_COMMAND_MANIFEST_KIND
    assert payload["mode"] == "dry-run"
    assert payload["argv"] == [str(config), "--dry-run"]
    assert payload["resolved_profile"]["linear"]["profile"] == "pmg-deflated-baseline"
    assert (output / "data" / "resolved_run_manifest.json").exists()

    suite_payload = {"kind": "petsc_ssr_suite_command", "suite": "local-32-smoke"}
    command_path.write_text(json.dumps(suite_payload) + "\n", encoding="utf-8")
    _write_preflight_artifacts(
        translation,
        output,
        config,
        4,
        runner_argv=[str(config)],
        mode="run",
    )
    assert json.loads(command_path.read_text(encoding="utf-8")) == suite_payload
