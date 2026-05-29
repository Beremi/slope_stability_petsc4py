from __future__ import annotations

import json
import tomllib
from pathlib import Path

from petsc_ssr.runners import run_case_from_config


ROOT = Path(__file__).resolve().parents[2]


def test_mechanics_dry_run_writes_resolved_artifact_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "dry-run"
    case_toml = ROOT / "benchmarks" / "cases" / "2d-homogeneous-ssr" / "case.toml"

    status = run_case_from_config.main([str(case_toml), "--dry-run", "--output-dir", str(output_dir)])

    assert status == 0
    data_dir = output_dir / "data"
    assert (data_dir / "problem.json").exists()
    assert (data_dir / "resolved_options.txt").exists()
    assert (data_dir / "environment.json").exists()
    assert (data_dir / "resolved_run_manifest.json").exists()
    assert (data_dir / "resolved_config.toml").exists()
    assert (output_dir / "exports" / "resolved_config.toml").exists()
    assert (output_dir / "generated_case.toml").exists()
    assert (data_dir / "native_problem_manifest.json").exists()
    assert (data_dir / "mechanics_bc_labels.csv").exists()
    assert not (data_dir / "mechanics_bc_nodes.csv").exists()

    manifest = json.loads((data_dir / "resolved_run_manifest.json").read_text(encoding="utf-8"))
    resolved_config_text = (data_dir / "resolved_config.toml").read_text(encoding="utf-8")
    resolved_config = tomllib.loads(resolved_config_text)
    resolved_options = (data_dir / "resolved_options.txt").read_text(encoding="utf-8")
    assert manifest["case"] == "2d-homogeneous-ssr"
    assert manifest["linear"]["profile"] == "pmg-deflated-baseline"
    assert manifest["continuation"]["algorithm"] == "indirect"
    assert manifest["newton"]["algorithm"] == "indirect-ssr"
    assert manifest["linear"]["algorithm"] == "ksp_deflated"
    assert manifest["linear"]["native_algorithm"] == "pmg-deflated"
    assert manifest["linear"]["pc_backend"] == "pmg_shell"
    assert manifest["linear"]["pc_variant"] == "pmg"
    assert manifest["linear"]["requested_pc_variant"] == "pmg"
    assert manifest["linear"]["pc_variant_fallback_reason"] is None
    assert manifest["artifacts"]["native_problem_manifest"] == str(data_dir / "native_problem_manifest.json")
    assert manifest["artifacts"]["mechanics_bc_nodes_csv"] is None
    assert "-native_problem_manifest" in resolved_options
    assert "-continuation_algorithm indirect" in resolved_options
    assert "-newton_algorithm indirect-ssr" in resolved_options
    assert "-linear_algorithm pmg-deflated" in resolved_options
    assert "-mechanics_bc_nodes_csv" not in resolved_options
    assert resolved_config["resolved"]["kind"] == "petsc_ssr_resolved_config"
    assert resolved_config["case"]["id"] == "2d-homogeneous-ssr"
    assert resolved_config["mesh"]["element"] == "P2"
    assert resolved_config["linear"]["profile"] == "pmg-deflated-baseline"
    assert resolved_config["continuation"]["algorithm"] == "indirect"
    assert resolved_config["newton"]["algorithm"] == "indirect-ssr"
    assert resolved_config["linear"]["algorithm"] == "ksp_deflated"
    assert resolved_config["linear"]["native_algorithm"] == "pmg-deflated"
    assert resolved_config["linear"]["pc_backend"] == "pmg_shell"
    assert resolved_config["linear"]["pc_variant"] == "pmg"
    assert resolved_config["linear"]["requested_pc_variant"] == "pmg"
    assert resolved_config["artifacts"]["native_problem_manifest"] == str(data_dir / "native_problem_manifest.json")
    assert resolved_config_text.startswith("[resolved]\n")


def test_mechanics_dry_run_resolves_relative_output_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    case_toml = ROOT / "benchmarks" / "cases" / "2d-homogeneous-ssr" / "case.toml"

    status = run_case_from_config.main([str(case_toml), "--dry-run", "--output-dir", "relative-run"])

    assert status == 0
    data_dir = tmp_path / "relative-run" / "data"
    native_manifest = json.loads((data_dir / "native_problem_manifest.json").read_text(encoding="utf-8"))

    assert (data_dir / "mechanics_bc_labels.csv").exists()
    assert native_manifest["native_inputs"]["mechanics_label_constraints_csv"] == str(
        data_dir / "mechanics_bc_labels.csv"
    )


def test_coordinate_constraint_table_is_debug_opt_in(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "dry-run-coordinate-debug"
    case_toml = ROOT / "benchmarks" / "cases" / "2d-homogeneous-ssr" / "case.toml"

    def _write_coordinate_table(_translation, out_dir):
        path = Path(out_dir) / "data" / "mechanics_bc_nodes.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x,y,z,cx,cy,cz\n0,0,0,1,1,0\n", encoding="utf-8")
        return path

    monkeypatch.setattr(run_case_from_config, "write_mechanics_constraint_table", _write_coordinate_table)

    status = run_case_from_config.main(
        [
            str(case_toml),
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--write-coordinate-bc-table",
        ]
    )

    assert status == 0
    data_dir = output_dir / "data"
    resolved_options = (data_dir / "resolved_options.txt").read_text(encoding="utf-8")
    manifest = json.loads((data_dir / "resolved_run_manifest.json").read_text(encoding="utf-8"))
    native_manifest = json.loads((data_dir / "native_problem_manifest.json").read_text(encoding="utf-8"))

    assert (data_dir / "mechanics_bc_nodes.csv").exists()
    assert "-debug_coordinate_bc_table true" in resolved_options
    assert "-mechanics_bc_nodes_csv" in resolved_options
    assert manifest["artifacts"]["mechanics_bc_nodes_csv"] == str(data_dir / "mechanics_bc_nodes.csv")
    assert native_manifest["native_inputs"]["debug_coordinate_bc_table"] is True
    assert native_manifest["native_inputs"]["mechanics_coordinate_constraints_csv"] == str(
        data_dir / "mechanics_bc_nodes.csv"
    )
    assert native_manifest["compatibility"]["coordinate_constraint_table_fallback_available"] is True


def test_mechanics_dry_run_records_output_preset_override(tmp_path: Path) -> None:
    output_dir = tmp_path / "dry-run-output-none"
    case_toml = ROOT / "benchmarks" / "cases" / "2d-homogeneous-ssr" / "case.toml"

    status = run_case_from_config.main(
        [str(case_toml), "--dry-run", "--output-dir", str(output_dir), "--output-preset", "none"]
    )

    assert status == 0
    data_dir = output_dir / "data"
    manifest = json.loads((data_dir / "resolved_run_manifest.json").read_text(encoding="utf-8"))
    resolved_config = tomllib.loads((data_dir / "resolved_config.toml").read_text(encoding="utf-8"))
    resolved_options = (data_dir / "resolved_options.txt").read_text(encoding="utf-8")

    assert manifest["output"] == {"preset": "none", "write_solution": False, "write_history": False}
    assert resolved_config["output"] == {"preset": "none", "write_solution": False, "write_history": False}
    assert "-summary_json" in resolved_options
    assert "-solution_vtk" not in resolved_options
    assert "-solution_binary" not in resolved_options


def test_coupled_mechanics_dry_run_records_seepage_profile(tmp_path: Path) -> None:
    output_dir = tmp_path / "dry-run-coupled"
    case_toml = ROOT / "benchmarks" / "cases" / "3d-homogeneous-seepage-ssr-concave" / "case.toml"

    status = run_case_from_config.main([str(case_toml), "--dry-run", "--output-dir", str(output_dir)])

    assert status == 0
    data_dir = output_dir / "data"
    manifest = json.loads((data_dir / "resolved_run_manifest.json").read_text(encoding="utf-8"))
    native_manifest = json.loads((data_dir / "native_problem_manifest.json").read_text(encoding="utf-8"))
    resolved_config = tomllib.loads((data_dir / "resolved_config.toml").read_text(encoding="utf-8"))
    resolved_options = (data_dir / "resolved_options.txt").read_text(encoding="utf-8")
    pressure_csv = output_dir / "hydro_prepass" / "data" / "coupled_pressure_nodes.csv"
    assert manifest["case"] == "3d-homogeneous-seepage-ssr-concave"
    assert manifest["seepage"]["profile"] == "darcy-tight"
    assert manifest["seepage"]["coupled"] is True
    assert manifest["seepage"]["linear_max_iter"] == 500
    assert manifest["compatibility"]["seepage_pressure_coordinate_bridge_active"] is True
    assert manifest["compatibility"]["seepage_pressure_source"] == "hydro_prepass_coordinate_bridge"
    assert manifest["compatibility"]["seepage_pressure_csv"] == str(pressure_csv)
    assert manifest["artifacts"]["seepage_pressure_csv"] == str(pressure_csv)
    assert native_manifest["native_inputs"]["seepage_pressure_source"] == "hydro_prepass_coordinate_bridge"
    assert native_manifest["native_inputs"]["seepage_pressure_csv"] == str(pressure_csv)
    assert native_manifest["compatibility"]["coordinate_seepage_pressure_table_required"] is True
    assert "-seepage_pressure_source hydro_prepass_coordinate_bridge" in resolved_options
    assert f"-seepage_pressure_csv {pressure_csv}" in resolved_options
    assert "-seepage_grho 9.8100000000000005" in resolved_options
    assert not pressure_csv.exists()
    assert resolved_config["case"]["seepage_coupled"] is True
    assert resolved_config["seepage"]["profile"] == "darcy-tight"
    assert resolved_config["seepage"]["nonlinear_max_iter"] == 50
    assert resolved_config["continuation"]["algorithm"] == "indirect"
    assert resolved_config["newton"]["algorithm"] == "indirect-ssr"


def test_seepage_dry_run_writes_resolved_artifact_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "dry-run-seepage"
    case_toml = ROOT / "benchmarks" / "cases" / "2d-sloan2013-seepage" / "case.toml"

    status = run_case_from_config.main([str(case_toml), "--dry-run", "--output-dir", str(output_dir)])

    assert status == 0
    data_dir = output_dir / "data"
    assert (data_dir / "hydro_options.txt").exists()
    assert (data_dir / "resolved_options.txt").exists()
    assert (data_dir / "environment.json").exists()
    assert (data_dir / "resolved_run_manifest.json").exists()
    assert (data_dir / "resolved_config.toml").exists()
    assert (data_dir / "native_problem_manifest.json").exists()
    assert (data_dir / "seepage_boundary_labels.csv").exists()
    assert (output_dir / "exports" / "resolved_config.toml").exists()
    assert (output_dir / "generated_case.toml").exists()

    manifest = json.loads((data_dir / "resolved_run_manifest.json").read_text(encoding="utf-8"))
    resolved_config_text = (data_dir / "resolved_config.toml").read_text(encoding="utf-8")
    resolved_config = tomllib.loads(resolved_config_text)
    assert manifest["case"] == "2d-sloan2013-seepage"
    assert manifest["analysis"] == "seepage"
    assert manifest["linear"]["profile"] == "pmg-deflated-baseline"
    assert manifest["linear"]["algorithm"] == "ksp_deflated"
    assert manifest["linear"]["native_algorithm"] == "gamg"
    assert manifest["linear"]["pc_backend"] == "pmg_shell"
    assert manifest["linear"]["pc_variant"] == "gamg"
    assert manifest["linear"]["requested_pc_variant"] == "pmg"
    assert manifest["linear"]["pc_variant_fallback_reason"] == "p1_has_no_p_hierarchy"
    assert manifest["seepage"]["profile"] == "sloan2013-steady"
    assert manifest["seepage"]["newton_max_it"] == 100
    assert manifest["artifacts"]["hydro_options"] == str(data_dir / "hydro_options.txt")
    assert manifest["artifacts"]["native_problem_manifest"] == str(data_dir / "native_problem_manifest.json")
    assert "-hydro_mesh" in (data_dir / "resolved_options.txt").read_text(encoding="utf-8")
    assert resolved_config["resolved"]["kind"] == "petsc_ssr_resolved_config"
    assert resolved_config["case"]["id"] == "2d-sloan2013-seepage"
    assert resolved_config["case"]["analysis"] == "seepage"
    assert resolved_config["mesh"]["element"] == "P1"
    assert resolved_config["linear"]["pc_backend"] == "pmg_shell"
    assert resolved_config["linear"]["algorithm"] == "ksp_deflated"
    assert resolved_config["linear"]["native_algorithm"] == "gamg"
    assert resolved_config["linear"]["pc_variant"] == "gamg"
    assert resolved_config["linear"]["requested_pc_variant"] == "pmg"
    assert resolved_config["linear"]["pc_variant_fallback_reason"] == "p1_has_no_p_hierarchy"
    assert resolved_config["seepage"]["profile"] == "sloan2013-steady"
    assert resolved_config["seepage"]["newton_max_it"] == 100
    assert resolved_config["seepage"]["head_mode"] == "support_piecewise"
    assert resolved_config["artifacts"]["seepage_boundary_labels_csv"] == str(data_dir / "seepage_boundary_labels.csv")
