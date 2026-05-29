from __future__ import annotations

import argparse
from pathlib import Path

from petsc_ssr.cli import main as cli_main
from petsc_ssr.cli.commands.doctor import doctor_payload
from petsc_ssr.cli.commands.run import build_run_case_argv


ROOT = Path(__file__).resolve().parents[2]
CASE_ROOT = ROOT / "benchmarks" / "cases"


def test_run_command_builds_forwarded_argv_without_implicit_debug_override(tmp_path: Path, monkeypatch) -> None:
    from petsc_ssr.cli.commands import case as case_command

    monkeypatch.setattr(case_command, "ENGINE_ROOT", tmp_path)
    args = argparse.Namespace(
        case_toml=CASE_ROOT / "3d-heterogeneous-ssr-p4" / "case.toml",
        profile="pmg-deflated-baseline",
        output_dir=tmp_path / "out",
        omega_max=2.5,
        continuation_step_max=12,
        linear_rtol=1e-8,
        ksp_max_it=300,
        refine_levels=1,
        output_preset="performance",
        petsc_opt=["-ksp_monitor", "-pc_type", "gamg"],
        force_c_baseline=False,
        write_coordinate_bc_table=True,
    )

    argv = build_run_case_argv(args)

    assert argv[0].startswith(str(tmp_path / ".local" / "tmp" / "case_overrides"))
    assert "--force-c-baseline" not in argv
    assert "--write-coordinate-bc-table" in argv
    assert ["--output-dir", str(tmp_path / "out")] == argv[1:3]
    assert "--output-preset" in argv
    assert "--petsc-opt=-ksp_monitor" in argv
    assert "--petsc-opt=gamg" in argv


def test_mesh_command_payload_is_importable_without_cli_dispatch(monkeypatch) -> None:
    from petsc_ssr.cli.commands import mesh as mesh_command

    class _Problem:
        name = "tiny"
        analysis = "ssr"
        elem_type = "P2"

    class _Config:
        problem = _Problem()

        def validate(self):
            return self

    class _Resolved:
        asset_name = "tiny_asset"
        variant_name = "default"
        mesh_path = None
        dimension = 2

    class _Mesh:
        coord = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        elem = [[0], [1], [2]]
        surf = [[0, 1], [1, 2]]
        q_mask = [[True, False], [True, True]]
        region_id_by_name = {"soil": 1}
        boundary_id_by_name = {"base": 1}
        nodesets = {"crest": [2]}

    class _Mechanical:
        material_rows = [{"name": "soil"}]

    monkeypatch.setattr(mesh_command, "load_run_case_config", lambda _path: _Config())
    monkeypatch.setattr(mesh_command, "resolve_problem_asset_from_config", lambda _cfg: _Resolved())
    monkeypatch.setattr(mesh_command, "build_mesh_for_resolved_asset", lambda _resolved, *, elem_type: _Mesh())
    monkeypatch.setattr(mesh_command, "load_mechanical_problem_spec", lambda _resolved: _Mechanical())

    payload = mesh_command.mesh_report_payload(Path("case.toml"))

    assert payload["case"] == "tiny"
    assert payload["asset"] == "tiny_asset"
    assert payload["dimension"] == 2
    assert payload["nodes"] == 3
    assert payload["cells"] == 1
    assert payload["free_component_dofs"] == 3
    assert payload["constrained_component_dofs"] == 1
    assert payload["materials"] == 1


def test_mesh_only_cli_reports_missing_mesh_extra_without_traceback(monkeypatch, capsys) -> None:
    from petsc_ssr.cli.commands import mesh as mesh_command

    def _raise_missing_extra(_path: Path):
        raise mesh_command.MeshInspectionError("Reading .msh files requires the 'meshio' package.")

    monkeypatch.setattr(mesh_command, "mesh_report_payload", _raise_missing_extra)

    status = cli_main.main(["mesh-only", "case.toml"])

    assert status == 2
    out = capsys.readouterr().out
    assert '"ok": false' in out
    assert "meshio" in out
    assert "Traceback" not in out


def test_doctor_command_payload_is_importable_without_cli_dispatch() -> None:
    payload = doctor_payload(ROOT)

    assert "pmg-deflated-baseline" in payload["solver_profiles"]
    assert "gamg-p1-baseline" in payload["solver_profiles"]
    assert "direct-debug" in payload["solver_profiles"]
    assert "indirect-classic" in payload["continuation_profiles"]
    assert "local-32-smoke.toml" in payload["suites"]
