from __future__ import annotations

import sys
from pathlib import Path
import subprocess

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "benchmarks" / "tools"
CASE_ROOT = ROOT / "benchmarks" / "cases"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import notebook_support as nb  # noqa: E402


def _fake_artifacts(tmp_path: Path, run_info: dict[str, object], npz: dict[str, np.ndarray] | None = None) -> nb.RunArtifacts:
    data_dir = tmp_path / "data"
    plots_dir = tmp_path / "plots"
    exports_dir = tmp_path / "exports"
    data_dir.mkdir()
    plots_dir.mkdir()
    exports_dir.mkdir()
    return nb.RunArtifacts(
        out_dir=tmp_path,
        data_dir=data_dir,
        plots_dir=plots_dir,
        exports_dir=exports_dir,
        run_info=run_info,
        history={},
        npz={} if npz is None else npz,
        progress_events=[],
        vtu_path=exports_dir / "final_solution.vtu",
    )


def test_modern_notebook_smoke_profile_keeps_small_cases_mathematical() -> None:
    case_toml = CASE_ROOT / "2d-homogeneous-ssr" / "case.toml"
    sections = nb.load_case_sections(case_toml)

    smoke = nb._profile_sections(case_toml, sections, "smoke")

    assert smoke["mesh"]["element"] == "P2"
    assert smoke["linear"]["profile"] == "pmg-deflated-baseline"
    assert smoke["continuation"]["step_max"] == 2
    assert smoke["output"]["preset"] == "smoke"


def test_modern_notebook_smoke_profile_downshifts_known_heavy_cases() -> None:
    case_toml = CASE_ROOT / "2d-kozinec-ll" / "case.toml"
    sections = nb.load_case_sections(case_toml)

    smoke = nb._profile_sections(case_toml, sections, "smoke")

    assert smoke["mesh"]["element"] == "P1"
    assert smoke["linear"]["profile"] == "gamg-p1-baseline"
    assert smoke["newton"]["profile"] == "limit-load-regularized-it100"
    assert smoke["continuation"]["step_max"] == 2
    assert smoke["output"]["preset"] == "smoke"


def test_coupled_seepage_smoke_profile_uses_explicit_coordinate_bc_debug_override() -> None:
    case_toml = CASE_ROOT / "3d-heterogeneous-seepage-ssr-comsol" / "case.toml"

    assert nb._profile_solver_args(case_toml, "smoke") == ["--write-coordinate-bc-table"]
    assert nb._profile_solver_args(case_toml, "full") == []


def test_modern_p4_case_uses_explicit_boundary_visualisation_path() -> None:
    case_toml = CASE_ROOT / "3d-heterogeneous-ssr-p4" / "case.toml"

    assert nb._elem_type(case_toml) == "P4"
    assert nb._use_explicit_surface_builder(case_toml) is True
    assert nb.load_case_metadata(case_toml)["mpi_ranks"] == 32
    assert nb.load_case_metadata(case_toml)["surface_decimate_reduction"] == 0.0


def test_p4_surface_split_uses_all_boundary_face_nodes() -> None:
    split = nb._surface_display_local_split(15)

    assert split.shape == (16, 3)
    assert set(split.reshape(-1).tolist()) == set(range(15))

    surf = np.arange(15, dtype=np.int64)[:, None]
    triangles, face_ids = nb._build_plotting_mesh_with_face_ids(surf)

    np.testing.assert_array_equal(triangles, split)
    np.testing.assert_array_equal(face_ids, np.zeros(16, dtype=np.int64))


def test_field_surface_decimation_is_explicit_only() -> None:
    assert nb._field_surface_decimate_reduction(None) == 0.0
    assert nb._field_surface_decimate_reduction(0.75) == 0.75
    assert nb._field_surface_decimate_reduction(4.0) == 0.99


def test_high_order_norm_field_sanitizers() -> None:
    class Dataset:
        def __init__(self) -> None:
            self.point_data = {
                "displacement": np.asarray([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]),
                "displacement_magnitude": np.asarray([-10.0, 5.0]),
                "deviatoric_strain": np.asarray([-1.0, 2.5]),
            }
            self.cell_data = {"deviatoric_strain": np.asarray([-3.0, 4.0])}

    dataset = Dataset()

    nb._sync_displacement_magnitude(dataset)
    nb._sanitize_nonnegative_point_array(dataset, "deviatoric_strain")
    nb._sanitize_nonnegative_cell_array(dataset, "deviatoric_strain")

    np.testing.assert_allclose(dataset.point_data["displacement_magnitude"], [5.0, 0.0])
    np.testing.assert_allclose(dataset.point_data["deviatoric_strain"], [0.0, 2.5])
    np.testing.assert_allclose(dataset.cell_data["deviatoric_strain"], [0.0, 4.0])


def test_default_surface_warp_scale_caps_single_edge_spikes() -> None:
    class Surface:
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([3, 0, 1, 2], dtype=np.int64)
        point_data = {
            "displacement": np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        }

    assert nb._cap_display_warp_scale(Surface(), 1.0, max_edge_stretch=2.0) == pytest.approx(0.1)
    assert nb._surface_warp_scale(np.eye(3), np.ones((3, 3)), Surface(), override=0.75) == 0.75


def test_generated_artifact_config_reuses_owner_notebook_sidecar(tmp_path: Path) -> None:
    case_dir = tmp_path / "benchmarks" / "cases" / "example-p4"
    artifact_dir = case_dir / "artifacts" / "simulation"
    artifact_dir.mkdir(parents=True)
    (case_dir / "case.toml").write_text(
        """
        [case]
        id = "example-p4"
        [mesh]
        asset = "demo"
        variant = "default"
        element = "P4"
        """,
        encoding="utf-8",
    )
    (case_dir / "notebook.toml").write_text(
        """
        [notebook]
        family = "3d_continuation"
        jupyter_backend = "client"
        mpi_ranks = 12
        """,
        encoding="utf-8",
    )
    generated = artifact_dir / "generated_case.toml"
    generated.write_text(
        """
        [case]
        id = "example-p4"
        [mesh]
        asset = "demo"
        variant = "default"
        element = "P1"
        """,
        encoding="utf-8",
    )

    assert nb._elem_type(generated) == "P1"
    assert nb.load_case_metadata(generated)["jupyter_backend"] == "client"
    assert nb.load_case_metadata(generated)["mpi_ranks"] == 12


def test_notebook_runner_auto_builds_missing_native_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    probes = [False, True]
    build_calls: list[list[str]] = []

    def fake_probe(python_executable: Path, env: dict[str, str]) -> tuple[bool, str]:
        return probes.pop(0), ""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        cmd = list(args[0])  # type: ignore[index]
        build_calls.append([str(item) for item in cmd])
        return subprocess.CompletedProcess(cmd, 0, stdout="built ok")

    monkeypatch.setattr(nb, "_python_can_import_native_extension", fake_probe)
    monkeypatch.setattr(nb.subprocess, "run", fake_run)

    nb._ensure_native_extension_available(Path("/venv/bin/python"), {"PYTHONPATH": "src"})

    assert build_calls == [["/venv/bin/python", "setup.py", "build_ext", "--inplace"]]


def test_notebook_runner_reports_native_build_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_probe(python_executable: Path, env: dict[str, str]) -> tuple[bool, str]:
        return False, ""

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        cmd = list(args[0])  # type: ignore[index]
        return subprocess.CompletedProcess(cmd, 1, stdout="compiler exploded")

    monkeypatch.setattr(nb, "_python_can_import_native_extension", fake_probe)
    monkeypatch.setattr(nb.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="build_ext --inplace") as excinfo:
        nb._ensure_native_extension_available(Path("/venv/bin/python"), {"PYTHONPATH": "src"})

    assert "compiler exploded" in str(excinfo.value)


def test_parallel_failure_message_includes_solver_tail() -> None:
    message = nb._parallel_failure_message(1, [f"line {index:02d}\n" for index in range(45)])

    assert "Parallel solve failed with exit code 1" in message
    assert "line 05" in message
    assert "line 44" in message
    assert "line 04" not in message


def test_timing_breakdown_groups_are_disjoint_and_sum_to_continuation_wall(tmp_path: Path) -> None:
    artifacts = _fake_artifacts(
        tmp_path,
        {
            "c_hotpath_summary": {
                "continuation_wall_time": 100.0,
                "assembly_time": 10.0,
                "tangent_assembly_time": 4.0,
                "residual_assembly_time": 3.0,
                "operator_build_time": 2.0,
                "linear_solve_time": 60.0,
                "ksp_setup_time": 8.0,
                "deflation_base_pc_apply_time": 20.0,
                "pmg_operator_update_time": 3.0,
                "pmg_fine_smooth_time": 5.0,
                "pmg_p2_smooth_time": 4.0,
                "pmg_transfer_time": 3.0,
                "pmg_coarse_solve_time": 2.0,
                "pmg_residual_time": 1.0,
                "deflation_orthogonalization_time": 6.0,
                "deflation_projector_time": 4.0,
                "deflation_coarse_initial_time": 2.0,
                "linear_operator_matvec_time": 5.0,
                "krylov_orthogonalization_time": 4.0,
                "krylov_least_squares_time": 1.0,
                "krylov_solution_update_time": 1.0,
                "line_search_time": 10.0,
            }
        },
    )

    groups, continuation_wall = nb._timing_breakdown_groups(artifacts)
    series = nb._timing_breakdown_series(artifacts)

    assert continuation_wall == pytest.approx(100.0)
    assert series["Assembly"] == pytest.approx(10.0)
    assert series["Preconditioning"] == pytest.approx(28.0)
    assert series["Deflation"] == pytest.approx(12.0)
    assert series["Linear solver"] == pytest.approx(20.0)
    assert series["Line search"] == pytest.approx(10.0)
    assert series["Other"] == pytest.approx(20.0)
    assert sum(series.values()) == pytest.approx(continuation_wall)
    assert groups["Preconditioning"]["PMG operator update"] == pytest.approx(3.0)
    assert groups["Preconditioning"]["PC apply other"] == pytest.approx(5.0)
    assert groups["Linear solver"]["Linear other"] == pytest.approx(9.0)


def test_timing_breakdown_plot_autoscales_to_tallest_column(tmp_path: Path) -> None:
    artifacts = _fake_artifacts(
        tmp_path,
        {
            "c_hotpath_summary": {
                "continuation_wall_time": 100.0,
                "assembly_time": 10.0,
                "tangent_assembly_time": 10.0,
                "linear_solve_time": 70.0,
                "deflation_base_pc_apply_time": 30.0,
                "deflation_orthogonalization_time": 20.0,
                "linear_operator_matvec_time": 20.0,
                "line_search_time": 20.0,
            }
        },
    )

    fig = nb.plot_timing_breakdown(artifacts)
    try:
        assert fig.axes[0].get_ylim()[1] == pytest.approx(34.5)
    finally:
        nb.plt.close(fig)


def test_timing_breakdown_uses_nested_run_info_fallback(tmp_path: Path) -> None:
    artifacts = _fake_artifacts(
        tmp_path,
        {
            "timings": {
                "continuation_total_wall_time": 13.0,
                "assembly": {
                    "tangent_assembly_time": 2.0,
                    "residual_assembly_time": 1.0,
                    "operator_build_time": 1.0,
                },
                "preconditioning": {
                    "ksp_setup_time": 2.0,
                    "pmg_operator_update_time": 0.5,
                    "deflation_base_pc_apply_time": 3.0,
                    "pmg_fine_smooth_time": 1.0,
                    "pmg_p2_smooth_time": 0.5,
                    "pmg_transfer_time": 0.25,
                    "pmg_coarse_solve_time": 0.25,
                },
                "deflation": {
                    "deflation_orthogonalization_time": 1.0,
                    "deflation_projector_time": 0.5,
                    "deflation_coarse_initial_time": 0.5,
                },
                "linear": {
                    "attempt_linear_solve_time_total": 8.0,
                    "linear_operator_matvec_time": 1.0,
                },
                "line_search": {
                    "line_search_time": 1.0,
                },
            }
        },
    )

    series = nb._timing_breakdown_series(artifacts)

    assert series["Assembly"] == pytest.approx(4.0)
    assert series["Preconditioning"] == pytest.approx(5.0)
    assert series["Deflation"] == pytest.approx(2.0)
    assert series["Linear solver"] == pytest.approx(1.0)
    assert series["Line search"] == pytest.approx(1.0)


def test_iteration_history_plot_uses_cumulative_linear_iterations(tmp_path: Path) -> None:
    artifacts = _fake_artifacts(
        tmp_path,
        {},
        {
            "stats_step_index": np.asarray([1, 2, 3], dtype=np.int64),
            "stats_step_newton_iterations": np.asarray([3, 4, 2], dtype=np.int64),
            "stats_step_linear_iterations": np.asarray([10, 20, 5], dtype=np.int64),
        },
    )

    fig = nb.plot_iteration_history(artifacts)
    try:
        newton_line = fig.axes[0].lines[0]
        linear_line = fig.axes[1].lines[0]
        np.testing.assert_array_equal(newton_line.get_ydata(), [3, 4, 2])
        np.testing.assert_array_equal(linear_line.get_ydata(), [10, 30, 35])
    finally:
        nb.plt.close(fig)
