from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pytest

from slope_stability.cli.run_case_from_config import _case_runner_kwargs
from slope_stability.core.run_config import load_run_case_config


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT / "benchmarks"
EXPECTED_FAMILIES = {
    "2d_continuation",
    "2d_seepage",
    "2d_seepage_continuation",
    "3d_continuation",
    "3d_seepage",
    "3d_seepage_continuation",
}


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _support():
    return _load_module(BENCHMARKS_DIR / "notebook_support.py", "benchmarks_notebook_support")


def _generator():
    return _load_module(BENCHMARKS_DIR / "generate_benchmark_notebooks.py", "generate_benchmark_notebooks")


def _case_tomls() -> list[Path]:
    return sorted(BENCHMARKS_DIR.glob("*/case.toml"))


def _notebook_sources(path: Path) -> str:
    notebook = nbformat.read(path, as_version=4)
    return "\n".join("".join(cell.get("source", "")) for cell in notebook.cells)


def test_every_benchmark_case_has_metadata_and_notebook_section() -> None:
    for case_toml in _case_tomls():
        raw = tomllib.loads(case_toml.read_text(encoding="utf-8"))
        benchmark = dict(raw.get("benchmark", {}))
        notebook = dict(raw.get("notebook", {}))

        assert benchmark.get("title")
        assert benchmark.get("matlab_script")
        assert benchmark.get("comparison_kind")
        assert "suite" in benchmark
        assert notebook.get("family") in EXPECTED_FAMILIES


def test_default_3d_cases_have_explicit_export_blocks() -> None:
    for name in ("3d_hetero_ssr_default", "3d_homo_ssr_default"):
        raw = tomllib.loads((BENCHMARKS_DIR / name / "case.toml").read_text(encoding="utf-8"))
        export = dict(raw.get("export", {}))
        assert export.get("write_custom_debug_bundle") is True
        assert export.get("write_history_json") is True
        assert export.get("write_solution_vtu") is True


def test_every_benchmark_has_valid_generated_notebook() -> None:
    for case_toml in _case_tomls():
        for notebook_name in ("simulation.ipynb", "visualisation.ipynb"):
            notebook_path = case_toml.parent / notebook_name
            assert notebook_path.exists(), notebook_path
            notebook = nbformat.read(notebook_path, as_version=4)
            nbformat.validate(notebook)
        assert not (case_toml.parent / "pyvista_workflow.ipynb").exists()


def test_no_committed_generated_notebook_case_toml_remains() -> None:
    assert not (BENCHMARKS_DIR / "3d_hetero_ssr_default" / "notebook_case.generated.toml").exists()


def test_generator_builds_valid_notebooks_for_all_cases() -> None:
    module = _generator()
    for case_toml in _case_tomls():
        nbformat.validate(module.build_simulation_notebook(case_toml))
        nbformat.validate(module.build_visualisation_notebook(case_toml))


def test_simulation_notebooks_default_to_benchmark_profile() -> None:
    for case_toml in _case_tomls():
        source = _notebook_sources(case_toml.parent / "simulation.ipynb")
        assert 'RUN_MODE = "run"' in source
        assert 'EXECUTION_PROFILE = "benchmark"' in source


def test_support_module_imports_without_viz_extras() -> None:
    module = _support()
    status = module.viz_support_status()

    assert set(status) == {"pyvista", "ipywidgets", "trame"}
    message = module.viz_support_message()
    assert "Interactive 3D" in message


def test_load_case_sections_resolves_relative_paths_for_generated_configs() -> None:
    module = _support()
    sections = module.load_case_sections(BENCHMARKS_DIR / "3d_hetero_ssr_default" / "case.toml")

    assert Path(sections["problem"]["mesh_path"]).is_absolute()


def test_smoke_profile_uses_lightweight_textmesh_and_ll_overrides() -> None:
    module = _support()

    koz_sections = module._profile_sections(  # noqa: SLF001
        BENCHMARKS_DIR / "run_2d_kozinec_ssr" / "case.toml",
        module.load_case_sections(BENCHMARKS_DIR / "run_2d_kozinec_ssr" / "case.toml"),
        "smoke",
    )
    assert koz_sections["problem"]["elem_type"] == "P1"
    assert koz_sections["execution"]["mpi_distribute_by_nodes"] is False
    assert koz_sections["execution"]["constitutive_mode"] == "global"
    assert koz_sections["linear_solver"]["solver_type"] == "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE"

    ll_sections = module._profile_sections(  # noqa: SLF001
        BENCHMARKS_DIR / "run_3d_hetero_ll" / "case.toml",
        module.load_case_sections(BENCHMARKS_DIR / "run_3d_hetero_ll" / "case.toml"),
        "smoke",
    )
    assert ll_sections["execution"]["mpi_distribute_by_nodes"] is False
    assert ll_sections["execution"]["constitutive_mode"] == "global"
    assert ll_sections["linear_solver"]["solver_type"] == "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE"


def test_kozinec_ssr_defaults_to_p2_with_petsc_hypre() -> None:
    module = _support()

    sections = module.load_case_sections(BENCHMARKS_DIR / "run_2d_kozinec_ssr" / "case.toml")

    assert sections["problem"]["elem_type"] == "P2"
    assert sections["linear_solver"]["solver_type"] == "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE"


def test_3d_hetero_ssr_default_uses_p4_pmg_defaults() -> None:
    module = _support()

    sections = module.load_case_sections(BENCHMARKS_DIR / "3d_hetero_ssr_default" / "case.toml")
    metadata = module.load_case_metadata(BENCHMARKS_DIR / "3d_hetero_ssr_default" / "case.toml")

    assert sections["problem"]["elem_type"] == "P4"
    assert sections["continuation"]["omega_max"] == 6.7e6
    assert sections["continuation"]["init_newton_stopping_criterion"] == "relative_correction"
    assert sections["continuation"]["init_newton_stopping_tol"] == 1e-3
    assert sections["newton"]["stopping_criterion"] == "absolute_delta_lambda"
    assert sections["newton"]["stopping_tol"] == 1e-4
    assert sections["linear_solver"]["pc_backend"] == "pmg_shell"
    assert metadata["jupyter_backend"] == "client"
    assert metadata["nonlinear_surface_subdivision"] == 0
    assert metadata["surface_decimate_reduction"] == 0.75
    assert metadata["boundary_edge_overlay"] is False


def test_extract_surface_for_display_uses_high_order_subdivision() -> None:
    module = _support()
    calls: dict[str, object] = {}

    class DummyDataSet:
        def extract_surface(self, **kwargs):
            calls.update(kwargs)
            return "surface"

    surface = module._extract_surface_for_display(DummyDataSet())  # noqa: SLF001

    assert surface == "surface"
    assert calls["nonlinear_subdivision"] == 4
    assert calls["pass_pointid"] is True
    assert calls["pass_cellid"] is True


def test_display_nonlinear_surface_subdivision_reads_notebook_setting() -> None:
    module = _support()

    subdivision = module._display_nonlinear_surface_subdivision(  # noqa: SLF001
        BENCHMARKS_DIR / "3d_hetero_ssr_default" / "case.toml"
    )

    assert subdivision == 0


def test_display_nonlinear_surface_subdivision_honors_override() -> None:
    module = _support()

    subdivision = module._display_nonlinear_surface_subdivision(  # noqa: SLF001
        BENCHMARKS_DIR / "3d_hetero_ssr_default" / "case.toml",
        override=3,
    )

    assert subdivision == 3


def test_display_surface_decimate_reduction_reads_notebook_setting_and_honors_override() -> None:
    module = _support()
    case_toml = BENCHMARKS_DIR / "3d_hetero_ssr_default" / "case.toml"

    assert module._display_surface_decimate_reduction(case_toml) == 0.75  # noqa: SLF001
    assert module._display_surface_decimate_reduction(case_toml, override=0.5) == 0.5  # noqa: SLF001


def test_display_boundary_edge_overlay_reads_notebook_setting_and_honors_override() -> None:
    module = _support()
    case_toml = BENCHMARKS_DIR / "3d_hetero_ssr_default" / "case.toml"

    assert module._display_boundary_edge_overlay(case_toml) is False  # noqa: SLF001
    assert module._display_boundary_edge_overlay(case_toml, override=True) is True  # noqa: SLF001


def test_display_jupyter_backend_reads_notebook_setting_and_honors_override() -> None:
    module = _support()
    case_toml = BENCHMARKS_DIR / "3d_hetero_ssr_default" / "case.toml"

    assert module._display_jupyter_backend(case_toml) == "client"  # noqa: SLF001
    assert module._display_jupyter_backend(case_toml, override="static") == "static"  # noqa: SLF001


def test_3d_hetero_ssr_default_visualisation_includes_deviatoric_slices() -> None:
    source = _notebook_sources(BENCHMARKS_DIR / "3d_hetero_ssr_default" / "visualisation.ipynb")

    assert "show_3d_deviatoric_slices" in source
    assert 'JUPYTER_BACKEND_OVERRIDE = None' in source
    assert 'SURFACE_SUBDIVISION_OVERRIDE = None' in source
    assert 'SURFACE_DECIMATE_REDUCTION_OVERRIDE = None' in source
    assert 'BOUNDARY_EDGE_OVERLAY_OVERRIDE = None' in source
    assert "surface_subdivision=SURFACE_SUBDIVISION_OVERRIDE" in source
    assert "surface_decimate_reduction=SURFACE_DECIMATE_REDUCTION_OVERRIDE" in source
    assert "boundary_edge_overlay=BOUNDARY_EDGE_OVERLAY_OVERRIDE" in source
    assert "jupyter_backend=JUPYTER_BACKEND_OVERRIDE" in source
    assert "slice_planes_y=[35.0]" in source
    assert "slice_planes_z=[43.30127019, 64.95190529]" in source


def test_2d_seepage_continuation_cases_default_to_single_rank() -> None:
    for name in ("run_2d_luzec_ssr", "run_2d_franz_dam_ssr"):
        raw = tomllib.loads((BENCHMARKS_DIR / name / "case.toml").read_text(encoding="utf-8"))

        assert raw["benchmark"]["mpi_ranks"] == 1


def test_sloan_seepage_summary_omits_continuation_nan_metrics() -> None:
    module = _support()
    artifacts = module.load_run_artifacts(BENCHMARKS_DIR / "run_2D_sloan2013_seepage_capture" / "artifacts" / "simulation")

    summary = module._run_completion_summary(artifacts)  # noqa: SLF001

    assert summary["step_count"] == 0
    assert summary["lambda_last"] is None
    assert summary["omega_last"] is None
    assert module._format_optional_metric(summary["lambda_last"]) == "n/a"  # noqa: SLF001
    assert module._format_optional_metric(summary["omega_last"]) == "n/a"  # noqa: SLF001


def test_sloan_seepage_case_raises_nonlinear_iteration_cap() -> None:
    cfg = load_run_case_config(BENCHMARKS_DIR / "run_2D_sloan2013_seepage_capture" / "case.toml")

    _runner, kwargs = _case_runner_kwargs(cfg)  # noqa: SLF001

    assert cfg.seepage.nonlinear_max_iter == 100
    assert kwargs["nonlinear_max_iter"] == 100


def test_reuse_prefers_matching_generated_case_config(tmp_path: Path) -> None:
    module = _support()
    case_toml = BENCHMARKS_DIR / "run_2d_kozinec_ssr" / "case.toml"
    run_label = "reuse_test"
    out_dir = tmp_path / "artifacts" / run_label
    data_dir = out_dir / "data"
    exports_dir = out_dir / "exports"
    data_dir.mkdir(parents=True)
    exports_dir.mkdir(parents=True)
    (data_dir / "run_info.json").write_text(json.dumps({"run_info": {}}), encoding="utf-8")
    np.savez(data_dir / "petsc_run.npz", lambda_hist=np.array([0.0]), omega_hist=np.array([0.0]))
    (exports_dir / "final_solution.vtu").write_text("<VTKFile/>", encoding="utf-8")
    generated = out_dir / "generated_case.toml"
    generated.write_text("[problem]\nelem_type = \"P1\"\n", encoding="utf-8")

    execution = module.ensure_notebook_artifacts(
        case_toml=case_toml,
        sections=module.load_case_sections(case_toml),
        materials=module.load_case_materials(case_toml),
        run_label=run_label,
        run_mode="auto",
        execution_profile="smoke",
        root=tmp_path,
    )

    assert execution.reused_existing is True
    assert execution.active_config == generated


def test_matlab_warp_scale_matches_reference_formula() -> None:
    module = _support()
    coord = np.array([[0.0, 10.0, 20.0], [0.0, 5.0, 10.0]], dtype=np.float64)
    displacement = np.array([[0.0, 2.0, 4.0], [0.0, 1.0, 2.0]], dtype=np.float64)

    scale = module.matlab_warp_scale(coord, displacement)

    assert np.isclose(scale, 0.05 * 20.0 / 4.0)


def test_material_palette_lookup_returns_exact_palette_entries() -> None:
    module = _support()
    luzec = module.get_material_palette("luzec")

    assert luzec[0] == (0.0, 0.0, 0.0)
    assert luzec[4] == (1.0, 1.0, 0.0)
    assert luzec[6] == (1.0, 0.0, 1.0)


def test_compute_element_deviatoric_strain_2d_and_3d() -> None:
    module = _support()

    coord_2d = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    elem_2d = np.array([[0], [1], [2]], dtype=np.int64)
    u_2d = np.array([[0.0, 0.2, 0.0], [0.0, 0.0, 0.1]], dtype=np.float64)
    dev_2d = module.compute_element_deviatoric_strain(coord_2d, elem_2d, "P1", u_2d, dim=2)

    coord_3d = np.array(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    elem_3d = np.array([[0], [1], [2], [3]], dtype=np.int64)
    u_3d = np.array(
        [[0.0, 0.2, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0], [0.0, 0.0, 0.0, 0.15]],
        dtype=np.float64,
    )
    dev_3d = module.compute_element_deviatoric_strain(coord_3d, elem_3d, "P1", u_3d, dim=3)

    assert dev_2d.shape == (1,)
    assert dev_3d.shape == (1,)
    assert float(dev_2d[0]) > 0.0
    assert float(dev_3d[0]) > 0.0


def test_load_case_mesh_uses_artifact_mpi_size(monkeypatch) -> None:
    module = _support()
    seen: dict[str, object] = {}

    def fake_rebuild_case_mesh(cfg, *, mpi_size: int = 1):
        seen["cfg"] = cfg
        seen["mpi_size"] = mpi_size
        return "mesh"

    sentinel_cfg = object()
    monkeypatch.setattr(module, "_load_runtime_config", lambda case_toml: sentinel_cfg)

    import slope_stability.postprocess as postprocess

    monkeypatch.setattr(postprocess, "rebuild_case_mesh", fake_rebuild_case_mesh)
    artifacts = SimpleNamespace(run_info={"run_info": {"mpi_size": 8}})

    result = module._load_case_mesh(BENCHMARKS_DIR / "run_3D_hetero_seepage_SSR_comsol_capture" / "case.toml", artifacts=artifacts)  # noqa: SLF001

    assert result == "mesh"
    assert seen["cfg"] is sentinel_cfg
    assert seen["mpi_size"] == 8


def test_pore_pressure_field_reorders_old_comsol_ssr_artifacts(monkeypatch) -> None:
    module = _support()
    cfg = SimpleNamespace(
        problem=SimpleNamespace(case="3d_hetero_seepage_ssr_comsol"),
        execution=SimpleNamespace(node_ordering="block_metis"),
    )
    artifacts = SimpleNamespace(
        npz={"seepage_pw": np.array([10.0, 20.0, 30.0], dtype=np.float64)},
        run_info={"run_info": {"mpi_size": 8}},
    )

    monkeypatch.setattr(module, "_load_runtime_config", lambda case_toml: cfg)
    monkeypatch.setattr(module, "_comsol_ssr_node_permutation", lambda case_toml, artifacts: np.array([1, 0, 2], dtype=np.int64))

    values = module._pore_pressure_field(artifacts, BENCHMARKS_DIR / "run_3D_hetero_seepage_SSR_comsol_capture" / "case.toml")  # noqa: SLF001

    np.testing.assert_allclose(values, np.array([20.0, 10.0, 30.0]))


def test_2d_artifact_plots_use_vtu_topology_for_reused_generated_config() -> None:
    module = _support()
    out_dir = BENCHMARKS_DIR / "run_2d_franz_dam_ssr" / "artifacts" / "simulation"
    artifacts = module.load_run_artifacts(out_dir)
    active_config = out_dir / "generated_case.toml"

    plotters = (
        module.plot_2d_pore_pressure,
        module.plot_2d_saturation,
        module.plot_2d_displacement,
        module.plot_2d_deviatoric_strain,
    )
    for plot in plotters:
        fig = plot(artifacts, active_config)
        assert fig is not None
        plt.close(fig)


def test_saturation_field_falls_back_to_npz_when_vtu_cell_data_is_missing() -> None:
    module = _support()
    artifacts = module.load_run_artifacts(BENCHMARKS_DIR / "run_2D_sloan2013_seepage_capture" / "artifacts" / "simulation")
    vtu = module.load_vtu(BENCHMARKS_DIR / "run_2D_sloan2013_seepage_capture" / "artifacts" / "simulation" / "exports" / "final_solution.vtu")

    saturation = module._saturation_field(  # noqa: SLF001
        artifacts,
        vtu=module.VtuData(
            points=vtu.points,
            cell_blocks=vtu.cell_blocks,
            point_data=vtu.point_data,
            cell_data={},
        ),
        n_cells=artifacts.npz["mater_sat"].size,
    )

    np.testing.assert_allclose(saturation, np.asarray(artifacts.npz["mater_sat"], dtype=np.float64))


def test_2d_vtu_triangle6_subdivision_preserves_positive_area() -> None:
    module = _support()
    out_dir = BENCHMARKS_DIR / "run_2d_franz_dam_ssr" / "artifacts" / "simulation"
    vtu = module.load_vtu(out_dir / "exports" / "final_solution.vtu")

    coord, triangles, _parents, _elem, _elem_type = module._vtu_linear_triangles_2d(vtu)  # noqa: SLF001
    pts = coord.T
    areas = 0.5 * (
        (pts[triangles[:, 1], 0] - pts[triangles[:, 0], 0]) * (pts[triangles[:, 2], 1] - pts[triangles[:, 0], 1])
        - (pts[triangles[:, 1], 1] - pts[triangles[:, 0], 1]) * (pts[triangles[:, 2], 0] - pts[triangles[:, 0], 0])
    )

    assert np.all(areas > 0.0)


def test_show_3d_deviatoric_surface_view_uses_surface_cell_scalars(monkeypatch) -> None:
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    module = _support()
    artifacts = module.load_run_artifacts(BENCHMARKS_DIR / "run_3D_hetero_seepage_SSR_comsol_capture" / "artifacts" / "simulation")
    case_toml = BENCHMARKS_DIR / "run_3D_hetero_seepage_SSR_comsol_capture" / "case.toml"
    captured: dict[str, object] = {}
    original_add_mesh = pv.Plotter.add_mesh

    def wrapped_add_mesh(self, mesh, *args, **kwargs):
        captured["mesh"] = mesh
        captured["kwargs"] = kwargs
        return original_add_mesh(self, mesh, *args, **kwargs)

    monkeypatch.setattr(module, "_module_available", lambda name: name == "pyvista")
    monkeypatch.setattr(module, "_import_pyvista", lambda: pv)
    monkeypatch.setattr(module, "_new_plotter", lambda pv_mod, title: pv_mod.Plotter(off_screen=True))
    monkeypatch.setattr(module, "_show_plotter", lambda plotter, *args, **kwargs: plotter.close() or "shown")
    monkeypatch.setattr(pv.Plotter, "add_mesh", wrapped_add_mesh)

    result = module.show_3d_deviatoric_surface_view(artifacts, case_toml)

    assert result == "shown"
    assert captured["kwargs"]["scalars"] == "deviatoric_strain"
    assert captured["kwargs"]["preference"] == "cell"
    assert captured["kwargs"]["lighting"] is False
    assert "deviatoric_strain" in captured["mesh"].cell_data
    assert "deviatoric_strain" not in captured["mesh"].point_data


def test_show_3d_deviatoric_surface_view_can_overlay_boundary_edges(monkeypatch) -> None:
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    module = _support()
    artifacts = module.load_run_artifacts(BENCHMARKS_DIR / "run_3D_hetero_seepage_SSR_comsol_capture" / "artifacts" / "simulation")
    case_toml = BENCHMARKS_DIR / "run_3D_hetero_seepage_SSR_comsol_capture" / "case.toml"
    captured: list[dict[str, object]] = []
    original_add_mesh = pv.Plotter.add_mesh

    def wrapped_add_mesh(self, mesh, *args, **kwargs):
        captured.append({"mesh": mesh, "kwargs": kwargs})
        return original_add_mesh(self, mesh, *args, **kwargs)

    monkeypatch.setattr(module, "_module_available", lambda name: name == "pyvista")
    monkeypatch.setattr(module, "_import_pyvista", lambda: pv)
    monkeypatch.setattr(module, "_new_plotter", lambda pv_mod, title: pv_mod.Plotter(off_screen=True))
    monkeypatch.setattr(module, "_show_plotter", lambda plotter, *args, **kwargs: plotter.close() or "shown")
    monkeypatch.setattr(pv.Plotter, "add_mesh", wrapped_add_mesh)

    result = module.show_3d_deviatoric_surface_view(artifacts, case_toml, boundary_edge_overlay=True)

    assert result == "shown"
    assert len(captured) == 2
    assert captured[0]["kwargs"]["scalars"] == "deviatoric_strain"
    assert "scalars" not in captured[1]["kwargs"]
    assert captured[1]["kwargs"]["line_width"] == pytest.approx(1.2)


def test_surface_parent_elements_and_plotting_face_ids_for_p2_faces() -> None:
    module = _support()
    elem = np.array(
        [
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
            [9],
        ],
        dtype=np.int64,
    )
    surf = np.array(
        [
            [0],
            [1],
            [2],
            [4],
            [5],
            [6],
        ],
        dtype=np.int64,
    )

    triangles, face_ids = module._build_plotting_mesh_with_face_ids(surf)
    parent = module._surface_parent_elements(elem, surf)

    assert triangles.shape == (4, 3)
    assert np.array_equal(face_ids, np.array([0, 0, 0, 0], dtype=np.int64))
    assert np.array_equal(parent, np.array([0], dtype=np.int64))


def test_show_3d_deviatoric_slices_uses_single_scalar_bar(monkeypatch) -> None:
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    module = _support()
    artifacts = module.load_run_artifacts(BENCHMARKS_DIR / "run_3D_hetero_seepage_SSR_comsol_capture" / "artifacts" / "simulation")
    case_toml = BENCHMARKS_DIR / "run_3D_hetero_seepage_SSR_comsol_capture" / "case.toml"

    monkeypatch.setattr(module, "_module_available", lambda name: name == "pyvista")
    monkeypatch.setattr(module, "_import_pyvista", lambda: pv)
    monkeypatch.setattr(module, "_new_plotter", lambda pv_mod, title: pv_mod.Plotter(off_screen=True))

    def fake_show(plotter, *args, **kwargs):
        count = len(plotter.scalar_bars)
        plotter.close()
        return count

    monkeypatch.setattr(module, "_show_plotter", fake_show)

    count = module.show_3d_deviatoric_slices(
        artifacts,
        case_toml,
        slice_planes_y=[35.0],
        slice_planes_z=[1.0e-16, 21.6506],
    )

    assert count == 1


def test_refine_slice_for_display_subdivides_p4_slice_and_resamples() -> None:
    pv = pytest.importorskip("pyvista")
    module = _support()
    artifacts = module.load_run_artifacts(BENCHMARKS_DIR / "3d_hetero_ssr_default" / "artifacts" / "simulation")
    case_toml = BENCHMARKS_DIR / "3d_hetero_ssr_default" / "case.toml"

    grid = pv.read(artifacts.vtu_path)
    slc = grid.slice(
        normal=(0.0, 0.0, 1.0),
        origin=(grid.center[0], grid.center[1], 43.30127019),
        generate_triangles=True,
    )
    refined = module._refine_slice_for_display(slc, grid, case_toml=case_toml)  # noqa: SLF001

    assert refined.n_cells > slc.n_cells
    assert "deviatoric_strain" in refined.point_data


def test_show_3d_saturation_view_uses_region_surfaces(monkeypatch) -> None:
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    module = _support()
    artifacts = module.load_run_artifacts(BENCHMARKS_DIR / "run_3D_hetero_seepage_capture" / "artifacts" / "simulation")
    case_toml = BENCHMARKS_DIR / "run_3D_hetero_seepage_capture" / "case.toml"
    captured: list[dict[str, object]] = []
    original_add_mesh = pv.Plotter.add_mesh

    def wrapped_add_mesh(self, mesh, *args, **kwargs):
        captured.append({"mesh": mesh, "kwargs": kwargs})
        return original_add_mesh(self, mesh, *args, **kwargs)

    monkeypatch.setattr(module, "_module_available", lambda name: name == "pyvista")
    monkeypatch.setattr(module, "_import_pyvista", lambda: pv)
    monkeypatch.setattr(module, "_new_plotter", lambda pv_mod, title: pv_mod.Plotter(off_screen=True))
    monkeypatch.setattr(module, "_show_plotter", lambda plotter, *args, **kwargs: plotter.close() or "shown")
    monkeypatch.setattr(pv.Plotter, "add_mesh", wrapped_add_mesh)

    result = module.show_3d_saturation_view(artifacts, case_toml)

    assert result == "shown"
    assert len(captured) == 2
    assert all(entry["kwargs"]["show_edges"] is True for entry in captured)
    assert all(entry["kwargs"]["lighting"] is False for entry in captured)
    assert {tuple(entry["kwargs"]["color"]) for entry in captured} == {(1.0, 1.0, 0.0), (0.0, 0.0, 1.0)}


def test_family_specific_notebooks_include_expected_plot_cells() -> None:
    expected = {
        "run_2D_homo_SSR_capture": ["plot_2d_displacement", "plot_2d_deviatoric_strain"],
        "run_2D_sloan2013_seepage_capture": ["plot_2d_pore_pressure", "plot_2d_saturation"],
        "run_2d_luzec_ssr": [
            "plot_2d_heterogeneity",
            "plot_2d_mesh",
            "plot_2d_pore_pressure",
            "plot_2d_saturation",
            "plot_2d_displacement",
            "plot_2d_deviatoric_strain",
        ],
        "run_3D_hetero_SSR_capture": ["show_3d_displacement_view", "show_3d_deviatoric_surface_view"],
        "run_3d_hetero_ll": ["show_3d_deviatoric_slices"],
        "run_3D_hetero_seepage_capture": ["show_3d_pore_pressure_view", "show_3d_saturation_view"],
        "run_3D_hetero_seepage_SSR_comsol_capture": [
            "show_3d_mesh_view",
            "show_3d_pore_pressure_view",
            "show_3d_displacement_view",
            "show_3d_deviatoric_surface_view",
            "show_3d_deviatoric_slices",
        ],
    }

    for case_name, snippets in expected.items():
        notebook_path = BENCHMARKS_DIR / case_name / "visualisation.ipynb"
        source = _notebook_sources(notebook_path)
        for snippet in snippets:
            assert snippet in source, (case_name, snippet)


def test_3d_hetero_default_notebook_has_no_duplicate_legacy_sections() -> None:
    simulation_source = _notebook_sources(BENCHMARKS_DIR / "3d_hetero_ssr_default" / "simulation.ipynb")
    source = _notebook_sources(BENCHMARKS_DIR / "3d_hetero_ssr_default" / "visualisation.ipynb")

    assert "show_3d_displacement_view" in source
    assert "show_3d_deviatoric_surface_view" in source
    assert "plot_matlab_style_strain_surface" not in source
    assert "plot_static_deviatoric_slices" not in source
    assert "petsc_displacements_3D.png" not in source
    assert "petsc_deviatoric_strain_3D.png" not in source
    assert "show_3d_displacement_view" not in simulation_source
    assert "show_3d_deviatoric_surface_view" not in simulation_source
