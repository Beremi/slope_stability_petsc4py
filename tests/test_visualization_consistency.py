from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
import tomllib

import matplotlib.pyplot as plt
import numpy as np
import pytest

from slope_stability.cli.run_case_from_config import _case_runner_kwargs, _export_outputs
from slope_stability.core.run_config import load_run_case_config
from slope_stability.postprocess import rebuild_case_mesh, validate_case_mesh_alignment


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT / "benchmarks"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _support():
    return _load_module(BENCHMARKS_DIR / "notebook_support.py", "benchmarks_notebook_support_consistency")


def _case_tomls_with_artifacts() -> list[Path]:
    case_tomls: list[Path] = []
    for case_toml in sorted(BENCHMARKS_DIR.glob("*/case.toml")):
        out_dir = case_toml.parent / "artifacts" / "simulation"
        if (out_dir / "exports" / "final_solution.vtu").exists() and (out_dir / "data" / "petsc_run.npz").exists():
            case_tomls.append(case_toml)
    return case_tomls


ALL_ARTIFACT_CASES = [case_toml.parent.name for case_toml in _case_tomls_with_artifacts()]
TWO_D_ARTIFACT_CASES = [
    case_toml.parent.name
    for case_toml in _case_tomls_with_artifacts()
    if int(tomllib.loads(case_toml.read_text(encoding="utf-8")).get("problem", {}).get("dimension", -1)) == 2
]
THREE_D_ARTIFACT_CASES = [
    case_toml.parent.name
    for case_toml in _case_tomls_with_artifacts()
    if int(tomllib.loads(case_toml.read_text(encoding="utf-8")).get("problem", {}).get("dimension", -1)) == 3
]


def _artifacts_dir(case_name: str) -> Path:
    return BENCHMARKS_DIR / case_name / "artifacts" / "simulation"


def _generated_config(case_name: str) -> Path:
    return _artifacts_dir(case_name) / "generated_case.toml"


def _runtime_config(case_name: str) -> Path:
    generated = _generated_config(case_name)
    try:
        load_run_case_config(generated)
    except Exception:
        return BENCHMARKS_DIR / case_name / "case.toml"
    return generated


def _load_case_context(case_name: str):
    module = _support()
    artifacts = module.load_run_artifacts(_artifacts_dir(case_name))
    vtu = module.load_vtu(artifacts.vtu_path)
    return module, artifacts, vtu


def _assert_field_sizes(vtu) -> None:
    n_points = int(vtu.points.shape[0])
    n_cells = int(sum(block.shape[0] for _, block in vtu.cell_blocks))
    for name, values in vtu.point_data.items():
        arr = np.asarray(values)
        assert arr.shape[0] == n_points, (name, arr.shape, n_points)
    for name, values in vtu.cell_data.items():
        arr = np.asarray(values)
        assert arr.shape[0] == n_cells, (name, arr.shape, n_cells)


def _run_case_to_output(case_toml: Path, out_dir: Path) -> Path:
    cfg = load_run_case_config(case_toml)
    runner, kwargs = _case_runner_kwargs(cfg)
    sig = inspect.signature(runner)
    accepted = set(sig.parameters)
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    output_path = Path(out_dir)
    if "output_dir" in sig.parameters:
        runner(output_path, **filtered_kwargs)
    elif "out_dir" in sig.parameters:
        runner(out_dir=output_path, **filtered_kwargs)
    else:
        raise TypeError(f"Unsupported runner signature for {runner.__module__}.{runner.__name__}")
    _export_outputs(cfg, case_toml.resolve(), output_path)
    return output_path


@pytest.mark.parametrize("case_name", ALL_ARTIFACT_CASES)
def test_committed_vtu_data_lengths_match_mesh(case_name: str) -> None:
    _module, _artifacts, vtu = _load_case_context(case_name)
    _assert_field_sizes(vtu)


@pytest.mark.parametrize("case_name", TWO_D_ARTIFACT_CASES)
def test_committed_2d_artifact_npz_matches_rebuilt_case_mesh(case_name: str) -> None:
    module = _support()
    artifacts = module.load_run_artifacts(_artifacts_dir(case_name))
    cfg = load_run_case_config(_runtime_config(case_name))
    case_mesh = rebuild_case_mesh(cfg, mpi_size=int(artifacts.run_info.get("run_info", {}).get("mpi_size", 1)))

    validate_case_mesh_alignment(case_mesh, artifacts.npz)


def test_sloan_runtime_export_mesh_alignment(tmp_path: Path) -> None:
    case_toml = BENCHMARKS_DIR / "run_2D_sloan2013_seepage_capture" / "case.toml"
    out_dir = _run_case_to_output(case_toml, tmp_path / "sloan_runtime_export")
    module = _support()
    artifacts = module.load_run_artifacts(out_dir)
    vtu = module.load_vtu(artifacts.vtu_path)
    cfg = load_run_case_config(out_dir / "generated_case.toml")
    case_mesh = rebuild_case_mesh(cfg, mpi_size=int(artifacts.run_info.get("run_info", {}).get("mpi_size", 1)))

    validate_case_mesh_alignment(case_mesh, artifacts.npz)
    np.testing.assert_allclose(vtu.points, case_mesh.points)
    assert len(vtu.cell_blocks) == len(case_mesh.cell_blocks)
    for (vtu_type, vtu_cells), (mesh_type, mesh_cells) in zip(vtu.cell_blocks, case_mesh.cell_blocks, strict=True):
        assert vtu_type == mesh_type
        assert np.array_equal(vtu_cells, mesh_cells)
    np.testing.assert_allclose(np.asarray(vtu.point_data["pore_pressure"], dtype=np.float64), np.asarray(artifacts.npz["pw"], dtype=np.float64))


@pytest.mark.parametrize(
    ("case_name", "plotter_names"),
    [
        (
            "slope_stability_2D_Franz_dam_SSR",
            ("plot_2d_pore_pressure", "plot_2d_saturation", "plot_2d_displacement", "plot_2d_deviatoric_strain"),
        ),
        (
            "run_2D_homo_SSR_capture",
            ("plot_2d_displacement", "plot_2d_deviatoric_strain"),
        ),
        (
            "run_2D_sloan2013_seepage_capture",
            ("plot_2d_pore_pressure", "plot_2d_saturation"),
        ),
    ],
)
def test_ordering_regression_2d_plots_render(case_name: str, plotter_names: tuple[str, ...]) -> None:
    module = _support()
    out_dir = _artifacts_dir(case_name)
    artifacts = module.load_run_artifacts(out_dir)
    active_config = _generated_config(case_name)

    for plotter_name in plotter_names:
        fig = getattr(module, plotter_name)(artifacts, active_config)
        assert fig is not None
        plt.close(fig)


@pytest.mark.parametrize("case_name", THREE_D_ARTIFACT_CASES)
def test_committed_3d_vtus_load_cleanly(case_name: str) -> None:
    _module, _artifacts, vtu = _load_case_context(case_name)
    assert vtu.points.shape[0] > 0
    assert len(vtu.cell_blocks) > 0
    _assert_field_sizes(vtu)


@pytest.mark.parametrize(
    ("case_name", "plotter_name"),
    [
        ("run_3D_hetero_SSR_capture", "show_3d_mesh_view"),
        ("run_3D_hetero_seepage_capture", "show_3d_pore_pressure_view"),
        ("SIOPT_SSR", "show_3d_deviatoric_surface_view"),
    ],
)
def test_representative_3d_plotters_read_committed_vtus(monkeypatch, case_name: str, plotter_name: str) -> None:
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    module = _support()
    artifacts = module.load_run_artifacts(_artifacts_dir(case_name))
    case_toml = BENCHMARKS_DIR / case_name / "case.toml"

    monkeypatch.setattr(module, "_module_available", lambda name: name == "pyvista")
    monkeypatch.setattr(module, "_import_pyvista", lambda: pv)
    monkeypatch.setattr(module, "_new_plotter", lambda pv_mod, title: pv_mod.Plotter(off_screen=True))
    monkeypatch.setattr(module, "_show_plotter", lambda plotter, *args, **kwargs: plotter.close() or "shown")

    result = getattr(module, plotter_name)(artifacts, case_toml)

    assert result == "shown"
