from __future__ import annotations

from pathlib import Path

import numpy as np

from slope_stability.assets import available_problem_assets, load_problem_asset
from slope_stability.cli.run_case_from_config import _case_runner_kwargs
from slope_stability.core.run_config import load_run_case_config
from slope_stability.postprocess.case_mesh import rebuild_case_mesh
from slope_stability.problem_asset_runtime import (
    build_mesh_for_resolved_asset,
    build_seepage_boundary_for_resolved_asset,
    resolve_problem_asset,
    resolve_problem_asset_from_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_configs_resolve_assets_across_current_benchmarks() -> None:
    expected = {
        "benchmarks/run_2D_homo_SSR_capture/case.toml": ("2d_homo_slope", "h1.0.msh", "default"),
        "benchmarks/run_2D_sloan2013_seepage_capture/case.toml": ("2d_sloan2013", "default.msh", "default"),
        "benchmarks/slope_stability_2D_Kozinec_SSR/case.toml": ("2d_kozinec", "default.msh", "default"),
        "benchmarks/slope_stability_2D_Luzec_SSR/case.toml": ("2d_luzec", "default.msh", "default"),
        "benchmarks/slope_stability_2D_Franz_dam_SSR/case.toml": ("2d_franz_dam", "default.msh", "default"),
        "benchmarks/run_3D_hetero_SSR_capture/case.toml": ("3d_hetero_slope", "adaptive_family_a_l1.msh", "default"),
        "benchmarks/run_3D_hetero_seepage_capture/case.toml": ("3d_hetero_seepage", "concave_family_b.msh", "default"),
        "benchmarks/run_3D_hetero_seepage_SSR_comsol_capture/case.toml": ("3d_hetero_seepage_transition", "transition_default.msh", "fixed_base"),
        "benchmarks/SIOPT_SSR/case.toml": ("3d_siopt", "reference_l0.msh", "fixed_base"),
    }

    for rel_path, (asset_name, variant_name, profile) in expected.items():
        cfg = load_run_case_config(ROOT / rel_path)
        resolved = resolve_problem_asset_from_config(cfg)
        assert resolved.asset_name == asset_name
        assert resolved.variant_name == variant_name
        assert resolved.resolved_variant.profile == profile


def test_config_runner_routes_are_asset_first() -> None:
    expected = {
        "benchmarks/run_2D_homo_SSR_capture/case.toml": "slope_stability.cli.run_2d_mechanics_capture",
        "benchmarks/run_2D_sloan2013_seepage_capture/case.toml": "slope_stability.cli.run_2d_seepage_capture",
        "benchmarks/run_3D_hetero_SSR_capture/case.toml": "slope_stability.cli.run_3d_mechanics_capture",
        "benchmarks/run_3D_hetero_seepage_capture/case.toml": "slope_stability.cli.run_3d_seepage_capture",
        "benchmarks/run_3D_hetero_seepage_SSR_comsol_capture/case.toml": "slope_stability.cli.run_3d_seepage_ssr_capture",
    }
    forbidden_kwargs = {
        "conductivity",
        "hydraulic_conductivity",
        "material_rows",
        "mesh_boundary_type",
        "mesh_path",
        "seepage_water_unit_weight",
        "water_unit_weight",
    }

    for rel_path, module_name in expected.items():
        cfg = load_run_case_config(ROOT / rel_path)
        runner, kwargs = _case_runner_kwargs(cfg)
        assert runner.__module__ == module_name
        assert kwargs["asset_name"] == cfg.problem.asset
        assert "mesh_variant" in kwargs
        assert forbidden_kwargs.isdisjoint(kwargs)


def test_assets_build_solver_meshes_from_definitions() -> None:
    for asset_name in available_problem_assets():
        asset = load_problem_asset(asset_name)
        elem_type = "P2" if asset.dimension == 2 else "P1"
        resolved = resolve_problem_asset(asset_name=asset_name)
        mesh = build_mesh_for_resolved_asset(resolved, elem_type=elem_type)

        assert mesh.coord.shape[0] == asset.dimension
        assert mesh.elem.shape[1] > 0
        assert mesh.surf.shape[1] > 0
        assert mesh.q_mask.shape == mesh.coord.shape
        assert mesh.material_id.shape == (mesh.elem.shape[1],)
        assert np.all(mesh.material_id >= 0)


def test_seepage_boundaries_are_asset_owned() -> None:
    cases = [
        ("2d_sloan2013", "P2"),
        ("2d_luzec", "P2"),
        ("2d_franz_dam", "P2"),
        ("3d_hetero_seepage", "P1"),
        ("3d_hetero_seepage_transition", "P1"),
    ]
    for asset_name, elem_type in cases:
        resolved = resolve_problem_asset(asset_name=asset_name)
        mesh = build_mesh_for_resolved_asset(resolved, elem_type=elem_type)
        q_w, pw_d = build_seepage_boundary_for_resolved_asset(
            resolved,
            mesh.coord,
            mesh.surf,
            mesh.boundary_labels,
            grho=9.81,
        )

        assert q_w.shape == (mesh.coord.shape[1],)
        assert pw_d.shape == (mesh.coord.shape[1],)
        assert q_w.dtype == np.dtype(bool)
        assert np.all(np.isfinite(pw_d))
        assert np.any(~q_w)


def test_rebuild_case_mesh_uses_asset_runtime() -> None:
    for rel_path in (
        "benchmarks/run_2D_homo_SSR_capture/case.toml",
        "benchmarks/run_2D_sloan2013_seepage_capture/case.toml",
        "benchmarks/slope_stability_2D_Kozinec_SSR/case.toml",
        "benchmarks/slope_stability_2D_Luzec_SSR/case.toml",
        "benchmarks/run_3D_hetero_SSR_capture/case.toml",
    ):
        cfg = load_run_case_config(ROOT / rel_path)
        mesh = rebuild_case_mesh(cfg, mpi_size=1)
        assert mesh.coord.shape[1] > 0
        assert mesh.elem.shape[1] > 0
        assert mesh.material_id.shape[0] == mesh.elem.shape[1]
