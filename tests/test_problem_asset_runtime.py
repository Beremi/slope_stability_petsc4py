from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slope_stability.assets import available_problem_assets, load_problem_asset
from slope_stability.core.run_config import ExecutionConfig, ProblemConfig, RunCaseConfig, load_run_case_config
from slope_stability.execution.asset_case import RouteKind, case_runner_kwargs, select_case_route
from slope_stability.postprocess.case_mesh import rebuild_case_mesh
from slope_stability.problem_asset_runtime import (
    build_mesh_for_resolved_asset,
    build_seepage_boundary_for_resolved_asset,
    load_mechanical_problem_spec,
    load_seepage_problem_spec,
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
        "benchmarks/run_2D_homo_SSR_capture/case.toml": RouteKind.MECHANICS_2D,
        "benchmarks/run_2D_sloan2013_seepage_capture/case.toml": RouteKind.SEEPAGE_2D,
        "benchmarks/run_3D_hetero_SSR_capture/case.toml": RouteKind.MECHANICS_3D,
        "benchmarks/run_3D_hetero_seepage_capture/case.toml": RouteKind.SEEPAGE_3D,
        "benchmarks/run_3D_hetero_seepage_SSR_comsol_capture/case.toml": RouteKind.SEEPAGE_SSR_3D,
    }
    forbidden_kwargs = {
        "conductivity",
        "hydraulic_conductivity",
        "material_rows",
        "mesh_" + "boundary_type",
        "mesh_path",
        "seepage_water_unit_weight",
        "water_unit_weight",
    }

    for rel_path, route_kind in expected.items():
        cfg = load_run_case_config(ROOT / rel_path)
        assert select_case_route(cfg) == route_kind
        _runner, kwargs = case_runner_kwargs(cfg)
        assert kwargs["asset_name"] == cfg.problem.asset
        assert "mesh_variant" in kwargs
        assert forbidden_kwargs.isdisjoint(kwargs)


def test_3d_mechanics_config_routes_tangent_matrix_backend(tmp_path: Path) -> None:
    source = ROOT / "benchmarks/run_3D_hetero_SSR_capture/case.toml"
    text = source.read_text(encoding="utf-8").replace(
        "[execution]\n",
        "[execution]\ntangent_matrix_backend = \"petsc_aij_element\"\n",
        1,
    )
    config_path = tmp_path / "petsc_native_case.toml"
    config_path.write_text(text, encoding="utf-8")

    cfg = load_run_case_config(config_path)
    assert cfg.execution.tangent_matrix_backend == "petsc_aij_element"
    _runner, kwargs = case_runner_kwargs(cfg)
    assert kwargs["tangent_matrix_backend"] == "petsc_aij_element"


def test_petsc_native_experiment_configs_are_loadable() -> None:
    case_dir = ROOT / "benchmarks/experiment_petsc_native_assembly_3D_hetero_SSR_P4_L1"
    configs = sorted(case_dir.glob("*.toml"))
    assert {p.name for p in configs} >= {
        "owned_csr_pmg_shell_32.toml",
        "petsc_aij_pmg_shell_32.toml",
        "petsc_aij_gamg_32.toml",
        "petsc_aij_hypre_32.toml",
        "petsc_aij_bddc_32.toml",
    }

    for path in configs:
        cfg = load_run_case_config(path)
        _runner, kwargs = case_runner_kwargs(cfg)
        assert kwargs["asset_name"] == "3d_hetero_slope"
        assert kwargs["elem_type"] == "P4"
        assert kwargs["omega_max_stop"] == pytest.approx(7.0e6)


def test_run_config_rejects_unknown_tangent_matrix_backend() -> None:
    cfg = RunCaseConfig(
        problem=ProblemConfig(
            name="bad_tangent_matrix_backend",
            asset="3d_hetero_slope",
            mesh_variant="adaptive_family_a_l1.msh",
            analysis="ssr",
            elem_type="P4",
        ),
        execution=ExecutionConfig(tangent_matrix_backend="python_element_loop"),
    )

    with pytest.raises(ValueError, match="tangent_matrix_backend"):
        cfg.validate()


def test_seepage_capable_3d_ll_route_is_rejected() -> None:
    cfg = RunCaseConfig(
        problem=ProblemConfig(
            name="unsupported_ll_on_seepage_asset",
            asset="3d_hetero_seepage_transition",
            mesh_variant="transition_default.msh",
            profile="fixed_base",
            analysis="ll",
            elem_type="P2",
        )
    )

    with pytest.raises(ValueError, match="not supported for seepage-capable 3D asset"):
        select_case_route(cfg)
    with pytest.raises(ValueError, match="not supported for seepage-capable 3D asset"):
        case_runner_kwargs(cfg)


def test_mechanical_problem_spec_uses_selected_profile() -> None:
    fixed = load_mechanical_problem_spec(
        resolve_problem_asset(asset_name="3d_siopt", mesh_variant="reference_l0.msh", profile="fixed_base")
    )
    roller = load_mechanical_problem_spec(
        resolve_problem_asset(asset_name="3d_siopt", mesh_variant="reference_l0.msh", profile="roller_base")
    )

    fixed_base = next(rule for rule in fixed.dirichlet_rules if rule.target == "base")
    roller_base = next(rule for rule in roller.dirichlet_rules if rule.target == "base")
    assert tuple(fixed_base.components) == ("x", "y", "z")
    assert tuple(roller_base.components) == ("y",)


def test_run_config_rejects_unsupported_geometry_and_seepage_fields(tmp_path: Path) -> None:
    seepage_config = tmp_path / "bad_seepage.toml"
    seepage_config.write_text(
        """
[problem]
name = "bad_seepage"
asset = "2d_sloan2013"
analysis = "seepage"
elem_type = "P1"

[seepage]
head_bcs = []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"\[seepage\] fields \['head_bcs'\]"):
        load_run_case_config(seepage_config)

    geometry_config = tmp_path / "bad_geometry.toml"
    geometry_config.write_text(
        """
[problem]
name = "bad_geometry"
asset = "3d_homo_slope"
analysis = "ssr"
elem_type = "P1"

[geometry]
mesh_path = "legacy.msh"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"\[geometry\] fields \['mesh_path'\]"):
        load_run_case_config(geometry_config)


def test_run_config_rejects_unknown_top_level_path_sections(tmp_path: Path) -> None:
    config = tmp_path / "bad_top_level.toml"
    config.write_text(
        """
[problem]
name = "bad_top_level"
asset = "2d_sloan2013"
analysis = "seepage"
elem_type = "P1"

[legacy]
mesh_path = "legacy.msh"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"Top-level fields \['legacy'\]"):
        load_run_case_config(config)


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


def test_waterlevels_seepage_asset_matches_matlab_boundary_and_material_semantics() -> None:
    resolved = resolve_problem_asset(asset_name="3d_hetero_seepage", mesh_variant="concave_family_a.msh")
    mesh = build_mesh_for_resolved_asset(resolved, elem_type="P2")

    lateral = np.setdiff1d(mesh.nodesets["y_lateral_lock"], mesh.nodesets["base"])
    x_lock = np.setdiff1d(mesh.nodesets["x_lock"], mesh.nodesets["base"])

    assert lateral.size > 0
    assert x_lock.size > 0
    assert np.all(mesh.q_mask[1, lateral])
    assert not np.any(mesh.q_mask[2, lateral])
    assert not np.any(mesh.q_mask[0, x_lock])
    assert not np.any(mesh.q_mask[:, mesh.nodesets["base"]])

    expected_by_region = {
        "general_foundation": [15.0, 38.0, 0.0, 50000.0, 0.30, 22.0, 22.0],
        "weak_foundation": [10.0, 35.0, 0.0, 50000.0, 0.30, 21.0, 21.0],
        "slope_mass": [18.0, 32.0, 0.0, 20000.0, 0.33, 20.0, 20.0],
        "cover_layer": [15.0, 30.0, 0.0, 10000.0, 0.33, 19.0, 19.0],
    }
    rows = resolved.definition.material_rows()
    assert rows is not None
    for region, expected_row in expected_by_region.items():
        material_id = mesh.region_id_by_name[region]
        assert rows[material_id] == expected_row

    seepage = load_seepage_problem_spec(resolved)
    assert seepage.seepage.water_unit_weight == pytest.approx(9.81)
    np.testing.assert_allclose(seepage.conductivity, np.ones(4, dtype=np.float64))

    q_w, pw_d = build_seepage_boundary_for_resolved_asset(
        resolved,
        mesh.coord,
        mesh.surf,
        mesh.boundary_labels,
        grho=9.81,
    )
    head_dry = np.asarray(mesh.nodesets["head_dry"], dtype=np.int64)
    head_porous = np.asarray(mesh.nodesets["head_porous"], dtype=np.int64)
    head_free = np.asarray(mesh.nodesets["head_free"], dtype=np.int64)
    dry_only = np.setdiff1d(np.setdiff1d(head_dry, head_porous), head_free)

    assert not np.any(q_w[head_dry])
    assert not np.any(q_w[head_porous])
    assert not np.any(q_w[head_free])
    np.testing.assert_allclose(pw_d[dry_only], 0.0)
    np.testing.assert_allclose(pw_d[head_porous], 9.81 * np.maximum(55.0 - mesh.coord[1, head_porous], 0.0))
    np.testing.assert_allclose(pw_d[head_free], 9.81 * np.maximum(35.0 - mesh.coord[1, head_free], 0.0))


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
