from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slope_stability.assets import load_problem_asset
from slope_stability.core.run_config import load_run_case_config
from slope_stability.problem_assets import (
    load_hydraulic_conductivity_for_asset,
    load_material_rows_for_asset,
    load_seepage_spec_for_asset_definition,
    load_water_unit_weight_for_asset,
)
from slope_stability.problem_asset_runtime import (
    build_mesh_for_resolved_asset,
    build_seepage_boundary_for_resolved_asset,
    resolve_problem_asset,
)


ROOT = Path(__file__).resolve().parents[1]
WATERLEVELS_ASSET = "3d_hetero_seepage"
WATERLEVELS_VARIANT = "concave_family_b.msh"
WATERLEVELS_ALT_VARIANT = "concave_family_c.msh"
TRANSITION_ASSET = "3d_hetero_seepage_transition"
TRANSITION_VARIANT = "transition_default.msh"
WATERLEVELS_CASE_PATH = ROOT / "benchmarks" / "run_3D_hetero_seepage_capture" / "case.toml"
TRANSITION_CASE_PATH = ROOT / "benchmarks" / "run_3D_hetero_seepage_SSR_comsol_capture" / "case.toml"


def test_waterlevels_mesh_folder_resolves_physical_inputs() -> None:
    rows = load_material_rows_for_asset(WATERLEVELS_ASSET)
    assert rows == [
        [15.0, 30.0, 0.0, 10000.0, 0.33, 19.0, 19.0],
        [15.0, 38.0, 0.0, 50000.0, 0.30, 22.0, 22.0],
        [10.0, 35.0, 0.0, 50000.0, 0.30, 21.0, 21.0],
        [18.0, 32.0, 0.0, 20000.0, 0.33, 20.0, 20.0],
    ]
    assert np.array_equal(load_hydraulic_conductivity_for_asset(WATERLEVELS_ASSET), np.array([1.0, 1.0, 1.0, 1.0]))
    assert load_water_unit_weight_for_asset(WATERLEVELS_ASSET) == pytest.approx(9.81)
    spec = load_seepage_spec_for_asset_definition(load_problem_asset(WATERLEVELS_ASSET), required=True)
    assert spec is not None
    assert spec.conductivity_mode == "by_material"


def test_transition_mesh_folder_resolves_physical_inputs() -> None:
    rows = load_material_rows_for_asset(TRANSITION_ASSET)
    assert rows == [
        [15.0, 30.0, 0.0, 10000.0, 0.33, 19.0, 19.0],
        [15.0, 38.0, 0.0, 50000.0, 0.30, 22.0, 22.0],
        [10.0, 35.0, 0.0, 50000.0, 0.30, 21.0, 21.0],
        [18.0, 32.0, 0.0, 20000.0, 0.33, 20.0, 20.0],
    ]
    assert np.array_equal(load_hydraulic_conductivity_for_asset(TRANSITION_ASSET), np.array([1.0]))
    assert load_water_unit_weight_for_asset(TRANSITION_ASSET) == pytest.approx(9.81)
    spec = load_seepage_spec_for_asset_definition(load_problem_asset(TRANSITION_ASSET), required=True)
    assert spec is not None
    assert spec.conductivity_mode == "uniform"


def test_seepage_boundaries_match_current_behavior() -> None:
    resolved_waterlevels = resolve_problem_asset(asset_name=WATERLEVELS_ASSET, mesh_variant=WATERLEVELS_VARIANT)
    waterlevels = build_mesh_for_resolved_asset(resolved_waterlevels, elem_type="P2")
    q_w, pw_d = build_seepage_boundary_for_resolved_asset(
        resolved_waterlevels,
        waterlevels.coord,
        waterlevels.surf,
        waterlevels.boundary_labels,
        grho=9.81,
    )
    assert int((~q_w).sum()) == 8181
    assert int(np.count_nonzero(pw_d)) == 3807
    assert float(pw_d.max()) == pytest.approx(539.55)

    resolved_transition = resolve_problem_asset(
        asset_name=TRANSITION_ASSET,
        mesh_variant=TRANSITION_VARIANT,
        profile="fixed_base",
    )
    transition = build_mesh_for_resolved_asset(resolved_transition, elem_type="P2")
    q_w_transition, pw_d_transition = build_seepage_boundary_for_resolved_asset(
        resolved_transition,
        transition.coord,
        transition.surf,
        transition.boundary_labels,
        grho=9.81,
    )
    assert int((~q_w_transition).sum()) == 6403
    assert int(np.count_nonzero(pw_d_transition)) == 2265
    assert float(pw_d_transition.max()) == pytest.approx(539.55)


def test_waterlevels_folder_definition_applies_to_multiple_mesh_files() -> None:
    resolved_base = resolve_problem_asset(asset_name=WATERLEVELS_ASSET, mesh_variant=WATERLEVELS_VARIANT)
    resolved_alt = resolve_problem_asset(asset_name=WATERLEVELS_ASSET, mesh_variant=WATERLEVELS_ALT_VARIANT)
    assert resolved_base.definition.material_rows() == resolved_alt.definition.material_rows()
    assert np.array_equal(
        resolved_base.definition.hydraulic_conductivity(),
        resolved_alt.definition.hydraulic_conductivity(),
    )
    mesh = build_mesh_for_resolved_asset(resolved_alt, elem_type="P2")
    assert mesh.q_mask.shape[0] == 3
    assert np.array_equal(np.unique(mesh.material_id), np.array([0, 1, 2, 3], dtype=np.int64))


def test_config_runner_uses_asset_family_physical_inputs_for_migrated_3d_cases() -> None:
    cfg = load_run_case_config(WATERLEVELS_CASE_PATH)
    assert cfg.problem.asset == "3d_hetero_seepage"
    assert cfg.problem.mesh_variant == "concave_family_b.msh"
    assert not hasattr(cfg.seepage, "conductivity")

    cfg_transition = load_run_case_config(TRANSITION_CASE_PATH)
    assert cfg_transition.problem.asset == "3d_hetero_seepage_transition"
    assert cfg_transition.problem.mesh_variant == "transition_default.msh"
    assert cfg_transition.problem.profile == "fixed_base"
