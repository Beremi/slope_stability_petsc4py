from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slope_stability.core.run_config import load_run_case_config
from slope_stability.problem_assets import (
    build_seepage_boundary_for_path,
    load_hydraulic_conductivity_for_path,
    load_material_rows_for_path,
    load_seepage_spec_for_path,
    load_water_unit_weight_for_path,
)
from slope_stability.problem_asset_runtime import build_mesh_for_path


ROOT = Path(__file__).resolve().parents[1]
WATERLEVELS_MESH_PATH = ROOT / "meshes" / "3d_hetero_seepage" / "concave_family_b.msh"
WATERLEVELS_ALT_MESH_PATH = ROOT / "meshes" / "3d_hetero_seepage" / "concave_family_c.msh"
TRANSITION_MESH_PATH = ROOT / "meshes" / "3d_hetero_seepage_transition" / "transition_default.msh"
WATERLEVELS_CASE_PATH = ROOT / "benchmarks" / "run_3D_hetero_seepage_capture" / "case.toml"
TRANSITION_CASE_PATH = ROOT / "benchmarks" / "run_3D_hetero_seepage_SSR_comsol_capture" / "case.toml"


def test_waterlevels_mesh_folder_resolves_physical_inputs() -> None:
    rows = load_material_rows_for_path(WATERLEVELS_MESH_PATH)
    assert rows == [
        [15.0, 30.0, 0.0, 10000.0, 0.33, 19.0, 19.0],
        [15.0, 38.0, 0.0, 50000.0, 0.30, 22.0, 22.0],
        [10.0, 35.0, 0.0, 50000.0, 0.30, 21.0, 21.0],
        [18.0, 32.0, 0.0, 20000.0, 0.33, 20.0, 20.0],
    ]
    assert np.array_equal(load_hydraulic_conductivity_for_path(WATERLEVELS_MESH_PATH), np.array([1.0, 1.0, 1.0, 1.0]))
    assert load_water_unit_weight_for_path(WATERLEVELS_MESH_PATH) == pytest.approx(9.81)
    spec = load_seepage_spec_for_path(WATERLEVELS_MESH_PATH, required=True)
    assert spec is not None
    assert spec.conductivity_mode == "by_material"


def test_transition_mesh_folder_resolves_physical_inputs() -> None:
    rows = load_material_rows_for_path(TRANSITION_MESH_PATH)
    assert rows == [
        [15.0, 30.0, 0.0, 10000.0, 0.33, 19.0, 19.0],
        [15.0, 38.0, 0.0, 50000.0, 0.30, 22.0, 22.0],
        [10.0, 35.0, 0.0, 50000.0, 0.30, 21.0, 21.0],
        [18.0, 32.0, 0.0, 20000.0, 0.33, 20.0, 20.0],
    ]
    assert np.array_equal(load_hydraulic_conductivity_for_path(TRANSITION_MESH_PATH), np.array([1.0]))
    assert load_water_unit_weight_for_path(TRANSITION_MESH_PATH) == pytest.approx(9.81)
    spec = load_seepage_spec_for_path(TRANSITION_MESH_PATH, required=True)
    assert spec is not None
    assert spec.conductivity_mode == "uniform"


def test_seepage_boundaries_match_current_behavior() -> None:
    waterlevels = build_mesh_for_path(WATERLEVELS_MESH_PATH, elem_type="P2")
    q_w, pw_d = build_seepage_boundary_for_path(
        WATERLEVELS_MESH_PATH,
        waterlevels.coord,
        waterlevels.surf,
        waterlevels.boundary_labels,
        grho=9.81,
    )
    assert int((~q_w).sum()) == 8181
    assert int(np.count_nonzero(pw_d)) == 3807
    assert float(pw_d.max()) == pytest.approx(539.55)

    transition = build_mesh_for_path(TRANSITION_MESH_PATH, elem_type="P2", profile="fixed_base")
    q_w_transition, pw_d_transition = build_seepage_boundary_for_path(
        TRANSITION_MESH_PATH,
        transition.coord,
        transition.surf,
        transition.boundary_labels,
        grho=9.81,
    )
    assert int((~q_w_transition).sum()) == 6403
    assert int(np.count_nonzero(pw_d_transition)) == 2265
    assert float(pw_d_transition.max()) == pytest.approx(539.55)


def test_waterlevels_folder_definition_applies_to_multiple_mesh_files() -> None:
    assert load_material_rows_for_path(WATERLEVELS_MESH_PATH) == load_material_rows_for_path(WATERLEVELS_ALT_MESH_PATH)
    assert np.array_equal(
        load_hydraulic_conductivity_for_path(WATERLEVELS_MESH_PATH),
        load_hydraulic_conductivity_for_path(WATERLEVELS_ALT_MESH_PATH),
    )
    mesh = build_mesh_for_path(WATERLEVELS_ALT_MESH_PATH, elem_type="P2")
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
