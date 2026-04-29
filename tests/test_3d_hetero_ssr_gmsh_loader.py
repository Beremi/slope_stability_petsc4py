from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from slope_stability.core.simplex_lagrange import tetra_reference_nodes
from slope_stability.fem.basis import local_basis_volume_3d
from slope_stability.fem.quadrature import quadrature_volume_3d
from slope_stability.core.run_config import load_run_case_config
from slope_stability.io import load_mesh_file
from slope_stability.problem_asset_runtime import build_mesh_for_path, resolve_problem_asset
from slope_stability.problem_assets import load_material_rows_for_path


ROOT = Path(__file__).resolve().parents[1]
MESH_PATH = ROOT / "meshes" / "3d_hetero_slope" / "adaptive_family_a_l1.msh"
SIOPT_MESH_PATH = ROOT / "meshes" / "3d_siopt" / "reference_l0.msh"
WATERLEVELS_MESH_PATH = ROOT / "meshes" / "3d_hetero_seepage" / "concave_family_b.msh"
COMSOL_MESH_PATH = ROOT / "meshes" / "3d_hetero_seepage_transition" / "transition_default.msh"
CASE_PATH = ROOT / "benchmarks" / "slope_stability_3D_hetero_SSR_default" / "case.toml"


def test_family_materials_resolve_from_mesh_folder() -> None:
    rows = load_material_rows_for_path(MESH_PATH)
    assert rows is not None
    assert len(rows) == 4
    assert rows[0] == [15.0, 30.0, 0.0, 10000.0, 0.33, 19.0, 19.0]
    assert rows[3] == [18.0, 32.0, 0.0, 20000.0, 0.33, 20.0, 20.0]


def test_config_falls_back_to_family_materials() -> None:
    cfg = load_run_case_config(CASE_PATH)
    assert cfg.material_rows() == [
        [15.0, 30.0, 0.0, 10000.0, 0.33, 19.0, 19.0],
        [15.0, 38.0, 0.0, 50000.0, 0.30, 22.0, 22.0],
        [10.0, 35.0, 0.0, 50000.0, 0.30, 21.0, 21.0],
        [18.0, 32.0, 0.0, 20000.0, 0.33, 20.0, 20.0],
    ]


def test_canonical_loader_builds_p1_p2_and_p4_meshes() -> None:
    mesh_p1 = build_mesh_for_path(MESH_PATH, elem_type="P1")
    assert mesh_p1.coord.shape == (3, 3845)
    assert mesh_p1.elem.shape == (4, 18419)
    assert mesh_p1.surf.shape == (3, 6325)
    assert mesh_p1.elem_type == "P1"
    assert np.array_equal(np.unique(mesh_p1.material_id), np.array([0, 1, 2, 3], dtype=np.int64))

    mesh_p2 = build_mesh_for_path(MESH_PATH, elem_type="P2")
    assert mesh_p2.coord.shape == (3, 27605)
    assert mesh_p2.elem.shape == (10, 18419)
    assert mesh_p2.surf.shape == (6, 6325)
    assert mesh_p2.elem_type == "P2"
    assert int((~mesh_p2.q_mask[0]).sum()) == 395
    assert int((~mesh_p2.q_mask[1]).sum()) == 722
    assert int((~mesh_p2.q_mask[2]).sum()) == 1336

    mesh_p4 = build_mesh_for_path(MESH_PATH, elem_type="P4")
    assert mesh_p4.coord.shape == (3, 208549)
    assert mesh_p4.elem.shape == (35, 18419)
    assert mesh_p4.surf.shape == (15, 6325)
    assert mesh_p4.elem_type == "P4"
    assert int((~mesh_p4.q_mask[0]).sum()) == 1472
    assert int((~mesh_p4.q_mask[1]).sum()) == 2783
    assert int((~mesh_p4.q_mask[2]).sum()) == 5070


def test_generic_loaders_accept_canonical_asset_paths() -> None:
    direct = load_mesh_file(MESH_PATH, elem_type="P2")
    assert direct.coord.shape == (3, 27605)
    assert direct.elem.shape == (10, 18419)
    assert direct.q_mask.shape == (3, 27605)

    waterlevels = build_mesh_for_path(WATERLEVELS_MESH_PATH, elem_type="P2")
    assert waterlevels.coord.shape[0] == 3
    assert waterlevels.elem.shape[0] == 10
    assert waterlevels.q_mask.shape == (3, waterlevels.coord.shape[1])

    comsol = build_mesh_for_path(COMSOL_MESH_PATH, elem_type="P2")
    assert comsol.coord.shape[0] == 3
    assert comsol.elem.shape[0] == 10
    assert comsol.q_mask.shape == (3, comsol.coord.shape[1])


def test_p4_reference_basis_is_nodal() -> None:
    xi = tetra_reference_nodes(4)
    hatp, dhat1, dhat2, dhat3 = local_basis_volume_3d("P4", xi)
    assert hatp.shape == (35, 35)
    assert dhat1.shape == (35, 35)
    assert dhat2.shape == (35, 35)
    assert dhat3.shape == (35, 35)
    assert np.allclose(hatp, np.eye(35), atol=1e-12)


def test_p4_tetra_quadrature_is_degree_six_exact() -> None:
    xi, wf = quadrature_volume_3d("P4")

    def exact_monomial(a: int, b: int, c: int) -> float:
        return float(math.factorial(a) * math.factorial(b) * math.factorial(c) / math.factorial(a + b + c + 3))

    for total_degree in range(7):
        for a in range(total_degree + 1):
            for b in range(total_degree - a + 1):
                c = total_degree - a - b
                approx = float(np.sum(wf * (xi[0, :] ** a) * (xi[1, :] ** b) * (xi[2, :] ** c)))
                assert abs(approx - exact_monomial(a, b, c)) < 1.0e-12


def test_siopt_profiles_replace_boundary_type() -> None:
    asset = resolve_problem_asset(asset_name="3d_siopt", mesh_variant="reference_l0.msh", profile="roller_base")
    mesh_sliding = asset.definition.build_mesh(asset.resolved_variant, elem_type="P2")

    asset_fixed = resolve_problem_asset(asset_name="3d_siopt", mesh_variant="reference_l0.msh", profile="fixed_base")
    mesh_fixed = asset_fixed.definition.build_mesh(asset_fixed.resolved_variant, elem_type="P2")

    assert np.array_equal(np.unique(mesh_fixed.boundary_labels), np.array([0, 1, 2, 3], dtype=np.int64))

    bottom_nodes = set(np.unique(mesh_fixed.surf[:, mesh_fixed.boundary_labels == mesh_fixed.boundary_id_by_name["base"]].ravel()))
    other_boundary_nodes = set(np.unique(mesh_fixed.surf[:, mesh_fixed.boundary_labels != mesh_fixed.boundary_id_by_name["base"]].ravel()))
    bottom_only = np.asarray(sorted(bottom_nodes - other_boundary_nodes), dtype=np.int64)

    assert bottom_only.size > 0
    assert np.all(~mesh_sliding.q_mask[1, bottom_only])
    assert np.all(mesh_sliding.q_mask[0, bottom_only])
    assert np.all(mesh_sliding.q_mask[2, bottom_only])
    assert np.all(~mesh_fixed.q_mask[:, bottom_only])
