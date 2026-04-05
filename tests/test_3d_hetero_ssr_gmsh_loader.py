from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from slope_stability.core.simplex_lagrange import tetra_reference_nodes
from slope_stability.fem.basis import local_basis_volume_3d
from slope_stability.fem.quadrature import quadrature_volume_3d
from slope_stability.core.run_config import load_run_case_config
from slope_stability.io import load_mesh_file
from slope_stability.problem_assets import load_material_rows_for_path


ROOT = Path(__file__).resolve().parents[1]
MESH_PATH = ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh"
SIOPT_MESH_PATH = ROOT / "meshes" / "3d_siopt" / "SIOPT_L0.msh"
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


def test_gmsh_loader_reads_tet4_and_elevates_to_tet10() -> None:
    mesh_p1 = load_mesh_file(MESH_PATH, elem_type="P1")
    assert mesh_p1.coord.shape == (3, 3845)
    assert mesh_p1.elem.shape == (4, 18419)
    assert mesh_p1.surf.shape == (3, 6325)
    assert mesh_p1.elem_type == "P1"
    assert np.array_equal(np.unique(mesh_p1.material), np.array([0, 1, 2, 3], dtype=np.int64))
    assert np.array_equal(np.unique(mesh_p1.boundary), np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64))

    mesh_p2 = load_mesh_file(MESH_PATH, elem_type="P2")
    assert mesh_p2.coord.shape == (3, 27605)
    assert mesh_p2.elem.shape == (10, 18419)
    assert mesh_p2.surf.shape == (6, 6325)
    assert mesh_p2.elem_type == "P2"
    assert int((~mesh_p2.q_mask[0]).sum()) == 395
    assert int((~mesh_p2.q_mask[1]).sum()) == 722
    assert int((~mesh_p2.q_mask[2]).sum()) == 1336


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


def test_gmsh_loader_elevates_tet4_to_tet35() -> None:
    mesh_p4 = load_mesh_file(MESH_PATH, elem_type="P4")
    assert mesh_p4.coord.shape == (3, 208549)
    assert mesh_p4.elem.shape == (35, 18419)
    assert mesh_p4.surf.shape == (15, 6325)
    assert mesh_p4.elem_type == "P4"
    assert np.array_equal(np.unique(mesh_p4.material), np.array([0, 1, 2, 3], dtype=np.int64))
    assert np.array_equal(np.unique(mesh_p4.boundary), np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64))
    assert int((~mesh_p4.q_mask[0]).sum()) == 1472
    assert int((~mesh_p4.q_mask[1]).sum()) == 2783
    assert int((~mesh_p4.q_mask[2]).sum()) == 5070


def test_siopt_boundary_type_glues_bottom_in_all_directions() -> None:
    mesh_sliding = load_mesh_file(SIOPT_MESH_PATH, elem_type="P2", boundary_type=0)
    mesh_glued = load_mesh_file(SIOPT_MESH_PATH, elem_type="P2", boundary_type=1)

    assert np.array_equal(np.unique(mesh_glued.boundary), np.array([0, 1, 3, 5], dtype=np.int64))

    bottom_nodes = set(np.unique(mesh_glued.surf[:, mesh_glued.boundary == 5].ravel()))
    other_boundary_nodes = set(np.unique(mesh_glued.surf[:, mesh_glued.boundary != 5].ravel()))
    bottom_only = np.asarray(sorted(bottom_nodes - other_boundary_nodes), dtype=np.int64)

    assert bottom_only.size > 0
    assert np.all(~mesh_sliding.q_mask[1, bottom_only])
    assert np.all(mesh_sliding.q_mask[0, bottom_only])
    assert np.all(mesh_sliding.q_mask[2, bottom_only])
    assert np.all(~mesh_glued.q_mask[:, bottom_only])
