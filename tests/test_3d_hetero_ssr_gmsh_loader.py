from __future__ import annotations

from pathlib import Path

import numpy as np

from slope_stability.core.run_config import load_run_case_config
from slope_stability.io import load_mesh_file
from slope_stability.problem_assets import load_material_rows_for_path


ROOT = Path(__file__).resolve().parents[1]
MESH_PATH = ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh"
CASE_PATH = ROOT / "benchmarks" / "3d_hetero_ssr_default" / "case.toml"


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
