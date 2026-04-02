from __future__ import annotations

from pathlib import Path

import h5py
import meshio
import numpy as np
import pytest

from slope_stability.core.elements import simplex_vtk_cell_block
from slope_stability.core.simplex_lagrange import tetra_reference_nodes, tetra_lagrange_node_tuples, triangle_lagrange_node_tuples
from slope_stability.export import write_debug_bundle_h5, write_vtu


def test_simplex_vtk_cell_block_exports_p4_as_vtk_lagrange() -> None:
    tri = np.arange(15, dtype=np.int64).reshape(15, 1)
    tet = np.arange(35, dtype=np.int64).reshape(35, 1)

    tri_type, tri_cells = simplex_vtk_cell_block(2, tri, "P4")
    tet_type, tet_cells = simplex_vtk_cell_block(3, tet, "P4")

    assert tri_type == "VTK_LAGRANGE_TRIANGLE"
    assert tri_cells.shape == (1, 15)
    assert tet_type == "VTK_LAGRANGE_TETRAHEDRON"
    assert tet_cells.shape == (1, 35)


def test_simplex_vtk_cell_block_matches_vtk_quartic_triangle_order() -> None:
    vtk = pytest.importorskip("vtk")
    tri = np.arange(15, dtype=np.int64).reshape(15, 1)

    tri_type, tri_cells = simplex_vtk_cell_block(2, tri, "P4")

    lookup = {node: idx for idx, node in enumerate(triangle_lagrange_node_tuples(4))}
    expected: list[int] = []
    for idx in range(15):
        b = [0, 0, 0]
        vtk.vtkLagrangeTriangle.BarycentricIndex(idx, b, 4)
        expected.append(lookup[(b[2], b[0], b[1])])

    assert tri_type == "VTK_LAGRANGE_TRIANGLE"
    assert np.array_equal(tri_cells[0], np.asarray(expected, dtype=np.int64))


def test_simplex_vtk_cell_block_matches_vtk_quartic_tetra_order() -> None:
    vtk = pytest.importorskip("vtk")
    tet = np.arange(35, dtype=np.int64).reshape(35, 1)

    tet_type, tet_cells = simplex_vtk_cell_block(3, tet, "P4")

    lookup = {node: idx for idx, node in enumerate(tetra_lagrange_node_tuples(4))}
    expected: list[int] = []
    for idx in range(35):
        b = [0, 0, 0, 0]
        vtk.vtkLagrangeTetra.BarycentricIndex(idx, b, 4)
        expected.append(lookup[(b[3], b[0], b[1], b[2])])

    assert tet_type == "VTK_LAGRANGE_TETRAHEDRON"
    assert np.array_equal(tet_cells[0], np.asarray(expected, dtype=np.int64))


def test_write_vtu_round_trips_p4_lagrange_tetra(tmp_path: Path) -> None:
    out_path = tmp_path / "p4.vtu"
    points = np.zeros((35, 3), dtype=np.float64)
    points[:, :3] = tetra_reference_nodes(4).T
    points[:, 2] += np.linspace(0.0, 0.1, 35)
    cell_blocks = [("VTK_LAGRANGE_TETRAHEDRON", np.arange(35, dtype=np.int64).reshape(1, 35))]

    write_vtu(
        out_path,
        points=points,
        cell_blocks=cell_blocks,
        point_data={"deviatoric_strain": np.linspace(0.0, 1.0, 35)},
    )

    mesh = meshio.read(out_path)
    assert len(mesh.cells) == 1
    assert mesh.cells[0].type == "VTK_LAGRANGE_TETRAHEDRON"
    assert mesh.cells[0].data.shape == (1, 35)
    assert "deviatoric_strain" in mesh.point_data

    pv = pytest.importorskip("pyvista")
    grid = pv.read(out_path)
    assert grid.n_cells == 1
    assert grid.n_points == 35
    assert np.array_equal(np.unique(grid.celltypes), np.array([71], dtype=grid.celltypes.dtype))
    assert "deviatoric_strain" in grid.point_data


def test_write_debug_bundle_h5_accepts_unicode_arrays(tmp_path: Path) -> None:
    npz_path = tmp_path / "run.npz"
    run_info_path = tmp_path / "run_info.json"
    out_path = tmp_path / "run_debug.h5"
    np.savez(
        npz_path,
        lambda_hist=np.array([1.0, 1.1], dtype=np.float64),
        precision_mode=np.array(["base", "fine"], dtype="<U4"),
    )
    run_info_path.write_text('{"run_info": {"step_count": 2}}', encoding="utf-8")

    write_debug_bundle_h5(
        out_path=out_path,
        config_text="[problem]\nelem_type = \"P4\"\n",
        run_info_path=run_info_path,
        npz_path=npz_path,
    )

    with h5py.File(out_path, "r") as h5:
        data = [item.decode("utf-8") for item in h5["arrays"]["precision_mode"][...].tolist()]
    assert data == ["base", "fine"]
