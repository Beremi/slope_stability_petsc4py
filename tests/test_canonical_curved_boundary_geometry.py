from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
import pytest

from slope_stability.assets.support.canonical_gmsh import build_solver_mesh, load_canonical_gmsh_mesh


def ensure_2d_points(coord: np.ndarray) -> np.ndarray:
    coord_arr = np.asarray(coord, dtype=np.float64)
    out = np.zeros((coord_arr.shape[1], 3), dtype=np.float64)
    out[:, :2] = coord_arr.T
    return out


def ensure_3d_points(coord: np.ndarray) -> np.ndarray:
    return np.asarray(coord, dtype=np.float64).T.copy()


def write_named_msh(
    *,
    out_path: Path,
    points: np.ndarray,
    volume_type: str,
    volume_cells: np.ndarray,
    region_by_entity: list[str],
    boundary_type: str | None = None,
    boundary_cells: np.ndarray | None = None,
    boundary_by_entity: list[str] | None = None,
    boundary_geom_type: str | None = None,
    boundary_geom_cells: np.ndarray | None = None,
    boundary_geom_by_entity: list[str] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cells: list[tuple[str, np.ndarray]] = []
    physical_blocks: list[np.ndarray] = []
    geometrical_blocks: list[np.ndarray] = []
    field_data: dict[str, np.ndarray] = {}

    def add_block(*, cell_type: str, data: np.ndarray, names: list[str], prefix: str, dim: int) -> None:
        tag_by_name: dict[str, int] = {}
        tags = np.empty(len(names), dtype=np.int32)
        for idx, name in enumerate(names):
            tag = tag_by_name.get(name)
            if tag is None:
                tag = len(tag_by_name) + 1
                tag_by_name[name] = tag
                field_data[f"{prefix}:{name}"] = np.asarray([tag, dim], dtype=np.int32)
            tags[idx] = tag
        cells.append((cell_type, np.asarray(data, dtype=np.int64).T))
        physical_blocks.append(tags)
        geometrical_blocks.append(tags.copy())

    dim = 2 if volume_type == "triangle" else 3
    add_block(cell_type=volume_type, data=volume_cells, names=region_by_entity, prefix="region", dim=dim)
    if boundary_type is not None and boundary_cells is not None and boundary_by_entity is not None:
        add_block(
            cell_type=boundary_type,
            data=boundary_cells,
            names=boundary_by_entity,
            prefix="boundary",
            dim=dim - 1,
        )
    if boundary_geom_type is not None and boundary_geom_cells is not None and boundary_geom_by_entity is not None:
        add_block(
            cell_type=boundary_geom_type,
            data=boundary_geom_cells,
            names=boundary_geom_by_entity,
            prefix="boundary_geom",
            dim=dim - 1,
        )

    mesh = meshio.Mesh(
        points=np.asarray(points, dtype=np.float64),
        cells=cells,
        cell_data={
            "gmsh:physical": physical_blocks,
            "gmsh:geometrical": geometrical_blocks,
        },
        field_data=field_data,
    )
    meshio.write(out_path, mesh, file_format="gmsh22", binary=False)


def _curve_point(t: float, p0: np.ndarray, p1: np.ndarray, pmid: np.ndarray) -> np.ndarray:
    n0 = (1.0 - t) * (1.0 - 2.0 * t)
    n1 = t * (2.0 * t - 1.0)
    n2 = 4.0 * t * (1.0 - t)
    return (n0 * p0) + (n1 * p1) + (n2 * pmid)


def test_2d_curved_boundary_geometry_patch_repositions_promoted_nodes(tmp_path: Path) -> None:
    out_path = tmp_path / "curved_2d.msh"
    points = ensure_2d_points(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.5],
                [0.0, 0.0, 1.0, 0.2],
            ],
            dtype=np.float64,
        )
    )
    write_named_msh(
        out_path=out_path,
        points=points,
        volume_type="triangle",
        volume_cells=np.array([[0], [1], [2]], dtype=np.int64),
        region_by_entity=["domain"],
        boundary_type="line",
        boundary_cells=np.array([[0], [1]], dtype=np.int64),
        boundary_by_entity=["curved_edge"],
        boundary_geom_type="line3",
        boundary_geom_cells=np.array([[0], [1], [3]], dtype=np.int64),
        boundary_geom_by_entity=["arch"],
    )

    canonical = load_canonical_gmsh_mesh(out_path, dimension=2)
    assert canonical.coord.shape == (2, 3)
    mesh = build_solver_mesh(
        canonical,
        elem_type="P4",
        boundary_geometry_specs={"arch": ("curved_edge", 2)},
        region_id_by_name={"domain": 0},
        boundary_id_by_name={"curved_edge": 0},
    )

    support_idx = int(canonical.boundary_groups["curved_edge"][0])
    support_nodes = mesh.surf[:, support_idx]
    p0 = np.array([0.0, 0.0], dtype=np.float64)
    p1 = np.array([1.0, 0.0], dtype=np.float64)
    pmid = np.array([0.5, 0.2], dtype=np.float64)
    expected = np.column_stack(
        [
            _curve_point(0.0, p0, p1, pmid),
            _curve_point(1.0, p0, p1, pmid),
            _curve_point(0.5, p0, p1, pmid),
            _curve_point(0.25, p0, p1, pmid),
            _curve_point(0.75, p0, p1, pmid),
        ]
    )
    assert np.allclose(mesh.coord[:, support_nodes], expected, atol=1.0e-12)


def test_3d_curved_boundary_geometry_patch_uses_geometry_only_midside_nodes(tmp_path: Path) -> None:
    out_path = tmp_path / "curved_3d.msh"
    points = ensure_3d_points(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 1.0, 0.2, 0.2, 0.2],
            ],
            dtype=np.float64,
        )
    )
    write_named_msh(
        out_path=out_path,
        points=points,
        volume_type="tetra",
        volume_cells=np.array([[0], [1], [2], [3]], dtype=np.int64),
        region_by_entity=["domain"],
        boundary_type="triangle",
        boundary_cells=np.array([[0], [1], [2]], dtype=np.int64),
        boundary_by_entity=["curved_face"],
        boundary_geom_type="triangle6",
        boundary_geom_cells=np.array([[0], [1], [2], [4], [5], [6]], dtype=np.int64),
        boundary_geom_by_entity=["cap"],
    )

    canonical = load_canonical_gmsh_mesh(out_path, dimension=3)
    assert canonical.coord.shape == (3, 4)
    mesh = build_solver_mesh(
        canonical,
        elem_type="P2",
        boundary_geometry_specs={"cap": ("curved_face", 2)},
        region_id_by_name={"domain": 0},
        boundary_id_by_name={"curved_face": 0},
    )

    support_idx = int(canonical.boundary_groups["curved_face"][0])
    face_coords = mesh.coord[:, mesh.surf[:, support_idx]].T
    expected_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.2],
            [0.5, 0.5, 0.2],
            [0.0, 0.5, 0.2],
        ],
        dtype=np.float64,
    )
    observed = sorted(tuple(np.round(row, 12)) for row in face_coords)
    expected = sorted(tuple(np.round(row, 12)) for row in expected_points)
    assert observed == expected


def test_curved_boundary_geometry_requires_matching_support_simplex(tmp_path: Path) -> None:
    out_path = tmp_path / "bad_curved_2d.msh"
    points = ensure_2d_points(
        np.array(
            [
                [0.0, 1.0, 0.0, 0.5],
                [0.0, 0.0, 1.0, 0.2],
            ],
            dtype=np.float64,
        )
    )
    write_named_msh(
        out_path=out_path,
        points=points,
        volume_type="triangle",
        volume_cells=np.array([[0], [1], [2]], dtype=np.int64),
        region_by_entity=["domain"],
        boundary_type="line",
        boundary_cells=np.array([[0], [2]], dtype=np.int64),
        boundary_by_entity=["different_edge"],
        boundary_geom_type="line3",
        boundary_geom_cells=np.array([[0], [1], [3]], dtype=np.int64),
        boundary_geom_by_entity=["arch"],
    )

    canonical = load_canonical_gmsh_mesh(out_path, dimension=2)
    with pytest.raises(ValueError, match="no matching support edge"):
        build_solver_mesh(
            canonical,
            elem_type="P2",
            boundary_geometry_specs={"arch": ("different_edge", 2)},
            region_id_by_name={"domain": 0},
            boundary_id_by_name={"different_edge": 0},
        )
