"""Asset-specific canonical mesh conversion helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import meshio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from slope_stability.assets.support.selectors import SeepageDefinition, build_seepage_boundary
from slope_stability.mesh.sloan2013_2d import generate_sloan2013_mesh_2d
from slope_stability.mesh.slope_2d import generate_homogeneous_slope_mesh_2d
from slope_stability.mesh.textmesh_2d import (
    franz_dam_pressure_boundary,
    load_mesh_franz_dam_2d,
    load_mesh_kozinec_2d,
    load_mesh_luzec_2d,
    luzec_pressure_boundary,
)
from tools.mesh_conversion.common import (
    ensure_2d_points,
    orient_tetrahedra_positive,
    orient_triangles_positive,
    write_canonical_msh,
)


def _legacy_source_dir(asset_dir: Path) -> Path:
    legacy = asset_dir / "legacy" / "source"
    return legacy if legacy.exists() else asset_dir


def _legacy_source_root(asset_dir: Path) -> Path | None:
    legacy = asset_dir / "legacy" / "source"
    return legacy if legacy.exists() else None


def _legacy_suffix_map(field_data: dict[str, np.ndarray], *, prefix: str, dim: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    head = f"{prefix}_"
    for name, meta in field_data.items():
        text = str(name).strip().lower()
        if not text.startswith(head):
            continue
        arr = np.asarray(meta, dtype=np.int64).ravel()
        if arr.size < 2 or int(arr[1]) != int(dim):
            continue
        mapping[int(arr[0])] = int(text.split("_", 1)[1])
    return mapping


def _meshio_cells(mesh, cell_type: str) -> tuple[np.ndarray, np.ndarray]:
    physical = mesh.cell_data["gmsh:physical"]
    cells_out: list[np.ndarray] = []
    tags_out: list[np.ndarray] = []
    for block, tags in zip(mesh.cells, physical, strict=False):
        if str(block.type) != str(cell_type):
            continue
        cells_out.append(np.asarray(block.data, dtype=np.int64))
        tags_out.append(np.asarray(tags, dtype=np.int64).ravel())
    if not cells_out:
        width = {"tetra": 4, "triangle": 3, "line": 2}.get(cell_type, 0)
        return np.empty((width, 0), dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.vstack(cells_out).T, np.concatenate(tags_out)


def _nodes_on_edges(surf: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(surf[:, mask], dtype=np.int64).ravel()).astype(np.int64)


def _nodes_on_faces(surf: np.ndarray, labels: np.ndarray, selected_labels: set[int]) -> np.ndarray:
    mask = np.isin(np.asarray(labels, dtype=np.int64), np.asarray(sorted(selected_labels), dtype=np.int64))
    return np.unique(np.asarray(surf[:, mask], dtype=np.int64).ravel()).astype(np.int64)


def _write_2d_asset(
    out_path: Path,
    *,
    coord: np.ndarray,
    elem: np.ndarray,
    surf: np.ndarray,
    region_names: list[str],
    boundary_names: list[str],
    nodesets: dict[str, np.ndarray],
) -> None:
    write_canonical_msh(
        out_path=out_path,
        points=ensure_2d_points(coord),
        volume_type="triangle",
        volume_cells=orient_triangles_positive(coord, elem),
        region_by_entity=region_names,
        boundary_type="line",
        boundary_cells=np.asarray(surf, dtype=np.int64),
        boundary_by_entity=boundary_names,
        nodesets=nodesets,
    )


def convert_2d_homo_slope(asset_dir: Path) -> None:
    params = {
        "h1.0.msh": dict(h=1.0),
        "h0.5.msh": dict(h=0.5),
    }
    for name, extra in params.items():
        mesh = generate_homogeneous_slope_mesh_2d(
            elem_type="P1",
            h=float(extra["h"]),
            x1=15.0,
            x2=10.0,
            x3=15.0,
            y1=10.0,
            y2=10.0,
        )
        coord = np.asarray(mesh.coord, dtype=np.float64)
        elem = np.asarray(mesh.elem, dtype=np.int64)
        surf = np.asarray(mesh.surf[:2, :], dtype=np.int64)
        x_min = float(coord[0, :].min())
        x_max = float(coord[0, :].max())
        y_min = float(coord[1, :].min())
        y_max = float(coord[1, :].max())
        mids = coord[:, surf].mean(axis=1)
        boundary_names: list[str] = []
        for idx in range(surf.shape[1]):
            edge = surf[:, idx]
            if np.allclose(coord[0, edge], x_min):
                boundary_names.append("left")
            elif np.allclose(coord[0, edge], x_max):
                boundary_names.append("right")
            elif np.allclose(coord[1, edge], y_min):
                boundary_names.append("base")
            elif np.allclose(coord[1, edge], y_max):
                boundary_names.append("crest")
            else:
                boundary_names.append("slope_surface")
        nodesets = {
            "left": np.flatnonzero(np.isclose(coord[0, :], x_min)).astype(np.int64),
            "right": np.flatnonzero(np.isclose(coord[0, :], x_max)).astype(np.int64),
            "base": np.flatnonzero(np.isclose(coord[1, :], y_min)).astype(np.int64),
        }
        _write_2d_asset(
            asset_dir / name,
            coord=coord,
            elem=elem,
            surf=surf,
            region_names=["slope_mass"] * elem.shape[1],
            boundary_names=boundary_names,
            nodesets=nodesets,
        )


def convert_2d_sloan2013(asset_dir: Path) -> None:
    mesh = generate_sloan2013_mesh_2d(elem_type="P1")
    coord = np.asarray(mesh.coord, dtype=np.float64)
    elem = np.asarray(mesh.elem, dtype=np.int64)
    surf = np.asarray(mesh.surf[:2, :], dtype=np.int64)
    x_max = float(coord[0, :].max())
    y_min = float(coord[1, :].min())
    y_max = float(coord[1, :].max())
    y1 = 6.75 + 0.5 + 0.75
    y2 = 1.0 + 9.25 + 2.0
    beta = np.deg2rad(26.6)
    x2 = float(y2 / np.tan(beta))
    x1 = 15.0
    head_mask = (
        (coord[0, :] <= 1.0e-3)
        | (coord[0, :] >= x_max - 1.0e-3)
        | (coord[1, :] >= y_max - 1.0e-3)
        | ((coord[1, :] >= y1 - 1.0e-3) & (coord[0, :] >= x1 + x2 - 1.0e-3))
        | (
            (coord[1, :] >= y1 - 1.0e-3)
            & (coord[1, :] >= (-(y2 / x2) * coord[0, :] + y1 + y2 * (1.0 + x1 / x2) - 1.0e-3))
        )
    )
    boundary_names: list[str] = []
    for idx in range(surf.shape[1]):
        edge = surf[:, idx]
        if np.allclose(coord[0, edge], 0.0):
            boundary_names.append("left")
        elif np.allclose(coord[0, edge], x_max):
            boundary_names.append("right")
        elif np.allclose(coord[1, edge], y_min):
            boundary_names.append("base")
        elif np.all(head_mask[edge]):
            boundary_names.append("head_support")
        elif np.allclose(coord[1, edge], y_max):
            boundary_names.append("crest")
        else:
            boundary_names.append("slope_surface")
    _write_2d_asset(
        asset_dir / "default.msh",
        coord=coord,
        elem=elem,
        surf=surf,
        region_names=["weak_layer" if int(mid) == 1 else "slope_mass" for mid in np.asarray(mesh.material, dtype=np.int64)],
        boundary_names=boundary_names,
        nodesets={
            "left": np.flatnonzero(coord[0, :] <= 1.0e-3).astype(np.int64),
            "right": np.flatnonzero(coord[0, :] >= x_max - 1.0e-3).astype(np.int64),
            "base": np.flatnonzero(coord[1, :] <= y_min + 1.0e-3).astype(np.int64),
            "head_support": np.flatnonzero(head_mask).astype(np.int64),
        },
    )


def _convert_2d_text_asset(
    asset_dir: Path,
    *,
    loader,
    head_nodes_builder=None,
) -> None:
    mesh = loader("P1", asset_dir)
    coord = np.asarray(mesh.coord, dtype=np.float64)
    elem = np.asarray(mesh.elem, dtype=np.int64)
    surf = np.asarray(mesh.surf[:2, :], dtype=np.int64)
    x_min = float(coord[0, :].min())
    x_max = float(coord[0, :].max())
    y_min = float(coord[1, :].min())
    left = np.flatnonzero(coord[0, :] <= x_min + 0.2).astype(np.int64)
    right = np.flatnonzero(coord[0, :] >= x_max - 0.2).astype(np.int64)
    base = np.flatnonzero(coord[1, :] <= y_min + 0.2).astype(np.int64)
    head = np.empty(0, dtype=np.int64) if head_nodes_builder is None else np.asarray(head_nodes_builder(coord, surf), dtype=np.int64)

    boundary_names: list[str] = []
    for idx in range(surf.shape[1]):
        edge = surf[:, idx]
        edge_set = set(int(v) for v in edge)
        if edge_set <= set(left.tolist()):
            boundary_names.append("left")
        elif edge_set <= set(right.tolist()):
            boundary_names.append("right")
        elif edge_set <= set(base.tolist()):
            boundary_names.append("base")
        elif head.size and edge_set <= set(head.tolist()):
            boundary_names.append("head_support")
        else:
            boundary_names.append("exterior")

    nodesets = {
        "left": left,
        "right": right,
        "base": base,
    }
    if head.size:
        nodesets["head_support"] = head
    region_names = [str(int(mid) + 1) for mid in np.asarray(mesh.material, dtype=np.int64)]
    _write_2d_asset(
        asset_dir / "default.msh",
        coord=coord,
        elem=elem,
        surf=surf,
        region_names=region_names,
        boundary_names=boundary_names,
        nodesets=nodesets,
    )


def convert_2d_kozinec(asset_dir: Path) -> None:
    source_dir = _legacy_source_dir(asset_dir)
    mesh = load_mesh_kozinec_2d("P1", source_dir)
    coord = np.asarray(mesh.coord, dtype=np.float64)
    elem = np.asarray(mesh.elem, dtype=np.int64)
    surf = np.asarray(mesh.surf[:2, :], dtype=np.int64)
    x_min = float(coord[0, :].min())
    x_max = float(coord[0, :].max())
    y_min = float(coord[1, :].min())
    left = np.flatnonzero(coord[0, :] <= x_min + 0.2).astype(np.int64)
    right = np.flatnonzero(coord[0, :] >= x_max - 0.2).astype(np.int64)
    base = np.flatnonzero(coord[1, :] <= y_min + 0.2).astype(np.int64)
    boundary_names: list[str] = []
    for idx in range(surf.shape[1]):
        edge = surf[:, idx]
        edge_set = set(int(v) for v in edge)
        if edge_set <= set(left.tolist()):
            boundary_names.append("left")
        elif edge_set <= set(right.tolist()):
            boundary_names.append("right")
        elif edge_set <= set(base.tolist()):
            boundary_names.append("base")
        else:
            boundary_names.append("exterior")
    _write_2d_asset(
        asset_dir / "default.msh",
        coord=coord,
        elem=elem,
        surf=surf,
        region_names=[f"subdomain_{int(mid) + 1}" for mid in np.asarray(mesh.material, dtype=np.int64)],
        boundary_names=boundary_names,
        nodesets={"left": left, "right": right, "base": base},
    )


def convert_2d_luzec(asset_dir: Path) -> None:
    def _head_nodes(coord: np.ndarray, surf: np.ndarray) -> np.ndarray:
        q_w, _ = luzec_pressure_boundary(coord, surf, 9.81)
        return np.flatnonzero(~q_w).astype(np.int64)

    source_dir = _legacy_source_dir(asset_dir)
    mesh = load_mesh_luzec_2d("P1", source_dir)
    coord = np.asarray(mesh.coord, dtype=np.float64)
    elem = np.asarray(mesh.elem, dtype=np.int64)
    surf = np.asarray(mesh.surf[:2, :], dtype=np.int64)
    x_min = float(coord[0, :].min())
    x_max = float(coord[0, :].max())
    y_min = float(coord[1, :].min())
    left = np.flatnonzero(coord[0, :] <= x_min + 0.2).astype(np.int64)
    right = np.flatnonzero(coord[0, :] >= x_max - 0.2).astype(np.int64)
    base = np.flatnonzero(coord[1, :] <= y_min + 0.2).astype(np.int64)
    head = _head_nodes(coord, surf)
    boundary_names: list[str] = []
    head_set = set(int(v) for v in head)
    left_set = set(int(v) for v in left)
    right_set = set(int(v) for v in right)
    base_set = set(int(v) for v in base)
    for idx in range(surf.shape[1]):
        edge_set = set(int(v) for v in surf[:, idx])
        if edge_set <= left_set:
            boundary_names.append("left")
        elif edge_set <= right_set:
            boundary_names.append("right")
        elif edge_set <= base_set:
            boundary_names.append("base")
        elif edge_set <= head_set:
            boundary_names.append("head_support")
        else:
            boundary_names.append("exterior")
    _write_2d_asset(
        asset_dir / "default.msh",
        coord=coord,
        elem=elem,
        surf=surf,
        region_names=[f"S{int(mid) + 1}" for mid in np.asarray(mesh.material, dtype=np.int64)],
        boundary_names=boundary_names,
        nodesets={"left": left, "right": right, "base": base, "head_support": head},
    )


def convert_2d_franz_dam(asset_dir: Path) -> None:
    source_dir = _legacy_source_dir(asset_dir)
    mesh = load_mesh_franz_dam_2d("P1", source_dir)
    coord = np.asarray(mesh.coord, dtype=np.float64)
    elem = np.asarray(mesh.elem, dtype=np.int64)
    surf = np.asarray(mesh.surf[:2, :], dtype=np.int64)
    x_min = float(coord[0, :].min())
    x_max = float(coord[0, :].max())
    y_min = float(coord[1, :].min())
    left = np.flatnonzero(coord[0, :] <= x_min + 0.2).astype(np.int64)
    right = np.flatnonzero(coord[0, :] >= x_max - 0.2).astype(np.int64)
    base = np.flatnonzero(coord[1, :] <= y_min + 0.2).astype(np.int64)
    q_w, _ = franz_dam_pressure_boundary(coord, surf, 9.81)
    head = np.flatnonzero(~q_w).astype(np.int64)
    boundary_names: list[str] = []
    head_set = set(int(v) for v in head)
    left_set = set(int(v) for v in left)
    right_set = set(int(v) for v in right)
    base_set = set(int(v) for v in base)
    for idx in range(surf.shape[1]):
        edge_set = set(int(v) for v in surf[:, idx])
        if edge_set <= left_set:
            boundary_names.append("left")
        elif edge_set <= right_set:
            boundary_names.append("right")
        elif edge_set <= base_set:
            boundary_names.append("base")
        elif edge_set <= head_set:
            boundary_names.append("head_support")
        else:
            boundary_names.append("exterior")
    _write_2d_asset(
        asset_dir / "default.msh",
        coord=coord,
        elem=elem,
        surf=surf,
        region_names=[f"zone_{int(mid) + 1}" for mid in np.asarray(mesh.material, dtype=np.int64)],
        boundary_names=boundary_names,
        nodesets={"left": left, "right": right, "base": base, "head_support": head},
    )


def _read_legacy_3d(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    msh = meshio.read(path)
    points = np.asarray(msh.points[:, :3], dtype=np.float64).T
    tet, tet_tags = _meshio_cells(msh, "tetra")
    tri, tri_tags = _meshio_cells(msh, "triangle")
    material_map = _legacy_suffix_map(msh.field_data, prefix="material", dim=3)
    boundary_map = _legacy_suffix_map(msh.field_data, prefix="boundary", dim=2)
    mat_suffix = np.asarray([material_map[int(tag)] for tag in tet_tags], dtype=np.int64)
    bnd_suffix = np.asarray([boundary_map[int(tag)] for tag in tri_tags], dtype=np.int64)
    return points, tet, tri, mat_suffix, bnd_suffix, material_map, boundary_map


def _write_3d_asset(
    out_path: Path,
    *,
    coord: np.ndarray,
    tet: np.ndarray,
    tri: np.ndarray,
    region_names: list[str],
    boundary_names: list[str],
    nodesets: dict[str, np.ndarray],
) -> None:
    write_canonical_msh(
        out_path=out_path,
        points=coord.T,
        volume_type="tetra",
        volume_cells=orient_tetrahedra_positive(coord, tet),
        region_by_entity=region_names,
        boundary_type="triangle",
        boundary_cells=np.asarray(tri, dtype=np.int64),
        boundary_by_entity=boundary_names,
        nodesets=nodesets,
    )


def _boundary_nodes_from_suffixes(tri: np.ndarray, boundary_suffix: np.ndarray, suffixes: set[int]) -> np.ndarray:
    mask = np.isin(np.asarray(boundary_suffix, dtype=np.int64), np.asarray(sorted(suffixes), dtype=np.int64))
    return np.unique(np.asarray(tri[:, mask], dtype=np.int64).ravel()).astype(np.int64)


def convert_3d_homo_slope(asset_dir: Path) -> None:
    boundary_name_map = {
        0: "slope_surface",
        1: "x_max",
        2: "x_min",
        3: "z_min",
        4: "z_max",
        5: "base",
        6: "crest",
    }
    source_root = _legacy_source_root(asset_dir)
    if source_root is None:
        source_root = REPO_ROOT / "meshes"
    variants = {
        source_root / "3d_homo_ssr" / "SSR_homo_ada_L1.msh": asset_dir / "adaptive_family_a_l1.msh",
        source_root / "3d_homo_ssr" / "SSR_homo_ada_L2.msh": asset_dir / "adaptive_family_a_l2.msh",
        source_root / "3d_homo_ssr" / "SSR_homo_ada_L3.msh": asset_dir / "adaptive_family_a_l3.msh",
        source_root / "3d_homo_ssr" / "SSR_homo_ada_L4.msh": asset_dir / "adaptive_family_a_l4.msh",
        source_root / "3d_homo_ssr" / "SSR_homo_ada_L5.msh": asset_dir / "adaptive_family_a_l5.msh",
        source_root / "3d_homo_ssr" / "SSR_homo_uni.msh": asset_dir / "uniform_family_a.msh",
        source_root / "3d_homo_ll" / "LL_homo_ada_L1.msh": asset_dir / "adaptive_family_b_l1.msh",
        source_root / "3d_homo_ll" / "LL_homo_ada_L2.msh": asset_dir / "adaptive_family_b_l2.msh",
        source_root / "3d_homo_ll" / "LL_homo_ada_L3.msh": asset_dir / "adaptive_family_b_l3.msh",
        source_root / "3d_homo_ll" / "LL_homo_ada_L4.msh": asset_dir / "adaptive_family_b_l4.msh",
        source_root / "3d_homo_ll" / "LL_homo_ada_L5.msh": asset_dir / "adaptive_family_b_l5.msh",
        source_root / "3d_homo_ll" / "LL_homo_uni.msh": asset_dir / "uniform_family_b.msh",
    }
    for src, dst in variants.items():
        coord, tet, tri, _mat_suffix, boundary_suffix, _mat_map, _bnd_map = _read_legacy_3d(src)
        boundary_names = [boundary_name_map[int(sfx)] for sfx in boundary_suffix]
        nodesets = {
            "x_lock": _boundary_nodes_from_suffixes(tri, boundary_suffix, {1, 2}),
            "z_lock": _boundary_nodes_from_suffixes(tri, boundary_suffix, {3, 4}),
            "base": _boundary_nodes_from_suffixes(tri, boundary_suffix, {5}),
        }
        _write_3d_asset(
            dst,
            coord=coord,
            tet=tet,
            tri=tri,
            region_names=["slope_mass"] * tet.shape[1],
            boundary_names=boundary_names,
            nodesets=nodesets,
        )


def convert_3d_hetero_slope(asset_dir: Path) -> None:
    region_name_map = {
        0: "cover_layer",
        1: "general_foundation",
        2: "weak_foundation",
        3: "slope_mass",
    }
    boundary_name_map = {
        0: "slope_surface",
        1: "x_max",
        2: "x_min",
        3: "z_min",
        4: "z_max",
        5: "base",
        6: "crest",
    }
    source_root = _legacy_source_root(asset_dir)
    if source_root is None:
        source_root = REPO_ROOT / "meshes"
    variants = {
        source_root / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh": asset_dir / "adaptive_family_a_l1.msh",
        source_root / "3d_hetero_ssr" / "SSR_hetero_ada_L2.msh": asset_dir / "adaptive_family_a_l2.msh",
        source_root / "3d_hetero_ssr" / "SSR_hetero_ada_L3.msh": asset_dir / "adaptive_family_a_l3.msh",
        source_root / "3d_hetero_ssr" / "SSR_hetero_ada_L4.msh": asset_dir / "adaptive_family_a_l4.msh",
        source_root / "3d_hetero_ssr" / "SSR_hetero_ada_L5.msh": asset_dir / "adaptive_family_a_l5.msh",
        source_root / "3d_hetero_ssr" / "SSR_hetero_uni.msh": asset_dir / "uniform_family_a.msh",
        source_root / "3d_hetero_ll" / "LL_hetero_ada_L1.msh": asset_dir / "adaptive_family_b_l1.msh",
        source_root / "3d_hetero_ll" / "LL_hetero_ada_L2.msh": asset_dir / "adaptive_family_b_l2.msh",
        source_root / "3d_hetero_ll" / "LL_hetero_ada_L3.msh": asset_dir / "adaptive_family_b_l3.msh",
        source_root / "3d_hetero_ll" / "LL_hetero_ada_L4.msh": asset_dir / "adaptive_family_b_l4.msh",
        source_root / "3d_hetero_ll" / "LL_hetero_ada_L5.msh": asset_dir / "adaptive_family_b_l5.msh",
        source_root / "3d_hetero_ll" / "LL_hetero_uni.msh": asset_dir / "uniform_family_b.msh",
    }
    for src, dst in variants.items():
        coord, tet, tri, mat_suffix, boundary_suffix, _mat_map, _bnd_map = _read_legacy_3d(src)
        _write_3d_asset(
            dst,
            coord=coord,
            tet=tet,
            tri=tri,
            region_names=[region_name_map[int(sfx)] for sfx in mat_suffix],
            boundary_names=[boundary_name_map[int(sfx)] for sfx in boundary_suffix],
            nodesets={
                "x_lock": _boundary_nodes_from_suffixes(tri, boundary_suffix, {1, 2}),
                "z_lock": _boundary_nodes_from_suffixes(tri, boundary_suffix, {3, 4}),
                "base": _boundary_nodes_from_suffixes(tri, boundary_suffix, {5}),
            },
        )


def convert_3d_hetero_seepage(asset_dir: Path) -> None:
    region_name_map = {
        0: "general_foundation",
        1: "weak_foundation",
        2: "slope_mass",
        3: "cover_layer",
    }
    boundary_name_map = {
        5: "surface_a",
        6: "surface_b",
        7: "surface_c",
        8: "surface_d",
        9: "surface_e",
        10: "surface_f",
        11: "surface_g",
        12: "surface_h",
        13: "surface_i",
        14: "surface_j",
    }
    rename = {
        "slope_with_waterlevels.msh": "family_a.msh",
        "slope_with_waterlevels2.msh": "family_b.msh",
        "slope_with_waterlevels3.msh": "family_c.msh",
        "slope_with_waterlevels4.msh": "family_d.msh",
        "slope_with_waterlevels5.msh": "family_e.msh",
        "slope_with_waterlevels6.msh": "family_f.msh",
        "slope_with_waterlevels_concave.msh": "concave_family_a.msh",
        "slope_with_waterlevels_concave_L2.msh": "concave_family_b.msh",
        "slope_with_waterlevels_concave_L3.msh": "concave_family_c.msh",
        "slope_with_waterlevels_concave_L4.msh": "concave_family_d.msh",
    }
    source_dir = _legacy_source_dir(asset_dir)
    for old_name, new_name in rename.items():
        coord, tet, tri, mat_suffix, boundary_suffix, _mat_map, _bnd_map = _read_legacy_3d(source_dir / old_name)
        _write_3d_asset(
            asset_dir / new_name,
            coord=coord,
            tet=tet,
            tri=tri,
            region_names=[region_name_map[int(sfx)] for sfx in mat_suffix],
            boundary_names=[boundary_name_map[int(sfx)] for sfx in boundary_suffix],
            nodesets={
                "x_lock": _boundary_nodes_from_suffixes(tri, boundary_suffix, {7, 8, 9}),
                "y_lateral_lock": _boundary_nodes_from_suffixes(tri, boundary_suffix, {10, 11}),
                "base": _boundary_nodes_from_suffixes(tri, boundary_suffix, {12}),
                "head_dry": _boundary_nodes_from_suffixes(tri, boundary_suffix, {5, 7, 13}),
                "head_porous": _boundary_nodes_from_suffixes(tri, boundary_suffix, {8}),
                "head_free": _boundary_nodes_from_suffixes(tri, boundary_suffix, {6, 9, 14}),
            },
        )


def convert_3d_hetero_seepage_transition(asset_dir: Path) -> None:
    region_name_map = {
        0: "cover_layer",
        1: "general_foundation",
        2: "weak_foundation",
        3: "slope_mass",
    }
    boundary_name_map = {
        0: "surface_main",
        1: "surface_xmin",
        2: "surface_xmax",
        3: "surface_zmin",
        4: "surface_zmax",
        5: "base",
        6: "crest",
    }
    coord, tet, tri, mat_suffix, boundary_suffix, _mat_map, _bnd_map = _read_legacy_3d(
        ((_legacy_source_root(asset_dir) or REPO_ROOT / "meshes") / "3d_hetero_seepage_ssr_comsol" / "comsol_mesh.msh")
    )
    seepage = SeepageDefinition(
        water_unit_weight=9.81,
        conductivity_mode="uniform",
        conductivity=(1.0,),
        water_levels={"free": 35.0, "porous": 55.0},
        hydraulic_boundaries={
            "mode": "hybrid_transition",
            "dry_labels": [6],
            "porous_labels": [2],
            "free_labels": [1],
            "geometry_recipe": {
                "base_point": [55.0, 30.0, 0.0],
                "toe_point": [115.0, 60.0, 0.0],
                "apex_left": [30.0, 30.0, 43.3],
                "apex_right": [30.0, 30.0, -43.3],
                "bed_y": 30.0,
                "triangle_normal_tolerance": 1.0e-1,
                "plane_distance_tolerance": 1.0e-6,
                "bed_tolerance": 1.0e-1,
                "sector_tolerance": 1.0e-10,
            },
        },
    )
    q_w, pw_d = build_seepage_boundary(
        coord=coord,
        surf=tri,
        triangle_labels=boundary_suffix,
        seepage=seepage,
        asset_dir=asset_dir,
        grho=9.81,
    )
    dry = np.flatnonzero((~q_w) & np.isclose(pw_d, 0.0)).astype(np.int64)
    porous = np.flatnonzero((~q_w) & np.isclose(pw_d, 9.81 * (55.0 - coord[1, :]), atol=1.0e-8)).astype(np.int64)
    free = np.flatnonzero((~q_w) & np.isclose(pw_d, 9.81 * (35.0 - coord[1, :]), atol=1.0e-8) & ~np.isin(np.arange(coord.shape[1]), porous)).astype(np.int64)
    _write_3d_asset(
        asset_dir / "transition_default.msh",
        coord=coord,
        tet=tet,
        tri=tri,
        region_names=[region_name_map[int(sfx)] for sfx in mat_suffix],
        boundary_names=[boundary_name_map[int(sfx)] for sfx in boundary_suffix],
        nodesets={
            "x_lock": _boundary_nodes_from_suffixes(tri, boundary_suffix, {1, 2}),
            "z_lock": _boundary_nodes_from_suffixes(tri, boundary_suffix, {3, 4}),
            "base": _boundary_nodes_from_suffixes(tri, boundary_suffix, {5}),
            "head_dry": dry,
            "head_porous": porous,
            "head_free": free,
        },
    )


def convert_3d_siopt(asset_dir: Path) -> None:
    boundary_name_map = {
        0: "slope_surface",
        1: "x_max",
        2: "x_min",
        3: "z_min",
        4: "z_max",
        5: "base",
        6: "crest",
    }
    rename = {
        "SIOPT_L0.msh": "reference_l0.msh",
        "SIOPT_L1.msh": "reference_l1.msh",
        "SIOPT_L5.msh": "reference_l5.msh",
    }
    source_dir = _legacy_source_dir(asset_dir)
    for old_name, new_name in rename.items():
        coord, tet, tri, _mat_suffix, boundary_suffix, _mat_map, _bnd_map = _read_legacy_3d(source_dir / old_name)
        _write_3d_asset(
            asset_dir / new_name,
            coord=coord,
            tet=tet,
            tri=tri,
            region_names=["reference_mass"] * tet.shape[1],
            boundary_names=[boundary_name_map[int(sfx)] for sfx in boundary_suffix],
            nodesets={
                "x_lock": _boundary_nodes_from_suffixes(tri, boundary_suffix, {1}),
                "z_lock": _boundary_nodes_from_suffixes(tri, boundary_suffix, {3}),
                "base": _boundary_nodes_from_suffixes(tri, boundary_suffix, {5}),
            },
        )
