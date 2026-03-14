"""Mesh node reordering helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

try:
    import pymetis
except Exception:  # pragma: no cover - optional dependency
    pymetis = None


@dataclass(frozen=True)
class ReorderedMesh:
    coord: np.ndarray
    elem: np.ndarray
    surf: np.ndarray
    q_mask: np.ndarray
    permutation: np.ndarray
    inverse_permutation: np.ndarray


def _part1by2_64(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.uint64) & np.uint64(0x1FFFFF)
    x = (x | (x << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
    return x


def _morton_order(coord: np.ndarray) -> np.ndarray:
    dim, n_nodes = coord.shape
    if dim != 3:
        raise ValueError("Morton reordering currently expects 3D coordinates")
    xyz = np.asarray(coord, dtype=np.float64)
    mins = xyz.min(axis=1, keepdims=True)
    spans = np.maximum(xyz.max(axis=1, keepdims=True) - mins, 1.0e-12)
    scaled = np.floor((xyz - mins) / spans * ((1 << 21) - 1)).astype(np.uint64)
    codes = _part1by2_64(scaled[0]) | (_part1by2_64(scaled[1]) << np.uint64(1)) | (_part1by2_64(scaled[2]) << np.uint64(2))
    return np.argsort(codes, kind="stable")[:n_nodes].astype(np.int64)


def _xyz_order(coord: np.ndarray) -> np.ndarray:
    xyz = np.asarray(coord, dtype=np.float64)
    if xyz.ndim != 2:
        raise ValueError("Coordinate array must be 2D")
    if xyz.shape[0] == 2:
        keys = (xyz[1], xyz[0])
    elif xyz.shape[0] == 3:
        keys = (xyz[2], xyz[1], xyz[0])
    else:
        raise ValueError("XYZ reordering expects 2D or 3D coordinates")
    return np.lexsort(keys).astype(np.int64)


def _nodal_adjacency(elem: np.ndarray, n_nodes: int):
    conn = np.asarray(elem, dtype=np.int64).T
    rows = []
    cols = []
    for local in conn:
        rr = np.repeat(local, local.size)
        cc = np.tile(local, local.size)
        mask = rr != cc
        rows.append(rr[mask])
        cols.append(cc[mask])
    row = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    col = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    data = np.ones(row.size, dtype=np.int8)
    graph = coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes)).tocsr()
    graph.sum_duplicates()
    graph.data[:] = 1
    return graph


def _rcm_order(elem: np.ndarray, n_nodes: int) -> np.ndarray:
    return reverse_cuthill_mckee(_nodal_adjacency(elem, n_nodes), symmetric_mode=True).astype(np.int64)


def _block_metis_order(coord: np.ndarray, elem: np.ndarray, n_nodes: int, n_parts: int | None) -> np.ndarray:
    if n_parts is None or int(n_parts) <= 1:
        return _xyz_order(coord)
    if pymetis is None:
        raise RuntimeError("block_metis ordering requires the optional 'pymetis' dependency")

    graph = _nodal_adjacency(elem, n_nodes)
    _edgecuts, membership = pymetis.part_graph(
        int(n_parts),
        adjacency=pymetis.CSRAdjacency(
            np.asarray(graph.indptr, dtype=np.int32),
            np.asarray(graph.indices, dtype=np.int32),
        ),
    )
    membership_arr = np.asarray(membership, dtype=np.int64)
    xyz_rank = np.empty(n_nodes, dtype=np.int64)
    xyz_rank[_xyz_order(coord)] = np.arange(n_nodes, dtype=np.int64)
    return np.lexsort((xyz_rank, membership_arr)).astype(np.int64)


def compute_node_permutation(coord: np.ndarray, elem: np.ndarray, strategy: str, *, n_parts: int | None = None) -> np.ndarray:
    strategy = str(strategy).lower()
    n_nodes = int(coord.shape[1])
    if strategy in {"none", "original", "identity"}:
        return np.arange(n_nodes, dtype=np.int64)
    if strategy in {"xyz", "block_xyz", "lexicographic_xyz"}:
        return _xyz_order(coord)
    if strategy == "block_metis":
        return _block_metis_order(coord, elem, n_nodes, n_parts)
    if strategy == "morton":
        return _morton_order(coord)
    if strategy in {"rcm", "block_rcm"}:
        return _rcm_order(elem, n_nodes)
    raise ValueError(f"Unsupported node-ordering strategy {strategy!r}")


def reorder_mesh_nodes(
    coord: np.ndarray,
    elem: np.ndarray,
    surf: np.ndarray,
    q_mask: np.ndarray,
    *,
    strategy: str = "original",
    n_parts: int | None = None,
) -> ReorderedMesh:
    perm = compute_node_permutation(coord, elem, strategy, n_parts=n_parts)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(perm.size, dtype=np.int64)

    coord_new = np.asarray(coord, dtype=np.float64)[:, perm]
    q_new = np.asarray(q_mask, dtype=bool)[:, perm]
    elem_new = inv_perm[np.asarray(elem, dtype=np.int64)]
    surf_new = inv_perm[np.asarray(surf, dtype=np.int64)]

    return ReorderedMesh(
        coord=coord_new,
        elem=elem_new,
        surf=surf_new,
        q_mask=q_new,
        permutation=perm,
        inverse_permutation=inv_perm,
    )
