"""Distributed local-overlap assembly for the standalone elastic operator."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

from ..mesh.materials import MaterialSpec, heterogenous_materials
from ..utils import flatten_field, owned_block_range
from .assembly import assemble_strain_operator, build_elastic_stiffness_matrix, vector_volume


@dataclass(frozen=True)
class OwnedElasticRows:
    """Owned global rows assembled from a local overlap submesh."""

    owned_node_range: tuple[int, int]
    owned_row_range: tuple[int, int]
    overlap_nodes: np.ndarray
    overlap_elements: np.ndarray
    local_matrix: csr_matrix
    local_rhs: np.ndarray
    timings: dict[str, float]


def _global_dofs_for_nodes(nodes: np.ndarray, dim: int) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.int64).ravel()
    if nodes.size == 0:
        return np.empty(0, dtype=np.int64)
    return (dim * np.repeat(nodes, dim) + np.tile(np.arange(dim, dtype=np.int64), nodes.size)).astype(np.int64)


def find_overlap_partition(elem: np.ndarray, owned_node_range: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes/elements required to assemble the owned node block."""

    elem = np.asarray(elem, dtype=np.int64)
    node0, node1 = map(int, owned_node_range)
    if node1 < node0:
        raise ValueError("owned_node_range must satisfy end >= start")
    if elem.ndim != 2:
        raise ValueError("elem must be (nodes_per_element, n_elem)")

    touches_owned = np.any((elem >= node0) & (elem < node1), axis=0)
    overlap_elements = np.flatnonzero(touches_owned).astype(np.int64)
    if overlap_elements.size == 0:
        return np.empty(0, dtype=np.int64), overlap_elements
    overlap_nodes = np.unique(elem[:, overlap_elements].reshape(-1, order="F")).astype(np.int64)
    return overlap_nodes, overlap_elements


def assemble_owned_elastic_rows(
    coord: np.ndarray,
    elem: np.ndarray,
    q_mask: np.ndarray,
    material_identifier: np.ndarray,
    materials: list[MaterialSpec] | list[dict] | dict,
    owned_node_range: tuple[int, int],
    *,
    elem_type: str = "P2",
) -> OwnedElasticRows:
    """Assemble the full-system elastic operator rows owned by one rank.

    The global mesh is assumed to be already reordered. Only elements touching
    the owned node range are assembled locally, then the resulting overlap
    operator is restricted to the owned global rows.
    """

    coord = np.asarray(coord, dtype=np.float64)
    elem = np.asarray(elem, dtype=np.int64)
    q_mask = np.asarray(q_mask, dtype=bool)
    material_identifier = np.asarray(material_identifier, dtype=np.int64).ravel()

    dim, n_nodes = coord.shape
    node0, node1 = map(int, owned_node_range)
    if node0 < 0 or node1 > n_nodes:
        raise ValueError("owned_node_range is outside the reordered mesh")
    if q_mask.shape != (dim, n_nodes):
        raise ValueError("q_mask shape must match coord")
    if material_identifier.size != elem.shape[1]:
        raise ValueError("material_identifier must have one entry per element")

    row0 = dim * node0
    row1 = dim * node1
    local_rows = row1 - row0
    global_dofs = dim * n_nodes
    free_mask = q_mask.reshape(-1, order="F")

    t_overlap = perf_counter()
    overlap_nodes, overlap_elements = find_overlap_partition(elem, owned_node_range)
    overlap_time = perf_counter() - t_overlap

    if overlap_nodes.size == 0:
        return OwnedElasticRows(
            owned_node_range=(node0, node1),
            owned_row_range=(row0, row1),
            overlap_nodes=overlap_nodes,
            overlap_elements=overlap_elements,
            local_matrix=csr_matrix((local_rows, global_dofs), dtype=np.float64),
            local_rhs=np.zeros(local_rows, dtype=np.float64),
            timings={
                "local_overlap_s": overlap_time,
                "local_assembly_s": 0.0,
                "local_bc_s": 0.0,
            },
        )

    t_assembly = perf_counter()
    node_lids = np.full(n_nodes, -1, dtype=np.int64)
    node_lids[overlap_nodes] = np.arange(overlap_nodes.size, dtype=np.int64)

    coord_overlap = coord[:, overlap_nodes]
    elem_overlap = node_lids[elem[:, overlap_elements]]
    asm = assemble_strain_operator(coord_overlap, elem_overlap, elem_type, dim=dim)

    _c0, _phi, _psi, shear, bulk, lame, gamma = heterogenous_materials(
        material_identifier[overlap_elements],
        np.ones(asm.n_int, dtype=bool),
        asm.n_q,
        materials,
    )
    K_overlap, weight, _B = build_elastic_stiffness_matrix(asm, shear, lame, bulk)

    f_v_int = np.zeros((dim, asm.n_int), dtype=np.float64)
    if dim >= 2:
        f_v_int[1, :] = -gamma.astype(np.float64)
    f_overlap = flatten_field(vector_volume(asm, f_v_int, weight))
    assembly_time = perf_counter() - t_assembly

    overlap_global_dofs = _global_dofs_for_nodes(overlap_nodes, dim)
    active_overlap = free_mask[overlap_global_dofs].astype(np.float64, copy=False)
    K_overlap = (K_overlap @ diags(active_overlap, format="csr")).tocsr()

    owned_nodes = np.arange(node0, node1, dtype=np.int64)
    owned_local_nodes = node_lids[owned_nodes]
    if np.any(owned_local_nodes < 0):
        raise RuntimeError("Owned nodes must be present in the overlap submesh")
    owned_local_dofs = _global_dofs_for_nodes(owned_local_nodes, dim)
    owned_global_rows = np.arange(row0, row1, dtype=np.int64)

    owned_rows = K_overlap[owned_local_dofs, :].tocoo()
    local_matrix = coo_matrix(
        (
            owned_rows.data,
            (
                owned_rows.row,
                overlap_global_dofs[owned_rows.col],
            ),
        ),
        shape=(local_rows, global_dofs),
    ).tocsr()
    local_rhs = np.asarray(f_overlap[owned_local_dofs], dtype=np.float64).copy()

    t_bc = perf_counter()
    owned_free = free_mask[owned_global_rows]
    local_rhs[~owned_free] = 0.0
    if np.any(~owned_free):
        local_matrix = local_matrix.tolil()
        constrained_local = np.flatnonzero(~owned_free)
        constrained_global = owned_global_rows[constrained_local]
        for local_row, global_row in zip(constrained_local.tolist(), constrained_global.tolist(), strict=False):
            local_matrix.rows[local_row] = [int(global_row)]
            local_matrix.data[local_row] = [1.0]
        local_matrix = local_matrix.tocsr()
    bc_time = perf_counter() - t_bc

    return OwnedElasticRows(
        owned_node_range=(node0, node1),
        owned_row_range=(row0, row1),
        overlap_nodes=overlap_nodes,
        overlap_elements=overlap_elements,
        local_matrix=local_matrix,
        local_rhs=local_rhs,
        timings={
            "local_overlap_s": overlap_time,
            "local_assembly_s": assembly_time,
            "local_bc_s": bc_time,
        },
    )


def assemble_owned_elastic_rows_for_comm(
    coord: np.ndarray,
    elem: np.ndarray,
    q_mask: np.ndarray,
    material_identifier: np.ndarray,
    materials: list[MaterialSpec] | list[dict] | dict,
    comm,
    *,
    elem_type: str = "P2",
) -> OwnedElasticRows:
    """Assemble rows owned by the calling rank for block-aligned PETSc ownership."""

    dim, n_nodes = np.asarray(coord).shape
    row0, row1 = owned_block_range(n_nodes, dim, comm)
    return assemble_owned_elastic_rows(
        coord,
        elem,
        q_mask,
        material_identifier,
        materials,
        (row0 // dim, row1 // dim),
        elem_type=elem_type,
    )
