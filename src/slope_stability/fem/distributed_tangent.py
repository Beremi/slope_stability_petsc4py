"""Ownership-aligned local tangent assembly on a fixed CSR pattern."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from ..mesh.materials import MaterialSpec
from .assembly import assemble_strain_operator
from .distributed_elastic import OwnedElasticRows, assemble_owned_elastic_rows

try:  # pragma: no cover - compiled extension is optional during tests
    from slope_stability import _kernels
except Exception:  # pragma: no cover
    _kernels = None


@dataclass(frozen=True)
class OwnedTangentPattern:
    """Fixed owned-row tangent sparsity and local assembly metadata."""

    dim: int
    n_strain: int
    owned_node_range: tuple[int, int]
    owned_row_range: tuple[int, int]
    overlap_nodes: np.ndarray
    overlap_elements: np.ndarray
    overlap_global_dofs: np.ndarray
    overlap_B: csr_matrix
    owned_local_overlap_dofs: np.ndarray
    owned_free_mask: np.ndarray
    owned_free_local_rows: np.ndarray
    global_free_size: int
    local_matrix_pattern: csr_matrix
    elastic_values: np.ndarray
    overlap_assembly_weight: np.ndarray
    dphi1: np.ndarray
    dphi2: np.ndarray
    dphi3: np.ndarray
    local_int_indices: np.ndarray
    unique_nodes: np.ndarray
    unique_elements: np.ndarray
    unique_global_dofs: np.ndarray
    unique_B: csr_matrix
    unique_local_int_indices: np.ndarray
    scatter_map: np.ndarray
    constrained_diag_positions: np.ndarray
    n_p: int
    n_q: int
    timings: dict[str, float]


def _global_dofs_for_nodes(nodes: np.ndarray, dim: int) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.int64).ravel()
    if nodes.size == 0:
        return np.empty(0, dtype=np.int64)
    return (dim * np.repeat(nodes, dim) + np.tile(np.arange(dim, dtype=np.int64), nodes.size)).astype(np.int64)


def _build_scatter_map(
    *,
    elem: np.ndarray,
    overlap_elements: np.ndarray,
    q_mask: np.ndarray,
    row0: int,
    local_matrix: csr_matrix,
) -> tuple[np.ndarray, np.ndarray, int]:
    dim = int(np.asarray(q_mask).shape[0])
    free_mask = np.asarray(q_mask, dtype=bool).reshape(-1, order="F")
    owned_global_rows = np.arange(row0, row0 + local_matrix.shape[0], dtype=np.int64)
    owned_free = free_mask[owned_global_rows]
    n_local_rows = int(local_matrix.shape[0])
    row_maps: list[dict[int, int]] = []
    constrained_diag_positions: list[int] = []

    for local_row in range(n_local_rows):
        start = int(local_matrix.indptr[local_row])
        end = int(local_matrix.indptr[local_row + 1])
        cols = local_matrix.indices[start:end]
        positions = np.arange(start, end, dtype=np.int64)
        row_maps.append({int(col): int(pos) for col, pos in zip(cols.tolist(), positions.tolist(), strict=False)})
        if not owned_free[local_row]:
            diag_pos = row_maps[-1].get(int(owned_global_rows[local_row]))
            if diag_pos is None:
                raise RuntimeError("Constrained local row is missing its diagonal entry")
            constrained_diag_positions.append(int(diag_pos))

    n_p = int(elem.shape[0])
    n_local_dof = dim * n_p
    n_ld2 = n_local_dof * n_local_dof
    scatter = np.full((int(overlap_elements.size), n_ld2), -1, dtype=np.int64)
    missing = 0

    for e_local, e_global in enumerate(np.asarray(overlap_elements, dtype=np.int64).tolist()):
        elem_nodes = np.asarray(elem[:, int(e_global)], dtype=np.int64)
        dofs = _global_dofs_for_nodes(elem_nodes, dim)
        for a, row_global in enumerate(dofs.tolist()):
            if row_global < row0 or row_global >= row0 + n_local_rows:
                continue
            local_row = int(row_global - row0)
            if not owned_free[local_row]:
                continue
            row_lookup = row_maps[local_row]
            base = a * n_local_dof
            for b, col_global in enumerate(dofs.tolist()):
                if not free_mask[int(col_global)]:
                    continue
                pos = row_lookup.get(int(col_global))
                if pos is None:
                    missing += 1
                    continue
                scatter[e_local, base + b] = int(pos)

    return scatter, np.asarray(constrained_diag_positions, dtype=np.int64), missing


def _build_owned_row_structural_pattern(
    *,
    elem: np.ndarray,
    overlap_elements: np.ndarray,
    q_mask: np.ndarray,
    owned_row_range: tuple[int, int],
) -> csr_matrix:
    dim, n_nodes = np.asarray(q_mask).shape
    row0, row1 = (int(owned_row_range[0]), int(owned_row_range[1]))
    n_local_rows = row1 - row0
    global_dofs = dim * n_nodes
    free_mask = np.asarray(q_mask, dtype=bool).reshape(-1, order="F")
    row_cols: list[set[int]] = [set() for _ in range(n_local_rows)]

    for e_global in np.asarray(overlap_elements, dtype=np.int64).tolist():
        dofs = _global_dofs_for_nodes(np.asarray(elem[:, int(e_global)], dtype=np.int64), dim)
        dof_list = [int(v) for v in dofs.tolist()]
        free_cols = [int(v) for v in dof_list if free_mask[v]]
        for row_global in dof_list:
            if row_global < row0 or row_global >= row1:
                continue
            local_row = row_global - row0
            if free_mask[row_global]:
                row_cols[local_row].update(free_cols)

    owned_rows = np.arange(row0, row1, dtype=np.int64)
    for local_row, row_global in enumerate(owned_rows.tolist()):
        if not free_mask[row_global]:
            row_cols[local_row] = {int(row_global)}

    indptr = np.zeros(n_local_rows + 1, dtype=np.int32)
    indices_chunks: list[np.ndarray] = []
    for local_row, cols in enumerate(row_cols):
        cols_arr = np.asarray(sorted(cols), dtype=np.int32)
        indices_chunks.append(cols_arr)
        indptr[local_row + 1] = indptr[local_row] + cols_arr.size
    indices = np.concatenate(indices_chunks) if indices_chunks else np.empty(0, dtype=np.int32)
    data = np.ones(indices.size, dtype=np.float64)
    return csr_matrix((data, indices, indptr), shape=(n_local_rows, global_dofs))


def _project_values_onto_pattern(source: csr_matrix, pattern: csr_matrix) -> np.ndarray:
    values = np.zeros(pattern.nnz, dtype=np.float64)
    for local_row in range(pattern.shape[0]):
        p0 = int(pattern.indptr[local_row])
        p1 = int(pattern.indptr[local_row + 1])
        target_lookup = {int(col): int(pos) for pos, col in enumerate(pattern.indices[p0:p1], start=p0)}
        s0 = int(source.indptr[local_row])
        s1 = int(source.indptr[local_row + 1])
        for col, val in zip(source.indices[s0:s1].tolist(), source.data[s0:s1].tolist(), strict=False):
            pos = target_lookup.get(int(col))
            if pos is not None:
                values[pos] = float(val)
    return values


def prepare_owned_tangent_pattern(
    coord: np.ndarray,
    elem: np.ndarray,
    q_mask: np.ndarray,
    material_identifier: np.ndarray,
    materials: list[MaterialSpec] | list[dict] | dict,
    owned_node_range: tuple[int, int],
    *,
    elem_type: str = "P2",
    include_unique: bool = True,
) -> OwnedTangentPattern:
    """Prepare the fixed owned-row pattern for repeated tangent assembly."""

    t0 = perf_counter()
    elastic_rows: OwnedElasticRows = assemble_owned_elastic_rows(
        coord,
        elem,
        q_mask,
        material_identifier,
        materials,
        owned_node_range,
        elem_type=elem_type,
    )
    t_elastic = perf_counter() - t0

    overlap_nodes = np.asarray(elastic_rows.overlap_nodes, dtype=np.int64)
    overlap_elements = np.asarray(elastic_rows.overlap_elements, dtype=np.int64)
    dim = int(np.asarray(coord).shape[0])
    node0, node1 = tuple(int(v) for v in elastic_rows.owned_node_range)
    row0, row1 = tuple(int(v) for v in elastic_rows.owned_row_range)
    owned_global_rows = np.arange(row0, row1, dtype=np.int64)
    owned_free_mask = np.asarray(q_mask, dtype=bool).reshape(-1, order="F")[owned_global_rows]
    owned_free_local_rows = np.flatnonzero(owned_free_mask).astype(np.int64)
    global_free_size = int(np.count_nonzero(np.asarray(q_mask, dtype=bool)))

    t_overlap = perf_counter()
    node_lids = np.full(int(np.asarray(coord).shape[1]), -1, dtype=np.int64)
    node_lids[overlap_nodes] = np.arange(overlap_nodes.size, dtype=np.int64)
    coord_overlap = np.asarray(coord, dtype=np.float64)[:, overlap_nodes]
    elem_overlap = node_lids[np.asarray(elem, dtype=np.int64)[:, overlap_elements]]
    overlap_asm = assemble_strain_operator(coord_overlap, elem_overlap, elem_type, dim=dim)
    overlap_global_dofs = _global_dofs_for_nodes(overlap_nodes, dim)
    owned_local_nodes = node_lids[np.arange(node0, node1, dtype=np.int64)]
    if np.any(owned_local_nodes < 0):
        raise RuntimeError("Owned nodes must be present in the overlap submesh")
    owned_local_overlap_dofs = _global_dofs_for_nodes(owned_local_nodes, dim)
    overlap_time = perf_counter() - t_overlap

    t_unique = perf_counter()
    if include_unique:
        elem_owner_nodes = np.min(np.asarray(elem, dtype=np.int64), axis=0)
        unique_elements = np.flatnonzero((elem_owner_nodes >= node0) & (elem_owner_nodes < node1)).astype(np.int64)
        if unique_elements.size:
            unique_nodes = np.unique(np.asarray(elem, dtype=np.int64)[:, unique_elements].reshape(-1, order="F")).astype(np.int64)
            unique_node_lids = np.full(int(np.asarray(coord).shape[1]), -1, dtype=np.int64)
            unique_node_lids[unique_nodes] = np.arange(unique_nodes.size, dtype=np.int64)
            coord_unique = np.asarray(coord, dtype=np.float64)[:, unique_nodes]
            elem_unique = unique_node_lids[np.asarray(elem, dtype=np.int64)[:, unique_elements]]
            unique_asm = assemble_strain_operator(coord_unique, elem_unique, elem_type, dim=dim)
            unique_global_dofs = _global_dofs_for_nodes(unique_nodes, dim)
            unique_local_int_indices = (
                unique_elements[:, None] * overlap_asm.n_q + np.arange(overlap_asm.n_q, dtype=np.int64)[None, :]
            ).reshape(-1)
            unique_B = unique_asm.B.tocsr()
        else:
            unique_nodes = np.empty(0, dtype=np.int64)
            unique_global_dofs = np.empty(0, dtype=np.int64)
            unique_local_int_indices = np.empty(0, dtype=np.int64)
            unique_B = csr_matrix((0, 0), dtype=np.float64)
    else:
        unique_nodes = np.empty(0, dtype=np.int64)
        unique_global_dofs = np.empty(0, dtype=np.int64)
        unique_local_int_indices = np.empty(0, dtype=np.int64)
        unique_elements = np.empty(0, dtype=np.int64)
        unique_B = csr_matrix((0, 0), dtype=np.float64)
    unique_time = perf_counter() - t_unique

    t_pattern = perf_counter()
    pattern_matrix = _build_owned_row_structural_pattern(
        elem=np.asarray(elem, dtype=np.int64),
        overlap_elements=overlap_elements,
        q_mask=np.asarray(q_mask, dtype=bool),
        owned_row_range=elastic_rows.owned_row_range,
    )
    elastic_values = _project_values_onto_pattern(elastic_rows.local_matrix, pattern_matrix)
    pattern_time = perf_counter() - t_pattern

    t_scatter = perf_counter()
    scatter_map, constrained_diag_positions, missing = _build_scatter_map(
        elem=np.asarray(elem, dtype=np.int64),
        overlap_elements=overlap_elements,
        q_mask=np.asarray(q_mask, dtype=bool),
        row0=int(elastic_rows.owned_row_range[0]),
        local_matrix=pattern_matrix,
    )
    scatter_time = perf_counter() - t_scatter

    n_q = int(overlap_asm.n_q)
    local_int_indices = (
        overlap_elements[:, None] * n_q + np.arange(n_q, dtype=np.int64)[None, :]
    ).reshape(-1)

    return OwnedTangentPattern(
        dim=dim,
        n_strain=int(overlap_asm.n_strain),
        owned_node_range=tuple(int(v) for v in elastic_rows.owned_node_range),
        owned_row_range=tuple(int(v) for v in elastic_rows.owned_row_range),
        overlap_nodes=overlap_nodes,
        overlap_elements=overlap_elements,
        overlap_global_dofs=np.ascontiguousarray(overlap_global_dofs, dtype=np.int64),
        overlap_B=overlap_asm.B.tocsr(),
        owned_local_overlap_dofs=np.ascontiguousarray(owned_local_overlap_dofs, dtype=np.int64),
        owned_free_mask=np.ascontiguousarray(owned_free_mask, dtype=bool),
        owned_free_local_rows=np.ascontiguousarray(owned_free_local_rows, dtype=np.int64),
        global_free_size=global_free_size,
        local_matrix_pattern=pattern_matrix,
        elastic_values=elastic_values,
        overlap_assembly_weight=np.ascontiguousarray(overlap_asm.weight, dtype=np.float64),
        dphi1=np.ascontiguousarray(overlap_asm.dphi["dphi1"].T, dtype=np.float64),
        dphi2=np.ascontiguousarray(overlap_asm.dphi["dphi2"].T, dtype=np.float64),
        dphi3=np.ascontiguousarray(overlap_asm.dphi.get("dphi3", np.empty((0, 0), dtype=np.float64)).T, dtype=np.float64),
        local_int_indices=np.ascontiguousarray(local_int_indices, dtype=np.int64),
        unique_nodes=np.ascontiguousarray(unique_nodes, dtype=np.int64),
        unique_elements=np.ascontiguousarray(unique_elements, dtype=np.int64),
        unique_global_dofs=np.ascontiguousarray(unique_global_dofs, dtype=np.int64),
        unique_B=unique_B,
        unique_local_int_indices=np.ascontiguousarray(unique_local_int_indices, dtype=np.int64),
        scatter_map=np.ascontiguousarray(scatter_map, dtype=np.int64),
        constrained_diag_positions=constrained_diag_positions,
        n_p=int(elem.shape[0]),
        n_q=n_q,
        timings={
            "elastic_pattern_s": float(t_elastic),
            "overlap_geometry_s": float(overlap_time),
            "unique_geometry_s": float(unique_time),
            "structural_pattern_s": float(pattern_time),
            "scatter_map_s": float(scatter_time),
            "scatter_missing_entries": float(missing),
        },
    )


def _assemble_owned_tangent_values_python(pattern: OwnedTangentPattern, ds_local: np.ndarray) -> np.ndarray:
    n_elem = int(pattern.scatter_map.shape[0])
    n_p = int(pattern.n_p)
    n_q = int(pattern.n_q)
    dim = int(pattern.dim)
    n_strain = int(pattern.n_strain)
    n_local_dof = dim * n_p
    out = np.zeros_like(pattern.elastic_values)

    for e in range(n_elem):
        ke = np.zeros((n_local_dof, n_local_dof), dtype=np.float64)
        g_base = e * n_q
        for q in range(n_q):
            g = g_base + q
            d_eq = ds_local[g].reshape(n_strain, n_strain, order="F") * pattern.overlap_assembly_weight[g]
            b_eq = np.zeros((n_strain, n_local_dof), dtype=np.float64)
            for i in range(n_p):
                dN1 = pattern.dphi1[g, i]
                dN2 = pattern.dphi2[g, i]
                c = dim * i
                if dim == 2:
                    b_eq[0, c + 0] = dN1
                    b_eq[1, c + 1] = dN2
                    b_eq[2, c + 0] = dN2
                    b_eq[2, c + 1] = dN1
                elif dim == 3:
                    dN3 = pattern.dphi3[g, i]
                    b_eq[0, c + 0] = dN1
                    b_eq[1, c + 1] = dN2
                    b_eq[2, c + 2] = dN3
                    b_eq[3, c + 0] = dN2
                    b_eq[3, c + 1] = dN1
                    b_eq[4, c + 1] = dN3
                    b_eq[4, c + 2] = dN2
                    b_eq[5, c + 0] = dN3
                    b_eq[5, c + 2] = dN1
                else:
                    raise ValueError(f"Unsupported dimension {dim}")
            ke += b_eq.T @ d_eq @ b_eq

        smap = pattern.scatter_map[e]
        active = smap >= 0
        out[smap[active]] += ke.reshape(-1, order="C")[active]

    if pattern.constrained_diag_positions.size:
        out[pattern.constrained_diag_positions] = 1.0
    return out


def assemble_owned_tangent_values(
    pattern: OwnedTangentPattern,
    DS: np.ndarray,
    *,
    use_compiled: bool = True,
) -> np.ndarray:
    """Assemble tangent values on the fixed owned-row CSR pattern."""

    ds_global = np.asarray(DS, dtype=np.float64)
    expected_rows = int(pattern.n_strain * pattern.n_strain)
    if ds_global.ndim != 2 or ds_global.shape[0] != expected_rows:
        raise ValueError(f"DS must have shape ({expected_rows}, n_int) for owned tangent assembly")
    if ds_global.shape[1] == pattern.local_int_indices.size:
        ds_local = np.ascontiguousarray(ds_global.T, dtype=np.float64)
    else:
        ds_local = np.ascontiguousarray(ds_global[:, pattern.local_int_indices].T, dtype=np.float64)

    if use_compiled and _kernels is not None and int(pattern.dim) == 3 and int(pattern.n_strain) == 6:
        values = _kernels.assemble_tangent_values_3d(
            pattern.dphi1,
            pattern.dphi2,
            pattern.dphi3,
            ds_local,
            pattern.overlap_assembly_weight,
            pattern.scatter_map,
            int(pattern.elastic_values.size),
        )
        if pattern.constrained_diag_positions.size:
            values[np.asarray(pattern.constrained_diag_positions, dtype=np.int64)] = 1.0
        return np.asarray(values, dtype=np.float64)

    return _assemble_owned_tangent_values_python(pattern, ds_local)


def assemble_owned_tangent_matrix(
    pattern: OwnedTangentPattern,
    DS: np.ndarray,
    *,
    use_compiled: bool = True,
) -> csr_matrix:
    """Return the owned-row tangent matrix on the precomputed CSR pattern."""

    values = assemble_owned_tangent_values(pattern, DS, use_compiled=use_compiled)
    indptr = np.array(pattern.local_matrix_pattern.indptr, copy=True)
    indices = np.array(pattern.local_matrix_pattern.indices, copy=True)
    data = np.asarray(values, dtype=np.float64).copy()
    return csr_matrix((data, indices, indptr), shape=pattern.local_matrix_pattern.shape)


def assemble_owned_regularized_matrix(
    pattern: OwnedTangentPattern,
    DS: np.ndarray,
    r: float,
    *,
    use_compiled: bool = True,
) -> csr_matrix:
    """Return owned rows of ``K_r = r*K_elast + (1-r)*K_tangent`` on the fixed pattern."""

    tang = assemble_owned_tangent_values(pattern, DS, use_compiled=use_compiled)
    tang = np.asarray(tang, dtype=np.float64)
    tang *= 1.0 - float(r)
    tang += float(r) * np.asarray(pattern.elastic_values, dtype=np.float64)
    indptr = np.array(pattern.local_matrix_pattern.indptr, copy=True)
    indices = np.array(pattern.local_matrix_pattern.indices, copy=True)
    return csr_matrix((tang, indices, indptr), shape=pattern.local_matrix_pattern.shape)


def build_global_tangent_matrix(assembly, DS: np.ndarray) -> csr_matrix:
    """Reference global tangent assembly ``B^T D B`` for validation."""

    n_strain = int(assembly.n_strain)
    n_int = int(assembly.n_int)
    expected_rows = n_strain * n_strain
    if np.asarray(DS).shape[0] != expected_rows:
        raise ValueError(f"DS must have shape ({expected_rows}, n_int)")

    aux = np.arange(n_strain * n_int, dtype=np.int64).reshape(n_strain, n_int, order="F")
    iD = np.tile(aux, (n_strain, 1))
    jD = np.kron(aux, np.ones((n_strain, 1), dtype=np.int64))
    vD_pre = np.repeat(np.asarray(assembly.weight, dtype=np.float64)[None, :], n_strain * n_strain, axis=0).ravel(order="F")
    vD = vD_pre * np.asarray(DS, dtype=np.float64).ravel(order="F")
    D = csc_matrix((vD, (iD.ravel(order="F"), jD.ravel(order="F"))), shape=(n_strain * n_int, n_strain * n_int))
    out = (assembly.B.T @ D @ assembly.B).tocsr()
    return ((out + out.T) * 0.5).tocsr()
