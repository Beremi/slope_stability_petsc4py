"""Ownership-aligned local tangent assembly on a fixed CSR pattern."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from ..mesh.materials import MaterialSpec
from .assembly import assemble_strain_geometry, assemble_strain_operator
from .distributed_elastic import OwnedElasticRows, assemble_owned_elastic_rows

try:  # pragma: no cover - compiled extension is optional during tests
    from slope_stability import _kernels
except Exception:  # pragma: no cover
    _kernels = None
try:  # pragma: no cover - mpi4py is optional in some unit-test environments
    from mpi4py import MPI as PYMPI
except Exception:  # pragma: no cover
    PYMPI = None

INT32_MAX = int(np.iinfo(np.int32).max)
DEFAULT_TANGENT_KERNEL = "rows"
SUPPORTED_TANGENT_KERNELS = frozenset({"legacy", "rows"})


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
    overlap_B: csr_matrix | None
    overlap_element_dof_lids: np.ndarray
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
    local_overlap_owner_mask: np.ndarray
    local_overlap_to_unique_pos: np.ndarray
    recv_neighbor_ranks: np.ndarray
    recv_ptr: np.ndarray
    recv_overlap_pos: np.ndarray
    recv_global_ip: np.ndarray
    send_neighbor_ranks: np.ndarray
    send_ptr: np.ndarray
    send_unique_pos: np.ndarray
    send_global_ip: np.ndarray
    scatter_map: np.ndarray
    row_slot_ptr: np.ndarray
    slot_elem: np.ndarray
    slot_lrow: np.ndarray
    slot_pos: np.ndarray
    constrained_diag_positions: np.ndarray
    n_p: int
    n_q: int
    stats: dict[str, float]
    timings: dict[str, float]


def _global_dofs_for_nodes(nodes: np.ndarray, dim: int) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.int64).ravel()
    if nodes.size == 0:
        return np.empty(0, dtype=np.int64)
    return (dim * np.repeat(nodes, dim) + np.tile(np.arange(dim, dtype=np.int64), nodes.size)).astype(np.int64)


def _ensure_int32_capacity(name: str, value: int) -> None:
    if int(value) > INT32_MAX:
        raise OverflowError(f"{name}={int(value)} exceeds int32 capacity")


def _normalize_tangent_kernel(kernel: str | None) -> str:
    normalized = DEFAULT_TANGENT_KERNEL if kernel is None else str(kernel).strip().lower()
    if normalized not in SUPPORTED_TANGENT_KERNELS:
        raise ValueError(
            f"Unsupported tangent kernel {kernel!r}; expected one of {sorted(SUPPORTED_TANGENT_KERNELS)}"
        )
    return normalized


def _mpi_comm():
    return None if PYMPI is None else PYMPI.COMM_WORLD


def _owner_ranks_for_nodes(min_nodes: np.ndarray, owned_node_ranges: list[tuple[int, int]]) -> np.ndarray:
    if len(owned_node_ranges) == 0:
        return np.empty(0, dtype=np.int32)
    ends = np.asarray([int(stop) for _start, stop in owned_node_ranges], dtype=np.int64)
    owners = np.searchsorted(ends, np.asarray(min_nodes, dtype=np.int64), side="right")
    if np.any(owners < 0) or np.any(owners >= len(owned_node_ranges)):
        raise RuntimeError("Failed to map element-owner nodes onto rank ownership ranges")
    return np.asarray(owners, dtype=np.int32)


def _build_exchange_lists(request_lists: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbor_ranks = [rank for rank, payload in enumerate(request_lists) if int(np.asarray(payload).size) > 0]
    counts = np.asarray([int(np.asarray(request_lists[rank]).size) for rank in neighbor_ranks], dtype=np.int64)
    _ensure_int32_capacity("exchange_ip_count", int(counts.sum()))
    ptr64 = np.zeros(len(neighbor_ranks) + 1, dtype=np.int64)
    if counts.size:
        ptr64[1:] = np.cumsum(counts, dtype=np.int64)
    payload = (
        np.concatenate([np.asarray(request_lists[rank]) for rank in neighbor_ranks]).astype(np.int64, copy=False)
        if neighbor_ranks
        else np.empty(0, dtype=np.int64)
    )
    return (
        np.asarray(neighbor_ranks, dtype=np.int32),
        np.asarray(ptr64, dtype=np.int32),
        np.asarray(payload, dtype=np.int64),
    )


def _build_constitutive_exchange_metadata(
    *,
    elem: np.ndarray,
    overlap_elements: np.ndarray,
    local_int_indices: np.ndarray,
    unique_local_int_indices: np.ndarray,
    owned_node_range: tuple[int, int],
    n_q: int,
    include_unique: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_overlap_ip = int(np.asarray(local_int_indices, dtype=np.int64).size)
    local_overlap_owner_mask = np.zeros(n_overlap_ip, dtype=bool)
    local_overlap_to_unique_pos = np.full(n_overlap_ip, -1, dtype=np.int32)

    if not include_unique:
        return (
            local_overlap_owner_mask,
            local_overlap_to_unique_pos,
            np.empty(0, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int64),
        )

    unique_lookup = {
        int(global_ip): int(pos) for pos, global_ip in enumerate(np.asarray(unique_local_int_indices, dtype=np.int64).tolist())
    }
    for overlap_pos, global_ip in enumerate(np.asarray(local_int_indices, dtype=np.int64).tolist()):
        unique_pos = unique_lookup.get(int(global_ip))
        if unique_pos is not None:
            local_overlap_owner_mask[overlap_pos] = True
            local_overlap_to_unique_pos[overlap_pos] = int(unique_pos)

    comm = _mpi_comm()
    if comm is None or int(comm.Get_size()) == 1:
        if not np.all(local_overlap_owner_mask):
            raise RuntimeError("Serial constitutive exchange metadata expected all overlap integration points to be locally owned")
        return (
            local_overlap_owner_mask,
            local_overlap_to_unique_pos,
            np.empty(0, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int32),
            np.zeros(1, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int64),
        )

    size = int(comm.Get_size())
    rank = int(comm.Get_rank())
    owned_ranges = [(int(a), int(b)) for a, b in comm.allgather((int(owned_node_range[0]), int(owned_node_range[1])))]
    elem_owner_nodes = np.min(np.asarray(elem, dtype=np.int64), axis=0)
    elem_owner_ranks = _owner_ranks_for_nodes(elem_owner_nodes, owned_ranges)

    request_global_ip_lists: list[list[int]] = [[] for _ in range(size)]
    request_overlap_pos_lists: list[list[int]] = [[] for _ in range(size)]
    for overlap_pos, global_ip in enumerate(np.asarray(local_int_indices, dtype=np.int64).tolist()):
        if local_overlap_owner_mask[overlap_pos]:
            continue
        owner_rank = int(elem_owner_ranks[int(global_ip // n_q)])
        if owner_rank == rank:
            raise RuntimeError("Remote overlap integration point resolved to local owner unexpectedly")
        request_global_ip_lists[owner_rank].append(int(global_ip))
        request_overlap_pos_lists[owner_rank].append(int(overlap_pos))

    send_requests = [
        np.asarray(payload, dtype=np.int64) if payload else np.empty(0, dtype=np.int64)
        for payload in request_global_ip_lists
    ]
    recv_requests = comm.alltoall(send_requests)

    owner_send_unique_lists: list[np.ndarray] = []
    owner_send_global_lists: list[np.ndarray] = []
    for source_rank, request_global_ip in enumerate(recv_requests):
        request_arr = np.asarray(request_global_ip, dtype=np.int64)
        if request_arr.size == 0:
            owner_send_unique_lists.append(np.empty(0, dtype=np.int32))
            owner_send_global_lists.append(np.empty(0, dtype=np.int64))
            continue
        owner_send_pos = np.empty(request_arr.size, dtype=np.int32)
        for idx, global_ip in enumerate(request_arr.tolist()):
            unique_pos = unique_lookup.get(int(global_ip))
            if unique_pos is None:
                raise RuntimeError(
                    f"Rank {rank} received constitutive request for non-owned integration point {int(global_ip)} from rank {source_rank}"
                )
            owner_send_pos[idx] = int(unique_pos)
        owner_send_unique_lists.append(owner_send_pos)
        owner_send_global_lists.append(np.asarray(request_arr, dtype=np.int64))

    recv_neighbor_ranks, recv_ptr, recv_overlap_pos_flat64 = _build_exchange_lists(
        [
            np.asarray(payload, dtype=np.int64) if payload else np.empty(0, dtype=np.int64)
            for payload in request_overlap_pos_lists
        ]
    )
    _ensure_int32_capacity("recv_overlap_pos", int(np.asarray(recv_overlap_pos_flat64).size))
    recv_overlap_pos = np.asarray(recv_overlap_pos_flat64, dtype=np.int32)
    _recv_neighbor_ranks_check, _recv_ptr_check, recv_global_ip = _build_exchange_lists(send_requests)
    if not np.array_equal(recv_neighbor_ranks, _recv_neighbor_ranks_check) or not np.array_equal(recv_ptr, _recv_ptr_check):
        raise RuntimeError("Constitutive receive metadata lost neighbor ordering consistency")

    send_neighbor_ranks, send_ptr, send_unique_pos64 = _build_exchange_lists(owner_send_unique_lists)
    send_unique_pos = np.asarray(send_unique_pos64, dtype=np.int32)
    _send_neighbor_ranks_check, _send_ptr_check, send_global_ip = _build_exchange_lists(owner_send_global_lists)
    if not np.array_equal(send_neighbor_ranks, _send_neighbor_ranks_check) or not np.array_equal(send_ptr, _send_ptr_check):
        raise RuntimeError("Constitutive send metadata lost neighbor ordering consistency")

    return (
        local_overlap_owner_mask,
        local_overlap_to_unique_pos,
        recv_neighbor_ranks,
        recv_ptr,
        recv_overlap_pos,
        recv_global_ip,
        send_neighbor_ranks,
        send_ptr,
        send_unique_pos,
        send_global_ip,
    )


def _build_local_row_maps(
    *,
    q_mask: np.ndarray,
    row0: int,
    local_matrix: csr_matrix,
) -> tuple[list[dict[int, int]], np.ndarray, np.ndarray]:
    dim = int(np.asarray(q_mask).shape[0])
    free_mask = np.asarray(q_mask, dtype=bool).reshape(-1, order="F")
    owned_global_rows = np.arange(int(row0), int(row0) + int(local_matrix.shape[0]), dtype=np.int64)
    owned_free = free_mask[owned_global_rows]
    n_local_rows = int(local_matrix.shape[0])

    _ensure_int32_capacity("local_matrix.nnz", int(local_matrix.nnz))

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

    return row_maps, np.asarray(owned_free, dtype=bool), np.asarray(constrained_diag_positions, dtype=np.int32)


def _build_scatter_map(
    *,
    elem: np.ndarray,
    overlap_elements: np.ndarray,
    row0: int,
    row_maps: list[dict[int, int]],
    q_mask: np.ndarray,
    owned_free: np.ndarray,
    n_local_rows: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    dim = int(np.asarray(q_mask).shape[0])
    free_mask = np.asarray(q_mask, dtype=bool).reshape(-1, order="F")
    n_p = int(elem.shape[0])
    n_local_dof = dim * n_p
    n_ld2 = n_local_dof * n_local_dof
    scatter = np.full((int(overlap_elements.size), n_ld2), -1, dtype=np.int32)
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

    return scatter, np.asarray([], dtype=np.int32), missing


def _build_row_slot_metadata(
    *,
    elem: np.ndarray,
    overlap_elements: np.ndarray,
    q_mask: np.ndarray,
    row0: int,
    row_maps: list[dict[int, int]],
    owned_free: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    dim = int(np.asarray(q_mask).shape[0])
    free_mask = np.asarray(q_mask, dtype=bool).reshape(-1, order="F")
    n_local_rows = int(owned_free.size)
    n_p = int(elem.shape[0])
    n_local_dof = dim * n_p
    if n_local_dof > int(np.iinfo(np.uint8).max):
        raise OverflowError(f"n_local_dof={n_local_dof} exceeds uint8 slot_lrow capacity")

    row_counts = np.zeros(n_local_rows, dtype=np.int64)
    active_rows_per_elem = np.zeros(int(overlap_elements.size), dtype=np.int32)

    for e_local, e_global in enumerate(np.asarray(overlap_elements, dtype=np.int64).tolist()):
        dofs = _global_dofs_for_nodes(np.asarray(elem[:, int(e_global)], dtype=np.int64), dim)
        active_rows = 0
        for row_global in dofs.tolist():
            if row_global < row0 or row_global >= row0 + n_local_rows:
                continue
            local_row = int(row_global - row0)
            if not owned_free[local_row]:
                continue
            row_counts[local_row] += 1
            active_rows += 1
        active_rows_per_elem[e_local] = int(active_rows)

    total_slots = int(row_counts.sum())
    _ensure_int32_capacity("row_slot_count", total_slots)
    row_slot_ptr64 = np.zeros(n_local_rows + 1, dtype=np.int64)
    row_slot_ptr64[1:] = np.cumsum(row_counts, dtype=np.int64)
    row_slot_ptr = np.asarray(row_slot_ptr64, dtype=np.int32)
    slot_elem = np.empty(total_slots, dtype=np.int32)
    slot_lrow = np.empty(total_slots, dtype=np.uint8)
    slot_pos = np.full((total_slots, n_local_dof), -1, dtype=np.int32)
    write_ptr = row_slot_ptr64[:-1].copy()

    for e_local, e_global in enumerate(np.asarray(overlap_elements, dtype=np.int64).tolist()):
        dofs = _global_dofs_for_nodes(np.asarray(elem[:, int(e_global)], dtype=np.int64), dim)
        dof_list = [int(v) for v in dofs.tolist()]
        for a, row_global in enumerate(dof_list):
            if row_global < row0 or row_global >= row0 + n_local_rows:
                continue
            local_row = int(row_global - row0)
            if not owned_free[local_row]:
                continue
            slot_idx = int(write_ptr[local_row])
            write_ptr[local_row] += 1
            slot_elem[slot_idx] = int(e_local)
            slot_lrow[slot_idx] = int(a)
            row_lookup = row_maps[local_row]
            pos_row = slot_pos[slot_idx]
            for b, col_global in enumerate(dof_list):
                if not free_mask[col_global]:
                    continue
                pos = row_lookup.get(int(col_global))
                if pos is None:
                    raise RuntimeError("Owned row slot metadata is missing a structural column entry")
                pos_row[b] = int(pos)

    avg_active_rows = float(active_rows_per_elem.mean()) if active_rows_per_elem.size else 0.0
    max_active_rows = int(active_rows_per_elem.max()) if active_rows_per_elem.size else 0
    return row_slot_ptr, slot_elem, slot_lrow, slot_pos, avg_active_rows, max_active_rows


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


def _csr_storage_bytes(matrix: csr_matrix) -> int:
    matrix = matrix.tocsr()
    return int(np.asarray(matrix.data).nbytes + np.asarray(matrix.indices).nbytes + np.asarray(matrix.indptr).nbytes)


def _require_legacy_scatter_map(pattern: OwnedTangentPattern) -> None:
    if int(np.asarray(pattern.overlap_elements, dtype=np.int64).size) > 0 and int(np.asarray(pattern.scatter_map).shape[1]) == 0:
        raise ValueError("Legacy tangent kernel requires scatter_map; prepare the pattern with include_legacy_scatter=True")


def _require_overlap_B(pattern: OwnedTangentPattern) -> csr_matrix:
    if pattern.overlap_B is None:
        raise ValueError("This path requires overlap_B; prepare the pattern with include_overlap_B=True")
    return pattern.overlap_B


def _assemble_overlap_strain_python_3d(pattern: OwnedTangentPattern, u_overlap: np.ndarray) -> np.ndarray:
    n_elem = int(np.asarray(pattern.overlap_elements, dtype=np.int64).size)
    n_p = int(pattern.n_p)
    n_q = int(pattern.n_q)
    out = np.zeros((int(pattern.local_int_indices.size), 6), dtype=np.float64)

    for e in range(n_elem):
        dofs = np.asarray(pattern.overlap_element_dof_lids[e], dtype=np.int64)
        g_base = e * n_q
        for q in range(n_q):
            g = g_base + q
            e11 = 0.0
            e22 = 0.0
            e33 = 0.0
            e12 = 0.0
            e23 = 0.0
            e13 = 0.0
            for j in range(n_p):
                base = 3 * j
                ux = float(u_overlap[dofs[base + 0]])
                uy = float(u_overlap[dofs[base + 1]])
                uz = float(u_overlap[dofs[base + 2]])
                dx = float(pattern.dphi1[g, j])
                dy = float(pattern.dphi2[g, j])
                dz = float(pattern.dphi3[g, j])
                e11 += dx * ux
                e22 += dy * uy
                e33 += dz * uz
                e12 += dy * ux + dx * uy
                e23 += dz * uy + dy * uz
                e13 += dz * ux + dx * uz
            out[g, 0] = e11
            out[g, 1] = e22
            out[g, 2] = e33
            out[g, 3] = e12
            out[g, 4] = e23
            out[g, 5] = e13
    return out


def _assemble_owned_force_python_rows(pattern: OwnedTangentPattern, stress_local: np.ndarray) -> np.ndarray:
    if int(pattern.dim) != 3 or int(pattern.n_strain) != 6:
        overlap_B = _require_overlap_B(pattern)
        load = (np.asarray(pattern.overlap_assembly_weight, dtype=np.float64)[None, :] * stress_local).reshape(-1, order="F")
        overlap_force = overlap_B.T.dot(load)
        owned_force = np.asarray(overlap_force[np.asarray(pattern.owned_local_overlap_dofs, dtype=np.int64)], dtype=np.float64).copy()
        owned_force[~np.asarray(pattern.owned_free_mask, dtype=bool)] = 0.0
        return owned_force

    out = np.zeros(int(pattern.local_matrix_pattern.shape[0]), dtype=np.float64)
    n_q = int(pattern.n_q)
    for local_row in range(out.size):
        slot_start = int(pattern.row_slot_ptr[local_row])
        slot_end = int(pattern.row_slot_ptr[local_row + 1])
        if slot_start == slot_end:
            continue
        acc = 0.0
        for slot_idx in range(slot_start, slot_end):
            elem_id = int(pattern.slot_elem[slot_idx])
            alpha = int(pattern.slot_lrow[slot_idx])
            node_i = alpha // 3
            comp = alpha % 3
            g_base = elem_id * n_q
            for q in range(n_q):
                g = g_base + q
                w = float(pattern.overlap_assembly_weight[g])
                sigma = np.asarray(stress_local[:, g], dtype=np.float64)
                dxi = float(pattern.dphi1[g, node_i])
                dyi = float(pattern.dphi2[g, node_i])
                dzi = float(pattern.dphi3[g, node_i])
                if comp == 0:
                    acc += w * (dxi * sigma[0] + dyi * sigma[3] + dzi * sigma[5])
                elif comp == 1:
                    acc += w * (dxi * sigma[3] + dyi * sigma[1] + dzi * sigma[4])
                else:
                    acc += w * (dxi * sigma[5] + dyi * sigma[4] + dzi * sigma[2])
        out[local_row] = acc
    out[~np.asarray(pattern.owned_free_mask, dtype=bool)] = 0.0
    return out


def assemble_overlap_strain(
    pattern: OwnedTangentPattern,
    U: np.ndarray,
    *,
    use_compiled: bool = True,
) -> np.ndarray:
    u_flat = np.asarray(U, dtype=np.float64).reshape(-1, order="F")
    u_overlap = np.ascontiguousarray(u_flat[np.asarray(pattern.overlap_global_dofs, dtype=np.int64)], dtype=np.float64)
    if use_compiled and _kernels is not None and int(pattern.dim) == 3 and int(pattern.n_strain) == 6:
        values = _kernels.assemble_overlap_strain_3d(
            pattern.dphi1,
            pattern.dphi2,
            pattern.dphi3,
            u_overlap,
            pattern.overlap_element_dof_lids,
            int(pattern.n_q),
        )
        return np.asarray(values, dtype=np.float64).T
    if int(pattern.dim) == 3 and int(pattern.n_strain) == 6:
        return _assemble_overlap_strain_python_3d(pattern, u_overlap).T
    overlap_B = _require_overlap_B(pattern)
    return np.asarray(overlap_B @ u_overlap, dtype=np.float64).reshape(int(pattern.n_strain), -1, order="F")


def assemble_owned_force_from_local_stress(
    pattern: OwnedTangentPattern,
    stress_local: np.ndarray,
    *,
    use_compiled: bool = True,
) -> np.ndarray:
    stress_local = np.asarray(stress_local, dtype=np.float64)
    if stress_local.shape[0] != int(pattern.n_strain):
        raise ValueError(f"stress_local must have shape ({int(pattern.n_strain)}, n_int_local)")
    if use_compiled and _kernels is not None and int(pattern.dim) == 3 and int(pattern.n_strain) == 6:
        values = _kernels.assemble_force_3d_rows(
            pattern.dphi1,
            pattern.dphi2,
            pattern.dphi3,
            np.ascontiguousarray(stress_local.T, dtype=np.float64),
            pattern.overlap_assembly_weight,
            pattern.row_slot_ptr,
            pattern.slot_elem,
            pattern.slot_lrow,
            int(pattern.n_q),
        )
        values = np.asarray(values, dtype=np.float64)
        values[~np.asarray(pattern.owned_free_mask, dtype=bool)] = 0.0
        return values
    return _assemble_owned_force_python_rows(pattern, stress_local)


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
    include_legacy_scatter: bool = True,
    include_overlap_B: bool = True,
    elastic_rows: OwnedElasticRows | None = None,
) -> OwnedTangentPattern:
    """Prepare the fixed owned-row pattern for repeated tangent assembly."""

    reused_elastic_rows = elastic_rows is not None
    if elastic_rows is None:
        t0 = perf_counter()
        elastic_rows = assemble_owned_elastic_rows(
            coord,
            elem,
            q_mask,
            material_identifier,
            materials,
            owned_node_range,
            elem_type=elem_type,
        )
        t_elastic = perf_counter() - t0
    else:
        t_elastic = 0.0
        if tuple(int(v) for v in elastic_rows.owned_node_range) != tuple(int(v) for v in owned_node_range):
            raise ValueError("Prebuilt elastic_rows owned_node_range does not match prepare_owned_tangent_pattern request")

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
    overlap_asm = (
        assemble_strain_operator(coord_overlap, elem_overlap, elem_type, dim=dim)
        if include_overlap_B
        else assemble_strain_geometry(coord_overlap, elem_overlap, elem_type, dim=dim)
    )
    overlap_B = None if overlap_asm.B is None else overlap_asm.B.tocsr()
    overlap_global_dofs = _global_dofs_for_nodes(overlap_nodes, dim)
    n_local_dof = dim * int(elem.shape[0])
    overlap_element_dof_lids = np.empty((int(overlap_elements.size), n_local_dof), dtype=np.int32)
    for e_local in range(int(overlap_elements.size)):
        dofs = _global_dofs_for_nodes(np.asarray(elem_overlap[:, e_local], dtype=np.int64), dim)
        if dofs.size:
            _ensure_int32_capacity("overlap_element_dof_lids", int(np.max(np.asarray(dofs, dtype=np.int64))))
        overlap_element_dof_lids[e_local, :] = np.asarray(dofs, dtype=np.int32)
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
    row_maps, owned_free, constrained_diag_positions = _build_local_row_maps(
        q_mask=np.asarray(q_mask, dtype=bool),
        row0=int(elastic_rows.owned_row_range[0]),
        local_matrix=pattern_matrix,
    )
    if include_legacy_scatter:
        scatter_map, _unused_diag, missing = _build_scatter_map(
            elem=np.asarray(elem, dtype=np.int64),
            overlap_elements=overlap_elements,
            row0=int(elastic_rows.owned_row_range[0]),
            row_maps=row_maps,
            q_mask=np.asarray(q_mask, dtype=bool),
            owned_free=owned_free,
            n_local_rows=int(pattern_matrix.shape[0]),
        )
    else:
        scatter_map = np.empty((int(overlap_elements.size), 0), dtype=np.int32)
        missing = 0
    row_slot_ptr, slot_elem, slot_lrow, slot_pos, avg_active_rows, max_active_rows = _build_row_slot_metadata(
        elem=np.asarray(elem, dtype=np.int64),
        overlap_elements=overlap_elements,
        q_mask=np.asarray(q_mask, dtype=bool),
        row0=int(elastic_rows.owned_row_range[0]),
        row_maps=row_maps,
        owned_free=owned_free,
    )
    scatter_time = perf_counter() - t_scatter

    n_q = int(overlap_asm.n_q)
    local_int_indices = (
        overlap_elements[:, None] * n_q + np.arange(n_q, dtype=np.int64)[None, :]
    ).reshape(-1)
    (
        local_overlap_owner_mask,
        local_overlap_to_unique_pos,
        recv_neighbor_ranks,
        recv_ptr,
        recv_overlap_pos,
        recv_global_ip,
        send_neighbor_ranks,
        send_ptr,
        send_unique_pos,
        send_global_ip,
    ) = _build_constitutive_exchange_metadata(
        elem=np.asarray(elem, dtype=np.int64),
        overlap_elements=overlap_elements,
        local_int_indices=local_int_indices,
        unique_local_int_indices=np.asarray(unique_local_int_indices, dtype=np.int64),
        owned_node_range=(node0, node1),
        n_q=n_q,
        include_unique=bool(include_unique),
    )
    scatter_bytes = int(np.asarray(scatter_map).nbytes)
    row_slot_bytes = int(
        np.asarray(row_slot_ptr).nbytes
        + np.asarray(slot_elem).nbytes
        + np.asarray(slot_lrow).nbytes
        + np.asarray(slot_pos).nbytes
    )
    dphi1 = np.ascontiguousarray(overlap_asm.dphi["dphi1"].T, dtype=np.float64)
    dphi2 = np.ascontiguousarray(overlap_asm.dphi["dphi2"].T, dtype=np.float64)
    dphi3 = np.ascontiguousarray(overlap_asm.dphi.get("dphi3", np.empty((0, 0), dtype=np.float64)).T, dtype=np.float64)
    dphi_bytes = int(np.asarray(dphi1).nbytes + np.asarray(dphi2).nbytes + np.asarray(dphi3).nbytes)
    overlap_B_bytes = 0 if overlap_B is None else _csr_storage_bytes(overlap_B)
    unique_B_bytes = _csr_storage_bytes(unique_B)

    return OwnedTangentPattern(
        dim=dim,
        n_strain=int(overlap_asm.n_strain),
        owned_node_range=tuple(int(v) for v in elastic_rows.owned_node_range),
        owned_row_range=tuple(int(v) for v in elastic_rows.owned_row_range),
        overlap_nodes=overlap_nodes,
        overlap_elements=overlap_elements,
        overlap_global_dofs=np.ascontiguousarray(overlap_global_dofs, dtype=np.int64),
        overlap_B=overlap_B,
        overlap_element_dof_lids=np.ascontiguousarray(overlap_element_dof_lids, dtype=np.int32),
        owned_local_overlap_dofs=np.ascontiguousarray(owned_local_overlap_dofs, dtype=np.int64),
        owned_free_mask=np.ascontiguousarray(owned_free_mask, dtype=bool),
        owned_free_local_rows=np.ascontiguousarray(owned_free_local_rows, dtype=np.int64),
        global_free_size=global_free_size,
        local_matrix_pattern=pattern_matrix,
        elastic_values=elastic_values,
        overlap_assembly_weight=np.ascontiguousarray(overlap_asm.weight, dtype=np.float64),
        dphi1=dphi1,
        dphi2=dphi2,
        dphi3=dphi3,
        local_int_indices=np.ascontiguousarray(local_int_indices, dtype=np.int64),
        unique_nodes=np.ascontiguousarray(unique_nodes, dtype=np.int64),
        unique_elements=np.ascontiguousarray(unique_elements, dtype=np.int64),
        unique_global_dofs=np.ascontiguousarray(unique_global_dofs, dtype=np.int64),
        unique_B=unique_B,
        unique_local_int_indices=np.ascontiguousarray(unique_local_int_indices, dtype=np.int64),
        local_overlap_owner_mask=np.ascontiguousarray(local_overlap_owner_mask, dtype=bool),
        local_overlap_to_unique_pos=np.ascontiguousarray(local_overlap_to_unique_pos, dtype=np.int32),
        recv_neighbor_ranks=np.ascontiguousarray(recv_neighbor_ranks, dtype=np.int32),
        recv_ptr=np.ascontiguousarray(recv_ptr, dtype=np.int32),
        recv_overlap_pos=np.ascontiguousarray(recv_overlap_pos, dtype=np.int32),
        recv_global_ip=np.ascontiguousarray(recv_global_ip, dtype=np.int64),
        send_neighbor_ranks=np.ascontiguousarray(send_neighbor_ranks, dtype=np.int32),
        send_ptr=np.ascontiguousarray(send_ptr, dtype=np.int32),
        send_unique_pos=np.ascontiguousarray(send_unique_pos, dtype=np.int32),
        send_global_ip=np.ascontiguousarray(send_global_ip, dtype=np.int64),
        scatter_map=np.ascontiguousarray(scatter_map, dtype=np.int32),
        row_slot_ptr=np.ascontiguousarray(row_slot_ptr, dtype=np.int32),
        slot_elem=np.ascontiguousarray(slot_elem, dtype=np.int32),
        slot_lrow=np.ascontiguousarray(slot_lrow, dtype=np.uint8),
        slot_pos=np.ascontiguousarray(slot_pos, dtype=np.int32),
        constrained_diag_positions=np.ascontiguousarray(constrained_diag_positions, dtype=np.int32),
        n_p=int(elem.shape[0]),
        n_q=n_q,
        stats={
            "scatter_bytes": float(scatter_bytes),
            "row_slot_bytes": float(row_slot_bytes),
            "overlap_B_bytes": float(overlap_B_bytes),
            "unique_B_bytes": float(unique_B_bytes),
            "dphi_bytes": float(dphi_bytes),
            "avg_active_rows_per_overlap_element": float(avg_active_rows),
            "max_active_rows_per_overlap_element": float(max_active_rows),
            "legacy_scatter_enabled": float(bool(include_legacy_scatter)),
        },
        timings={
            "elastic_pattern_s": float(t_elastic),
            "overlap_geometry_s": float(overlap_time),
            "unique_geometry_s": float(unique_time),
            "structural_pattern_s": float(pattern_time),
            "scatter_map_s": float(scatter_time),
            "scatter_missing_entries": float(missing),
            "elastic_pattern_reused": float(bool(reused_elastic_rows)),
        },
    )


def _assemble_owned_tangent_values_python_legacy(pattern: OwnedTangentPattern, ds_local: np.ndarray) -> np.ndarray:
    _require_legacy_scatter_map(pattern)
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


def _assemble_owned_tangent_values_python_rows(pattern: OwnedTangentPattern, ds_local: np.ndarray) -> np.ndarray:
    if int(pattern.dim) != 3 or int(pattern.n_strain) != 6:
        return _assemble_owned_tangent_values_python_legacy(pattern, ds_local)

    n_p = int(pattern.n_p)
    n_q = int(pattern.n_q)
    out = np.zeros_like(pattern.elastic_values)

    for local_row in range(int(pattern.local_matrix_pattern.shape[0])):
        slot_start = int(pattern.row_slot_ptr[local_row])
        slot_end = int(pattern.row_slot_ptr[local_row + 1])
        if slot_start == slot_end:
            continue
        for slot_idx in range(slot_start, slot_end):
            elem_id = int(pattern.slot_elem[slot_idx])
            alpha = int(pattern.slot_lrow[slot_idx])
            node_i = alpha // 3
            comp = alpha % 3
            pos = pattern.slot_pos[slot_idx]
            g_base = elem_id * n_q

            for q in range(n_q):
                g = g_base + q
                w = float(pattern.overlap_assembly_weight[g])
                dsg = ds_local[g]
                dxi = float(pattern.dphi1[g, node_i])
                dyi = float(pattern.dphi2[g, node_i])
                dzi = float(pattern.dphi3[g, node_i])

                if comp == 0:
                    t0 = w * (dxi * dsg[0] + dyi * dsg[3] + dzi * dsg[5])
                    t1 = w * (dxi * dsg[6] + dyi * dsg[9] + dzi * dsg[11])
                    t2 = w * (dxi * dsg[12] + dyi * dsg[15] + dzi * dsg[17])
                    t3 = w * (dxi * dsg[18] + dyi * dsg[21] + dzi * dsg[23])
                    t4 = w * (dxi * dsg[24] + dyi * dsg[27] + dzi * dsg[29])
                    t5 = w * (dxi * dsg[30] + dyi * dsg[33] + dzi * dsg[35])
                elif comp == 1:
                    t0 = w * (dyi * dsg[1] + dxi * dsg[3] + dzi * dsg[4])
                    t1 = w * (dyi * dsg[7] + dxi * dsg[9] + dzi * dsg[10])
                    t2 = w * (dyi * dsg[13] + dxi * dsg[15] + dzi * dsg[16])
                    t3 = w * (dyi * dsg[19] + dxi * dsg[21] + dzi * dsg[22])
                    t4 = w * (dyi * dsg[25] + dxi * dsg[27] + dzi * dsg[28])
                    t5 = w * (dyi * dsg[31] + dxi * dsg[33] + dzi * dsg[34])
                elif comp == 2:
                    t0 = w * (dzi * dsg[2] + dyi * dsg[4] + dxi * dsg[5])
                    t1 = w * (dzi * dsg[8] + dyi * dsg[10] + dxi * dsg[11])
                    t2 = w * (dzi * dsg[14] + dyi * dsg[16] + dxi * dsg[17])
                    t3 = w * (dzi * dsg[20] + dyi * dsg[22] + dxi * dsg[23])
                    t4 = w * (dzi * dsg[26] + dyi * dsg[28] + dxi * dsg[29])
                    t5 = w * (dzi * dsg[32] + dyi * dsg[34] + dxi * dsg[35])
                else:  # pragma: no cover - guarded by alpha//3 decomposition above
                    raise ValueError(f"Unsupported row component {comp}")

                for j in range(n_p):
                    dxj = float(pattern.dphi1[g, j])
                    dyj = float(pattern.dphi2[g, j])
                    dzj = float(pattern.dphi3[g, j])
                    pos_x = int(pos[3 * j + 0])
                    pos_y = int(pos[3 * j + 1])
                    pos_z = int(pos[3 * j + 2])
                    if pos_x >= 0:
                        out[pos_x] += t0 * dxj + t3 * dyj + t5 * dzj
                    if pos_y >= 0:
                        out[pos_y] += t3 * dxj + t1 * dyj + t4 * dzj
                    if pos_z >= 0:
                        out[pos_z] += t5 * dxj + t4 * dyj + t2 * dzj

    if pattern.constrained_diag_positions.size:
        out[pattern.constrained_diag_positions] = 1.0
    return out


def assemble_owned_tangent_values(
    pattern: OwnedTangentPattern,
    DS: np.ndarray,
    *,
    use_compiled: bool = True,
    kernel: str = DEFAULT_TANGENT_KERNEL,
) -> np.ndarray:
    """Assemble tangent values on the fixed owned-row CSR pattern."""

    kernel_name = _normalize_tangent_kernel(kernel)
    ds_global = np.asarray(DS, dtype=np.float64)
    expected_rows = int(pattern.n_strain * pattern.n_strain)
    if ds_global.ndim != 2 or ds_global.shape[0] != expected_rows:
        raise ValueError(f"DS must have shape ({expected_rows}, n_int) for owned tangent assembly")
    if ds_global.shape[1] == pattern.local_int_indices.size:
        ds_local = np.ascontiguousarray(ds_global.T, dtype=np.float64)
    else:
        ds_local = np.ascontiguousarray(ds_global[:, pattern.local_int_indices].T, dtype=np.float64)

    if kernel_name == "legacy":
        _require_legacy_scatter_map(pattern)

    if use_compiled and _kernels is not None and int(pattern.dim) == 3 and int(pattern.n_strain) == 6:
        if kernel_name == "rows":
            values = _kernels.assemble_tangent_values_3d_rows(
                pattern.dphi1,
                pattern.dphi2,
                pattern.dphi3,
                ds_local,
                pattern.overlap_assembly_weight,
                pattern.row_slot_ptr,
                pattern.slot_elem,
                pattern.slot_lrow,
                pattern.slot_pos,
                int(pattern.n_q),
                int(pattern.elastic_values.size),
            )
        else:
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

    if kernel_name == "rows":
        return _assemble_owned_tangent_values_python_rows(pattern, ds_local)
    return _assemble_owned_tangent_values_python_legacy(pattern, ds_local)


def assemble_owned_tangent_matrix(
    pattern: OwnedTangentPattern,
    DS: np.ndarray,
    *,
    use_compiled: bool = True,
    kernel: str = DEFAULT_TANGENT_KERNEL,
) -> csr_matrix:
    """Return the owned-row tangent matrix on the precomputed CSR pattern."""

    values = assemble_owned_tangent_values(pattern, DS, use_compiled=use_compiled, kernel=kernel)
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
    kernel: str = DEFAULT_TANGENT_KERNEL,
) -> csr_matrix:
    """Return owned rows of ``K_r = r*K_elast + (1-r)*K_tangent`` on the fixed pattern."""

    tang = assemble_owned_tangent_values(pattern, DS, use_compiled=use_compiled, kernel=kernel)
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
