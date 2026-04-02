"""Helpers for 3D PMG hierarchies and transfer construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mpi4py import MPI
from scipy.sparse import coo_matrix, csr_matrix

from ..core.simplex_lagrange import tetra_reference_nodes
from ..fem.basis import local_basis_volume_3d
from ..fem.distributed_elastic import find_overlap_partition
from ..mesh import MaterialSpec, load_mesh_from_file, reorder_mesh_nodes
from ..problem_assets import load_material_rows_for_path
from ..utils import owned_block_range, q_to_free_indices


@dataclass(frozen=True)
class PMGLevel:
    """One reordered FE level represented in free-DOF numbering."""

    order: int
    elem_type: str
    coord: np.ndarray
    elem: np.ndarray
    surf: np.ndarray
    q_mask: np.ndarray
    material_identifier: np.ndarray
    freedofs: np.ndarray
    total_to_free_orig: np.ndarray
    perm: np.ndarray
    iperm: np.ndarray
    owned_node_range: tuple[int, int]
    owned_total_range: tuple[int, int]
    owned_free_range: tuple[int, int]

    @property
    def dim(self) -> int:
        return int(self.coord.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.coord.shape[1])

    @property
    def total_size(self) -> int:
        return int(self.dim * self.n_nodes)

    @property
    def free_size(self) -> int:
        return int(self.freedofs.size)

    @property
    def global_size(self) -> int:
        return int(self.free_size)

    @property
    def lo(self) -> int:
        return int(self.owned_free_range[0])

    @property
    def hi(self) -> int:
        return int(self.owned_free_range[1])

    @property
    def owned_row_range(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.owned_free_range)


@dataclass(frozen=True)
class PMGTransfer:
    """Owned-row prolongation in reordered free-space coordinates."""

    coarse_order: int
    fine_order: int
    local_matrix: csr_matrix
    global_shape: tuple[int, int]
    owned_row_range: tuple[int, int]
    coo_rows: np.ndarray
    coo_cols: np.ndarray
    coo_data: np.ndarray


@dataclass(frozen=True)
class ElasticPMGHierarchy:
    """Static three-level PMG hierarchy metadata and transfers."""

    level_p1: PMGLevel
    level_p2: PMGLevel
    level_p4: PMGLevel
    prolongation_p21: PMGTransfer
    prolongation_p42: PMGTransfer
    materials: tuple[MaterialSpec, ...]
    mesh_path: Path
    node_ordering: str

    @property
    def levels(self) -> tuple[PMGLevel, PMGLevel, PMGLevel]:
        return (self.level_p1, self.level_p2, self.level_p4)

    @property
    def prolongations(self) -> tuple[PMGTransfer, PMGTransfer]:
        return (self.prolongation_p21, self.prolongation_p42)

    @property
    def coarse_level(self) -> PMGLevel:
        return self.level_p1

    @property
    def mid_level(self) -> PMGLevel:
        return self.level_p2

    @property
    def fine_level(self) -> PMGLevel:
        return self.level_p4


PMGHierarchy = ElasticPMGHierarchy


@dataclass(frozen=True)
class GeneralPMGHierarchy:
    """Static multi-level PMG hierarchy metadata and transfers."""

    levels_tuple: tuple[PMGLevel, ...]
    prolongations_tuple: tuple[PMGTransfer, ...]
    materials: tuple[MaterialSpec, ...]
    mesh_path: Path
    node_ordering: str

    @property
    def levels(self) -> tuple[PMGLevel, ...]:
        return self.levels_tuple

    @property
    def prolongations(self) -> tuple[PMGTransfer, ...]:
        return self.prolongations_tuple

    @property
    def coarse_level(self) -> PMGLevel:
        return self.levels_tuple[0]

    @property
    def mid_level(self) -> PMGLevel:
        if len(self.levels_tuple) < 2:
            return self.levels_tuple[0]
        return self.levels_tuple[-2]

    @property
    def fine_level(self) -> PMGLevel:
        return self.levels_tuple[-1]


def _materials_from_rows(material_rows: list[list[float]] | None, *, mesh_path: Path) -> tuple[MaterialSpec, ...]:
    rows = material_rows
    if rows is None:
        rows = load_material_rows_for_path(mesh_path)
    if rows is None:
        raise ValueError(f"No material rows found for {mesh_path}")
    return tuple(
        MaterialSpec(
            c0=float(row[0]),
            phi=float(row[1]),
            psi=float(row[2]),
            young=float(row[3]),
            poisson=float(row[4]),
            gamma_sat=float(row[5]),
            gamma_unsat=float(row[6]),
        )
        for row in rows
    )


def _identity_free_permutation(n_free: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.arange(int(n_free), dtype=np.int64)
    return perm, perm.copy()


def _prune_level_to_active_free(level: PMGLevel, active_free_mask: np.ndarray) -> PMGLevel:
    active_mask = np.asarray(active_free_mask, dtype=bool).reshape(-1)
    if active_mask.size != level.free_size:
        raise ValueError(
            f"Active free mask size {active_mask.size} does not match level free size {level.free_size}."
        )
    if bool(np.all(active_mask)):
        return level

    q_flat = np.asarray(level.q_mask, dtype=bool).reshape(-1, order="F").copy()
    inactive_total = np.asarray(level.freedofs[~active_mask], dtype=np.int64)
    q_flat[inactive_total] = False
    q_mask = q_flat.reshape(level.q_mask.shape, order="F")

    freedofs = np.asarray(level.freedofs[active_mask], dtype=np.int64)
    total_to_free_orig = np.full(level.total_size, -1, dtype=np.int64)
    total_to_free_orig[freedofs] = np.arange(freedofs.size, dtype=np.int64)
    perm, iperm = _identity_free_permutation(freedofs.size)
    lo = int(np.searchsorted(freedofs, level.owned_total_range[0], side="left"))
    hi = int(np.searchsorted(freedofs, level.owned_total_range[1], side="left"))

    return PMGLevel(
        order=int(level.order),
        elem_type=str(level.elem_type),
        coord=np.asarray(level.coord, dtype=np.float64),
        elem=np.asarray(level.elem, dtype=np.int64),
        surf=np.asarray(level.surf, dtype=np.int64),
        q_mask=q_mask,
        material_identifier=np.asarray(level.material_identifier, dtype=np.int64),
        freedofs=freedofs,
        total_to_free_orig=total_to_free_orig,
        perm=perm,
        iperm=iperm,
        owned_node_range=tuple(int(v) for v in level.owned_node_range),
        owned_total_range=tuple(int(v) for v in level.owned_total_range),
        owned_free_range=(lo, hi),
    )


def _prune_transfer_columns(
    transfer: PMGTransfer,
    active_coarse_mask: np.ndarray,
    *,
    coarse_level: PMGLevel,
) -> PMGTransfer:
    active_mask = np.asarray(active_coarse_mask, dtype=bool).reshape(-1)
    if active_mask.size != int(transfer.global_shape[1]):
        raise ValueError(
            f"Active coarse mask size {active_mask.size} does not match transfer column size {transfer.global_shape[1]}."
        )
    if bool(np.all(active_mask)):
        return transfer

    old_to_new = np.full(active_mask.size, -1, dtype=np.int64)
    old_to_new[active_mask] = np.arange(int(np.count_nonzero(active_mask)), dtype=np.int64)
    keep = active_mask[np.asarray(transfer.coo_cols, dtype=np.int64)]
    cols = old_to_new[np.asarray(transfer.coo_cols[keep], dtype=np.int64)]
    rows = np.asarray(transfer.coo_rows[keep], dtype=np.int64)
    data = np.asarray(transfer.coo_data[keep], dtype=np.float64)
    local_rows = rows - int(transfer.owned_row_range[0])
    local_matrix = coo_matrix(
        (data, (local_rows, cols)),
        shape=(int(transfer.owned_row_range[1] - transfer.owned_row_range[0]), coarse_level.free_size),
        dtype=np.float64,
    ).tocsr()
    return PMGTransfer(
        coarse_order=int(transfer.coarse_order),
        fine_order=int(transfer.fine_order),
        local_matrix=local_matrix,
        global_shape=(int(transfer.global_shape[0]), int(coarse_level.free_size)),
        owned_row_range=tuple(int(v) for v in transfer.owned_row_range),
        coo_rows=rows,
        coo_cols=cols,
        coo_data=data,
    )


def _prune_transfer_rows(
    transfer: PMGTransfer,
    active_fine_mask: np.ndarray,
    *,
    fine_level: PMGLevel,
) -> PMGTransfer:
    active_mask = np.asarray(active_fine_mask, dtype=bool).reshape(-1)
    if active_mask.size != int(transfer.global_shape[0]):
        raise ValueError(
            f"Active fine mask size {active_mask.size} does not match transfer row size {transfer.global_shape[0]}."
        )
    if bool(np.all(active_mask)):
        return transfer

    old_to_new = np.full(active_mask.size, -1, dtype=np.int64)
    old_to_new[active_mask] = np.arange(int(np.count_nonzero(active_mask)), dtype=np.int64)
    keep = active_mask[np.asarray(transfer.coo_rows, dtype=np.int64)]
    rows = old_to_new[np.asarray(transfer.coo_rows[keep], dtype=np.int64)]
    cols = np.asarray(transfer.coo_cols[keep], dtype=np.int64)
    data = np.asarray(transfer.coo_data[keep], dtype=np.float64)
    local_rows = rows - int(fine_level.owned_row_range[0])
    local_matrix = coo_matrix(
        (data, (local_rows, cols)),
        shape=(int(fine_level.owned_row_range[1] - fine_level.owned_row_range[0]), int(transfer.global_shape[1])),
        dtype=np.float64,
    ).tocsr()
    return PMGTransfer(
        coarse_order=int(transfer.coarse_order),
        fine_order=int(transfer.fine_order),
        local_matrix=local_matrix,
        global_shape=(int(fine_level.free_size), int(transfer.global_shape[1])),
        owned_row_range=tuple(int(v) for v in fine_level.owned_row_range),
        coo_rows=rows,
        coo_cols=cols,
        coo_data=data,
    )


def _build_level(
    *,
    mesh_path: Path,
    elem_type: str,
    node_ordering: str,
    reorder_parts: int | None,
    boundary_type: int,
    comm,
) -> PMGLevel:
    mesh = load_mesh_from_file(mesh_path, boundary_type=boundary_type, elem_type=elem_type)
    reordered = reorder_mesh_nodes(
        mesh.coord,
        mesh.elem,
        mesh.surf,
        mesh.q_mask,
        strategy=node_ordering,
        n_parts=reorder_parts if str(node_ordering).lower() == "block_metis" else None,
    )
    coord = np.asarray(reordered.coord, dtype=np.float64)
    elem = np.asarray(reordered.elem, dtype=np.int64)
    surf = np.asarray(reordered.surf, dtype=np.int64)
    q_mask = np.asarray(reordered.q_mask, dtype=bool)
    material_identifier = np.asarray(mesh.material, dtype=np.int64).ravel()

    freedofs_total = q_to_free_indices(q_mask)
    total_to_free_orig = np.full(coord.shape[0] * coord.shape[1], -1, dtype=np.int64)
    total_to_free_orig[freedofs_total] = np.arange(freedofs_total.size, dtype=np.int64)
    perm, iperm = _identity_free_permutation(freedofs_total.size)
    freedofs = np.asarray(freedofs_total[perm], dtype=np.int64)

    owned_total_range = owned_block_range(coord.shape[1], coord.shape[0], comm)
    node0 = int(owned_total_range[0] // coord.shape[0])
    node1 = int(owned_total_range[1] // coord.shape[0])
    lo = int(np.searchsorted(freedofs_total, owned_total_range[0], side="left"))
    hi = int(np.searchsorted(freedofs_total, owned_total_range[1], side="left"))

    return PMGLevel(
        order=int(elem_type[1:]),
        elem_type=str(elem_type),
        coord=coord,
        elem=elem,
        surf=surf,
        q_mask=q_mask,
        material_identifier=material_identifier,
        freedofs=freedofs,
        total_to_free_orig=total_to_free_orig,
        perm=perm,
        iperm=iperm,
        owned_node_range=(node0, node1),
        owned_total_range=(int(owned_total_range[0]), int(owned_total_range[1])),
        owned_free_range=(lo, hi),
    )


def _sorted_coo_arrays(entries: dict[tuple[int, int], float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not entries:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    keys = np.asarray(list(entries.keys()), dtype=np.int64)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    rows = np.asarray(keys[order, 0], dtype=np.int64)
    cols = np.asarray(keys[order, 1], dtype=np.int64)
    data = np.asarray([entries[tuple(keys[idx])] for idx in order], dtype=np.float64)
    return rows, cols, data


def _adjacent_level_prolongation(
    coarse: PMGLevel,
    fine: PMGLevel,
    *,
    coarse_order: int,
    fine_order: int,
    duplicate_tolerance: float = 1.0e-12,
) -> PMGTransfer:
    if coarse.dim != fine.dim:
        raise ValueError("PMG levels must have the same vector dimension")
    if coarse.material_identifier.shape != fine.material_identifier.shape or not np.array_equal(
        coarse.material_identifier,
        fine.material_identifier,
    ):
        raise ValueError("PMG levels must share the same element-material ordering.")
    if coarse.elem.shape[1] != fine.elem.shape[1]:
        raise ValueError("PMG levels must share the same element count.")

    fine_ref = tetra_reference_nodes(int(fine_order))
    coarse_hatp = np.asarray(local_basis_volume_3d(coarse.elem_type, fine_ref)[0], dtype=np.float64)
    overlap_nodes, overlap_elements = find_overlap_partition(fine.elem, fine.owned_node_range)
    _ = overlap_nodes

    entries: dict[tuple[int, int], float] = {}
    for elem_id in np.asarray(overlap_elements, dtype=np.int64).tolist():
        fine_nodes = np.asarray(fine.elem[:, elem_id], dtype=np.int64)
        coarse_nodes = np.asarray(coarse.elem[:, elem_id], dtype=np.int64)
        for fine_local_idx, fine_node in enumerate(fine_nodes.tolist()):
            for comp in range(fine.dim):
                fine_total = int(fine.dim * fine_node + comp)
                if fine_total < fine.owned_total_range[0] or fine_total >= fine.owned_total_range[1]:
                    continue
                fine_free_orig = int(fine.total_to_free_orig[fine_total])
                if fine_free_orig < 0:
                    continue
                fine_row = int(fine.iperm[fine_free_orig])
                coeff = np.asarray(coarse_hatp[:, fine_local_idx], dtype=np.float64)
                for coarse_local_idx in np.flatnonzero(np.abs(coeff) > duplicate_tolerance).tolist():
                    coarse_total = int(coarse.dim * coarse_nodes[coarse_local_idx] + comp)
                    coarse_free_orig = int(coarse.total_to_free_orig[coarse_total])
                    if coarse_free_orig < 0:
                        continue
                    coarse_col = int(coarse.iperm[coarse_free_orig])
                    value = float(coeff[coarse_local_idx])
                    key = (fine_row, coarse_col)
                    previous = entries.get(key)
                    if previous is None:
                        entries[key] = value
                        continue
                    if abs(previous - value) > duplicate_tolerance:
                        raise ValueError(
                            f"Inconsistent PMG interpolation entry for free row {fine_row} coarse col {coarse_col}: "
                            f"{previous} vs {value}"
                        )

    rows, cols, data = _sorted_coo_arrays(entries)
    local_rows = rows - fine.lo
    local_matrix = coo_matrix(
        (data, (local_rows, cols)),
        shape=(fine.hi - fine.lo, coarse.free_size),
        dtype=np.float64,
    ).tocsr()
    return PMGTransfer(
        coarse_order=int(coarse_order),
        fine_order=int(fine_order),
        local_matrix=local_matrix,
        global_shape=(fine.free_size, coarse.free_size),
        owned_row_range=fine.owned_row_range,
        coo_rows=rows,
        coo_cols=cols,
        coo_data=data,
    )


def _prepare_p1_search_geometry(level: PMGLevel) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if str(level.elem_type).upper() != "P1" or int(level.elem.shape[0]) != 4:
        raise ValueError("Cross-mesh PMG transfer currently requires a P1 tetrahedral coarse level.")

    coord = np.asarray(level.coord, dtype=np.float64)
    elem = np.asarray(level.elem, dtype=np.int64)
    x0 = coord[:, elem[0, :]].T
    x1 = coord[:, elem[1, :]].T
    x2 = coord[:, elem[2, :]].T
    x3 = coord[:, elem[3, :]].T
    mins = np.minimum.reduce([x0, x1, x2, x3])
    maxs = np.maximum.reduce([x0, x1, x2, x3])
    jac = np.stack((x1 - x0, x2 - x0, x3 - x0), axis=2)
    det = np.linalg.det(jac)
    if np.any(np.abs(det) <= 1.0e-14):
        raise ValueError("Cross-mesh PMG transfer encountered a degenerate coarse tetrahedron.")
    jac_inv = np.linalg.inv(jac)
    return x0, mins, maxs, jac_inv, elem


def _locate_point_in_p1_tets(
    point: np.ndarray,
    *,
    x0: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    jac_inv: np.ndarray,
    tolerance: float,
) -> tuple[int, np.ndarray]:
    pt = np.asarray(point, dtype=np.float64).reshape(3)
    candidates = np.flatnonzero(np.all((mins <= (pt + tolerance)) & (pt <= (maxs + tolerance)), axis=1))
    if candidates.size == 0:
        raise ValueError(f"No coarse tetrahedron bounding box contains point {pt.tolist()}.")

    best_elem = -1
    best_bary: np.ndarray | None = None
    best_margin = -np.inf
    for elem_id in candidates.tolist():
        xi = np.asarray(jac_inv[elem_id] @ (pt - x0[elem_id]), dtype=np.float64).reshape(3)
        bary = np.array((1.0 - float(np.sum(xi)), xi[0], xi[1], xi[2]), dtype=np.float64)
        if np.all(bary >= -tolerance) and np.all(bary <= 1.0 + tolerance):
            margin = float(np.min(bary))
            if margin > best_margin:
                best_elem = int(elem_id)
                best_margin = margin
                best_bary = bary
    if best_elem < 0 or best_bary is None:
        raise ValueError(
            f"Point {pt.tolist()} was not found inside any coarse tetrahedron among {int(candidates.size)} candidates."
        )
    return best_elem, best_bary


def _cross_mesh_p1_to_p1_prolongation(
    coarse: PMGLevel,
    fine: PMGLevel,
    *,
    duplicate_tolerance: float = 1.0e-12,
    location_tolerance: float = 1.0e-9,
) -> PMGTransfer:
    if coarse.dim != fine.dim:
        raise ValueError("PMG levels must have the same vector dimension")
    if int(coarse.order) != 1 or int(fine.order) != 1:
        raise ValueError("Cross-mesh PMG transfer currently supports only P1 -> P1 interpolation.")

    x0, mins, maxs, jac_inv, coarse_elem = _prepare_p1_search_geometry(coarse)
    entries: dict[tuple[int, int], float] = {}
    node0, node1 = tuple(int(v) for v in fine.owned_node_range)

    for fine_node in range(node0, node1):
        fine_point = np.asarray(fine.coord[:, fine_node], dtype=np.float64)
        coarse_elem_id, bary = _locate_point_in_p1_tets(
            fine_point,
            x0=x0,
            mins=mins,
            maxs=maxs,
            jac_inv=jac_inv,
            tolerance=location_tolerance,
        )
        coarse_nodes = np.asarray(coarse_elem[:, coarse_elem_id], dtype=np.int64)
        for comp in range(fine.dim):
            fine_total = int(fine.dim * fine_node + comp)
            fine_free_orig = int(fine.total_to_free_orig[fine_total])
            if fine_free_orig < 0:
                continue
            fine_row = int(fine.iperm[fine_free_orig])
            for coarse_local_idx, weight in enumerate(np.asarray(bary, dtype=np.float64).tolist()):
                if abs(weight) <= duplicate_tolerance:
                    continue
                coarse_total = int(coarse.dim * coarse_nodes[coarse_local_idx] + comp)
                coarse_free_orig = int(coarse.total_to_free_orig[coarse_total])
                if coarse_free_orig < 0:
                    continue
                coarse_col = int(coarse.iperm[coarse_free_orig])
                key = (fine_row, coarse_col)
                previous = entries.get(key)
                if previous is None:
                    entries[key] = float(weight)
                    continue
                if abs(previous - float(weight)) > duplicate_tolerance:
                    raise ValueError(
                        f"Inconsistent cross-mesh PMG entry for free row {fine_row} coarse col {coarse_col}: "
                        f"{previous} vs {weight}"
                    )

    rows, cols, data = _sorted_coo_arrays(entries)
    local_rows = rows - fine.lo
    local_matrix = coo_matrix(
        (data, (local_rows, cols)),
        shape=(fine.hi - fine.lo, coarse.free_size),
        dtype=np.float64,
    ).tocsr()
    return PMGTransfer(
        coarse_order=int(coarse.order),
        fine_order=int(fine.order),
        local_matrix=local_matrix,
        global_shape=(fine.free_size, coarse.free_size),
        owned_row_range=fine.owned_row_range,
        coo_rows=rows,
        coo_cols=cols,
        coo_data=data,
    )


def build_3d_pmg_hierarchy(
    mesh_path: str | Path,
    *,
    boundary_type: int = 0,
    node_ordering: str = "block_metis",
    reorder_parts: int | None = None,
    material_rows: list[list[float]] | None = None,
    comm,
) -> ElasticPMGHierarchy:
    """Build the static same-mesh `P1 -> P2 -> P4` PMG hierarchy."""

    mesh_path = Path(mesh_path).resolve()
    materials = _materials_from_rows(material_rows, mesh_path=mesh_path)
    level_p1 = _build_level(
        mesh_path=mesh_path,
        elem_type="P1",
        node_ordering=node_ordering,
        reorder_parts=reorder_parts,
        boundary_type=boundary_type,
        comm=comm,
    )
    level_p2 = _build_level(
        mesh_path=mesh_path,
        elem_type="P2",
        node_ordering=node_ordering,
        reorder_parts=reorder_parts,
        boundary_type=boundary_type,
        comm=comm,
    )
    level_p4 = _build_level(
        mesh_path=mesh_path,
        elem_type="P4",
        node_ordering=node_ordering,
        reorder_parts=reorder_parts,
        boundary_type=boundary_type,
        comm=comm,
    )
    prolongation_p21 = _adjacent_level_prolongation(
        level_p1,
        level_p2,
        coarse_order=1,
        fine_order=2,
    )
    prolongation_p42 = _adjacent_level_prolongation(
        level_p2,
        level_p4,
        coarse_order=2,
        fine_order=4,
    )
    return ElasticPMGHierarchy(
        level_p1=level_p1,
        level_p2=level_p2,
        level_p4=level_p4,
        prolongation_p21=prolongation_p21,
        prolongation_p42=prolongation_p42,
        materials=materials,
        mesh_path=mesh_path,
        node_ordering=str(node_ordering),
    )


def build_3d_same_mesh_pmg_hierarchy(
    mesh_path: str | Path,
    *,
    fine_elem_type: str = "P4",
    boundary_type: int = 0,
    node_ordering: str = "block_metis",
    reorder_parts: int | None = None,
    material_rows: list[list[float]] | None = None,
    comm,
) -> GeneralPMGHierarchy:
    """Build a same-mesh hierarchy such as `P1 -> P2` or `P1 -> P2 -> P4`."""

    fine_elem_type_norm = str(fine_elem_type).strip().upper()
    if fine_elem_type_norm not in {"P2", "P4"}:
        raise ValueError(f"Same-mesh PMG hierarchy currently supports fine_elem_type P2 or P4, got {fine_elem_type!r}.")

    mesh_path = Path(mesh_path).resolve()
    materials = _materials_from_rows(material_rows, mesh_path=mesh_path)
    level_p1 = _build_level(
        mesh_path=mesh_path,
        elem_type="P1",
        node_ordering=node_ordering,
        reorder_parts=reorder_parts,
        boundary_type=boundary_type,
        comm=comm,
    )
    level_p2 = _build_level(
        mesh_path=mesh_path,
        elem_type="P2",
        node_ordering=node_ordering,
        reorder_parts=reorder_parts,
        boundary_type=boundary_type,
        comm=comm,
    )
    prolongation_p21 = _adjacent_level_prolongation(
        level_p1,
        level_p2,
        coarse_order=1,
        fine_order=2,
    )
    if fine_elem_type_norm == "P2":
        return GeneralPMGHierarchy(
            levels_tuple=(level_p1, level_p2),
            prolongations_tuple=(prolongation_p21,),
            materials=materials,
            mesh_path=mesh_path,
            node_ordering=str(node_ordering),
        )

    level_p4 = _build_level(
        mesh_path=mesh_path,
        elem_type="P4",
        node_ordering=node_ordering,
        reorder_parts=reorder_parts,
        boundary_type=boundary_type,
        comm=comm,
    )
    prolongation_p42 = _adjacent_level_prolongation(
        level_p2,
        level_p4,
        coarse_order=2,
        fine_order=4,
    )
    return GeneralPMGHierarchy(
        levels_tuple=(level_p1, level_p2, level_p4),
        prolongations_tuple=(prolongation_p21, prolongation_p42),
        materials=materials,
        mesh_path=mesh_path,
        node_ordering=str(node_ordering),
    )


def build_3d_elastic_pmg_hierarchy(*args, **kwargs) -> ElasticPMGHierarchy:
    """Compatibility alias for the corrected PMG hierarchy builder."""

    return build_3d_pmg_hierarchy(*args, **kwargs)


def build_3d_mixed_pmg_hierarchy(
    fine_mesh_path: str | Path,
    coarse_mesh_path: str | Path,
    *,
    fine_elem_type: str = "P2",
    boundary_type: int = 0,
    node_ordering: str = "original",
    reorder_parts: int | None = None,
    material_rows: list[list[float]] | None = None,
    comm,
) -> ElasticPMGHierarchy:
    """Build a mixed three-level hierarchy such as `P1(L1) -> P1(L2) -> P2(L2)`."""

    fine_mesh_path = Path(fine_mesh_path).resolve()
    coarse_mesh_path = Path(coarse_mesh_path).resolve()
    materials = _materials_from_rows(material_rows, mesh_path=fine_mesh_path)
    level_coarse = _build_level(
        mesh_path=coarse_mesh_path,
        elem_type="P1",
        node_ordering=node_ordering,
        reorder_parts=reorder_parts,
        boundary_type=boundary_type,
        comm=comm,
    )
    level_mid = _build_level(
        mesh_path=fine_mesh_path,
        elem_type="P1",
        node_ordering=node_ordering,
        reorder_parts=reorder_parts,
        boundary_type=boundary_type,
        comm=comm,
    )
    level_fine = _build_level(
        mesh_path=fine_mesh_path,
        elem_type=str(fine_elem_type),
        node_ordering=node_ordering,
        reorder_parts=reorder_parts,
        boundary_type=boundary_type,
        comm=comm,
    )
    prolongation_h = _cross_mesh_p1_to_p1_prolongation(level_coarse, level_mid)
    local_active = np.diff(prolongation_h.local_matrix.tocsc().indptr).astype(np.int32) > 0
    if hasattr(comm, "tompi4py"):
        mpi_comm = comm.tompi4py()
    else:
        mpi_comm = comm
    global_active = np.asarray(mpi_comm.allreduce(local_active.astype(np.int32), op=MPI.SUM) > 0, dtype=bool)
    if not bool(np.all(global_active)):
        level_coarse = _prune_level_to_active_free(level_coarse, global_active)
        prolongation_h = _prune_transfer_columns(
            prolongation_h,
            global_active,
            coarse_level=level_coarse,
        )
    prolongation_p = _adjacent_level_prolongation(
        level_mid,
        level_fine,
        coarse_order=int(level_mid.order),
        fine_order=int(level_fine.order),
    )
    return ElasticPMGHierarchy(
        level_p1=level_coarse,
        level_p2=level_mid,
        level_p4=level_fine,
        prolongation_p21=prolongation_h,
        prolongation_p42=prolongation_p,
        materials=materials,
        mesh_path=fine_mesh_path,
        node_ordering=str(node_ordering),
    )


def _prune_general_hierarchy(
    levels: list[PMGLevel],
    prolongations: list[PMGTransfer],
    *,
    comm,
) -> tuple[list[PMGLevel], list[PMGTransfer]]:
    if hasattr(comm, "tompi4py"):
        mpi_comm = comm.tompi4py()
    else:
        mpi_comm = comm

    active_fine_mask: np.ndarray | None = None
    for transfer_idx in range(len(prolongations) - 1, -1, -1):
        fine_level = levels[transfer_idx + 1]
        if active_fine_mask is not None and not bool(np.all(active_fine_mask)):
            fine_level = _prune_level_to_active_free(fine_level, active_fine_mask)
            levels[transfer_idx + 1] = fine_level
            prolongations[transfer_idx] = _prune_transfer_rows(
                prolongations[transfer_idx],
                active_fine_mask,
                fine_level=fine_level,
            )

        local_active = np.diff(prolongations[transfer_idx].local_matrix.tocsc().indptr).astype(np.int32) > 0
        active_coarse_mask = np.asarray(
            mpi_comm.allreduce(local_active.astype(np.int32), op=MPI.SUM) > 0,
            dtype=bool,
        )
        if not bool(np.all(active_coarse_mask)):
            coarse_level = _prune_level_to_active_free(levels[transfer_idx], active_coarse_mask)
            levels[transfer_idx] = coarse_level
            prolongations[transfer_idx] = _prune_transfer_columns(
                prolongations[transfer_idx],
                active_coarse_mask,
                coarse_level=coarse_level,
            )
            if transfer_idx > 0:
                prolongations[transfer_idx - 1] = _prune_transfer_rows(
                    prolongations[transfer_idx - 1],
                    active_coarse_mask,
                    fine_level=coarse_level,
                )
            active_fine_mask = np.ones(coarse_level.free_size, dtype=bool)
        else:
            active_fine_mask = np.asarray(active_coarse_mask, dtype=bool)
    return levels, prolongations


def build_3d_mixed_pmg_hierarchy_with_intermediate_p2(
    fine_mesh_path: str | Path,
    coarse_mesh_path: str | Path,
    *,
    boundary_type: int = 0,
    node_ordering: str = "original",
    reorder_parts: int | None = None,
    material_rows: list[list[float]] | None = None,
    comm,
) -> GeneralPMGHierarchy:
    """Build `P1(L1) -> P1(L2) -> P2(L2) -> P4(L2)` for mixed-shell PMG."""

    fine_mesh_path = Path(fine_mesh_path).resolve()
    coarse_mesh_path = Path(coarse_mesh_path).resolve()
    materials = _materials_from_rows(material_rows, mesh_path=fine_mesh_path)

    levels = [
        _build_level(
            mesh_path=coarse_mesh_path,
            elem_type="P1",
            node_ordering=node_ordering,
            reorder_parts=reorder_parts,
            boundary_type=boundary_type,
            comm=comm,
        ),
        _build_level(
            mesh_path=fine_mesh_path,
            elem_type="P1",
            node_ordering=node_ordering,
            reorder_parts=reorder_parts,
            boundary_type=boundary_type,
            comm=comm,
        ),
        _build_level(
            mesh_path=fine_mesh_path,
            elem_type="P2",
            node_ordering=node_ordering,
            reorder_parts=reorder_parts,
            boundary_type=boundary_type,
            comm=comm,
        ),
        _build_level(
            mesh_path=fine_mesh_path,
            elem_type="P4",
            node_ordering=node_ordering,
            reorder_parts=reorder_parts,
            boundary_type=boundary_type,
            comm=comm,
        ),
    ]

    prolongations = [
        _cross_mesh_p1_to_p1_prolongation(levels[0], levels[1]),
        _adjacent_level_prolongation(levels[1], levels[2], coarse_order=1, fine_order=2),
        _adjacent_level_prolongation(levels[2], levels[3], coarse_order=2, fine_order=4),
    ]
    levels, prolongations = _prune_general_hierarchy(levels, prolongations, comm=comm)
    return GeneralPMGHierarchy(
        levels_tuple=tuple(levels),
        prolongations_tuple=tuple(prolongations),
        materials=materials,
        mesh_path=fine_mesh_path,
        node_ordering=str(node_ordering),
    )


def build_3d_mixed_pmg_chain_hierarchy(
    fine_mesh_path: str | Path,
    coarse_mesh_paths: list[str | Path] | tuple[str | Path, ...],
    *,
    fine_elem_type: str = "P2",
    boundary_type: int = 0,
    node_ordering: str = "original",
    reorder_parts: int | None = None,
    material_rows: list[list[float]] | None = None,
    comm,
) -> GeneralPMGHierarchy:
    """Build a mixed multi-level hierarchy such as `P1(L1)->...->P1(L5)->P2(L5)`."""

    fine_mesh_path = Path(fine_mesh_path).resolve()
    coarse_mesh_paths = tuple(Path(path).resolve() for path in coarse_mesh_paths)
    if not coarse_mesh_paths:
        raise ValueError("Mixed PMG chain hierarchy requires at least one coarse mesh path.")

    materials = _materials_from_rows(material_rows, mesh_path=fine_mesh_path)
    p1_tail_paths = tuple(reversed(coarse_mesh_paths)) + (fine_mesh_path,)
    levels: list[PMGLevel] = []
    for mesh_path in p1_tail_paths:
        levels.append(
            _build_level(
                mesh_path=mesh_path,
                elem_type="P1",
                node_ordering=node_ordering,
                reorder_parts=reorder_parts,
                boundary_type=boundary_type,
                comm=comm,
            )
        )
    levels.append(
        _build_level(
            mesh_path=fine_mesh_path,
            elem_type=str(fine_elem_type),
            node_ordering=node_ordering,
            reorder_parts=reorder_parts,
            boundary_type=boundary_type,
            comm=comm,
        )
    )

    prolongations: list[PMGTransfer] = []
    for level_idx in range(len(levels) - 2):
        prolongations.append(_cross_mesh_p1_to_p1_prolongation(levels[level_idx], levels[level_idx + 1]))
    prolongations.append(
        _adjacent_level_prolongation(
            levels[-2],
            levels[-1],
            coarse_order=int(levels[-2].order),
            fine_order=int(levels[-1].order),
        )
    )

    levels, prolongations = _prune_general_hierarchy(levels, prolongations, comm=comm)

    return GeneralPMGHierarchy(
        levels_tuple=tuple(levels),
        prolongations_tuple=tuple(prolongations),
        materials=materials,
        mesh_path=fine_mesh_path,
        node_ordering=str(node_ordering),
    )
