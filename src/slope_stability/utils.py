"""Utility helpers shared across PETSc/FEM/solver modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
try:  # pragma: no cover - PETSc is optional in some test/benchmark environments
    from petsc4py import PETSc
except Exception:  # pragma: no cover
    PETSc = None

_PETSC_MAT_CSR_REFS: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_PETSC_MAT_COO_REFS: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_PETSC_MAT_METADATA: dict[int, dict[str, object]] = {}
_PETSC_MAT_IS_REFS: dict[int, tuple[object, object]] = {}


@dataclass
class IterationHistory:
    iterations: list[int]
    solve_time: list[float]
    preconditioner_time: list[float]
    orthogonalization_time: list[float]


def q_to_free_indices(q_mask: np.ndarray) -> np.ndarray:
    """Return free DOF indices from a MATLAB-style free mask ``Q``.

    In MATLAB examples ``Q`` is a logical matrix ``(dim, n_nodes)`` with ``True``
    for unknown degrees of freedom. The same ordering is kept here:
    column-major flattening with ``order='F'``.
    """

    return np.flatnonzero(np.asarray(q_mask, dtype=bool).reshape(-1, order="F")).astype(np.int64)


def flatten_field(field: np.ndarray) -> np.ndarray:
    """Flatten a ``(dim, n_nodes)`` field in MATLAB column-major order."""

    return np.asarray(field, dtype=np.float64).reshape(-1, order="F")


def unflatten_field(vec: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`flatten_field`."""

    dim, n_nodes = shape
    return np.asarray(vec, dtype=np.float64).reshape((dim, n_nodes), order="F")


def full_field_from_free_values(values: np.ndarray, free_idx: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Lift free DOF values back into a full ``(dim, n_nodes)`` field.

    The mapping follows the same MATLAB-style column-major ordering used by
    :func:`q_to_free_indices`.
    """

    flat = np.zeros(int(np.prod(shape, dtype=np.int64)), dtype=np.float64)
    flat[np.asarray(free_idx, dtype=np.int64)] = np.asarray(values, dtype=np.float64).reshape(-1)
    return unflatten_field(flat, shape)


def to_numpy_vector(x) -> np.ndarray:
    """Convert a PETSc vector or array-like to ``np.ndarray``."""

    if PETSc is not None and isinstance(x, PETSc.Vec):
        return x.getArray(readonly=False).copy()
    return np.asarray(x, dtype=np.float64)


def _require_petsc() -> None:
    if PETSc is None:
        raise RuntimeError("petsc4py is required for this operation")


def to_petsc_vector(vec: np.ndarray, comm=None):
    """Create a PETSc vector from ``np.ndarray`` without extra casts."""

    _require_petsc()
    if comm is None:
        comm = PETSc.COMM_WORLD
    arr = np.asarray(vec, dtype=np.float64).ravel()
    v = PETSc.Vec().create(comm=comm)
    v.setSizes(len(arr))
    v.setArray(arr)
    return v


def owned_block_range(n_blocks: int, block_size: int, comm) -> tuple[int, int]:
    """Return a contiguous row-ownership range aligned to node blocks."""

    size = int(comm.getSize())
    rank = int(comm.getRank())
    start_block = (rank * n_blocks) // size
    end_block = ((rank + 1) * n_blocks) // size
    return start_block * block_size, end_block * block_size


def global_array_to_petsc_vec(
    vec: np.ndarray,
    *,
    comm,
    ownership_range: tuple[int, int] | None = None,
    bsize: int | None = None,
):
    """Create a PETSc vector from a global dense array using local ownership."""

    _require_petsc()
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if ownership_range is None or int(comm.getSize()) == 1:
        return PETSc.Vec().createWithArray(arr.copy(), size=arr.size, bsize=bsize, comm=comm)

    r0, r1 = ownership_range
    local = np.array(arr[r0:r1], dtype=np.float64, copy=True)
    return PETSc.Vec().createWithArray(local, size=(r1 - r0, arr.size), bsize=bsize, comm=comm)


def local_array_to_petsc_vec(
    local: np.ndarray,
    *,
    global_size: int,
    comm,
    bsize: int | None = None,
):
    """Create a distributed PETSc vector from already-owned local entries."""

    _require_petsc()
    arr = np.asarray(local, dtype=np.float64).reshape(-1)
    return PETSc.Vec().createWithArray(arr.copy(), size=(arr.size, global_size), bsize=bsize, comm=comm)


def petsc_vec_to_global_array(vec) -> np.ndarray:
    """Collect a distributed PETSc vector onto all ranks as a dense array."""

    _require_petsc()
    local = vec.getArray(readonly=False).copy()
    comm = vec.getComm()
    if int(comm.getSize()) == 1:
        return local
    parts = comm.tompi4py().allgather(local)
    return np.concatenate(parts) if parts else local


def to_petsc_aij_matrix(
    A,
    comm=None,
    *,
    block_size: int | None = None,
    ownership_range: tuple[int, int] | None = None,
):
    """Convert a dense/scipy matrix into a PETSc AIJ matrix."""

    if PETSc is not None and isinstance(A, PETSc.Mat):
        return A

    import scipy.sparse as sp

    _require_petsc()
    if comm is None:
        comm = PETSc.COMM_SELF

    if sp.issparse(A):
        csr = A.tocsr()
    else:
        csr = sp.csr_matrix(np.asarray(A, dtype=np.float64))

    if block_size is not None and int(comm.getSize()) > 1:
        if csr.shape[0] != csr.shape[1]:
            raise ValueError("Distributed block ownership currently expects a square matrix")
        if csr.shape[0] % block_size != 0:
            raise ValueError(f"Global rows {csr.shape[0]} are not divisible by block_size {block_size}")

    if ownership_range is None and int(comm.getSize()) > 1 and block_size is not None:
        ownership_range = owned_block_range(csr.shape[0] // block_size, block_size, comm)

    if ownership_range is None or int(comm.getSize()) == 1:
        indptr = np.array(csr.indptr, dtype=PETSc.IntType, copy=True)
        indices = np.array(csr.indices, dtype=PETSc.IntType, copy=True)
        data = np.array(csr.data, dtype=np.float64, copy=True)
        size = csr.shape
    else:
        r0, r1 = ownership_range
        row0 = int(csr.indptr[r0])
        row1 = int(csr.indptr[r1])
        indptr = np.array(csr.indptr[r0 : r1 + 1] - row0, dtype=PETSc.IntType, copy=True)
        indices = np.array(csr.indices[row0:row1], dtype=PETSc.IntType, copy=True)
        data = np.array(csr.data[row0:row1], dtype=np.float64, copy=True)
        local_rows = r1 - r0
        size = ((local_rows, csr.shape[0]), (local_rows, csr.shape[1]))

    mat = PETSc.Mat().createAIJ(size=size, csr=(indptr, indices, data), comm=comm)
    if block_size is not None:
        mat.setBlockSize(int(block_size))
    mat.assemble()
    # Keep CSR buffers alive for PETSc builds that borrow Python-managed memory.
    _PETSC_MAT_CSR_REFS[int(mat.handle)] = (indptr, indices, data)
    return mat


def local_csr_to_petsc_aij_matrix(
    A_local,
    *,
    global_shape: tuple[int, int],
    comm,
    block_size: int | None = None,
    local_col_size: int | None = None,
):
    """Create a distributed PETSc AIJ matrix from already-owned CSR rows."""

    import scipy.sparse as sp

    _require_petsc()
    csr = A_local.tocsr() if sp.issparse(A_local) else sp.csr_matrix(np.asarray(A_local, dtype=np.float64))
    indptr = np.array(csr.indptr, dtype=PETSc.IntType, copy=True)
    indices = np.array(csr.indices, dtype=PETSc.IntType, copy=True)
    data = np.array(csr.data, dtype=np.float64, copy=True)
    if local_col_size is not None:
        local_cols = int(local_col_size)
    elif int(comm.getSize()) == 1:
        local_cols = int(csr.shape[1])
    elif (
        block_size is not None
        and int(global_shape[0]) == int(global_shape[1])
        and int(csr.shape[0]) % int(block_size) == 0
    ):
        # Preserve node-aligned ownership for distributed square block operators.
        local_cols = int(csr.shape[0])
    else:
        local_cols = PETSc.DECIDE
    mat = PETSc.Mat().createAIJ(
        size=((csr.shape[0], int(global_shape[0])), (local_cols, int(global_shape[1]))),
        csr=(indptr, indices, data),
        comm=comm,
    )
    if (
        block_size is not None
        and int(csr.shape[0]) % int(block_size) == 0
        and int(global_shape[0]) % int(block_size) == 0
        and int(global_shape[1]) % int(block_size) == 0
        and int(local_cols) != int(PETSc.DECIDE)
        and int(local_cols) % int(block_size) == 0
    ):
        mat.setBlockSize(int(block_size))
    mat.assemble()
    _PETSC_MAT_CSR_REFS[int(mat.handle)] = (indptr, indices, data)
    return mat


def owned_coo_to_petsc_aij_matrix(
    rows,
    cols,
    data,
    *,
    global_shape: tuple[int, int],
    owned_row_range: tuple[int, int],
    comm,
    local_col_size: int | None = None,
):
    """Create a distributed PETSc AIJ matrix from owned global COO triplets."""

    _require_petsc()
    row_arr = np.asarray(rows, dtype=PETSc.IntType).reshape(-1)
    col_arr = np.asarray(cols, dtype=PETSc.IntType).reshape(-1)
    data_arr = np.asarray(data, dtype=np.float64).reshape(-1)
    if row_arr.size != col_arr.size or row_arr.size != data_arr.size:
        raise ValueError("COO rows, cols, and data must have the same length")

    row0, row1 = (int(owned_row_range[0]), int(owned_row_range[1]))
    if row_arr.size:
        if int(row_arr.min()) < row0 or int(row_arr.max()) >= row1:
            raise ValueError("COO rows must be locally owned by the target rank")
    local_rows = int(row1 - row0)
    local_cols = int(local_col_size) if local_col_size is not None else PETSc.DECIDE

    mat = PETSc.Mat().createAIJ(
        size=((local_rows, int(global_shape[0])), (local_cols, int(global_shape[1]))),
        comm=comm,
    )
    if row_arr.size:
        mat.setPreallocationCOO(row_arr, col_arr)
        mat.setValuesCOO(data_arr)
    mat.assemble()
    _PETSC_MAT_COO_REFS[int(mat.handle)] = (row_arr, col_arr, data_arr)
    return mat


def update_petsc_aij_matrix_csr(
    A,
    *,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
):
    """Overwrite values of an existing AIJ matrix on the same CSR pattern."""

    _require_petsc()
    if not isinstance(A, PETSc.Mat):
        raise TypeError("A must be a PETSc Mat")
    A.setValuesCSR(
        np.asarray(indptr, dtype=PETSc.IntType),
        np.asarray(indices, dtype=PETSc.IntType),
        np.asarray(data, dtype=np.float64),
    )
    A.assemble()
    return A


def set_petsc_matrix_metadata(A, **metadata) -> None:
    """Attach Python-side metadata to a PETSc matrix handle."""

    if PETSc is not None and isinstance(A, PETSc.Mat):
        store = _PETSC_MAT_METADATA.setdefault(int(A.handle), {})
        store.update(metadata)


def get_petsc_matrix_metadata(A) -> dict[str, object]:
    """Return Python-side metadata previously attached to a PETSc matrix."""

    if PETSc is None or not isinstance(A, PETSc.Mat):
        return {}
    return dict(_PETSC_MAT_METADATA.get(int(A.handle), {}))


def bddc_pc_coordinates_from_metadata(A):
    """Return coordinates in the shape expected by PCBDDC for ``A``.

    Metadata stores node-wise coordinates when available. MATIS in this repo is
    assembled on dof-level maps, so PETSc may still expect one tuple per local
    vector entry. Expand node-wise coordinates to the local MATIS vector layout
    when needed.
    """

    metadata = get_petsc_matrix_metadata(A)
    coordinates = metadata.get("bddc_local_coordinates")
    if coordinates is None:
        return None
    arr = np.asarray(coordinates, dtype=np.float64)
    if arr.ndim != 2 or PETSc is None or not isinstance(A, PETSc.Mat):
        return arr
    expected = int(metadata.get("matis_vector_local_size", arr.shape[0]))
    if arr.shape[0] == expected:
        return arr
    block_size = int(A.getBlockSize() or 1)
    if block_size > 1 and int(arr.shape[0]) * block_size == expected:
        return np.repeat(arr, block_size, axis=0)
    return arr


def get_petsc_is_local_mat(A):
    """Return the locally stored SeqAIJ matrix for a MATIS matrix if registered."""

    if PETSc is None or not isinstance(A, PETSc.Mat):
        return None
    refs = _PETSC_MAT_IS_REFS.get(int(A.handle))
    if refs is None:
        return None
    return refs[0]


def local_csr_to_petsc_seq_matrix(A_local, *, mat_type: str = "aij", block_size: int | None = None):
    """Create a sequential PETSc sparse matrix from local CSR data."""

    import scipy.sparse as sp

    _require_petsc()
    csr = A_local.tocsr() if sp.issparse(A_local) else sp.csr_matrix(np.asarray(A_local, dtype=np.float64))
    normalized = str(mat_type).strip().lower()
    if normalized not in {"aij", "sbaij"}:
        raise ValueError(f"Unsupported local PETSc matrix type {mat_type!r}; expected 'aij' or 'sbaij'")
    if normalized == "sbaij":
        if csr.shape[0] != csr.shape[1]:
            raise ValueError("SBAIJ local matrices must be square")
    indptr = np.array(csr.indptr, dtype=PETSc.IntType, copy=True)
    indices = np.array(csr.indices, dtype=PETSc.IntType, copy=True)
    data = np.array(csr.data, dtype=np.float64, copy=True)
    if normalized == "sbaij":
        bs = 1 if block_size is None else int(block_size)
        if csr.shape[0] % bs != 0 or csr.shape[1] % bs != 0:
            raise ValueError(f"SBAIJ local matrix shape {csr.shape} is not divisible by block size {bs}")
        bsr = csr.tobsr(blocksize=(bs, bs))
        block_indptr = [0]
        block_indices: list[int] = []
        block_data: list[np.ndarray] = []
        for block_row in range(int(bsr.indptr.size) - 1):
            start = int(bsr.indptr[block_row])
            stop = int(bsr.indptr[block_row + 1])
            for pos in range(start, stop):
                block_col = int(bsr.indices[pos])
                if block_col >= block_row:
                    block_indices.append(block_col)
                    block_data.append(np.asarray(bsr.data[pos], dtype=np.float64))
            block_indptr.append(len(block_indices))
        block_indptr_arr = np.asarray(block_indptr, dtype=PETSc.IntType)
        block_indices_arr = np.asarray(block_indices, dtype=PETSc.IntType)
        if block_data:
            block_data_arr = np.ascontiguousarray(np.asarray(block_data, dtype=np.float64))
        else:
            block_data_arr = np.empty((0, bs, bs), dtype=np.float64)
        mat = PETSc.Mat().createSBAIJ(
            size=csr.shape,
            bsize=bs,
            csr=(block_indptr_arr, block_indices_arr, block_data_arr),
            comm=PETSc.COMM_SELF,
        )
    else:
        mat = PETSc.Mat().createAIJ(size=csr.shape, csr=(indptr, indices, data), comm=PETSc.COMM_SELF)
    if block_size is not None and normalized != "sbaij":
        mat.setBlockSize(int(block_size))
    mat.assemble()
    _PETSC_MAT_CSR_REFS[int(mat.handle)] = (indptr, indices, data)
    return mat


def local_csr_to_petsc_seq_aij_matrix(A_local):
    """Create a sequential AIJ PETSc matrix from local CSR data."""

    return local_csr_to_petsc_seq_matrix(A_local, mat_type="aij")


def _build_seq_nullspace(basis: np.ndarray | None):
    if PETSc is None or basis is None:
        return None, []
    arr = np.asarray(basis, dtype=np.float64)
    if arr.size == 0:
        return None, []
    if arr.ndim == 1:
        arr = arr[:, None]
    vecs = [
        PETSc.Vec().createWithArray(np.array(arr[:, j], dtype=np.float64, copy=True), size=arr.shape[0], comm=PETSc.COMM_SELF)
        for j in range(arr.shape[1])
    ]
    nsp = PETSc.NullSpace().create(constant=False, vectors=vecs, comm=PETSc.COMM_SELF)
    return nsp, vecs


def _build_dist_nullspace(
    comm,
    basis: np.ndarray | None,
    *,
    global_size: int | None = None,
    ownership_range: tuple[int, int] | None = None,
    block_size: int | None = None,
):
    if PETSc is None or basis is None:
        return None, []
    arr = np.asarray(basis, dtype=np.float64)
    if arr.size == 0:
        return None, []
    if arr.ndim == 1:
        arr = arr[:, None]
    local_size = None if ownership_range is None else int(ownership_range[1] - ownership_range[0])
    vecs = [
        (
            local_array_to_petsc_vec(
                np.asarray(arr[:, j], dtype=np.float64),
                global_size=int(global_size if global_size is not None else arr.shape[0]),
                comm=comm,
                bsize=block_size,
            )
            if ownership_range is not None and local_size is not None and int(arr.shape[0]) == local_size
            else global_array_to_petsc_vec(
                np.asarray(arr[:, j], dtype=np.float64),
                comm=comm,
                ownership_range=ownership_range,
                bsize=block_size,
            )
        )
        for j in range(arr.shape[1])
    ]
    nsp = PETSc.NullSpace().create(constant=False, vectors=vecs, comm=comm)
    return nsp, vecs


def local_csr_to_petsc_matis_matrix(
    A_local,
    *,
    global_size: int,
    local_to_global: np.ndarray,
    comm,
    block_size: int | None = None,
    local_vector_size: int | None = None,
    local_mat_type: str = "aij",
    metadata: dict[str, object] | None = None,
):
    """Create a PETSc MATIS matrix from a local square CSR matrix."""

    _require_petsc()
    stored_metadata = dict(metadata or {})
    local_mat = local_csr_to_petsc_seq_matrix(A_local, mat_type=local_mat_type, block_size=block_size)
    if block_size is not None:
        local_mat.setBlockSize(int(block_size))
    local_nullspace, local_nullspace_vecs = _build_seq_nullspace(stored_metadata.get("bddc_local_nullspace_basis"))
    if local_nullspace is not None:
        local_mat.setNullSpace(local_nullspace)
        stored_metadata["bddc_local_nullspace"] = local_nullspace
        stored_metadata["bddc_local_nullspace_vecs"] = local_nullspace_vecs
    local_near_nullspace, local_near_nullspace_vecs = _build_seq_nullspace(
        stored_metadata.get("bddc_local_near_nullspace_basis")
    )
    if local_near_nullspace is not None:
        local_mat.setNearNullSpace(local_near_nullspace)
        stored_metadata["bddc_local_near_nullspace"] = local_near_nullspace
        stored_metadata["bddc_local_near_nullspace_vecs"] = local_near_nullspace_vecs
    local_to_global_arr = np.asarray(local_to_global, dtype=PETSc.IntType)
    lgmap = PETSc.LGMap().create(local_to_global_arr, comm=comm)
    n_local = int(local_to_global_arr.size)
    vec_local = int(n_local if local_vector_size is None else local_vector_size)
    mat = PETSc.Mat().createIS(
        size=((vec_local, int(global_size)), (vec_local, int(global_size))),
        bsize=block_size,
        lgmapr=lgmap,
        lgmapc=lgmap,
        comm=comm,
    )
    mat.setISAllowRepeated(True)
    mat.setISLocalMat(local_mat)
    if block_size is not None:
        mat.setBlockSize(int(block_size))
    global_near_nullspace, global_near_nullspace_vecs = _build_dist_nullspace(
        comm,
        stored_metadata.get("bddc_global_near_nullspace_basis"),
        global_size=int(global_size),
        ownership_range=mat.getOwnershipRange() if int(comm.getSize()) > 1 else None,
        block_size=block_size,
    )
    if global_near_nullspace is not None:
        mat.setNearNullSpace(global_near_nullspace)
        stored_metadata["bddc_global_near_nullspace"] = global_near_nullspace
        stored_metadata["bddc_global_near_nullspace_vecs"] = global_near_nullspace_vecs
    mat.assemble()
    _PETSC_MAT_IS_REFS[int(mat.handle)] = (local_mat, lgmap)
    stored_metadata.setdefault("local_to_global", np.asarray(local_to_global_arr, dtype=np.int64))
    stored_metadata.setdefault("matis_local_size", int(n_local))
    stored_metadata.setdefault("matis_vector_local_size", int(vec_local))
    stored_metadata.setdefault("matis_global_size", int(global_size))
    stored_metadata.setdefault("matis_local_mat_type", str(local_mat_type).strip().lower())
    _PETSC_MAT_METADATA[int(mat.handle)] = stored_metadata
    return mat


def release_petsc_aij_matrix(A) -> None:
    """Release cached CSR buffers associated with a PETSc matrix."""

    if PETSc is not None and isinstance(A, PETSc.Mat):
        _PETSC_MAT_CSR_REFS.pop(int(A.handle), None)
        _PETSC_MAT_COO_REFS.pop(int(A.handle), None)
        metadata = _PETSC_MAT_METADATA.pop(int(A.handle), None)
        if metadata:
            for value in metadata.values():
                if isinstance(value, (list, tuple)):
                    for item in value:
                        if hasattr(item, "destroy"):
                            try:
                                item.destroy()
                            except Exception:
                                pass
                elif hasattr(value, "destroy"):
                    try:
                        value.destroy()
                    except Exception:
                        pass
        is_refs = _PETSC_MAT_IS_REFS.pop(int(A.handle), None)
        if is_refs is not None:
            local_mat, lgmap = is_refs
            try:
                release_petsc_aij_matrix(local_mat)
            except Exception:
                pass
            try:
                local_mat.destroy()
            except Exception:
                pass
            try:
                lgmap.destroy()
            except Exception:
                pass


def matvec_to_numpy(A, x: np.ndarray) -> np.ndarray:
    """Apply matrix ``A`` to ``x`` and return dense vector.

    Supports dense/sparse/scipy and PETSc matrices.
    """

    if PETSc is not None and isinstance(A, PETSc.Mat):
        xv = global_array_to_petsc_vec(
            x,
            comm=A.getComm(),
            ownership_range=A.getOwnershipRange(),
            bsize=A.getBlockSize() or None,
        )
        y = A.createVecLeft()
        A.mult(xv, y)
        return petsc_vec_to_global_array(y)
    return np.asarray(A @ np.asarray(x, dtype=np.float64), dtype=np.float64).ravel()


def extract_submatrix_free(A, free_idx: np.ndarray):
    """Extract the free-free block used for constrained systems."""

    if PETSc is not None and isinstance(A, PETSc.Mat):
        free_arr = np.asarray(free_idx, dtype=PETSc.IntType).reshape(-1)
        comm = A.getComm()
        if int(comm.getSize()) > 1:
            row0, row1 = A.getOwnershipRange()
            mask = (free_arr >= int(row0)) & (free_arr < int(row1))
            free_arr = np.asarray(free_arr[mask], dtype=PETSc.IntType)
        iset = PETSc.IS().createGeneral(free_arr, comm=comm)
        sub = A.createSubMatrix(iset, iset)
        try:
            block_size = int(sub.getBlockSize() or 1)
            n_global, _m_global = sub.getSize()
            if block_size > 1 and int(n_global) % block_size != 0:
                sub.setBlockSize(1)
        except Exception:
            pass
        return sub
    return A[np.ix_(free_idx, free_idx)]


def to_scipy_csr_from_petsc(A):
    """Convert PETSc matrix into :class:`scipy.sparse.csr_matrix`."""

    import scipy.sparse as sp

    if PETSc is not None and isinstance(A, PETSc.Mat):
        rows, cols, vals = A.getValuesCSR()
        return sp.csr_matrix((vals, cols, rows), shape=A.size)
    return A.tocsr()

def ensure_vectorized_material(values: np.ndarray, target: int, name: str = "material") -> np.ndarray:
    """Broadcast scalar material constants to integration-point arrays."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 1:
        return np.full(target, float(arr[0]), dtype=np.float64)
    if arr.size != target:
        raise ValueError(f"{name} has size {arr.size}, expected {target}.")
    return arr.astype(np.float64)


def flatten_iterable(values: Iterable[float]) -> np.ndarray:
    """Convert scalar/array-like to a 1D float numpy array."""

    return np.asarray(tuple(values), dtype=np.float64)
