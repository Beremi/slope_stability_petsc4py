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
):
    """Create a distributed PETSc AIJ matrix from already-owned CSR rows."""

    import scipy.sparse as sp

    _require_petsc()
    csr = A_local.tocsr() if sp.issparse(A_local) else sp.csr_matrix(np.asarray(A_local, dtype=np.float64))
    indptr = np.array(csr.indptr, dtype=PETSc.IntType, copy=True)
    indices = np.array(csr.indices, dtype=PETSc.IntType, copy=True)
    data = np.array(csr.data, dtype=np.float64, copy=True)
    mat = PETSc.Mat().createAIJ(
        size=((csr.shape[0], int(global_shape[0])), (csr.shape[0], int(global_shape[1]))),
        csr=(indptr, indices, data),
        comm=comm,
    )
    if block_size is not None:
        mat.setBlockSize(int(block_size))
    mat.assemble()
    _PETSC_MAT_CSR_REFS[int(mat.handle)] = (indptr, indices, data)
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


def get_petsc_is_local_mat(A):
    """Return the locally stored SeqAIJ matrix for a MATIS matrix if registered."""

    if PETSc is None or not isinstance(A, PETSc.Mat):
        return None
    refs = _PETSC_MAT_IS_REFS.get(int(A.handle))
    if refs is None:
        return None
    return refs[0]


def local_csr_to_petsc_seq_aij_matrix(A_local):
    """Create a sequential AIJ PETSc matrix from local CSR data."""

    import scipy.sparse as sp

    _require_petsc()
    csr = A_local.tocsr() if sp.issparse(A_local) else sp.csr_matrix(np.asarray(A_local, dtype=np.float64))
    indptr = np.array(csr.indptr, dtype=PETSc.IntType, copy=True)
    indices = np.array(csr.indices, dtype=PETSc.IntType, copy=True)
    data = np.array(csr.data, dtype=np.float64, copy=True)
    mat = PETSc.Mat().createAIJ(size=csr.shape, csr=(indptr, indices, data), comm=PETSc.COMM_SELF)
    mat.assemble()
    _PETSC_MAT_CSR_REFS[int(mat.handle)] = (indptr, indices, data)
    return mat


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


def local_csr_to_petsc_matis_matrix(
    A_local,
    *,
    global_size: int,
    local_to_global: np.ndarray,
    comm,
    block_size: int | None = None,
    local_vector_size: int | None = None,
    metadata: dict[str, object] | None = None,
):
    """Create a PETSc MATIS matrix from a local square CSR matrix."""

    _require_petsc()
    stored_metadata = dict(metadata or {})
    local_mat = local_csr_to_petsc_seq_aij_matrix(A_local)
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
    mat.assemble()
    _PETSC_MAT_IS_REFS[int(mat.handle)] = (local_mat, lgmap)
    stored_metadata.setdefault("local_to_global", np.asarray(local_to_global_arr, dtype=np.int64))
    stored_metadata.setdefault("matis_local_size", int(n_local))
    stored_metadata.setdefault("matis_vector_local_size", int(vec_local))
    stored_metadata.setdefault("matis_global_size", int(global_size))
    _PETSC_MAT_METADATA[int(mat.handle)] = stored_metadata
    return mat


def release_petsc_aij_matrix(A) -> None:
    """Release cached CSR buffers associated with a PETSc matrix."""

    if PETSc is not None and isinstance(A, PETSc.Mat):
        _PETSC_MAT_CSR_REFS.pop(int(A.handle), None)
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
        iset = PETSc.IS().createGeneral(free_idx.astype(PETSc.IntType), comm=A.getComm())
        return A.createSubMatrix(iset, iset)
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
