"""A-orthogonalization utilities.

These utilities mirror ``A_orthogonalize`` from the MATLAB code.
"""

from __future__ import annotations

import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

try:
    from .. import _kernels as kernels
except Exception:  # pragma: no cover - optional extension
    kernels = None

from ..utils import global_array_to_petsc_vec, petsc_vec_to_global_array


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64)
    if b_arr.ndim == 0:
        return float(a_arr * b_arr)
    if b_arr.ndim == 1:
        b_arr = b_arr.ravel()
        if kernels is None:
            return float(np.dot(a_arr, b_arr))
        try:
            return float(kernels.dot(a_arr, b_arr))
        except Exception:
            return float(np.dot(a_arr, b_arr))
    if b_arr.ndim == 2:
        if b_arr.shape[0] == 1:
            return float(np.dot(a_arr, b_arr[0, :]))
        if b_arr.shape[1] == 1:
            return float(np.dot(a_arr, b_arr[:, 0]))
        if b_arr.shape[1] == a_arr.size:
            return b_arr @ a_arr
        if b_arr.shape[0] == a_arr.size:
            return b_arr.T @ a_arr
        # Fallback for unexpected layout.
        return np.dot(a_arr, b_arr.T)
    if kernels is None:
        return float(np.dot(a_arr, b_arr))
    return float(kernels.dot(a_arr, b_arr))


def _a_orthogonalize_impl(
    W: np.ndarray,
    A,
    eps_add: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror MATLAB ``A_orthogonalize`` and return basis, signed norms, and kept source indices."""

    if W is None:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    W_mat = np.asarray(W, dtype=np.float64)
    if W_mat.size == 0:
        rows = int(W_mat.shape[0]) if W_mat.ndim >= 1 else 0
        return np.empty((rows, 0), dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    if W_mat.ndim == 1:
        W_mat = W_mat[:, None]

    if W_mat.shape[1] == 0:
        return np.empty((W_mat.shape[0], 0), dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    def apply_A(x):
        if isinstance(A, np.ndarray):
            return A @ x
        if hasattr(A, "mult"):
            comm = A.getComm() if hasattr(A, "getComm") else PETSc.COMM_WORLD
            ownership_range = A.getOwnershipRange() if hasattr(A, "getOwnershipRange") and int(comm.getSize()) > 1 else None
            x_vec = global_array_to_petsc_vec(
                x,
                comm=comm,
                ownership_range=ownership_range,
                bsize=A.getBlockSize() if hasattr(A, "getBlockSize") else None,
            )
            y_vec = A.createVecRight()
            A.mult(x_vec, y_vec)
            return petsc_vec_to_global_array(y_vec)
        return A @ x

    def a_orthogonalize_petsc_distributed() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        comm = A.getComm()
        mpi_comm = comm.tompi4py()
        r0, r1 = A.getOwnershipRange()
        n_local = int(r1 - r0)

        x_vec = A.createVecRight()
        y_vec = A.createVecRight()
        x_arr = x_vec.getArray(readonly=False)

        w_orth = np.zeros((n_rows, n_cols), dtype=np.float64)
        norms = np.ones(n_cols, dtype=np.float64)
        keep = np.zeros(n_cols, dtype=bool)
        keep_source_idx = np.full(n_cols, -1, dtype=np.int64)

        for i in range(n_cols):
            idx = n_cols - i - 1
            v = W_mat[:, idx].copy()

            x_arr[...] = v[r0:r1]
            A.mult(x_vec, y_vec)
            a_v_local = np.asarray(y_vec.getArray(readonly=True), dtype=np.float64).copy()

            if i > 0:
                w_prev_local = w_orth[r0:r1, :i]
                coeff_local = w_prev_local.T @ a_v_local
                coeff = np.asarray(mpi_comm.allreduce(coeff_local, op=MPI.SUM), dtype=np.float64)
                v = v - (w_orth[:, :i] * norms[:i]) @ coeff

            v_local = v[r0:r1]
            norm_val = float(mpi_comm.allreduce(float(np.dot(v_local, a_v_local)), op=MPI.SUM))

            if abs(norm_val) > eps_add:
                if norm_val > 0:
                    scale = np.sqrt(norm_val)
                else:
                    scale = np.sqrt(abs(norm_val))
                    norms[i] = -1.0
                w_orth[:, i] = v / scale
                keep[i] = True
                keep_source_idx[i] = idx

        if not np.any(keep):
            return np.empty((n_rows, 0), dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

        kept_cols = w_orth[:, keep]
        kept_norms = norms[keep]
        kept_source_idx = keep_source_idx[keep]
        return kept_cols[:, ::-1], kept_norms[::-1], kept_source_idx[::-1]

    n_rows, n_cols = W_mat.shape
    if hasattr(A, "mult") and hasattr(A, "getComm") and int(A.getComm().getSize()) > 1:
        return a_orthogonalize_petsc_distributed()

    w_orth = np.zeros((n_rows, n_cols), dtype=np.float64)
    norms = np.ones(n_cols, dtype=np.float64)
    keep = np.zeros(n_cols, dtype=bool)
    keep_source_idx = np.full(n_cols, -1, dtype=np.int64)
    for i in range(n_cols):
        idx = n_cols - i - 1
        v = W_mat[:, idx].copy()
        a_v = apply_A(v)
        if i > 0:
            coeff = np.asarray(_dot(a_v, w_orth[:, :i].T), dtype=np.float64).reshape(-1)
            v = v - (w_orth[:, :i] * norms[:i]) @ coeff
        norm_val = _dot(v, a_v)

        if abs(norm_val) > eps_add:
            if norm_val > 0:
                scale = np.sqrt(norm_val)
            else:
                scale = np.sqrt(abs(norm_val))
                norms[i] = -1.0
            w_orth[:, i] = v / scale
            keep[i] = True
            keep_source_idx[i] = idx

    if not np.any(keep):
        return np.empty((n_rows, 0), dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    kept_cols = w_orth[:, keep]
    kept_norms = norms[keep]
    kept_source_idx = keep_source_idx[keep]
    return kept_cols[:, ::-1], kept_norms[::-1], kept_source_idx[::-1]


def a_orthogonalize_with_local_metadata(
    W: np.ndarray,
    A,
    eps_add: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return distributed A-orthogonal basis using only owned-row local slices.

    ``W`` is still accepted as the full dense basis for compatibility, but only
    the owned row range of the PETSc matrix is touched in the hot path.
    """

    if W is None:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    if not (hasattr(A, "mult") and hasattr(A, "getComm") and int(A.getComm().getSize()) > 1):
        basis, norms, kept = _a_orthogonalize_impl(W, A, eps_add)
        return basis, norms, kept

    W_mat = np.asarray(W, dtype=np.float64)
    if W_mat.ndim == 1:
        W_mat = W_mat[:, None]
    if W_mat.size == 0 or W_mat.shape[1] == 0:
        n_local = int(A.getOwnershipRange()[1] - A.getOwnershipRange()[0])
        return np.empty((n_local, 0), dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    mpi_comm = A.getComm().tompi4py()
    r0, r1 = A.getOwnershipRange()
    n_local = int(r1 - r0)
    n_cols = int(W_mat.shape[1])
    W_local = np.asarray(W_mat[r0:r1, :], dtype=np.float64)

    x_vec = A.createVecRight()
    y_vec = A.createVecRight()
    x_arr = x_vec.getArray(readonly=False)

    w_orth_local = np.zeros((n_local, n_cols), dtype=np.float64)
    norms = np.ones(n_cols, dtype=np.float64)
    keep = np.zeros(n_cols, dtype=bool)
    keep_source_idx = np.full(n_cols, -1, dtype=np.int64)

    for i in range(n_cols):
        idx = n_cols - i - 1
        v_local = W_local[:, idx].copy()

        x_arr[...] = v_local
        A.mult(x_vec, y_vec)
        a_v_local = np.asarray(y_vec.getArray(readonly=True), dtype=np.float64).copy()

        if i > 0:
            w_prev_local = w_orth_local[:, :i]
            coeff_local = w_prev_local.T @ a_v_local
            coeff = np.asarray(mpi_comm.allreduce(coeff_local, op=MPI.SUM), dtype=np.float64)
            v_local = v_local - (w_prev_local * norms[:i]) @ coeff

        norm_val = float(mpi_comm.allreduce(float(np.dot(v_local, a_v_local)), op=MPI.SUM))
        if abs(norm_val) > eps_add:
            if norm_val > 0:
                scale = np.sqrt(norm_val)
            else:
                scale = np.sqrt(abs(norm_val))
                norms[i] = -1.0
            w_orth_local[:, i] = v_local / scale
            keep[i] = True
            keep_source_idx[i] = idx

    if not np.any(keep):
        return np.empty((n_local, 0), dtype=np.float64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    kept_cols = w_orth_local[:, keep]
    kept_norms = norms[keep]
    kept_source_idx = keep_source_idx[keep]
    return kept_cols[:, ::-1], kept_norms[::-1], kept_source_idx[::-1]


def a_orthogonalize(W: np.ndarray, A, eps_add: float = 1e-3) -> np.ndarray:
    """A-orthogonalize columns of ``W`` in-place."""

    basis, _norms, _kept = _a_orthogonalize_impl(W, A, eps_add)
    return basis


def a_orthogonalize_with_info(W: np.ndarray, A, eps_add: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Return MATLAB-style A-orthogonal basis together with signed column norms."""

    basis, norms, _kept = _a_orthogonalize_impl(W, A, eps_add)
    return basis, norms


def a_orthogonalize_with_metadata(
    W: np.ndarray,
    A,
    eps_add: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return MATLAB-style A-orthogonal basis, signed norms, and kept original column indices."""

    return _a_orthogonalize_impl(W, A, eps_add)
