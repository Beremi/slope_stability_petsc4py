"""Preconditioners used by custom PETSc-based solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from petsc4py import PETSc

from ..utils import global_array_to_petsc_vec, to_petsc_aij_matrix


PreconditionerApply = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PreconditionerSetupResult:
    apply: PreconditionerApply
    metadata: dict[str, float | int | str]


def _as_dense_matrix(A) -> np.ndarray:
    if isinstance(A, np.ndarray):
        return np.asarray(A, dtype=np.float64)
    if hasattr(A, "convert"):
        return A.convert("dense").getDenseArray()
    return np.asarray(A.todense())


def make_near_nullspace_elasticity(
    coord: np.ndarray,
    q_mask: np.ndarray | None = None,
    center_coordinates: bool = True,
    return_full: bool = False,
) -> np.ndarray:
    """Build near-nullspace vectors for linear elasticity.

    Returns columns of an orthonormal basis for rigid modes.
    """

    dim, n_nodes = coord.shape
    if q_mask is None:
        q_mask = np.ones((dim, n_nodes), dtype=bool)
    q_mask = np.asarray(q_mask, dtype=bool)
    if q_mask.shape != (dim, n_nodes):
        raise ValueError("q_mask must have shape (dim, n_nodes)")

    free_mask = q_mask.ravel(order="F")
    if not free_mask.any():
        n_rows = coord.size if return_full else 0
        return np.empty((n_rows, 0), dtype=np.float64)

    if dim == 2:
        x = coord[0, :].copy()
        y = coord[1, :].copy()
        if center_coordinates:
            x -= x.mean()
            y -= y.mean()

        modes = np.zeros((coord.size, 3), dtype=np.float64)
        # tx
        mode = np.zeros((dim, n_nodes), dtype=np.float64)
        mode[0, :] = 1.0
        modes[:, 0] = mode.ravel(order="F")
        # ty
        mode = np.zeros((dim, n_nodes), dtype=np.float64)
        mode[1, :] = 1.0
        modes[:, 1] = mode.ravel(order="F")
        # rigid rotation
        mode = np.zeros((dim, n_nodes), dtype=np.float64)
        mode[0, :] = -y
        mode[1, :] = x
        modes[:, 2] = mode.ravel(order="F")
    elif dim == 3:
        x = coord[0, :].copy()
        y = coord[1, :].copy()
        z = coord[2, :].copy()
        if center_coordinates:
            x -= x.mean()
            y -= y.mean()
            z -= z.mean()

        modes = np.zeros((coord.size, 6), dtype=np.float64)
        # translations
        for i, (a, b, c) in enumerate(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))):
            mode = np.zeros((dim, n_nodes), dtype=np.float64)
            mode[0, :] = a
            mode[1, :] = b
            mode[2, :] = c
            modes[:, i] = mode.ravel(order="F")

        # rotation about x, y, z
        mode = np.zeros((dim, n_nodes), dtype=np.float64)
        mode[1, :] = -z
        mode[2, :] = y
        modes[:, 3] = mode.ravel(order="F")

        mode = np.zeros((dim, n_nodes), dtype=np.float64)
        mode[0, :] = z
        mode[2, :] = -x
        modes[:, 4] = mode.ravel(order="F")

        mode = np.zeros((dim, n_nodes), dtype=np.float64)
        mode[0, :] = -y
        mode[1, :] = x
        modes[:, 5] = mode.ravel(order="F")
    else:
        raise ValueError("Unsupported dimension for nullspace construction")

    # Restrict to free DOFs
    modes_free = modes[free_mask, :]
    # Stable Gram-Schmidt with Euclidean inner product
    basis = []
    for i in range(modes_free.shape[1]):
        v = modes_free[:, i].copy()
        if not basis:
            pass
        for existing in basis:
            v -= existing * float(existing @ v)
        nrm = np.linalg.norm(v)
        if nrm > 1e-12 * np.sqrt(max(1, v.size)):
            basis.append(v / nrm)

    if not basis:
        n_rows = coord.size if return_full else free_mask.sum()
        return np.empty((n_rows, 0), dtype=np.float64)
    Z = np.column_stack(basis)
    if not return_full:
        return Z

    Z_full = np.zeros((coord.size, Z.shape[1]), dtype=np.float64)
    Z_full[free_mask, :] = Z
    return Z_full


def _build_petsc_nullspace(
    comm: PETSc.Comm,
    basis: np.ndarray | None,
    *,
    ownership_range: tuple[int, int] | None = None,
    block_size: int | None = None,
):
    if basis is None:
        return None, []

    basis = np.asarray(basis, dtype=np.float64)
    if basis.size == 0:
        return None, []
    if basis.ndim == 1:
        basis = basis[:, None]

    vecs = [
        global_array_to_petsc_vec(
            basis[:, j],
            comm=comm,
            ownership_range=ownership_range,
            bsize=block_size,
        )
        for j in range(basis.shape[1])
    ]
    nsp = PETSc.NullSpace().create(constant=False, vectors=vecs, comm=comm)
    return nsp, vecs


def attach_near_nullspace(A, basis: np.ndarray | None):
    """Attach a PETSc near-nullspace to ``A`` and return owned PETSc objects."""

    A_petsc = to_petsc_aij_matrix(A, comm=A.getComm() if hasattr(A, "getComm") else PETSc.COMM_SELF)
    block_size = A_petsc.getBlockSize() or None
    nsp, vecs = _build_petsc_nullspace(
        A_petsc.getComm(),
        basis,
        ownership_range=A_petsc.getOwnershipRange() if int(A_petsc.getComm().getSize()) > 1 else None,
        block_size=block_size,
    )
    if nsp is not None:
        A_petsc.setNearNullSpace(nsp)
    return A_petsc, nsp, vecs


class JacobiPreconditioner:
    """Simple block-diagonal preconditioner with blocks defined by Q rows."""

    def __init__(self, A, Q: np.ndarray):
        self._comm = A.getComm() if hasattr(A, "getComm") else PETSc.COMM_WORLD
        self.blocks: list[np.ndarray] = []
        q_mask = np.asarray(Q, dtype=bool)
        if q_mask.size == 0:
            self.blocks = []
            return

        free = q_mask.ravel(order="F")
        ncomp = q_mask.shape[0]
        A_dense = _as_dense_matrix(A)

        for i in range(ncomp):
            row_mask = np.zeros_like(free)
            row_mask[i::ncomp] = q_mask[i].ravel(order="F")
            idx = np.flatnonzero(row_mask)
            if idx.size == 0:
                continue
            block = np.asarray(A_dense[np.ix_(idx, idx)], dtype=np.float64)
            reg = 1e-2 * np.maximum(1.0, np.abs(np.diag(block)))
            block = block + np.diag(reg)
            self.blocks.append((idx, np.linalg.cholesky(block).T))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(np.asarray(x, dtype=np.float64))
        if len(self.blocks) == 0:
            return y
        for idx, L in self.blocks:
            rhs = x[idx]
            z = np.linalg.solve(L.T, rhs)
            y[idx] = np.linalg.solve(L, z)
        return y


class GAMGPreconditioner:
    """PETSc GAMG-based preconditioner used inside custom Krylov solves."""

    def __init__(self, A, null_space: Optional[np.ndarray] = None, options: Optional[dict] = None):
        if options is None:
            options = {}
        self._A, self._near_nullspace, self._near_nullspace_vecs = attach_near_nullspace(A, null_space)
        self._comm = self._A.getComm()
        self.ksp = PETSc.KSP().create(comm=self._comm)
        self.ksp.setOperators(self._A)
        self.ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = self.ksp.getPC()
        pc.setType(PETSc.PC.Type.GAMG)

        # Optional GAMG tuning.
        print_level = int(options.get("print_level", 0))
        self.ksp.setTolerances(rtol=1e-6, atol=1e-30, max_it=50)

        # setOptionsPrefix can be used by advanced users via command-line options too
        if print_level > 0:
            self.ksp.setInitialGuessNonzero(False)
        else:
            self.ksp.setInitialGuessNonzero(False)

        pc.setFromOptions()
        self.ksp.setUp()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_vec = PETSc.Vec().createWithArray(np.asarray(x, dtype=np.float64), comm=self._comm)
        y_vec = self._A.createVecLeft()
        y_vec.set(0)
        self.ksp.solve(x_vec, y_vec)
        if self.ksp.getConvergedReason() <= 0:
            # Fallback gracefully.
            return x
        return y_vec.getArray(readonly=False).copy()


def build_preconditioner(name: str, A, q_mask: np.ndarray, coord: np.ndarray | None = None, **kwargs) -> PreconditionerSetupResult:
    name = name.upper()
    if name == "GAMG":
        null_space = kwargs.get("null_space")
        if null_space is None and coord is not None:
            null_space = make_near_nullspace_elasticity(
                coord,
                q_mask=q_mask,
                center_coordinates=True,
                return_full=False,
            )
        precond = GAMGPreconditioner(A, null_space=null_space, options=kwargs)
    elif name == "JACOBI":
        precond = JacobiPreconditioner(A, q_mask)
    else:
        precond = JacobiPreconditioner(A, q_mask)
    return PreconditionerSetupResult(apply=precond, metadata={"name": name})
