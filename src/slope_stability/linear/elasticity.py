"""Helpers for standalone linear-elasticity PETSc probes."""

from __future__ import annotations

import numpy as np
from petsc4py import PETSc
from scipy.sparse import diags

from ..utils import flatten_field


def impose_zero_dirichlet_full_system(A, rhs, q_mask: np.ndarray):
    """Impose homogeneous Dirichlet conditions without destroying node ordering.

    The matrix keeps its full size. Constrained rows and columns are zeroed and
    constrained diagonal entries are set to one.
    """

    free_mask = np.asarray(q_mask, dtype=bool).reshape(-1, order="F")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if A.shape[0] != free_mask.size:
        raise ValueError("A and q_mask size mismatch")

    rhs_arr = np.asarray(rhs, dtype=np.float64)
    rhs_vec = flatten_field(rhs_arr) if rhs_arr.ndim == 2 else rhs_arr.reshape(-1).copy()
    if rhs_vec.size != free_mask.size:
        raise ValueError("rhs and q_mask size mismatch")

    active = free_mask.astype(np.float64)
    fixed = (~free_mask).astype(np.float64)

    A_csr = A.tocsr()
    # Masking rows and columns preserves the original sparsity pattern order.
    A_bc = A_csr.multiply(active[:, None]).multiply(active[None, :]) + diags(fixed, format="csr")
    A_bc.eliminate_zeros()

    rhs_bc = rhs_vec.copy()
    rhs_bc[~free_mask] = 0.0
    return A_bc, rhs_bc, free_mask


def create_rigid_body_near_nullspace(coord: np.ndarray, *, comm: PETSc.Comm = PETSc.COMM_SELF):
    """Create PETSc rigid-body near-nullspace data from nodal coordinates."""

    coord = np.asarray(coord, dtype=np.float64)
    if coord.ndim != 2:
        raise ValueError("coord must be (dim, n_nodes)")

    dim = coord.shape[0]
    coords_flat = coord.reshape(-1, order="F").copy()
    coords_vec = PETSc.Vec().createWithArray(coords_flat, comm=comm)
    coords_vec.setBlockSize(dim)
    near_nullspace = PETSc.NullSpace().createRigidBody(coords_vec)
    return coords_vec, near_nullspace


def attach_rigid_body_near_nullspace(A: PETSc.Mat, coord: np.ndarray):
    """Attach PETSc's rigid-body near-nullspace to a matrix."""

    dim = int(np.asarray(coord).shape[0])
    A.setBlockSize(dim)
    coords_vec, near_nullspace = create_rigid_body_near_nullspace(coord, comm=A.getComm())
    A.setNearNullSpace(near_nullspace)
    return coords_vec, near_nullspace
