from __future__ import annotations

import json

import numpy as np
from petsc4py import PETSc
from scipy.sparse import csr_matrix

from slope_stability.linear.solver import PetscMatlabExactDFGMRESSolver
from slope_stability.utils import local_csr_to_petsc_aij_matrix, local_csr_to_petsc_matis_matrix, matvec_to_numpy


def _distributed_tridiag_operator():
    comm = PETSc.COMM_WORLD
    rank = int(comm.getRank())
    if int(comm.getSize()) != 2:
        raise RuntimeError("This overlap check expects exactly 2 ranks")

    row_base = 2 * rank
    data: list[float] = []
    rows: list[int] = []
    cols: list[int] = []
    for row_local in range(2):
        row_global = row_base + row_local
        rows.append(row_local)
        cols.append(row_global)
        data.append(4.0)
        if row_global > 0:
            rows.append(row_local)
            cols.append(row_global - 1)
            data.append(-1.0)
        if row_global < 3:
            rows.append(row_local)
            cols.append(row_global + 1)
            data.append(-1.0)
    local = csr_matrix((data, (rows, cols)), shape=(2, 4), dtype=np.float64)
    return local_csr_to_petsc_aij_matrix(local, global_shape=(4, 4), comm=comm, block_size=1)


def _overlap_bddc_preconditioner():
    comm = PETSc.COMM_WORLD
    rank = int(comm.getRank())
    if rank == 0:
        local_to_global = np.asarray([0, 1, 2], dtype=np.int64)
    else:
        local_to_global = np.asarray([1, 2, 3], dtype=np.int64)
    local = csr_matrix(
        np.array(
            [
                [4.0, -1.0, 0.0],
                [-1.0, 4.0, -1.0],
                [0.0, -1.0, 4.0],
            ],
            dtype=np.float64,
        )
    )
    return local_csr_to_petsc_matis_matrix(
        local,
        global_size=4,
        local_to_global=local_to_global,
        local_vector_size=2,
        comm=comm,
        block_size=1,
        metadata={
            "bddc_field_is_local": (
                PETSc.IS().createGeneral(np.asarray([0, 1, 2], dtype=PETSc.IntType), comm=PETSc.COMM_SELF),
            ),
            "bddc_dirichlet_local": np.asarray([], dtype=np.int32),
            "bddc_local_adjacency": (
                np.asarray(local.indptr, dtype=PETSc.IntType),
                np.asarray(local.indices, dtype=PETSc.IntType),
            ),
        },
    )


def main() -> None:
    comm = PETSc.COMM_WORLD
    A = _distributed_tridiag_operator()
    P = _overlap_bddc_preconditioner()
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type="JACOBI",
        q_mask=np.ones((1, 4), dtype=bool),
        coord=np.arange(4, dtype=np.float64)[None, :],
        preconditioner_options={
            "pc_backend": "bddc",
            "preconditioner_matrix_policy": "current",
            "preconditioner_rebuild_policy": "every_newton",
        },
        tolerance=1.0e-10,
        max_iterations=50,
    )
    rhs = np.ones(4, dtype=np.float64)
    solver.setup_preconditioner(A, full_matrix=A, preconditioning_matrix=P)
    x = solver.solve(A, rhs, full_rhs=rhs)
    residual = rhs - matvec_to_numpy(A, x)
    payload = {
        "solution": np.asarray(x, dtype=np.float64).tolist(),
        "residual_norm": float(np.linalg.norm(residual)),
        **solver.get_preconditioner_diagnostics(),
    }
    gathered = comm.tompi4py().gather(payload, root=0)
    if int(comm.getRank()) == 0:
        print(json.dumps({"results": gathered}, sort_keys=True))
    solver.release_iteration_resources()
    P.destroy()
    A.destroy()


if __name__ == "__main__":
    main()
