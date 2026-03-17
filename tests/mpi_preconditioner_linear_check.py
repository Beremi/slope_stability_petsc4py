from __future__ import annotations

import json

import numpy as np
from petsc4py import PETSc
from scipy.sparse import csr_matrix

from slope_stability.linear.solver import PetscMatlabExactDFGMRESSolver
from slope_stability.utils import local_csr_to_petsc_aij_matrix, local_csr_to_petsc_matis_matrix, matvec_to_numpy


def _distributed_block_operator():
    comm = PETSc.COMM_WORLD
    rank = int(comm.getRank())
    size = int(comm.getSize())
    dim = 2
    n = dim * size
    row_base = dim * rank
    data: list[float] = []
    rows: list[int] = []
    cols: list[int] = []
    for comp in range(dim):
        rows.append(comp)
        cols.append(row_base + comp)
        data.append(4.0)
        if rank > 0:
            rows.append(comp)
            cols.append(row_base + comp - dim)
            data.append(-1.0)
        if rank + 1 < size:
            rows.append(comp)
            cols.append(row_base + comp + dim)
            data.append(-1.0)
    local = csr_matrix((data, (rows, cols)), shape=(dim, n), dtype=np.float64)
    return local_csr_to_petsc_aij_matrix(local, global_shape=(n, n), comm=comm, block_size=dim), n


def _bddc_preconditioner(global_size: int):
    comm = PETSc.COMM_WORLD
    rank = int(comm.getRank())
    local = csr_matrix(np.diag([4.0, 4.0]), dtype=np.float64)
    field_is = tuple(
        PETSc.IS().createGeneral(np.asarray([comp], dtype=PETSc.IntType), comm=PETSc.COMM_SELF)
        for comp in range(2)
    )
    return local_csr_to_petsc_matis_matrix(
        local,
        global_size=global_size,
        local_to_global=np.asarray([2 * rank, 2 * rank + 1], dtype=np.int64),
        comm=comm,
        block_size=2,
        metadata={
            "bddc_field_is_local": field_is,
            "bddc_dirichlet_local": np.asarray([], dtype=np.int32),
            "bddc_local_adjacency": (
                np.asarray(local.indptr, dtype=PETSc.IntType),
                np.asarray(local.indices, dtype=PETSc.IntType),
            ),
        },
    )


def _solve_case(name: str, A, global_size: int) -> dict[str, object]:
    opts = {
        "pc_backend": "hypre" if "hypre" in name else "bddc",
        "preconditioner_matrix_policy": "current",
        "preconditioner_rebuild_policy": "every_newton",
    }
    pc_type = "HYPRE" if "hypre" in name else "JACOBI"
    if name == "hypre_lagged_current":
        opts["preconditioner_matrix_policy"] = "lagged"
        opts["preconditioner_rebuild_policy"] = "accepted_step"
    solver = PetscMatlabExactDFGMRESSolver(
        pc_type=pc_type,
        q_mask=np.ones((2, global_size // 2), dtype=bool),
        coord=np.vstack((np.arange(global_size // 2, dtype=np.float64), np.zeros(global_size // 2, dtype=np.float64))),
        preconditioner_options=opts,
        tolerance=1.0e-10,
        max_iterations=50,
    )
    rhs = np.ones(global_size, dtype=np.float64)
    P = _bddc_preconditioner(global_size) if name == "bddc" else None
    solver.setup_preconditioner(A, full_matrix=A, preconditioning_matrix=P)
    x = solver.solve(A, rhs, full_rhs=rhs)
    residual = rhs - matvec_to_numpy(A, x)
    payload = {
        "case": name,
        "solution_norm": float(np.linalg.norm(x)),
        "residual_norm": float(np.linalg.norm(residual)),
        **solver.get_preconditioner_diagnostics(),
    }
    solver.release_iteration_resources()
    if P is not None:
        P.destroy()
    return payload


def main() -> None:
    A, global_size = _distributed_block_operator()
    results = [_solve_case(name, A, global_size) for name in ("hypre_current", "hypre_lagged_current", "bddc")]
    gathered = PETSc.COMM_WORLD.tompi4py().gather(results if PETSc.COMM_WORLD.getRank() == 0 else None, root=0)
    if PETSc.COMM_WORLD.getRank() == 0:
        print(json.dumps({"results": results}, sort_keys=True))


if __name__ == "__main__":
    main()
