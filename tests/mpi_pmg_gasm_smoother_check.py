from __future__ import annotations

import json
from types import SimpleNamespace

from mpi4py import MPI
import numpy as np
from petsc4py import PETSc
from scipy.sparse import csr_matrix

from slope_stability.linear.solver import PetscMatlabExactDFGMRESSolver, _ManualPMGShellPC
from slope_stability.utils import local_csr_to_petsc_aij_matrix


def _operator():
    comm = PETSc.COMM_WORLD
    rank = int(comm.getRank())
    size = int(comm.getSize())
    rows: list[int] = [0]
    cols: list[int] = [rank]
    data: list[float] = [2.0]
    if rank > 0:
        rows.append(0)
        cols.append(rank - 1)
        data.append(-0.25)
    if rank + 1 < size:
        rows.append(0)
        cols.append(rank + 1)
        data.append(-0.25)
    local = csr_matrix((data, (rows, cols)), shape=(1, size), dtype=np.float64)
    return local_csr_to_petsc_aij_matrix(local, global_shape=(size, size), comm=comm)


def main() -> None:
    comm = PETSc.COMM_WORLD
    size = int(comm.getSize())
    if size != 4:
        raise RuntimeError(f"Expected 4 MPI ranks for the PMG GASM smoke, got {size}.")

    A = _operator()
    dummy_solver = SimpleNamespace(
        preconditioner_options={
            "mg_levels_ksp_type": "richardson",
            "mg_levels_ksp_max_it": 3,
            "pmg_smoother_pc_type": "gasm",
            "pmg_smoother_gasm_total_subdomains": 2,
            "pmg_smoother_gasm_grouping": "contiguous",
            "pmg_smoother_gasm_overlap": 1,
            "pmg_smoother_gasm_type": "restrict",
            "pmg_smoother_gasm_sub_ksp_type": "preonly",
            "pmg_smoother_gasm_sub_ksp_max_it": 1,
            "pmg_smoother_gasm_sub_pc_type": "jacobi",
            "pmg_smoother_gasm_view_subdomains": False,
        },
        _set_petsc_option=PetscMatlabExactDFGMRESSolver._set_petsc_option,
    )
    context = _ManualPMGShellPC(dummy_solver)
    smoother = context._build_smoother(A, prefix="pmg_gasm_smoke_", hierarchy=SimpleNamespace(levels=()))

    b = A.createVecRight()
    x = A.createVecRight()
    Ax = A.createVecRight()
    b.set(1.0)
    x.set(0.0)
    smoother.solve(b, x)
    A.mult(x, Ax)
    residual = np.asarray(b.getArray(readonly=True), dtype=np.float64) - np.asarray(
        Ax.getArray(readonly=True), dtype=np.float64
    )
    local_norm = float(np.linalg.norm(residual))
    max_norm = float(comm.tompi4py().allreduce(local_norm, op=MPI.MAX))
    payload = {
        "pc_type": str(smoother.getPC().getType()),
        "ksp_type": str(smoother.getType()),
        "sub_ksp_type": "preonly",
        "total_subdomains": 2,
        "ranks_per_subdomain": 2,
        "residual_norm_max": max_norm,
        "solution_norm_local": float(np.linalg.norm(np.asarray(x.getArray(readonly=True), dtype=np.float64))),
    }
    if int(comm.getRank()) == 0:
        print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
