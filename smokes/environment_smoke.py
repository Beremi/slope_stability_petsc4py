#!/usr/bin/env python
"""Fast dependency, MPI, PETSc, and asset-registry smoke check."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _module_version(name: str) -> str:
    module = importlib.import_module(name)
    return str(getattr(module, "__version__", "unknown"))


def _tiny_petsc_solve(PETSc, comm, *, pc_type: str, factor_solver_type: str | None = None) -> dict[str, Any]:
    from mpi4py import MPI

    rank = int(comm.getRank())
    size = int(comm.getSize())
    n = max(1, size)
    mat = None
    ksp = None
    rhs = None
    sol = None
    try:
        mat = PETSc.Mat().createAIJ([n, n], nnz=1, comm=comm)
        mat.setUp()
        r0, r1 = mat.getOwnershipRange()
        for row in range(int(r0), int(r1)):
            mat.setValue(row, row, 2.0)
        mat.assemblyBegin()
        mat.assemblyEnd()

        rhs = mat.createVecRight()
        sol = mat.createVecRight()
        rhs.set(4.0)
        sol.set(0.0)

        ksp = PETSc.KSP().create(comm=comm)
        ksp.setType(PETSc.KSP.Type.PREONLY if pc_type == "lu" else PETSc.KSP.Type.CG)
        ksp.setOperators(mat)
        pc = ksp.getPC()
        pc.setType(pc_type)
        if factor_solver_type:
            pc.setFactorSolverType(factor_solver_type)
        ksp.setTolerances(rtol=1.0e-12, max_it=20)
        ksp.setUp()
        ksp.solve(rhs, sol)
        local_error = float(max(abs(value - 2.0) for value in sol.getArray(readonly=True))) if r1 > r0 else 0.0
        error = float(comm.tompi4py().allreduce(local_error, op=MPI.MAX))
        ok = bool(error < 1.0e-9)
        return {
            "ok": ok,
            "rank": rank,
            "pc_type": pc_type,
            "factor_solver_type": factor_solver_type,
            "iterations": int(ksp.getIterationNumber()),
            "max_error": error,
        }
    except Exception as exc:  # pragma: no cover - depends on site PETSc build
        return {
            "ok": False,
            "rank": rank,
            "pc_type": pc_type,
            "factor_solver_type": factor_solver_type,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        for obj in (ksp, rhs, sol, mat):
            destroy = getattr(obj, "destroy", None)
            if destroy is not None:
                destroy()


def _asset_probe() -> dict[str, Any]:
    from slope_stability.assets import available_problem_assets, load_problem_asset

    assets = available_problem_assets()
    asset = load_problem_asset("2d_homo_slope")
    variant = asset.resolve_variant("h1.0.msh")
    mesh = asset.build_mesh(variant, elem_type="P1")
    return {
        "asset_count": len(assets),
        "has_2d_homo_slope": "2d_homo_slope" in assets,
        "probe_asset": asset.asset_id,
        "probe_variant": variant.name,
        "probe_nodes": int(mesh.coord.shape[1]),
        "probe_elements": int(mesh.elem.shape[1]),
        "probe_free_unknowns": int(mesh.q_mask.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-mpi-size", type=int, default=1, help="Fail unless MPI size is at least this value.")
    parser.add_argument("--require-hypre", action="store_true", help="Fail if the tiny HYPRE PETSc probe fails.")
    parser.add_argument("--require-mumps", action="store_true", help="Fail if the tiny MUMPS PETSc probe fails.")
    args = parser.parse_args()

    imports = {
        "numpy": _module_version("numpy"),
        "scipy": _module_version("scipy"),
        "h5py": _module_version("h5py"),
        "meshio": _module_version("meshio"),
        "mpi4py": _module_version("mpi4py"),
        "petsc4py": _module_version("petsc4py"),
    }

    from mpi4py import MPI
    from petsc4py import PETSc

    comm = PETSc.COMM_WORLD
    rank = int(comm.getRank())
    size = int(comm.getSize())
    mpi_comm = comm.tompi4py()

    petsc = {
        "version": ".".join(str(part) for part in PETSc.Sys.getVersion()),
        "scalar_type": str(PETSc.ScalarType),
        "real_type": str(PETSc.RealType),
    }
    package_probes = {
        "jacobi": _tiny_petsc_solve(PETSc, comm, pc_type="jacobi"),
        "hypre": _tiny_petsc_solve(PETSc, comm, pc_type="hypre"),
        "mumps": _tiny_petsc_solve(PETSc, comm, pc_type="lu", factor_solver_type="mumps"),
    }
    asset = _asset_probe()

    local = {
        "rank": rank,
        "size": size,
        "processor": MPI.Get_processor_name(),
        "package_probes": package_probes,
    }
    ranks = mpi_comm.gather(local, root=0)

    failures: list[str] = []
    if size < int(args.min_mpi_size):
        failures.append(f"MPI size {size} is smaller than required {args.min_mpi_size}")
    if args.require_hypre and not bool(package_probes["hypre"]["ok"]):
        failures.append("HYPRE probe failed")
    if args.require_mumps and not bool(package_probes["mumps"]["ok"]):
        failures.append("MUMPS probe failed")

    failed_flags = mpi_comm.gather(failures, root=0)
    if rank == 0:
        all_failures = [item for group in failed_flags for item in group]
        print(
            json.dumps(
                {
                    "status": "ok" if not all_failures else "failed",
                    "imports": imports,
                    "mpi": {"size": size},
                    "petsc": petsc,
                    "asset_probe": asset,
                    "ranks": ranks,
                    "failures": all_failures,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
