#!/usr/bin/env python3
"""Replay exported SSR linear systems with the original petsc4py PMG solver.

This is intentionally a small local investigation driver.  It loads the
exported free matrix/RHS/deflation basis and runs the Python MATLAB-style
DFGMRES implementation on exactly those frozen systems.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from slope_stability.linear.pmg import build_3d_pmg_hierarchy
from slope_stability.linear.solver import PetscMatlabExactDFGMRESSolver
from slope_stability.problem_asset_runtime import (
    load_mechanical_problem_spec,
    resolve_problem_asset,
)
from slope_stability.utils import petsc_vec_to_global_array


def _load_vec(path: Path, comm) -> PETSc.Vec:
    viewer = PETSc.Viewer().createBinary(str(path), mode="r", comm=comm)
    vec = PETSc.Vec().load(viewer)
    viewer.destroy()
    return vec


def _load_mat(path: Path, comm, *, global_size: int | None = None, ownership_range=None) -> PETSc.Mat:
    viewer = PETSc.Viewer().createBinary(str(path), mode="r", comm=comm)
    if global_size is None or ownership_range is None:
        mat = PETSc.Mat().load(viewer)
    else:
        lo, hi = (int(ownership_range[0]), int(ownership_range[1]))
        n_local = int(hi - lo)
        mat = PETSc.Mat().create(comm=comm)
        mat.setSizes(((n_local, int(global_size)), (n_local, int(global_size))))
        mat.setType(PETSc.Mat.Type.AIJ)
        mat.load(viewer)
    viewer.destroy()
    return mat


def _global_array(vec: PETSc.Vec) -> np.ndarray:
    return np.asarray(petsc_vec_to_global_array(vec), dtype=np.float64)


def _load_basis(sample_dir: Path, basis_cols: int, comm) -> np.ndarray:
    cols: list[np.ndarray] = []
    for idx in range(int(basis_cols)):
        vec = _load_vec(sample_dir / f"basis_{idx:04d}.vec", comm)
        try:
            cols.append(_global_array(vec))
        finally:
            vec.destroy()
    if not cols:
        return np.empty((0, 0), dtype=np.float64)
    return np.column_stack(cols)


def _make_hierarchy(comm):
    resolved = resolve_problem_asset(
        asset_name="3d_hetero_slope",
        mesh_variant="adaptive_family_a_l1.msh",
    )
    spec = load_mechanical_problem_spec(resolved)
    return build_3d_pmg_hierarchy(
        resolved,
        node_ordering="block_metis",
        reorder_parts=int(comm.getSize()),
        material_rows=spec.material_rows,
        comm=comm,
    )


def _make_solver(*, hierarchy, rtol: float, max_it: int, q_mask, coord) -> PetscMatlabExactDFGMRESSolver:
    opts = {
        "pc_backend": "pmg_shell",
        "pmg_hierarchy": hierarchy,
        "mpi_distribute_by_nodes": True,
        "full_system_preconditioner": False,
        "compiled_outer": False,
        "recycle_preconditioner": True,
        "distributed_deflation_basis_local": True,
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_ksp_max_it": 3,
        "mg_levels_pc_type": "jacobi",
        "mg_coarse_ksp_type": "preonly",
        "mg_coarse_pc_type": "hypre",
        "mg_coarse_pc_hypre_type": "boomeramg",
        "mg_coarse_max_it": 1,
        "mg_coarse_rtol": 0.0,
        "pc_hypre_boomeramg_max_iter": 4,
        "pc_hypre_boomeramg_tol": 0.0,
        "pc_hypre_boomeramg_coarsen_type": "HMIS",
        "pc_hypre_boomeramg_interp_type": "ext+i",
        "pc_hypre_boomeramg_P_max": 4,
        "pc_hypre_boomeramg_strong_threshold": 0.5,
        "pc_hypre_boomeramg_grid_sweeps_all": 1,
        "pc_hypre_boomeramg_cycle_type": "V",
        "pc_hypre_boomeramg_agg_nl": 0,
    }
    solver = PetscMatlabExactDFGMRESSolver(
        "HYPRE",
        tolerance=float(rtol),
        max_iterations=int(max_it),
        tolerance_deflation_basis=1.0e-3,
        verbose=False,
        q_mask=q_mask,
        coord=coord,
        preconditioner_options=opts,
    )
    solver._diagnostics_enabled = True
    return solver


def _sample_key(path: Path) -> str:
    return path.name


def _run_one_solve(
    *,
    solver: PetscMatlabExactDFGMRESSolver,
    A: PETSc.Mat,
    rhs_path: Path,
    solve_label: str,
    rank: int,
) -> dict[str, object]:
    rhs_vec = _load_vec(rhs_path, A.getComm())
    try:
        rhs_global = _global_array(rhs_vec)
    finally:
        rhs_vec.destroy()
    t0 = time.perf_counter()
    solver._linear_replay_debug_label = solve_label
    solver.solve(A, rhs_global)
    elapsed = time.perf_counter() - t0
    info = dict(getattr(solver, "_last_solve_info", {}) or {})
    hist = list(info.get("reported_residual_history", []) or [])
    if rank == 0:
        print(
            "PETSC4PY_REPLAY_RESULT "
            f"solve={solve_label} iterations={int(info.get('iterations', -1))} "
            f"initial_rel={(hist[0] if hist else float('nan')):.16e} "
            f"final_rel={float(info.get('reported_residual_final', float('nan'))):.16e} "
            f"time={elapsed:.6f}",
            flush=True,
        )
    return {
        "solve": solve_label,
        "iterations": int(info.get("iterations", -1)),
        "initial_rel": float(hist[0]) if hist else float("nan"),
        "final_rel": float(info.get("reported_residual_final", float("nan"))),
        "time_s": float(elapsed),
        "history": [float(v) for v in hist],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-root", default="/tmp/ssr_hard_step_petsc4py")
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--max-it", type=int, default=600)
    args = parser.parse_args()

    comm = PETSc.COMM_WORLD
    rank = int(comm.getRank())
    size = int(comm.getSize())
    mpi_comm = comm.tompi4py()

    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    mpi_comm.Barrier()

    if rank == 0:
        print(f"PETSC4PY_REPLAY_START ranks={size} rtol={args.rtol} max_it={args.max_it}", flush=True)

    hierarchy = _make_hierarchy(comm)
    q_mask = hierarchy.fine_level.q_mask.astype(bool, copy=False)
    coord = hierarchy.fine_level.coord.astype(np.float64, copy=False)

    summary_rows: list[dict[str, object]] = []
    history_payload: dict[str, object] = {}

    for sample_name in args.samples:
        sample_dir = Path(args.sample_root) / sample_name
        meta = json.loads((sample_dir / "meta.json").read_text())
        if rank == 0:
            print(
                "PETSC4PY_REPLAY_SAMPLE "
                f"sample={sample_name} omega={float(meta.get('omega', 0.0)):.9e} "
                f"lambda={float(meta.get('lambda', 0.0)):.9e} "
                f"newton={int(meta.get('newton_iteration', -1))} "
                f"basis={int(meta.get('basis_cols', 0))}",
                flush=True,
            )
        fine_owned = tuple(int(v) for v in hierarchy.fine_level.owned_free_range)
        A = _load_mat(
            sample_dir / "A_free.mat",
            comm,
            global_size=int(meta.get("global_size", hierarchy.fine_level.free_size)),
            ownership_range=fine_owned,
        )
        basis = _load_basis(sample_dir, int(meta.get("basis_cols", 0)), comm)
        solver = _make_solver(
            hierarchy=hierarchy,
            rtol=args.rtol,
            max_it=args.max_it,
            q_mask=q_mask,
            coord=coord,
        )
        solver.deflation_basis = basis
        solver.setup_preconditioner(A)
        solver.A_orthogonalize(A)
        if rank == 0:
            ortho = getattr(solver, "_last_orthogonalization_info", {}) or {}
            print(
                "PETSC4PY_REPLAY_BASIS "
                f"sample={sample_name} exported={int(meta.get('basis_cols', 0))} "
                f"after={int(ortho.get('basis_cols_after', -1))} "
                f"passes={int(ortho.get('basis_reorth_passes', -1))}",
                flush=True,
            )

        for solve_label, rhs_name in (("dW", "rhs_w.vec"), ("dV", "rhs_v.vec")):
            result = _run_one_solve(
                solver=solver,
                A=A,
                rhs_path=sample_dir / rhs_name,
                solve_label=solve_label,
                rank=rank,
            )
            expected = meta.get("expected", {}) or {}
            row = {
                "sample": sample_name,
                "omega": float(meta.get("omega", 0.0)),
                "lambda": float(meta.get("lambda", 0.0)),
                "newton_it": int(meta.get("newton_iteration", -1)),
                "basis": int(meta.get("basis_cols", 0)),
                "solve": solve_label,
                "profile": "petsc4py_original",
                "petsc4py_1e1_its": int(expected.get(f"{solve_label}_iterations", -1)),
                "petsc4py_1e6_its": int(result["iterations"]),
                "initial_rel": float(result["initial_rel"]),
                "final_rel": float(result["final_rel"]),
                "time_s": float(result["time_s"]),
            }
            summary_rows.append(row)
            history_payload[f"{sample_name}:{solve_label}"] = {
                **row,
                "history": result["history"],
            }

        solver.release_iteration_resources()
        A.destroy()
        del solver
        mpi_comm.Barrier()

    if rank == 0:
        csv_path = out_dir / "petsc4py_replay_summary.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        json_path = out_dir / "petsc4py_replay_histories.json"
        json_path.write_text(json.dumps(history_payload, indent=2))
        print(f"PETSC4PY_REPLAY_DONE csv={csv_path} histories={json_path}", flush=True)

    PETSc.garbage_cleanup(comm=comm)
    mpi_comm.Barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
