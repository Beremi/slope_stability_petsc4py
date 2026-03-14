#!/usr/bin/env python3
"""Run the 2D Sloan2013 seepage benchmark and save MATLAB-comparison artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from petsc4py import PETSc

ROOT = Path(__file__).resolve().parents[3]

from slope_stability.linear.solver import SolverFactory
from slope_stability.mesh import generate_sloan2013_mesh_2d
from slope_stability.fem import quadrature_volume_2d
from slope_stability.seepage import heter_conduct, seepage_problem_2d


def _plot_pore_pressure(coord: np.ndarray, elem: np.ndarray, pw: np.ndarray, out_path: Path) -> None:
    tri = np.asarray(elem[:3, :].T, dtype=np.int64)
    triangulation = mtri.Triangulation(coord[0, :], coord[1, :], triangles=tri)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tcf = ax.tricontourf(triangulation, pw, levels=30, cmap="viridis")
    fig.colorbar(tcf, ax=ax, label="pore pressure [kPa]")
    ax.set_aspect("equal")
    ax.set_title("PETSc pore pressure")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_saturation(coord: np.ndarray, elem: np.ndarray, mater_sat: np.ndarray, out_path: Path) -> None:
    tri = np.asarray(elem[:3, :].T, dtype=np.int64)
    facecolors = np.asarray(mater_sat, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    coll = ax.tripcolor(coord[0, :], coord[1, :], tri, facecolors=facecolors, edgecolors="none", cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(coll, ax=ax, label="saturated")
    ax.set_aspect("equal")
    ax.set_title("PETSc saturation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_capture(
    *,
    out_dir: Path,
    elem_type: str = "P1",
    solver_type: str = "PETSC_MATLAB_DFGMRES_HYPRE",
    linear_tolerance: float = 1.0e-10,
    linear_max_iter: int = 300,
) -> dict[str, object]:
    comm = PETSc.COMM_WORLD
    rank = int(comm.getRank())
    size = int(comm.getSize())
    if size > 1 and rank != 0:
        return {
            "output": str(out_dir),
            "runtime": 0.0,
            "mpi_mode": "root_only",
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    plots_dir = out_dir / "plots"
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    elem_type = str(elem_type).upper()
    mesh = generate_sloan2013_mesh_2d(elem_type=elem_type)
    coord, elem, material_identifier = mesh.coord, mesh.elem, mesh.material

    x1 = 15.0
    x3 = 20.0
    y11 = 6.75
    y12 = 0.5
    y13 = 0.75
    y21 = 1.0
    y22 = 9.25
    y23 = 2.0
    y1 = y11 + y12 + y13
    y2 = y21 + y22 + y23
    beta_deg = 26.6
    beta = np.deg2rad(beta_deg)
    x2 = y2 / np.tan(beta)
    grho = 9.81
    k = np.array([1.0, 1.0], dtype=np.float64)
    n_q = int(quadrature_volume_2d(elem_type)[0].shape[1])
    conduct0 = heter_conduct(material_identifier, n_q, k)

    q_w = np.ones(coord.shape[1], dtype=bool)
    q_w[coord[0, :] <= 0.001] = False
    q_w[coord[0, :] >= x1 + x2 + x3 - 0.001] = False
    q_w[coord[1, :] >= y1 + y2 - 0.001] = False
    q_w[(coord[1, :] >= y1 - 0.001) & (coord[0, :] >= x1 + x2 - 0.001)] = False
    q_w[(coord[1, :] >= y1 - 0.001) & (coord[1, :] >= -(y2 / x2) * coord[0, :] + y1 + y2 * (1.0 + x1 / x2) - 0.001)] = False

    pw_d = np.zeros(coord.shape[1], dtype=np.float64)
    x_bar = x1 + (1.0 - y21 / y2) * x2
    part1 = (coord[0, :] < x_bar) & (coord[1, :] <= -(y22 / x_bar) * coord[0, :] + y1 + y21 + y22)
    part2 = coord[0, :] >= x_bar
    pw_d[part1] = grho * ((y22 / x_bar) * (x_bar - coord[0, part1]) + y1 + y21 - coord[1, part1])
    pw_d[part2] = grho * (y1 + y21 - coord[1, part2])

    solver = SolverFactory.create(
        solver_type,
        tolerance=linear_tolerance,
        max_iterations=linear_max_iter,
        deflation_basis_tolerance=1.0e-3,
        verbose=False,
        q_mask=None,
        coord=None,
        preconditioner_options={
            "threads": 16,
            "print_level": 0,
            "use_as_preconditioner": True,
            "pc_hypre_boomeramg_coarsen_type": "HMIS",
            "pc_hypre_boomeramg_interp_type": "ext+i",
        },
    )

    t0 = perf_counter()
    pw, grad_p, mater_sat, history, assembly = seepage_problem_2d(
        coord,
        elem,
        q_w,
        pw_d,
        grho,
        conduct0,
        elem_type=elem_type,
        linear_system_solver=solver,
        it_max=50,
        tol=1.0e-10,
    )
    runtime = perf_counter() - t0

    _plot_pore_pressure(coord, elem, pw, plots_dir / "petsc_pore_pressure_2D.png")
    _plot_saturation(coord, elem, mater_sat, plots_dir / "petsc_saturation_2D.png")

    np.savez(
        data_dir / "petsc_run.npz",
        coord=coord,
        elem=elem,
        material_identifier=material_identifier,
        q_w=q_w,
        pw_d=pw_d,
        conduct0=conduct0,
        pw=pw,
        grad_p=grad_p,
        mater_sat=mater_sat,
        criterion=np.asarray(history["criterion"], dtype=np.float64),
        linear_iterations=np.asarray(history["linear_iterations"], dtype=np.int64),
    )

    result = {
        "run_info": {
            "runtime_seconds": runtime,
            "mpi_size": size,
            "mpi_mode": "root_only" if size > 1 else "serial",
            "mesh_nodes": int(coord.shape[1]),
            "mesh_elements": int(elem.shape[1]),
            "n_int": int(assembly.n_int),
            "solver_type": solver_type,
        },
        "params": {
            "elem_type": elem_type,
            "h": 0.5,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "y11": y11,
            "y12": y12,
            "y13": y13,
            "y21": y21,
            "y22": y22,
            "y23": y23,
            "y1": y1,
            "y2": y2,
            "beta_deg": beta_deg,
            "grho": grho,
            "k": k.tolist(),
            "linear_tolerance": linear_tolerance,
            "linear_max_iter": linear_max_iter,
        },
        "timings": {
            "linear": {
                "init_linear": history["init_linear"],
                "newton_linear_iterations": history["linear_iterations"],
                "newton_linear_solve_time": history["linear_solve_time"],
                "newton_linear_preconditioner_time": history["linear_preconditioner_time"],
                "newton_linear_orthogonalization_time": history["linear_orthogonalization_time"],
            }
        },
        "history": {
            "criterion": history["criterion"],
            "iterations": history["iterations"],
            "converged": bool(history["converged"]),
            "K_D_nnz": int(history["K_D_nnz"]),
        },
    }
    with open(data_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return {
        **result,
        "output": str(out_dir),
        "npz": str(data_dir / "petsc_run.npz"),
        "json": str(data_dir / "run_info.json"),
        "runtime": float(runtime),
        "mpi_mode": "root_only" if size > 1 else "serial",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2D Sloan2013 seepage capture.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--elem_type", type=str, default="P1", choices=["P1", "P2", "P4"])
    parser.add_argument("--solver_type", type=str, default="PETSC_MATLAB_DFGMRES_HYPRE")
    parser.add_argument("--linear_tolerance", type=float, default=1.0e-10)
    parser.add_argument("--linear_max_iter", type=int, default=300)
    args = parser.parse_args()
    run_capture(
        out_dir=args.out_dir,
        elem_type=args.elem_type,
        solver_type=args.solver_type,
        linear_tolerance=args.linear_tolerance,
        linear_max_iter=args.linear_max_iter,
    )


if __name__ == "__main__":
    main()
