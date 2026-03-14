#!/usr/bin/env python3
"""Run the 3D heterogeneous seepage benchmark and save MATLAB-comparison artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from petsc4py import PETSc

ROOT = Path(__file__).resolve().parents[3]

from slope_stability.linear.solver import SolverFactory
from slope_stability.core.elements import validate_supported_elem_type
from slope_stability.mesh import load_mesh_gmsh_waterlevels, seepage_boundary_3d_hetero
from slope_stability.seepage import heter_conduct, seepage_problem_3d


def _plot_pore_pressure_surface(coord: np.ndarray, surf: np.ndarray, pw: np.ndarray, out_path: Path) -> None:
    tri = np.asarray(surf[:3, :].T, dtype=np.int64)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    coll = ax.plot_trisurf(
        coord[0, :],
        coord[2, :],
        coord[1, :],
        triangles=tri,
        cmap="viridis",
        linewidth=0.05,
        antialiased=True,
        shade=False,
        array=np.asarray(pw, dtype=np.float64),
    )
    fig.colorbar(coll, ax=ax, shrink=0.75, label="pore pressure [kPa]")
    ax.set_title("PETSc pore pressure 3D")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_saturation_centroids(coord: np.ndarray, elem: np.ndarray, mater_sat: np.ndarray, out_path: Path) -> None:
    centers = np.mean(coord[:, elem[:4, :]], axis=1)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        centers[0, :],
        centers[2, :],
        centers[1, :],
        c=np.asarray(mater_sat, dtype=np.float64),
        s=4.0,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    fig.colorbar(sc, ax=ax, shrink=0.75, label="saturated")
    ax.set_title("PETSc saturation 3D")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_capture(
    *,
    out_dir: Path,
    mesh_path: Path,
    elem_type: str = "P2",
    solver_type: str = "PETSC_MATLAB_DFGMRES_HYPRE",
    linear_tolerance: float = 1.0e-10,
    linear_max_iter: int = 500,
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

    elem_type = validate_supported_elem_type(3, elem_type)
    if elem_type != "P2":
        raise NotImplementedError(
            f"3D heterogeneous seepage currently uses the Gmsh waterlevels P2 mesh family; requested {elem_type!r}."
        )

    mesh = load_mesh_gmsh_waterlevels(mesh_path)
    coord, elem, surf = mesh.coord, mesh.elem, mesh.surf
    material_identifier, triangle_labels = mesh.material, mesh.triangle_labels

    grho = 9.81
    k = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    conduct0 = heter_conduct(material_identifier, 11, k)
    q_w, pw_d = seepage_boundary_3d_hetero(coord, surf, triangle_labels, grho)

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
    pw, grad_p, mater_sat, history, assembly = seepage_problem_3d(
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

    _plot_pore_pressure_surface(coord, surf, pw, plots_dir / "petsc_pore_pressure_3D.png")
    _plot_saturation_centroids(coord, elem, mater_sat, plots_dir / "petsc_saturation_3D.png")

    np.savez(
        data_dir / "petsc_run.npz",
        coord=coord,
        elem=elem,
        surf=surf,
        material_identifier=material_identifier,
        triangle_labels=triangle_labels,
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
            "mesh_file": str(mesh_path),
            "mesh_nodes": int(coord.shape[1]),
            "mesh_elements": int(elem.shape[1]),
            "n_int": int(assembly.n_int),
            "solver_type": solver_type,
        },
        "params": {
            "elem_type": elem_type,
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
    parser = argparse.ArgumentParser(description="Run 3D hetero seepage capture.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--mesh_path",
        type=Path,
        default=ROOT / "meshes" / "3d_hetero_seepage" / "slope_with_waterlevels_concave_L2.h5",
    )
    parser.add_argument("--elem_type", type=str, default="P2", choices=["P1", "P2", "P4"])
    parser.add_argument("--solver_type", type=str, default="PETSC_MATLAB_DFGMRES_HYPRE")
    parser.add_argument("--linear_tolerance", type=float, default=1.0e-10)
    parser.add_argument("--linear_max_iter", type=int, default=500)
    args = parser.parse_args()
    run_capture(
        out_dir=args.out_dir,
        mesh_path=args.mesh_path,
        elem_type=args.elem_type,
        solver_type=args.solver_type,
        linear_tolerance=args.linear_tolerance,
        linear_max_iter=args.linear_max_iter,
    )


if __name__ == "__main__":
    main()
