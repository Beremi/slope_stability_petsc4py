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
from slope_stability.mesh import reorder_mesh_nodes
from slope_stability.problem_asset_runtime import (
    build_mesh_for_resolved_asset,
    build_seepage_boundary_for_resolved_asset,
    load_seepage_problem_spec,
    resolve_problem_asset,
)
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
    asset_name: str | None = "2d_sloan2013",
    mesh_variant: str | None = None,
    profile: str | None = None,
    elem_type: str = "P1",
    node_ordering: str = "block_metis",
    solver_type: str = "PETSC_MATLAB_DFGMRES_HYPRE",
    linear_tolerance: float = 1.0e-10,
    linear_max_iter: int = 300,
    nonlinear_max_iter: int = 50,
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
    resolved_asset = resolve_problem_asset(asset_name=str(asset_name or "2d_sloan2013"), mesh_variant=mesh_variant, profile=profile)
    built_mesh = build_mesh_for_resolved_asset(resolved_asset, elem_type=elem_type)
    partition_count = int(size) if str(node_ordering).lower() == "block_metis" else None
    reordered = reorder_mesh_nodes(
        built_mesh.coord,
        built_mesh.elem,
        built_mesh.surf,
        built_mesh.q_mask,
        strategy=node_ordering,
        n_parts=partition_count,
    )
    seepage_spec = load_seepage_problem_spec(resolved_asset)
    coord = np.asarray(reordered.coord, dtype=np.float64)
    elem = np.asarray(reordered.elem, dtype=np.int64)
    surf = np.asarray(reordered.surf, dtype=np.int64)
    material_identifier = np.asarray(built_mesh.material_id, dtype=np.int64)

    params = dict(resolved_asset.variant.get("source", {}).get("parameters", resolved_asset.variant.get("parameters", {})))
    x1 = float(params.get("x1", 15.0))
    x3 = float(params.get("x3", 20.0))
    y11 = float(params.get("y11", 6.75))
    y12 = float(params.get("y12", 0.5))
    y13 = float(params.get("y13", 0.75))
    y21 = float(params.get("y21", 1.0))
    y22 = float(params.get("y22", 9.25))
    y23 = float(params.get("y23", 2.0))
    y1 = y11 + y12 + y13
    y2 = y21 + y22 + y23
    beta_deg = float(params.get("beta_deg", 26.6))
    beta = np.deg2rad(beta_deg)
    x2 = float(y2 / np.tan(beta))
    grho = float(seepage_spec.seepage.water_unit_weight)
    k = np.asarray(seepage_spec.conductivity, dtype=np.float64)
    n_q = int(quadrature_volume_2d(elem_type)[0].shape[1])
    conduct0 = heter_conduct(material_identifier, n_q, k)
    q_w, pw_d = build_seepage_boundary_for_resolved_asset(
        resolved_asset,
        coord,
        surf,
        built_mesh.boundary_labels,
        grho=grho,
    )

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
        it_max=nonlinear_max_iter,
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
            "node_ordering": str(node_ordering),
        },
        "params": {
            "elem_type": elem_type,
            "asset_name": resolved_asset.asset_name,
            "mesh_variant": resolved_asset.variant_name,
            "node_ordering": str(node_ordering),
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
            "nonlinear_max_iter": nonlinear_max_iter,
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
    parser.add_argument("--node_ordering", type=str, default="block_metis")
    parser.add_argument("--solver_type", type=str, default="PETSC_MATLAB_DFGMRES_HYPRE")
    parser.add_argument("--linear_tolerance", type=float, default=1.0e-10)
    parser.add_argument("--linear_max_iter", type=int, default=300)
    parser.add_argument("--nonlinear_max_iter", type=int, default=50)
    args = parser.parse_args()
    run_capture(
        out_dir=args.out_dir,
        elem_type=args.elem_type,
        node_ordering=args.node_ordering,
        solver_type=args.solver_type,
        linear_tolerance=args.linear_tolerance,
        linear_max_iter=args.linear_max_iter,
        nonlinear_max_iter=args.nonlinear_max_iter,
    )


if __name__ == "__main__":
    main()
