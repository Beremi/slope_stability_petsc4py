#!/usr/bin/env python3
"""Run a 3D asset-backed seepage case and save artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from petsc4py import PETSc

from slope_stability.linear.pmg import build_3d_same_mesh_scalar_pmg_hierarchy, validate_pmg_fine_level_alignment
from slope_stability.linear.solver import SolverFactory
from slope_stability.core.elements import validate_supported_elem_type
from slope_stability.fem.quadrature import quadrature_volume_3d
from slope_stability.mesh import reorder_mesh_nodes
from slope_stability.problem_asset_runtime import (
    build_mesh_for_resolved_asset,
    build_seepage_boundary_for_resolved_asset,
    load_seepage_problem_spec,
    resolve_problem_asset,
)
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


def _load_reordered_mesh(
    resolved_asset,
    *,
    elem_type: str,
    node_ordering: str,
    partition_count: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mesh = build_mesh_for_resolved_asset(resolved_asset, elem_type=elem_type)
    reordered = reorder_mesh_nodes(
        mesh.coord,
        mesh.elem,
        mesh.surf,
        mesh.q_mask,
        strategy=node_ordering,
        n_parts=partition_count if str(node_ordering).lower() == "block_metis" else None,
    )
    return (
        np.asarray(reordered.coord, dtype=np.float64),
        np.asarray(reordered.elem, dtype=np.int64),
        np.asarray(reordered.surf, dtype=np.int64),
        np.asarray(mesh.material_id, dtype=np.int64),
        np.asarray(mesh.boundary_labels, dtype=np.int64),
    )


def _first_linear_history_payload(init_linear: dict[str, object]) -> dict[str, object]:
    solve_info = dict(init_linear.get("solve_info", {}) or {})
    reported = [float(v) for v in solve_info.get("reported_residual_history", []) or []]
    true = [float(v) for v in solve_info.get("true_residual_history", []) or []]
    limit = min(11, max(len(reported), len(true)))
    reported0 = reported[0] if reported else None
    true0 = true[0] if true else None
    rows = []
    for idx in range(limit):
        reported_val = reported[idx] if idx < len(reported) else None
        true_val = true[idx] if idx < len(true) else None
        rows.append(
            {
                "history_index": idx,
                "reported_residual": reported_val,
                "reported_ratio_to_initial": None
                if reported_val is None or reported0 in {None, 0.0}
                else float(reported_val / reported0),
                "true_residual": true_val,
                "true_ratio_to_initial": None if true_val is None or true0 in {None, 0.0} else float(true_val / true0),
            }
        )
    return {
        "iterations": int(init_linear.get("iterations", 0) or 0),
        "solve_time_s": float(init_linear.get("solve_time", 0.0) or 0.0),
        "preconditioner_time_s": float(init_linear.get("preconditioner_time", 0.0) or 0.0),
        "orthogonalization_time_s": float(init_linear.get("orthogonalization_time", 0.0) or 0.0),
        "reported_residual_history": reported,
        "true_residual_history": true,
        "rows": rows,
    }


def _write_first_linear_artifacts(data_dir: Path, payload: dict[str, object]) -> None:
    json_path = data_dir / "first_linear_system.json"
    csv_path = data_dir / "first_linear_system.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "history_index",
                "reported_residual",
                "reported_ratio_to_initial",
                "true_residual",
                "true_ratio_to_initial",
            ],
        )
        writer.writeheader()
        writer.writerows(payload.get("rows", []))


def run_capture(
    *,
    out_dir: Path,
    asset_name: str,
    mesh_variant: str | None = None,
    profile: str | None = None,
    elem_type: str = "P2",
    node_ordering: str = "block_metis",
    partition_count_override: int | None = None,
    solver_type: str = "PETSC_MATLAB_DFGMRES_HYPRE",
    pc_backend: str = "hypre",
    linear_tolerance: float = 1.0e-10,
    linear_max_iter: int = 500,
    petsc_opt: list[str] | None = None,
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
    if elem_type not in {"P2", "P4"}:
        raise NotImplementedError(f"3D seepage currently supports P2 and P4 meshes; requested {elem_type!r}.")

    resolved_asset = resolve_problem_asset(asset_name=str(asset_name), mesh_variant=mesh_variant, profile=profile)
    seepage_spec = load_seepage_problem_spec(resolved_asset)
    if resolved_asset.mesh_path is None:
        raise ValueError(f"Asset {resolved_asset.asset_name!r} variant {resolved_asset.variant_name!r} has no mesh file.")
    mesh_path = resolved_asset.mesh_path
    profile = resolved_asset.resolved_variant.profile

    partition_count = (
        int(partition_count_override)
        if partition_count_override is not None
        else (int(size) if str(node_ordering).lower() == "block_metis" else None)
    )
    coord, elem, surf, material_identifier, triangle_labels = _load_reordered_mesh(
        resolved_asset,
        elem_type=elem_type,
        node_ordering=node_ordering,
        partition_count=partition_count,
    )

    grho = float(seepage_spec.seepage.water_unit_weight)
    conductivity_values = np.asarray(seepage_spec.conductivity, dtype=np.float64).ravel()
    required_conductivity_count = int(material_identifier.max()) + 1 if material_identifier.size else 1
    if conductivity_values.size == 1 and required_conductivity_count > 1:
        conductivity_values = np.repeat(conductivity_values, required_conductivity_count)
    if conductivity_values.size < required_conductivity_count:
        raise ValueError(
            f"Conductivity vector has {conductivity_values.size} entries, "
            f"but seepage material ids require {required_conductivity_count}."
        )
    n_q = int(quadrature_volume_3d(elem_type)[0].shape[1])
    conduct0 = heter_conduct(material_identifier, n_q, conductivity_values)
    q_w, pw_d = build_seepage_boundary_for_resolved_asset(resolved_asset, coord, surf, triangle_labels, grho=grho)

    pc_backend_norm = str(pc_backend).strip().lower()
    preconditioner_options = {
        "threads": 16,
        "print_level": 0,
        "use_as_preconditioner": True,
        "pc_hypre_boomeramg_coarsen_type": "HMIS",
        "pc_hypre_boomeramg_interp_type": "ext+i",
        "null_space": np.empty((0, 0), dtype=np.float64),
    }
    if petsc_opt:
        for entry in petsc_opt:
            text = str(entry)
            if "=" in text:
                key, value = text.split("=", 1)
            else:
                key, value = text, "1"
            preconditioner_options[key.strip()] = value.strip()
    if pc_backend_norm in {"pmg", "pmg_shell"}:
        def _q_mask_builder(level_coord, level_surf, level_triangle_labels):
            if level_triangle_labels is None:
                raise ValueError("PMG seepage hierarchy requires triangle labels for boundary detection.")
            level_q_w, _ = build_seepage_boundary_for_resolved_asset(
                resolved_asset,
                level_coord,
                level_surf,
                level_triangle_labels,
                grho=grho,
            )
            return np.asarray(level_q_w, dtype=bool).reshape(1, -1)

        pmg_hierarchy = build_3d_same_mesh_scalar_pmg_hierarchy(
            resolved_asset,
            fine_elem_type=elem_type,
            node_ordering=node_ordering,
            reorder_parts=partition_count,
            q_mask_builder=_q_mask_builder,
            comm=PETSc.COMM_SELF,
        )
        validate_pmg_fine_level_alignment(
            pmg_hierarchy,
            coord=coord,
            elem=elem,
            surf=surf,
            q_mask=np.asarray(q_w, dtype=bool).reshape(1, -1),
            comm=PETSc.COMM_SELF,
        )
        preconditioner_options.update(
            {
                "pc_backend": pc_backend_norm,
                "pmg_hierarchy": pmg_hierarchy,
            }
        )

    solver = SolverFactory.create(
        solver_type,
        tolerance=linear_tolerance,
        max_iterations=linear_max_iter,
        deflation_basis_tolerance=1.0e-3,
        verbose=False,
        q_mask=np.asarray(q_w, dtype=bool).reshape(1, -1),
        coord=coord,
        preconditioner_options=preconditioner_options,
    )
    if hasattr(solver, "enable_diagnostics"):
        solver.enable_diagnostics(True)

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
    first_linear_payload = _first_linear_history_payload(dict(history["init_linear"]))
    _write_first_linear_artifacts(data_dir, first_linear_payload)

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
            "pc_backend": pc_backend_norm,
        },
        "params": {
            "elem_type": elem_type,
            "node_ordering": str(node_ordering),
            "grho": grho,
            "k": conductivity_values.tolist(),
            "linear_tolerance": linear_tolerance,
            "linear_max_iter": linear_max_iter,
            "partition_count": partition_count,
            "pc_backend": pc_backend_norm,
            "petsc_opt": list(petsc_opt or []),
        },
        "timings": {
            "linear": {
                "init_linear": history["init_linear"],
                "newton_linear_iterations": history["linear_iterations"],
                "newton_linear_solve_time": history["linear_solve_time"],
                "newton_linear_preconditioner_time": history["linear_preconditioner_time"],
                "newton_linear_orthogonalization_time": history["linear_orthogonalization_time"],
                "newton_linear_solve_info": history["linear_solve_info"],
            }
        },
        "history": {
            "criterion": history["criterion"],
            "iterations": history["iterations"],
            "converged": bool(history["converged"]),
            "K_D_nnz": int(history["K_D_nnz"]),
        },
        "first_linear_system": first_linear_payload,
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
    parser = argparse.ArgumentParser(description="Run a 3D asset-backed seepage case.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--asset", type=str, required=True)
    parser.add_argument("--mesh_variant", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--elem_type", type=str, default="P2", choices=["P1", "P2", "P4"])
    parser.add_argument("--node_ordering", type=str, default="block_metis")
    parser.add_argument("--partition_count_override", type=int, default=None)
    parser.add_argument("--solver_type", type=str, default="PETSC_MATLAB_DFGMRES_HYPRE")
    parser.add_argument("--pc_backend", type=str, default="hypre")
    parser.add_argument("--linear_tolerance", type=float, default=1.0e-10)
    parser.add_argument("--linear_max_iter", type=int, default=500)
    parser.add_argument("--petsc_opt", action="append", default=None)
    args = parser.parse_args()
    run_capture(
        out_dir=args.out_dir,
        asset_name=args.asset,
        mesh_variant=args.mesh_variant,
        profile=args.profile,
        elem_type=args.elem_type,
        node_ordering=args.node_ordering,
        partition_count_override=args.partition_count_override,
        solver_type=args.solver_type,
        pc_backend=args.pc_backend,
        linear_tolerance=args.linear_tolerance,
        linear_max_iter=args.linear_max_iter,
        petsc_opt=args.petsc_opt,
    )


if __name__ == "__main__":
    main()
