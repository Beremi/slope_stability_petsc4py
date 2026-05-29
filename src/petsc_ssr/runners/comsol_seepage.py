from __future__ import annotations

import argparse
import json
from pathlib import Path

from petsc_ssr.hydro import (
    DEFAULT_COMSOL_SEEPAGE_MESH,
    HydroMesh,
    HydroResult,
    print_hydro_result,
    solve_comsol_seepage,
)
from petsc_ssr.runtime.options import quote_option_tokens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone COMSOL-transition seepage port.")
    parser.add_argument("--backend", choices=["petsc", "scipy"], default="petsc")
    parser.add_argument("--mesh", type=Path, default=DEFAULT_COMSOL_SEEPAGE_MESH)
    parser.add_argument("--elem-type", choices=["P1", "P2"], default="P2")
    parser.add_argument("--linear-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--linear-max-it", type=int, default=500)
    parser.add_argument("--newton-max-it", type=int, default=50)
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(".local/tmp/comsol_seepage"))
    parser.add_argument("--pressure-output", type=Path, default=None)
    parser.add_argument("--pc-variant", choices=["pmg", "gamg", "none"], default="pmg")
    parser.add_argument("--pmg-smoother-max-it", type=int, default=10)
    parser.add_argument("--pmg-coarse-max-it", type=int, default=5)
    parser.add_argument("--pmg-coarse-pc-type", default="gamg")
    parser.add_argument("--log-view", action="store_true")
    parser.add_argument("--petsc-opt", action="append", default=[])
    return parser


def run_petsc_backend(args: argparse.Namespace) -> int:
    from mpi4py import MPI
    from petsc4py import PETSc  # noqa: F401

    from petsc_ssr.native import _core

    comm = MPI.COMM_WORLD
    data_dir = args.output_dir / "data"
    pressure_output = args.pressure_output or (data_dir / "hydro_pressure.bin")
    if comm.rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
        pressure_output.parent.mkdir(parents=True, exist_ok=True)
        options_record = dict(vars(args))
        options_record["pressure_output_effective"] = pressure_output
        (data_dir / "hydro_options.json").write_text(json.dumps(options_record, default=str, indent=2), encoding="utf-8")
    comm.Barrier()
    tokens = [
        "-hydro_mesh",
        str(args.mesh),
        "-hydro_elem_type",
        args.elem_type,
        "-hydro_pc_variant",
        args.pc_variant,
        "-hydro_newton_tol",
        f"{args.linear_tolerance:.17g}",
        "-hydro_newton_max_it",
        str(args.newton_max_it),
        "-hydro_ksp_rtol",
        f"{args.linear_tolerance:.17g}",
        "-hydro_ksp_max_it",
        str(args.linear_max_it),
        "-hydro_pmg_smoother_max_it",
        str(args.pmg_smoother_max_it),
        "-hydro_pmg_coarse_max_it",
        str(args.pmg_coarse_max_it),
        "-hydro_pmg_coarse_pc_type",
        args.pmg_coarse_pc_type,
        "-hydro_summary_json",
        str(data_dir / "hydro_summary.json"),
        "-hydro_pressure_binary",
        str(pressure_output),
    ]
    if args.log_view:
        tokens.append("-hydro_log_view")
    tokens.extend(args.petsc_opt)
    _core.run_hydro_options(quote_option_tokens(tokens))
    comm.Barrier()
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.backend == "petsc":
        if args.parse_only:
            raise SystemExit("--parse-only is only supported by --backend scipy")
        return run_petsc_backend(args)
    result = solve_comsol_seepage(
        mesh_path=args.mesh,
        elem_type=args.elem_type,
        linear_tolerance=args.linear_tolerance,
        linear_max_iter=args.linear_max_it,
        newton_max_it=args.newton_max_it,
        parse_only=args.parse_only,
    )
    print_hydro_result(result)
    if isinstance(result, HydroResult):
        summary_path, npz_path = result.write_outputs(args.output_dir)
        print(f"HYDRO_OUTPUT summary={summary_path} npz={npz_path}")
    elif isinstance(result, HydroMesh):
        print("HYDRO_OUTPUT parse_only=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
