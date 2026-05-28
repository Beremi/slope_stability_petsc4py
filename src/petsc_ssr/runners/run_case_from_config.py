from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from mpi4py import MPI

from ..context import SsrContext
from ..hydro_cases import run_hydro_case, translate_hydro_case_toml, write_coupled_pressure_table, write_hydro_case_outputs
from ..case_config import (
    benchmark_capability_rows,
    ensure_engine_imports,
    translate_case_toml,
    write_capability_report,
    write_mechanics_constraint_table,
    write_case_outputs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PETSc SSR case TOML files when supported.")
    parser.add_argument("config", nargs="?", help="Benchmark case.toml to run")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--out_dir", dest="output_dir", type=Path, help="Compatibility alias used by benchmark notebooks")
    parser.add_argument("--refine-levels", type=int, default=None, help="Uniform DMPlex refinement override")
    parser.add_argument("--omega-max", type=float, default=None)
    parser.add_argument("--continuation-method", choices=["indirect", "direct"], default=None)
    parser.add_argument("--continuation-step-max", type=int, default=None)
    parser.add_argument("--linear-rtol", type=float, default=None)
    parser.add_argument("--ksp-max-it", type=int, default=None)
    parser.add_argument("--force-c-baseline", action="store_true", help="Use the maintained C baseline deflated FGMRES profile")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-benchmarks", action="store_true")
    parser.add_argument("--capability-report", type=Path, default=None)
    args = parser.parse_args(argv)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if args.list_benchmarks:
        rows = benchmark_capability_rows()
        if rank == 0:
            if args.capability_report:
                write_capability_report(args.capability_report, rows)
            print(json.dumps(rows, indent=2), flush=True)
        return 0

    if not args.config:
        parser.error("config is required unless --list-benchmarks is used")

    hydro_translation = translate_hydro_case_toml(args.config)
    if hydro_translation.supported:
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = Path(".local") / "tmp" / "cases" / Path(args.config).parent.name
        if rank == 0:
            print(f"CASE_TRANSLATION supported=true reason={hydro_translation.reason}", flush=True)
        if args.dry_run:
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "analysis": "seepage",
                            "mesh": str(hydro_translation.resolved.mesh_path),
                            "elem_type": hydro_translation.elem_type,
                            "reason": hydro_translation.reason,
                        },
                        indent=2,
                    ),
                    flush=True,
                )
            return 0
        result = run_hydro_case(hydro_translation, output_dir)
        write_hydro_case_outputs(result, hydro_translation, args.config)
        if rank == 0:
            print(
                "CASE_RESULT "
                f"output={result.output_dir} seepage_final={result.summary.get('final_criterion')} "
                f"linear={result.summary.get('linear_iterations')} wall={result.summary.get('wall_time')}",
                flush=True,
            )
        return 0

    translation = translate_case_toml(args.config, refine_levels=args.refine_levels, force_full_c_baseline=args.force_c_baseline)
    if rank == 0:
        print(
            f"CASE_TRANSLATION supported={str(translation.supported).lower()} reason={translation.reason}",
            flush=True,
        )
    if not translation.supported:
        return 2
    assert translation.problem is not None
    assert translation.options is not None
    if args.omega_max is not None:
        translation.options.omega_max = args.omega_max
    if args.continuation_method is not None:
        translation.options.continuation_method = args.continuation_method
    if args.continuation_step_max is not None:
        translation.options.continuation_step_max = args.continuation_step_max
    if args.linear_rtol is not None:
        translation.options.linear.rtol = args.linear_rtol
    if args.ksp_max_it is not None:
        translation.options.linear.max_it = args.ksp_max_it

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(".local") / "tmp" / "cases" / Path(args.config).parent.name

    if args.dry_run:
        if rank == 0:
                print(json.dumps({"problem": translation.problem.to_dict(), "options": translation.options.option_tokens()}, indent=2), flush=True)
        return 0

    metadata = dict(translation.problem.metadata)
    bc_nodes_csv = output_dir / "data" / "mechanics_bc_nodes.csv"
    if rank == 0:
        bc_nodes_csv = write_mechanics_constraint_table(translation, output_dir)
    comm.Barrier()
    metadata["mechanics_bc_nodes_csv"] = str(bc_nodes_csv)
    translation = replace(translation, problem=replace(translation.problem, metadata=metadata))

    if bool(translation.problem.metadata.get("seepage_coupled", False)):
        hydro_translation = translate_hydro_case_toml(args.config, allow_coupled=True)
        if not hydro_translation.supported:
            if rank == 0:
                print(f"CASE_COUPLED_HYDRO supported=false reason={hydro_translation.reason}", flush=True)
            return 2
        hydro_output = output_dir / "hydro_prepass"
        if rank == 0:
            print(f"CASE_COUPLED_HYDRO supported=true reason={hydro_translation.reason} output={hydro_output}", flush=True)
        hydro_result = run_hydro_case(hydro_translation, hydro_output)
        write_hydro_case_outputs(hydro_result, hydro_translation, args.config)
        pressure_csv = write_coupled_pressure_table(hydro_result, hydro_translation)
        ensure_engine_imports()
        from petsc_ssr.problem_asset_runtime import load_seepage_problem_spec

        seepage_grho = float(load_seepage_problem_spec(hydro_translation.resolved).seepage.water_unit_weight)
        metadata = dict(translation.problem.metadata)
        metadata["seepage_pressure_csv"] = str(pressure_csv)
        metadata["seepage_grho"] = seepage_grho
        translation = replace(translation, problem=replace(translation.problem, metadata=metadata))

    with SsrContext(translation.problem, translation.options, output_dir=output_dir) as ctx:
        result = ctx.run()
    write_case_outputs(result, translation, args.config)
    if rank == 0:
        print(
            "CASE_RESULT "
            f"output={result.output_dir} lambda={result.summary.get('lambda_last')} "
            f"omega={result.summary.get('omega_last')} linear={result.summary.get('total_linear_its')} "
            f"wall={result.summary.get('wall_time')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
