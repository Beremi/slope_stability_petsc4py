from __future__ import annotations

import argparse
import json
import shutil
import shlex
import sys
from dataclasses import replace
from pathlib import Path

from mpi4py import MPI

from ..config.manifest import (
    build_environment_manifest,
    build_resolved_config,
    build_resolved_run_manifest,
    build_run_command_manifest,
    dumps_resolved_config_toml,
)
from ..hydro_cases import (
    run_hydro_case,
    translate_hydro_case_toml,
    write_coupled_pressure_table,
    write_hydro_case_outputs,
    write_hydro_preflight_artifacts,
)
from ..runtime.options import quote_option_tokens, resolve_run_option_tokens
from ..case_config import (
    benchmark_capability_rows,
    ensure_engine_imports,
    translate_case_toml,
    write_capability_report,
    write_mechanics_constraint_table,
    write_mechanics_label_constraint_table,
    write_mechanics_neumann_label_table,
    write_native_problem_manifest,
    planned_mechanics_neumann_label_table,
    planned_seepage_label_table,
    write_seepage_label_table,
    write_case_outputs,
)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
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
    parser.add_argument("--output-preset", default=None, help="Override the named output preset for this run.")
    parser.add_argument(
        "--petsc-opt",
        action="append",
        default=[],
        help="Append one PETSc option token after profile resolution; repeat for option/value pairs.",
    )
    parser.add_argument("--force-c-baseline", action="store_true", help="Use the maintained C baseline deflated FGMRES profile")
    parser.add_argument(
        "--write-coordinate-bc-table",
        action="store_true",
        help="Debug compatibility: also write/pass mechanics_bc_nodes.csv coordinate constraints.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-benchmarks", action="store_true")
    parser.add_argument("--capability-report", type=Path, default=None)
    args = parser.parse_args(raw_argv)

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

    hydro_translation = translate_hydro_case_toml(args.config, output_preset=args.output_preset)
    if hydro_translation.supported:
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = Path(".local") / "tmp" / "cases" / Path(args.config).parent.name
        if rank == 0:
            print(f"CASE_TRANSLATION supported=true reason={hydro_translation.reason}", flush=True)
        if args.dry_run:
            if rank == 0:
                write_hydro_preflight_artifacts(
                    hydro_translation,
                    output_dir,
                    args.config,
                    comm.Get_size(),
                    runner_argv=raw_argv,
                    mode="dry-run",
                )
                print(
                    json.dumps(
                        {
                            "analysis": "seepage",
                            "mesh": str(hydro_translation.resolved.mesh_path),
                            "elem_type": hydro_translation.elem_type,
                            "reason": hydro_translation.reason,
                            "artifacts": str(output_dir / "data" / "resolved_run_manifest.json"),
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

    translation = translate_case_toml(
        args.config,
        refine_levels=args.refine_levels,
        force_full_c_baseline=args.force_c_baseline,
        output_preset=args.output_preset,
    )
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
        translation.options.continuation_algorithm = args.continuation_method
        translation.options.newton_algorithm = "indirect-ssr" if args.continuation_method == "indirect" else "fixed-load"
    if args.continuation_step_max is not None:
        translation.options.continuation_step_max = args.continuation_step_max
    if args.linear_rtol is not None:
        translation.options.linear.rtol = args.linear_rtol
    if args.ksp_max_it is not None:
        translation.options.linear.max_it = args.ksp_max_it
    for raw in args.petsc_opt:
        translation.options.petsc_options.extend(shlex.split(str(raw)))

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(".local") / "tmp" / "cases" / Path(args.config).parent.name
    output_dir = Path(output_dir).resolve()

    bc_nodes_csv: Path | None = None
    metadata = dict(translation.problem.metadata)
    native_problem_manifest = output_dir / "data" / "native_problem_manifest.json"
    bc_labels_csv = output_dir / "data" / "mechanics_bc_labels.csv"
    neumann_labels_csv = planned_mechanics_neumann_label_table(translation, output_dir)
    seepage_labels_csv = planned_seepage_label_table(translation, output_dir)
    metadata["native_problem_manifest"] = str(native_problem_manifest)
    metadata["mechanics_bc_labels_csv"] = str(bc_labels_csv)
    if neumann_labels_csv is not None:
        metadata["mechanics_neumann_labels_csv"] = str(neumann_labels_csv)
    if seepage_labels_csv is not None:
        metadata["seepage_boundary_labels_csv"] = str(seepage_labels_csv)
    translation = replace(translation, problem=replace(translation.problem, metadata=metadata))

    if args.dry_run:
        if rank == 0:
            write_mechanics_label_constraint_table(translation, output_dir)
            write_mechanics_neumann_label_table(translation, output_dir)
            write_seepage_label_table(translation, output_dir)
            bc_nodes_csv = (
                _write_optional_mechanics_constraint_table(translation, output_dir)
                if args.write_coordinate_bc_table
                else None
            )
            if bc_nodes_csv is not None:
                metadata = dict(translation.problem.metadata)
                metadata["mechanics_bc_nodes_csv"] = str(bc_nodes_csv)
                metadata["debug_coordinate_bc_table"] = True
                translation = replace(translation, problem=replace(translation.problem, metadata=metadata))
            elif args.write_coordinate_bc_table:
                metadata = dict(translation.problem.metadata)
                metadata.pop("mechanics_bc_nodes_csv", None)
                metadata.pop("debug_coordinate_bc_table", None)
                translation = replace(translation, problem=replace(translation.problem, metadata=metadata))
            try:
                translation = _plan_coupled_seepage_pressure_bridge(translation, output_dir, args.config, args.output_preset)
            except RuntimeError as exc:
                print(f"CASE_COUPLED_HYDRO supported=false reason={exc}", flush=True)
                return 2
            native_problem_manifest = write_native_problem_manifest(
                translation,
                output_dir,
                mechanics_coordinate_constraint_table=bc_nodes_csv,
            )
            _validate_native_problem_artifacts(native_problem_manifest)
            _write_preflight_artifacts(
                translation,
                output_dir,
                args.config,
                comm.Get_size(),
                runner_argv=raw_argv,
                mode="dry-run",
            )
            print(
                json.dumps(
                    {
                        "problem": translation.problem.to_dict(),
                        "problem_options": translation.problem.option_tokens(),
                        "solver_options": translation.options.option_tokens(),
                        "options": [*translation.problem.option_tokens(), *translation.options.option_tokens()],
                        "artifacts": str(output_dir / "data" / "resolved_run_manifest.json"),
                    },
                    indent=2,
                ),
                flush=True,
            )
        return 0

    if rank == 0:
        bc_labels_csv = write_mechanics_label_constraint_table(translation, output_dir)
        neumann_labels_csv = write_mechanics_neumann_label_table(translation, output_dir)
        seepage_labels_csv = write_seepage_label_table(translation, output_dir)
        bc_nodes_csv = (
            _write_optional_mechanics_constraint_table(translation, output_dir)
            if args.write_coordinate_bc_table
            else None
        )
        native_problem_manifest = write_native_problem_manifest(
            translation,
            output_dir,
            mechanics_coordinate_constraint_table=bc_nodes_csv,
        )
        _validate_native_problem_artifacts(native_problem_manifest)
    bc_nodes_csv_value = comm.bcast(str(bc_nodes_csv) if bc_nodes_csv is not None else None, root=0)
    bc_nodes_csv = Path(bc_nodes_csv_value) if bc_nodes_csv_value else None
    comm.Barrier()
    metadata = dict(translation.problem.metadata)
    metadata["native_problem_manifest"] = str(native_problem_manifest)
    metadata["mechanics_bc_labels_csv"] = str(bc_labels_csv)
    if bc_nodes_csv is not None:
        metadata["mechanics_bc_nodes_csv"] = str(bc_nodes_csv)
        metadata["debug_coordinate_bc_table"] = True
    else:
        metadata.pop("mechanics_bc_nodes_csv", None)
        metadata.pop("debug_coordinate_bc_table", None)
    if neumann_labels_csv is not None:
        metadata["mechanics_neumann_labels_csv"] = str(neumann_labels_csv)
    if seepage_labels_csv is not None:
        metadata["seepage_boundary_labels_csv"] = str(seepage_labels_csv)
    translation = replace(translation, problem=replace(translation.problem, metadata=metadata))

    if bool(translation.problem.metadata.get("seepage_coupled", False)):
        hydro_translation = translate_hydro_case_toml(args.config, allow_coupled=True, output_preset=args.output_preset)
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
        metadata["seepage_pressure_source"] = "hydro_prepass_coordinate_bridge"
        metadata["seepage_grho"] = seepage_grho
        translation = replace(translation, problem=replace(translation.problem, metadata=metadata))
        if rank == 0:
            native_problem_manifest = write_native_problem_manifest(
                translation,
                output_dir,
                mechanics_coordinate_constraint_table=translation.problem.metadata.get("mechanics_bc_nodes_csv"),
            )
            _validate_native_problem_artifacts(native_problem_manifest)
        comm.Barrier()
        metadata = dict(translation.problem.metadata)
        metadata["native_problem_manifest"] = str(native_problem_manifest)
        translation = replace(translation, problem=replace(translation.problem, metadata=metadata))

    if rank == 0:
        _write_preflight_artifacts(
            translation,
            output_dir,
            args.config,
            comm.Get_size(),
            runner_argv=raw_argv,
            mode="run",
        )
    comm.Barrier()

    from ..context import SsrContext

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

def _write_preflight_artifacts(
    translation,
    output_dir: Path,
    config_path: str | Path,
    mpi_size: int,
    *,
    runner_argv: list[str] | None = None,
    mode: str = "run",
) -> None:
    assert translation.problem is not None
    assert translation.options is not None
    data_dir = output_dir / "data"
    logs_dir = output_dir / "logs"
    exports_dir = output_dir / "exports"
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)

    options_string = quote_option_tokens(
        resolve_run_option_tokens(
            translation.problem,
            translation.options,
            output_dir,
            write_solution_vtu=(translation.config is None or bool(translation.config.export.write_solution_vtu)),
        )
    )
    (data_dir / "problem.json").write_text(json.dumps(translation.problem.to_dict(), indent=2), encoding="utf-8")
    (data_dir / "options.txt").write_text(options_string + "\n", encoding="utf-8")
    (data_dir / "resolved_options.txt").write_text(options_string + "\n", encoding="utf-8")
    (data_dir / "environment.json").write_text(
        json.dumps(build_environment_manifest(mpi_size=int(mpi_size)), indent=2) + "\n",
        encoding="utf-8",
    )
    resolved_run_manifest = build_resolved_run_manifest(
        translation.problem,
        translation.options,
        output_dir=output_dir,
        mpi_size=int(mpi_size),
    )
    (data_dir / "resolved_run_manifest.json").write_text(
        json.dumps(resolved_run_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    command_json = output_dir / "command.json"
    if not command_json.exists():
        command_json.write_text(
            json.dumps(
                build_run_command_manifest(
                    output_dir=output_dir,
                    mpi_size=int(mpi_size),
                    argv=[] if runner_argv is None else runner_argv,
                    mode=mode,
                    entrypoint="petsc_ssr.runners.run_case_from_config",
                    resolved_run_manifest=resolved_run_manifest,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    resolved_config = dumps_resolved_config_toml(
        build_resolved_config(
            translation.problem,
            translation.options,
            output_dir=output_dir,
            mpi_size=int(mpi_size),
        )
    )
    (data_dir / "resolved_config.toml").write_text(resolved_config, encoding="utf-8")
    (exports_dir / "resolved_config.toml").write_text(resolved_config, encoding="utf-8")
    _copy_config_if_different(config_path, output_dir / "generated_case.toml")


def _write_optional_mechanics_constraint_table(translation, output_dir: Path) -> Path | None:
    try:
        return write_mechanics_constraint_table(translation, output_dir)
    except ImportError as exc:
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "preflight_warnings.txt").write_text(
            f"mechanics_bc_nodes.csv skipped: {exc}\n"
            "Native mechanics constraints use DMPlex label tables first; install optional mesh dependencies to write the coordinate compatibility table.\n",
            encoding="utf-8",
        )
        return None


def _plan_coupled_seepage_pressure_bridge(translation, output_dir: Path, config_path: str | Path, output_preset: str | None):
    assert translation.problem is not None
    if not bool(translation.problem.metadata.get("seepage_coupled", False)):
        return translation
    hydro_translation = translate_hydro_case_toml(config_path, allow_coupled=True, output_preset=output_preset)
    if not hydro_translation.supported:
        raise RuntimeError(f"Coupled hydro prepass is not supported for dry-run planning: {hydro_translation.reason}")
    assert hydro_translation.resolved is not None
    ensure_engine_imports()
    from petsc_ssr.problem_asset_runtime import load_seepage_problem_spec

    pressure_csv = Path(output_dir) / "hydro_prepass" / "data" / "coupled_pressure_nodes.csv"
    seepage_grho = float(load_seepage_problem_spec(hydro_translation.resolved).seepage.water_unit_weight)
    metadata = dict(translation.problem.metadata)
    metadata["seepage_pressure_csv"] = str(pressure_csv)
    metadata["seepage_pressure_source"] = "hydro_prepass_coordinate_bridge"
    metadata["seepage_grho"] = seepage_grho
    return replace(translation, problem=replace(translation.problem, metadata=metadata))


def _validate_native_problem_artifacts(manifest_path: str | Path) -> None:
    ensure_engine_imports()
    from petsc_ssr.problem_asset_runtime import validate_native_problem_artifact_contract

    validate_native_problem_artifact_contract(manifest_path)


def _copy_config_if_different(src: str | Path, dst: str | Path) -> None:
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src_path.resolve() == dst_path.resolve():
            return
    except FileNotFoundError:
        pass
    shutil.copyfile(src_path, dst_path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
