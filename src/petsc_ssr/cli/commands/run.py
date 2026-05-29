"""Run command adapter helpers."""

from __future__ import annotations

import argparse

from petsc_ssr.cli.commands.case import case_override


def build_run_case_argv(args: argparse.Namespace) -> list[str]:
    case_toml = case_override(args.case_toml, profile=args.profile, output_preset=args.output_preset)
    argv = [str(case_toml)]
    if args.output_dir is not None:
        argv.extend(["--output-dir", str(args.output_dir)])
    if args.refine_levels is not None:
        argv.extend(["--refine-levels", str(args.refine_levels)])
    if args.omega_max is not None:
        argv.extend(["--omega-max", str(args.omega_max)])
    if args.continuation_step_max is not None:
        argv.extend(["--continuation-step-max", str(args.continuation_step_max)])
    if args.linear_rtol is not None:
        argv.extend(["--linear-rtol", str(args.linear_rtol)])
    if args.ksp_max_it is not None:
        argv.extend(["--ksp-max-it", str(args.ksp_max_it)])
    if args.output_preset is not None:
        argv.extend(["--output-preset", str(args.output_preset)])
    for token in args.petsc_opt:
        argv.append(f"--petsc-opt={token}")
    if args.force_c_baseline:
        argv.append("--force-c-baseline")
    if args.write_coordinate_bc_table:
        argv.append("--write-coordinate-bc-table")
    return argv


def run_case(args: argparse.Namespace) -> int:
    from petsc_ssr.runners import run_case_from_config

    return run_case_from_config.main(build_run_case_argv(args))
