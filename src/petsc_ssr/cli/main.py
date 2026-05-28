from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from mpi4py import MPI

from petsc_ssr.config import load_run_case_config
from petsc_ssr.problem_asset_runtime import build_mesh_for_resolved_asset, load_mechanical_problem_spec, resolve_problem_asset_from_config
from petsc_ssr.runners import run_case_from_config


ENGINE_ROOT = Path(__file__).resolve().parents[3]
CASE_ROOT = ENGINE_ROOT / "benchmarks" / "cases"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="petsc-ssr", description="Standalone PETSc SSR engine.")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run a case TOML through the maintained PETSc/C engine.")
    run.add_argument("case_toml", type=Path)
    run.add_argument("--profile", default=None, help="Solver profile override, e.g. baseline-pmg-deflated.")
    run.add_argument("--output", "--output-dir", dest="output_dir", type=Path, default=None)
    run.add_argument("--omega-max", type=float, default=None)
    run.add_argument("--continuation-step-max", type=int, default=None)
    run.add_argument("--linear-rtol", type=float, default=None)
    run.add_argument("--ksp-max-it", type=int, default=None)
    run.add_argument("--refine-levels", type=int, default=None)

    case = sub.add_parser("case", help="Inspect case TOML files.")
    case_sub = case.add_subparsers(dest="case_command")
    validate = case_sub.add_parser("validate", help="Validate a case TOML and print resolved metadata.")
    validate.add_argument("case_toml", type=Path)
    dry = case_sub.add_parser("dry-run", help="Show the translated problem and PETSc options without solving.")
    dry.add_argument("case_toml", type=Path)
    dry.add_argument("--profile", default=None)
    dry.add_argument("--output", "--output-dir", dest="output_dir", type=Path, default=None)

    mesh = sub.add_parser("mesh-only", help="Inspect mesh, labels, materials, and constrained DOFs for a case.")
    mesh.add_argument("case_toml", type=Path)
    mesh.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")

    bench = sub.add_parser("benchmark", help="Benchmark maintenance helpers.")
    bench_sub = bench.add_subparsers(dest="benchmark_command")
    init = bench_sub.add_parser("init", help="Generate README and notebooks for one case or all cases.")
    init.add_argument("case", nargs="?", help="Case slug or case.toml. Omit to regenerate all.")

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run_case(args)
    if args.command == "case":
        if args.case_command == "validate":
            return _validate_case(args.case_toml)
        if args.case_command == "dry-run":
            return _dry_run_case(args)
    if args.command == "mesh-only":
        return _mesh_only(args.case_toml, args.output)
    if args.command == "benchmark" and args.benchmark_command == "init":
        return _benchmark_init(args.case)
    parser.print_help()
    return 2


def _profile_override(path: Path, profile: str | None) -> Path:
    if not profile:
        return path
    import tomllib

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if "linear" in data:
        data.setdefault("linear", {})["profile"] = profile
    elif "linear_solver" in data:
        data.setdefault("linear", {"profile": profile})
    else:
        data["linear"] = {"profile": profile}
    out = ENGINE_ROOT / ".local" / "tmp" / "profile_overrides" / path.parent.name / f"{profile}.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_dumps_simple_toml(data), encoding="utf-8")
    return out


def _run_case(args: argparse.Namespace) -> int:
    case_toml = _profile_override(args.case_toml, args.profile)
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
    argv.append("--force-c-baseline")
    return run_case_from_config.main(argv)


def _validate_case(case_toml: Path) -> int:
    cfg = load_run_case_config(case_toml).validate()
    resolved = resolve_problem_asset_from_config(cfg)
    payload = {
        "case": cfg.problem.name,
        "analysis": cfg.problem.analysis,
        "dimension": resolved.dimension,
        "asset": resolved.asset_name,
        "mesh_variant": resolved.variant_name,
        "mesh_path": None if resolved.mesh_path is None else str(resolved.mesh_path),
        "element": cfg.problem.elem_type,
        "refine_levels": cfg.problem.refine_levels,
        "partitioner": cfg.problem.partitioner,
        "linear_profile": getattr(cfg.linear_solver, "pmg_profile", None),
    }
    _rank0_print(json.dumps(payload, indent=2))
    return 0


def _dry_run_case(args: argparse.Namespace) -> int:
    case_toml = _profile_override(args.case_toml, args.profile)
    argv = [str(case_toml), "--dry-run"]
    if args.output_dir is not None:
        argv.extend(["--output-dir", str(args.output_dir)])
    return run_case_from_config.main(argv)


def _mesh_only(case_toml: Path, output: Path | None) -> int:
    cfg = load_run_case_config(case_toml).validate()
    resolved = resolve_problem_asset_from_config(cfg)
    mesh = build_mesh_for_resolved_asset(resolved, elem_type=cfg.problem.elem_type)
    mechanical = None
    if cfg.problem.analysis.lower() != "seepage":
        mechanical = load_mechanical_problem_spec(resolved)
    q_mask = np.asarray(getattr(mesh, "q_mask", np.empty((0, 0), dtype=bool)), dtype=bool)
    report: dict[str, Any] = {
        "case": cfg.problem.name,
        "asset": resolved.asset_name,
        "mesh_variant": resolved.variant_name,
        "mesh_path": None if resolved.mesh_path is None else str(resolved.mesh_path),
        "dimension": resolved.dimension,
        "element": cfg.problem.elem_type,
        "nodes": int(np.asarray(mesh.coord).shape[1]),
        "cells": int(np.asarray(mesh.elem).shape[1]),
        "boundary_entities": int(np.asarray(mesh.surf).shape[1]),
        "regions": sorted(getattr(mesh, "region_id_by_name", {}).keys()),
        "boundaries": sorted(getattr(mesh, "boundary_id_by_name", {}).keys()),
        "nodesets": sorted(getattr(mesh, "nodesets", {}).keys()),
        "free_component_dofs": int(np.count_nonzero(q_mask)) if q_mask.size else None,
        "constrained_component_dofs": int(q_mask.size - np.count_nonzero(q_mask)) if q_mask.size else None,
        "materials": 0 if mechanical is None else len(mechanical.material_rows),
    }
    text = json.dumps(report, indent=2)
    if output is not None and MPI.COMM_WORLD.Get_rank() == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    _rank0_print(text)
    return 0


def _benchmark_init(case: str | None) -> int:
    from petsc_ssr.benchmarks import generators

    if case:
        case_path = Path(case)
        if not case_path.exists():
            case_path = CASE_ROOT / case / "case.toml"
        case_dir = case_path.parent
        generators.generate_case_readme(case_dir / "case.toml")
        generators.generate_case_notebooks(case_dir / "case.toml")
    else:
        generators.generate_all(CASE_ROOT)
    return 0


def _rank0_print(text: str) -> None:
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(text, flush=True)


def _dumps_simple_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []
    scalars = {key: value for key, value in data.items() if not isinstance(value, dict)}
    for key, value in scalars.items():
        lines.append(f"{key} = {_toml_value(value)}")
    for section, payload in data.items():
        if not isinstance(payload, dict):
            continue
        lines.append("")
        lines.append(f"[{section}]")
        for key, value in payload.items():
            if isinstance(value, dict):
                lines.append("")
                lines.append(f"[{section}.{key}]")
                for inner_key, inner_value in value.items():
                    lines.append(f"{inner_key} = {_toml_value(inner_value)}")
            else:
                lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines).strip() + "\n"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if value is None:
        return '""'
    return json.dumps(str(value))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
