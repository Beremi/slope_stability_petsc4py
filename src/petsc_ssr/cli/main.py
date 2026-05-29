from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mpi4py import MPI


ENGINE_ROOT = Path(__file__).resolve().parents[3]
CASE_ROOT = ENGINE_ROOT / "benchmarks" / "cases"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="petsc-ssr", description="Standalone PETSc SSR engine.")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run a case TOML through the maintained PETSc/C engine.")
    run.add_argument("case_toml", type=Path)
    run.add_argument("--profile", default=None, help="Solver profile override, e.g. pmg-deflated-baseline.")
    run.add_argument("--output", "--output-dir", dest="output_dir", type=Path, default=None)
    run.add_argument("--omega-max", type=float, default=None)
    run.add_argument("--continuation-step-max", type=int, default=None)
    run.add_argument("--linear-rtol", type=float, default=None)
    run.add_argument("--ksp-max-it", type=int, default=None)
    run.add_argument("--refine-levels", type=int, default=None)
    run.add_argument(
        "--output-preset",
        default=None,
        choices=("standard", "standard-continuation", "standard-seepage", "performance", "smoke", "none"),
    )
    run.add_argument(
        "--petsc-opt",
        action="append",
        default=[],
        help="Append one PETSc option token after profile resolution; repeat for option/value pairs.",
    )
    run.add_argument(
        "--force-c-baseline",
        action="store_true",
        help="Debug escape hatch: force the maintained C baseline outer Krylov choice after profile resolution.",
    )
    run.add_argument(
        "--write-coordinate-bc-table",
        action="store_true",
        help="Debug compatibility: write/pass mechanics_bc_nodes.csv coordinate constraints.",
    )

    case = sub.add_parser("case", help="Inspect case TOML files.")
    case_sub = case.add_subparsers(dest="case_command")
    validate = case_sub.add_parser("validate", help="Validate a case TOML and print resolved metadata.")
    validate.add_argument("case_toml", nargs="?", type=Path)
    validate.add_argument("--all", dest="all_cases", action="store_true", help="Validate every benchmark case TOML.")
    validate.add_argument("--cases-root", type=Path, default=CASE_ROOT, help=argparse.SUPPRESS)
    explain = case_sub.add_parser("explain", help="Print the resolved case/profile model without solving.")
    explain.add_argument("case_toml", type=Path)
    dry = case_sub.add_parser("dry-run", help="Show the translated problem and PETSc options without solving.")
    dry.add_argument("case_toml", type=Path)
    dry.add_argument("--profile", default=None)
    dry.add_argument("--output", "--output-dir", dest="output_dir", type=Path, default=None)
    dry.add_argument(
        "--output-preset",
        default=None,
        choices=("standard", "standard-continuation", "standard-seepage", "performance", "smoke", "none"),
    )
    dry.add_argument(
        "--write-coordinate-bc-table",
        action="store_true",
        help="Debug compatibility: write/pass mechanics_bc_nodes.csv coordinate constraints.",
    )

    profile = sub.add_parser("profile", help="Inspect continuation, Newton, seepage, and solver profiles.")
    profile_sub = profile.add_subparsers(dest="profile_command")
    profile_explain = profile_sub.add_parser("explain", help="Print a resolved profile without launching a case.")
    profile_explain.add_argument("name", help="Profile name, e.g. pmg-deflated-baseline.")
    profile_explain.add_argument("--kind", choices=("solver", "continuation", "newton", "seepage"), default="solver")
    profile_explain.add_argument("--world-size", type=int, default=None, help="MPI world size used to resolve rank-adaptive solver policy.")
    profile_explain.add_argument("--element", default="P4", help="Element order used to resolve concrete PC variant, e.g. P1 or P4.")
    profile_validate = profile_sub.add_parser("validate", help="Validate committed profile registries.")
    profile_validate.add_argument("--kind", choices=("all", "solver", "continuation", "newton", "seepage"), default="all")
    profile_validate.add_argument("--world-size", type=int, action="append", default=None, help="MPI world size used to check solver rank policy; repeatable.")
    profile_validate.add_argument("--element", action="append", default=None, help="Element order used to check concrete solver PC policy; repeatable.")

    mesh = sub.add_parser("mesh-only", help="Inspect mesh, labels, materials, and constrained DOFs for a case.")
    mesh.add_argument("case_toml", type=Path)
    mesh.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")

    asset = sub.add_parser("asset", help="Inspect mesh assets.")
    asset_sub = asset.add_subparsers(dest="asset_command")
    asset_validate = asset_sub.add_parser("validate", help="Validate an asset definition and declared supports.")
    asset_validate.add_argument("asset", nargs="?", help="Asset id or meshes/<asset> path.")
    asset_validate.add_argument("--all", dest="all_assets", action="store_true", help="Validate every registered mesh asset.")

    sub.add_parser("doctor", help="Inspect runtime, optional dependencies, assets, profiles, and suites.")

    bench = sub.add_parser("benchmark", help="Benchmark maintenance helpers.")
    bench_sub = bench.add_subparsers(dest="benchmark_command")
    init = bench_sub.add_parser("init", help="Create a benchmark case from an asset, or regenerate existing artifacts.")
    init.add_argument("case", nargs="?", help="Case slug or case.toml. Omit to regenerate all existing cases.")
    init.add_argument("--asset", help="Asset id for a new case skeleton, e.g. 3d_hetero_slope.")
    init.add_argument("--variant", help="Mesh variant for a new case. Defaults to the asset default variant.")
    init.add_argument("--element", default="P2", help="Element order for a new case, e.g. P1, P2, P4.")
    init.add_argument("--analysis", default="ssr", choices=("ssr", "ll", "seepage"), help="Analysis type for a new case.")
    init.add_argument("--title", help="Human-readable title for a new case.")
    init.add_argument("--linear-profile", default="pmg-deflated-baseline", help="Solver profile for a new case.")
    init.add_argument("--cases-root", type=Path, default=CASE_ROOT, help=argparse.SUPPRESS)
    init.add_argument("--suites-root", type=Path, default=ENGINE_ROOT / "benchmarks" / "suites", help=argparse.SUPPRESS)
    init.add_argument("--targets-root", type=Path, default=ENGINE_ROOT / "benchmarks" / "targets", help=argparse.SUPPRESS)
    init.add_argument("--overwrite", action="store_true", help="Replace generated files for a new case skeleton.")
    init.add_argument("--no-notebooks", action="store_true", help="Create or update README/run files without generating ipynb notebooks.")
    init.add_argument("--check", action="store_true", help="Validate generated benchmark scaffolding without rewriting files.")
    bench_list = bench_sub.add_parser("list", help="List registered benchmark cases, suites, and targets.")
    bench_list.add_argument("--kind", choices=("all", "cases", "suites", "targets"), default="all")
    bench_list.add_argument("--cases-root", type=Path, default=CASE_ROOT, help=argparse.SUPPRESS)
    bench_list.add_argument("--suites-root", type=Path, default=ENGINE_ROOT / "benchmarks" / "suites", help=argparse.SUPPRESS)
    bench_list.add_argument("--targets-root", type=Path, default=ENGINE_ROOT / "benchmarks" / "targets", help=argparse.SUPPRESS)

    suite = sub.add_parser("suite", help="Run and report benchmark suites.")
    suite_sub = suite.add_subparsers(dest="suite_command")
    suite_validate = suite_sub.add_parser("validate", help="Validate a suite TOML without writing a manifest.")
    suite_validate.add_argument("suite_toml", type=Path)
    suite_expand = suite_sub.add_parser("expand", help="Expand a suite TOML to a resolved manifest.")
    suite_expand.add_argument("suite_toml", type=Path)
    suite_expand.add_argument("--output", type=Path, default=None)
    suite_run = suite_sub.add_parser("run", help="Run a suite or write its manifest with --dry-run.")
    suite_run.add_argument("suite_toml", type=Path)
    suite_run.add_argument("--output", "--run-root", dest="run_root", type=Path, default=None)
    suite_run.add_argument("--dry-run", action="store_true")
    suite_run.add_argument("--max-runs", type=int, default=None)
    suite_report = suite_sub.add_parser("report", help="Write a Markdown/CSV report from a suite run root.")
    suite_report.add_argument("run_root", type=Path)
    suite_report.add_argument("--output", type=Path, default=None)

    targets = sub.add_parser("targets", help="Validate and compare benchmark target scaffolds.")
    targets_sub = targets.add_subparsers(dest="targets_command")
    targets_validate = targets_sub.add_parser("validate", help="Validate benchmark target JSON files.")
    targets_validate.add_argument("target", type=Path, help="Target JSON file or directory.")
    compare = targets_sub.add_parser("compare", help="Compare a suite run root with a target directory.")
    compare.add_argument("run_root", type=Path)
    compare.add_argument("target_root", type=Path)
    compare.add_argument("--output", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run_case(args)
    if args.command == "case":
        if args.case_command == "validate":
            if args.all_cases:
                return _validate_all_cases(args.cases_root)
            if args.case_toml is None:
                parser.error("case validate requires a case TOML unless --all is used")
            return _validate_case(args.case_toml)
        if args.case_command == "explain":
            return _explain_case(args.case_toml)
        if args.case_command == "dry-run":
            return _dry_run_case(args)
    if args.command == "profile":
        if args.profile_command == "explain":
            return _explain_profile(args)
        if args.profile_command == "validate":
            return _validate_profiles(args)
    if args.command == "mesh-only":
        return _mesh_only(args.case_toml, args.output)
    if args.command == "asset" and args.asset_command == "validate":
        if args.all_assets:
            return _asset_validate_all()
        if not args.asset:
            parser.error("asset validate requires an asset id/path unless --all is used")
        return _asset_validate(args.asset)
    if args.command == "doctor":
        return _doctor()
    if args.command == "benchmark":
        if args.benchmark_command == "init":
            return _benchmark_init(args)
        if args.benchmark_command == "list":
            return _benchmark_list(args)
    if args.command == "suite":
        if args.suite_command == "validate":
            return _suite_validate(args)
        return _suite_command(args)
    if args.command == "targets":
        if args.targets_command == "validate":
            return _targets_validate(args)
        if args.targets_command == "compare":
            return _targets_compare(args)
    parser.print_help()
    return 2


def _run_case(args: argparse.Namespace) -> int:
    from petsc_ssr.cli.commands.run import run_case

    return run_case(args)


def _validate_case(case_toml: Path) -> int:
    from petsc_ssr.cli.commands.case import validate_case_payload

    _rank0_print(json.dumps(validate_case_payload(case_toml), indent=2))
    return 0


def _validate_all_cases(cases_root: Path) -> int:
    from petsc_ssr.cli.commands.case import validate_all_cases_payload

    payload = validate_all_cases_payload(cases_root)
    _rank0_print(json.dumps(payload, indent=2))
    return 0 if payload["errors"] == 0 else 2


def _explain_case(case_toml: Path) -> int:
    from petsc_ssr.cli.commands.case import explain_case_payload

    _rank0_print(json.dumps(explain_case_payload(case_toml), indent=2))
    return 0


def _dry_run_case(args: argparse.Namespace) -> int:
    from petsc_ssr.cli.commands.case import dry_run_case

    return dry_run_case(args)


def _explain_profile(args: argparse.Namespace) -> int:
    from petsc_ssr.cli.commands.profile import explain_profile_payload

    payload = explain_profile_payload(
        args.name,
        kind=args.kind,
        world_size=args.world_size,
        element=args.element,
    )
    _rank0_print(json.dumps(payload, indent=2))
    return 0


def _validate_profiles(args: argparse.Namespace) -> int:
    from petsc_ssr.cli.commands.profile import validate_profiles_payload

    payload = validate_profiles_payload(
        kind=args.kind,
        world_sizes=args.world_size,
        elements=args.element,
    )
    _rank0_print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 2


def _mesh_only(case_toml: Path, output: Path | None) -> int:
    from petsc_ssr.cli.commands.mesh import MeshInspectionError, mesh_report_payload

    try:
        report = mesh_report_payload(case_toml)
    except MeshInspectionError as exc:
        _rank0_print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2
    text = json.dumps(report, indent=2)
    if output is not None and MPI.COMM_WORLD.Get_rank() == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    _rank0_print(text)
    return 0


def _benchmark_init(args: argparse.Namespace) -> int:
    from petsc_ssr.cli.commands.benchmark import benchmark_init_result

    result = benchmark_init_result(args)
    if result.payload is not None:
        _rank0_print(json.dumps(result.payload, indent=2))
    if result.path is not None:
        _rank0_print(str(result.path))
    return result.status


def _benchmark_list(args: argparse.Namespace) -> int:
    from petsc_ssr.cli.commands.benchmark import benchmark_list_payload

    payload = benchmark_list_payload(
        kind=args.kind,
        cases_root=args.cases_root,
        suites_root=args.suites_root,
        targets_root=args.targets_root,
    )
    _rank0_print(json.dumps(payload, indent=2))
    return 0


def _asset_validate(asset: str) -> int:
    from petsc_ssr.cli.commands.asset import validate_asset_payload

    payload = validate_asset_payload(asset)
    _rank0_print(json.dumps(payload, indent=2))
    return 0 if not payload["errors"] else 2


def _asset_validate_all() -> int:
    from petsc_ssr.cli.commands.asset import validate_all_assets_payload

    payload = validate_all_assets_payload()
    _rank0_print(json.dumps(payload, indent=2))
    return 0 if payload["errors"] == 0 else 2


def _doctor() -> int:
    from petsc_ssr.cli.commands.doctor import doctor_payload

    _rank0_print(json.dumps(doctor_payload(ENGINE_ROOT), indent=2))
    return 0


def _suite_command(args: argparse.Namespace) -> int:
    from petsc_ssr.cli.commands.suite import suite_command_path

    path = suite_command_path(args)
    if path is not None:
        _rank0_print(str(path))
        return 0
    return 2


def _suite_validate(args: argparse.Namespace) -> int:
    from petsc_ssr.cli.commands.suite import validate_suite_payload

    _rank0_print(json.dumps(validate_suite_payload(args.suite_toml), indent=2))
    return 0


def _targets_compare(args: argparse.Namespace) -> int:
    from petsc_ssr.cli.commands.suite import compare_suite_targets

    path = compare_suite_targets(args.run_root, args.target_root, output=args.output)
    _rank0_print(str(path))
    return 0


def _targets_validate(args: argparse.Namespace) -> int:
    from petsc_ssr.cli.commands.targets import validate_targets_payload

    payload = validate_targets_payload(args.target)
    _rank0_print(json.dumps(payload, indent=2))
    return 0 if payload["errors"] == 0 else 2


def _rank0_print(text: str) -> None:
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(text, flush=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
