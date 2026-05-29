"""Benchmark maintenance command helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkInitResult:
    status: int
    payload: dict[str, Any] | None = None
    path: Path | None = None


def benchmark_init_result(args: argparse.Namespace) -> BenchmarkInitResult:
    if args.check:
        payload = benchmark_check_payload(
            args.case,
            cases_root=args.cases_root,
            suites_root=getattr(args, "suites_root", None),
            targets_root=getattr(args, "targets_root", None),
            check_notebooks=not bool(args.no_notebooks),
        )
        return BenchmarkInitResult(status=0 if payload["ok"] else 1, payload=payload)

    if args.asset:
        path = create_benchmark_case_from_args(args)
        return BenchmarkInitResult(status=0, path=path)

    regenerate_benchmark_artifacts(
        args.case,
        cases_root=args.cases_root,
        generate_notebooks=not bool(args.no_notebooks),
    )
    return BenchmarkInitResult(status=0)


def benchmark_check_payload(
    case: str | None,
    *,
    cases_root: Path,
    suites_root: Path | None = None,
    targets_root: Path | None = None,
    check_notebooks: bool,
) -> dict[str, Any]:
    from petsc_ssr.benchmarks import generators
    from petsc_ssr.benchmarks.registry import DEFAULT_SUITES_ROOT, DEFAULT_TARGETS_ROOT

    if case:
        case_path = Path(case)
        target = case_path if case_path.exists() else cases_root / case / "case.toml"
        issues = generators.check_case_artifacts(target, check_notebooks=check_notebooks)
    else:
        issues = generators.check_generated_cases(cases_root, check_notebooks=check_notebooks)
        suites = Path(suites_root) if suites_root is not None else DEFAULT_SUITES_ROOT
        targets = Path(targets_root) if targets_root is not None else DEFAULT_TARGETS_ROOT
        issues.extend(_benchmark_registry_issues(suites_root=suites, targets_root=targets))
    return {
        "ok": not issues,
        "cases_root": str(cases_root),
        "suites_root": None if case else str(Path(suites_root) if suites_root is not None else DEFAULT_SUITES_ROOT),
        "targets_root": None if case else str(Path(targets_root) if targets_root is not None else DEFAULT_TARGETS_ROOT),
        "check_notebooks": check_notebooks,
        "issues": issues,
    }


def _benchmark_registry_issues(*, suites_root: Path, targets_root: Path) -> list[str]:
    from petsc_ssr.benchmarks.registry import discover_suites, discover_targets
    from petsc_ssr.cli.commands.profile import validate_profiles_payload

    issues: list[str] = []
    profile_payload = validate_profiles_payload()
    for issue in profile_payload["issues"]:
        issues.append(f"invalid profile registry at {issue['path']}: {issue['error']}")
    for label, loader, root in (
        ("suite", discover_suites, suites_root),
        ("target", discover_targets, targets_root),
    ):
        try:
            loader(root)
        except Exception as exc:
            issues.append(f"invalid benchmark {label} registry at {root}: {exc}")
    return issues


def create_benchmark_case_from_args(args: argparse.Namespace) -> Path:
    from petsc_ssr.benchmarks import generators

    if not args.case:
        raise ValueError("benchmark init --asset requires a case slug.")
    return generators.create_case_skeleton(
        args.case,
        asset=args.asset,
        cases_root=args.cases_root,
        variant=args.variant,
        element=args.element,
        analysis=args.analysis,
        title=args.title,
        linear_profile=args.linear_profile,
        overwrite=bool(args.overwrite),
        generate_notebooks=not bool(args.no_notebooks),
    )


def regenerate_benchmark_artifacts(
    case: str | None,
    *,
    cases_root: Path,
    generate_notebooks: bool,
) -> None:
    from petsc_ssr.benchmarks import generators

    if case:
        case_path = Path(case)
        if not case_path.exists():
            case_path = cases_root / case / "case.toml"
        if not case_path.exists():
            raise FileNotFoundError(f"No existing case found at {case_path}; pass --asset to create a new benchmark case.")
        case_dir = case_path.parent
        generators.generate_case_readme(case_dir / "case.toml")
        if generate_notebooks:
            generators.generate_case_notebooks(case_dir / "case.toml")
        return

    if generate_notebooks:
        generators.generate_all(cases_root)
        return
    for case_toml in sorted(cases_root.glob("*/case.toml")):
        generators.generate_case_readme(case_toml)


def benchmark_list_payload(
    *,
    kind: str,
    cases_root: Path,
    suites_root: Path,
    targets_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    from petsc_ssr.benchmarks.registry import discover_benchmark_registry, registry_subset

    registry = discover_benchmark_registry(
        cases_root=cases_root,
        suites_root=suites_root,
        targets_root=targets_root,
    )
    return registry_subset(registry, kind)
