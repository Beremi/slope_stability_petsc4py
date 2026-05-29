"""Suite and target comparison command helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


ENGINE_ROOT = Path(__file__).resolve().parents[4]


def validate_suite_payload(suite_toml: Path) -> dict[str, Any]:
    from petsc_ssr.benchmarks.suites import expand_suite, load_suite

    spec = load_suite(suite_toml)
    runs = expand_suite(spec, run_root=ENGINE_ROOT / ".local" / "runs" / spec.id)
    return {
        "suite": {
            "id": spec.id,
            "title": spec.title,
            "source": None if spec.source is None else str(spec.source),
            "cases": list(spec.cases),
            "profiles": list(spec.profiles),
            "ranks": list(spec.ranks),
            "repeats": int(spec.repeats),
            "timeout": spec.timeout,
            "resources": spec.resources,
            "environment": spec.environment,
            "collect": spec.collect,
        },
        "sweeps": {
            "refine_levels": [value for value in spec.refine_levels if value is not None],
            "linear_rtol": [value for value in spec.linear_rtols if value is not None],
            "continuation_step_max": [value for value in spec.continuation_step_max if value is not None],
        },
        "run_count": len(runs),
        "resolved_profile_groups": sorted({f"{run.profile}@{run.ranks}" for run in runs}),
    }


def expand_suite_manifest(suite_toml: Path, *, output: Path | None = None) -> Path:
    from petsc_ssr.benchmarks.suites import expand_suite, load_suite, write_manifest

    spec = load_suite(suite_toml)
    manifest_path = output or (ENGINE_ROOT / ".local" / "runs" / spec.id / "manifest.json")
    runs = expand_suite(spec, run_root=manifest_path.parent)
    return write_manifest(spec, runs, manifest_path)


def run_suite_manifest(
    suite_toml: Path,
    *,
    run_root: Path | None = None,
    dry_run: bool = False,
    max_runs: int | None = None,
) -> Path:
    from petsc_ssr.benchmarks.suites import load_suite, run_suite

    spec = load_suite(suite_toml)
    return run_suite(spec, run_root=run_root, dry_run=dry_run, max_runs=max_runs)


def write_suite_report(run_root: Path, *, output: Path | None = None) -> Path:
    from petsc_ssr.benchmarks.report import write_report

    return write_report(run_root, output=output)


def compare_suite_targets(run_root: Path, target_root: Path, *, output: Path | None = None) -> Path:
    from petsc_ssr.benchmarks.compare import compare_targets

    return compare_targets(run_root, target_root, output=output)


def suite_command_path(args: argparse.Namespace) -> Path | None:
    if args.suite_command == "expand":
        return expand_suite_manifest(args.suite_toml, output=args.output)
    if args.suite_command == "run":
        return run_suite_manifest(args.suite_toml, run_root=args.run_root, dry_run=bool(args.dry_run), max_runs=args.max_runs)
    if args.suite_command == "report":
        return write_suite_report(args.run_root, output=args.output)
    return None
