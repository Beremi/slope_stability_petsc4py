"""Registry helpers for user-facing benchmark cases, suites, and targets."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from .suites import load_suite
from .targets import load_target, target_metric_names


ENGINE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES_ROOT = ENGINE_ROOT / "benchmarks" / "cases"
DEFAULT_SUITES_ROOT = ENGINE_ROOT / "benchmarks" / "suites"
DEFAULT_TARGETS_ROOT = ENGINE_ROOT / "benchmarks" / "targets"


def discover_benchmark_registry(
    *,
    cases_root: str | Path = DEFAULT_CASES_ROOT,
    suites_root: str | Path = DEFAULT_SUITES_ROOT,
    targets_root: str | Path = DEFAULT_TARGETS_ROOT,
) -> dict[str, list[dict[str, Any]]]:
    """Return a machine-readable registry of benchmark cases, suites, and targets."""

    return {
        "cases": discover_cases(cases_root),
        "suites": discover_suites(suites_root),
        "targets": discover_targets(targets_root),
    }


def discover_cases(cases_root: str | Path = DEFAULT_CASES_ROOT) -> list[dict[str, Any]]:
    root = Path(cases_root)
    entries: list[dict[str, Any]] = []
    for case_toml in sorted(root.glob("*/case.toml")):
        payload = tomllib.loads(case_toml.read_text(encoding="utf-8"))
        case = dict(payload.get("case", {}))
        mesh = dict(payload.get("mesh", {}))
        physics = dict(payload.get("physics", {}))
        continuation = dict(payload.get("continuation", {}))
        newton = dict(payload.get("newton", {}))
        linear = dict(payload.get("linear", {}))
        entry = {
            "id": str(case.get("id") or case.get("name") or case_toml.parent.name),
            "slug": case_toml.parent.name,
            "title": str(case.get("title", case_toml.parent.name)),
            "path": str(case_toml),
            "tags": [str(tag) for tag in case.get("tags", [])],
            "analysis": _analysis_from_physics(physics),
            "asset": mesh.get("asset"),
            "variant": mesh.get("variant"),
            "element": mesh.get("element"),
            "profiles": {
                "continuation": continuation.get("profile"),
                "newton": newton.get("profile"),
                "linear": linear.get("profile"),
            },
            "notebook": str(case_toml.parent / "notebook.toml") if (case_toml.parent / "notebook.toml").exists() else None,
        }
        entries.append(entry)
    return entries


def discover_suites(suites_root: str | Path = DEFAULT_SUITES_ROOT) -> list[dict[str, Any]]:
    root = Path(suites_root)
    entries: list[dict[str, Any]] = []
    for suite_toml in sorted(root.glob("*.toml")):
        spec = load_suite(suite_toml)
        entries.append(
            {
                "id": spec.id,
                "title": spec.title,
                "path": str(suite_toml),
                "cases": list(spec.cases),
                "profiles": list(spec.profiles),
                "ranks": list(spec.ranks),
                "repeats": int(spec.repeats),
                "sweeps": {
                    "refine_levels": [value for value in spec.refine_levels if value is not None],
                    "linear_rtol": [value for value in spec.linear_rtols if value is not None],
                    "continuation_step_max": [value for value in spec.continuation_step_max if value is not None],
                },
                "resources": spec.resources,
                "environment": spec.environment,
            }
        )
    return entries


def discover_targets(targets_root: str | Path = DEFAULT_TARGETS_ROOT) -> list[dict[str, Any]]:
    root = Path(targets_root)
    entries: list[dict[str, Any]] = []
    for target_path in sorted(root.rglob("*.json")):
        payload = json.loads(target_path.read_text(encoding="utf-8"))
        if "case" not in payload:
            continue
        payload = load_target(target_path)
        rank_metrics = payload.get("rank_metrics", {})
        groups = payload.get("groups", [])
        entries.append(
            {
                "case": payload.get("case", target_path.stem),
                "profile": payload.get("profile"),
                "suite": payload.get("suite"),
                "status": payload.get("status"),
                "path": str(target_path),
                "target_set": _target_set(root, target_path),
                "metrics": target_metric_names(payload),
                "rank_metric_groups": len(rank_metrics) if isinstance(rank_metrics, dict) else 0,
                "sweep_metric_groups": len(groups) if isinstance(groups, list) else 0,
            }
        )
    return entries


def registry_subset(registry: dict[str, list[dict[str, Any]]], kind: str) -> dict[str, list[dict[str, Any]]]:
    if kind == "all":
        return registry
    if kind not in registry:
        raise ValueError(f"Unknown benchmark registry kind {kind!r}.")
    return {kind: registry[kind]}


def _analysis_from_physics(physics: dict[str, Any]) -> str:
    mechanics = dict(physics.get("mechanics", {}))
    seepage = dict(physics.get("seepage", {}))
    model = str(mechanics.get("model", "")).strip().lower()
    if model:
        return "ll" if "limit" in model else "ssr"
    if seepage:
        return "seepage"
    return "unknown"


def _target_set(root: Path, target_path: Path) -> str:
    parent = target_path.parent
    try:
        rel = parent.relative_to(root)
    except ValueError:
        return parent.name
    text = rel.as_posix()
    return "." if text == "." else text
