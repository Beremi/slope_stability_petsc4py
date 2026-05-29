"""Benchmark target comparison helpers and measured-run summaries."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

from petsc_ssr.benchmarks.logs import options_left_status
from petsc_ssr.benchmarks.targets import load_target, target_metric_payload
from petsc_ssr.runtime.results import load_run_summary

__all__ = ["collect_run_rows", "compare_targets", "summarize_run_group", "target_comparison_rows"]


def collect_run_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in manifest.get("runs", []):
        summary = load_run_summary(run["output_dir"])
        resolved_profile = dict(run.get("resolved_profile", {}))
        linear_profile = dict(resolved_profile.get("linear", {}))
        pc_profile = dict(resolved_profile.get("pc", {}))
        pmg_profile = dict(resolved_profile.get("pmg", {}))
        artifacts = dict(run.get("artifacts", {}))
        pc_fallback = pc_profile.get("fallback_reason")
        if pc_fallback is None:
            pc_fallback = summary.get("pc_variant_fallback_reason")
        rows.append(
            {
                "case": run["case"],
                "profile": run["profile"],
                "ranks": run["ranks"],
                "refine_levels": run.get("refine_levels"),
                "linear_rtol": run.get("linear_rtol"),
                "continuation_step_max": run.get("continuation_step_max"),
                "repeat": run["repeat"],
                "resource": run.get("resource"),
                "launcher": " ".join(str(token) for token in run.get("launcher", []) or []),
                "status": "complete" if summary else "planned",
                "profile_algorithm": linear_profile.get("algorithm") or summary.get("profile_algorithm"),
                "native_linear_algorithm": linear_profile.get("native_algorithm")
                or summary.get("native_linear_algorithm")
                or summary.get("linear_algorithm"),
                "pc_variant": pc_profile.get("variant") or summary.get("pc_variant") or summary.get("variant"),
                "requested_pc_variant": pc_profile.get("requested_variant") or summary.get("requested_pc_variant"),
                "pc_variant_fallback_reason": pc_fallback,
                "pmg_p2_active_ranks": pmg_profile.get("p2_active_ranks") or summary.get("pmg_p2_active_ranks"),
                "pmg_p1_active_ranks": pmg_profile.get("p1_active_ranks") or summary.get("pmg_p1_active_ranks"),
                "lambda_last": summary.get("lambda_last"),
                "omega_last": summary.get("omega_last"),
                "final_rel": summary.get("final_rel"),
                "final_rel_correction": summary.get("final_rel_correction"),
                "wall_time": summary.get("wall_time"),
                "continuation_wall_time": summary.get("continuation_wall_time"),
                "elastic_assembly_time": summary.get("elastic_assembly_time"),
                "global_dofs": summary.get("global_dofs"),
                "accepted_steps": summary.get("accepted_steps"),
                "total_newton_its": summary.get("total_newton_its"),
                "total_linear_its": summary.get("total_linear_its"),
                "total_line_search_its": summary.get("total_line_search_its"),
                "deflation_orthogonalization_time": summary.get("deflation_orthogonalization_time"),
                "deflation_pc_apply_time": summary.get("deflation_pc_apply_time"),
                "deflation_projector_time": summary.get("deflation_projector_time"),
                "options_left": options_left_status(Path(run["output_dir"])),
                "output_dir": run["output_dir"],
                "command_manifest": artifacts.get("command_json"),
                "resolved_run_manifest": artifacts.get("resolved_run_manifest_json"),
                "resolved_options": artifacts.get("resolved_options_txt"),
                "summary": artifacts.get("summary_json"),
                "petsc_log": artifacts.get("petsc_log_txt"),
                "options_left_artifact": artifacts.get("options_left_txt"),
            }
        )
    return rows


def summarize_run_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        "wall_time": "wall_time_median",
        "continuation_wall_time": "continuation_wall_time_median",
        "total_newton_its": "total_newton_its_median",
        "total_linear_its": "total_linear_its_median",
        "total_line_search_its": "total_line_search_its_median",
        "accepted_steps": "accepted_steps_median",
        "lambda_last": "lambda_last_median",
        "omega_last": "omega_last_median",
        "final_rel": "final_rel_median",
        "global_dofs": "global_dofs_median",
        "elastic_assembly_time": "elastic_assembly_time_median",
        "deflation_orthogonalization_time": "deflation_orthogonalization_time_median",
        "deflation_pc_apply_time": "deflation_pc_apply_time_median",
        "deflation_projector_time": "deflation_projector_time_median",
    }
    out: dict[str, Any] = {}
    for source, target in fields.items():
        values = [_number(row.get(source)) for row in rows]
        numeric = [value for value in values if value is not None]
        out[target] = median(numeric) if numeric else None
    statuses = sorted({str(row.get("options_left", "missing")) for row in rows})
    out["options_left"] = ",".join(statuses)
    return out


def compare_targets(run_root: str | Path, target_root: str | Path, *, output: str | Path | None = None) -> Path:
    root = Path(run_root)
    targets = Path(target_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"No suite manifest found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = target_comparison_rows(manifest, targets)
    out = Path(output) if output is not None else root / "target-comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"suite": manifest.get("suite", {}), "comparison_level": "median", "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    return out


def target_comparison_rows(manifest: dict[str, Any], targets: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in collect_run_rows(manifest):
        key = (
            row["case"],
            row["profile"],
            row.get("refine_levels"),
            row.get("linear_rtol"),
            row.get("continuation_step_max"),
            int(row["ranks"]),
        )
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    suite_id = str(manifest.get("suite", {}).get("id", "") or "")
    for (case, profile, refine, rtol, step_max, ranks), group_rows in grouped.items():
        first_row = group_rows[0]
        completed = [row for row in group_rows if row.get("status") == "complete"]
        summary = summarize_run_group(completed) if completed else {}
        target_path, target = _load_target_for_case(targets, case, suite_id=suite_id)
        row = {
            "case": case,
            "profile": profile,
            "ranks": ranks,
            "refine_levels": refine,
            "linear_rtol": rtol,
            "continuation_step_max": step_max,
            "repeats": len(completed),
            "planned_repeats": len(group_rows),
            "options_left": summary.get("options_left") if summary else "missing",
            "native_linear_algorithm": first_row.get("native_linear_algorithm"),
            "pc_variant": first_row.get("pc_variant"),
            "requested_pc_variant": first_row.get("requested_pc_variant"),
            "pc_variant_fallback_reason": first_row.get("pc_variant_fallback_reason"),
            "pmg_p2_active_ranks": first_row.get("pmg_p2_active_ranks"),
            "pmg_p1_active_ranks": first_row.get("pmg_p1_active_ranks"),
        }
        rows.append(_compare_target_group(row, target_path, target, summary))
    return rows


def _load_target_for_case(targets: Path, case: str, *, suite_id: str) -> tuple[Path, dict[str, Any]]:
    direct = targets / f"{case}.json"
    if direct.exists():
        return direct, load_target(direct)

    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(targets.rglob(f"{case}.json")):
        try:
            payload = load_target(path)
        except json.JSONDecodeError:
            continue
        if str(payload.get("case", path.stem)) == case:
            candidates.append((path, payload))
    for path, payload in candidates:
        if suite_id and str(payload.get("suite", "")) == suite_id:
            return path, payload
    if candidates:
        return candidates[0]
    return direct, {}


def _compare_target_group(row: dict[str, Any], target_path: Path, target: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    metrics = _target_metrics_for_row(target, row)
    options_left_failure = _options_left_failure(row.get("options_left")) if summary else None
    if options_left_failure is not None:
        status = options_left_failure
        results: dict[str, Any] = {}
    elif not target:
        status = "missing_target"
        results = {}
    elif not summary:
        status = "missing_summary"
        results = {}
    elif not metrics:
        status = "no_metric_targets"
        results = {}
    else:
        results = {name: _compare_metric(name, spec, summary) for name, spec in metrics.items()}
        status = "pass" if all(item["status"] == "pass" for item in results.values()) else "fail"
    return {
        **row,
        "target": str(target_path),
        "status": status,
        "metrics": metrics,
        "results": results,
    }


def _options_left_failure(value: object) -> str | None:
    statuses = {item.strip().lower() for item in str(value or "").split(",") if item.strip()}
    if "check" in statuses:
        return "options_left_check"
    if not statuses or "missing" in statuses:
        return "options_left_missing"
    if "unknown" in statuses:
        return "options_left_unknown"
    if statuses == {"clean"}:
        return None
    return "options_left_unknown"


def _target_metrics_for_row(target: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(target.get("metrics", {}))

    rank_metrics = target.get("rank_metrics", {})
    if isinstance(rank_metrics, dict):
        rank_override = rank_metrics.get(str(row["ranks"]))
        if isinstance(rank_override, dict):
            metrics.update(target_metric_payload(rank_override))

    for group in target.get("groups", []):
        if isinstance(group, dict) and _target_group_matches(row, group):
            metrics.update(target_metric_payload(group))

    return metrics


def _target_group_matches(row: dict[str, Any], group: dict[str, Any]) -> bool:
    for key in ("profile", "ranks", "refine_levels", "linear_rtol", "continuation_step_max"):
        if key not in group:
            continue
        if row.get(key) != group.get(key):
            return False
    return True


def _compare_metric(name: str, spec: object, summary: dict[str, Any]) -> dict[str, Any]:
    actual_name = name
    actual = summary.get(actual_name)
    if actual is None and not name.endswith("_median"):
        actual_name = f"{name}_median"
        actual = summary.get(actual_name)
    if actual is None:
        return {"status": "missing_actual", "actual": None}
    if not isinstance(spec, dict):
        expected = float(spec)
        abs_tol = 0.0
        rel_tol = 0.0
    elif "max" in spec:
        maximum = float(spec["max"])
        value = float(actual)
        return {"status": "pass" if value <= maximum else "fail", "actual": value, "actual_metric": actual_name, "max": maximum}
    else:
        expected_raw = spec.get("expected", spec.get("value"))
        if expected_raw is None:
            return {"status": "invalid_target", "actual": actual}
        expected = float(expected_raw)
        abs_tol = float(spec.get("abs_tol", 0.0))
        rel_tol = float(spec.get("rel_tol", 0.0))
    value = float(actual)
    allowed = max(abs_tol, abs(expected) * rel_tol)
    delta = abs(value - expected)
    return {
        "status": "pass" if delta <= allowed else "fail",
        "actual": value,
        "actual_metric": actual_name,
        "expected": expected,
        "abs_delta": delta,
        "allowed_delta": allowed,
    }


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
