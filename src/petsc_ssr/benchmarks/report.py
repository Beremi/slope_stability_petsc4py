"""Benchmark report rendering helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from petsc_ssr.benchmarks.compare import collect_run_rows, summarize_run_group
from petsc_ssr.benchmarks.logs import petsc_log_top_events

__all__ = ["scaling_rows", "write_report"]


def write_report(run_root: str | Path, *, output: str | Path | None = None) -> Path:
    root = Path(run_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"No suite manifest found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = collect_run_rows(manifest)
    scale_rows = scaling_rows(rows)
    petsc_event_rows = _collect_petsc_event_rows(manifest)
    out = Path(output) if output is not None else root / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render_markdown_report(manifest, rows, scale_rows, petsc_event_rows), encoding="utf-8")
    _write_csv(out.with_suffix(".csv"), rows)
    _write_csv(out.with_suffix(".scaling.csv"), scale_rows)
    _write_csv(out.with_suffix(".petsc-events.csv"), petsc_event_rows)
    return out


def scaling_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "complete" or _number(row.get("wall_time")) is None:
            continue
        key = (
            row["case"],
            row["profile"],
            row.get("refine_levels"),
            row.get("linear_rtol"),
            row.get("continuation_step_max"),
            int(row["ranks"]),
        )
        grouped.setdefault(key, []).append(row)

    by_sweep: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for (case, profile, refine, rtol, step_max, ranks), group_rows in grouped.items():
        summary = summarize_run_group(group_rows)
        summary.update(
            {
                "case": case,
                "profile": profile,
                "refine_levels": refine,
                "linear_rtol": rtol,
                "continuation_step_max": step_max,
                "ranks": ranks,
                "repeats": len(group_rows),
            }
        )
        by_sweep.setdefault((case, profile, refine, rtol, step_max), []).append(summary)

    out: list[dict[str, Any]] = []
    for _sweep, rank_rows in sorted(by_sweep.items(), key=lambda item: tuple("" if value is None else str(value) for value in item[0])):
        rank_rows.sort(key=lambda row: int(row["ranks"]))
        baseline = rank_rows[0]
        base_ranks = int(baseline["ranks"])
        base_wall = _number(baseline.get("wall_time_median"))
        for row in rank_rows:
            wall = _number(row.get("wall_time_median"))
            ranks = int(row["ranks"])
            if base_wall is not None and wall is not None and wall > 0.0:
                speedup = base_wall / wall
                efficiency = speedup / (ranks / base_ranks)
            else:
                speedup = None
                efficiency = None
            row["baseline_ranks"] = base_ranks
            row["speedup"] = speedup
            row["parallel_efficiency"] = efficiency
            out.append(row)
    return out


def _collect_petsc_event_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in manifest.get("runs", []):
        for rank, event in enumerate(petsc_log_top_events(run["output_dir"], limit=10), start=1):
            rows.append(
                {
                    "case": run["case"],
                    "profile": run["profile"],
                    "ranks": run["ranks"],
                    "repeat": run["repeat"],
                    "event_rank": rank,
                    "event": event["event"],
                    "count": event["count"],
                    "time_s": event["time_s"],
                    "output_dir": run["output_dir"],
                }
            )
    return rows


def _render_markdown_report(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    scale_rows: list[dict[str, Any]],
    petsc_event_rows: list[dict[str, Any]],
) -> str:
    suite = manifest["suite"]
    lines = [
        f"# {suite['title']}",
        "",
        f"- Suite: `{suite['id']}`",
        f"- Runs: {len(rows)}",
        f"- Ranks: {suite.get('ranks', [])}",
        "",
        "| case | profile | ranks | resource | refine | linear rtol | step max | repeat | status | options-left | lambda | omega | wall | newton | linear | native linear | pc | pmg p2 | pmg p1 | output |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {case} | {profile} | {ranks} | {resource} | {refine_levels} | {linear_rtol} | {continuation_step_max} | {repeat} | {status} | {options_left} | {lambda_last} | {omega_last} | {wall_time} | {total_newton_its} | {total_linear_its} | {native_linear_algorithm} | {pc_variant} | {pmg_p2_active_ranks} | {pmg_p1_active_ranks} | {output_dir} |".format(
                **_display_row(row)
            )
        )
    lines.append("")
    lines.extend(_render_scaling_section(scale_rows))
    lines.extend(_render_iteration_section(scale_rows))
    lines.extend(_render_numerical_section(scale_rows))
    lines.extend(_render_petsc_event_section(petsc_event_rows))
    lines.extend(_render_artifact_section(manifest))
    lines.append("Options-left checks are reported when run artifacts exist.")
    return "\n".join(lines) + "\n"


def _render_scaling_section(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["", "## Scaling Summary", ""]
    if not rows:
        lines.extend(["No completed runs with `wall_time` are available yet.", ""])
        return lines
    lines.extend(
        [
            "| case | profile | ranks | repeats | wall median | speedup | efficiency | options-left |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {case} | {profile} | {ranks} | {repeats} | {wall_time_median} | {speedup} | {parallel_efficiency} | {options_left} |".format(
                **_display_row(row)
            )
        )
    lines.append("")
    return lines


def _render_iteration_section(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## Iteration Summary", ""]
    if not rows:
        lines.extend(["No completed runs are available yet.", ""])
        return lines
    lines.extend(
        [
            "| case | ranks | accepted steps | newton median | linear median | line-search median |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {case} | {ranks} | {accepted_steps_median} | {total_newton_its_median} | {total_linear_its_median} | {total_line_search_its_median} |".format(
                **_display_row(row)
            )
        )
    lines.append("")
    return lines


def _render_numerical_section(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## Numerical Summary", ""]
    if not rows:
        lines.extend(["No completed runs are available yet.", ""])
        return lines
    lines.extend(
        [
            "| case | ranks | lambda median | omega median | final rel median | global dofs |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {case} | {ranks} | {lambda_last_median} | {omega_last_median} | {final_rel_median} | {global_dofs_median} |".format(
                **_display_row(row)
            )
        )
    lines.append("")
    return lines


def _render_petsc_event_section(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## PETSc Log Events", ""]
    if not rows:
        lines.extend(["No completed runs with `logs/petsc_log.txt` are available yet.", ""])
        return lines
    lines.extend(
        [
            "| case | ranks | repeat | rank | event | count | time |",
            "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {case} | {ranks} | {repeat} | {event_rank} | {event} | {count} | {time_s} |".format(
                **_display_row(row)
            )
        )
    lines.append("")
    return lines


def _render_artifact_section(manifest: dict[str, Any]) -> list[str]:
    lines = ["## Artifact Paths", ""]
    runs = list(manifest.get("runs", []))
    if not runs:
        lines.extend(["No runs are present in the suite manifest.", ""])
        return lines
    lines.extend(
        [
            "| case | profile | ranks | resource | launcher | repeat | output | command | resolved manifest | resolved options | summary | PETSc log | options-left |",
            "| --- | --- | ---: | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for run in runs:
        artifacts = dict(run.get("artifacts", {}))
        row = {
            "case": run.get("case"),
            "profile": run.get("profile"),
            "ranks": run.get("ranks"),
            "resource": run.get("resource"),
            "launcher": " ".join(str(token) for token in run.get("launcher", []) or []),
            "repeat": run.get("repeat"),
            "output": run.get("output_dir"),
            "command": artifacts.get("command_json"),
            "resolved_manifest": artifacts.get("resolved_run_manifest_json"),
            "resolved_options": artifacts.get("resolved_options_txt"),
            "summary": artifacts.get("summary_json"),
            "petsc_log": artifacts.get("petsc_log_txt"),
            "options_left": artifacts.get("options_left_txt"),
        }
        lines.append(
            "| {case} | {profile} | {ranks} | {resource} | {launcher} | {repeat} | {output} | {command} | {resolved_manifest} | {resolved_options} | {summary} | {petsc_log} | {options_left} |".format(
                **_display_row(row)
            )
        )
    lines.append("")
    return lines


def _display_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: "" if value is None else _format_value(value) for key, value in row.items()}


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
