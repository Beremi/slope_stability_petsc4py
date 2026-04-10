#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = ROOT / "artifacts" / "comparisons" / "p4_l1_pmg_shell_fix_20260410"
REPORT_PATH = ROOT / "benchmarks" / "slope_stability_3D_hetero_SSR_default" / "archive" / "report_p4_l1_pmg_shell_fix_20260410.md"

RUN_SPECS = (
    {
        "name": "original_baseline",
        "label": "Original baseline artifact",
        "run_dir": OUT_ROOT / "original_baseline",
        "kind": "baseline",
    },
    {
        "name": "postfix_default",
        "label": "Post-fix source default",
        "run_dir": ROOT / "benchmarks" / "slope_stability_3D_hetero_SSR_default" / "artifacts" / "simulation",
        "kind": "default",
    },
    {
        "name": "postfix_armijo",
        "label": "Post-fix Armijo residual",
        "run_dir": OUT_ROOT / "armijo_run",
        "kind": "armijo",
    },
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rel(from_path: Path, to_path: Path) -> str:
    return os.path.relpath(to_path.resolve(), start=from_path.parent.resolve())


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [_coerce_row(dict(row)) for row in reader]


def _coerce_row(row: dict[str, str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in row.items():
        if value is None or value == "":
            out[key] = None
            continue
        text = str(value).strip()
        if text.lower() in {"true", "false"}:
            out[key] = text.lower() == "true"
            continue
        try:
            number = float(text)
        except ValueError:
            out[key] = text
            continue
        out[key] = int(number) if number.is_integer() else number
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: object) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except Exception:
        return math.nan


def _safe_int(value: object) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except Exception:
        return 0


def _fmt(value: object, digits: int = 3) -> str:
    number = _safe_float(value)
    if not np.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _fmt_pct(value: float) -> str:
    if not np.isfinite(float(value)):
        return "-"
    return f"{100.0 * float(value):+.1f}%"


def _fmt_pct_abs(value: float) -> str:
    if not np.isfinite(float(value)):
        return "-"
    return f"{100.0 * abs(float(value)):.1f}%"


def _load_run(spec: dict[str, object]) -> dict[str, object]:
    run_dir = Path(spec["run_dir"])
    history = _read_json(run_dir / "exports" / "continuation_history.json")
    run_info = _read_json(run_dir / "data" / "run_info.json")
    accepted_rows = _read_csv_rows(run_dir / "exports" / "accepted_continuation_steps.csv")
    attempt_rows = _read_csv_rows(run_dir / "exports" / "all_attempts_summary.csv")
    newton_rows = _read_csv_rows(run_dir / "exports" / "all_newton_iterations.csv")

    step_map = {int(row["accepted_step"]): row for row in accepted_rows if row.get("accepted_step") is not None}
    continuation_rows = [row for row in newton_rows if row.get("phase") == "continuation"]
    continuation_attempt_rows = [row for row in attempt_rows if row.get("phase") == "continuation"]

    lambda_hist = np.asarray(history.get("lambda_hist", []), dtype=np.float64)
    omega_hist = np.asarray(history.get("omega_hist", []), dtype=np.float64)
    u_hist = np.asarray(history.get("Umax_hist", []), dtype=np.float64)
    runtime = float(run_info["run_info"]["runtime_seconds"])
    timings_linear = dict(run_info.get("timings", {}).get("linear", {}))

    summary = {
        "name": str(spec["name"]),
        "label": str(spec["label"]),
        "kind": str(spec["kind"]),
        "run_dir": str(run_dir),
        "runtime_s": runtime,
        "init_linear_iterations": int(timings_linear.get("init_linear_iterations", 0)),
        "init_linear_solve_time_s": float(timings_linear.get("init_linear_solve_time", 0.0)),
        "accepted_continuation_steps": int(len(accepted_rows)),
        "continuation_newton_total": int(sum(_safe_int(row.get("step_newton_iterations")) for row in accepted_rows)),
        "continuation_linear_total": int(sum(_safe_int(row.get("step_linear_iterations")) for row in accepted_rows)),
        "continuation_line_search_total": int(sum(_safe_int(row.get("line_search_iterations")) for row in accepted_rows)),
        "continuation_line_search_fallback_count": int(
            sum(int(bool(row.get("line_search_fallback_used", False))) for row in continuation_rows)
        ),
        "continuation_deflation_basis_dim_solve_max": int(
            max((_safe_int(row.get("deflation_basis_dim_solve")) for row in continuation_rows), default=0)
        ),
        "continuation_deflation_basis_dim_end_last": int(
            _safe_int(continuation_rows[-1].get("deflation_basis_dim_end")) if continuation_rows else 0
        ),
        "final_lambda": float(lambda_hist[-1]) if lambda_hist.size else math.nan,
        "final_omega": float(omega_hist[-1]) if omega_hist.size else math.nan,
        "final_u_max": float(u_hist[-1]) if u_hist.size else math.nan,
        "step_map": step_map,
        "accepted_rows": accepted_rows,
        "attempt_rows": attempt_rows,
        "continuation_attempt_rows": continuation_attempt_rows,
        "newton_rows": newton_rows,
        "continuation_newton_rows": continuation_rows,
        "lambda_hist": lambda_hist,
        "omega_hist": omega_hist,
    }
    return summary


def _comparison_rows(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline = next(run for run in runs if run["kind"] == "baseline")
    baseline_runtime = float(baseline["runtime_s"])
    rows = []
    for run in runs:
        rows.append(
            {
                "variant": run["label"],
                "runtime_s": float(run["runtime_s"]),
                "runtime_vs_baseline_pct": (
                    (float(run["runtime_s"]) / baseline_runtime) - 1.0 if baseline_runtime > 0.0 else math.nan
                ),
                "init_linear_iterations": int(run["init_linear_iterations"]),
                "continuation_newton_total": int(run["continuation_newton_total"]),
                "continuation_linear_total": int(run["continuation_linear_total"]),
                "continuation_line_search_total": int(run["continuation_line_search_total"]),
                "continuation_line_search_fallback_count": int(run["continuation_line_search_fallback_count"]),
                "continuation_deflation_basis_dim_solve_max": int(run["continuation_deflation_basis_dim_solve_max"]),
                "final_lambda": float(run["final_lambda"]),
                "final_omega": float(run["final_omega"]),
                "accepted_continuation_steps": int(run["accepted_continuation_steps"]),
            }
        )
    return rows


def _accepted_step_comparison(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    all_steps = sorted({int(step) for run in runs for step in run["step_map"]})
    rows: list[dict[str, object]] = []
    for step in all_steps:
        row: dict[str, object] = {"step": int(step)}
        for run in runs:
            prefix = str(run["name"])
            step_row = run["step_map"].get(step, {})
            row[f"{prefix}_omega"] = step_row.get("omega_value")
            row[f"{prefix}_lambda"] = step_row.get("lambda_value")
            row[f"{prefix}_u_max"] = step_row.get("u_max")
            row[f"{prefix}_newton"] = step_row.get("step_newton_iterations")
            row[f"{prefix}_linear"] = step_row.get("step_linear_iterations")
            row[f"{prefix}_wall_s"] = step_row.get("step_wall_time_s")
        rows.append(row)
    return rows


def _plot_overlay(runs: list[dict[str, object]], *, y_key: str, ylabel: str, title: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(8.0, 5.0), dpi=180)
    for run in runs:
        x = np.asarray([_safe_int(row.get("accepted_step")) for row in run["accepted_rows"]], dtype=np.int64)
        y = np.asarray([_safe_float(row.get(y_key)) for row in run["accepted_rows"]], dtype=np.float64)
        if not x.size or not y.size:
            continue
        plt.plot(x, y, marker="o", linewidth=1.6, label=str(run["label"]))
    plt.xlabel("Accepted continuation step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_trajectory(runs: list[dict[str, object]], out_path: Path) -> None:
    fig = plt.figure(figsize=(8.0, 5.0), dpi=180)
    for run in runs:
        omega = np.asarray(run["omega_hist"], dtype=np.float64)
        lam = np.asarray(run["lambda_hist"], dtype=np.float64)
        if not omega.size or not lam.size:
            continue
        plt.plot(omega, lam, marker="o", linewidth=1.6, label=str(run["label"]))
    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$\lambda$")
    plt.title("Accepted continuation trajectory")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _write_summary_json(path: Path, *, runs: list[dict[str, object]], summary_rows: list[dict[str, object]], step_rows: list[dict[str, object]]) -> None:
    payload = {
        "runs": [
            {
                "name": run["name"],
                "label": run["label"],
                "kind": run["kind"],
                "run_dir": run["run_dir"],
                "runtime_s": run["runtime_s"],
                "init_linear_iterations": run["init_linear_iterations"],
                "continuation_newton_total": run["continuation_newton_total"],
                "continuation_linear_total": run["continuation_linear_total"],
                "continuation_line_search_total": run["continuation_line_search_total"],
                "continuation_line_search_fallback_count": run["continuation_line_search_fallback_count"],
                "continuation_deflation_basis_dim_solve_max": run["continuation_deflation_basis_dim_solve_max"],
                "final_lambda": run["final_lambda"],
                "final_omega": run["final_omega"],
                "accepted_continuation_steps": run["accepted_continuation_steps"],
            }
            for run in runs
        ],
        "summary_rows": summary_rows,
        "accepted_step_rows": step_rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _overall_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Variant | Runtime [s] | Vs baseline | Init lin | Cont. Newton | Cont. linear | LS total | Fallbacks | Max defl. basis | Final lambda | Final omega |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    _fmt(row["runtime_s"], 3),
                    _fmt_pct(float(row["runtime_vs_baseline_pct"])),
                    str(int(row["init_linear_iterations"])),
                    str(int(row["continuation_newton_total"])),
                    str(int(row["continuation_linear_total"])),
                    str(int(row["continuation_line_search_total"])),
                    str(int(row["continuation_line_search_fallback_count"])),
                    str(int(row["continuation_deflation_basis_dim_solve_max"])),
                    _fmt(row["final_lambda"], 6),
                    _fmt(row["final_omega"], 1),
                ]
            )
            + " |"
        )
    return lines


def _accepted_table(runs: list[dict[str, object]]) -> list[str]:
    baseline, fixed_default, armijo = runs
    lines = [
        "| Step | Omega [e6] | Lambda baseline | Lambda fixed | Lambda Armijo | Linear baseline | Linear fixed | Linear Armijo | Wall baseline [s] | Wall fixed [s] | Wall Armijo [s] |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    all_steps = sorted({int(step) for run in runs for step in run["step_map"]})
    for step in all_steps:
        b = baseline["step_map"].get(step, {})
        d = fixed_default["step_map"].get(step, {})
        a = armijo["step_map"].get(step, {})
        omega_value = _safe_float(d.get("omega_value", b.get("omega_value", a.get("omega_value")))) / 1.0e6
        lines.append(
            f"| {int(step)} | {_fmt(omega_value, 3)} | {_fmt(b.get('lambda_value'), 6)} | {_fmt(d.get('lambda_value'), 6)} | {_fmt(a.get('lambda_value'), 6)} | "
            f"{_safe_int(b.get('step_linear_iterations')) or '-'} | {_safe_int(d.get('step_linear_iterations')) or '-'} | {_safe_int(a.get('step_linear_iterations')) or '-'} | "
            f"{_fmt(b.get('step_wall_time_s'), 3)} | {_fmt(d.get('step_wall_time_s'), 3)} | {_fmt(a.get('step_wall_time_s'), 3)} |"
        )
    return lines


def _late_step_table(runs: list[dict[str, object]]) -> list[str]:
    baseline, fixed_default, armijo = runs
    lines = [
        "| Step | Metric | Baseline | Fixed default | Armijo |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for step in (8, 9):
        for metric, key, digits in (
            ("Newton", "step_newton_iterations", 0),
            ("Linear", "step_linear_iterations", 0),
            ("Wall [s]", "step_wall_time_s", 3),
        ):
            b = baseline["step_map"].get(step, {})
            d = fixed_default["step_map"].get(step, {})
            a = armijo["step_map"].get(step, {})
            if digits == 0:
                b_text = str(_safe_int(b.get(key))) if b else "-"
                d_text = str(_safe_int(d.get(key))) if d else "-"
                a_text = str(_safe_int(a.get(key))) if a else "-"
            else:
                b_text = _fmt(b.get(key), digits)
                d_text = _fmt(d.get(key), digits)
                a_text = _fmt(a.get(key), digits)
            lines.append(f"| {step} | {metric} | {b_text} | {d_text} | {a_text} |")
    return lines


def _diagnostics_table(runs: list[dict[str, object]]) -> list[str]:
    fixed_default = next(run for run in runs if run["kind"] == "default")
    armijo = next(run for run in runs if run["kind"] == "armijo")
    lines = [
        "| Variant | Continuation LS total | Fallback count | Max deflation basis | Final deflation basis |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for run in (fixed_default, armijo):
        final_basis = 0
        if run["continuation_newton_rows"]:
            final_basis = _safe_int(run["continuation_newton_rows"][-1].get("deflation_basis_dim_end"))
        lines.append(
            f"| {run['label']} | {int(run['continuation_line_search_total'])} | {int(run['continuation_line_search_fallback_count'])} | "
            f"{int(run['continuation_deflation_basis_dim_solve_max'])} | {int(final_basis)} |"
        )
    return lines


def _write_report(path: Path, *, runs: list[dict[str, object]], summary_rows: list[dict[str, object]], plot_paths: dict[str, Path]) -> None:
    baseline = next(run for run in runs if run["kind"] == "baseline")
    fixed_default = next(run for run in runs if run["kind"] == "default")
    armijo = next(run for run in runs if run["kind"] == "armijo")

    runtime_gain = 1.0 - (float(fixed_default["runtime_s"]) / float(baseline["runtime_s"]))
    runtime_gain_armijo = 1.0 - (float(armijo["runtime_s"]) / float(baseline["runtime_s"]))

    lines = [
        "# P4(L1) PMG-Shell Fix Continuation Report",
        "",
        "## Executive Summary",
        "",
        "- Scope: `P4(L1)` on `SSR_hetero_ada_L1.msh`, SSR indirect continuation, `mpi_ranks = 8`.",
        "- The original artifact compared here is the pre-fix benchmark-local bundle that was snapshotted before rerun.",
        "- The permanent PMG-shell fix is the parallel smoother switch to `chebyshev + jacobi` for shell hierarchies with level orders `(1, 2, 4)` or `(1, 1, 2)` at MPI size `> 1`.",
        f"- On this benchmark, the post-fix source-default run reduced full runtime from `{_fmt(baseline['runtime_s'], 3)} s` to `{_fmt(fixed_default['runtime_s'], 3)} s` ({_fmt_pct_abs(runtime_gain)} lower wall time).",
        f"- The Armijo residual variant finished on essentially the same continuation path in `{_fmt(armijo['runtime_s'], 3)} s` ({_fmt_pct_abs(runtime_gain_armijo)} lower wall time vs baseline), so it remains an experimental option rather than a new default.",
        "- Only `8`-rank runs are compared here. The earlier `32`-rank evidence is context for the fix, not a rerun in this report.",
        "",
        "## Configuration Summary",
        "",
        "- Baseline and post-fix default share the same benchmark case: `solver_type = PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE`, `pc_backend = pmg_shell`, `elem_type = P4`, `omega_max_stop = 6.7e6`.",
        "- The fixed default differs from the original artifact only in the PMG-shell smoother selection for the robust parallel shell case.",
        "- The Armijo variant keeps the same fixed PMG-shell path and overrides only:",
        "  - `newton.line_search = \"armijo_residual\"`",
        "  - `newton.armijo_max_ls = 10`",
        "  - `newton.armijo_rescale_trial_to_omega = true`",
        "  - `newton.armijo_fallback_to_alg5 = true`",
        "- In this implementation, the Armijo path is applied to the continuation Newton corrections; init remains on the existing path.",
        "",
        "## What Changed In PMG-Shell",
        "",
        "- Before the fix, the benchmark-local artifact used `richardson + sor` on the shell levels even for the parallel `P4` hierarchy with orders `(1, 2, 4)`.",
        "- The permanent driver change now detects that robust parallel shell case and switches those levels to `chebyshev + jacobi` while keeping the coarse HYPRE solve and the rest of the continuation stack unchanged.",
        "",
        "## Overall Metrics",
        "",
        *_overall_table(summary_rows),
        "",
        "## Accepted Continuation Steps",
        "",
        *_accepted_table(runs),
        "",
        "## Late-Step Focus",
        "",
        *_late_step_table(runs),
        "",
        "## Line Search And Deflation Diagnostics",
        "",
        *_diagnostics_table(runs),
        "",
        "## Comparison Plots",
        "",
        f"### Accepted trajectory overlay\n\n![Accepted trajectory overlay]({_rel(path, plot_paths['trajectory'])})",
        "",
        f"### Step Newton iteration overlay\n\n![Step Newton overlay]({_rel(path, plot_paths['newton'])})",
        "",
        f"### Step linear iteration overlay\n\n![Step linear overlay]({_rel(path, plot_paths['linear'])})",
        "",
        f"### Step wall-time overlay\n\n![Step wall overlay]({_rel(path, plot_paths['wall'])})",
        "",
        "## Recommendation",
        "",
        "- Keep the PMG-shell smoother correction permanent for parallel `P4(L1)` source-default runs.",
        "- Keep the rest of the source-default continuation path unchanged.",
        "- Keep `armijo_residual` as an optional debug and experiment mode. On this benchmark it matches the fixed default path closely, but it does not buy a material runtime or iteration reduction.",
        "",
        "## Artifact Locations",
        "",
        f"- Baseline snapshot: `{baseline['run_dir']}`",
        f"- Post-fix source-default benchmark-local artifact: `{fixed_default['run_dir']}`",
        f"- Post-fix Armijo run: `{armijo['run_dir']}`",
        f"- Comparison root: `{OUT_ROOT}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _ensure_dir(OUT_ROOT)
    plot_dir = _ensure_dir(OUT_ROOT / "plots")

    runs = [_load_run(spec) for spec in RUN_SPECS]
    summary_rows = _comparison_rows(runs)
    step_rows = _accepted_step_comparison(runs)

    plot_paths = {
        "trajectory": plot_dir / "trajectory_overlay.png",
        "newton": plot_dir / "step_newton_overlay.png",
        "linear": plot_dir / "step_linear_overlay.png",
        "wall": plot_dir / "step_wall_overlay.png",
    }
    _plot_trajectory(runs, plot_paths["trajectory"])
    _plot_overlay(runs, y_key="step_newton_iterations", ylabel="Newton iterations", title="Step Newton iterations", out_path=plot_paths["newton"])
    _plot_overlay(runs, y_key="step_linear_iterations", ylabel="Linear iterations", title="Step linear iterations", out_path=plot_paths["linear"])
    _plot_overlay(runs, y_key="step_wall_time_s", ylabel="Wall time [s]", title="Step wall time", out_path=plot_paths["wall"])

    _write_csv(OUT_ROOT / "overall_summary.csv", summary_rows)
    _write_csv(OUT_ROOT / "accepted_steps_comparison.csv", step_rows)
    _write_summary_json(OUT_ROOT / "comparison_summary.json", runs=runs, summary_rows=summary_rows, step_rows=step_rows)
    _write_report(REPORT_PATH, runs=runs, summary_rows=summary_rows, plot_paths=plot_paths)


if __name__ == "__main__":
    main()
