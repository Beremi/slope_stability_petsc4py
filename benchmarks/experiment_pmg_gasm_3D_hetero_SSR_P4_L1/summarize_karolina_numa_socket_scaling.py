#!/usr/bin/env python3
"""Summarize and plot Karolina NUMA-coalesced PMG socket-scaling runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = ROOT / "artifacts/experiments/pmg_numa_coalesced_karolina_socket_scaling_p4_l1_omega7"
TARGET_OMEGA = 7.0e6

CORE_METRICS = {
    "wall_s": "Wall time",
    "runtime_s": "Driver runtime",
    "linear_solve_total_s": "Linear solve total",
    "linear_pc_total_s": "Linear PC total",
    "pc_setup_s": "PC setup total",
    "pc_apply_s": "PC apply total",
    "orthogonalization_s": "Orthogonalization total",
}

PMG_SETUP_METRICS = {
    "manualmg_setup_destroy_time_total_s": "Destroy/reuse cleanup",
    "manualmg_setup_galerkin_ptap_time_total_s": "Galerkin PtAP",
    "manualmg_setup_coarse_operator_time_total_s": "Coarse operator",
    "manualmg_setup_near_nullspace_time_total_s": "Near nullspace",
    "manualmg_setup_smoother_time_total_s": "All smoothers",
    "manualmg_setup_fine_smoother_time_total_s": "Fine smoother",
    "manualmg_setup_mid_smoother_time_total_s": "Mid smoother",
    "manualmg_setup_other_smoother_time_total_s": "Other smoother",
    "manualmg_setup_coarse_ksp_time_total_s": "Coarse KSP",
    "manualmg_setup_work_vector_time_total_s": "Work vectors",
}

PMG_APPLY_METRICS = {
    "manualmg_fine_pre_smoother_time_total_s": "Fine pre-smoother",
    "manualmg_fine_post_smoother_time_total_s": "Fine post-smoother",
    "manualmg_mid_pre_smoother_time_total_s": "Mid pre-smoother",
    "manualmg_mid_post_smoother_time_total_s": "Mid post-smoother",
    "manualmg_restrict_fine_to_mid_time_total_s": "Restrict fine to mid",
    "manualmg_restrict_mid_to_coarse_time_total_s": "Restrict mid to coarse",
    "manualmg_prolong_coarse_to_mid_time_total_s": "Prolong coarse to mid",
    "manualmg_prolong_mid_to_fine_time_total_s": "Prolong mid to fine",
    "manualmg_fine_residual_time_total_s": "Fine residual",
    "manualmg_mid_residual_time_total_s": "Mid residual",
    "manualmg_vector_sum_time_total_s": "Vector sums",
    "manualmg_coarse_hypre_time_total_s": "Coarse solve",
}

ALL_TIMED_METRICS = {**CORE_METRICS, **PMG_SETUP_METRICS, **PMG_APPLY_METRICS}

_NEWTON_RE = re.compile(
    r"^\s*N\d+(?:\s+conv)?\s+\|.*?\|\s+lin=(?P<lin>\d+)\s+\|\s+"
    r"solve=(?P<solve>[0-9.eE+-]+)s\s+\|\s+pc=(?P<pc>[0-9.eE+-]+)s\s+\|\s+"
    r"orth=(?P<orth>[0-9.eE+-]+)s"
)
_INIT_RE = re.compile(
    r"^\[init\].*?omega=\[[^,\]]+,\s*(?P<omega>[0-9.eE+-]+)\].*?"
    r"\|\s+lin=(?P<lin>\d+)\s+\|\s+t=(?P<time>[0-9.eE+-]+)s"
)
_STEP_OK_RE = re.compile(
    r"^\[step\s+(?P<step>\d+)\s+ok\].*?omega=(?P<omega>[0-9.eE+-]+).*?"
    r"\|\s+lin=(?P<lin>\d+)\s+\|\s+t=(?P<time>[0-9.eE+-]+)s"
)
_STEP_TRY_RE = re.compile(r"^\[step\s+(?P<step>\d+)\s+try\s+\d+\]\s+omega=(?P<omega>[0-9.eE+-]+)")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_float(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _read_int(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def _final_progress(run_out: Path) -> dict[str, Any]:
    npz_path = run_out / "data/petsc_run.npz"
    if not npz_path.exists():
        return {}
    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception:
        return {}
    progress: dict[str, Any] = {}
    for source_key, target_key in (
        ("omega_hist", "omega"),
        ("lambda_hist", "lambda"),
        ("load_factor_hist", "lambda"),
    ):
        if source_key not in data:
            continue
        values = np.asarray(data[source_key])
        if values.ndim == 0 or not values.size:
            continue
        progress[f"last_{target_key}"] = float(values[-1])
        progress[f"count_{target_key}"] = int(values.size)
    return progress


def _stdout_progress(run_dir: Path) -> dict[str, Any]:
    log_path = run_dir / "stdout.log"
    if not log_path.exists():
        return {}

    progress: dict[str, Any] = {
        "stdout_linear_iterations": 0,
        "stdout_linear_solve_total_s": 0.0,
        "stdout_linear_pc_total_s": 0.0,
        "stdout_orthogonalization_s": 0.0,
    }
    accepted_steps = 0
    init_done = False
    last_accepted_omega: float | None = None
    last_attempt_omega: float | None = None

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := _NEWTON_RE.match(line):
            progress["stdout_linear_iterations"] += int(match.group("lin"))
            progress["stdout_linear_solve_total_s"] += float(match.group("solve"))
            progress["stdout_linear_pc_total_s"] += float(match.group("pc"))
            progress["stdout_orthogonalization_s"] += float(match.group("orth"))
            continue
        if match := _INIT_RE.match(line):
            init_done = True
            last_accepted_omega = float(match.group("omega"))
            continue
        if match := _STEP_OK_RE.match(line):
            accepted_steps += 1
            last_accepted_omega = float(match.group("omega"))
            continue
        if match := _STEP_TRY_RE.match(line):
            last_attempt_omega = float(match.group("omega"))

    total_lin = int(progress["stdout_linear_iterations"])
    if total_lin <= 0 and last_accepted_omega is None and last_attempt_omega is None:
        return {}
    progress["linear_iterations"] = total_lin or None
    if last_accepted_omega is not None:
        progress["last_omega"] = last_accepted_omega
    elif last_attempt_omega is not None:
        progress["last_omega"] = last_attempt_omega
    if last_attempt_omega is not None:
        progress["last_attempt_omega"] = last_attempt_omega
    if init_done:
        progress["count_omega"] = 2 + accepted_steps
    elif accepted_steps:
        progress["count_omega"] = accepted_steps
    return progress


def _linear_iteration_count(linear: dict[str, Any]) -> int | None:
    total = int(linear.get("init_linear_iterations") or 0) + int(linear.get("attempt_linear_iterations_total") or 0)
    return total or None


def _sum_present(linear: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    values = [linear.get(key) for key in keys]
    if all(value is None for value in values):
        return None
    return float(sum(float(value or 0.0) for value in values))


def _row_from_run_dir(run_dir: Path) -> dict[str, Any]:
    run_out = run_dir / "run"
    payload = _read_json(run_out / "data/run_info.json")
    info = payload.get("run_info") or payload
    params = payload.get("params") or {}
    timings = payload.get("timings") or {}
    linear = timings.get("linear") or {}
    metadata = _read_json(run_dir / "job_metadata.json")
    progress = _final_progress(run_out)
    stdout_progress = _stdout_progress(run_dir)
    progress = {**stdout_progress, **progress}
    exit_code = _read_int(run_dir / "exit_code.txt")
    omega_last = progress.get("last_omega")
    completed = exit_code == 0 and omega_last is not None and float(omega_last) >= TARGET_OMEGA

    nodes = metadata.get("nodes") or info.get("numa_layout_node_count")
    ranks = metadata.get("total_ranks") or info.get("mpi_size")
    sockets = (
        linear.get("manualmg_smoother_gasm_total_subdomains")
        or info.get("numa_layout_total_domains")
        or (int(nodes) * int(metadata.get("numa_domains_per_node")) if nodes and metadata.get("numa_domains_per_node") else None)
    )

    row: dict[str, Any] = {
        "case": metadata.get("case_name") or run_dir.parent.name,
        "job_id": run_dir.name,
        "nodes": nodes,
        "ranks": ranks,
        "sockets": sockets,
        "ranks_per_socket": linear.get("manualmg_smoother_gasm_ranks_per_subdomain")
        or info.get("numa_layout_ranks_per_numa"),
        "exit_code": exit_code,
        "completed": completed,
        "wall_s": _read_float(run_dir / "wall_seconds.txt") or info.get("runtime_seconds"),
        "runtime_s": info.get("runtime_seconds"),
        "omega_last": omega_last,
        "omega_attempt_last": progress.get("last_attempt_omega"),
        "lambda_last": progress.get("last_lambda"),
        "load_steps": info.get("step_count") or progress.get("count_omega"),
        "linear_iterations": _linear_iteration_count(linear) or progress.get("linear_iterations"),
        "setup_count": linear.get("manualmg_setup_count") or linear.get("preconditioner_rebuild_count"),
        "apply_count": linear.get("manualmg_apply_count"),
        "path": str(run_dir),
    }
    row["linear_solve_total_s"] = _sum_present(
        linear,
        ("init_linear_solve_time", "attempt_linear_solve_time_total"),
    ) or progress.get("stdout_linear_solve_total_s")
    row["linear_pc_total_s"] = _sum_present(
        linear,
        ("init_linear_preconditioner_time", "attempt_linear_preconditioner_time_total"),
    ) or progress.get("stdout_linear_pc_total_s")
    row["orthogonalization_s"] = _sum_present(
        linear,
        ("init_linear_orthogonalization_time", "attempt_linear_orthogonalization_time_total"),
    ) or progress.get("stdout_orthogonalization_s")
    row["pc_setup_s"] = linear.get("preconditioner_setup_time_total")
    row["pc_apply_s"] = linear.get("preconditioner_apply_time_total")
    for key in {**PMG_SETUP_METRICS, **PMG_APPLY_METRICS}:
        row[key] = linear.get(key)
    return row


def _discover_attempts(out_root: Path) -> list[dict[str, Any]]:
    run_dirs = sorted(path.parent for path in (out_root / "runs").glob("*/*/job_metadata.json"))
    return [_row_from_run_dir(path) for path in run_dirs]


def _best_rows(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in attempts:
        grouped.setdefault(str(row["case"]), []).append(row)
    best: list[dict[str, Any]] = []
    for rows in grouped.values():
        rows.sort(key=lambda row: (bool(row.get("completed")), int(row.get("job_id") or 0)))
        best.append(rows[-1])
    return sorted(best, key=lambda row: int(row.get("sockets") or 0))


def _write_tsv(rows: list[dict[str, Any]], out_path: Path, *, include_all_metrics: bool = True) -> None:
    base_columns = [
        "case",
        "job_id",
        "nodes",
        "ranks",
        "sockets",
        "ranks_per_socket",
        "exit_code",
        "completed",
        "wall_s",
        "runtime_s",
        "omega_last",
        "omega_attempt_last",
        "lambda_last",
        "load_steps",
        "linear_iterations",
        "setup_count",
        "apply_count",
    ]
    metric_columns = list(ALL_TIMED_METRICS) if include_all_metrics else []
    columns = base_columns + [key for key in metric_columns if key not in base_columns] + ["path"]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(_fmt(row.get(column)) for column in columns))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown(rows: list[dict[str, Any]], out_path: Path) -> None:
    columns = [
        ("case", "case"),
        ("job_id", "job"),
        ("nodes", "nodes"),
        ("ranks", "ranks"),
        ("sockets", "sockets"),
        ("exit_code", "exit"),
        ("completed", "done"),
        ("wall_s", "wall s"),
        ("omega_last", "omega"),
        ("omega_attempt_last", "try omega"),
        ("load_steps", "steps"),
        ("linear_iterations", "lin it"),
        ("pc_setup_s", "pc setup s"),
        ("pc_apply_s", "pc apply s"),
        ("manualmg_setup_galerkin_ptap_time_total_s", "PtAP s"),
        ("manualmg_setup_smoother_time_total_s", "smoother setup s"),
        ("manualmg_setup_coarse_ksp_time_total_s", "coarse setup s"),
        ("manualmg_coarse_hypre_time_total_s", "coarse solve s"),
    ]
    lines = ["| " + " | ".join(label for _, label in columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key, _ in columns) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _valid_xy(rows: list[dict[str, Any]], metric: str) -> tuple[list[int], list[float]]:
    pairs = []
    for row in rows:
        x = row.get("sockets")
        y = row.get(metric)
        if x is None or y is None:
            continue
        y = float(y)
        if y <= 0.0 or not math.isfinite(y):
            continue
        pairs.append((int(x), y))
    pairs.sort()
    return [x for x, _ in pairs], [y for _, y in pairs]


def _plot_metric_group(rows: list[dict[str, Any]], metrics: dict[str, str], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.88)
    plotted = 0
    for metric, label in metrics.items():
        xs, ys = _valid_xy(rows, metric)
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linewidth=2.0, label=label)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which="both", alpha=0.28)
    ax.set_xlabel("Logical NUMA/GASM domains")
    ax.set_ylabel("seconds")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=180)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def _plot_individual_metrics(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, label in ALL_TIMED_METRICS.items():
        xs, ys = _valid_xy(rows, metric)
        if not xs:
            continue
        fig, ax = plt.subplots(figsize=(7.8, 5.2))
        fig.subplots_adjust(left=0.13, right=0.97, bottom=0.16, top=0.88)
        ax.plot(xs, ys, marker="o", linewidth=2.5, color="#1f77b4")
        if xs and 1 in xs:
            base = ys[xs.index(1)]
            ideal_xs = sorted(xs)
            ideal_ys = [base / x for x in ideal_xs]
            ax.plot(ideal_xs, ideal_ys, linestyle="--", color="#666666", linewidth=1.4, label="ideal 1/x from 1 socket")
            ax.legend(loc="best", fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 4, 8, 16])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, which="both", alpha=0.28)
        ax.set_xlabel("Logical NUMA/GASM domains")
        ax.set_ylabel("seconds")
        ax.set_title(label)
        safe_name = metric.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"{safe_name}.png", dpi=180)
        fig.savefig(out_dir / f"{safe_name}.svg")
        plt.close(fig)


def _write_speedup(rows: list[dict[str, Any]], out_path: Path) -> None:
    completed = [row for row in rows if row.get("completed")]
    by_socket = {int(row["sockets"]): row for row in completed if row.get("sockets")}
    baseline = by_socket.get(1)
    columns = ["metric", "sockets", "seconds", "speedup_vs_1_socket", "parallel_efficiency"]
    lines = ["\t".join(columns)]
    if baseline is not None:
        for metric in ALL_TIMED_METRICS:
            base_value = baseline.get(metric)
            if base_value is None or float(base_value) <= 0.0:
                continue
            for sockets in sorted(by_socket):
                value = by_socket[sockets].get(metric)
                if value is None or float(value) <= 0.0:
                    continue
                speedup = float(base_value) / float(value)
                efficiency = speedup / float(sockets)
                lines.append(
                    "\t".join(
                        (
                            metric,
                            str(sockets),
                            _fmt(float(value)),
                            _fmt(speedup),
                            _fmt(efficiency),
                        )
                    )
                )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    attempts = _discover_attempts(out_root)
    rows = _best_rows(attempts)
    out_root.mkdir(parents=True, exist_ok=True)
    _write_tsv(attempts, out_root / "attempts.tsv")
    _write_tsv(rows, out_root / "summary.tsv")
    _write_markdown(rows, out_root / "summary.md")
    _write_speedup(rows, out_root / "speedup.tsv")
    _plot_metric_group(rows, CORE_METRICS, out_root / "scaling_core_timing_loglog.png", "NUMA PMG Socket Scaling: Core Timings")
    _plot_metric_group(
        rows,
        PMG_SETUP_METRICS,
        out_root / "scaling_pmg_setup_parts_loglog.png",
        "NUMA PMG Socket Scaling: Setup Parts",
    )
    _plot_metric_group(
        rows,
        PMG_APPLY_METRICS,
        out_root / "scaling_pmg_apply_parts_loglog.png",
        "NUMA PMG Socket Scaling: V-cycle Apply Parts",
    )
    _plot_individual_metrics(rows, out_root / "per_metric_plots")
    print(f"Wrote {len(rows)} selected row(s), {len(attempts)} attempt row(s) to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
