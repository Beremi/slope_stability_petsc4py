#!/usr/bin/env python3
"""Summarize and plot Karolina multi-node full-occupancy scaling runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = ROOT / "artifacts/experiments/pmg_gasm_karolina_multinode_full_occupancy_p4_l1_omega7"
DEFAULT_ONE_NODE_SUMMARY = ROOT / "artifacts/experiments/pmg_gasm_karolina_qexp_one_node_p4_l1_omega7/summary.tsv"
RANKS_PER_NODE = 128
TARGET_OMEGA = 7.0e6


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _read_float(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return float(path.read_text().strip())
    except ValueError:
        return None


def _read_int(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except ValueError:
        return None


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
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


def _linear_iteration_count(linear: dict[str, Any]) -> int | None:
    total = int(linear.get("init_linear_iterations") or 0) + int(linear.get("attempt_linear_iterations_total") or 0)
    return total or None


def _row_from_run_dir(run_dir: Path) -> dict[str, Any]:
    run_out = run_dir / "run"
    run_info_path = run_out / "data/run_info.json"
    payload = _read_json(run_info_path)
    info = payload.get("run_info") or payload
    timings = payload.get("timings") or {}
    linear = timings.get("linear") or {}
    metadata = _read_json(run_dir / "case_metadata.json")
    progress = _final_progress(run_out)
    exit_code = _read_int(run_dir / "exit_code.txt")
    omega_last = progress.get("last_omega")
    completed = exit_code == 0 and omega_last is not None and omega_last >= TARGET_OMEGA
    nodes = metadata.get("nodes")
    ranks = metadata.get("ranks") or info.get("mpi_size")
    if nodes is None and ranks:
        nodes = int(ranks) // RANKS_PER_NODE

    pc_setup_total = linear.get("preconditioner_setup_time_total")
    pc_apply_total = linear.get("preconditioner_apply_time_total")
    pc_setup_count = linear.get("manualmg_setup_count") or linear.get("preconditioner_rebuild_count")
    pc_apply_count = linear.get("manualmg_apply_count")

    return {
        "case": metadata.get("case_name") or run_dir.parent.name,
        "kind": metadata.get("kind") or "unknown",
        "nodes": nodes,
        "ranks": ranks,
        "sockets": metadata.get("sockets") or linear.get("manualmg_smoother_gasm_total_subdomains"),
        "ranks_per_socket": metadata.get("ranks_per_socket")
        or linear.get("manualmg_smoother_gasm_ranks_per_subdomain"),
        "exit_code": exit_code,
        "completed": completed,
        "wall_s": _read_float(run_dir / "wall_seconds.txt") or info.get("runtime_seconds"),
        "omega_last": omega_last,
        "lambda_last": progress.get("last_lambda"),
        "load_steps": info.get("step_count") or progress.get("count_omega"),
        "linear_iterations": _linear_iteration_count(linear),
        "pc_setup_s": pc_setup_total,
        "pc_apply_s": pc_apply_total,
        "setup_count": pc_setup_count,
        "setup_per_setup_s": pc_setup_total / pc_setup_count if pc_setup_total and pc_setup_count else None,
        "apply_per_apply_s": pc_apply_total / pc_apply_count if pc_apply_total and pc_apply_count else None,
        "path": str(run_dir),
        "source": "multinode",
    }


def _rows_from_one_node_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("case") not in {"baseline_128", "gasm_8x16"}:
                continue
            ranks = int(row["ranks"])
            wall_s = float(row["wall_s"])
            omega_last = float(row["omega_last"]) if row.get("omega_last") else None
            rows.append(
                {
                    "case": row["case"],
                    "kind": row["kind"],
                    "nodes": ranks // RANKS_PER_NODE,
                    "ranks": ranks,
                    "sockets": row.get("sockets") or None,
                    "ranks_per_socket": row.get("ranks_per_socket") or None,
                    "exit_code": int(row["exit_code"]) if row.get("exit_code") else None,
                    "completed": omega_last is not None and omega_last >= TARGET_OMEGA,
                    "wall_s": wall_s,
                    "omega_last": omega_last,
                    "lambda_last": float(row["lambda_last"]) if row.get("lambda_last") else None,
                    "load_steps": int(row["load_steps"]) if row.get("load_steps") else None,
                    "linear_iterations": int(row["linear_iterations"]) if row.get("linear_iterations") else None,
                    "pc_setup_s": float(row["pc_setup_s"]) if row.get("pc_setup_s") else None,
                    "pc_apply_s": float(row["pc_apply_s"]) if row.get("pc_apply_s") else None,
                    "setup_count": int(float(row["setup_count"])) if row.get("setup_count") else None,
                    "setup_per_setup_s": float(row["setup_per_setup_s"]) if row.get("setup_per_setup_s") else None,
                    "apply_per_apply_s": float(row["apply_per_apply_s"]) if row.get("apply_per_apply_s") else None,
                    "path": row.get("path"),
                    "source": "one_node_anchor",
                }
            )
    return rows


def _discover_rows(out_root: Path, one_node_summary: Path | None) -> list[dict[str, Any]]:
    run_dirs = sorted(path.parent for path in (out_root / "runs").glob("*/*/case_metadata.json"))
    rows = [_row_from_run_dir(path) for path in run_dirs]
    if one_node_summary is not None:
        rows.extend(_rows_from_one_node_summary(one_node_summary))
    return sorted(rows, key=lambda row: (int(row.get("nodes") or 0), str(row.get("kind")), str(row.get("case"))))


def _write_tsv(rows: list[dict[str, Any]], out_path: Path) -> None:
    columns = [
        "case",
        "kind",
        "nodes",
        "ranks",
        "sockets",
        "ranks_per_socket",
        "exit_code",
        "completed",
        "wall_s",
        "omega_last",
        "lambda_last",
        "load_steps",
        "linear_iterations",
        "pc_setup_s",
        "pc_apply_s",
        "setup_count",
        "setup_per_setup_s",
        "apply_per_apply_s",
        "source",
        "path",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(_fmt(row.get(column)) for column in columns))
    out_path.write_text("\n".join(lines) + "\n")


def _write_markdown(rows: list[dict[str, Any]], out_path: Path) -> None:
    columns = [
        ("case", "case"),
        ("nodes", "nodes"),
        ("ranks", "ranks"),
        ("sockets", "sockets"),
        ("exit_code", "exit"),
        ("completed", "done"),
        ("wall_s", "wall s"),
        ("omega_last", "omega"),
        ("linear_iterations", "lin it"),
        ("pc_setup_s", "setup s"),
        ("pc_apply_s", "apply s"),
        ("source", "source"),
    ]
    lines = ["| " + " | ".join(label for _, label in columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key, _ in columns) + " |")
    out_path.write_text("\n".join(lines) + "\n")


def _plot(rows: list[dict[str, Any]], out_root: Path) -> None:
    completed = [row for row in rows if row.get("completed") and row.get("wall_s") and row.get("nodes")]
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for row in completed:
        by_kind.setdefault(str(row["kind"]), []).append(row)
    if not by_kind:
        return
    for group in by_kind.values():
        group.sort(key=lambda row: int(row["nodes"]))

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.18, top=0.88)
    colors = {"baseline": "#1f77b4", "gasm": "#d95f02"}
    labels = {"baseline": "Baseline PMG shell", "gasm": "PMG + GASM smoother"}

    for kind in ("baseline", "gasm"):
        group = by_kind.get(kind, [])
        if not group:
            continue
        xs = [int(row["nodes"]) for row in group]
        ys = [float(row["wall_s"]) for row in group]
        ax.plot(xs, ys, marker="o" if kind == "baseline" else "s", linewidth=2.6, color=colors[kind], label=labels[kind])
        if 1 in xs:
            y1 = ys[xs.index(1)]
            ideal_xs = sorted(xs)
            ideal_ys = [y1 / node for node in ideal_xs]
            ax.plot(
                ideal_xs,
                ideal_ys,
                linestyle="--" if kind == "baseline" else ":",
                linewidth=1.8,
                color=colors[kind],
                alpha=0.5,
                label=f"Ideal from {labels[kind]} 1 node",
            )
        for row in group:
            label = f"{float(row['wall_s']):.0f}s"
            if kind == "gasm" and row.get("sockets"):
                label = f"{int(float(row['sockets']))}x{int(float(row['ranks_per_socket']))}\n{label}"
            ax.annotate(
                label,
                (int(row["nodes"]), float(row["wall_s"])),
                textcoords="offset points",
                xytext=(0, 9 if kind == "baseline" else -36),
                ha="center",
                color=colors[kind],
                fontsize=9.5,
            )

    incomplete = [row for row in rows if not row.get("completed") and row.get("wall_s") and row.get("nodes")]
    for row in incomplete:
        color = colors.get(str(row.get("kind")), "#555555")
        ax.scatter([int(row["nodes"])], [float(row["wall_s"])], marker="x", s=80, color=color, linewidth=2.0)
        ax.annotate(
            f"exit {row.get('exit_code')}\n{float(row['wall_s']):.0f}s",
            (int(row["nodes"]), float(row["wall_s"])),
            textcoords="offset points",
            xytext=(0, -35),
            ha="center",
            color=color,
            fontsize=9,
        )

    nodes = sorted({int(row["nodes"]) for row in rows if row.get("nodes")})
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xticks(nodes)
    ax.set_xticklabels([str(node) for node in nodes])
    ax.set_xlabel("Nodes, full occupancy (128 MPI ranks/node)")
    ax.set_ylabel("Wall time to omega = 7e6 [s]")
    ax.set_title("P4(L1) Hetero 3D SSR Full-Occupancy Scaling")
    ax.grid(True, which="major", color="#d5d5d5", linewidth=0.85)
    ax.grid(True, which="minor", color="#eeeeee", linewidth=0.5)
    ax.legend(loc="best", frameon=True, framealpha=0.95)
    fig.text(
        0.11,
        0.055,
        "Karolina full-occupancy runs. Multi-node jobs use a 10 min limit; 1-node anchor is from the previous run.",
        fontsize=9.2,
        color="#555555",
    )
    fig.savefig(out_root / "multinode_full_occupancy_scaling_loglog.png", dpi=180)
    fig.savefig(out_root / "multinode_full_occupancy_scaling_loglog.svg")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--one-node-summary", type=Path, default=DEFAULT_ONE_NODE_SUMMARY)
    parser.add_argument("--no-one-node-anchor", action="store_true")
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    one_node_summary = None if args.no_one_node_anchor else args.one_node_summary.resolve()
    rows = _discover_rows(out_root, one_node_summary)
    _write_tsv(rows, out_root / "summary.tsv")
    _write_markdown(rows, out_root / "summary.md")
    _plot(rows, out_root)
    print(f"Wrote {len(rows)} row(s) to {out_root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
