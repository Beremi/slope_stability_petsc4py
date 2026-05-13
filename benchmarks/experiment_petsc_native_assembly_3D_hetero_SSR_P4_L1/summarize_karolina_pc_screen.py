#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _sources(info: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = [info]
    while stack:
        current = stack.pop(0)
        sources.append(current)
        for value in current.values():
            if isinstance(value, dict):
                stack.append(value)
    return sources


def _pick(info: dict[str, Any], *keys: str) -> Any:
    for source in _sources(info):
        for key in keys:
            if key in source:
                return source[key]
    return None


def _sum_pick(info: dict[str, Any], *keys: str) -> Any:
    values = [_pick(info, key) for key in keys]
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return None
    total = sum(numeric)
    if all(float(value).is_integer() for value in numeric):
        return int(total)
    return total


def _history_counts(run_dir: Path) -> tuple[int | None, int | None]:
    npz_path = run_dir / "data" / "petsc_run.npz"
    if not npz_path.exists():
        return None, None
    import numpy as np

    with np.load(npz_path, allow_pickle=True) as npz:
        accepted = np.asarray(npz["accepted_lambdas"]) if "accepted_lambdas" in npz.files else np.asarray([])
        if accepted.size == 0 and "lambda_hist" in npz.files:
            accepted = np.asarray(npz["lambda_hist"])
        newton = np.asarray(npz["newton_iterations"]) if "newton_iterations" in npz.files else np.asarray([])
        if newton.size == 0:
            pieces = []
            for key in ("stats_init_newton_iterations", "stats_step_newton_iterations_total"):
                if key in npz.files:
                    pieces.append(np.asarray(npz[key], dtype=np.float64).reshape(-1))
            newton = np.concatenate(pieces) if pieces else np.asarray([])
    return int(accepted.size), int(newton.sum()) if newton.size else None


def _read_manifest(root: Path) -> dict[str, dict[str, str]]:
    manifest = root / "manifest.tsv"
    if not manifest.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            run_name = Path(row["run_out"]).name
            rows[run_name] = row
    return rows


def _status(run_dir: Path, info: dict[str, Any]) -> str:
    explicit = str(info.get("status") or "")
    if explicit:
        return explicit
    if (run_dir / "data" / "run_info.json").exists():
        return "completed"
    if (run_dir / "stderr.txt").exists() or (run_dir / "stdout.txt").exists():
        return "started"
    return "missing"


def summarize(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    run_root = root / "runs" if (root / "runs").exists() else root
    manifest = _read_manifest(root)
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        info = _load_json(run_dir / "data" / "run_info.json")
        progress = _load_json(run_dir / "progress_latest.json")
        meta = manifest.get(run_dir.name, {})
        accepted, nonlinear = _history_counts(run_dir)
        if accepted is None and progress:
            accepted = progress.get("accepted_steps")
        rows.append(
            {
                "run": run_dir.name,
                "variant": meta.get("variant", run_dir.name),
                "layout": meta.get("layout", ""),
                "nodes": meta.get("nodes", ""),
                "ranks_per_node": meta.get("ranks_per_node", ""),
                "ranks": meta.get("ranks", _pick(info, "mpi_size", "world_size")),
                "status": _status(run_dir, info),
                "accepted_steps": accepted,
                "nonlinear_iterations": nonlinear,
                "linear_iterations": _pick(info, "linear_iterations_total", "linear_iteration_count_total")
                or _sum_pick(info, "init_linear_iterations", "attempt_linear_iterations_total"),
                "wall_time_s": _pick(
                    info,
                    "elapsed_wall_time_s",
                    "wall_time_s",
                    "total_time_s",
                    "runtime_seconds",
                    "continuation_total_wall_time",
                ),
                "linear_solve_s": _pick(info, "linear_solve_time_total_s", "time_linear_solve_total_s")
                or _sum_pick(info, "init_linear_solve_time", "attempt_linear_solve_time_total"),
                "pc_setup_s": _pick(info, "preconditioner_setup_time_total_s", "preconditioner_setup_time_total", "pc_setup_time_total_s"),
                "pc_apply_s": _pick(info, "preconditioner_time_total_s", "pc_apply_time_total_s", "preconditioner_apply_time_total"),
                "orthogonalize_s": _pick(info, "orthogonalization_time_total_s", "time_orthogonalization_total_s")
                or _sum_pick(info, "init_linear_orthogonalization_time", "attempt_linear_orthogonalization_time_total"),
                "tangent_s": _pick(info, "time_build_tangent_local_total_s", "build_tangent_local_total_s", "build_tangent_local"),
                "force_s": _pick(info, "time_build_force_total_s", "build_force_total_s", "build_F"),
                "manualmg_setup_s": _pick(info, "manualmg_setup_time_total_s", "manualmg_setup_time_s"),
                "manualmg_fine_s": _pick(info, "manualmg_fine_smoother_time_total_s")
                or _sum_pick(info, "manualmg_fine_pre_smoother_time_total_s", "manualmg_fine_post_smoother_time_total_s"),
                "manualmg_mid_s": _pick(info, "manualmg_mid_smoother_time_total_s")
                or _sum_pick(info, "manualmg_mid_pre_smoother_time_total_s", "manualmg_mid_post_smoother_time_total_s"),
                "manualmg_coarse_hypre_s": _pick(info, "manualmg_coarse_hypre_time_total_s"),
                "manualmg_residual_s": _pick(info, "manualmg_residual_time_total_s")
                or _sum_pick(info, "manualmg_fine_residual_time_total_s", "manualmg_mid_residual_time_total_s"),
                "manualmg_transfer_s": _pick(info, "manualmg_transfer_time_total_s")
                or _sum_pick(
                    info,
                    "manualmg_restrict_fine_to_mid_time_total_s",
                    "manualmg_restrict_mid_to_coarse_time_total_s",
                    "manualmg_prolong_coarse_to_mid_time_total_s",
                    "manualmg_prolong_mid_to_fine_time_total_s",
                ),
                "manualmg_coarse_pc": _pick(info, "manualmg_coarse_pc_type"),
                "manualmg_coarse_groups": _pick(info, "manualmg_coarse_redundant_group_count"),
                "manualmg_coarse_subcomm": _pick(info, "manualmg_coarse_subcomm_size"),
            }
        )
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--format", choices=("tsv", "markdown"), default="markdown")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = summarize(args.root)
    fields = [
        "variant",
        "layout",
        "ranks",
        "status",
        "accepted_steps",
        "nonlinear_iterations",
        "linear_iterations",
        "wall_time_s",
        "linear_solve_s",
        "pc_setup_s",
        "pc_apply_s",
        "orthogonalize_s",
        "tangent_s",
        "force_s",
        "manualmg_coarse_hypre_s",
        "manualmg_fine_s",
        "manualmg_mid_s",
        "manualmg_transfer_s",
        "manualmg_coarse_pc",
        "manualmg_coarse_groups",
        "manualmg_coarse_subcomm",
    ]
    if args.format == "tsv":
        lines = ["\t".join(fields)]
        lines.extend("\t".join(_fmt(row.get(field)) for field in fields) for row in rows)
    else:
        lines = [
            "| " + " | ".join(fields) + " |",
            "| " + " | ".join("---" for _ in fields) + " |",
        ]
        lines.extend("| " + " | ".join(_fmt(row.get(field)) for field in fields) + " |" for row in rows)
    text = "\n".join(lines) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
