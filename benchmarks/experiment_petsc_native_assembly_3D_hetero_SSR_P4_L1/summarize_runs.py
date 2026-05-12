#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _history_counts(npz_path: Path) -> tuple[int | None, int | None]:
    if not npz_path.exists():
        return None, None
    import numpy as np

    with np.load(npz_path, allow_pickle=True) as npz:
        accepted = np.asarray(npz["accepted_lambdas"]) if "accepted_lambdas" in npz.files else np.asarray([])
        newton = np.asarray(npz["newton_iterations"]) if "newton_iterations" in npz.files else np.asarray([])
    return int(accepted.size), int(newton.sum()) if newton.size else None


def _sources(info: dict) -> list[dict]:
    sources = [info]
    for section in ("run_info", "params", "timings", "diagnostics", "solver_diagnostics"):
        value = info.get(section)
        if isinstance(value, dict):
            sources.append(value)
    return sources


def _pick(info: dict, *keys: str):
    for source in _sources(info):
        for key in keys:
            if key in source:
                return source[key]
    return None


def summarize(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        data_dir = run_dir / "data"
        info = _load_json(data_dir / "run_info.json")
        accepted, nonlinear = _history_counts(data_dir / "petsc_run.npz")
        rows.append(
            {
                "case": run_dir.name,
                "status": info.get("status", ""),
                "backend": _pick(info, "tangent_matrix_backend", "execution_tangent_matrix_backend"),
                "pc_backend": _pick(info, "pc_backend", "linear_solver_pc_backend"),
                "accepted_steps": accepted,
                "nonlinear_iterations": nonlinear,
                "linear_iterations": _pick(info, "linear_iterations_total", "linear_iteration_count_total"),
                "tangent_time_s": _pick(info, "time_build_tangent_local_total_s", "build_tangent_local_total_s"),
                "pc_setup_s": _pick(info, "preconditioner_setup_time_total_s", "pc_setup_time_total_s"),
                "pc_apply_s": _pick(info, "preconditioner_time_total_s", "pc_apply_time_total_s"),
                "wall_time_s": _pick(info, "elapsed_wall_time_s", "wall_time_s", "total_time_s"),
            }
        )
    return rows


def main() -> None:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
    rows = summarize(root)
    fieldnames = [
        "case",
        "status",
        "backend",
        "pc_backend",
        "accepted_steps",
        "nonlinear_iterations",
        "linear_iterations",
        "tangent_time_s",
        "pc_setup_s",
        "pc_apply_s",
        "wall_time_s",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
