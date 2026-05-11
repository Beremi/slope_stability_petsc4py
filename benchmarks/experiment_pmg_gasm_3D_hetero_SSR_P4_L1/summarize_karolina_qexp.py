#!/usr/bin/env python3
"""Summarize Karolina PMG/GASM one-node Qexp runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = ROOT / "artifacts/experiments/pmg_gasm_karolina_qexp_one_node_p4_l1_omega7"


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


def _row_for_run_info(run_info_path: Path) -> dict[str, Any]:
    run_out = run_info_path.parents[1]
    run_dir = run_out.parent if run_out.name == "run" else run_out
    metadata = _read_json(run_dir / "case_metadata.json")
    payload = _read_json(run_info_path)
    info = payload.get("run_info") or payload
    params = payload.get("params") or {}
    timings = payload.get("timings") or {}
    linear = timings.get("linear") or {}
    progress = _final_progress(run_out)

    pc_setup_total = linear.get("preconditioner_setup_time_total")
    pc_apply_total = linear.get("preconditioner_apply_time_total")
    pc_setup_count = linear.get("manualmg_setup_count") or linear.get("preconditioner_rebuild_count")
    pc_apply_count = linear.get("manualmg_apply_count")
    full_setup_count = linear.get("manualmg_setup_full_count")
    reuse_setup_count = linear.get("manualmg_setup_reuse_count")
    linear_iterations = (
        int(linear.get("init_linear_iterations") or 0)
        + int(linear.get("attempt_linear_iterations_total") or 0)
    )
    if not linear_iterations:
        linear_iterations = None

    return {
        "case": metadata.get("case_name") or run_dir.name,
        "kind": metadata.get("kind") or ("gasm" if params.get("pmg_smoother_pc_type") == "gasm" else "baseline"),
        "ranks": metadata.get("ranks") or info.get("mpi_size"),
        "sockets": metadata.get("sockets") or linear.get("manualmg_smoother_gasm_total_subdomains"),
        "ranks_per_socket": metadata.get("ranks_per_socket")
        or linear.get("manualmg_smoother_gasm_ranks_per_subdomain"),
        "exit_code": _read_int(run_dir / "exit_code.txt"),
        "wall_s": _read_float(run_dir / "wall_seconds.txt") or info.get("runtime_seconds"),
        "omega_last": progress.get("last_omega"),
        "lambda_last": progress.get("last_lambda"),
        "load_steps": info.get("step_count") or progress.get("count_omega"),
        "linear_iterations": linear_iterations,
        "pc_setup_s": pc_setup_total,
        "pc_apply_s": pc_apply_total,
        "setup_count": pc_setup_count,
        "full_setup_count": full_setup_count,
        "reuse_setup_count": reuse_setup_count,
        "setup_per_setup_s": pc_setup_total / pc_setup_count if pc_setup_total and pc_setup_count else None,
        "apply_per_apply_s": pc_apply_total / pc_apply_count if pc_apply_total and pc_apply_count else None,
        "path": str(run_dir),
    }


def _discover_rows(out_root: Path) -> list[dict[str, Any]]:
    paths = set((out_root / "runs").glob("*/*/run/data/run_info.json"))
    paths.update(out_root.glob("*/data/run_info.json"))
    paths = sorted(paths)
    return [_row_for_run_info(path) for path in paths]


def _write_tsv(rows: list[dict[str, Any]], out_path: Path) -> None:
    columns = [
        "case",
        "kind",
        "ranks",
        "sockets",
        "ranks_per_socket",
        "exit_code",
        "wall_s",
        "omega_last",
        "lambda_last",
        "load_steps",
        "linear_iterations",
        "pc_setup_s",
        "pc_apply_s",
        "setup_count",
        "full_setup_count",
        "reuse_setup_count",
        "setup_per_setup_s",
        "apply_per_apply_s",
        "path",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(_fmt(row.get(column)) for column in columns))
    out_path.write_text("\n".join(lines) + "\n")


def _write_markdown(rows: list[dict[str, Any]], out_path: Path) -> None:
    columns = [
        ("case", "case"),
        ("ranks", "ranks"),
        ("sockets", "sockets"),
        ("ranks_per_socket", "r/socket"),
        ("exit_code", "exit"),
        ("wall_s", "wall s"),
        ("omega_last", "omega"),
        ("load_steps", "steps"),
        ("linear_iterations", "lin it"),
        ("pc_setup_s", "setup s"),
        ("pc_apply_s", "apply s"),
        ("setup_per_setup_s", "setup/setup"),
        ("apply_per_apply_s", "apply/apply"),
    ]
    lines = ["| " + " | ".join(label for _, label in columns) + " |"]
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key, _ in columns) + " |")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    rows = _discover_rows(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    _write_tsv(rows, out_root / "summary.tsv")
    _write_markdown(rows, out_root / "summary.md")
    print(f"Wrote {len(rows)} row(s) to {out_root / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
