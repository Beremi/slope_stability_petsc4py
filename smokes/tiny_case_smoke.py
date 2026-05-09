#!/usr/bin/env python
"""Run a one-step serial mechanics smoke case under artifacts/."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = ROOT / "benchmarks" / "run_2D_homo_SSR_capture" / "case.toml"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "smokes" / "tiny_2d_homo_ssr"


def _replace_in_section(text: str, section: str, key: str, value: str) -> str:
    header = f"[{section}]"
    start = text.find(header)
    if start < 0:
        return f"{text.rstrip()}\n\n{header}\n{key} = {value}\n"
    next_section = text.find("\n[", start + len(header))
    if next_section < 0:
        next_section = len(text)
    block = text[start:next_section]
    pattern = re.compile(rf"(?m)^({re.escape(key)}\s*=\s*).*$")
    if pattern.search(block):
        block = pattern.sub(rf"\g<1>{value}", block)
    else:
        block = block.rstrip() + f"\n{key} = {value}\n"
    return text[:start] + block + text[next_section:]


def _write_smoke_config(base_config: Path, config_path: Path, *, factor_solver_type: str | None) -> None:
    text = base_config.read_text(encoding="utf-8")
    edits = [
        ("problem", "name", '"tiny_2d_homo_ssr_smoke"'),
        ("problem", "elem_type", '"P1"'),
        ("execution", "node_ordering", '"none"'),
        ("execution", "mpi_distribute_by_nodes", "false"),
        ("execution", "constitutive_mode", '"overlap"'),
        ("continuation", "step_max", "1"),
        ("newton", "it_max", "8"),
        ("newton", "it_damp_max", "4"),
        ("linear_solver", "solver_type", '"KSPPREONLY_LU"'),
        ("linear_solver", "tolerance", "1e-10"),
        ("linear_solver", "max_iterations", "1"),
        ("linear_solver", "recycle_preconditioner", "false"),
        ("export", "write_custom_debug_bundle", "false"),
        ("export", "write_history_json", "true"),
        ("export", "write_solution_vtu", "false"),
    ]
    if factor_solver_type:
        edits.append(("linear_solver", "factor_solver_type", json.dumps(factor_solver_type)))
    for section, key, value in edits:
        text = _replace_in_section(text, section, key, value)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python executable to use for the CLI run.")
    parser.add_argument("--factor-solver-type", default=None, help="Optional PETSc LU package, for example mumps.")
    args = parser.parse_args()

    try:
        from petsc4py import PETSc

        mpi_size = int(PETSc.COMM_WORLD.getSize())
    except Exception:
        mpi_size = 1
    if mpi_size != 1:
        raise SystemExit("tiny_case_smoke.py is intentionally serial; use environment_smoke.py for MPI validation.")

    out_dir = args.out_dir.resolve()
    config_path = out_dir / "case.toml"
    run_dir = out_dir / "run"
    _write_smoke_config(args.base_config.resolve(), config_path, factor_solver_type=args.factor_solver_type)

    cmd = [
        str(args.python),
        "-m",
        "slope_stability.cli.run_case_from_config",
        str(config_path),
        "--out_dir",
        str(run_dir),
    ]
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        completed.check_returncode()

    run_info_path = run_dir / "data" / "run_info.json"
    petsc_npz_path = run_dir / "data" / "petsc_run.npz"
    if not run_info_path.exists():
        raise SystemExit(f"missing expected {run_info_path}")
    if not petsc_npz_path.exists():
        raise SystemExit(f"missing expected {petsc_npz_path}")

    data = json.loads(run_info_path.read_text(encoding="utf-8"))
    info = data.get("run_info", {})
    mesh = data.get("mesh", {})
    coord_shape = mesh.get("coord_shape") or [None, None]
    elem_shape = mesh.get("elem_shape") or [None, None]
    print(
        json.dumps(
            {
                "status": "ok",
                "case": str(config_path.relative_to(ROOT)),
                "run_dir": str(run_dir.relative_to(ROOT)),
                "elem_type": data.get("params", {}).get("elem_type"),
                "mesh_nodes": coord_shape[1] if len(coord_shape) > 1 else None,
                "mesh_elements": elem_shape[1] if len(elem_shape) > 1 else None,
                "rank_count": info.get("rank_count"),
                "step_count": info.get("step_count"),
                "runtime_seconds": info.get("runtime_seconds"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
