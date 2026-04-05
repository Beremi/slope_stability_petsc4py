#!/usr/bin/env python
"""Regenerate and minify benchmark notebook artifacts for committed reuse."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = ROOT / "benchmarks"
SRC_DIR = ROOT / "src"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import notebook_support as nb
from slope_stability.export import write_vtu


NPZ_KEEP_KEYS = (
    "U",
    "lambda_hist",
    "omega_hist",
    "Umax_hist",
    "pore_pressure_export",
    "pw_export",
    "seepage_pw_reordered",
    "pw_reordered",
    "pw",
    "seepage_pw",
    "saturation",
    "mater_sat",
    "seepage_mater_sat",
)
VTU_POINT_KEEP = {
    "displacement",
    "displacement_magnitude",
    "pore_pressure",
    "deviatoric_strain",
}
VTU_CELL_KEEP = {
    "material_id",
    "saturation",
    "deviatoric_strain",
}
REQUIRED_RELATIVE_PATHS = (
    Path("generated_case.toml"),
    Path("data/run_info.json"),
    Path("data/petsc_run.npz"),
    Path("exports/final_solution.vtu"),
)


@dataclass(frozen=True)
class SnapshotSummary:
    benchmark: str
    out_dir: Path
    runtime_seconds: float
    lambda_last: float | None
    omega_last: float | None
    step_count: int
    bytes_before: int
    bytes_after: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional benchmark directory names to run. Defaults to every benchmarks/*/case.toml.",
    )
    parser.add_argument("--run-label", default="simulation", help="Artifact label to refresh.")
    parser.add_argument(
        "--execution-profile",
        default="benchmark",
        choices=("smoke", "benchmark", "full"),
        help="Notebook execution profile to use when generating snapshots.",
    )
    parser.add_argument(
        "--mpi-ranks",
        type=int,
        default=None,
        help="Optional MPI rank override. Defaults to the notebook profile setting.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON file path for the generated size/runtime summary.",
    )
    return parser.parse_args()


def _selected_case_tomls(case_names: list[str] | None) -> list[Path]:
    all_cases = {path.parent.name: path for path in nb.benchmark_case_tomls(BENCHMARKS_DIR)}
    if not case_names:
        return [all_cases[name] for name in sorted(all_cases)]
    missing = [name for name in case_names if name not in all_cases]
    if missing:
        raise SystemExit(f"Unknown benchmark case(s): {', '.join(missing)}")
    return [all_cases[name] for name in case_names]


def _dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _keep_npz_arrays(npz: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    kept: dict[str, np.ndarray] = {}
    for key in NPZ_KEEP_KEYS:
        if key in npz:
            kept[key] = np.asarray(npz[key])
    return kept


def _rewrite_npz(npz_path: Path) -> None:
    with np.load(npz_path, allow_pickle=True) as npz_file:
        arrays = {name: np.asarray(npz_file[name]) for name in npz_file.files}
    kept = _keep_npz_arrays(arrays)
    tmp_path = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **kept)
    tmp_path.replace(npz_path)


def _rewrite_vtu(vtu_path: Path) -> None:
    vtu = nb.load_vtu(vtu_path)
    point_data = {name: values for name, values in vtu.point_data.items() if name in VTU_POINT_KEEP}
    cell_data = {name: values for name, values in vtu.cell_data.items() if name in VTU_CELL_KEEP}
    write_vtu(
        vtu_path,
        points=vtu.points,
        cell_blocks=vtu.cell_blocks,
        point_data=point_data,
        cell_data=cell_data,
    )


def _prune_artifact_dir(out_dir: Path) -> None:
    keep = {path.as_posix() for path in REQUIRED_RELATIVE_PATHS}
    for item in sorted(out_dir.rglob("*")):
        if not item.is_file():
            continue
        rel = item.relative_to(out_dir).as_posix()
        if rel not in keep:
            item.unlink()
    for directory in sorted((path for path in out_dir.rglob("*") if path.is_dir()), reverse=True):
        if directory == out_dir:
            continue
        try:
            directory.rmdir()
        except OSError:
            continue


def _prepare_case(case_toml: Path, *, run_label: str, execution_profile: str, mpi_ranks: int | None) -> SnapshotSummary:
    sections = nb.load_case_sections(case_toml)
    materials = nb.load_case_materials(case_toml)
    execution = nb.ensure_notebook_artifacts(
        case_toml=case_toml,
        sections=sections,
        materials=materials,
        run_label=run_label,
        run_mode="run",
        execution_profile=execution_profile,
        mpi_ranks=mpi_ranks,
        root=case_toml.parent,
    )
    out_dir = execution.out_dir
    bytes_before = _dir_size(out_dir)
    _rewrite_npz(out_dir / "data" / "petsc_run.npz")
    _rewrite_vtu(out_dir / "exports" / "final_solution.vtu")
    _prune_artifact_dir(out_dir)
    bytes_after = _dir_size(out_dir)
    artifacts = nb.load_run_artifacts(out_dir)
    summary = nb._run_completion_summary(artifacts)  # noqa: SLF001
    return SnapshotSummary(
        benchmark=case_toml.parent.name,
        out_dir=out_dir,
        runtime_seconds=float(summary["runtime_seconds"]),
        lambda_last=summary["lambda_last"],
        omega_last=summary["omega_last"],
        step_count=int(summary["step_count"]),
        bytes_before=bytes_before,
        bytes_after=bytes_after,
    )


def _summary_payload(results: list[SnapshotSummary]) -> dict[str, object]:
    total_before = int(sum(item.bytes_before for item in results))
    total_after = int(sum(item.bytes_after for item in results))
    return {
        "total_bytes_before": total_before,
        "total_bytes_after": total_after,
        "total_bytes_saved": total_before - total_after,
        "cases": [
            {
                "benchmark": item.benchmark,
                "out_dir": str(item.out_dir),
                "runtime_seconds": item.runtime_seconds,
                "lambda_last": item.lambda_last,
                "omega_last": item.omega_last,
                "step_count": item.step_count,
                "bytes_before": item.bytes_before,
                "bytes_after": item.bytes_after,
                "bytes_saved": item.bytes_before - item.bytes_after,
            }
            for item in results
        ],
    }


def main() -> None:
    args = _parse_args()
    results = [
        _prepare_case(
            case_toml,
            run_label=args.run_label,
            execution_profile=args.execution_profile,
            mpi_ranks=args.mpi_ranks,
        )
        for case_toml in _selected_case_tomls(args.cases)
    ]
    payload = _summary_payload(results)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"{'benchmark':<40} {'runtime[s]':>11} {'steps':>7} {'before[MB]':>12} {'after[MB]':>11} {'saved[MB]':>11}",
        flush=True,
    )
    for item in results:
        print(
            f"{item.benchmark:<40} "
            f"{item.runtime_seconds:>11.3f} "
            f"{item.step_count:>7d} "
            f"{item.bytes_before / (1024 ** 2):>12.2f} "
            f"{item.bytes_after / (1024 ** 2):>11.2f} "
            f"{(item.bytes_before - item.bytes_after) / (1024 ** 2):>11.2f}",
            flush=True,
        )
    print(
        f"{'TOTAL':<40} "
        f"{'':>11} "
        f"{'':>7} "
        f"{payload['total_bytes_before'] / (1024 ** 2):>12.2f} "
        f"{payload['total_bytes_after'] / (1024 ** 2):>11.2f} "
        f"{payload['total_bytes_saved'] / (1024 ** 2):>11.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
