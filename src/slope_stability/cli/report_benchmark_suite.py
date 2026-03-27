#!/usr/bin/env python3
"""Generate the benchmark-suite README from completed case artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def _load_toml(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _case_dirs(suite_dir: Path) -> tuple[list[Path], list[Path]]:
    benchmark_case_dirs: list[Path] = []
    runnable_case_dirs: list[Path] = []
    for path in sorted(p for p in suite_dir.iterdir() if p.is_dir() and (p / "case.toml").exists()):
        config = _load_toml(path / "case.toml")
        benchmark = dict(config.get("benchmark", {}))
        if benchmark.get("suite", False):
            benchmark_case_dirs.append(path)
        else:
            runnable_case_dirs.append(path)
    return benchmark_case_dirs, runnable_case_dirs


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_h5_arrays(path: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as h5:
        def visit(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset):
                arrays[name] = np.asarray(obj[()])
        h5.visititems(visit)
    return arrays


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def _rel(path: Path, base: Path) -> str:
    return str(path.relative_to(base))


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    denom = max(float(np.linalg.norm(a)), 1.0e-30)
    return float(np.linalg.norm(a - b) / denom)


def _mesh_count(summary: dict[str, object], kind: str) -> int:
    mesh = dict(summary.get("mesh", {}))
    run_info = dict(summary.get("run_info", {}))
    direct = mesh.get(kind)
    if isinstance(direct, (int, float)) and direct:
        return int(direct)
    from_run = run_info.get(kind)
    if isinstance(from_run, (int, float)) and from_run:
        return int(from_run)
    shape = mesh.get("coord_shape" if kind == "n_nodes" else "elem_shape")
    if isinstance(shape, list) and shape:
        dims = [int(v) for v in shape if isinstance(v, (int, float))]
        if dims:
            return max(dims)
    return 0


def _build_case_row(case_dir: Path, artifacts_root: Path) -> tuple[str, str]:
    config = _load_toml(case_dir / "case.toml")
    meta = dict(config.get("benchmark", {}))
    title = str(meta.get("title", case_dir.name))
    kind = str(meta.get("comparison_kind", "")).lower()
    matlab_dir = artifacts_root / case_dir.name / "matlab"
    petsc_dir = artifacts_root / case_dir.name / "petsc"
    if not (matlab_dir / "summary.json").exists() or not (petsc_dir / "data" / "run_info.json").exists():
        return (
            f"| `{case_dir.name}` | {title} | {kind or '-'} | missing | - | - | - | - | [case]({_rel(case_dir, case_dir.parent)}) |",
            "",
        )

    matlab_summary = _load_json(matlab_dir / "summary.json")
    petsc_run = _load_json(petsc_dir / "data" / "run_info.json")
    matlab_runtime = float(dict(matlab_summary.get("run_info", {})).get("runtime_seconds", 0.0))
    petsc_runtime = float(dict(petsc_run.get("run_info", {})).get("runtime_seconds", 0.0))
    notes = []

    if "seepage" in kind and "continuation" not in kind:
        matlab_h5 = _load_h5_arrays(matlab_dir / "summary.h5")
        petsc_npz = _load_npz(petsc_dir / "data" / "petsc_run.npz")
        pw_m = np.asarray(matlab_h5.get("seepage/pw", []), dtype=np.float64)
        grad_m = np.asarray(matlab_h5.get("seepage/grad_p", []), dtype=np.float64)
        sat_m = np.asarray(matlab_h5.get("seepage/mater_sat", []), dtype=np.float64)
        pw_p = np.asarray(petsc_npz.get("pw", petsc_npz.get("seepage_pw", [])), dtype=np.float64)
        grad_p = np.asarray(petsc_npz.get("grad_p", petsc_npz.get("seepage_grad_p", [])), dtype=np.float64)
        sat_p = np.asarray(petsc_npz.get("mater_sat", petsc_npz.get("seepage_mater_sat", [])), dtype=np.float64)
        if grad_m.ndim == 2 and grad_p.ndim == 2 and grad_m.shape == grad_p.T.shape:
            grad_m = grad_m.T
        pw_rel = _relative_error(pw_m, pw_p)
        grad_rel = _relative_error(grad_m, grad_p)
        sat_mismatch = (
            int(np.count_nonzero((sat_m.reshape(-1) > 0.5) != (sat_p.reshape(-1) > 0.5)))
            if sat_m.size and sat_p.size
            else -1
        )
        summary = f"`pw {pw_rel:.2e}`, `grad {grad_rel:.2e}`, `sat {sat_mismatch}`"
        notes.append(f"- Mesh: `{_mesh_count(matlab_summary, 'n_nodes')}` nodes, `{_mesh_count(matlab_summary, 'n_elements')}` elements")
        mpi_mode = dict(petsc_run.get("run_info", {})).get("mpi_mode", "serial")
        if mpi_mode != "distributed":
            notes.append(f"- PETSc MPI mode: `{mpi_mode}`")
    else:
        matlab_h5 = _load_h5_arrays(matlab_dir / "summary.h5")
        petsc_npz = _load_npz(petsc_dir / "data" / "petsc_run.npz")
        lambda_m = np.asarray(dict(matlab_summary.get("continuation", {})).get("lambda_hist", []), dtype=np.float64)
        omega_m = np.asarray(dict(matlab_summary.get("continuation", {})).get("omega_hist", []), dtype=np.float64)
        lambda_p = np.asarray(petsc_npz.get("lambda_hist", []), dtype=np.float64)
        omega_p = np.asarray(petsc_npz.get("omega_hist", []), dtype=np.float64)
        lambda_rel = _relative_error(lambda_m, lambda_p)
        omega_rel = _relative_error(omega_m, omega_p)
        summary = (
            f"`steps {lambda_m.size}/{lambda_p.size}`, "
            f"`lambda {lambda_rel:.2e}`, "
            f"`omega {omega_rel:.2e}`"
        )
        step_linear_m = np.asarray(matlab_h5.get("continuation/stats/step_linear_iterations", []), dtype=np.float64).reshape(-1)
        step_linear_p = np.asarray(petsc_npz.get("stats_step_linear_iterations", []), dtype=np.float64).reshape(-1)
        if step_linear_m.size and step_linear_p.size:
            notes.append(
                f"- Linear iterations total: MATLAB `{int(step_linear_m.sum())}`, PETSc `{int(step_linear_p.sum())}`"
            )

    row = (
        f"| `{case_dir.name}` | {title} | {kind or '-'} | done | "
        f"{matlab_runtime:.3f} | {petsc_runtime:.3f} | {summary} | "
        f"[README]({_rel(case_dir / 'README.md', case_dir.parent)}) | "
        f"[run.sh]({_rel(case_dir / 'run.sh', case_dir.parent)}) |"
    )
    detail = ""
    if notes:
        detail = "\n".join(notes)
    return row, detail


def _build_runnable_case_row(case_dir: Path) -> str:
    config = _load_toml(case_dir / "case.toml")
    problem = dict(config.get("problem", {}))
    return (
        f"| `{case_dir.name}` | "
        f"{problem.get('case', '-')} | "
        f"{problem.get('analysis', '-')} | "
        f"{problem.get('dimension', '-')}D | "
        f"{problem.get('elem_type', '-')} | "
        f"[README]({_rel(case_dir / 'README.md', case_dir.parent)}) | "
        f"[run.sh]({_rel(case_dir / 'run.sh', case_dir.parent)}) | "
        f"[case.toml]({_rel(case_dir / 'case.toml', case_dir.parent)}) |"
    )


def generate_suite_readme(suite_dir: Path) -> None:
    suite_dir = suite_dir.resolve()
    artifacts_root = ROOT / "artifacts" / "benchmarks" / "mpi8"
    benchmark_case_dirs, runnable_case_dirs = _case_dirs(suite_dir)
    rows: list[str] = []
    detail_blocks: list[str] = []
    for case_dir in benchmark_case_dirs:
        row, detail = _build_case_row(case_dir, artifacts_root)
        rows.append(row)
        if detail:
            detail_blocks.append(f"### `{case_dir.name}`\n\n{detail}")
    runnable_rows = [_build_runnable_case_row(case_dir) for case_dir in runnable_case_dirs]

    report = """# MATLAB Original Script Benchmarks

Unified case registry.

Each case folder contains at least:
- `case.toml`
- `run.sh`
- `README.md`

The canonical MATLAB-parity benchmark suite is the subset with `[benchmark].suite = true` in `case.toml`.

Run the full parity suite:

```bash
./.venv/bin/python -m slope_stability.cli.run_benchmark_suite
```

Run any single case from its folder with `./run.sh`.

## MATLAB-Parity Benchmarks

| Case | Title | Kind | Status | MATLAB [s] | PETSc [s] | Parity summary | Results | Run |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
"""
    report += "\n".join(rows) + "\n\n"
    if runnable_rows:
        report += """## Additional Runnable Cases

These folders are part of the unified case registry, but they are not included in the canonical MATLAB-parity suite.

| Folder | Problem case | Analysis | Dimension | Element | README | Run | Config |
| --- | --- | --- | --- | --- | --- | --- | --- |
"""
        report += "\n".join(runnable_rows) + "\n\n"
    if detail_blocks:
        report += "## Notes\n\n" + "\n\n".join(detail_blocks) + "\n"
    (suite_dir / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the benchmark-suite README.")
    parser.add_argument(
        "--suite_dir",
        type=Path,
        default=ROOT / "benchmarks",
    )
    args = parser.parse_args()
    generate_suite_readme(args.suite_dir)


if __name__ == "__main__":
    main()
