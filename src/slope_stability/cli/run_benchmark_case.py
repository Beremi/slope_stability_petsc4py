#!/usr/bin/env python3
"""Run one canonical MATLAB-vs-PETSc benchmark case from a benchmark TOML."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import tomllib

import h5py
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from slope_stability.core.run_config import load_run_case_config


CANONICAL_PETSC_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _read_benchmark_meta(config_path: Path) -> dict[str, object]:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    meta = dict(raw.get("benchmark", {}))
    if not meta:
        raise KeyError(f"{config_path} is missing a [benchmark] section.")
    return meta


def _single_quote(text: str) -> str:
    return text.replace("'", "''")


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(cmd, cwd=cwd, env=merged_env, check=check, text=True)


def _run_matlab(case_dir: Path, meta: dict[str, object], matlab_dir: Path) -> tuple[Path, Path]:
    script_name = str(meta["matlab_script"])
    matlab_out = matlab_dir / "matlab_run.mat"
    summary_json = matlab_dir / "summary.json"
    summary_h5 = matlab_dir / "summary.h5"
    solver_override = str(meta.get("matlab_solver_override", "")).strip()

    args = [str(matlab_out), str(matlab_dir)]
    if solver_override:
        args.append(solver_override)
    args_expr = ", ".join(f"'{_single_quote(arg)}'" for arg in args)
    matlab_root = ROOT / "slope_stability_matlab"
    matlab_scripts = matlab_root / "scripts"
    batch = (
        f"cd('{_single_quote(str(ROOT.resolve()))}'); "
        f"addpath(genpath('{_single_quote(str(matlab_root.resolve()))}')); "
        f"addpath('{_single_quote(str(matlab_scripts.resolve()))}'); "
        f"{script_name}({args_expr}); "
        f"export_benchmark_summary('{_single_quote(str(matlab_out))}', "
        f"'{_single_quote(str(summary_json))}', "
        f"'{_single_quote(str(summary_h5))}');"
    )
    proc = subprocess.Popen(
        ["/usr/local/bin/matlab", "-batch", batch],
        cwd=ROOT,
        env=os.environ.copy(),
        text=True,
    )
    success_files = (matlab_out, summary_json, summary_h5)
    success_seen_at: float | None = None
    while True:
        rc = proc.poll()
        if rc is not None:
            if rc != 0 and not all(path.exists() for path in success_files):
                raise subprocess.CalledProcessError(rc, proc.args)
            break
        if all(path.exists() for path in success_files):
            if success_seen_at is None:
                success_seen_at = time.monotonic()
            elif time.monotonic() - success_seen_at >= 5.0:
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10.0)
                break
        time.sleep(1.0)
    return summary_json, summary_h5


def _run_petsc(config_path: Path, meta: dict[str, object], petsc_dir: Path) -> None:
    mpi_ranks = int(meta.get("mpi_ranks", 8))
    cmd = [
        "mpirun",
        "-n",
        str(mpi_ranks),
        str(ROOT / ".venv" / "bin" / "python"),
        "-m",
        "slope_stability.cli.run_case_from_config",
        str(config_path),
        "--out_dir",
        str(petsc_dir),
    ]
    _run(cmd, cwd=ROOT, env=CANONICAL_PETSC_ENV)


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


def _load_petsc_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def _rel(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve())


def _make_continuation_plot(matlab_summary: dict[str, object], petsc_npz: dict[str, np.ndarray], out_path: Path) -> None:
    lambda_m = np.asarray(matlab_summary["continuation"]["lambda_hist"], dtype=np.float64)
    omega_m = np.asarray(matlab_summary["continuation"]["omega_hist"], dtype=np.float64)
    lambda_p = np.asarray(petsc_npz["lambda_hist"], dtype=np.float64)
    omega_p = np.asarray(petsc_npz["omega_hist"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=180)
    ax.plot(omega_m, lambda_m, marker="o", linewidth=1.4, label="MATLAB")
    ax.plot(omega_p, lambda_p, marker="s", linewidth=1.2, label="PETSc")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title("Continuation history")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _make_iteration_plot(matlab_h5: dict[str, np.ndarray], petsc_npz: dict[str, np.ndarray], out_path: Path) -> bool:
    step_newton_m = np.asarray(matlab_h5.get("continuation/stats/step_newton_iterations", []), dtype=np.float64).reshape(-1)
    step_linear_m = np.asarray(matlab_h5.get("continuation/stats/step_linear_iterations", []), dtype=np.float64).reshape(-1)
    step_newton_p = np.asarray(petsc_npz.get("stats_step_newton_iterations", []), dtype=np.float64).reshape(-1)
    step_linear_p = np.asarray(petsc_npz.get("stats_step_linear_iterations", []), dtype=np.float64).reshape(-1)
    if not (step_newton_m.size or step_newton_p.size or step_linear_m.size or step_linear_p.size):
        return False

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.0), dpi=180, sharex=True)
    if step_newton_m.size:
        axes[0].plot(np.arange(1, step_newton_m.size + 1), step_newton_m, marker="o", label="MATLAB")
    if step_newton_p.size:
        axes[0].plot(np.arange(1, step_newton_p.size + 1), step_newton_p, marker="s", label="PETSc")
    axes[0].set_ylabel("Newton iters")
    axes[0].grid(True)
    axes[0].legend()

    if step_linear_m.size:
        axes[1].plot(np.arange(1, step_linear_m.size + 1), step_linear_m, marker="o", label="MATLAB")
    if step_linear_p.size:
        axes[1].plot(np.arange(1, step_linear_p.size + 1), step_linear_p, marker="s", label="PETSc")
    axes[1].set_ylabel("Linear iters")
    axes[1].set_xlabel("Accepted step")
    axes[1].grid(True)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = max(float(np.linalg.norm(a)), 1.0e-30)
    return float(np.linalg.norm(a - b) / denom)


def _write_continuation_report(
    *,
    case_dir: Path,
    config_path: Path,
    meta: dict[str, object],
    matlab_summary: dict[str, object],
    matlab_h5: dict[str, np.ndarray],
    petsc_run_info: dict[str, object],
    petsc_npz: dict[str, np.ndarray],
    matlab_dir: Path,
    petsc_dir: Path,
) -> None:
    figures_dir = case_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    continuation_fig = figures_dir / "continuation_history.png"
    _make_continuation_plot(matlab_summary, petsc_npz, continuation_fig)
    iteration_fig = figures_dir / "iterations.png"
    has_iteration_fig = _make_iteration_plot(matlab_h5, petsc_npz, iteration_fig)

    lambda_m = np.asarray(matlab_summary["continuation"]["lambda_hist"], dtype=np.float64)
    omega_m = np.asarray(matlab_summary["continuation"]["omega_hist"], dtype=np.float64)
    umax_m = np.asarray(matlab_summary["continuation"]["umax_hist"], dtype=np.float64)
    lambda_p = np.asarray(petsc_npz["lambda_hist"], dtype=np.float64)
    omega_p = np.asarray(petsc_npz["omega_hist"], dtype=np.float64)
    umax_p = np.asarray(petsc_npz.get("Umax_hist", []), dtype=np.float64)
    matched = min(lambda_m.size, lambda_p.size, omega_m.size, omega_p.size, umax_m.size if umax_m.size else 10**9, umax_p.size if umax_p.size else 10**9)
    lambda_rel = _relative_error(lambda_m[:matched], lambda_p[:matched]) if matched else float("nan")
    omega_rel = _relative_error(omega_m[:matched], omega_p[:matched]) if matched else float("nan")
    umax_rel = _relative_error(umax_m[:matched], umax_p[:matched]) if matched and umax_m.size and umax_p.size else float("nan")
    final_umax_m = float(umax_m[-1]) if umax_m.size else float("nan")
    final_umax_p = float(umax_p[-1]) if umax_p.size else float("nan")

    matlab_runtime = float(matlab_summary["run_info"].get("runtime_seconds", 0.0))
    petsc_runtime = float(petsc_run_info["run_info"]["runtime_seconds"])
    step_newton_m = np.asarray(matlab_h5.get("continuation/stats/step_newton_iterations", []), dtype=np.int64).reshape(-1)
    step_linear_m = np.asarray(matlab_h5.get("continuation/stats/step_linear_iterations", []), dtype=np.int64).reshape(-1)
    step_newton_p = np.asarray(petsc_npz.get("stats_step_newton_iterations", []), dtype=np.int64).reshape(-1)
    step_linear_p = np.asarray(petsc_npz.get("stats_step_linear_iterations", []), dtype=np.int64).reshape(-1)

    step_lines = []
    step_count = max(step_newton_m.size, step_newton_p.size, step_linear_m.size, step_linear_p.size, lambda_m.size, lambda_p.size)
    for i in range(step_count):
        step_lines.append(
            "| {step} | {lm} | {lp} | {om} | {op} | {nm} | {np} | {linm} | {linp} |".format(
                step=i + 1,
                lm=_fmt(lambda_m, i),
                lp=_fmt(lambda_p, i),
                om=_fmt(omega_m, i),
                op=_fmt(omega_p, i),
                nm=_fmt(step_newton_m, i, integer=True),
                np=_fmt(step_newton_p, i, integer=True),
                linm=_fmt(step_linear_m, i, integer=True),
                linp=_fmt(step_linear_p, i, integer=True),
            )
        )

    matlab_plot_dir = matlab_dir / "matlab_plots"
    petsc_plot_candidates = [
        ("Displacement", "matlab_displacements_2D.png", "petsc_displacements_2D.png"),
        ("Strain", "matlab_deviatoric_strain_2D.png", "petsc_deviatoric_strain_2D.png"),
        ("Curve", "matlab_omega_lambda_2D.png", "petsc_omega_lambda_2D.png"),
    ]
    if "3d" in str(meta["title"]).lower():
        petsc_plot_candidates = [
            ("Displacement", "matlab_displacements_3D.png", "petsc_displacements_3D.png"),
            ("Strain", "matlab_deviatoric_strain_3D.png", "petsc_deviatoric_strain_3D.png"),
            ("Curve", "matlab_omega_lambda.png", "petsc_omega_lambda.png"),
        ]
    image_sections = []
    petsc_plot_dir = petsc_dir / "plots"
    for label, matlab_name, petsc_name in petsc_plot_candidates:
        matlab_path = matlab_plot_dir / matlab_name
        petsc_path = petsc_plot_dir / petsc_name
        if matlab_path.exists() and petsc_path.exists():
            image_sections.append(
                f"### {label}\n\n| MATLAB | PETSc |\n| --- | --- |\n| ![]({_rel(matlab_path, case_dir)}) | ![]({_rel(petsc_path, case_dir)}) |\n"
            )

    report = f"""# {meta['title']}

## Setup

- MATLAB script: `{meta['matlab_script']}`
- PETSc config: [`case.toml`](case.toml)
- Run command: [`run.sh`](run.sh)
- MPI ranks: `{meta.get('mpi_ranks', 8)}`

## Summary

| Metric | MATLAB | PETSc |
| --- | ---: | ---: |
| Runtime [s] | {matlab_runtime:.3f} | {petsc_runtime:.3f} |
| Accepted steps | {lambda_m.size} | {lambda_p.size} |
| Final lambda | {lambda_m[-1]:.12g} | {lambda_p[-1]:.12g} |
| Final omega | {omega_m[-1]:.12g} | {omega_p[-1]:.12g} |
| Final Umax | {final_umax_m:.12g} | {final_umax_p:.12g} |
| Relative lambda history error | {lambda_rel:.3e} | - |
| Relative omega history error | {omega_rel:.3e} | - |
| Relative Umax history error | {umax_rel:.3e} | - |

## Generated Comparison

![]({_rel(continuation_fig, case_dir)})

"""
    if has_iteration_fig:
        report += f"![]({_rel(iteration_fig, case_dir)})\n\n"
    report += """## Accepted-Step Table

| Step | MATLAB lambda | PETSc lambda | MATLAB omega | PETSc omega | MATLAB Newton | PETSc Newton | MATLAB linear | PETSc linear |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    report += "\n".join(step_lines) + "\n\n"
    if image_sections:
        report += "## Side-by-Side Figures\n\n" + "\n".join(image_sections)
    report += (
        "## Raw Outputs\n\n"
        f"- MATLAB artifacts: `{_rel(matlab_dir, case_dir)}`\n"
        f"- PETSc artifacts: `{_rel(petsc_dir, case_dir)}`\n"
    )
    for name in ("report.md", "README.md"):
        (case_dir / name).write_text(report, encoding="utf-8")


def _write_seepage_report(
    *,
    case_dir: Path,
    meta: dict[str, object],
    matlab_summary: dict[str, object],
    matlab_h5: dict[str, np.ndarray],
    petsc_run_info: dict[str, object],
    petsc_npz: dict[str, np.ndarray],
    matlab_dir: Path,
    petsc_dir: Path,
) -> None:
    pw_m = np.asarray(matlab_h5.get("seepage/pw", []), dtype=np.float64)
    grad_m = np.asarray(matlab_h5.get("seepage/grad_p", []), dtype=np.float64)
    sat_m = np.asarray(matlab_h5.get("seepage/mater_sat", []), dtype=np.float64)
    pw_p = np.asarray(petsc_npz.get("pw", petsc_npz.get("seepage_pw", [])), dtype=np.float64)
    grad_p = np.asarray(petsc_npz.get("grad_p", petsc_npz.get("seepage_grad_p", [])), dtype=np.float64)
    sat_p = np.asarray(petsc_npz.get("mater_sat", petsc_npz.get("seepage_mater_sat", [])), dtype=np.float64)
    if grad_m.ndim == 2 and grad_p.ndim == 2 and grad_m.shape == grad_p.T.shape:
        grad_m = grad_m.T

    pw_rel = _relative_error(pw_m, pw_p) if pw_m.size and pw_p.size else float("nan")
    grad_rel = _relative_error(grad_m, grad_p) if grad_m.size and grad_p.size else float("nan")
    sat_mismatch = int(np.count_nonzero(np.asarray(sat_m.reshape(-1) > 0.5, dtype=np.int8) != np.asarray(sat_p.reshape(-1) > 0.5, dtype=np.int8))) if sat_m.size and sat_p.size else -1

    matlab_runtime = float(matlab_summary["run_info"].get("runtime_seconds", 0.0))
    petsc_runtime = float(petsc_run_info["run_info"]["runtime_seconds"])

    image_rows = []
    figure_pairs = [
        ("Pore pressure", matlab_dir / "matlab_pore_pressure_2D.png", petsc_dir / "plots" / "petsc_pore_pressure_2D.png"),
        ("Saturation", matlab_dir / "matlab_saturation_2D.png", petsc_dir / "plots" / "petsc_saturation_2D.png"),
        ("Pore pressure", matlab_dir / "matlab_pore_pressure_3D.png", petsc_dir / "plots" / "petsc_pore_pressure_3D.png"),
        ("Saturation", matlab_dir / "matlab_saturation_3D.png", petsc_dir / "plots" / "petsc_saturation_3D.png"),
    ]
    for label, matlab_path, petsc_path in figure_pairs:
        if matlab_path.exists() and petsc_path.exists():
            image_rows.append(
                f"### {label}\n\n| MATLAB | PETSc |\n| --- | --- |\n| ![]({_rel(matlab_path, case_dir)}) | ![]({_rel(petsc_path, case_dir)}) |\n"
            )

    report = f"""# {meta['title']}

## Setup

- MATLAB script: `{meta['matlab_script']}`
- PETSc config: [`case.toml`](case.toml)
- Run command: [`run.sh`](run.sh)
- MPI ranks requested: `{meta.get('mpi_ranks', 8)}`
- PETSc MPI mode: `{petsc_run_info['run_info'].get('mpi_mode', 'serial')}`

## Summary

| Metric | MATLAB | PETSc |
| --- | ---: | ---: |
| Runtime [s] | {matlab_runtime:.3f} | {petsc_runtime:.3f} |
| Mesh nodes | {_mesh_count(matlab_summary, 'n_nodes')} | {int(petsc_run_info['run_info'].get('mesh_nodes', 0))} |
| Mesh elements | {_mesh_count(matlab_summary, 'n_elements')} | {int(petsc_run_info['run_info'].get('mesh_elements', 0))} |
| Relative pore-pressure error | {pw_rel:.3e} | - |
| Relative gradient error | {grad_rel:.3e} | - |
| Saturation mismatch count | {sat_mismatch} | - |

"""
    if image_rows:
        report += "## Side-by-Side Figures\n\n" + "\n".join(image_rows)
    report += (
        "## Raw Outputs\n\n"
        f"- MATLAB artifacts: `{_rel(matlab_dir, case_dir)}`\n"
        f"- PETSc artifacts: `{_rel(petsc_dir, case_dir)}`\n"
    )
    for name in ("report.md", "README.md"):
        (case_dir / name).write_text(report, encoding="utf-8")


def _fmt(arr: np.ndarray, idx: int, *, integer: bool = False) -> str:
    if idx >= arr.size:
        return "-"
    value = arr[idx]
    return str(int(value)) if integer else f"{float(value):.6g}"


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
    if isinstance(shape, list) and len(shape) >= 2:
        return int(shape[-1])
    return 0


def run_benchmark(config_path: Path, *, reuse_existing: bool = False) -> None:
    config_path = config_path.resolve()
    case_dir = config_path.parent
    cfg = load_run_case_config(config_path)
    meta = _read_benchmark_meta(config_path)

    artifacts_root = ROOT / "artifacts" / "benchmarks" / "mpi8" / case_dir.name
    matlab_dir = artifacts_root / "matlab"
    petsc_dir = artifacts_root / "petsc"
    matlab_dir.mkdir(parents=True, exist_ok=True)
    petsc_dir.mkdir(parents=True, exist_ok=True)

    summary_json = matlab_dir / "summary.json"
    summary_h5 = matlab_dir / "summary.h5"
    if not (reuse_existing and summary_json.exists() and summary_h5.exists()):
        summary_json, summary_h5 = _run_matlab(case_dir, meta, matlab_dir)

    if not (reuse_existing and (petsc_dir / "data" / "run_info.json").exists() and (petsc_dir / "data" / "petsc_run.npz").exists()):
        _run_petsc(config_path, meta, petsc_dir)

    matlab_summary = _load_json(summary_json)
    matlab_h5 = _load_h5_arrays(summary_h5)
    petsc_run_info = _load_json(petsc_dir / "data" / "run_info.json")
    petsc_npz = _load_petsc_npz(petsc_dir / "data" / "petsc_run.npz")

    kind = str(meta.get("comparison_kind", matlab_summary.get("kind", cfg.problem.analysis))).lower()
    if "continuation" in kind or cfg.problem.analysis.lower() in {"ssr", "ll"}:
        _write_continuation_report(
            case_dir=case_dir,
            config_path=config_path,
            meta=meta,
            matlab_summary=matlab_summary,
            matlab_h5=matlab_h5,
            petsc_run_info=petsc_run_info,
            petsc_npz=petsc_npz,
            matlab_dir=matlab_dir,
            petsc_dir=petsc_dir,
        )
    else:
        _write_seepage_report(
            case_dir=case_dir,
            meta=meta,
            matlab_summary=matlab_summary,
            matlab_h5=matlab_h5,
            petsc_run_info=petsc_run_info,
            petsc_npz=petsc_npz,
            matlab_dir=matlab_dir,
            petsc_dir=petsc_dir,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one canonical MATLAB-vs-PETSc benchmark case.")
    parser.add_argument("config", type=Path, help="Benchmark case TOML.")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing MATLAB/PETSc artifacts if present and only regenerate the report.")
    args = parser.parse_args()
    run_benchmark(args.config, reuse_existing=args.reuse_existing)


if __name__ == "__main__":
    main()
