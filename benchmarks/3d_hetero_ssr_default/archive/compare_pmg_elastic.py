from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "archive" else SCRIPT_DIR
ROOT = BENCHMARK_DIR.parents[1]
DEFAULT_OUT_ROOT = ROOT / "artifacts" / "p4_pmg_elastic"
DEFAULT_REPORT = SCRIPT_DIR / "report_p4_pmg_elastic.md"
DEFAULT_SUMMARY = DEFAULT_OUT_ROOT / "summary.json"
DEFAULT_MESH = ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh"
STATE_ROOT = ROOT / "artifacts" / "p4_scaling_step2"
LINEAR_TOL = 1.0e-3


@dataclass(frozen=True)
class RunSpec:
    name: str
    stage: str
    ranks: int
    timeout_s: int
    command: tuple[str, ...]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_or_none(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run(spec: RunSpec, *, out_root: Path) -> dict[str, object]:
    run_dir = _ensure_dir(out_root / spec.stage / spec.name)
    data_dir = _ensure_dir(run_dir / "data")
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env["PYTHONPATH"] = str(ROOT / "src")

    t0 = time.perf_counter()
    status = "completed"
    return_code = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                spec.command,
                cwd=ROOT,
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout=spec.timeout_s,
                check=False,
            )
        return_code = int(completed.returncode)
        if completed.returncode != 0:
            status = "failed"
    except subprocess.TimeoutExpired:
        status = "timeout"
    elapsed = float(time.perf_counter() - t0)

    run_info = _json_or_none(data_dir / "run_info.json")
    entry: dict[str, object] = {
        "name": spec.name,
        "stage": spec.stage,
        "ranks": int(spec.ranks),
        "status": status,
        "timeout_s": int(spec.timeout_s),
        "elapsed_wall_s": elapsed,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "run_info_path": str(data_dir / "run_info.json"),
        "command": list(spec.command),
    }
    if return_code is not None:
        entry["return_code"] = int(return_code)
    if run_info is not None:
        entry["metrics"] = {
            "pc_backend": run_info.get("pc_backend"),
            "iteration_count": run_info.get("iteration_count", run_info.get("solve_delta", {}).get("iterations")),
            "setup_elapsed_s": run_info.get("setup_elapsed_s"),
            "solve_elapsed_s": run_info.get("solve_elapsed_s"),
            "solve_plus_setup_elapsed_s": run_info.get("solve_plus_setup_elapsed_s"),
            "final_relative_residual": run_info.get("final_relative_residual"),
            "steps": run_info.get("steps"),
            "lambda_last": run_info.get("lambda_last"),
            "omega_last": run_info.get("omega_last"),
        }
    return entry


def _linear_converged(entry: dict[str, object], *, tolerance: float) -> bool:
    if entry.get("stage") != "linear" or entry.get("status") != "completed":
        return False
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        return False
    residual = metrics.get("final_relative_residual")
    if residual is None:
        return False
    try:
        return float(residual) <= float(tolerance)
    except (TypeError, ValueError):
        return False


def _frozen_specs(python: str, out_root: Path) -> list[RunSpec]:
    probe = ROOT / "benchmarks" / "3d_hetero_ssr_default" / "archive" / "probe_hypre_frozen.py"
    common = (
        "--state-selector",
        "hard",
        "--outer-solver-family",
        "repo",
        "--solver-type",
        "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        "--linear-tolerance",
        str(LINEAR_TOL),
        "--linear-max-iter",
        "80",
    )
    specs: list[RunSpec] = []
    for ranks in (1, 8):
        state_npz = STATE_ROOT / f"rank{ranks}" / "data" / "petsc_run.npz"
        state_run_info = STATE_ROOT / f"rank{ranks}" / "data" / "run_info.json"
        mpiexec = ("/usr/bin/mpiexec", "-n", str(ranks)) if ranks > 1 else ()
        specs.append(
            RunSpec(
                name=f"rank{ranks}_hypre_current",
                stage="linear",
                ranks=ranks,
                timeout_s=1800,
                command=(
                    *mpiexec,
                    python,
                    str(probe),
                    "--out-dir",
                    str(out_root / "linear" / f"rank{ranks}_hypre_current"),
                    "--state-npz",
                    str(state_npz),
                    "--state-run-info",
                    str(state_run_info),
                    "--pc-backend",
                    "hypre",
                    "--pmat-source",
                    "tangent",
                    *common,
                ),
            )
        )
        specs.append(
            RunSpec(
                name=f"rank{ranks}_pmg_elastic",
                stage="linear",
                ranks=ranks,
                timeout_s=1800,
                command=(
                    *mpiexec,
                    python,
                    str(probe),
                    "--out-dir",
                    str(out_root / "linear" / f"rank{ranks}_pmg_elastic"),
                    "--state-npz",
                    str(state_npz),
                    "--state-run-info",
                    str(state_run_info),
                    "--pc-backend",
                    "pmg",
                    "--pmat-source",
                    "elastic",
                    *common,
                ),
            )
        )
    return specs


def _nonlinear_specs(python: str, out_root: Path) -> list[RunSpec]:
    runner = "slope_stability.cli.run_3D_hetero_SSR_capture"
    base = (
        "--mesh_path",
        str(DEFAULT_MESH),
        "--elem_type",
        "P4",
        "--step_max",
        "2",
        "--solver_type",
        "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        "--linear_tolerance",
        "1e-1",
        "--linear_max_iter",
        "100",
        "--no-store_step_u",
    )
    return [
        RunSpec(
            name="rank8_hypre_current_step2",
            stage="nonlinear",
            ranks=8,
            timeout_s=3600,
            command=(
                "/usr/bin/mpiexec",
                "-n",
                "8",
                python,
                "-m",
                runner,
                "--out_dir",
                str(out_root / "nonlinear" / "rank8_hypre_current_step2"),
                "--pc_backend",
                "hypre",
                "--preconditioner_matrix_source",
                "tangent",
                *base,
            ),
        ),
        RunSpec(
            name="rank8_pmg_elastic_step2",
            stage="nonlinear",
            ranks=8,
            timeout_s=3600,
            command=(
                "/usr/bin/mpiexec",
                "-n",
                "8",
                python,
                "-m",
                runner,
                "--out_dir",
                str(out_root / "nonlinear" / "rank8_pmg_elastic_step2"),
                "--pc_backend",
                "pmg",
                "--preconditioner_matrix_source",
                "elastic",
                *base,
            ),
        ),
    ]


def _write_report(*, entries: list[dict[str, object]], report_path: Path, summary_path: Path) -> None:
    lines = [
        "# P4 Elastic PMG Report",
        "",
        f"- Summary JSON: `{summary_path}`",
        f"- Linear gate tolerance: `{LINEAR_TOL:.1e}`",
        "",
        "## Linear Gate",
        "",
        "| Run | Status | Converged | Backend | Setup [s] | Solve [s] | Total [s] | Iterations | Final relative residual |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    linear_entries = [entry for entry in entries if entry["stage"] == "linear"]
    for entry in linear_entries:
        metrics = entry.get("metrics", {})
        lines.append(
            "| {name} | {status} | {converged} | {backend} | {setup} | {solve} | {total} | {iters} | {resid} |".format(
                name=entry["name"],
                status=entry["status"],
                converged="yes" if _linear_converged(entry, tolerance=LINEAR_TOL) else "no",
                backend=metrics.get("pc_backend", "-"),
                setup=("{:.3f}".format(metrics["setup_elapsed_s"]) if metrics.get("setup_elapsed_s") is not None else "-"),
                solve=("{:.3f}".format(metrics["solve_elapsed_s"]) if metrics.get("solve_elapsed_s") is not None else "-"),
                total=(
                    "{:.3f}".format(metrics["solve_plus_setup_elapsed_s"])
                    if metrics.get("solve_plus_setup_elapsed_s") is not None
                    else "{:.3f}".format(float(entry["elapsed_wall_s"]))
                ),
                iters=metrics.get("iteration_count", "-"),
                resid=("{:.3e}".format(metrics["final_relative_residual"]) if metrics.get("final_relative_residual") is not None else "-"),
            )
        )

    lines.extend(
        [
            "",
            "## Nonlinear Gate",
            "",
            "| Run | Status | Backend | Total [s] | Steps | Final lambda | Final omega |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    nonlinear_entries = [entry for entry in entries if entry["stage"] == "nonlinear"]
    for entry in nonlinear_entries:
        metrics = entry.get("metrics", {})
        lines.append(
            "| {name} | {status} | {backend} | {total} | {steps} | {lam} | {omega} |".format(
                name=entry["name"],
                status=entry["status"],
                backend=metrics.get("pc_backend", "-"),
                total=(
                    "{:.3f}".format(metrics["solve_plus_setup_elapsed_s"])
                    if metrics.get("solve_plus_setup_elapsed_s") is not None
                    else "{:.3f}".format(float(entry["elapsed_wall_s"]))
                ),
                steps=metrics.get("steps", "-"),
                lam=("{:.6f}".format(metrics["lambda_last"]) if metrics.get("lambda_last") is not None else "-"),
                omega=("{:.6f}".format(metrics["omega_last"]) if metrics.get("omega_last") is not None else "-"),
            )
        )

    lines.extend(
        [
            "",
            "## Commands",
            "",
        ]
    )
    for entry in entries:
        lines.append(f"### {entry['name']}")
        lines.append("")
        lines.append("```bash")
        lines.append(" ".join(str(part) for part in entry["command"]))
        lines.append("```")
        lines.append("")

    skipped_entries = [entry for entry in entries if entry["status"] == "skipped"]
    if skipped_entries:
        lines.extend(
            [
                "## Notes",
                "",
            ]
        )
        for entry in skipped_entries:
            reason = entry.get("reason", "-")
            lines.append(f"- `{entry['name']}`: {reason}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the P4 elastic PMG comparison suite.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--skip-nonlinear", action="store_true", default=False)
    args = parser.parse_args()

    out_root = _ensure_dir(Path(args.out_root))
    python = sys.executable
    entries: list[dict[str, object]] = []

    for spec in _frozen_specs(python, out_root):
        entries.append(_run(spec, out_root=out_root))

    pmg_linear_entries = [
        entry for entry in entries if entry["stage"] == "linear" and "pmg" in str(entry["name"])
    ]
    pmg_linear_ok = bool(pmg_linear_entries) and all(
        _linear_converged(entry, tolerance=LINEAR_TOL) for entry in pmg_linear_entries
    )
    if pmg_linear_ok and not bool(args.skip_nonlinear):
        for spec in _nonlinear_specs(python, out_root):
            entries.append(_run(spec, out_root=out_root))
    else:
        if bool(args.skip_nonlinear):
            reason = "Nonlinear stage skipped by command-line flag."
        elif not pmg_linear_entries:
            reason = "PMG linear gate produced no results."
        else:
            reason = f"PMG linear gate did not converge to <= {LINEAR_TOL:.1e}."
        entries.extend(
            {
                "name": spec.name,
                "stage": spec.stage,
                "ranks": spec.ranks,
                "status": "skipped",
                "timeout_s": spec.timeout_s,
                "elapsed_wall_s": 0.0,
                "command": list(spec.command),
                "reason": reason,
            }
            for spec in _nonlinear_specs(python, out_root)
        )

    summary = {
        "out_root": str(out_root),
        "entries": entries,
    }
    args.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_report(entries=entries, report_path=args.report_path, summary_path=args.summary_path)


if __name__ == "__main__":
    main()
