#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
BENCHMARK_DIR = SCRIPT_DIR if (SCRIPT_DIR / "case.toml").exists() else SCRIPT_DIR.parent
ROOT = BENCHMARK_DIR.parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
SRC_DIR = ROOT / "src"
MESH_DIR = (ROOT / "meshes" / "3d_hetero_ssr").resolve()
COARSE_MESH_PATH = (MESH_DIR / "SSR_hetero_ada_L1.msh").resolve()
FINE_MESH_PATH = (MESH_DIR / "SSR_hetero_ada_L2.msh").resolve()
HELPERS_PATH = SCRIPT_DIR / "compare_secant_newton_precision.py"

BASELINE_ARTIFACT_ROOT = (
    ROOT
    / "artifacts"
    / "comparisons"
    / "slope_stability_3D_hetero_SSR_default"
    / "pmg_levels_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6"
)
BASELINE_RUN_DIR = BASELINE_ARTIFACT_ROOT / "runs" / "p4_l2"
BASELINE_CONFIG_PATH = BASELINE_ARTIFACT_ROOT / "commands" / "p4_l2.json"

DEFAULT_REPORT_PATH = SCRIPT_DIR / "comparisons_p4_l2_pmg_p2_intermediate.md"
DEFAULT_ARTIFACT_ROOT = (
    ROOT
    / "artifacts"
    / "comparisons"
    / "slope_stability_3D_hetero_SSR_default"
    / "p4_l2_pmg_p2_intermediate_abs_dlambda_1e4_initrelcorr_1e3_omega6p7e6"
)

P4_PETSC_OPTIONS: tuple[str, ...] = (
    "pc_hypre_boomeramg_max_iter=4",
    "pc_hypre_boomeramg_tol=0.0",
)

CASE_COLORS: dict[str, str] = {
    "current": "#6a4c93",
    "with_p2_intermediate": "#0b63a3",
}

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import notebook_support as nb

_HELPERS_SPEC = importlib.util.spec_from_file_location("compare_secant_newton_precision_helpers", HELPERS_PATH)
if _HELPERS_SPEC is None or _HELPERS_SPEC.loader is None:
    raise RuntimeError(f"Failed to load comparison helpers from {HELPERS_PATH}")
helpers = importlib.util.module_from_spec(_HELPERS_SPEC)
sys.modules[_HELPERS_SPEC.name] = helpers
_HELPERS_SPEC.loader.exec_module(helpers)
helpers.CASE_COLORS.update(CASE_COLORS)


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    hierarchy: str
    mesh_path: Path
    elem_type: str
    node_ordering: str
    coarse_mesh_path: Path | None
    fine_hierarchy_mode: str
    out_dir: Path
    config_path: Path
    petsc_opts: tuple[str, ...]
    tol_multiplier: float = 1.0
    tol: float = 1.0e-4
    stopping_criterion: str = "absolute_delta_lambda"
    stopping_tol: float = 1.0e-4
    d_lambda_init: float = 0.1
    init_stopping_criterion: str = "relative_correction"
    init_stopping_tol: float = 1.0e-3
    max_deflation_basis_vectors: int = 48
    reusable_only: bool = False


@dataclass(frozen=True)
class CaseDetails:
    summary: helpers.CaseSummary
    hierarchy: str
    unknowns: int
    mesh_nodes: int
    mesh_elements: int
    manualmg_levels: int
    manualmg_level_orders: list[int]
    manualmg_level_global_sizes: list[int]
    manualmg_coarse_operator_source: str
    manualmg_coarse_ksp_type: str
    manualmg_coarse_pc_type: str
    manualmg_coarse_hypre_type: str
    manualmg_fine_ksp_type: str
    manualmg_fine_pc_type: str
    manualmg_mid_ksp_type: str
    manualmg_mid_pc_type: str
    init_linear_solve_time: float
    init_pc_apply_time: float
    init_orthogonalization_time: float
    continuation_linear_solve_time: float
    continuation_pc_apply_time: float
    continuation_orthogonalization_time: float
    pc_setup_total: float
    constitutive_tangent_time: float
    constitutive_force_time: float
    constitutive_other_time: float
    attempt_count: int
    successful_attempt_count: int
    preconditioner_rebuild_count: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current mixed P4(L2) PMG against a 4-level hierarchy "
            "P4(L2) -> P2(L2) -> P1(L2) -> P1(L1)."
        )
    )
    parser.add_argument("--omega-target", type=float, default=6.7e6)
    parser.add_argument("--mpi-ranks", type=int, default=8)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--report-title",
        type=str,
        default="P4(L2) PMG Comparison: Current vs P2-Intermediate Hierarchy",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def _default_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def _case_specs(*, artifact_root: Path) -> list[CaseSpec]:
    runs_root = artifact_root / "runs"
    commands_root = artifact_root / "commands"
    return [
        CaseSpec(
            key="current",
            label="Current P4(L2)",
            hierarchy="P4(L2) -> P1(L2) -> P1(L1)",
            mesh_path=FINE_MESH_PATH,
            elem_type="P4",
            node_ordering="block_metis",
            coarse_mesh_path=COARSE_MESH_PATH,
            fine_hierarchy_mode="default",
            out_dir=BASELINE_RUN_DIR,
            config_path=BASELINE_CONFIG_PATH,
            petsc_opts=P4_PETSC_OPTIONS,
            reusable_only=True,
        ),
        CaseSpec(
            key="with_p2_intermediate",
            label="P4(L2) with P2(L2) Intermediate",
            hierarchy="P4(L2) -> P2(L2) -> P1(L2) -> P1(L1)",
            mesh_path=FINE_MESH_PATH,
            elem_type="P4",
            node_ordering="block_metis",
            coarse_mesh_path=COARSE_MESH_PATH,
            fine_hierarchy_mode="p4_p2_intermediate",
            out_dir=runs_root / "p4_l2_p2_intermediate",
            config_path=commands_root / "p4_l2_p2_intermediate.json",
            petsc_opts=P4_PETSC_OPTIONS,
        ),
    ]


def _case_command(*, spec: CaseSpec, omega_target: float, mpi_ranks: int) -> list[str]:
    cmd = [
        "mpirun",
        "-n",
        str(int(mpi_ranks)),
        str(PYTHON),
        "-m",
        "slope_stability.cli.run_3D_hetero_SSR_capture",
        "--out_dir",
        str(spec.out_dir),
        "--mesh_path",
        str(spec.mesh_path),
        "--elem_type",
        str(spec.elem_type),
        "--node_ordering",
        str(spec.node_ordering),
        "--omega_max_stop",
        str(float(omega_target)),
        "--solver_type",
        "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        "--pc_backend",
        "pmg_shell",
        "--pmg_coarse_mesh_path",
        str(spec.coarse_mesh_path),
        "--pmg_fine_hierarchy_mode",
        str(spec.fine_hierarchy_mode),
        "--preconditioner_matrix_source",
        "tangent",
        "--d_lambda_init",
        str(float(spec.d_lambda_init)),
        "--tol",
        str(float(spec.tol)),
        "--newton_stopping_criterion",
        str(spec.stopping_criterion),
        "--newton_stopping_tol",
        str(float(spec.stopping_tol)),
        "--init_newton_stopping_criterion",
        str(spec.init_stopping_criterion),
        "--init_newton_stopping_tol",
        str(float(spec.init_stopping_tol)),
        "--max_deflation_basis_vectors",
        str(int(spec.max_deflation_basis_vectors)),
        "--recycle_preconditioner",
    ]
    for opt in spec.petsc_opts:
        cmd.extend(["--petsc-opt", str(opt)])
    return cmd


def _artifact_set_complete(path: Path) -> bool:
    path = Path(path)
    required = (
        path / "data" / "run_info.json",
        path / "data" / "petsc_run.npz",
        path / "data" / "progress_latest.json",
        path / "plots" / "petsc_omega_lambda.png",
        path / "plots" / "petsc_displacements_3D.png",
        path / "plots" / "petsc_deviatoric_strain_3D.png",
        path / "plots" / "petsc_step_displacement.png",
    )
    return all(item.exists() for item in required)


def _write_command_manifest(*, spec: CaseSpec, omega_target: float, mpi_ranks: int) -> None:
    if spec.reusable_only:
        return
    spec.config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": spec.key,
        "label": spec.label,
        "hierarchy": spec.hierarchy,
        "mesh_path": str(spec.mesh_path),
        "elem_type": spec.elem_type,
        "node_ordering": spec.node_ordering,
        "pmg_coarse_mesh_path": None if spec.coarse_mesh_path is None else str(spec.coarse_mesh_path),
        "pmg_fine_hierarchy_mode": str(spec.fine_hierarchy_mode),
        "omega_target": float(omega_target),
        "mpi_ranks": int(mpi_ranks),
        "newton_stopping_criterion": spec.stopping_criterion,
        "newton_stopping_tol": float(spec.stopping_tol),
        "init_newton_stopping_criterion": spec.init_stopping_criterion,
        "init_newton_stopping_tol": float(spec.init_stopping_tol),
        "d_lambda_init": float(spec.d_lambda_init),
        "max_deflation_basis_vectors": int(spec.max_deflation_basis_vectors),
        "petsc_opts": list(spec.petsc_opts),
        "command": _case_command(spec=spec, omega_target=omega_target, mpi_ranks=mpi_ranks),
    }
    spec.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_case(*, spec: CaseSpec, omega_target: float, mpi_ranks: int, force: bool) -> None:
    if spec.reusable_only:
        if not _artifact_set_complete(spec.out_dir):
            raise RuntimeError(f"Baseline artifact is incomplete: {spec.out_dir}")
        print(f"[reuse] {spec.label}: {spec.out_dir}", flush=True)
        return
    if force and spec.out_dir.exists():
        shutil.rmtree(spec.out_dir)
    if not force and _artifact_set_complete(spec.out_dir):
        print(f"[reuse] {spec.label}: {spec.out_dir}", flush=True)
        return
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    cmd = _case_command(spec=spec, omega_target=omega_target, mpi_ranks=mpi_ranks)
    print(f"[run] {spec.label}: {' '.join(cmd)}", flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=_default_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{spec.key}] {line.rstrip()}", flush=True)
    return_code = int(process.wait())
    if return_code != 0:
        raise RuntimeError(f"{spec.label} failed with exit code {return_code}")
    if not _artifact_set_complete(spec.out_dir):
        raise RuntimeError(f"{spec.label} did not produce a complete artifact set at {spec.out_dir}")


def _parse_progress_counts(run_dir: Path) -> tuple[int, int]:
    progress_path = run_dir / "data" / "progress.jsonl"
    if not progress_path.exists():
        return 0, 0
    attempt_count = 0
    success_count = 0
    for raw_line in progress_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(event.get("event", "")) != "attempt_complete":
            continue
        attempt_count += 1
        if bool(event.get("success", False)):
            success_count += 1
    return attempt_count, success_count


def _load_case_details(spec: CaseSpec, *, default_runtime: float | None = None) -> CaseDetails:
    summary = helpers._build_case_summary(spec, default_runtime=default_runtime)
    artifacts = nb.load_run_artifacts(spec.out_dir)
    run_info = dict(artifacts.run_info.get("run_info", {}))
    timings = dict(artifacts.run_info.get("timings", {}))
    linear = dict(timings.get("linear", {}))
    constitutive = dict(timings.get("constitutive", {}))
    attempt_count, success_count = _parse_progress_counts(spec.out_dir)
    constitutive_other = float(
        constitutive.get("local_strain", 0.0)
        + constitutive.get("local_constitutive", 0.0)
        + constitutive.get("stress", 0.0)
        + constitutive.get("stress_tangent", 0.0)
    )
    return CaseDetails(
        summary=summary,
        hierarchy=spec.hierarchy,
        unknowns=int(run_info.get("unknowns", 0)),
        mesh_nodes=int(run_info.get("mesh_nodes", 0)),
        mesh_elements=int(run_info.get("mesh_elements", 0)),
        manualmg_levels=int(linear.get("manualmg_levels", 0)),
        manualmg_level_orders=[int(v) for v in list(linear.get("manualmg_level_orders", []))],
        manualmg_level_global_sizes=[int(v) for v in list(linear.get("manualmg_level_global_sizes", []))],
        manualmg_coarse_operator_source=str(linear.get("manualmg_coarse_operator_source", "")),
        manualmg_coarse_ksp_type=str(linear.get("manualmg_coarse_ksp_type", "")),
        manualmg_coarse_pc_type=str(linear.get("manualmg_coarse_pc_type", "")),
        manualmg_coarse_hypre_type=str(linear.get("manualmg_coarse_hypre_type", "")),
        manualmg_fine_ksp_type=str(linear.get("manualmg_fine_ksp_type", "")),
        manualmg_fine_pc_type=str(linear.get("manualmg_fine_pc_type", "")),
        manualmg_mid_ksp_type=str(linear.get("manualmg_mid_ksp_type", "")),
        manualmg_mid_pc_type=str(linear.get("manualmg_mid_pc_type", "")),
        init_linear_solve_time=float(linear.get("init_linear_solve_time", 0.0)),
        init_pc_apply_time=float(linear.get("init_linear_preconditioner_time", 0.0)),
        init_orthogonalization_time=float(linear.get("init_linear_orthogonalization_time", 0.0)),
        continuation_linear_solve_time=float(linear.get("attempt_linear_solve_time_total", 0.0)),
        continuation_pc_apply_time=float(linear.get("attempt_linear_preconditioner_time_total", 0.0)),
        continuation_orthogonalization_time=float(linear.get("attempt_linear_orthogonalization_time_total", 0.0)),
        pc_setup_total=float(linear.get("preconditioner_setup_time_total", 0.0)),
        constitutive_tangent_time=float(constitutive.get("build_tangent_local", 0.0)),
        constitutive_force_time=float(constitutive.get("build_F", 0.0)),
        constitutive_other_time=constitutive_other,
        attempt_count=int(attempt_count),
        successful_attempt_count=int(success_count),
        preconditioner_rebuild_count=int(linear.get("preconditioner_rebuild_count", 0)),
    )


def _plot_histories(cases: list[helpers.CaseSummary], *, plots_dir: Path) -> dict[str, Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    def plot_state_series(filename: str, title: str, ylabel: str, getter) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
        for case in cases:
            x = np.arange(1, len(getter(case)) + 1, dtype=np.int64)
            y = np.asarray(getter(case), dtype=np.float64)
            ax.plot(x, y, marker="o", linewidth=1.5, label=case.label, color=CASE_COLORS.get(case.key))
        ax.set_xlabel("Accepted state")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = plots_dir / filename
        fig.savefig(path)
        plt.close(fig)
        out[filename] = path

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    for case in cases:
        ax.plot(
            np.asarray(case.omega_hist, dtype=np.float64) / 1.0e6,
            np.asarray(case.lambda_hist, dtype=np.float64),
            marker="o",
            linewidth=1.5,
            label=case.label,
            color=CASE_COLORS.get(case.key),
        )
    ax.set_xlabel(r"$\omega$ [$10^6$]")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(r"Continuation Curve to $\omega = 6.7 \times 10^6$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "lambda_omega_overlay.png"
    fig.savefig(path)
    plt.close(fig)
    out["lambda_omega_overlay.png"] = path

    plot_state_series("lambda_vs_state.png", "Lambda by Accepted State", r"$\lambda$", lambda case: case.lambda_hist)
    plot_state_series("omega_vs_state.png", "Omega by Accepted State", r"$\omega$", lambda case: case.omega_hist)

    def plot_step_series(filename: str, title: str, ylabel: str, getter, *, logy: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
        for case in cases:
            x = np.asarray(case.step_indices, dtype=np.int64)
            y = np.asarray(getter(case), dtype=np.float64)
            if x.size == 0 or y.size == 0:
                continue
            ax.plot(x, y, marker="o", linewidth=1.5, label=case.label, color=CASE_COLORS.get(case.key))
        ax.set_xlabel("Accepted continuation step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = plots_dir / filename
        fig.savefig(path)
        plt.close(fig)
        out[filename] = path

    plot_step_series(
        "step_newton_iterations.png",
        "Newton Iterations per Accepted Step",
        "Newton iterations",
        lambda case: case.step_newton_iterations,
    )
    plot_step_series(
        "step_linear_iterations.png",
        "Linear Iterations per Accepted Step",
        "Linear iterations",
        lambda case: case.step_linear_iterations,
    )
    plot_step_series(
        "step_wall_time.png",
        "Wall Time per Accepted Step",
        "Seconds",
        lambda case: case.step_wall_time,
    )
    plot_step_series(
        "step_relres_end.png",
        "Final Newton Relative Residual per Accepted Step",
        "Relative residual",
        lambda case: case.step_relres,
        logy=True,
    )
    plot_step_series(
        "step_relcorr_end.png",
        "Final Newton Relative Correction per Accepted Step",
        r"$||\alpha \Delta U|| / ||U||$",
        lambda case: case.step_relcorr,
        logy=True,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    for case in cases:
        x = np.asarray(case.step_indices, dtype=np.int64)
        numer = np.asarray(case.step_linear_iterations, dtype=np.float64)
        denom = np.asarray(case.step_newton_iterations, dtype=np.float64)
        y = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=np.abs(denom) > 0.0)
        if x.size == 0 or y.size == 0:
            continue
        ax.plot(x, y, marker="o", linewidth=1.5, label=case.label, color=CASE_COLORS.get(case.key))
    ax.set_xlabel("Accepted continuation step")
    ax.set_ylabel("Linear / Newton")
    ax.set_title("Linear Iterations per Newton Iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / "step_linear_per_newton.png"
    fig.savefig(path)
    plt.close(fig)
    out["step_linear_per_newton.png"] = path

    labels = [case.label for case in cases]
    runtimes = [case.runtime_seconds for case in cases]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)
    bars = ax.bar(labels, runtimes, color=[CASE_COLORS.get(case.key) for case in cases])
    ax.set_ylabel("Seconds")
    ax.set_title("Runtime by Case")
    for bar, value in zip(bars, runtimes, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = plots_dir / "runtime_by_case.png"
    fig.savefig(path)
    plt.close(fig)
    out["runtime_by_case.png"] = path

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    stacks = {
        "Constitutive": np.asarray([case.total_constitutive_seconds for case in cases], dtype=np.float64),
        "Linear solve": np.asarray([case.total_linear_solve_seconds for case in cases], dtype=np.float64),
        "PC apply": np.asarray([case.total_pc_apply_seconds for case in cases], dtype=np.float64),
        "PC setup": np.asarray([case.total_pc_setup_seconds for case in cases], dtype=np.float64),
        "Orthogonalization": np.asarray([case.total_orthogonalization_seconds for case in cases], dtype=np.float64),
        "Other": np.asarray([case.total_other_seconds for case in cases], dtype=np.float64),
    }
    bottom = np.zeros(len(cases), dtype=np.float64)
    palette = {
        "Constitutive": "#3465a4",
        "Linear solve": "#cc7000",
        "PC apply": "#73a857",
        "PC setup": "#99582a",
        "Orthogonalization": "#7b6db4",
        "Other": "#7f7f7f",
    }
    for name, values in stacks.items():
        ax.bar(labels, values, bottom=bottom, label=name, color=palette[name])
        bottom += values
    ax.set_ylabel("Seconds")
    ax.set_title("Runtime Breakdown")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = plots_dir / "timing_breakdown_stacked.png"
    fig.savefig(path)
    plt.close(fig)
    out["timing_breakdown_stacked.png"] = path
    return out


def _write_report(
    *,
    report_path: Path,
    report_title: str,
    artifact_root: Path,
    omega_target: float,
    mpi_ranks: int,
    details: list[CaseDetails],
    comparison_plots: dict[str, Path],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cases = [detail.summary for detail in details]
    baseline = details[0]
    candidate = details[1]
    runtime_ratio = candidate.summary.runtime_seconds / baseline.summary.runtime_seconds
    lambda_delta = candidate.summary.final_lambda - baseline.summary.final_lambda
    linear_delta = candidate.summary.continuation_linear_total - baseline.summary.continuation_linear_total
    lines: list[str] = [
        f"# {report_title}",
        "",
        "This compares the existing mixed `P4(L2)` PMG hierarchy against a rerun that inserts a same-mesh `P2(L2)` intermediate level.",
        "",
        "- Common benchmark base: `benchmarks/slope_stability_3D_hetero_SSR_default/case.toml`",
        f"- Common stop target: `omega = {omega_target:.1e}`",
        "- Common continuation Newton stop: `|Δlambda| < 1e-4`",
        "- Common init Newton stop: `relative correction < 1e-3`",
        "- Common `d_lambda_init = 0.1`",
        "- Deflation: on (`max_deflation_basis_vectors = 48`)",
        f"- MPI ranks: `{int(mpi_ranks)}`",
        "- PETSc opts: " + ", ".join(f"`{opt}`" for opt in P4_PETSC_OPTIONS),
        f"- Artifact root: `{artifact_root}`",
        "",
        "## Headline",
        "",
        f"- Runtime ratio (`new/current`): `{runtime_ratio:.3f}x`",
        f"- Final lambda shift (`new - current`): `{lambda_delta:+.6f}`",
        f"- Continuation linear-iteration shift (`new - current`): `{linear_delta:+d}`",
        "",
        "## Summary",
        "",
        "| Case | Hierarchy | Unknowns | Runtime [s] | Accepted states | Continuation steps | Final lambda | Final omega | Init Newton | Continuation Newton | Continuation linear | Linear / Newton | Final relres | Final `ΔU/U` |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for detail in details:
        case = detail.summary
        lines.append(
            "| "
            + " | ".join(
                [
                    case.label,
                    detail.hierarchy,
                    str(detail.unknowns),
                    helpers._format_float(case.runtime_seconds, digits=3),
                    str(case.accepted_states),
                    str(case.continuation_steps),
                    helpers._format_float(case.final_lambda),
                    helpers._format_float(case.final_omega, digits=1),
                    str(case.init_newton_total),
                    str(case.continuation_newton_total),
                    str(case.continuation_linear_total),
                    helpers._format_float(case.total_linear_per_newton, digits=3),
                    helpers._format_sci(case.final_step_relres),
                    helpers._format_sci(case.final_step_relcorr),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.extend(
        helpers._build_step_table(
            cases,
            title="Accepted-Step Lambda",
            suffix="lambda",
            getter=lambda case: case.step_lambda,
            digits=6,
        )
    )
    lines.extend(
        helpers._build_step_table(
            cases,
            title="Accepted-Step Omega",
            suffix="omega",
            getter=lambda case: case.step_omega,
            digits=1,
        )
    )
    lines.extend(
        helpers._build_step_table(
            cases,
            title="Accepted-Step Newton Iterations",
            suffix="Newton",
            getter=lambda case: case.step_newton_iterations,
            digits=0,
        )
    )
    lines.extend(
        helpers._build_step_table(
            cases,
            title="Accepted-Step Linear Iterations",
            suffix="linear",
            getter=lambda case: case.step_linear_iterations,
            digits=0,
        )
    )
    lines.extend(
        helpers._build_step_table(
            cases,
            title="Accepted-Step Final Relative Correction",
            suffix="ΔU/U",
            getter=lambda case: case.step_relcorr,
            digits=6,
        )
    )
    lines.append("## Timing Totals")
    lines.append("")
    lines.append(
        "| Case | Constitutive [s] | Linear solve [s] | PC apply [s] | PC setup [s] | Orthogonalization [s] | Other [s] |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for detail in details:
        case = detail.summary
        lines.append(
            "| "
            + " | ".join(
                [
                    case.label,
                    helpers._format_float(case.total_constitutive_seconds, digits=3),
                    helpers._format_float(case.total_linear_solve_seconds, digits=3),
                    helpers._format_float(case.total_pc_apply_seconds, digits=3),
                    helpers._format_float(case.total_pc_setup_seconds, digits=3),
                    helpers._format_float(case.total_orthogonalization_seconds, digits=3),
                    helpers._format_float(case.total_other_seconds, digits=3),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## PMG Layout")
    lines.append("")
    lines.append(
        "| Case | ManualMG levels | Level orders | Level global sizes | Coarse operator | Coarse KSP/PC/Hypre | Fine smoother | Mid smoother |"
    )
    lines.append("| --- | ---: | --- | --- | --- | --- | --- | --- |")
    for detail in details:
        coarse_desc = "/".join(
            value
            for value in [
                detail.manualmg_coarse_ksp_type,
                detail.manualmg_coarse_pc_type,
                detail.manualmg_coarse_hypre_type,
            ]
            if value
        )
        fine_desc = "/".join(value for value in [detail.manualmg_fine_ksp_type, detail.manualmg_fine_pc_type] if value)
        mid_desc = "/".join(value for value in [detail.manualmg_mid_ksp_type, detail.manualmg_mid_pc_type] if value)
        lines.append(
            "| "
            + " | ".join(
                [
                    detail.summary.label,
                    str(detail.manualmg_levels),
                    str(detail.manualmg_level_orders),
                    str(detail.manualmg_level_global_sizes),
                    detail.manualmg_coarse_operator_source or "n/a",
                    coarse_desc or "n/a",
                    fine_desc or "n/a",
                    mid_desc or "n/a",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Comparison Plots")
    lines.append("")
    for filename, title in [
        ("lambda_omega_overlay.png", "Lambda vs omega"),
        ("lambda_vs_state.png", "Lambda by accepted state"),
        ("omega_vs_state.png", "Omega by accepted state"),
        ("runtime_by_case.png", "Runtime by case"),
        ("timing_breakdown_stacked.png", "Timing breakdown"),
        ("step_newton_iterations.png", "Newton iterations per step"),
        ("step_linear_iterations.png", "Linear iterations per step"),
        ("step_linear_per_newton.png", "Linear per Newton"),
        ("step_wall_time.png", "Wall time per step"),
        ("step_relres_end.png", "Final relative residual per step"),
        ("step_relcorr_end.png", "Final relative correction per step"),
    ]:
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}]({helpers._rel(comparison_plots[filename], report_path)})")
        lines.append("")
    lines.append("## Existing Per-Run Plots")
    lines.append("")
    lines.extend(
        helpers._build_image_table(
            cases=cases,
            report_path=report_path,
            title="Continuation Curve",
            path_getter=lambda case: case.plot_omega_lambda,
            alt_suffix="omega-lambda",
        )
    )
    lines.extend(
        helpers._build_image_table(
            cases=cases,
            report_path=report_path,
            title="Displacements",
            path_getter=lambda case: case.plot_displacements,
            alt_suffix="displacement",
        )
    )
    lines.extend(
        helpers._build_image_table(
            cases=cases,
            report_path=report_path,
            title="Deviatoric Strain",
            path_getter=lambda case: case.plot_deviatoric_strain,
            alt_suffix="deviatoric strain",
        )
    )
    lines.extend(
        helpers._build_image_table(
            cases=cases,
            report_path=report_path,
            title="Step Displacement History",
            path_getter=lambda case: case.plot_step_displacement,
            alt_suffix="step displacement",
        )
    )
    lines.append("## Run Artifacts")
    lines.append("")
    for detail in details:
        lines.append(
            f"- {detail.summary.label}: command `{helpers._rel(detail.summary.config_path, report_path)}`, artifact `{helpers._rel(detail.summary.run_dir, report_path)}`"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    artifact_root = args.artifact_root.resolve()
    report_path = args.report_path.resolve()
    specs = _case_specs(artifact_root=artifact_root)
    for spec in specs:
        _write_command_manifest(spec=spec, omega_target=float(args.omega_target), mpi_ranks=int(args.mpi_ranks))

    if not args.report_only:
        for spec in specs:
            _run_case(spec=spec, omega_target=float(args.omega_target), mpi_ranks=int(args.mpi_ranks), force=bool(args.force))

    details: list[CaseDetails] = []
    default_runtime: float | None = None
    for idx, spec in enumerate(specs):
        detail = _load_case_details(spec, default_runtime=default_runtime)
        details.append(detail)
        if idx == 0:
            default_runtime = detail.summary.runtime_seconds

    cases = [detail.summary for detail in details]
    plots_dir = artifact_root / "report" / "plots"
    comparison_plots = _plot_histories(cases, plots_dir=plots_dir)

    summary_payload = {
        "report_title": str(args.report_title),
        "omega_target": float(args.omega_target),
        "mpi_ranks": int(args.mpi_ranks),
        "artifact_root": str(artifact_root),
        "report_path": str(report_path),
        "cases": [
            {
                "summary": asdict(detail.summary),
                "hierarchy": detail.hierarchy,
                "unknowns": int(detail.unknowns),
                "mesh_nodes": int(detail.mesh_nodes),
                "mesh_elements": int(detail.mesh_elements),
                "manualmg_levels": int(detail.manualmg_levels),
                "manualmg_level_orders": list(detail.manualmg_level_orders),
                "manualmg_level_global_sizes": list(detail.manualmg_level_global_sizes),
                "manualmg_coarse_operator_source": detail.manualmg_coarse_operator_source,
                "manualmg_coarse_ksp_type": detail.manualmg_coarse_ksp_type,
                "manualmg_coarse_pc_type": detail.manualmg_coarse_pc_type,
                "manualmg_coarse_hypre_type": detail.manualmg_coarse_hypre_type,
                "manualmg_fine_ksp_type": detail.manualmg_fine_ksp_type,
                "manualmg_fine_pc_type": detail.manualmg_fine_pc_type,
                "manualmg_mid_ksp_type": detail.manualmg_mid_ksp_type,
                "manualmg_mid_pc_type": detail.manualmg_mid_pc_type,
                "init_linear_solve_time": float(detail.init_linear_solve_time),
                "init_pc_apply_time": float(detail.init_pc_apply_time),
                "init_orthogonalization_time": float(detail.init_orthogonalization_time),
                "continuation_linear_solve_time": float(detail.continuation_linear_solve_time),
                "continuation_pc_apply_time": float(detail.continuation_pc_apply_time),
                "continuation_orthogonalization_time": float(detail.continuation_orthogonalization_time),
                "pc_setup_total": float(detail.pc_setup_total),
                "constitutive_tangent_time": float(detail.constitutive_tangent_time),
                "constitutive_force_time": float(detail.constitutive_force_time),
                "constitutive_other_time": float(detail.constitutive_other_time),
                "attempt_count": int(detail.attempt_count),
                "successful_attempt_count": int(detail.successful_attempt_count),
                "preconditioner_rebuild_count": int(detail.preconditioner_rebuild_count),
            }
            for detail in details
        ],
        "comparison_plots": {name: str(path) for name, path in comparison_plots.items()},
    }
    summary_path = artifact_root / "report" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")

    _write_report(
        report_path=report_path,
        report_title=str(args.report_title),
        artifact_root=artifact_root,
        omega_target=float(args.omega_target),
        mpi_ranks=int(args.mpi_ranks),
        details=details,
        comparison_plots=comparison_plots,
    )
    print(f"[done] report: {report_path}")
    print(f"[done] summary: {summary_path}")


if __name__ == "__main__":
    main()
