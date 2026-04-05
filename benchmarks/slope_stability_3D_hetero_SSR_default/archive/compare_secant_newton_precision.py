#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
BENCHMARK_DIR = SCRIPT_DIR if (SCRIPT_DIR / "case.toml").exists() else SCRIPT_DIR.parent
ROOT = BENCHMARK_DIR.parents[1]
CASE_TOML = BENCHMARK_DIR / "case.toml"
DEFAULT_REPORT_PATH = SCRIPT_DIR / "comparisons.md"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "comparisons" / "slope_stability_3D_hetero_SSR_default" / "secant_newton_precision_omega6p7e6"
PYTHON = ROOT / ".venv" / "bin" / "python"
SRC_DIR = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import notebook_support as nb


TOL_VARIANTS: tuple[tuple[str, str, float], ...] = (
    ("less_precise_x100", "Less precise x100", 100.0),
    ("less_precise_x10", "Less precise x10", 10.0),
    ("default", "Default", 1.0),
    ("precision_x10", "More precise x10", 0.1),
    ("precision_x100", "More precise x100", 0.01),
)
RELATIVE_CORRECTION_CASE: tuple[str, str, float] = (
    "relative_correction_1e_2",
    "Relative correction 1e-2",
    1.0e-2,
)
CASE_COLORS: dict[str, str] = {
    "less_precise_x100": "#9b2226",
    "less_precise_x10": "#d96c06",
    "default": "#0b63a3",
    "relative_correction_1e_2": "#111111",
    "precision_x10": "#3c8c3a",
    "precision_x100": "#5c4d9b",
}
DEFAULT_CASE_KEYS: tuple[str, ...] = (
    "less_precise_x100",
    "less_precise_x10",
    "default",
    "precision_x10",
    "precision_x100",
    "relative_correction_1e_2",
)


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    tol_multiplier: float
    tol: float
    stopping_criterion: str
    stopping_tol: float
    out_dir: Path
    config_path: Path


@dataclass(frozen=True)
class CaseSummary:
    key: str
    label: str
    tol_multiplier: float
    tol: float
    stopping_criterion: str
    stopping_tol: float
    runtime_seconds: float
    speedup_vs_default: float
    accepted_states: int
    continuation_steps: int
    final_lambda: float
    final_omega: float
    init_newton_total: int
    continuation_newton_total: int
    init_linear_total: int
    continuation_linear_total: int
    final_step_relres: float
    final_step_relcorr: float
    max_step_newton: float
    max_step_linear: float
    max_step_wall_time: float
    total_linear_per_newton: float
    total_constitutive_seconds: float
    total_linear_solve_seconds: float
    total_pc_apply_seconds: float
    total_pc_setup_seconds: float
    total_orthogonalization_seconds: float
    total_other_seconds: float
    plot_omega_lambda: Path
    plot_displacements: Path
    plot_deviatoric_strain: Path
    plot_step_displacement: Path
    run_dir: Path
    config_path: Path
    lambda_hist: list[float]
    omega_hist: list[float]
    step_indices: list[int]
    step_lambda: list[float]
    step_omega: list[float]
    step_newton_iterations: list[float]
    step_linear_iterations: list[float]
    step_wall_time: list[float]
    step_relres: list[float]
    step_relcorr: list[float]


def _available_case_keys() -> list[str]:
    return list(DEFAULT_CASE_KEYS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run secant continuation to omega=6.7e6 with different Newton tolerances and write comparisons.md."
    )
    parser.add_argument(
        "--omega-target",
        type=float,
        default=6.7e6,
        help="Continuation omega stop target for all runs.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Root directory for generated configs, runs, and comparison plots.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown report path to write.",
    )
    parser.add_argument(
        "--report-title",
        type=str,
        default="Secant Newton-Precision Comparison",
        help="Top-level markdown heading for the report.",
    )
    parser.add_argument(
        "--problem-elem-type",
        type=str,
        default=None,
        help="Override `[problem].elem_type` for all generated configs.",
    )
    parser.add_argument(
        "--problem-mesh-path",
        type=str,
        default=None,
        help="Override `[problem].mesh_path` for all generated configs.",
    )
    parser.add_argument(
        "--case-keys",
        nargs="+",
        default=list(DEFAULT_CASE_KEYS),
        help=f"Subset and order of comparison cases. Available: {', '.join(_available_case_keys())}.",
    )
    parser.add_argument(
        "--append-step-newton-sections",
        action="store_true",
        help="Append a per-accepted-step Newton-trace section at the end of the report.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rerun existing complete artifacts before rebuilding the report.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip solver execution and build the report from existing artifacts.",
    )
    return parser.parse_args()


def _deepcopy_sections(sections: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: copy.deepcopy(values) for name, values in sections.items()}


def _case_specs(*, base_tol: float, artifact_root: Path, case_keys: list[str] | tuple[str, ...] | None = None) -> list[CaseSpec]:
    runs_root = artifact_root / "runs"
    configs_root = artifact_root / "configs"
    all_specs: list[CaseSpec] = []
    for key, label, tol_multiplier in TOL_VARIANTS:
        tol = float(base_tol * float(tol_multiplier))
        all_specs.append(
            CaseSpec(
                key=key,
                label=label,
                tol_multiplier=float(tol_multiplier),
                tol=tol,
                stopping_criterion="relative_residual",
                stopping_tol=float(tol),
                out_dir=runs_root / key,
                config_path=configs_root / f"{key}.toml",
            )
        )
    relcorr_key, relcorr_label, relcorr_tol = RELATIVE_CORRECTION_CASE
    all_specs.append(
        CaseSpec(
            key=relcorr_key,
            label=relcorr_label,
            tol_multiplier=1.0,
            tol=float(base_tol),
            stopping_criterion="relative_correction",
            stopping_tol=float(relcorr_tol),
            out_dir=runs_root / relcorr_key,
            config_path=configs_root / f"{relcorr_key}.toml",
        )
    )
    if case_keys is None:
        return all_specs
    requested = [str(key).strip() for key in case_keys if str(key).strip()]
    requested_set = set(requested)
    available = {spec.key for spec in all_specs}
    unknown = [key for key in requested if key not in available]
    if unknown:
        raise ValueError(f"Unknown case key(s): {', '.join(unknown)}. Available: {', '.join(sorted(available))}")
    spec_by_key = {spec.key: spec for spec in all_specs}
    return [spec_by_key[key] for key in requested]


def _write_case_config(
    *,
    spec: CaseSpec,
    base_sections: dict[str, dict[str, Any]],
    materials: list[dict[str, Any]],
    omega_target: float,
    problem_elem_type: str | None = None,
    problem_mesh_path: str | None = None,
) -> None:
    sections = _deepcopy_sections(base_sections)
    sections.setdefault("problem", {})
    sections.setdefault("continuation", {})
    sections.setdefault("newton", {})
    sections.setdefault("export", {})
    if problem_elem_type is not None:
        sections["problem"]["elem_type"] = str(problem_elem_type)
    if problem_mesh_path is not None:
        sections["problem"]["mesh_path"] = str(problem_mesh_path)
    sections["continuation"]["method"] = "indirect"
    sections["continuation"]["predictor"] = "secant"
    sections["continuation"]["omega_max"] = float(omega_target)
    sections["newton"]["tol"] = float(spec.tol)
    sections["newton"]["stopping_criterion"] = str(spec.stopping_criterion)
    sections["newton"]["stopping_tol"] = float(spec.stopping_tol)
    sections["export"]["write_custom_debug_bundle"] = False
    text = nb.render_case_toml(sections, materials)
    spec.config_path.parent.mkdir(parents=True, exist_ok=True)
    spec.config_path.write_text(text, encoding="utf-8")


def _run_serial_case(*, spec: CaseSpec, force: bool) -> None:
    if force and spec.out_dir.exists():
        shutil.rmtree(spec.out_dir)
    if not force and nb.artifact_dir_complete(spec.out_dir):
        print(f"[reuse] {spec.label}: {spec.out_dir}")
        return
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        "-m",
        "slope_stability.cli.run_case_from_config",
        str(spec.config_path),
        "--out_dir",
        str(spec.out_dir),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_DIR)
    print(f"[run] {spec.label}: {' '.join(cmd)}", flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
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
    if not nb.artifact_dir_complete(spec.out_dir):
        raise RuntimeError(f"{spec.label} did not produce a complete artifact set at {spec.out_dir}")


def _to_list(array: np.ndarray) -> list[float]:
    arr = np.asarray(array)
    if arr.size == 0:
        return []
    if arr.dtype == object:
        return [float(v) for v in arr.tolist()]
    return [float(v) for v in arr.astype(np.float64).reshape(-1).tolist()]


def _to_scalar_int(array: np.ndarray) -> int:
    arr = np.asarray(array)
    if arr.size == 0:
        return 0
    return int(np.asarray(arr).reshape(-1)[0])


def _safe_ratio(numer: float, denom: float) -> float:
    if not np.isfinite(float(denom)) or abs(float(denom)) <= 1.0e-30:
        return float("nan")
    return float(numer / denom)


def _sum_finite(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.nansum(arr))


def _max_finite(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def _final_finite(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(finite[-1])


def _build_case_summary(spec: CaseSpec, *, default_runtime: float | None = None) -> CaseSummary:
    artifacts = nb.load_run_artifacts(spec.out_dir)
    npz = artifacts.npz
    run_info = dict(artifacts.run_info.get("run_info", {}))
    timings = dict(artifacts.run_info.get("timings", {}))
    constitutive = dict(timings.get("constitutive", {}))
    linear = dict(timings.get("linear", {}))

    lambda_hist = np.asarray(npz.get("lambda_hist", []), dtype=np.float64)
    omega_hist = np.asarray(npz.get("omega_hist", []), dtype=np.float64)
    step_indices = np.asarray(npz.get("stats_step_index", []), dtype=np.int64)
    step_lambda = np.asarray(npz.get("stats_step_lambda", []), dtype=np.float64)
    step_omega = np.asarray(npz.get("stats_step_omega", []), dtype=np.float64)
    step_newton = np.asarray(npz.get("stats_step_newton_iterations", []), dtype=np.float64)
    step_linear = np.asarray(npz.get("stats_step_linear_iterations", []), dtype=np.float64)
    step_wall = np.asarray(npz.get("stats_step_wall_time", []), dtype=np.float64)
    step_relres = np.asarray(npz.get("stats_step_newton_relres_end", []), dtype=np.float64)
    step_relcorr = np.asarray(npz.get("stats_step_newton_relcorr_end", []), dtype=np.float64)
    init_newton = np.asarray(npz.get("stats_init_newton_iterations", []), dtype=np.float64)
    init_linear_total = _to_scalar_int(npz.get("stats_init_linear_iterations", np.asarray([], dtype=np.float64)))

    runtime = float(run_info.get("runtime_seconds", 0.0))
    constitutive_total = float(sum(float(v) for v in constitutive.values()))
    linear_solve_total = float(linear.get("init_linear_solve_time", 0.0) + linear.get("attempt_linear_solve_time_total", 0.0))
    pc_apply_total = float(
        linear.get("init_linear_preconditioner_time", 0.0) + linear.get("attempt_linear_preconditioner_time_total", 0.0)
    )
    pc_setup_total = float(linear.get("preconditioner_setup_time_total", 0.0))
    orth_total = float(
        linear.get("init_linear_orthogonalization_time", 0.0)
        + linear.get("attempt_linear_orthogonalization_time_total", 0.0)
    )
    accounted = constitutive_total + linear_solve_total + pc_apply_total + pc_setup_total + orth_total
    other_total = float(max(runtime - accounted, 0.0))

    continuation_newton_total = int(round(_sum_finite(step_newton)))
    continuation_linear_total = int(round(_sum_finite(step_linear)))
    init_newton_total = int(round(_sum_finite(init_newton)))
    total_linear_per_newton = _safe_ratio(float(continuation_linear_total), float(continuation_newton_total))

    speedup_vs_default = 1.0 if default_runtime is None else _safe_ratio(float(default_runtime), float(runtime))
    plots_dir = spec.out_dir / "plots"
    return CaseSummary(
        key=spec.key,
        label=spec.label,
        tol_multiplier=spec.tol_multiplier,
        tol=float(spec.tol),
        stopping_criterion=str(spec.stopping_criterion),
        stopping_tol=float(spec.stopping_tol),
        runtime_seconds=runtime,
        speedup_vs_default=float(speedup_vs_default),
        accepted_states=int(lambda_hist.size),
        continuation_steps=int(step_indices.size),
        final_lambda=float(lambda_hist[-1]) if lambda_hist.size else float("nan"),
        final_omega=float(omega_hist[-1]) if omega_hist.size else float("nan"),
        init_newton_total=init_newton_total,
        continuation_newton_total=continuation_newton_total,
        init_linear_total=int(init_linear_total),
        continuation_linear_total=continuation_linear_total,
        final_step_relres=_final_finite(step_relres),
        final_step_relcorr=_final_finite(step_relcorr),
        max_step_newton=_max_finite(step_newton),
        max_step_linear=_max_finite(step_linear),
        max_step_wall_time=_max_finite(step_wall),
        total_linear_per_newton=float(total_linear_per_newton),
        total_constitutive_seconds=constitutive_total,
        total_linear_solve_seconds=linear_solve_total,
        total_pc_apply_seconds=pc_apply_total,
        total_pc_setup_seconds=pc_setup_total,
        total_orthogonalization_seconds=orth_total,
        total_other_seconds=other_total,
        plot_omega_lambda=plots_dir / "petsc_omega_lambda.png",
        plot_displacements=plots_dir / "petsc_displacements_3D.png",
        plot_deviatoric_strain=plots_dir / "petsc_deviatoric_strain_3D.png",
        plot_step_displacement=plots_dir / "petsc_step_displacement.png",
        run_dir=spec.out_dir,
        config_path=spec.config_path,
        lambda_hist=_to_list(lambda_hist),
        omega_hist=_to_list(omega_hist),
        step_indices=[int(v) for v in step_indices.tolist()],
        step_lambda=_to_list(step_lambda),
        step_omega=_to_list(step_omega),
        step_newton_iterations=_to_list(step_newton),
        step_linear_iterations=_to_list(step_linear),
        step_wall_time=_to_list(step_wall),
        step_relres=_to_list(step_relres),
        step_relcorr=_to_list(step_relcorr),
    )


def _plot_histories(cases: list[CaseSummary], *, plots_dir: Path) -> dict[str, Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    colors = CASE_COLORS
    out: dict[str, Path] = {}

    def plot_state_series(filename: str, title: str, ylabel: str, getter):
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
        for case in cases:
            x = np.arange(1, len(getter(case)) + 1, dtype=np.int64)
            y = np.asarray(getter(case), dtype=np.float64)
            ax.plot(x, y, marker="o", linewidth=1.5, label=case.label, color=colors.get(case.key))
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
            color=colors.get(case.key),
        )
    ax.set_xlabel(r"$\omega$ [$10^6$]")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(r"Continuation Curve to $\omega = 6.7 \times 10^6$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    lambda_omega = plots_dir / "lambda_omega_overlay.png"
    fig.savefig(lambda_omega)
    plt.close(fig)
    out["lambda_omega_overlay.png"] = lambda_omega

    plot_state_series("lambda_vs_state.png", "Lambda by Accepted State", r"$\lambda$", lambda case: case.lambda_hist)
    plot_state_series("omega_vs_state.png", "Omega by Accepted State", r"$\omega$", lambda case: case.omega_hist)

    def plot_step_series(filename: str, title: str, ylabel: str, getter, *, logy: bool = False):
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
        for case in cases:
            x = np.asarray(case.step_indices, dtype=np.int64)
            y = np.asarray(getter(case), dtype=np.float64)
            if x.size == 0 or y.size == 0:
                continue
            ax.plot(x, y, marker="o", linewidth=1.5, label=case.label, color=colors.get(case.key))
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
        ax.plot(x, y, marker="o", linewidth=1.5, label=case.label, color=colors.get(case.key))
    ax.set_xlabel("Accepted continuation step")
    ax.set_ylabel("Linear / Newton")
    ax.set_title("Linear Iterations per Newton Iteration")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    linear_per_newton = plots_dir / "step_linear_per_newton.png"
    fig.savefig(linear_per_newton)
    plt.close(fig)
    out["step_linear_per_newton.png"] = linear_per_newton

    labels = [case.label for case in cases]
    runtimes = [case.runtime_seconds for case in cases]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    bars = ax.bar(labels, runtimes, color=[colors.get(case.key) for case in cases])
    ax.set_ylabel("Seconds")
    ax.set_title("Runtime by Newton Stopping Case")
    for bar, value in zip(bars, runtimes, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    runtime_plot = plots_dir / "runtime_by_case.png"
    fig.savefig(runtime_plot)
    plt.close(fig)
    out["runtime_by_case.png"] = runtime_plot

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)
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
    timing_plot = plots_dir / "timing_breakdown_stacked.png"
    fig.savefig(timing_plot)
    plt.close(fig)
    out["timing_breakdown_stacked.png"] = timing_plot

    return out


def _rel(path: Path, report_path: Path) -> str:
    return os.path.relpath(path, start=report_path.parent).replace(os.sep, "/")


def _format_float(value: float, *, digits: int = 6) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _format_sci(value: float) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.3e}"


def _stop_label(criterion: str) -> str:
    mode = str(criterion).strip().lower()
    if mode == "relative_correction":
        return "relative correction"
    return "relative residual"


def _lookup_step_value(case: CaseSummary, step: int, values: list[float]) -> float:
    if step not in case.step_indices:
        return float("nan")
    idx = case.step_indices.index(step)
    return float(values[idx])


def _build_step_table(
    cases: list[CaseSummary],
    *,
    title: str,
    suffix: str,
    getter,
    digits: int,
) -> list[str]:
    headers = ["Step"] + [f"{case.label} {suffix}" for case in cases]
    lines = [
        f"## {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in cases]) + " |",
    ]
    all_steps = sorted({step for case in cases for step in case.step_indices})
    for step in all_steps:
        row = [str(step)]
        for case in cases:
            row.append(_format_float(_lookup_step_value(case, step, getter(case)), digits=digits))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _build_image_table(
    *,
    cases: list[CaseSummary],
    report_path: Path,
    title: str,
    path_getter,
    alt_suffix: str,
) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| " + " | ".join(case.label for case in cases) + " |",
        "| " + " | ".join(["---" for _ in cases]) + " |",
        "| "
        + " | ".join(
            f"![{case.label} {alt_suffix}]({_rel(path_getter(case), report_path)})"
            for case in cases
        )
        + " |",
        "",
    ]
    return lines


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _extract_successful_step_traces(case: CaseSummary) -> dict[int, dict[str, Any]]:
    artifacts = nb.load_run_artifacts(case.run_dir)
    events = list(artifacts.progress_events)
    step_accept_by_step: dict[int, dict[str, Any]] = {}
    success_attempt_by_step: dict[int, dict[str, Any]] = {}
    newton_events_by_key: dict[tuple[int, int], list[dict[str, Any]]] = {}

    for event in events:
        kind = str(event.get("event", ""))
        if kind == "step_accepted":
            step = _coerce_int(event.get("accepted_step"), -1)
            if step >= 1:
                step_accept_by_step[step] = event
            continue
        if kind == "attempt_complete":
            step = _coerce_int(event.get("target_step"), -1)
            if step >= 1 and bool(event.get("success", False)):
                success_attempt_by_step[step] = event
            continue
        if kind != "newton_iteration":
            continue
        step = _coerce_int(event.get("target_step"), -1)
        attempt_in_step = _coerce_int(event.get("attempt_in_step"), 1)
        if step < 1:
            continue
        newton_events_by_key.setdefault((step, attempt_in_step), []).append(event)

    traces: dict[int, dict[str, Any]] = {}
    for step in sorted(step_accept_by_step):
        step_accept = step_accept_by_step[step]
        attempt_complete = success_attempt_by_step.get(step)
        attempt_in_step = (
            _coerce_int(attempt_complete.get("attempt_in_step"), 1) if attempt_complete is not None else None
        )
        if attempt_in_step is None or (step, attempt_in_step) not in newton_events_by_key:
            available_attempts = sorted(attempt for (target_step, attempt) in newton_events_by_key if target_step == step)
            if not available_attempts:
                continue
            attempt_in_step = int(available_attempts[-1])
            if attempt_complete is None:
                attempt_complete = next(
                    (
                        event
                        for event in events
                        if str(event.get("event", "")) == "attempt_complete"
                        and _coerce_int(event.get("target_step"), -1) == step
                        and _coerce_int(event.get("attempt_in_step"), 1) == attempt_in_step
                    ),
                    None,
                )

        step_events = sorted(
            newton_events_by_key.get((step, attempt_in_step), []),
            key=lambda event: _coerce_int(event.get("iteration"), 0),
        )
        if not step_events:
            continue

        traces[step] = {
            "step": int(step),
            "key": case.key,
            "label": case.label,
            "attempt_in_step": int(attempt_in_step),
            "iterations": [_coerce_int(event.get("iteration"), 0) for event in step_events],
            "criterion": [_coerce_float(event.get("criterion")) for event in step_events],
            "rel_residual": [_coerce_float(event.get("rel_residual")) for event in step_events],
            "lambda": [_coerce_float(event.get("lambda_value")) for event in step_events],
            "delta_lambda": [_coerce_float(event.get("delta_lambda")) for event in step_events],
            "accepted_correction_norm": [_coerce_float(event.get("accepted_correction_norm")) for event in step_events],
            "accepted_relative_correction_norm": [
                _coerce_float(event.get("accepted_relative_correction_norm")) for event in step_events
            ],
            "final_lambda": _coerce_float(step_accept.get("lambda_value")),
            "final_omega": _coerce_float(step_accept.get("omega_value")),
            "step_wall_time": _coerce_float(step_accept.get("step_wall_time")),
            "newton_iterations": (
                _coerce_int(step_accept.get("step_newton_iterations"), 0)
                if step_accept.get("step_newton_iterations") is not None
                else _coerce_int(
                    None if attempt_complete is None else attempt_complete.get("newton_iterations"),
                    len(step_events),
                )
            ),
            "final_relres": _coerce_float(
                step_accept.get(
                    "step_newton_relres_end",
                    None if attempt_complete is None else attempt_complete.get("newton_relres_end"),
                )
            ),
            "final_relcorr": _coerce_float(
                step_accept.get(
                    "step_newton_relcorr_end",
                    None if attempt_complete is None else attempt_complete.get("newton_relcorr_end"),
                )
            ),
        }
    return traces


def _plot_step_iteration_series(
    traces: list[dict[str, Any]],
    *,
    plots_dir: Path,
    filename: str,
    ylabel: str,
    title: str,
    value_getter,
    logy: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    for trace in traces:
        x = np.asarray(trace["iterations"], dtype=np.int64)
        y = np.asarray(value_getter(trace), dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if logy:
            mask &= y > 0.0
        if not np.any(mask):
            continue
        ax.plot(
            x[mask],
            y[mask],
            marker="o",
            linewidth=1.5,
            label=trace["label"],
            color=CASE_COLORS.get(trace["key"]),
        )
    ax.set_xlabel("Newton iteration")
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
    return path


def _plot_step_xy(
    traces: list[dict[str, Any]],
    *,
    plots_dir: Path,
    filename: str,
    xlabel: str,
    ylabel: str,
    title: str,
    x_getter,
    y_getter,
    logx: bool = False,
    logy: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    for trace in traces:
        x = np.asarray(x_getter(trace), dtype=np.float64)
        y = np.asarray(y_getter(trace), dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if logx:
            mask &= x > 0.0
        if logy:
            mask &= y > 0.0
        if not np.any(mask):
            continue
        ax.plot(
            x[mask],
            y[mask],
            marker="o",
            linewidth=1.5,
            label=trace["label"],
            color=CASE_COLORS.get(trace["key"]),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = plots_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def _build_step_newton_sections(cases: list[CaseSummary], *, artifact_root: Path) -> list[dict[str, Any]]:
    traces_by_case = {case.key: _extract_successful_step_traces(case) for case in cases}
    all_steps = sorted({step for step_traces in traces_by_case.values() for step in step_traces})
    plots_root = artifact_root / "report" / "plots" / "newton_by_step"
    sections: list[dict[str, Any]] = []

    for step in all_steps:
        step_traces = [traces_by_case[case.key][step] for case in cases if step in traces_by_case[case.key]]
        if not step_traces:
            continue
        step_dir = plots_root / f"step_{step:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        plots = {
            "criterion.png": _plot_step_iteration_series(
                step_traces,
                plots_dir=step_dir,
                filename="criterion.png",
                ylabel="Criterion",
                title=f"Accepted Step {step}: Criterion",
                value_getter=lambda trace: trace["criterion"],
                logy=True,
            ),
            "lambda.png": _plot_step_iteration_series(
                step_traces,
                plots_dir=step_dir,
                filename="lambda.png",
                ylabel=r"$\lambda$",
                title=f"Accepted Step {step}: Lambda",
                value_getter=lambda trace: trace["lambda"],
            ),
            "delta_lambda.png": _plot_step_iteration_series(
                step_traces,
                plots_dir=step_dir,
                filename="delta_lambda.png",
                ylabel=r"$|\Delta \lambda|$",
                title=f"Accepted Step {step}: Absolute Delta Lambda",
                value_getter=lambda trace: np.abs(np.asarray(trace["delta_lambda"], dtype=np.float64)),
                logy=True,
            ),
            "delta_u.png": _plot_step_iteration_series(
                step_traces,
                plots_dir=step_dir,
                filename="delta_u.png",
                ylabel=r"$||\alpha \Delta U||$",
                title=f"Accepted Step {step}: Newton Correction Norm",
                value_getter=lambda trace: trace["accepted_correction_norm"],
                logy=True,
            ),
            "delta_u_over_u.png": _plot_step_iteration_series(
                step_traces,
                plots_dir=step_dir,
                filename="delta_u_over_u.png",
                ylabel=r"$||\alpha \Delta U|| / ||U||$",
                title=f"Accepted Step {step}: Relative Newton Correction",
                value_getter=lambda trace: trace["accepted_relative_correction_norm"],
                logy=True,
            ),
            "correction_norm_vs_lambda.png": _plot_step_xy(
                step_traces,
                plots_dir=step_dir,
                filename="correction_norm_vs_lambda.png",
                xlabel=r"$\lambda$",
                ylabel=r"$||\alpha \Delta U||$",
                title=f"Accepted Step {step}: Correction Norm vs Lambda",
                x_getter=lambda trace: trace["lambda"],
                y_getter=lambda trace: trace["accepted_correction_norm"],
                logy=True,
            ),
            "correction_norm_vs_criterion.png": _plot_step_xy(
                step_traces,
                plots_dir=step_dir,
                filename="correction_norm_vs_criterion.png",
                xlabel="Criterion",
                ylabel=r"$||\alpha \Delta U||$",
                title=f"Accepted Step {step}: Correction Norm vs Criterion",
                x_getter=lambda trace: trace["criterion"],
                y_getter=lambda trace: trace["accepted_correction_norm"],
                logx=True,
                logy=True,
            ),
            "lambda_vs_criterion.png": _plot_step_xy(
                step_traces,
                plots_dir=step_dir,
                filename="lambda_vs_criterion.png",
                xlabel="Criterion",
                ylabel=r"$\lambda$",
                title=f"Accepted Step {step}: Lambda vs Criterion",
                x_getter=lambda trace: trace["criterion"],
                y_getter=lambda trace: trace["lambda"],
                logx=True,
            ),
            "relative_increment_vs_lambda.png": _plot_step_xy(
                step_traces,
                plots_dir=step_dir,
                filename="relative_increment_vs_lambda.png",
                xlabel=r"$\lambda$",
                ylabel=r"$||\alpha \Delta U|| / ||U||$",
                title=f"Accepted Step {step}: Relative Correction vs Lambda",
                x_getter=lambda trace: trace["lambda"],
                y_getter=lambda trace: trace["accepted_relative_correction_norm"],
                logy=True,
            ),
            "relative_increment_vs_criterion.png": _plot_step_xy(
                step_traces,
                plots_dir=step_dir,
                filename="relative_increment_vs_criterion.png",
                xlabel="Criterion",
                ylabel=r"$||\alpha \Delta U|| / ||U||$",
                title=f"Accepted Step {step}: Relative Correction vs Criterion",
                x_getter=lambda trace: trace["criterion"],
                y_getter=lambda trace: trace["accepted_relative_correction_norm"],
                logx=True,
                logy=True,
            ),
        }
        sections.append({"step": int(step), "traces": step_traces, "plots": plots})

    return sections


def _build_plot_grid(
    *,
    report_path: Path,
    cells: list[tuple[str, Path | None]],
    columns: int,
) -> list[str]:
    lines: list[str] = []
    for start in range(0, len(cells), columns):
        chunk = list(cells[start : start + columns])
        while len(chunk) < columns:
            chunk.append(("", None))
        titles = [title for title, _ in chunk]
        images = [
            "" if path is None else f"![{title}]({_rel(path, report_path)})"
            for title, path in chunk
        ]
        lines.append("| " + " | ".join(titles) + " |")
        lines.append("| " + " | ".join(["---" for _ in chunk]) + " |")
        lines.append("| " + " | ".join(images) + " |")
        lines.append("")
    return lines


def _residual_sweep_line(cases: list[CaseSummary], *, base_r_min: float) -> str:
    residual_cases = [case for case in cases if str(case.stopping_criterion).strip().lower() == "relative_residual"]
    if not residual_cases:
        return f"- Residual `r_min` fixed at `{base_r_min:.1e}`."
    parts: list[str] = []
    for case in residual_cases:
        if case.key == "default":
            parts.append(f"default (`{case.tol:.1e}`)")
        elif case.tol_multiplier > 1.0:
            parts.append(f"looser by `{case.tol_multiplier:.0f}x` (`{case.tol:.1e}`)")
        else:
            factor = 1.0 / float(case.tol_multiplier)
            parts.append(f"tighter by `{factor:.0f}x` (`{case.tol:.1e}`)")
    return f"- Residual-tolerance sweep: {', '.join(parts)}, with `r_min` fixed at `{base_r_min:.1e}`."


def _write_report(
    *,
    report_path: Path,
    report_title: str,
    problem_description: str,
    problem_override_lines: list[str],
    artifact_root: Path,
    omega_target: float,
    base_tol: float,
    base_r_min: float,
    cases: list[CaseSummary],
    comparison_plots: dict[str, Path],
    step_newton_sections: list[dict[str, Any]] | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    has_default = any(case.key == "default" for case in cases)
    speedup_header = "Speedup vs default" if has_default else "Speedup"
    lines.append(f"# {report_title}")
    lines.append("")
    lines.append(
        f"This compares {problem_description} using the standard secant predictor with the continuation stop forced to `omega = {omega_target:.1e}`."
    )
    lines.append("")
    lines.append("- Base case: `benchmarks/slope_stability_3D_hetero_SSR_default/case.toml`")
    lines.extend(problem_override_lines)
    lines.append("- Predictor: `secant`")
    lines.append(f"- Continuation stop: `omega_max = {omega_target:.1e}`")
    lines.append(_residual_sweep_line(cases, base_r_min=base_r_min))
    if any(str(case.stopping_criterion).strip().lower() == "relative_correction" for case in cases):
        lines.append(
            "- Additional case: stop on relative Newton correction `||alpha ΔU|| / ||U|| <= 1e-2` with residual `tol` kept at the default `1e-4`."
        )
    lines.append(f"- Artifact root: `{artifact_root}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Case | Residual tol | Stop criterion | Stop tol | Runtime [s] | {speedup_header} | Accepted states | Continuation steps | Final lambda | Final omega | Init Newton | Continuation Newton | Init linear | Continuation linear | Linear / Newton | Final relres | Final `ΔU/U` |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for case in cases:
        lines.append(
            "| "
            + " | ".join(
                [
                    case.label,
                    f"`{case.tol:.1e}`",
                    _stop_label(case.stopping_criterion),
                    f"`{case.stopping_tol:.1e}`",
                    _format_float(case.runtime_seconds, digits=3),
                    _format_float(case.speedup_vs_default, digits=3),
                    str(case.accepted_states),
                    str(case.continuation_steps),
                    _format_float(case.final_lambda),
                    _format_float(case.final_omega, digits=1),
                    str(case.init_newton_total),
                    str(case.continuation_newton_total),
                    str(case.init_linear_total),
                    str(case.continuation_linear_total),
                    _format_float(case.total_linear_per_newton, digits=3),
                    _format_sci(case.final_step_relres),
                    _format_sci(case.final_step_relcorr),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.extend(
        _build_step_table(
            cases,
            title="Accepted-Step Lambda",
            suffix="lambda",
            getter=lambda case: case.step_lambda,
            digits=6,
        )
    )
    lines.extend(
        _build_step_table(
            cases,
            title="Accepted-Step Omega",
            suffix="omega",
            getter=lambda case: case.step_omega,
            digits=1,
        )
    )
    lines.extend(
        _build_step_table(
            cases,
            title="Accepted-Step Newton Iterations",
            suffix="Newton",
            getter=lambda case: case.step_newton_iterations,
            digits=0,
        )
    )
    lines.extend(
        _build_step_table(
            cases,
            title="Accepted-Step Linear Iterations",
            suffix="linear",
            getter=lambda case: case.step_linear_iterations,
            digits=0,
        )
    )
    lines.extend(
        _build_step_table(
            cases,
            title="Accepted-Step Final Relative Correction",
            suffix="ΔU/U",
            getter=lambda case: case.step_relcorr,
            digits=6,
        )
    )

    lines.append("## Comparison Plots")
    lines.append("")
    comparison_titles = [
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
    ]
    for filename, title in comparison_titles:
        path = comparison_plots[filename]
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}]({_rel(path, report_path)})")
        lines.append("")

    lines.append("## Existing Per-Run Plots")
    lines.append("")
    lines.extend(
        _build_image_table(
            cases=cases,
            report_path=report_path,
            title="Continuation Curve",
            path_getter=lambda case: case.plot_omega_lambda,
            alt_suffix="omega-lambda",
        )
    )
    lines.extend(
        _build_image_table(
            cases=cases,
            report_path=report_path,
            title="Displacements",
            path_getter=lambda case: case.plot_displacements,
            alt_suffix="displacement",
        )
    )
    lines.extend(
        _build_image_table(
            cases=cases,
            report_path=report_path,
            title="Deviatoric Strain",
            path_getter=lambda case: case.plot_deviatoric_strain,
            alt_suffix="deviatoric strain",
        )
    )
    lines.extend(
        _build_image_table(
            cases=cases,
            report_path=report_path,
            title="Step Displacement History",
            path_getter=lambda case: case.plot_step_displacement,
            alt_suffix="step displacement",
        )
    )
    if step_newton_sections:
        lines.append("## Accepted-Step Newton Solves")
        lines.append("")
        lines.append(
            "Each section below overlays the successful Newton solve that produced the accepted continuation step for every case that reached that step."
        )
        lines.append("")
        for section in step_newton_sections:
            step = int(section["step"])
            traces = list(section["traces"])
            plots = dict(section["plots"])
            trace_by_key = {trace["key"]: trace for trace in traces}
            lines.append(f"### Accepted Continuation Step {step}")
            lines.append("")
            lines.append("| Case | Attempt in step | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for case in cases:
                trace = trace_by_key.get(case.key)
                if trace is None:
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                case.label,
                                "n/a",
                                "n/a",
                                "n/a",
                                "n/a",
                                "n/a",
                                "n/a",
                                "n/a",
                            ]
                        )
                        + " |"
                    )
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            case.label,
                            str(trace["attempt_in_step"]),
                            str(trace["newton_iterations"]),
                            _format_float(trace["step_wall_time"], digits=3),
                            _format_float(trace["final_lambda"]),
                            _format_float(trace["final_omega"], digits=1),
                            _format_sci(trace["final_relres"]),
                            _format_sci(trace["final_relcorr"]),
                        ]
                    )
                    + " |"
                )
            lines.append("")
            lines.extend(
                _build_plot_grid(
                    report_path=report_path,
                    columns=2,
                    cells=[
                        ("Criterion", plots["criterion.png"]),
                        ("Lambda", plots["lambda.png"]),
                        ("Abs Delta Lambda", plots["delta_lambda.png"]),
                        ("Delta U", plots["delta_u.png"]),
                        ("Delta U / U", plots["delta_u_over_u.png"]),
                    ],
                )
            )
            lines.extend(
                _build_plot_grid(
                    report_path=report_path,
                    columns=3,
                    cells=[
                        ("Delta U vs Lambda", plots["correction_norm_vs_lambda.png"]),
                        ("Delta U vs Criterion", plots["correction_norm_vs_criterion.png"]),
                        ("Lambda vs Criterion", plots["lambda_vs_criterion.png"]),
                        ("Delta U / U vs Lambda", plots["relative_increment_vs_lambda.png"]),
                        ("Delta U / U vs Criterion", plots["relative_increment_vs_criterion.png"]),
                    ],
                )
            )
    lines.append("## Run Artifacts")
    lines.append("")
    for case in cases:
        lines.append(
            f"- {case.label}: config `{_rel(case.config_path, report_path)}`, artifact `{_rel(case.run_dir, report_path)}`"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    artifact_root = args.artifact_root.resolve()
    report_path = args.report_path.resolve()
    problem_elem_type = None if args.problem_elem_type is None else str(args.problem_elem_type).strip() or None
    problem_mesh_path = None if args.problem_mesh_path is None else str(args.problem_mesh_path).strip() or None

    base_sections = nb.load_case_sections(CASE_TOML)
    materials = nb.load_case_materials(CASE_TOML)
    base_tol = float(base_sections.get("newton", {}).get("tol", 1.0e-4))
    base_r_min = float(base_sections.get("newton", {}).get("r_min", 1.0e-4))

    specs = _case_specs(base_tol=base_tol, artifact_root=artifact_root, case_keys=args.case_keys)
    for spec in specs:
        _write_case_config(
            spec=spec,
            base_sections=base_sections,
            materials=materials,
            omega_target=float(args.omega_target),
            problem_elem_type=problem_elem_type,
            problem_mesh_path=problem_mesh_path,
        )

    if not args.report_only:
        for spec in specs:
            _run_serial_case(spec=spec, force=bool(args.force))

    cases: list[CaseSummary] = []
    default_spec = next((spec for spec in specs if spec.key == "default"), None)
    default_summary = None if default_spec is None else _build_case_summary(default_spec)
    for spec in specs:
        if default_summary is not None and spec.key == "default":
            cases.append(default_summary)
        else:
            cases.append(
                _build_case_summary(
                    spec,
                    default_runtime=None if default_summary is None else default_summary.runtime_seconds,
                )
            )

    plots_dir = artifact_root / "report" / "plots"
    comparison_plots = _plot_histories(cases, plots_dir=plots_dir)
    step_newton_sections = (
        _build_step_newton_sections(cases, artifact_root=artifact_root) if bool(args.append_step_newton_sections) else []
    )

    problem_override_lines: list[str] = []
    problem_description = "the current default `slope_stability_3D_hetero_SSR_default` case"
    override_parts: list[str] = []
    if problem_elem_type is not None:
        override_parts.append(f"`elem_type = \"{problem_elem_type}\"`")
        problem_override_lines.append(f"- Problem override: `elem_type = \"{problem_elem_type}\"`")
    if problem_mesh_path is not None:
        override_parts.append(f"`mesh_path = \"{problem_mesh_path}\"`")
        problem_override_lines.append(f"- Problem override: `mesh_path = \"{problem_mesh_path}\"`")
    if override_parts:
        problem_description = (
            "the current default `slope_stability_3D_hetero_SSR_default` case with "
            + " and ".join(override_parts)
            + " and all other settings left unchanged"
        )

    summary_payload = {
        "omega_target": float(args.omega_target),
        "base_tol": base_tol,
        "base_r_min": base_r_min,
        "report_title": str(args.report_title),
        "problem_elem_type": problem_elem_type,
        "problem_mesh_path": problem_mesh_path,
        "artifact_root": str(artifact_root),
        "report_path": str(report_path),
        "cases": [asdict(case) for case in cases],
        "comparison_plots": {name: str(path) for name, path in comparison_plots.items()},
        "step_newton_sections": [
            {
                "step": int(section["step"]),
                "plots": {name: str(path) for name, path in dict(section["plots"]).items()},
                "cases": [
                    {
                        "key": trace["key"],
                        "label": trace["label"],
                        "attempt_in_step": int(trace["attempt_in_step"]),
                        "newton_iterations": int(trace["newton_iterations"]),
                        "step_wall_time": float(trace["step_wall_time"]),
                        "final_lambda": float(trace["final_lambda"]),
                        "final_omega": float(trace["final_omega"]),
                        "final_relres": float(trace["final_relres"]),
                        "final_relcorr": float(trace["final_relcorr"]),
                    }
                    for trace in list(section["traces"])
                ],
            }
            for section in step_newton_sections
        ],
    }
    summary_path = artifact_root / "report" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2, default=str), encoding="utf-8")

    _write_report(
        report_path=report_path,
        report_title=str(args.report_title),
        problem_description=problem_description,
        problem_override_lines=problem_override_lines,
        artifact_root=artifact_root,
        omega_target=float(args.omega_target),
        base_tol=base_tol,
        base_r_min=base_r_min,
        cases=cases,
        comparison_plots=comparison_plots,
        step_newton_sections=step_newton_sections,
    )
    print(f"[done] report: {report_path}")
    print(f"[done] summary: {summary_path}")


if __name__ == "__main__":
    main()
