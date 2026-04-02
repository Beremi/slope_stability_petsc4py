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
BENCHMARK_DIR = SCRIPT_PATH.parent
ROOT = SCRIPT_PATH.parents[2]
CASE_TOML = BENCHMARK_DIR / "case.toml"
DEFAULT_REPORT_PATH = BENCHMARK_DIR / "comparisons_p4_l1_pmg.md"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "comparisons" / "3d_hetero_ssr_default" / "p4_l1_pmg_newton_stops_omega6p7e6"
PYTHON = ROOT / ".venv" / "bin" / "python"
SRC_DIR = ROOT / "src"
HELPERS_PATH = BENCHMARK_DIR / "compare_secant_newton_precision.py"
MESH_PATH = (ROOT / "meshes" / "3d_hetero_ssr" / "SSR_hetero_ada_L1.msh").resolve()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import notebook_support as nb

_HELPERS_SPEC = importlib.util.spec_from_file_location("compare_secant_newton_precision_helpers", HELPERS_PATH)
if _HELPERS_SPEC is None or _HELPERS_SPEC.loader is None:
    raise RuntimeError(f"Failed to load comparison helpers from {HELPERS_PATH}")
helpers = importlib.util.module_from_spec(_HELPERS_SPEC)
sys.modules[_HELPERS_SPEC.name] = helpers
_HELPERS_SPEC.loader.exec_module(helpers)

P4_PETSC_OPTIONS: tuple[str, ...] = (
    "pc_hypre_boomeramg_max_iter=4",
    "pc_hypre_boomeramg_tol=0.0",
)
CASE_ORDER: tuple[str, ...] = (
    "default",
    "less_precise_x100",
    "relative_correction_1e_2",
    "absolute_delta_lambda_1e_2",
    "absolute_delta_lambda_1e_3",
    "absolute_delta_lambda_1e_3_cap_initial_segment",
)
CAP_CASE_KEYS: frozenset[str] = frozenset(
    {
        "absolute_delta_lambda_1e_3_cap_initial_segment",
        "absolute_delta_lambda_1e_3_cap_initial_segment_initrelcorr_dlambda0p1",
        "hybrid_rough_fine_history_box",
        "hybrid_rough_fine_history_box_dlambda0p1_no_flat_stop",
    }
)
HYBRID_CASE_KEYS: frozenset[str] = frozenset(
    {
        "hybrid_rough_fine_history_box",
        "hybrid_rough_fine_history_box_dlambda0p1_no_flat_stop",
    }
)
CASE_COLORS: dict[str, str] = {
    "default": "#0b63a3",
    "less_precise_x100": "#d96c06",
    "relative_correction_1e_2": "#111111",
    "relative_correction_1e_2_rerun": "#5c5c5c",
    "absolute_delta_lambda_1e_2": "#7a3e9d",
    "absolute_delta_lambda_1e_3": "#2d6a4f",
    "absolute_delta_lambda_1e_3_cap_initial_segment": "#9b2226",
    "absolute_delta_lambda_1e_3_cap_initial_segment_initrelcorr_dlambda0p1": "#c84c0c",
    "absolute_delta_lambda_1e_3_initrelcorr_dlambda0p1": "#33658a",
    "absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1": "#0a9396",
    "absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1_no_deflation": "#bb3e03",
    "relative_correction_1e_3_initrelcorr_dlambda0p1": "#6a4c93",
    "hybrid_rough_fine_history_box": "#1f7a8c",
    "hybrid_rough_fine_history_box_dlambda0p1_no_flat_stop": "#0f9d58",
}
helpers.CASE_COLORS.update(CASE_COLORS)

_BASE_SECTIONS = nb.load_case_sections(CASE_TOML)
_DEFAULT_MPI_RANKS = int(_BASE_SECTIONS.get("benchmark", {}).get("mpi_ranks", 8))


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
    cli_args: tuple[str, ...] = ()
    petsc_opts: tuple[str, ...] = P4_PETSC_OPTIONS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run P4(L1) PMG secant continuation to omega=6.7e6 with multiple Newton stopping rules."
    )
    parser.add_argument("--omega-target", type=float, default=6.7e6)
    parser.add_argument("--mpi-ranks", type=int, default=_DEFAULT_MPI_RANKS)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--report-title", type=str, default="P4(L1) PMG Newton-Stop Comparison")
    parser.add_argument("--case-keys", nargs="+", default=list(CASE_ORDER))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def _case_specs(*, base_tol: float, artifact_root: Path, case_keys: list[str] | tuple[str, ...]) -> list[CaseSpec]:
    case_defs: dict[str, dict[str, object]] = {
        "default": {
            "label": "Default",
            "tol_multiplier": 1.0,
            "stopping_criterion": "relative_residual",
            "stopping_tol": float(base_tol),
            "cli_args": (),
        },
        "less_precise_x100": {
            "label": "100x less precision",
            "tol_multiplier": 100.0,
            "stopping_criterion": "relative_residual",
            "stopping_tol": float(base_tol * 100.0),
            "cli_args": (),
        },
        "relative_correction_1e_2": {
            "label": "Relative correction 1e-2",
            "tol_multiplier": 1.0,
            "stopping_criterion": "relative_correction",
            "stopping_tol": 1.0e-2,
            "cli_args": (),
        },
        "relative_correction_1e_2_rerun": {
            "label": "Relative correction 1e-2 (rerun current code)",
            "tol_multiplier": 1.0,
            "stopping_criterion": "relative_correction",
            "stopping_tol": 1.0e-2,
            "cli_args": (),
        },
        "absolute_delta_lambda_1e_2": {
            "label": "Abs Delta Lambda 1e-2",
            "tol_multiplier": 1.0,
            "stopping_criterion": "absolute_delta_lambda",
            "stopping_tol": 1.0e-2,
            "cli_args": (),
        },
        "absolute_delta_lambda_1e_3": {
            "label": "Abs Delta Lambda 1e-3",
            "tol_multiplier": 1.0,
            "stopping_criterion": "absolute_delta_lambda",
            "stopping_tol": 1.0e-3,
            "cli_args": (),
        },
        "absolute_delta_lambda_1e_3_cap_initial_segment": {
            "label": "Abs Delta Lambda 1e-3 + History-Box Cap",
            "tol_multiplier": 1.0,
            "stopping_criterion": "absolute_delta_lambda",
            "stopping_tol": 1.0e-3,
            "cli_args": ("--step_length_cap_mode", "history_box", "--step_length_cap_factor", "1.0"),
        },
        "absolute_delta_lambda_1e_3_cap_initial_segment_initrelcorr_dlambda0p1": {
            "label": "Abs Delta Lambda 1e-3 + History-Box Cap (init rel corr 1e-2, d_lambda_init=0.1)",
            "tol_multiplier": 1.0,
            "stopping_criterion": "absolute_delta_lambda",
            "stopping_tol": 1.0e-3,
            "cli_args": (
                "--d_lambda_init",
                "0.1",
                "--init_newton_stopping_criterion",
                "relative_correction",
                "--init_newton_stopping_tol",
                "1e-2",
                "--step_length_cap_mode",
                "history_box",
                "--step_length_cap_factor",
                "1.0",
            ),
        },
        "absolute_delta_lambda_1e_3_initrelcorr_dlambda0p1": {
            "label": "Abs Delta Lambda 1e-3 (init rel corr 1e-2, d_lambda_init=0.1)",
            "tol_multiplier": 1.0,
            "stopping_criterion": "absolute_delta_lambda",
            "stopping_tol": 1.0e-3,
            "cli_args": (
                "--d_lambda_init",
                "0.1",
                "--init_newton_stopping_criterion",
                "relative_correction",
                "--init_newton_stopping_tol",
                "1e-2",
            ),
        },
        "absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1": {
            "label": "Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1)",
            "tol_multiplier": 1.0,
            "stopping_criterion": "absolute_delta_lambda",
            "stopping_tol": 1.0e-4,
            "cli_args": (
                "--d_lambda_init",
                "0.1",
                "--init_newton_stopping_criterion",
                "relative_correction",
                "--init_newton_stopping_tol",
                "1e-2",
            ),
        },
        "absolute_delta_lambda_1e_4_initrelcorr_dlambda0p1_no_deflation": {
            "label": "Abs Delta Lambda 1e-4 (init rel corr 1e-2, d_lambda_init=0.1, deflation off)",
            "tol_multiplier": 1.0,
            "stopping_criterion": "absolute_delta_lambda",
            "stopping_tol": 1.0e-4,
            "cli_args": (
                "--d_lambda_init",
                "0.1",
                "--init_newton_stopping_criterion",
                "relative_correction",
                "--init_newton_stopping_tol",
                "1e-2",
                "--max_deflation_basis_vectors",
                "0",
            ),
        },
        "relative_correction_1e_3_initrelcorr_dlambda0p1": {
            "label": "Relative correction 1e-3 (init rel corr 1e-2, d_lambda_init=0.1)",
            "tol_multiplier": 1.0,
            "stopping_criterion": "relative_correction",
            "stopping_tol": 1.0e-3,
            "cli_args": (
                "--d_lambda_init",
                "0.1",
                "--init_newton_stopping_criterion",
                "relative_correction",
                "--init_newton_stopping_tol",
                "1e-2",
            ),
        },
        "hybrid_rough_fine_history_box": {
            "label": "Hybrid Rough/Fine + History-Box Cap",
            "tol_multiplier": 1.0,
            "stopping_criterion": "absolute_delta_lambda",
            "stopping_tol": 1.0e-2,
            "cli_args": (
                "--d_lambda_init",
                "0.05",
                "--init_newton_stopping_criterion",
                "relative_correction",
                "--init_newton_stopping_tol",
                "1e-2",
                "--fine_newton_stopping_criterion",
                "absolute_delta_lambda",
                "--fine_newton_stopping_tol",
                "1e-3",
                "--step_length_cap_mode",
                "history_box",
                "--step_length_cap_factor",
                "1.0",
                "--fine_switch_mode",
                "history_box_cumulative_distance",
                "--fine_switch_distance_factor",
                "2.0",
            ),
        },
        "hybrid_rough_fine_history_box_dlambda0p1_no_flat_stop": {
            "label": "Hybrid Rough/Fine + History-Box Cap (d_lambda_init=0.1, flat-stop off)",
            "tol_multiplier": 1.0,
            "stopping_criterion": "absolute_delta_lambda",
            "stopping_tol": 1.0e-2,
            "cli_args": (
                "--d_lambda_init",
                "0.1",
                "--d_lambda_diff_scaled_min",
                "0.0",
                "--init_newton_stopping_criterion",
                "relative_correction",
                "--init_newton_stopping_tol",
                "1e-2",
                "--fine_newton_stopping_criterion",
                "absolute_delta_lambda",
                "--fine_newton_stopping_tol",
                "1e-3",
                "--step_length_cap_mode",
                "history_box",
                "--step_length_cap_factor",
                "1.0",
                "--fine_switch_mode",
                "history_box_cumulative_distance",
                "--fine_switch_distance_factor",
                "2.0",
            ),
        },
    }
    requested = [str(key).strip() for key in case_keys if str(key).strip()]
    unknown = [key for key in requested if key not in case_defs]
    if unknown:
        raise ValueError(f"Unknown case key(s): {', '.join(unknown)}. Available: {', '.join(sorted(case_defs))}")
    runs_root = artifact_root / "runs"
    commands_root = artifact_root / "commands"
    out: list[CaseSpec] = []
    for key in requested:
        entry = dict(case_defs[key])
        tol_multiplier = float(entry["tol_multiplier"])
        tol = float(base_tol * tol_multiplier)
        out.append(
            CaseSpec(
                key=key,
                label=str(entry["label"]),
                tol_multiplier=tol_multiplier,
                tol=tol,
                stopping_criterion=str(entry["stopping_criterion"]),
                stopping_tol=float(entry["stopping_tol"]),
                out_dir=runs_root / key,
                config_path=commands_root / f"{key}.json",
                cli_args=tuple(str(v) for v in tuple(entry.get("cli_args", ()))),
            )
        )
    return out


def _default_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


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
        str(MESH_PATH),
        "--elem_type",
        "P4",
        "--node_ordering",
        "block_metis",
        "--omega_max_stop",
        str(float(omega_target)),
        "--solver_type",
        "PETSC_MATLAB_DFGMRES_HYPRE_NULLSPACE",
        "--pc_backend",
        "pmg_shell",
        "--preconditioner_matrix_source",
        "tangent",
        "--tol",
        str(float(spec.tol)),
        "--newton_stopping_criterion",
        str(spec.stopping_criterion),
        "--newton_stopping_tol",
        str(float(spec.stopping_tol)),
    ]
    cmd.extend(spec.cli_args)
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
    spec.config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": spec.key,
        "label": spec.label,
        "omega_target": float(omega_target),
        "mpi_ranks": int(mpi_ranks),
        "mesh_path": str(MESH_PATH),
        "elem_type": "P4",
        "pc_backend": "pmg_shell",
        "petsc_opts": list(spec.petsc_opts),
        "command": _case_command(spec=spec, omega_target=omega_target, mpi_ranks=mpi_ranks),
    }
    spec.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_case(*, spec: CaseSpec, omega_target: float, mpi_ranks: int, force: bool) -> None:
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


def _stop_label(criterion: str) -> str:
    mode = str(criterion).strip().lower()
    if mode == "relative_correction":
        return "relative correction"
    if mode == "absolute_delta_lambda":
        return "|Δlambda|"
    return "relative residual"


def _precision_label(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized == "fine":
        return "fine"
    if normalized == "rough":
        return "rough"
    return "base"


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_progress_events(run_dir: Path) -> list[dict[str, object]]:
    progress_path = Path(run_dir) / "data" / "progress.jsonl"
    if not progress_path.exists():
        return []
    events: list[dict[str, object]] = []
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _extract_successful_step_traces_from_progress(case: helpers.CaseSummary) -> dict[int, dict[str, object]]:
    events = _load_progress_events(case.run_dir)
    step_accept_by_step: dict[int, dict[str, object]] = {}
    success_attempt_by_step: dict[int, dict[str, object]] = {}
    newton_events_by_key: dict[tuple[int, int], list[dict[str, object]]] = {}

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

    traces: dict[int, dict[str, object]] = {}
    for step in sorted(step_accept_by_step):
        step_accept = step_accept_by_step[step]
        attempt_complete = success_attempt_by_step.get(step)
        attempt_in_step = _coerce_int(None if attempt_complete is None else attempt_complete.get("attempt_in_step"), 1)
        if (step, attempt_in_step) not in newton_events_by_key:
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
                else _coerce_int(None if attempt_complete is None else attempt_complete.get("newton_iterations"), len(step_events))
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
            "precision_mode": str(step_accept.get("precision_mode", "base")),
            "stopping_criterion": str(
                step_accept.get(
                    "stopping_criterion",
                    None if attempt_complete is None else attempt_complete.get("stopping_criterion", case.stopping_criterion),
                )
            ),
            "stopping_tol": _coerce_float(
                step_accept.get(
                    "stopping_tol",
                    None if attempt_complete is None else attempt_complete.get("stopping_tol", case.stopping_tol),
                )
            ),
            "fine_switch_triggered": bool(
                step_accept.get(
                    "fine_switch_triggered",
                    False if attempt_complete is None else attempt_complete.get("fine_switch_triggered", False),
                )
            ),
            "fine_switch_cumulative_distance": _coerce_float(
                step_accept.get(
                    "fine_switch_cumulative_distance",
                    None if attempt_complete is None else attempt_complete.get("fine_switch_cumulative_distance"),
                )
            ),
            "fine_switch_current_length": _coerce_float(
                step_accept.get(
                    "fine_switch_current_length",
                    None if attempt_complete is None else attempt_complete.get("fine_switch_current_length"),
                )
            ),
            "fine_switch_distance_with_current": _coerce_float(
                step_accept.get(
                    "fine_switch_distance_with_current",
                    None if attempt_complete is None else attempt_complete.get("fine_switch_distance_with_current"),
                )
            ),
            "fine_switch_threshold": _coerce_float(
                step_accept.get(
                    "fine_switch_threshold",
                    None if attempt_complete is None else attempt_complete.get("fine_switch_threshold"),
                )
            ),
            "fine_reference_step": _coerce_int(
                step_accept.get(
                    "fine_reference_step",
                    None if attempt_complete is None else attempt_complete.get("fine_reference_step", 0),
                ),
                0,
            ),
        }
    return traces


def _plot_step_iteration_series(
    traces: list[dict[str, object]],
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
            label=str(trace["label"]),
            color=CASE_COLORS.get(str(trace["key"])),
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
    traces: list[dict[str, object]],
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
            label=str(trace["label"]),
            color=CASE_COLORS.get(str(trace["key"])),
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


def _build_step_newton_sections_from_progress(
    cases: list[helpers.CaseSummary], *, artifact_root: Path
) -> list[dict[str, object]]:
    traces_by_case = {case.key: _extract_successful_step_traces_from_progress(case) for case in cases}
    all_steps = sorted({step for step_traces in traces_by_case.values() for step in step_traces})
    plots_root = artifact_root / "report" / "plots" / "newton_by_step"
    sections: list[dict[str, object]] = []

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


def _write_report(
    *,
    report_path: Path,
    report_title: str,
    artifact_root: Path,
    omega_target: float,
    mpi_ranks: int,
    petsc_opts: tuple[str, ...],
    cases: list[helpers.CaseSummary],
    comparison_plots: dict[str, Path],
    step_newton_sections: list[dict[str, object]],
    step_newton_sections_cap: list[dict[str, object]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    speedup_header = "Speedup vs default" if any(case.key == "default" for case in cases) else "Speedup"
    case_order_line = ", ".join(case.label for case in cases)
    includes_hybrid = any(case.key in HYBRID_CASE_KEYS for case in cases)
    lines: list[str] = [
        f"# {report_title}",
        "",
        "This compares `P4(L1)` with the PMG backend using the standard secant predictor, stopping at "
        f"`omega = {omega_target:.1e}`.",
        "",
        "- Base benchmark: `benchmarks/3d_hetero_ssr_default/case.toml`",
        f"- Overrides: `elem_type = \"P4\"`, `mesh_path = \"{MESH_PATH}\"`, `pc_backend = \"pmg_shell\"`, `node_ordering = \"block_metis\"`",
        f"- MPI ranks: `{int(mpi_ranks)}`",
        "- PMG PETSc opts: " + ", ".join(f"`{opt}`" for opt in petsc_opts),
        f"- Cases: {case_order_line}",
        "- History-box step-length cap: affine-rescale the full current `lambda-omega` history into `[0,1]^2`, measure the first segment (`lambda 1.0 -> 1.1`) there, and limit the next step so the projected last-segment direction has at most the same normalized length.",
        "- Hybrid rough/fine trigger: use `|Δlambda| < 1e-2` by default, then switch the crossing step to `|Δlambda| < 1e-3` once the cumulative accepted rough-path distance plus the current capped projected step exceeds `2x` the initial fine-segment length."
        if includes_hybrid
        else "",
        f"- Artifact root: `{artifact_root}`",
        "",
        "## Summary",
        "",
        f"| Case | Residual tol | Stop criterion | Stop tol | Runtime [s] | {speedup_header} | Accepted states | Continuation steps | Final lambda | Final omega | Init Newton | Continuation Newton | Init linear | Continuation linear | Linear / Newton | Final relres | Final `ΔU/U` |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        lines.append(
            "| "
            + " | ".join(
                [
                    case.label,
                    f"`{case.tol:.1e}`",
                    _stop_label(case.stopping_criterion),
                    f"`{case.stopping_tol:.1e}`",
                    helpers._format_float(case.runtime_seconds, digits=3),
                    helpers._format_float(case.speedup_vs_default, digits=3),
                    str(case.accepted_states),
                    str(case.continuation_steps),
                    helpers._format_float(case.final_lambda),
                    helpers._format_float(case.final_omega, digits=1),
                    str(case.init_newton_total),
                    str(case.continuation_newton_total),
                    str(case.init_linear_total),
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
    if includes_hybrid:
        hybrid_case = next(case for case in cases if case.key in HYBRID_CASE_KEYS)
        hybrid_traces = _extract_successful_step_traces_from_progress(hybrid_case)
        if hybrid_traces:
            lines.append("## Hybrid Trigger Summary")
            lines.append("")
            lines.append(
                "| Accepted step | Precision mode | Stop criterion | Stop tol | Cum rough dist | Current length | Dist + current | Threshold | Reference step | Triggered |"
            )
            lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
            for step in sorted(hybrid_traces):
                trace = hybrid_traces[step]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(int(step)),
                            _precision_label(str(trace["precision_mode"])),
                            _stop_label(str(trace["stopping_criterion"])),
                            helpers._format_sci(trace["stopping_tol"]),
                            helpers._format_float(trace["fine_switch_cumulative_distance"], digits=6),
                            helpers._format_float(trace["fine_switch_current_length"], digits=6),
                            helpers._format_float(trace["fine_switch_distance_with_current"], digits=6),
                            helpers._format_float(trace["fine_switch_threshold"], digits=6),
                            str(int(trace["fine_reference_step"])) if int(trace["fine_reference_step"]) > 0 else "n/a",
                            "yes" if bool(trace["fine_switch_triggered"]) else "no",
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
    def append_step_newton_chapter(
        *,
        chapter_title: str,
        chapter_intro: str,
        chapter_cases: list[helpers.CaseSummary],
        chapter_sections: list[dict[str, object]],
    ) -> None:
        if not chapter_sections:
            return
        lines.append(f"## {chapter_title}")
        lines.append("")
        lines.append(chapter_intro)
        lines.append("")
        for section in chapter_sections:
            step = int(section["step"])
            traces = list(section["traces"])
            plots = dict(section["plots"])
            trace_by_key = {trace["key"]: trace for trace in traces}
            lines.append(f"### Accepted Continuation Step {step}")
            lines.append("")
            lines.append(
                "| Case | Attempt in step | Precision mode | Stop criterion | Stop tol | Newton iterations | Step wall [s] | Final lambda | Final omega | Final relres | Final `ΔU/U` | Cum rough dist | Current length | Threshold | Ref step | Triggered |"
            )
            lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
            for case in chapter_cases:
                trace = trace_by_key.get(case.key)
                if trace is None:
                    lines.append(f"| {case.label} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            case.label,
                            str(trace["attempt_in_step"]),
                            _precision_label(str(trace["precision_mode"])),
                            _stop_label(str(trace["stopping_criterion"])),
                            helpers._format_sci(trace["stopping_tol"]),
                            str(trace["newton_iterations"]),
                            helpers._format_float(trace["step_wall_time"], digits=3),
                            helpers._format_float(trace["final_lambda"]),
                            helpers._format_float(trace["final_omega"], digits=1),
                            helpers._format_sci(trace["final_relres"]),
                            helpers._format_sci(trace["final_relcorr"]),
                            helpers._format_float(trace["fine_switch_cumulative_distance"], digits=6),
                            helpers._format_float(trace["fine_switch_current_length"], digits=6),
                            helpers._format_float(trace["fine_switch_threshold"], digits=6),
                            str(int(trace["fine_reference_step"])) if int(trace["fine_reference_step"]) > 0 else "n/a",
                            "yes" if bool(trace["fine_switch_triggered"]) else "no",
                        ]
                    )
                    + " |"
                )
            lines.append("")
            lines.extend(
                helpers._build_plot_grid(
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
                helpers._build_plot_grid(
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

    non_cap_cases = [case for case in cases if case.key not in CAP_CASE_KEYS]
    cap_cases = [case for case in cases if case.key in CAP_CASE_KEYS]

    append_step_newton_chapter(
        chapter_title="Accepted-Step Newton Solves",
        chapter_intro=(
            "These sections overlay the successful Newton solve that produced each accepted continuation step for the main PMG cases without the step-length cap."
        ),
        chapter_cases=non_cap_cases,
        chapter_sections=step_newton_sections,
    )
    append_step_newton_chapter(
        chapter_title="Accepted-Step Newton Solves With Step-Length Cap",
        chapter_intro=(
            "These sections show the separate Newton convergence history for the cases that use the moving history-box step-length cap, including the hybrid rough/fine run when present."
        ),
        chapter_cases=cap_cases,
        chapter_sections=step_newton_sections_cap,
    )
    lines.append("## Run Artifacts")
    lines.append("")
    for case in cases:
        lines.append(
            f"- {case.label}: command `{helpers._rel(case.config_path, report_path)}`, artifact `{helpers._rel(case.run_dir, report_path)}`"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    artifact_root = args.artifact_root.resolve()
    report_path = args.report_path.resolve()
    base_tol = float(_BASE_SECTIONS.get("newton", {}).get("tol", 1.0e-4))

    specs = _case_specs(base_tol=base_tol, artifact_root=artifact_root, case_keys=args.case_keys)
    for spec in specs:
        _write_command_manifest(spec=spec, omega_target=float(args.omega_target), mpi_ranks=int(args.mpi_ranks))

    if not args.report_only:
        for spec in specs:
            _run_case(spec=spec, omega_target=float(args.omega_target), mpi_ranks=int(args.mpi_ranks), force=bool(args.force))

    default_summary = None
    default_spec = next((spec for spec in specs if spec.key == "default"), None)
    if default_spec is not None:
        default_summary = helpers._build_case_summary(default_spec)
    cases: list[helpers.CaseSummary] = []
    for spec in specs:
        if default_summary is not None and spec.key == "default":
            cases.append(default_summary)
        else:
            cases.append(
                helpers._build_case_summary(
                    spec,
                    default_runtime=None if default_summary is None else default_summary.runtime_seconds,
                )
            )

    plots_dir = artifact_root / "report" / "plots"
    comparison_plots = helpers._plot_histories(cases, plots_dir=plots_dir)
    step_newton_sections = _build_step_newton_sections_from_progress(
        [case for case in cases if case.key not in CAP_CASE_KEYS],
        artifact_root=artifact_root / "report" / "main_cases",
    )
    step_newton_sections_cap = _build_step_newton_sections_from_progress(
        [case for case in cases if case.key in CAP_CASE_KEYS],
        artifact_root=artifact_root / "report" / "step_length_cap_case",
    )

    summary_payload = {
        "report_title": str(args.report_title),
        "omega_target": float(args.omega_target),
        "mpi_ranks": int(args.mpi_ranks),
        "artifact_root": str(artifact_root),
        "report_path": str(report_path),
        "mesh_path": str(MESH_PATH),
        "pc_backend": "pmg_shell",
        "petsc_opts": list(P4_PETSC_OPTIONS),
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
                        "precision_mode": str(trace["precision_mode"]),
                        "stopping_criterion": str(trace["stopping_criterion"]),
                        "stopping_tol": float(trace["stopping_tol"]),
                        "fine_switch_triggered": bool(trace["fine_switch_triggered"]),
                        "fine_switch_cumulative_distance": float(trace["fine_switch_cumulative_distance"]),
                        "fine_switch_current_length": float(trace["fine_switch_current_length"]),
                        "fine_switch_distance_with_current": float(trace["fine_switch_distance_with_current"]),
                        "fine_switch_threshold": float(trace["fine_switch_threshold"]),
                        "fine_reference_step": int(trace["fine_reference_step"]),
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
        "step_newton_sections_cap": [
            {
                "step": int(section["step"]),
                "plots": {name: str(path) for name, path in dict(section["plots"]).items()},
                "cases": [
                    {
                        "key": trace["key"],
                        "label": trace["label"],
                        "attempt_in_step": int(trace["attempt_in_step"]),
                        "precision_mode": str(trace["precision_mode"]),
                        "stopping_criterion": str(trace["stopping_criterion"]),
                        "stopping_tol": float(trace["stopping_tol"]),
                        "fine_switch_triggered": bool(trace["fine_switch_triggered"]),
                        "fine_switch_cumulative_distance": float(trace["fine_switch_cumulative_distance"]),
                        "fine_switch_current_length": float(trace["fine_switch_current_length"]),
                        "fine_switch_distance_with_current": float(trace["fine_switch_distance_with_current"]),
                        "fine_switch_threshold": float(trace["fine_switch_threshold"]),
                        "fine_reference_step": int(trace["fine_reference_step"]),
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
            for section in step_newton_sections_cap
        ],
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
        petsc_opts=P4_PETSC_OPTIONS,
        cases=cases,
        comparison_plots=comparison_plots,
        step_newton_sections=step_newton_sections,
        step_newton_sections_cap=step_newton_sections_cap,
    )
    print(f"[done] report: {report_path}")
    print(f"[done] summary: {summary_path}")


if __name__ == "__main__":
    main()
