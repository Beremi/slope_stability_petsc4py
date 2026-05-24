"""petsc4py-facing wrapper for the pure C DMPlex indirect SSR hot path."""

from __future__ import annotations

import csv
import json
import shlex
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


ROOT = Path(__file__).resolve().parents[3]
STANDALONE_DIR = ROOT / "standalone_petsc_indirect_ssr"
DEFAULT_OPTIONS_FILE = STANDALONE_DIR / "options" / "pmg_shell_split_smoother.opts"


def _option_tokens_from_file(path: Path) -> list[str]:
    tokens: list[str] = []
    if not path.exists():
        return tokens
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if line:
            tokens.extend(shlex.split(line))
    return tokens


def _append_option(tokens: list[str], key: str, value: object | None = None) -> None:
    if not key.startswith("-"):
        key = "-" + key
    tokens.append(key)
    if value is not None:
        tokens.append(str(value))


def _options_string(tokens: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(token)) for token in tokens)


def _curve_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        return np.empty(0), np.empty(0), np.empty(0)
    steps: list[int] = []
    omega: list[float] = []
    lambdas: list[float] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            steps.append(int(row["step"]))
            omega.append(float(row["omega"]))
            lambdas.append(float(row["lambda"]))
    return np.asarray(steps, dtype=np.int64), np.asarray(omega, dtype=np.float64), np.asarray(lambdas, dtype=np.float64)


def run_c_hotpath_capture(
    output_dir: Path,
    *,
    mesh_path: str | Path,
    lambda_init: float,
    d_lambda_init: float,
    d_lambda_min: float,
    d_lambda_diff_scaled_min: float,
    omega_max_stop: float,
    step_max: int,
    it_newt_max: int,
    it_damp_max: int,
    tol: float,
    r_min: float,
    newton_stopping_criterion: str,
    newton_stopping_tol: float | None,
    init_newton_stopping_criterion: str | None,
    init_newton_stopping_tol: float | None,
    linear_tolerance: float,
    linear_max_iter: int,
    pmg_shell_p2_active_ranks: int | None,
    pmg_shell_p1_active_ranks: int | None,
    pmg_shell_subcomm_type: str | None,
    pmg_shell_fine_ksp_max_it: int | None,
    pmg_shell_p2_ksp_max_it: int | None,
    pmg_shell_p1_pc_type: str | None,
    pmg_shell_p1_redundant_number: int | None,
    pmg_shell_p1_redundant_ksp_type: str | None,
    pmg_shell_p1_redundant_ksp_rtol: float | None,
    pmg_shell_p1_redundant_ksp_max_it: int | None,
    pmg_shell_p1_redundant_pc_type: str | None,
    petsc_opt: list[str] | None = None,
) -> dict[str, object]:
    """Run the same C DMPlex solver from the petsc4py config route.

    This backend intentionally keeps the large PETSc objects in C/PETSc and
    writes only curated summaries back through Python, avoiding the legacy
    petsc4py array/CSR duplication that dominates memory use.
    """

    try:
        from slope_stability import _petsc_ssr
    except Exception as exc:  # pragma: no cover - build-environment dependent
        raise RuntimeError(
            "mechanics_backend='dmplex_c_hotpath' requires the slope_stability._petsc_ssr "
            "extension. Rebuild with PETSC_DIR/PETSC_ARCH set to the local PETSc build."
        ) from exc

    comm = PETSc.COMM_WORLD
    rank = int(comm.getRank())
    mpi_comm = comm.tompi4py()
    out_dir = Path(output_dir)
    data_dir = out_dir / "data"
    if rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)
    mpi_comm.Barrier()

    curve_csv = data_dir / "continuation_curve.csv"
    summary_json = data_dir / "c_hotpath_summary.json"
    tokens = _option_tokens_from_file(DEFAULT_OPTIONS_FILE)
    explicit: list[str] = []
    for key, value in (
        ("-mesh", Path(mesh_path)),
        ("-refine_levels", 0),
        ("-lambda_init", lambda_init),
        ("-d_lambda_init", d_lambda_init),
        ("-d_lambda_min", d_lambda_min),
        ("-d_lambda_diff_scaled_min", d_lambda_diff_scaled_min),
        ("-omega_max", omega_max_stop),
        ("-continuation_step_max", step_max),
        ("-newton_max_it", it_newt_max),
        ("-newton_rtol", tol),
        ("-it_damp_max", it_damp_max),
        ("-r_min", r_min),
        ("-newton_stopping_criterion", newton_stopping_criterion),
        ("-linear_rtol", linear_tolerance),
        ("-ksp_max_it", linear_max_iter),
        ("-petscpartitioner_type", "parmetis"),
        ("-curve_csv", curve_csv),
        ("-summary_json", summary_json),
    ):
        _append_option(explicit, key, value)
    if newton_stopping_tol is not None:
        _append_option(explicit, "-newton_stopping_tol", newton_stopping_tol)
    if init_newton_stopping_criterion is not None:
        _append_option(explicit, "-init_newton_stopping_criterion", init_newton_stopping_criterion)
    if init_newton_stopping_tol is not None:
        _append_option(explicit, "-init_newton_stopping_tol", init_newton_stopping_tol)
    for key, value in (
        ("-pmg_shell_p2_active_ranks", pmg_shell_p2_active_ranks),
        ("-pmg_shell_p1_active_ranks", pmg_shell_p1_active_ranks),
        ("-pmg_shell_subcomm_type", pmg_shell_subcomm_type),
        ("-pmg_shell_fine_ksp_max_it", pmg_shell_fine_ksp_max_it),
        ("-pmg_shell_p2_ksp_max_it", pmg_shell_p2_ksp_max_it),
        ("-pmg_shell_p1_pc_type", pmg_shell_p1_pc_type),
        ("-pmg_shell_p1_pc_redundant_number", pmg_shell_p1_redundant_number),
        ("-pmg_shell_p1_redundant_ksp_type", pmg_shell_p1_redundant_ksp_type),
        ("-pmg_shell_p1_redundant_ksp_rtol", pmg_shell_p1_redundant_ksp_rtol),
        ("-pmg_shell_p1_redundant_ksp_max_it", pmg_shell_p1_redundant_ksp_max_it),
        ("-pmg_shell_p1_redundant_pc_type", pmg_shell_p1_redundant_pc_type),
    ):
        if value is not None:
            _append_option(explicit, key, value)
    for entry in petsc_opt or []:
        parts = shlex.split(str(entry).replace("=", " ", 1))
        explicit.extend(parts)

    t0 = perf_counter()
    _petsc_ssr.run_options(_options_string(tokens + explicit))
    wall = perf_counter() - t0
    mpi_comm.Barrier()

    summary: dict[str, object] = {}
    if rank == 0 and summary_json.exists():
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
    summary = mpi_comm.bcast(summary, root=0)

    steps, omega_hist, lambda_hist = _curve_arrays(curve_csv) if rank == 0 else (np.empty(0), np.empty(0), np.empty(0))
    if rank == 0:
        run_payload = {
            "run_info": {
                "timestamp": np.datetime64("now").astype(str),
                "python_version": "petsc4py dmplex C hotpath",
                "runtime_seconds": float(summary.get("wall_time", wall)),
                "mpi_size": int(summary.get("ranks", comm.getSize())),
                "mesh_nodes": None,
                "mesh_elements": None,
                "unknowns": int(summary.get("global_dofs", 0)),
                "analysis": "ssr",
                "mechanics_backend": "dmplex_c_hotpath",
                "solver_type": "C_DFGMRES_HOTPATH",
                "step_count": int(summary.get("accepted_steps", int(len(steps)))),
            },
            "params": {
                "mechanics_backend": "dmplex_c_hotpath",
                "mesh_file": str(mesh_path),
                "lambda_init": float(lambda_init),
                "d_lambda_init": float(d_lambda_init),
                "omega_max_stop": float(omega_max_stop),
                "linear_tolerance": float(linear_tolerance),
                "linear_max_iter": int(linear_max_iter),
            },
            "timings": {
                "continuation_total_wall_time": float(summary.get("continuation_wall_time", wall)),
                "linear": {
                    "init_linear_iterations": 0,
                    "attempt_linear_iterations_total": int(summary.get("total_linear_its", 0)),
                    "manualmg_active_layout_status": "c_hotpath_active_layout",
                    "deflation_orthogonalization_time": float(summary.get("deflation_orthogonalization_time", 0.0)),
                    "deflation_pc_apply_time": float(summary.get("deflation_pc_apply_time", 0.0)),
                    "deflation_projector_time": float(summary.get("deflation_projector_time", 0.0)),
                },
            },
            "c_hotpath_summary": summary,
            "lambda_last": float(summary.get("lambda_last", lambda_hist[-1] if lambda_hist.size else 0.0)),
            "omega_last": float(summary.get("omega_last", omega_hist[-1] if omega_hist.size else 0.0)),
            "final_rel": float(summary.get("final_rel", 0.0)),
            "final_rel_correction": float(summary.get("final_rel_correction", 0.0)),
            "newton_iterations_total": int(summary.get("total_newton_its", 0)),
        }
        (data_dir / "run_info.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
        np.savez_compressed(
            data_dir / "petsc_run.npz",
            step=steps,
            omega_hist=omega_hist,
            lambda_hist=lambda_hist,
            load_factor_hist=lambda_hist,
            U=np.empty((0, 0), dtype=np.float64),
            Umax_hist=np.empty(0, dtype=np.float64),
            step_U=np.empty((0, 3, 0), dtype=np.float64),
        )
        print(
            f"[done] backend=dmplex_c_hotpath steps={run_payload['run_info']['step_count']} "
            f"lambda={run_payload['lambda_last']:.8e} omega={run_payload['omega_last']:.8e}",
            flush=True,
        )

    mpi_comm.Barrier()
    return {
        "output": str(out_dir),
        "npz": str(data_dir / "petsc_run.npz"),
        "json": str(data_dir / "run_info.json"),
        "runtime": float(summary.get("wall_time", wall)),
        "lambda_last": float(summary.get("lambda_last", 0.0)),
        "omega_last": float(summary.get("omega_last", 0.0)),
        "steps": int(summary.get("accepted_steps", 0)),
    }
