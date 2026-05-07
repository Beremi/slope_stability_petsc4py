#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
from time import perf_counter
import time
import subprocess
import signal
import sys
import tomllib
import traceback

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_DIR = SCRIPT_DIR.parent
REPO_ROOT = STUDY_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from slope_stability.fem import available_tetra_quadrature_rules, quadrature_volume_3d


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _load_study(study_path: Path) -> dict:
    with study_path.open("rb") as handle:
        raw = tomllib.load(handle)

    study = raw["study"]
    continuation = raw["continuation"]
    newton = raw["newton"]
    linear_solver = raw["linear_solver"]
    execution = raw["execution"]

    element_types = [str(value).upper() for value in study["element_types"]]
    matrix_raw = study.get("quadrature_matrix")
    if matrix_raw is None:
        shared_rules = [int(value) for value in study["quadrature_rules"]]
        quadrature_matrix = {elem_type: list(shared_rules) for elem_type in element_types}
    else:
        keyed_matrix = {str(key).upper(): value for key, value in dict(matrix_raw).items()}
        quadrature_matrix: dict[str, list[int]] = {}
        missing_elements = [elem_type for elem_type in element_types if elem_type not in keyed_matrix]
        if missing_elements:
            raise ValueError(f"Missing quadrature_matrix entries for elements {missing_elements}.")
        for elem_type in element_types:
            quadrature_matrix[elem_type] = [int(value) for value in keyed_matrix[elem_type]]

    quadrature_rules: list[int] = []
    for elem_type in element_types:
        for rule in quadrature_matrix[elem_type]:
            if rule not in quadrature_rules:
                quadrature_rules.append(int(rule))
    allowed_rules = set(available_tetra_quadrature_rules())
    invalid_rules = [rule for rule in quadrature_rules if rule not in allowed_rules]
    if invalid_rules:
        raise ValueError(f"Unsupported quadrature rules {invalid_rules}; available rules are {sorted(allowed_rules)}.")

    reference_rule = int(study["reference_quadrature_rule"])
    missing_reference = [elem_type for elem_type in element_types if reference_rule not in quadrature_matrix[elem_type]]
    if missing_reference:
        raise ValueError(
            "reference_quadrature_rule must be included in every element-specific quadrature list; "
            f"missing for {missing_reference}."
        )

    def _parse_timeout(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return float(value)

    default_timeout = _parse_timeout(study.get("default_timeout_seconds", study.get("timeout_seconds")))
    timeout_overrides_raw = {str(key).upper(): value for key, value in dict(study.get("timeout_seconds_by_element", {})).items()}
    timeout_seconds_by_element = {
        elem_type: _parse_timeout(timeout_overrides_raw[elem_type])
        for elem_type in element_types
        if elem_type in timeout_overrides_raw
    }

    return {
        "study_dir": study_path.parent.resolve(),
        "asset": str(study["asset"]),
        "mesh_variant": str(study["mesh_variant"]),
        "artifact_dir": (REPO_ROOT / study["artifact_dir"]).resolve(),
        "report_basename": str(study["report_basename"]),
        "name": str(study["name"]),
        "mpi_ranks": int(study["mpi_ranks"]),
        "omp_threads": int(study["omp_threads"]),
        "element_types": element_types,
        "quadrature_matrix": quadrature_matrix,
        "quadrature_rules": quadrature_rules,
        "reference_quadrature_rule": reference_rule,
        "omega_final": float(study["omega_final"]),
        "default_timeout_seconds": default_timeout,
        "timeout_seconds_by_element": timeout_seconds_by_element,
        "continuation": {
            "lambda_init": float(continuation["lambda_init"]),
            "d_lambda_init": float(continuation["d_lambda_init"]),
            "d_lambda_min": float(continuation["d_lambda_min"]),
            "d_lambda_diff_scaled_min": float(continuation["d_lambda_diff_scaled_min"]),
            "step_max": int(continuation["step_max"]),
            "predictor": str(continuation["predictor"]),
            "omega_step_controller": str(continuation["omega_step_controller"]),
        },
        "newton": {
            "it_max": int(newton["it_max"]),
            "it_damp_max": int(newton["it_damp_max"]),
            "tol": float(newton["tol"]),
            "r_min": float(newton["r_min"]),
            "stopping_criterion": str(newton["stopping_criterion"]),
            "stopping_tol": float(newton["stopping_tol"]),
            "init_stopping_criterion": str(newton["init_stopping_criterion"]),
            "init_stopping_tol": float(newton["init_stopping_tol"]),
        },
        "linear_solver": {
            "solver_type": str(linear_solver["solver_type"]),
            "tolerance": float(linear_solver["tolerance"]),
            "max_iterations": int(linear_solver["max_iterations"]),
            "pc_backend": str(linear_solver["pc_backend"]),
            "preconditioner_threads": int(linear_solver["preconditioner_threads"]),
        },
        "execution": {
            "node_ordering": str(execution["node_ordering"]),
            "mpi_distribute_by_nodes": bool(execution["mpi_distribute_by_nodes"]),
            "constitutive_mode": str(execution["constitutive_mode"]),
            "recycle_preconditioner": bool(execution["recycle_preconditioner"]),
            "store_step_u": bool(execution["store_step_u"]),
        },
    }


def _run_id(elem_type: str, quadrature_rule: int) -> str:
    return f"{str(elem_type).lower()}_q{int(quadrature_rule):02d}"


def _iter_study_cases(config: dict) -> list[tuple[str, int]]:
    cases: list[tuple[str, int]] = []
    for elem_type in config["element_types"]:
        for quadrature_rule in config["quadrature_matrix"][str(elem_type)]:
            cases.append((str(elem_type), int(quadrature_rule)))
    return cases


def _resolve_timeout_seconds(config: dict, elem_type: str) -> float | None:
    elem_key = str(elem_type).upper()
    if elem_key in config["timeout_seconds_by_element"]:
        return config["timeout_seconds_by_element"][elem_key]
    return config["default_timeout_seconds"]


def _float_matches(lhs: object, rhs: float, *, rel_tol: float = 1.0e-12, abs_tol: float = 1.0e-12) -> bool:
    try:
        lhs_value = float(lhs)
    except (TypeError, ValueError):
        return False
    return math.isclose(lhs_value, float(rhs), rel_tol=rel_tol, abs_tol=abs_tol)


def _record_matches_config(record: dict, config: dict, *, elem_type: str, quadrature_rule: int) -> bool:
    required_pairs: list[tuple[str, object]] = [
        ("elem_type", str(elem_type)),
        ("quadrature_rule", int(quadrature_rule)),
        ("asset", str(config["asset"])),
        ("mesh_variant", str(config["mesh_variant"])),
        ("solver_type", str(config["linear_solver"]["solver_type"])),
        ("pc_backend", str(config["linear_solver"]["pc_backend"])),
        ("newton_stopping_criterion", str(config["newton"]["stopping_criterion"])),
        ("init_newton_stopping_criterion", str(config["newton"]["init_stopping_criterion"])),
    ]
    for key, expected in required_pairs:
        if record.get(key) != expected:
            return False

    if int(record.get("mpi_ranks", -1)) != int(config["mpi_ranks"]):
        return False
    if int(record.get("linear_max_iterations", -1)) != int(config["linear_solver"]["max_iterations"]):
        return False
    if not _float_matches(record.get("omega_target"), config["omega_final"]):
        return False
    if not _float_matches(record.get("linear_tolerance"), config["linear_solver"]["tolerance"]):
        return False
    if not _float_matches(record.get("newton_stopping_tol"), config["newton"]["stopping_tol"]):
        return False
    if not _float_matches(record.get("init_newton_stopping_tol"), config["newton"]["init_stopping_tol"]):
        return False
    if not _float_matches(
        record.get("continuation_d_lambda_diff_scaled_min"),
        config["continuation"]["d_lambda_diff_scaled_min"],
    ):
        return False
    return True


def _resume_decision(config: dict, *, run_dir: Path, elem_type: str, quadrature_rule: int, resume: bool) -> tuple[bool, str]:
    if not resume:
        return False, "resume disabled"
    record_path = run_dir / "record.json"
    curve_path = run_dir / "curve.csv"
    if not record_path.exists():
        return False, "no normalized record"
    try:
        record = json.loads(record_path.read_text())
    except Exception:
        return False, "record unreadable"
    if not _record_matches_config(record, config, elem_type=elem_type, quadrature_rule=quadrature_rule):
        return False, "record config mismatch"
    if str(record.get("status", "")) != "success":
        return False, f"record status={record.get('status', 'unknown')}"
    if bool(record.get("timed_out", False)):
        return False, "timed-out record"
    if not curve_path.exists():
        return False, "missing normalized curve"
    return True, "completed matching run"


def _monotone_non_decreasing(values: np.ndarray, *, tol: float = 1.0e-10) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return True
    return bool(np.all(np.diff(arr) >= -float(tol)))


def _interp_at_target(x: np.ndarray, y: np.ndarray, target: float) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size == 0:
        return float("nan")
    x_min = float(np.min(x_arr))
    x_max = float(np.max(x_arr))
    if target < x_min or target > x_max:
        return float("nan")
    return float(np.interp(float(target), x_arr, y_arr))


def _safe_sum(npz, key: str) -> float:
    if key not in npz.files:
        return float("nan")
    return float(np.nansum(np.asarray(npz[key], dtype=np.float64)))


def _safe_max(npz, key: str) -> float:
    if key not in npz.files:
        return float("nan")
    arr = np.asarray(npz[key], dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def _write_curve_csv(
    curve_path: Path,
    *,
    curve_rows: list[dict],
) -> None:
    fieldnames = [
        "run_id",
        "elem_type",
        "quadrature_rule",
        "step",
        "omega",
        "lambda",
        "umax",
        "newton_iterations",
        "linear_iterations",
        "newton_relres_end",
        "newton_relcorr_end",
        "attempt_count",
        "branch_efficiency",
    ]
    with curve_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curve_rows)


def _load_progress_events(progress_jsonl_path: Path) -> list[dict]:
    events: list[dict] = []
    if not progress_jsonl_path.exists():
        return events
    for raw_line in progress_jsonl_path.read_text().splitlines():
        text = raw_line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _resolve_stop_reason(
    *,
    progress: dict | None,
    progress_events: list[dict],
    forced_stop_reason: str,
    timed_out: bool,
    returncode: int | None,
) -> str:
    if progress is not None:
        reason = str(progress.get("stop_reason", "")).strip()
        if reason:
            return reason
    for event in reversed(progress_events):
        reason = str(event.get("stop_reason", "")).strip()
        if reason:
            return reason
    if forced_stop_reason:
        return str(forced_stop_reason)
    if timed_out:
        return "timeout"
    if returncode == 0:
        return "finished"
    return ""


def _curve_rows_from_progress_events(
    events: list[dict],
    *,
    run_id: str,
    elem_type: str,
    quadrature_rule: int,
) -> list[dict]:
    rows: list[dict] = []
    init_complete = next(
        (
            event
            for event in events
            if str(event.get("event", "")) == "init_complete"
            and str(event.get("continuation_kind", "")) == "ssr_indirect"
        ),
        None,
    )
    if init_complete is not None:
        lambda_hist = [float(value) for value in init_complete.get("lambda_hist", [])]
        omega_hist = [float(value) for value in init_complete.get("omega_hist", [])]
        for idx, (omega_value, lambda_value) in enumerate(zip(omega_hist, lambda_hist, strict=True), start=1):
            rows.append(
                {
                    "run_id": run_id,
                    "elem_type": elem_type,
                    "quadrature_rule": int(quadrature_rule),
                    "step": int(idx),
                    "omega": float(omega_value),
                    "lambda": float(lambda_value),
                    "umax": float("nan"),
                    "newton_iterations": float("nan"),
                    "linear_iterations": float("nan"),
                    "newton_relres_end": float("nan"),
                    "newton_relcorr_end": float("nan"),
                    "attempt_count": float("nan"),
                    "branch_efficiency": float("nan"),
                }
            )

    accepted_events = [
        event
        for event in events
        if str(event.get("event", "")) == "step_accepted"
        and str(event.get("continuation_kind", "")) == "ssr_indirect"
    ]
    for event in accepted_events:
        step = int(event.get("accepted_step", len(rows) + 1))
        rows.append(
            {
                "run_id": run_id,
                "elem_type": elem_type,
                "quadrature_rule": int(quadrature_rule),
                "step": step,
                "omega": float(event.get("omega_value", float("nan"))),
                "lambda": float(event.get("lambda_value", float("nan"))),
                "umax": float(event.get("u_max", float("nan"))),
                "newton_iterations": float(event.get("step_newton_iterations", float("nan"))),
                "linear_iterations": float(event.get("step_linear_iterations", float("nan"))),
                "newton_relres_end": float(event.get("step_newton_relres_end", float("nan"))),
                "newton_relcorr_end": float(event.get("step_newton_relcorr_end", float("nan"))),
                "attempt_count": float(event.get("step_attempt_count", float("nan"))),
                "branch_efficiency": float("nan"),
            }
        )

    rows.sort(key=lambda row: int(row["step"]))
    return rows


def _load_existing_record(run_dir: Path) -> tuple[dict, list[dict]]:
    record = json.loads((run_dir / "record.json").read_text())
    curve_rows: list[dict] = []
    curve_path = run_dir / "curve.csv"
    if curve_path.exists():
        curve_rows = pd.read_csv(curve_path).to_dict(orient="records")
    return record, curve_rows


def _normalize_completed_raw_artifacts(config: dict, *, run_dir: Path, elem_type: str, quadrature_rule: int) -> tuple[dict, list[dict]] | None:
    data_dir = run_dir / "data"
    run_info_path = data_dir / "run_info.json"
    progress_path = data_dir / "progress_latest.json"
    if not run_info_path.exists():
        return None

    run_id = _run_id(elem_type, quadrature_rule)
    curve_csv_path = run_dir / "curve.csv"
    record_json_path = run_dir / "record.json"
    wall_runtime = float("nan")
    try:
        run_info = json.loads(run_info_path.read_text())
        wall_runtime = float(run_info.get("run_info", {}).get("runtime_seconds", float("nan")))
    except Exception:
        wall_runtime = float("nan")

    forced_stop_reason = ""
    if progress_path.exists():
        try:
            forced_stop_reason = str(json.loads(progress_path.read_text()).get("stop_reason", ""))
        except Exception:
            forced_stop_reason = ""

    record, curve_rows = _status_from_artifacts(
        config=config,
        run_dir=run_dir,
        run_id=run_id,
        elem_type=elem_type,
        quadrature_rule=quadrature_rule,
        wall_runtime=wall_runtime,
        returncode=0,
        timed_out=False,
        error="",
        tb_text="",
        forced_stop_reason=forced_stop_reason,
    )
    record_json_path.write_text(json.dumps(record, indent=2) + "\n")
    _write_curve_csv(curve_csv_path, curve_rows=curve_rows)
    print(f"[normalize] {run_id} status={record['status']} time={record['runtime_seconds_wall']:.2f}s", flush=True)
    return record, curve_rows


def _toml_bool(value: bool) -> str:
    return "true" if bool(value) else "false"


def _quote_toml(value: object) -> str:
    return json.dumps(str(value))


def _write_capture_case_toml(config: dict, *, out_dir: Path, elem_type: str, quadrature_rule: int) -> Path:
    case_path = out_dir / "study_case.toml"
    text = f"""
[problem]
name = {_quote_toml(_run_id(elem_type, quadrature_rule))}
asset = {_quote_toml(config["asset"])}
mesh_variant = {_quote_toml(config["mesh_variant"])}
analysis = "ssr"
elem_type = {_quote_toml(elem_type)}
davis_type = "B"

[geometry]
quadrature_rule = {int(quadrature_rule)}

[execution]
node_ordering = {_quote_toml(config["execution"]["node_ordering"])}
mpi_distribute_by_nodes = {_toml_bool(config["execution"]["mpi_distribute_by_nodes"])}
constitutive_mode = {_quote_toml(config["execution"]["constitutive_mode"])}
store_step_u = {_toml_bool(config["execution"]["store_step_u"])}

[continuation]
lambda_init = {float(config["continuation"]["lambda_init"]):.16g}
d_lambda_init = {float(config["continuation"]["d_lambda_init"]):.16g}
d_lambda_min = {float(config["continuation"]["d_lambda_min"]):.16g}
d_lambda_diff_scaled_min = {float(config["continuation"]["d_lambda_diff_scaled_min"]):.16g}
omega_max = {float(config["omega_final"]):.16g}
predictor = {_quote_toml(config["continuation"]["predictor"])}
omega_step_controller = {_quote_toml(config["continuation"]["omega_step_controller"])}
step_max = {int(config["continuation"]["step_max"])}
init_newton_stopping_criterion = {_quote_toml(config["newton"]["init_stopping_criterion"])}
init_newton_stopping_tol = {float(config["newton"]["init_stopping_tol"]):.16g}

[newton]
it_max = {int(config["newton"]["it_max"])}
it_damp_max = {int(config["newton"]["it_damp_max"])}
tol = {float(config["newton"]["tol"]):.16g}
r_min = {float(config["newton"]["r_min"]):.16g}
stopping_criterion = {_quote_toml(config["newton"]["stopping_criterion"])}
stopping_tol = {float(config["newton"]["stopping_tol"]):.16g}

[linear_solver]
solver_type = {_quote_toml(config["linear_solver"]["solver_type"])}
tolerance = {float(config["linear_solver"]["tolerance"]):.16g}
max_iterations = {int(config["linear_solver"]["max_iterations"])}
pc_backend = {_quote_toml(config["linear_solver"]["pc_backend"])}
threads = {int(config["linear_solver"]["preconditioner_threads"])}
recycle_preconditioner = {_toml_bool(config["execution"]["recycle_preconditioner"])}
""".lstrip()
    case_path.write_text(text, encoding="utf-8")
    return case_path


def _capture_command(config: dict, *, out_dir: Path, elem_type: str, quadrature_rule: int) -> list[str]:
    python_exec = str(REPO_ROOT / ".venv" / "bin" / "python")
    case_path = _write_capture_case_toml(config, out_dir=out_dir, elem_type=elem_type, quadrature_rule=quadrature_rule)

    return [
        "mpirun",
        "-n",
        str(config["mpi_ranks"]),
        python_exec,
        "-m",
        "slope_stability.cli.run_case_from_config",
        str(case_path),
        "--out_dir",
        str(out_dir),
    ]


def _curve_rows_from_npz(npz, *, run_id: str, elem_type: str, quadrature_rule: int) -> list[dict]:
    lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64)
    omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64)
    umax_hist = np.asarray(npz["Umax_hist"], dtype=np.float64)
    n_steps = int(lambda_hist.size)

    def _series(name: str) -> np.ndarray:
        if name not in npz.files:
            return np.full(n_steps, np.nan, dtype=np.float64)
        arr = np.asarray(npz[name], dtype=np.float64)
        if arr.size == n_steps:
            return arr
        return np.full(n_steps, np.nan, dtype=np.float64)

    newton_iterations = _series("stats_step_newton_iterations")
    linear_iterations = _series("stats_step_linear_iterations")
    newton_relres_end = _series("stats_step_newton_relres_end")
    newton_relcorr_end = _series("stats_step_newton_relcorr_end")
    attempt_count = _series("stats_step_attempt_count")
    branch_efficiency = _series("stats_step_branch_efficiency")

    rows: list[dict] = []
    for idx in range(n_steps):
        rows.append(
            {
                "run_id": run_id,
                "elem_type": elem_type,
                "quadrature_rule": int(quadrature_rule),
                "step": int(idx + 1),
                "omega": float(omega_hist[idx]),
                "lambda": float(lambda_hist[idx]),
                "umax": float(umax_hist[idx]),
                "newton_iterations": float(newton_iterations[idx]),
                "linear_iterations": float(linear_iterations[idx]),
                "newton_relres_end": float(newton_relres_end[idx]),
                "newton_relcorr_end": float(newton_relcorr_end[idx]),
                "attempt_count": float(attempt_count[idx]),
                "branch_efficiency": float(branch_efficiency[idx]),
            }
        )
    return rows


def _status_from_artifacts(
    *,
    config: dict,
    run_dir: Path,
    run_id: str,
    elem_type: str,
    quadrature_rule: int,
    wall_runtime: float,
    returncode: int | None,
    timed_out: bool,
    error: str,
    tb_text: str,
    forced_stop_reason: str,
) -> tuple[dict, list[dict]]:
    data_dir = run_dir / "data"
    run_info_path = data_dir / "run_info.json"
    progress_path = data_dir / "progress_latest.json"
    progress_jsonl_path = data_dir / "progress.jsonl"
    npz_path = data_dir / "petsc_run.npz"

    run_info = None
    progress = None
    progress_events: list[dict] = []
    curve_rows: list[dict] = []
    lambda_hist = np.asarray([], dtype=np.float64)
    omega_hist = np.asarray([], dtype=np.float64)
    umax_hist = np.asarray([], dtype=np.float64)
    step_newton_iterations_total = float("nan")
    step_linear_iterations_total = float("nan")
    max_step_newton_iterations = float("nan")
    max_step_linear_iterations = float("nan")
    attempt_count_total = float("nan")

    if run_info_path.exists():
        run_info = json.loads(run_info_path.read_text())
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
    if progress_jsonl_path.exists():
        progress_events = _load_progress_events(progress_jsonl_path)
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=True) as npz:
            lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64)
            omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64)
            umax_hist = np.asarray(npz["Umax_hist"], dtype=np.float64)
            curve_rows = _curve_rows_from_npz(npz, run_id=run_id, elem_type=elem_type, quadrature_rule=quadrature_rule)
            step_newton_iterations_total = _safe_sum(npz, "stats_step_newton_iterations")
            step_linear_iterations_total = _safe_sum(npz, "stats_step_linear_iterations")
            max_step_newton_iterations = _safe_max(npz, "stats_step_newton_iterations")
            max_step_linear_iterations = _safe_max(npz, "stats_step_linear_iterations")
            attempt_count_total = _safe_sum(npz, "stats_step_attempt_count")
    elif progress_events:
        curve_rows = _curve_rows_from_progress_events(
            progress_events,
            run_id=run_id,
            elem_type=elem_type,
            quadrature_rule=quadrature_rule,
        )
        if curve_rows:
            lambda_hist = np.asarray([float(row["lambda"]) for row in curve_rows], dtype=np.float64)
            omega_hist = np.asarray([float(row["omega"]) for row in curve_rows], dtype=np.float64)
            umax_hist = np.asarray([float(row["umax"]) for row in curve_rows], dtype=np.float64)
            step_newton_iterations = np.asarray(
                [float(row["newton_iterations"]) for row in curve_rows if np.isfinite(float(row["newton_iterations"]))],
                dtype=np.float64,
            )
            step_linear_iterations = np.asarray(
                [float(row["linear_iterations"]) for row in curve_rows if np.isfinite(float(row["linear_iterations"]))],
                dtype=np.float64,
            )
            attempt_counts = np.asarray(
                [float(row["attempt_count"]) for row in curve_rows if np.isfinite(float(row["attempt_count"]))],
                dtype=np.float64,
            )
            if step_newton_iterations.size:
                step_newton_iterations_total = float(np.nansum(step_newton_iterations))
                max_step_newton_iterations = float(np.nanmax(step_newton_iterations))
            if step_linear_iterations.size:
                step_linear_iterations_total = float(np.nansum(step_linear_iterations))
                max_step_linear_iterations = float(np.nanmax(step_linear_iterations))
            if attempt_counts.size:
                attempt_count_total = float(np.nansum(attempt_counts))

    runtime_capture = float("nan")
    mesh_nodes = 0
    mesh_elements = 0
    unknowns = 0
    mpi_size = config["mpi_ranks"]
    n_q = int(quadrature_volume_3d(elem_type, quadrature_rule)[0].shape[1])
    n_int = 0
    solver_type = config["linear_solver"]["solver_type"]

    if run_info is not None:
        run_meta = run_info.get("run_info", {})
        runtime_capture = float(run_meta.get("runtime_seconds", float("nan")))
        mesh_nodes = int(run_meta.get("mesh_nodes", 0))
        mesh_elements = int(run_meta.get("mesh_elements", 0))
        unknowns = int(run_meta.get("unknowns", 0))
        mpi_size = int(run_meta.get("mpi_size", config["mpi_ranks"]))
        solver_type = str(run_meta.get("solver_type", solver_type))
        if mesh_elements > 0:
            n_int = int(mesh_elements * n_q)
    elif progress_events:
        total_wall_candidates = [
            float(event.get("total_wall_time", float("nan")))
            for event in progress_events
            if np.isfinite(float(event.get("total_wall_time", float("nan"))))
        ]
        if total_wall_candidates:
            runtime_capture = float(total_wall_candidates[-1])

    omega_last = float(omega_hist[-1]) if omega_hist.size else float("nan")
    lambda_last = float(lambda_hist[-1]) if lambda_hist.size else float("nan")
    umax_last = float(umax_hist[-1]) if umax_hist.size else float("nan")
    reached_target = bool(omega_hist.size and omega_last >= float(config["omega_final"]) - 1.0e-8)
    omega_monotone = _monotone_non_decreasing(omega_hist)
    lambda_monotone = _monotone_non_decreasing(lambda_hist)

    stop_reason = _resolve_stop_reason(
        progress=progress,
        progress_events=progress_events,
        forced_stop_reason=forced_stop_reason,
        timed_out=timed_out,
        returncode=returncode,
    )

    if returncode == 0 and not timed_out:
        status = "success" if reached_target else "incomplete"
    elif lambda_hist.size:
        status = "incomplete"
    else:
        status = "failed"

    record = {
        "run_id": run_id,
        "elem_type": elem_type,
        "quadrature_rule": int(quadrature_rule),
        "omega_target": float(config["omega_final"]),
        "status": status,
        "returncode": None if returncode is None else int(returncode),
        "timed_out": bool(timed_out),
        "runtime_seconds_wall": float(wall_runtime),
        "runtime_seconds_capture": runtime_capture,
        "mpi_ranks": int(mpi_size),
        "asset": str(config["asset"]),
        "mesh_variant": str(config["mesh_variant"]),
        "mesh_nodes": int(mesh_nodes),
        "mesh_elements": int(mesh_elements),
        "unknowns": int(unknowns),
        "n_q": int(n_q),
        "n_int": int(n_int),
        "solver_type": str(solver_type),
        "linear_tolerance": float(config["linear_solver"]["tolerance"]),
        "linear_max_iterations": int(config["linear_solver"]["max_iterations"]),
        "pc_backend": str(config["linear_solver"]["pc_backend"]),
        "newton_stopping_criterion": str(config["newton"]["stopping_criterion"]),
        "newton_stopping_tol": float(config["newton"]["stopping_tol"]),
        "init_newton_stopping_criterion": str(config["newton"]["init_stopping_criterion"]),
        "init_newton_stopping_tol": float(config["newton"]["init_stopping_tol"]),
        "continuation_d_lambda_diff_scaled_min": float(config["continuation"]["d_lambda_diff_scaled_min"]),
        "accepted_steps": int(lambda_hist.size),
        "omega_last": omega_last,
        "lambda_last": lambda_last,
        "umax_last": umax_last,
        "omega_monotone": bool(omega_monotone),
        "lambda_monotone": bool(lambda_monotone),
        "reached_omega_target": bool(reached_target),
        "lambda_at_omega_target": _interp_at_target(omega_hist, lambda_hist, float(config["omega_final"])),
        "umax_at_omega_target": _interp_at_target(omega_hist, umax_hist, float(config["omega_final"])),
        "stop_reason": stop_reason,
        "step_newton_iterations_total": step_newton_iterations_total,
        "step_linear_iterations_total": step_linear_iterations_total,
        "max_step_newton_iterations": max_step_newton_iterations,
        "max_step_linear_iterations": max_step_linear_iterations,
        "attempt_count_total": attempt_count_total,
        "error": str(error),
        "traceback": str(tb_text),
    }
    return record, curve_rows


def _read_progress_snapshot(run_dir: Path) -> dict | None:
    progress_path = run_dir / "data" / "progress_latest.json"
    if not progress_path.exists():
        return None
    try:
        return json.loads(progress_path.read_text())
    except Exception:
        return None


def _progress_indicates_zero_correction_stall(progress: dict | None) -> bool:
    if not progress:
        return False
    if str(progress.get("phase", "")) != "init":
        return False
    if str(progress.get("event", "")) != "newton_iteration":
        return False
    if int(progress.get("init_attempt", 0)) < 2:
        return False
    if int(progress.get("iteration", 0)) < 10:
        return False
    rel_residual = float(progress.get("rel_residual", float("nan")))
    alpha = float(progress.get("alpha", float("nan")))
    accepted_correction_norm = float(progress.get("accepted_correction_norm", float("nan")))
    accepted_relative_correction_norm = float(progress.get("accepted_relative_correction_norm", float("nan")))
    linear_iterations = int(progress.get("linear_iterations", 0))
    return (
        np.isfinite(rel_residual)
        and rel_residual >= 0.999999
        and np.isfinite(alpha)
        and abs(alpha) <= 1.0e-15
        and np.isfinite(accepted_correction_norm)
        and abs(accepted_correction_norm) <= 1.0e-15
        and np.isfinite(accepted_relative_correction_norm)
        and abs(accepted_relative_correction_norm) <= 1.0e-15
        and linear_iterations <= 1
    )


def _progress_indicates_capped_linear_zero_correction_stall(
    progress_events: list[dict],
    *,
    linear_max_iterations: int,
    window: int = 8,
) -> bool:
    init_newton_events = [
        event
        for event in progress_events
        if str(event.get("phase", "")) == "init" and str(event.get("event", "")) == "newton_iteration"
    ]
    if len(init_newton_events) < int(window):
        return False
    recent = init_newton_events[-int(window):]
    for event in recent:
        try:
            alpha = float(event.get("alpha", float("nan")))
            accepted_correction_norm = float(event.get("accepted_correction_norm", float("nan")))
            accepted_relative_correction_norm = float(event.get("accepted_relative_correction_norm", float("nan")))
            rel_residual = float(event.get("rel_residual", float("nan")))
            linear_iterations = int(event.get("linear_iterations", 0))
        except (TypeError, ValueError):
            return False
        if not np.isfinite(alpha) or abs(alpha) > 1.0e-15:
            return False
        if not np.isfinite(accepted_correction_norm) or abs(accepted_correction_norm) > 1.0e-15:
            return False
        if not np.isfinite(accepted_relative_correction_norm) or abs(accepted_relative_correction_norm) > 1.0e-15:
            return False
        if not np.isfinite(rel_residual) or rel_residual < 0.99:
            return False
        if linear_iterations < int(linear_max_iterations):
            return False

    criterion_values = np.asarray(
        [float(event.get("criterion", float("nan"))) for event in recent],
        dtype=np.float64,
    )
    finite = criterion_values[np.isfinite(criterion_values)]
    if finite.size < int(window):
        return False
    baseline = max(1.0, float(np.max(np.abs(finite))))
    return bool(float(np.ptp(finite)) <= 1.0e-10 * baseline)


def _terminate_process_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass


def _wait_with_watchdogs(
    proc: subprocess.Popen,
    *,
    run_dir: Path,
    timeout_seconds: float | None,
    linear_max_iterations: int,
) -> tuple[int | None, bool, str, str, str]:
    start = perf_counter()
    forced_stop_reason = ""
    error = ""
    tb_text = ""
    progress_jsonl_path = run_dir / "data" / "progress.jsonl"
    while True:
        returncode = proc.poll()
        if returncode is not None:
            return int(returncode), False, error, tb_text, forced_stop_reason

        elapsed = perf_counter() - start
        progress = _read_progress_snapshot(run_dir)
        progress_events = _load_progress_events(progress_jsonl_path) if progress_jsonl_path.exists() else []
        if _progress_indicates_zero_correction_stall(progress):
            forced_stop_reason = "zero_correction_stall"
            error = "Aborted after repeated zero-correction Newton stall in init attempt."
            _terminate_process_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                _terminate_process_group(proc, signal.SIGKILL)
                proc.wait(timeout=15.0)
            return None, False, error, tb_text, forced_stop_reason
        if _progress_indicates_capped_linear_zero_correction_stall(
            progress_events,
            linear_max_iterations=linear_max_iterations,
        ):
            forced_stop_reason = "capped_linear_zero_correction_stall"
            error = "Aborted after repeated capped-linear zero-correction Newton stall in init solve."
            _terminate_process_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                _terminate_process_group(proc, signal.SIGKILL)
                proc.wait(timeout=15.0)
            return None, False, error, tb_text, forced_stop_reason

        if timeout_seconds is not None and elapsed >= float(timeout_seconds):
            forced_stop_reason = "timeout"
            error = f"Timed out after {timeout_seconds} s."
            _terminate_process_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=15.0)
            except subprocess.TimeoutExpired as exc:
                tb_text = "".join(traceback.format_exception(exc))
                _terminate_process_group(proc, signal.SIGKILL)
                proc.wait(timeout=15.0)
            return None, True, error, tb_text, forced_stop_reason

        time.sleep(5.0)


def _run_single_case(
    config: dict,
    *,
    elem_type: str,
    quadrature_rule: int,
    resume: bool,
) -> tuple[dict, list[dict]]:
    artifact_root = Path(config["artifact_dir"])
    run_id = _run_id(elem_type, quadrature_rule)
    run_dir = artifact_root / run_id
    record_json_path = run_dir / "record.json"
    curve_csv_path = run_dir / "curve.csv"
    reusable, reason = _resume_decision(
        config,
        run_dir=run_dir,
        elem_type=elem_type,
        quadrature_rule=quadrature_rule,
        resume=resume,
    )
    if reusable:
        print(f"[resume] {run_id} ({reason})", flush=True)
        return _load_existing_record(run_dir)

    if resume and run_dir.exists():
        normalized = _normalize_completed_raw_artifacts(
            config,
            run_dir=run_dir,
            elem_type=elem_type,
            quadrature_rule=quadrature_rule,
        )
        if normalized is not None:
            return normalized

    if run_dir.exists():
        print(f"[rerun] {run_id} ({reason})", flush=True)
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    command = _capture_command(config, out_dir=run_dir, elem_type=elem_type, quadrature_rule=quadrature_rule)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(config["omp_threads"])
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    (run_dir / "command.txt").write_text(subprocess.list2cmdline(command) + "\n")
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "OMP_NUM_THREADS": env["OMP_NUM_THREADS"],
                "OPENBLAS_NUM_THREADS": env["OPENBLAS_NUM_THREADS"],
                "MKL_NUM_THREADS": env["MKL_NUM_THREADS"],
                "PYTHONPATH": env["PYTHONPATH"],
            },
            indent=2,
        )
        + "\n"
    )

    print(f"[run] {run_id} -> omega={config['omega_final']:.6g}", flush=True)
    start = perf_counter()
    returncode: int | None = None
    timed_out = False
    error = ""
    tb_text = ""
    forced_stop_reason = ""
    with (run_dir / "run.log").open("w") as log_handle:
        log_handle.write(subprocess.list2cmdline(command) + "\n\n")
        log_handle.flush()
        try:
            proc = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            returncode, timed_out, error, tb_text, forced_stop_reason = _wait_with_watchdogs(
                proc,
                run_dir=run_dir,
                timeout_seconds=_resolve_timeout_seconds(config, elem_type),
                linear_max_iterations=int(config["linear_solver"]["max_iterations"]),
            )
            if error:
                log_handle.write("\n\n[study-watchdog]\n")
                log_handle.write(error + "\n")
                if tb_text:
                    log_handle.write(tb_text)
        except Exception as exc:  # pragma: no cover - defensive outer wrapper
            returncode = None
            error = str(exc)
            tb_text = "".join(traceback.format_exception(exc))
            log_handle.write("\n\n[study-exception]\n")
            log_handle.write(tb_text)

    wall_runtime = perf_counter() - start
    record, curve_rows = _status_from_artifacts(
        config=config,
        run_dir=run_dir,
        run_id=run_id,
        elem_type=elem_type,
        quadrature_rule=quadrature_rule,
        wall_runtime=wall_runtime,
        returncode=returncode,
        timed_out=timed_out,
        error=error,
        tb_text=tb_text,
        forced_stop_reason=forced_stop_reason,
    )

    record_json_path.write_text(json.dumps(record, indent=2) + "\n")
    _write_curve_csv(curve_csv_path, curve_rows=curve_rows)
    print(
        f"[done] {run_id} status={record['status']} time={record['runtime_seconds_wall']:.2f}s "
        f"omega_last={record['omega_last']:.6g}",
        flush=True,
    )
    return record, curve_rows


def _write_frames(config: dict, *, records: list[dict], curve_rows: list[dict]) -> None:
    data_dir = Path(config["study_dir"]) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    runs_df = pd.DataFrame(records)
    curves_df = pd.DataFrame(curve_rows)
    runs_df.to_csv(data_dir / "runs.csv", index=False)
    curves_df.to_csv(data_dir / "continuation_curves.csv", index=False)
    (data_dir / "study_meta.json").write_text(
        json.dumps(
            {
                "study_name": config["name"],
                "report_basename": config["report_basename"],
                "asset": str(config["asset"]),
                "mesh_variant": str(config["mesh_variant"]),
                "artifact_dir": _repo_relative(Path(config["artifact_dir"])),
                "omega_final": float(config["omega_final"]),
                "element_types": list(config["element_types"]),
                "quadrature_matrix": {key: list(value) for key, value in config["quadrature_matrix"].items()},
                "quadrature_rules": list(config["quadrature_rules"]),
                "reference_quadrature_rule": int(config["reference_quadrature_rule"]),
                "mpi_ranks": int(config["mpi_ranks"]),
                "omp_threads": int(config["omp_threads"]),
                "default_timeout_seconds": config["default_timeout_seconds"],
                "timeout_seconds_by_element": config["timeout_seconds_by_element"],
                "continuation": config["continuation"],
                "newton": config["newton"],
                "linear_solver": config["linear_solver"],
                "execution": config["execution"],
                "benchmark_note": (
                    "The standard heterogeneous 3D SSR benchmark on asset `3d_hetero_slope` with mesh variant `adaptive_family_a_l1.msh` reaches "
                    "omega on the order of 6.7e6 only in the indirect continuation formulation. "
                    "The earlier direct-study omega scale was therefore not the same benchmark."
                ),
            },
            indent=2,
        )
        + "\n"
    )


def _collect_existing_records(config: dict) -> tuple[list[dict], list[dict]]:
    artifact_root = Path(config["artifact_dir"])
    records: list[dict] = []
    curve_rows: list[dict] = []
    if not artifact_root.exists():
        return records, curve_rows
    for elem_type, quadrature_rule in _iter_study_cases(config):
        run_dir = artifact_root / _run_id(elem_type, quadrature_rule)
        record_path = run_dir / "record.json"
        if not record_path.exists():
            continue
        record = json.loads(record_path.read_text())
        records.append(record)
        curve_path = run_dir / "curve.csv"
        if curve_path.exists():
            curve_rows.extend(pd.read_csv(curve_path).to_dict(orient="records"))
    return records, curve_rows


def _print_run_plan(config: dict, *, selected_ids: set[str], resume: bool) -> None:
    print(
        f"[plan] omega_final={config['omega_final']:.6g} reference=q{int(config['reference_quadrature_rule'])} "
        f"default_timeout={config['default_timeout_seconds']}",
        flush=True,
    )
    for elem_type in config["element_types"]:
        rules = [int(rule) for rule in config["quadrature_matrix"][elem_type]]
        timeout_value = _resolve_timeout_seconds(config, elem_type)
        print(
            f"[plan] {elem_type}: rules={','.join(f'q{rule}' for rule in rules)} "
            f"timeout={'none' if timeout_value is None else f'{timeout_value:.0f}s'}",
            flush=True,
        )
    for elem_type, quadrature_rule in _iter_study_cases(config):
        run_id = _run_id(elem_type, quadrature_rule)
        if selected_ids and run_id not in selected_ids:
            continue
        reusable, reason = _resume_decision(
            config,
            run_dir=Path(config["artifact_dir"]) / run_id,
            elem_type=elem_type,
            quadrature_rule=quadrature_rule,
            resume=resume,
        )
        action = "reuse" if reusable else "run"
        print(
            f"[plan] {run_id}: {action} ({reason}); timeout="
            f"{'none' if _resolve_timeout_seconds(config, elem_type) is None else f'{_resolve_timeout_seconds(config, elem_type):.0f}s'}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the heterogeneous 3D SSR integration-point study.")
    parser.add_argument("--study", type=Path, default=STUDY_DIR / "study.toml")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--only", action="append", default=[], help="Optional run id(s) like p4_q24 to execute.")
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()

    config = _load_study(Path(args.study).resolve())
    Path(config["artifact_dir"]).mkdir(parents=True, exist_ok=True)

    selected_ids = {str(value).strip().lower() for value in args.only if str(value).strip()}
    if args.dry_run:
        _print_run_plan(config, selected_ids=selected_ids, resume=bool(args.resume))
        return

    for elem_type, quadrature_rule in _iter_study_cases(config):
        run_id = _run_id(elem_type, quadrature_rule)
        if selected_ids and run_id not in selected_ids:
            continue
        _run_single_case(
            config,
            elem_type=elem_type,
            quadrature_rule=quadrature_rule,
            resume=bool(args.resume),
        )
    records, curve_rows = _collect_existing_records(config)
    _write_frames(config, records=records, curve_rows=curve_rows)


if __name__ == "__main__":
    main()
