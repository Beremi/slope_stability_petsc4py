from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import h5py
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from study_common import load_horizons, load_study, read_json


def _to_float(value) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _load_petsc_stop_reason(npz_path: Path, run_info: dict) -> str:
    stop_reason = str(run_info.get("run_info", {}).get("stop_reason", "")).strip()
    if stop_reason:
        return stop_reason
    with np.load(npz_path, allow_pickle=True) as npz:
        if "stats_stop_reason" not in npz:
            return ""
        raw = npz["stats_stop_reason"]
        if np.ndim(raw) == 0:
            return str(raw.item())
        arr = np.asarray(raw).reshape(-1)
        return "" if arr.size == 0 else str(arr[0])


def _infer_stop_reason(
    stop_reason: str,
    *,
    final_omega: float | str,
    omega_max_stop: float | str,
    d_lambda_diff_scaled_min: float | str,
) -> str:
    text = str(stop_reason).strip()
    if text:
        return text
    omega_value = _to_float(final_omega)
    omega_limit = _to_float(omega_max_stop)
    if math.isfinite(omega_value) and math.isfinite(omega_limit):
        scale = max(abs(omega_limit), 1.0)
        if abs(omega_value - omega_limit) <= 1.0e-9 * scale:
            return "omega_max_stop"
    d_lambda_floor = _to_float(d_lambda_diff_scaled_min)
    if math.isfinite(d_lambda_floor) and d_lambda_floor > 0.0:
        return "d_lambda_diff_scaled_min"
    return ""


def _load_h5_dataset(h5: h5py.File, name: str) -> np.ndarray:
    if name not in h5:
        return np.asarray([], dtype=np.float64)
    return np.asarray(h5[name][()])


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect raw PETSc/MATLAB study runs into report CSVs.")
    parser.add_argument("--study", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    study = load_study(args.study, manifest_path=args.manifest, allow_missing_manifest=True)
    horizons = load_horizons(study["artifact_root"] / "horizons.json")
    case_tol_map = {case["id"]: case["tol"] for case in study["cases"]}
    run_meta_paths = sorted(study["artifact_root"].rglob("study_run.json"))

    runs_rows: list[dict] = []
    continuation_rows: list[dict] = []
    newton_rows: list[dict] = []
    horizon_rows: list[dict] = []

    for meta_path in run_meta_paths:
        meta = read_json(meta_path)
        output_dir = meta_path.parent
        engine = meta["engine"]
        phase = meta["phase"]

        if engine == "petsc":
            run_info_path = output_dir / "data" / "run_info.json"
            npz_path = output_dir / "data" / "petsc_run.npz"
            if not run_info_path.exists() or not npz_path.exists():
                continue
            run_info = read_json(run_info_path)
            with np.load(npz_path, allow_pickle=True) as npz:
                lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64).reshape(-1)
                omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64).reshape(-1)
                umax_hist = np.asarray(npz.get("Umax_hist", []), dtype=np.float64).reshape(-1)
                final_lambda = lambda_hist[-1] if lambda_hist.size else ""
                final_omega = omega_hist[-1] if omega_hist.size else ""
                stop_reason = _infer_stop_reason(
                    _load_petsc_stop_reason(npz_path, run_info),
                    final_omega=final_omega,
                    omega_max_stop=meta["shared_omega_max_stop"],
                    d_lambda_diff_scaled_min=meta["d_lambda_diff_scaled_min"],
                )

                runs_rows.append(
                    {
                        "case_id": meta["case_id"],
                        "case_label": meta["case_label"],
                        "level_id": meta["level_id"],
                        "level_label": meta["level_label"],
                        "level_order": meta["level_order"],
                        "engine": engine,
                        "variant": meta["variant"],
                        "phase": phase,
                        "mesh_nodes": run_info["run_info"].get("mesh_nodes", ""),
                        "mesh_elements": run_info["run_info"].get("mesh_elements", ""),
                        "unknowns": run_info["run_info"].get("unknowns", ""),
                        "runtime_seconds": run_info["run_info"].get("runtime_seconds", ""),
                        "accepted_steps": lambda_hist.size,
                        "final_lambda": final_lambda,
                        "final_omega": final_omega,
                        "stop_reason": stop_reason,
                        "omega_max_stop": meta["shared_omega_max_stop"],
                        "d_lambda_diff_scaled_min": meta["d_lambda_diff_scaled_min"],
                        "newton_residual_tol": meta.get("newton_residual_tol", case_tol_map.get(meta["case_id"], "")),
                        "newton_stopping_criterion": meta["newton_stopping_criterion"],
                        "newton_stopping_tol": meta["newton_stopping_tol"] if meta["newton_stopping_tol"] is not None else "",
                        "output_dir": str(output_dir),
                    }
                )

                if phase == "smoke":
                    horizon_rows.append(
                        {
                            "case_id": meta["case_id"],
                            "case_label": meta["case_label"],
                            "level_id": meta["level_id"],
                            "level_label": meta["level_label"],
                            "smoke_final_omega": lambda_hist.size and omega_hist[-1] or "",
                            "chosen_omega_max_stop": horizons.get(meta["case_id"], {}).get(
                                "omega_max_stop",
                                meta["shared_omega_max_stop"],
                            ),
                            "runtime_seconds": run_info["run_info"].get("runtime_seconds", ""),
                            "accepted_steps": lambda_hist.size,
                            "output_dir": str(output_dir),
                        }
                    )

                if phase == "main":
                    for step_idx, (omega_value, lambda_value) in enumerate(zip(omega_hist, lambda_hist), start=1):
                        umax_value = umax_hist[step_idx - 1] if step_idx - 1 < umax_hist.size else ""
                        continuation_rows.append(
                            {
                                "case_id": meta["case_id"],
                                "case_label": meta["case_label"],
                                "level_id": meta["level_id"],
                                "level_label": meta["level_label"],
                                "level_order": meta["level_order"],
                                "engine": engine,
                                "variant": meta["variant"],
                                "step": step_idx,
                                "omega": omega_value,
                                "lambda": lambda_value,
                                "umax": umax_value,
                            }
                        )

                    step_newton = np.asarray(npz.get("stats_step_newton_iterations", []), dtype=np.float64).reshape(-1)
                    step_linear = np.asarray(npz.get("stats_step_linear_iterations", []), dtype=np.float64).reshape(-1)
                    step_linear_solve = np.asarray(npz.get("stats_step_linear_solve_time", []), dtype=np.float64).reshape(-1)
                    step_linear_prec = np.asarray(npz.get("stats_step_linear_preconditioner_time", []), dtype=np.float64).reshape(-1)
                    step_linear_orth = np.asarray(npz.get("stats_step_linear_orthogonalization_time", []), dtype=np.float64).reshape(-1)
                    step_relres = np.asarray(npz.get("stats_step_newton_relres_end", []), dtype=np.float64).reshape(-1)
                    step_count = max(
                        step_newton.size,
                        step_linear.size,
                        step_linear_solve.size,
                        step_linear_prec.size,
                        step_linear_orth.size,
                        step_relres.size,
                    )
                    for idx in range(step_count):
                        newton_rows.append(
                            {
                                "case_id": meta["case_id"],
                                "case_label": meta["case_label"],
                                "level_id": meta["level_id"],
                                "level_label": meta["level_label"],
                                "level_order": meta["level_order"],
                                "engine": engine,
                                "variant": meta["variant"],
                                "step": idx + 1,
                                "newton_iterations": step_newton[idx] if idx < step_newton.size else "",
                                "linear_iterations": step_linear[idx] if idx < step_linear.size else "",
                                "linear_solve_time": step_linear_solve[idx] if idx < step_linear_solve.size else "",
                                "linear_preconditioner_time": step_linear_prec[idx] if idx < step_linear_prec.size else "",
                                "linear_orthogonalization_time": step_linear_orth[idx] if idx < step_linear_orth.size else "",
                                "newton_relres_end": step_relres[idx] if idx < step_relres.size else "",
                                "output_dir": str(output_dir),
                            }
                        )

        elif engine == "matlab":
            summary_json = output_dir / "summary.json"
            summary_h5 = output_dir / "summary.h5"
            if not summary_json.exists() or not summary_h5.exists():
                continue
            summary = read_json(summary_json)
            cont = summary.get("continuation", {})
            lambda_hist = np.asarray(cont.get("lambda_hist", []), dtype=np.float64).reshape(-1)
            omega_hist = np.asarray(cont.get("omega_hist", []), dtype=np.float64).reshape(-1)
            umax_hist = np.asarray(cont.get("umax_hist", []), dtype=np.float64).reshape(-1)
            stop_reason = str(cont.get("stop_reason", summary.get("run_info", {}).get("stop_reason", ""))).strip()

            runs_rows.append(
                {
                    "case_id": meta["case_id"],
                    "case_label": meta["case_label"],
                    "level_id": meta["level_id"],
                    "level_label": meta["level_label"],
                    "level_order": meta["level_order"],
                    "engine": engine,
                    "variant": meta["variant"],
                    "phase": phase,
                    "mesh_nodes": summary.get("mesh", {}).get("n_nodes", ""),
                    "mesh_elements": summary.get("mesh", {}).get("n_elements", ""),
                    "unknowns": summary.get("mesh", {}).get("n_unknown", ""),
                    "runtime_seconds": summary.get("run_info", {}).get("runtime_seconds", ""),
                    "accepted_steps": cont.get("accepted_steps", ""),
                    "final_lambda": cont.get("final_lambda", ""),
                    "final_omega": cont.get("final_omega", ""),
                    "stop_reason": stop_reason,
                    "omega_max_stop": meta["shared_omega_max_stop"],
                    "d_lambda_diff_scaled_min": meta["d_lambda_diff_scaled_min"],
                    "newton_residual_tol": meta.get("newton_residual_tol", case_tol_map.get(meta["case_id"], "")),
                    "newton_stopping_criterion": meta["newton_stopping_criterion"],
                    "newton_stopping_tol": meta["newton_stopping_tol"] if meta["newton_stopping_tol"] is not None else "",
                    "output_dir": str(output_dir),
                }
            )

            if phase == "main":
                for step_idx, (omega_value, lambda_value) in enumerate(zip(omega_hist, lambda_hist), start=1):
                    umax_value = umax_hist[step_idx - 1] if step_idx - 1 < umax_hist.size else ""
                    continuation_rows.append(
                        {
                            "case_id": meta["case_id"],
                            "case_label": meta["case_label"],
                            "level_id": meta["level_id"],
                            "level_label": meta["level_label"],
                            "level_order": meta["level_order"],
                            "engine": engine,
                            "variant": meta["variant"],
                            "step": step_idx,
                            "omega": omega_value,
                            "lambda": lambda_value,
                            "umax": umax_value,
                        }
                    )

                with h5py.File(summary_h5, "r") as h5:
                    step_newton = _load_h5_dataset(h5, "continuation/stats/step_newton_iterations").reshape(-1)
                    step_linear = _load_h5_dataset(h5, "continuation/stats/step_linear_iterations").reshape(-1)
                    step_linear_solve = _load_h5_dataset(h5, "continuation/stats/step_linear_solve_time").reshape(-1)
                    step_linear_prec = _load_h5_dataset(h5, "continuation/stats/step_linear_preconditioner_time").reshape(-1)
                    step_linear_orth = _load_h5_dataset(h5, "continuation/stats/step_linear_orthogonalization_time").reshape(-1)
                    step_relres = _load_h5_dataset(h5, "continuation/stats/step_newton_relres_end").reshape(-1)
                step_count = max(
                    step_newton.size,
                    step_linear.size,
                    step_linear_solve.size,
                    step_linear_prec.size,
                    step_linear_orth.size,
                    step_relres.size,
                )
                for idx in range(step_count):
                    newton_rows.append(
                        {
                            "case_id": meta["case_id"],
                            "case_label": meta["case_label"],
                            "level_id": meta["level_id"],
                            "level_label": meta["level_label"],
                            "level_order": meta["level_order"],
                            "engine": engine,
                            "variant": meta["variant"],
                            "step": idx + 1,
                            "newton_iterations": step_newton[idx] if idx < step_newton.size else "",
                            "linear_iterations": step_linear[idx] if idx < step_linear.size else "",
                            "linear_solve_time": step_linear_solve[idx] if idx < step_linear_solve.size else "",
                            "linear_preconditioner_time": step_linear_prec[idx] if idx < step_linear_prec.size else "",
                            "linear_orthogonalization_time": step_linear_orth[idx] if idx < step_linear_orth.size else "",
                            "newton_relres_end": step_relres[idx] if idx < step_relres.size else "",
                            "output_dir": str(output_dir),
                        }
                    )

    runs_rows.sort(key=lambda row: (row["case_id"], int(row["level_order"]), row["phase"], row["engine"], row["variant"]))
    continuation_rows.sort(key=lambda row: (row["case_id"], int(row["level_order"]), row["variant"], row["engine"], int(row["step"])))
    newton_rows.sort(key=lambda row: (row["case_id"], int(row["level_order"]), row["variant"], row["engine"], int(row["step"])))
    horizon_rows.sort(key=lambda row: row["case_id"])

    _write_csv(
        study["data_dir"] / "runs.csv",
        runs_rows,
        [
            "case_id",
            "case_label",
            "level_id",
            "level_label",
            "level_order",
            "engine",
            "variant",
            "phase",
            "mesh_nodes",
            "mesh_elements",
            "unknowns",
            "runtime_seconds",
            "accepted_steps",
            "final_lambda",
            "final_omega",
            "stop_reason",
            "omega_max_stop",
            "d_lambda_diff_scaled_min",
            "newton_residual_tol",
            "newton_stopping_criterion",
            "newton_stopping_tol",
            "output_dir",
        ],
    )
    _write_csv(
        study["data_dir"] / "continuation_curves.csv",
        continuation_rows,
        ["case_id", "case_label", "level_id", "level_label", "level_order", "engine", "variant", "step", "omega", "lambda", "umax"],
    )
    _write_csv(
        study["data_dir"] / "newton_step_stats.csv",
        newton_rows,
        [
            "case_id",
            "case_label",
            "level_id",
            "level_label",
            "level_order",
            "engine",
            "variant",
            "step",
            "newton_iterations",
            "linear_iterations",
            "linear_solve_time",
            "linear_preconditioner_time",
            "linear_orthogonalization_time",
            "newton_relres_end",
            "output_dir",
        ],
    )
    _write_csv(
        study["data_dir"] / "horizon_estimates.csv",
        horizon_rows,
        ["case_id", "case_label", "level_id", "level_label", "smoke_final_omega", "chosen_omega_max_stop", "runtime_seconds", "accepted_steps", "output_dir"],
    )


if __name__ == "__main__":
    main()
