#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = ROOT / "artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_smart_controller_v2_rank8_step100/data"
NEW_DIR = ROOT / "artifacts/p4_l1_smart_three_param_exact_omega_full/rank8_step100/data"
OUT_DIR = ROOT / "artifacts/p4_l1_smart_three_param_exact_omega_full/report"


def _load_case(data_dir: Path) -> dict:
    run_info = json.loads((data_dir / "run_info.json").read_text())
    npz = np.load(data_dir / "petsc_run.npz", allow_pickle=True)
    return {"run_info": run_info, "npz": npz}


def _arr(npz, key: str, dtype=float) -> np.ndarray:
    name = f"stats_{key}"
    if name not in npz:
        return np.asarray([], dtype=dtype)
    return np.asarray(npz[name], dtype=dtype)


def _summary(case: dict) -> dict:
    ri = case["run_info"]
    npz = case["npz"]
    timing = ri["timings"]["linear"]
    predictor = ri.get("predictor_diagnostics", {})
    lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64)
    omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64)
    step_idx = _arr(npz, "step_index", int)
    step_newton = _arr(npz, "step_newton_iterations", int)
    step_linear = _arr(npz, "step_linear_iterations", int)
    return {
        "runtime": float(ri["run_info"]["runtime_seconds"]),
        "accepted_steps": int(step_idx.size),
        "final_lambda": float(lambda_hist[-1]),
        "final_omega": float(omega_hist[-1]),
        "continuation_newton_total": int(np.sum(step_newton)),
        "continuation_linear_total": int(np.sum(step_linear)),
        "continuation_linear_per_newton": float(np.sum(step_linear) / max(np.sum(step_newton), 1)),
        "predictor_wall_total": float(predictor.get("step_predictor_wall_time_total", np.nan)),
        "predictor_u_diff_total": float(predictor.get("step_initial_guess_displacement_diff_volume_integral_total", np.nan)),
        "predictor_dev_diff_total": float(predictor.get("step_initial_guess_deviatoric_strain_diff_volume_integral_total", np.nan)),
        "secant_ref_u_diff_total": float(predictor.get("step_secant_reference_displacement_diff_volume_integral_total", np.nan)),
        "secant_ref_dev_diff_total": float(predictor.get("step_secant_reference_deviatoric_strain_diff_volume_integral_total", np.nan)),
        "linear_solve_total": float(timing["init_linear_solve_time"] + timing["attempt_linear_solve_time_total"]),
        "linear_prec_total": float(timing["init_linear_preconditioner_time"] + timing["attempt_linear_preconditioner_time_total"]),
        "linear_orth_total": float(timing["init_linear_orthogonalization_time"] + timing["attempt_linear_orthogonalization_time_total"]),
    }


def _plot_series(out_dir: Path, baseline: dict, new: dict) -> None:
    bnpz = baseline["npz"]
    nnpz = new["npz"]

    plt.figure(figsize=(7, 5))
    plt.plot(np.asarray(bnpz["omega_hist"], dtype=np.float64), np.asarray(bnpz["lambda_hist"], dtype=np.float64), marker="o", label="baseline secant")
    plt.plot(np.asarray(nnpz["omega_hist"], dtype=np.float64), np.asarray(nnpz["lambda_hist"], dtype=np.float64), marker="o", label="3-param exact-omega")
    plt.xlabel("omega")
    plt.ylabel("lambda")
    plt.title("P4(L1) Smart Controller: Lambda-Omega")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "lambda_omega.png", dpi=180)
    plt.close()

    for key, ylabel, fname in [
        ("step_wall_time", "Step wall time [s]", "step_wall_time.png"),
        ("step_newton_iterations", "Newton iterations", "step_newton_iterations.png"),
        ("step_linear_iterations", "Linear iterations", "step_linear_iterations.png"),
        ("step_predictor_wall_time", "Predictor wall time [s]", "predictor_wall_time.png"),
    ]:
        plt.figure(figsize=(7, 5))
        bx = _arr(bnpz, "step_index", int)
        nx = _arr(nnpz, "step_index", int)
        by = _arr(bnpz, key, float if "time" in key else int)
        ny = _arr(nnpz, key, float if "time" in key else int)
        plt.plot(bx, by, marker="o", label="baseline secant")
        plt.plot(nx, ny, marker="o", label="3-param exact-omega")
        plt.xlabel("Accepted continuation step")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=180)
        plt.close()

    plt.figure(figsize=(7, 5))
    bx = _arr(bnpz, "step_index", int)
    nx = _arr(nnpz, "step_index", int)
    bnewt = _arr(bnpz, "step_newton_iterations", float)
    nnewt = _arr(nnpz, "step_newton_iterations", float)
    blin = _arr(bnpz, "step_linear_iterations", float)
    nlin = _arr(nnpz, "step_linear_iterations", float)
    plt.plot(bx, blin / np.maximum(bnewt, 1.0), marker="o", label="baseline secant")
    plt.plot(nx, nlin / np.maximum(nnewt, 1.0), marker="o", label="3-param exact-omega")
    plt.xlabel("Accepted continuation step")
    plt.ylabel("Linear / Newton")
    plt.title("Linear iterations per Newton step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "linear_per_newton.png", dpi=180)
    plt.close()

    nx = _arr(nnpz, "step_index", int)
    n_actual_u = _arr(nnpz, "step_initial_guess_displacement_diff_volume_integral", float)
    n_secant_u = _arr(nnpz, "step_secant_reference_displacement_diff_volume_integral", float)
    n_actual_dev = _arr(nnpz, "step_initial_guess_deviatoric_strain_diff_volume_integral", float)
    n_secant_dev = _arr(nnpz, "step_secant_reference_deviatoric_strain_diff_volume_integral", float)

    plt.figure(figsize=(7, 5))
    plt.plot(nx, n_actual_u, marker="o", label="3-param actual predictor")
    plt.plot(nx, n_secant_u, marker="o", label="retrospective secant reference")
    plt.xlabel("Accepted continuation step")
    plt.ylabel("Displacement diff integral")
    plt.title("Predictor vs retrospective secant: displacement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "predictor_vs_secant_u_diff.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(nx, n_actual_dev, marker="o", label="3-param actual predictor")
    plt.plot(nx, n_secant_dev, marker="o", label="retrospective secant reference")
    plt.xlabel("Accepted continuation step")
    plt.ylabel("Deviatoric strain diff integral")
    plt.title("Predictor vs retrospective secant: deviatoric strain")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "predictor_vs_secant_dev_diff.png", dpi=180)
    plt.close()

    bsum = _summary(baseline)
    nsum = _summary(new)
    plt.figure(figsize=(6, 5))
    plt.scatter([bsum["runtime"], nsum["runtime"]], [bsum["final_lambda"], nsum["final_lambda"]], s=90)
    for x, y, label in [
        (bsum["runtime"], bsum["final_lambda"], "baseline secant"),
        (nsum["runtime"], nsum["final_lambda"], "3-param exact-omega"),
    ]:
        plt.annotate(label, (x, y), xytext=(5, 5), textcoords="offset points")
    plt.xscale("log")
    plt.xlabel("Runtime [s] (log scale)")
    plt.ylabel("Final lambda")
    plt.title("Final lambda vs runtime")
    plt.tight_layout()
    plt.savefig(out_dir / "final_lambda_vs_runtime.png", dpi=180)
    plt.close()


def main() -> None:
    baseline = _load_case(BASE_DIR)
    new = _load_case(NEW_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_series(OUT_DIR, baseline, new)
    bsum = _summary(baseline)
    nsum = _summary(new)

    def _fmt_ratio(a: float, b: float) -> str:
        if not np.isfinite(a) or not np.isfinite(b) or b == 0.0:
            return "nan"
        return f"{a / b:.3f}x"

    report = f"""# P4(L1) Smart Controller: Baseline Secant vs 3-Parameter Exact-omega

Baseline artifact:
- [run_info.json]({BASE_DIR.relative_to(OUT_DIR.parent).as_posix()}/run_info.json)

New artifact:
- [run_info.json]({NEW_DIR.relative_to(OUT_DIR.parent).as_posix()}/run_info.json)

## Totals

| Metric | Baseline secant | 3-param exact-omega | Ratio new / base |
| --- | ---: | ---: | ---: |
| Runtime [s] | `{bsum['runtime']:.3f}` | `{nsum['runtime']:.3f}` | `{_fmt_ratio(nsum['runtime'], bsum['runtime'])}` |
| Accepted continuation steps | `{bsum['accepted_steps']}` | `{nsum['accepted_steps']}` | `{_fmt_ratio(float(nsum['accepted_steps']), float(bsum['accepted_steps']))}` |
| Final omega | `{bsum['final_omega']:.6e}` | `{nsum['final_omega']:.6e}` | `-` |
| Final lambda | `{bsum['final_lambda']:.9f}` | `{nsum['final_lambda']:.9f}` | `-` |
| Continuation Newton total | `{bsum['continuation_newton_total']}` | `{nsum['continuation_newton_total']}` | `{_fmt_ratio(float(nsum['continuation_newton_total']), float(bsum['continuation_newton_total']))}` |
| Continuation linear total | `{bsum['continuation_linear_total']}` | `{nsum['continuation_linear_total']}` | `{_fmt_ratio(float(nsum['continuation_linear_total']), float(bsum['continuation_linear_total']))}` |
| Linear / Newton | `{bsum['continuation_linear_per_newton']:.3f}` | `{nsum['continuation_linear_per_newton']:.3f}` | `{_fmt_ratio(nsum['continuation_linear_per_newton'], bsum['continuation_linear_per_newton'])}` |
| Linear solve total [s] | `{bsum['linear_solve_total']:.3f}` | `{nsum['linear_solve_total']:.3f}` | `{_fmt_ratio(nsum['linear_solve_total'], bsum['linear_solve_total'])}` |
| PC apply/setup total [s] | `{bsum['linear_prec_total']:.3f}` | `{nsum['linear_prec_total']:.3f}` | `{_fmt_ratio(nsum['linear_prec_total'], bsum['linear_prec_total'])}` |
| Orthogonalization total [s] | `{bsum['linear_orth_total']:.3f}` | `{nsum['linear_orth_total']:.3f}` | `{_fmt_ratio(nsum['linear_orth_total'], bsum['linear_orth_total'])}` |
| Predictor wall total [s] | `{bsum['predictor_wall_total']:.3f}` | `{nsum['predictor_wall_total']:.3f}` | `{_fmt_ratio(nsum['predictor_wall_total'], max(bsum['predictor_wall_total'], 1.0e-30))}` |

## New Run Predictor-vs-Secant Diagnostics

| Metric | 3-param actual | Retrospective secant reference |
| --- | ---: | ---: |
| Displacement diff integral total | `{nsum['predictor_u_diff_total']:.3f}` | `{nsum['secant_ref_u_diff_total']:.3f}` |
| Deviatoric diff integral total | `{nsum['predictor_dev_diff_total']:.3f}` | `{nsum['secant_ref_dev_diff_total']:.3f}` |

## Plots

![Lambda Omega](lambda_omega.png)

![Step Wall Time](step_wall_time.png)

![Step Newton Iterations](step_newton_iterations.png)

![Step Linear Iterations](step_linear_iterations.png)

![Linear per Newton](linear_per_newton.png)

![Predictor vs Secant U Diff](predictor_vs_secant_u_diff.png)

![Predictor vs Secant Deviatoric Diff](predictor_vs_secant_dev_diff.png)

![Predictor Wall Time](predictor_wall_time.png)

![Final Lambda vs Runtime](final_lambda_vs_runtime.png)
"""
    (OUT_DIR / "README.md").write_text(report)
    (OUT_DIR / "summary.json").write_text(json.dumps({"baseline": bsum, "new": nsum}, indent=2))


if __name__ == "__main__":
    main()
