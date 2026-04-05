#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from petsc4py import PETSc

from slope_stability.cli.run_3D_hetero_SSR_capture import (
    _collector_delta,
    _collector_snapshot,
    _newton_guess_difference_volume_integrals,
)
from slope_stability.continuation.indirect import (
    _free_dot,
    _predictor_residual_penalty_merit,
    _secant_predictor,
    _three_param_penalty_predictor,
)
from slope_stability.nonlinear.newton import newton_ind_ssr


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "artifacts/p4_l1_three_param_bad_steps"
STATE_DIR = ROOT / "artifacts/p4_l1_alpha_refine_compare/rank8_secant_step12/data"
REPLAY_SCRIPT = ROOT / "benchmarks/slope_stability_3D_hetero_SSR_default/archive/replay_p4_l1_step13_predictor_compare.py"
TARGET_STEPS = (4, 5, 6, 7, 8)


def _load_replay_module():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("replay_step13_mod", REPLAY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load replay helper from {REPLAY_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["replay_step13_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def _predictor_merit(
    *,
    U: np.ndarray,
    lambda_ini: float,
    omega_target: float,
    q_mask: np.ndarray,
    f_v: np.ndarray,
    const_builder,
) -> tuple[float, float, float]:
    merit = _predictor_residual_penalty_merit(
        U=np.asarray(U, dtype=np.float64),
        lambda_value=float(lambda_ini),
        omega_target=float(omega_target),
        Q=q_mask,
        f=f_v,
        constitutive_matrix_builder=const_builder,
        penalty_weight=0.0,
    )
    omega_now = _free_dot(f_v, U, q_mask)
    omega_diff = float(omega_now - float(omega_target))
    return float(merit), omega_now, omega_diff


def _build_step_inputs(step_u: np.ndarray, omega_hist: np.ndarray, lambda_hist: np.ndarray, target_step: int) -> dict[str, object]:
    if target_step < 4:
        raise ValueError("Target step must be at least 4 for the three-parameter predictor replay.")
    if target_step > len(step_u):
        raise ValueError(f"Target step {target_step} exceeds saved state history length {len(step_u)}.")
    current_index = target_step - 2
    previous_index = target_step - 3
    return {
        "U_i": np.asarray(step_u[current_index], dtype=np.float64),
        "U_im1": np.asarray(step_u[previous_index], dtype=np.float64),
        "omega_now": float(omega_hist[current_index]),
        "omega_old": float(omega_hist[previous_index]),
        "lambda_now": float(lambda_hist[current_index]),
        "omega_target": float(omega_hist[target_step - 1]),
        "predictor_u_hist": tuple(np.asarray(v, dtype=np.float64) for v in step_u[: target_step - 1]),
    }


def _run_case_for_step(mod, run_info: dict, target_step: int, predictor: str) -> dict[str, object]:
    state_npz = np.load(STATE_DIR / "petsc_run.npz", allow_pickle=True)
    step_u = np.asarray(state_npz["step_U"], dtype=np.float64)
    omega_hist = np.asarray(state_npz["omega_hist"], dtype=np.float64)
    lambda_hist = np.asarray(state_npz["lambda_hist"], dtype=np.float64)
    step_inputs = _build_step_inputs(step_u, omega_hist, lambda_hist, target_step)

    built = mod._build_case(run_info)
    coord = built["coord"]
    elem = built["elem"]
    q_mask = built["q_mask"]
    f_v = built["f_V"]
    k_elast = built["K_elast"]
    const_builder = built["const_builder"]
    solver = built["solver"]

    U_i = np.asarray(step_inputs["U_i"], dtype=np.float64)
    U_im1 = np.asarray(step_inputs["U_im1"], dtype=np.float64)
    omega_now = float(step_inputs["omega_now"])
    omega_old = float(step_inputs["omega_old"])
    lambda_now = float(step_inputs["lambda_now"])
    omega_target = float(step_inputs["omega_target"])

    if predictor == "secant":
        t0 = perf_counter()
        U_ini, lambda_ini, predictor_kind = _secant_predictor(
            omega_old=omega_old,
            omega=omega_now,
            omega_target=omega_target,
            U_old=U_im1,
            U=U_i,
            lambda_value=lambda_now,
        )
        predictor_info = {
            "predictor_alpha": float((omega_target - omega_now) / (omega_now - omega_old)),
            "predictor_beta": np.nan,
            "predictor_gamma": np.nan,
            "energy_eval_count": np.nan,
            "predictor_wall_time": float(perf_counter() - t0),
        }
    elif predictor == "three_param_penalty":
        U_ini, lambda_ini, predictor_kind, predictor_info = _three_param_penalty_predictor(
            omega_old=omega_old,
            omega=omega_now,
            omega_target=omega_target,
            U_old=U_im1,
            U=U_i,
            lambda_value=lambda_now,
            predictor_u_hist=step_inputs["predictor_u_hist"],
            Q=q_mask,
            f=f_v,
            constitutive_matrix_builder=const_builder,
        )
    else:
        raise ValueError(f"Unsupported predictor {predictor!r}")

    merit, omega_pred, omega_diff = _predictor_merit(
        U=U_ini,
        lambda_ini=float(lambda_ini),
        omega_target=float(omega_target),
        q_mask=q_mask,
        f_v=f_v,
        const_builder=const_builder,
    )

    snap_before = _collector_snapshot(solver)
    t_newton = perf_counter()
    U_sol, lambda_sol, flag, it_newt, _history = newton_ind_ssr(
        U_ini,
        omega_target,
        lambda_ini,
        int(run_info["params"]["it_newt_max"]),
        int(run_info["params"]["it_damp_max"]),
        float(run_info["params"]["tol"]),
        float(run_info["params"]["r_min"]),
        k_elast,
        q_mask,
        f_v,
        const_builder,
        solver,
    )
    newton_wall = float(perf_counter() - t_newton)
    snap_after = _collector_snapshot(solver)
    delta = _collector_delta(snap_before, snap_after)
    guess_diag = _newton_guess_difference_volume_integrals(coord, elem, str(run_info["params"]["elem_type"]), U_ini, U_sol)

    close_solver = getattr(solver, "close", None)
    if callable(close_solver):
        close_solver()

    return {
        "target_step": int(target_step),
        "predictor_kind": str(predictor_kind),
        "predictor_alpha": float(predictor_info.get("predictor_alpha", np.nan)),
        "predictor_beta": float(predictor_info.get("predictor_beta", np.nan)),
        "predictor_gamma": float(predictor_info.get("predictor_gamma", np.nan)),
        "predictor_eval_count": float(predictor_info.get("energy_eval_count", np.nan)),
        "predictor_wall_time": float(predictor_info.get("predictor_wall_time", np.nan)),
        "predictor_merit": float(merit),
        "predictor_omega": float(omega_pred),
        "predictor_omega_diff": float(omega_diff),
        "lambda_initial": float(lambda_ini),
        "lambda_solution": float(lambda_sol),
        "newton_flag": int(flag),
        "newton_iterations": int(it_newt),
        "newton_wall_time": float(newton_wall),
        "linear_iterations": int(delta["iterations"]),
        "linear_solve_time": float(delta["solve_time"]),
        "linear_preconditioner_time": float(delta["preconditioner_time"]),
        "linear_orthogonalization_time": float(delta["orthogonalization_time"]),
        "u_init_to_solution_displacement_integral": float(guess_diag["displacement_diff_volume_integral"]),
        "u_init_to_solution_deviatoric_integral": float(guess_diag["deviatoric_strain_diff_volume_integral"]),
    }


def _write_report(results: dict[int, dict[str, dict[str, object]]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))

    lines: list[str] = []
    lines.append("# P4(L1) Bad-Step Replay: Secant vs 3-Term Predictor")
    lines.append("")
    lines.append("Source saved secant branch:")
    lines.append(f"- [run_info.json]({STATE_DIR / 'run_info.json'})")
    lines.append("")
    lines.append("This compares isolated one-step replays on the saved secant branch for steps where the killed full 3-term continuation was clearly worse than secant.")
    lines.append("Each row below is a cold rank-8 replay from the same saved state for two predictors:")
    lines.append("- standard secant")
    lines.append("- corrected exact-omega 3-term predictor")
    lines.append("")
    lines.append("| Step | Predictor | Merit | Newton iters | Linear iters | `u_ini -> u_final` disp. int. | `u_ini -> u_final` dev. int. | Predictor time [s] | Newton wall [s] | alpha | beta | gamma |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for step in sorted(results):
        for key in ("secant", "three_param_penalty"):
            row = results[step][key]
            lines.append(
                f"| {step} | `{row['predictor_kind']}` | "
                f"{row['predictor_merit']:.6e} | {row['newton_iterations']} | {row['linear_iterations']} | "
                f"{row['u_init_to_solution_displacement_integral']:.3f} | {row['u_init_to_solution_deviatoric_integral']:.3f} | "
                f"{row['predictor_wall_time']:.3f} | {row['newton_wall_time']:.3f} | "
                f"{row['predictor_alpha']:.6f} | {row['predictor_beta']:.6f} | {row['predictor_gamma']:.6f} |"
            )
        sec = results[step]["secant"]
        tri = results[step]["three_param_penalty"]
        lines.append(
            f"| {step} | `delta (3-term - secant)` | "
            f"{(tri['predictor_merit'] - sec['predictor_merit']):.6e} | "
            f"{tri['newton_iterations'] - sec['newton_iterations']} | "
            f"{tri['linear_iterations'] - sec['linear_iterations']} | "
            f"{tri['u_init_to_solution_displacement_integral'] - sec['u_init_to_solution_displacement_integral']:.3f} | "
            f"{tri['u_init_to_solution_deviatoric_integral'] - sec['u_init_to_solution_deviatoric_integral']:.3f} | "
            f"{tri['predictor_wall_time'] - sec['predictor_wall_time']:.3f} | "
            f"{tri['newton_wall_time'] - sec['newton_wall_time']:.3f} | - | - | - |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Merit is the current corrected predictor merit: residual-only `res_rel^2` on the exact-`omega` manifold.")
    lines.append("- Both predictors satisfy the target `omega` for these replays, so the constraint part is zero.")
    lines.append("- These are isolated one-step replays from the saved secant branch, not slices from the killed full 3-term continuation.")
    (OUT_DIR / "README.md").write_text("\n".join(lines))


def main() -> None:
    mod = _load_replay_module()
    run_info = json.loads((STATE_DIR / "run_info.json").read_text())
    results: dict[int, dict[str, dict[str, object]]] = {}
    for step in TARGET_STEPS:
        if PETSc.COMM_WORLD.rank == 0:
            print(f"[replay] target accepted step {step}")
        results[step] = {
            "secant": _run_case_for_step(mod, run_info, step, "secant"),
            "three_param_penalty": _run_case_for_step(mod, run_info, step, "three_param_penalty"),
        }
    if PETSc.COMM_WORLD.rank == 0:
        _write_report(results)


if __name__ == "__main__":
    main()
