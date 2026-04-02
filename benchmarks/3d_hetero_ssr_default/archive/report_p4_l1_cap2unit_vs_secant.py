#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
MICRO_ROOT = ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk"
SECANT_PROGRESS = ROOT / "artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_rank8_step12/data/progress.jsonl"
OUT_DIR = MICRO_ROOT / "cap2unit_vs_secant_report"
PLOTS_DIR = OUT_DIR / "plots"
OMEGA_HORIZON = 6.7e6

CASES = [
    {
        "key": "rank8_micro100_adaptive_cap652_negdl_rrb_roll10",
        "label": "Older Conservative Roll10, Stall x0.8",
        "path": MICRO_ROOT / "rank8_micro100_adaptive_cap652_negdl_rrb_roll10/data/progress.jsonl",
        "color": "#2ca02c",
    },
    {
        "key": "rank8_micro100_adaptive_cap652_negdl_rrb_roll10_stall0p9",
        "label": "Conservative Roll10, Stall x0.9",
        "path": MICRO_ROOT / "rank8_micro100_adaptive_cap652_negdl_rrb_roll10_stall0p9/data/progress.jsonl",
        "color": "#17becf",
    },
    {
        "key": "rank8_micro100_adaptive_cap652_negdl_rrb_roll10_stall0p9_floor",
        "label": "Conservative Roll10, Stall x0.9, Floor",
        "path": MICRO_ROOT / "rank8_micro100_adaptive_cap652_negdl_rrb_roll10_stall0p9_floor/data/progress.jsonl",
        "color": "#8c564b",
    },
    {
        "key": "full_rel_fixedomega_currentstate_omega6p7e6_cap2unit_absgate",
        "label": "Cap 2.0, Absolute Gate, Bug-Fixed",
        "path": MICRO_ROOT / "full_rel_fixedomega_currentstate_omega6p7e6_cap2unit_absgate/data/progress.jsonl",
        "color": "#d62728",
    },
    {
        "key": "full_rel_fixedomega_currentstate_omega6p7e6_cap2unit_absgate_growth1p35_shrink0p9_floor",
        "label": "Cap 2.0, Absolute Gate, Growth x1.35, Stall x0.9, Floor",
        "path": MICRO_ROOT
        / "full_rel_fixedomega_currentstate_omega6p7e6_cap2unit_absgate_growth1p35_shrink0p9_floor/data/progress.jsonl",
        "color": "#1f77b4",
    },
]


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _save_plot(fig: plt.Figure, filename: str) -> str:
    out = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return filename


def _safe_interp(x_new: float, x: np.ndarray, y: np.ndarray) -> float | None:
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return None
    xf = x[finite]
    yf = y[finite]
    order = np.argsort(xf)
    xf = xf[order]
    yf = yf[order]
    if x_new < xf[0] or x_new > xf[-1]:
        return None
    return float(np.interp(x_new, xf, yf))


def _secant_curve() -> dict[str, object]:
    records = _load_jsonl(SECANT_PROGRESS)
    init = next(r for r in records if r.get("event") == "init_complete")
    accepted = [r for r in records if r.get("event") == "step_accepted"]

    init_time = float(init.get("total_wall_time", 0.0))
    omega = [float(init["omega_hist"][1])]
    lam = [float(init["lambda_hist"][1])]
    time = [0.0]
    accepted_step = [1]
    for step in accepted:
        omega.append(float(step["omega_value"]))
        lam.append(float(step["lambda_value"]))
        time.append(float(step["total_wall_time"]) - init_time)
        accepted_step.append(int(step["accepted_step"]))

    omega_arr = np.asarray(omega, dtype=np.float64)
    lam_arr = np.asarray(lam, dtype=np.float64)
    time_arr = np.asarray(time, dtype=np.float64)

    uniq_o: list[float] = []
    uniq_l: list[float] = []
    uniq_t: list[float] = []
    for o, l, t in zip(omega_arr, lam_arr, time_arr):
        if not uniq_o or o > uniq_o[-1] + 1.0e-9:
            uniq_o.append(float(o))
            uniq_l.append(float(l))
            uniq_t.append(float(t))

    return {
        "label": "Original Secant",
        "color": "black",
        "omega": np.asarray(uniq_o, dtype=np.float64),
        "lambda": np.asarray(uniq_l, dtype=np.float64),
        "time": np.asarray(uniq_t, dtype=np.float64),
        "accepted_step": np.asarray(accepted_step[: len(uniq_o)], dtype=np.float64),
        "path": str(SECANT_PROGRESS),
        "reached_horizon": float(np.max(uniq_o)) >= OMEGA_HORIZON - 1.0e-9,
        "time_to_horizon": _safe_interp(OMEGA_HORIZON, np.asarray(uniq_o, dtype=np.float64), np.asarray(uniq_t, dtype=np.float64)),
    }


def _settings_summary(init: dict[str, object]) -> str:
    parts: list[str] = []
    mode = init.get("mode")
    if mode:
        parts.append(str(mode))
    if "fixed_omega_use_current_state" in init:
        parts.append("fixed-omega=current-state" if bool(init["fixed_omega_use_current_state"]) else "fixed-omega=secant")
    roll = init.get("rolling_basis_size")
    if roll is not None:
        comp = str(init.get("basis_compression", "none"))
        suffix = "" if comp == "none" else f"+{comp}"
        parts.append(f"roll{int(roll)}{suffix}")
    if bool(init.get("stall_recovery", False)):
        parts.append("stall-recovery")
    gate = str(init.get("correction_gate_mode", ""))
    if gate == "relative":
        parts.append(f"rel-gate={float(init.get('correction_rel_threshold', math.nan)):.3g}")
    elif gate == "absolute":
        parts.append(f"abs-gate={float(init.get('correction_abs_threshold', math.nan)):.3g}")
    growth = init.get("omega_growth_factor")
    if growth is not None:
        parts.append(f"growth x{float(growth):.3g}")
    if "step_length_cap_mode" in init and str(init["step_length_cap_mode"]) != "none":
        parts.append(
            f"cap {init['step_length_cap_mode']} x{float(init.get('step_length_cap_factor', 1.0)):.3g}"
        )
    shrink = init.get("no_progress_shrink_factor")
    if shrink is not None:
        parts.append(f"no-progress x{float(shrink):.3g}")
    return ", ".join(parts)


def _load_micro_case(spec: dict[str, object], secant: dict[str, object]) -> dict[str, object]:
    path = Path(spec["path"])
    records = _load_jsonl(path)
    init = next(r for r in records if r.get("event") == "init_complete")
    steps = [r for r in records if r.get("event") == "micro_step_complete"]
    final = next((r for r in reversed(records) if r.get("event") == "run_complete"), None)
    if final is None:
        last = steps[-1]
        final = {
            "event": "run_complete_partial",
            "stop_reason": "manual_stop_or_partial",
            "final_omega": float(last.get("committed_omega_after", last.get("omega_after", math.nan))),
            "final_lambda": float(last.get("committed_lambda_after", last.get("lambda_after", math.nan))),
            "runtime_seconds": float(last.get("total_wall_time", math.nan)),
            "micro_steps_completed": int(len(steps)),
        }

    omega = np.asarray(
        [float(step.get("committed_omega_after", step.get("omega_after", math.nan))) for step in steps],
        dtype=np.float64,
    )
    lam = np.asarray(
        [float(step.get("committed_lambda_after", step.get("lambda_after", math.nan))) for step in steps],
        dtype=np.float64,
    )
    total_wall = np.asarray([float(step.get("total_wall_time", math.nan)) for step in steps], dtype=np.float64)
    linear = np.asarray([float(step.get("linear_iterations", 0.0)) for step in steps], dtype=np.float64)
    rel_after = np.asarray([float(step.get("rel_residual_after", math.nan)) for step in steps], dtype=np.float64)
    domega_used = np.asarray([float(step.get("domega_used", 0.0)) for step in steps], dtype=np.float64)
    correction = np.asarray([float(step.get("correction_free_norm", math.nan)) for step in steps], dtype=np.float64)

    sec_o = np.asarray(secant["omega"], dtype=np.float64)
    sec_l = np.asarray(secant["lambda"], dtype=np.float64)
    sec_ref = np.interp(omega, sec_o, sec_l)
    deviation = lam - sec_ref

    init_time = float(init.get("total_wall_time", 0.0))
    init_omega = float(init["omega_hist"][1])
    init_lambda = float(init["lambda_hist"][1])
    cont_time = total_wall - init_time

    omega_time = np.concatenate(([init_omega], omega))
    lambda_time = np.concatenate(([init_lambda], lam))
    cont_time_with_init = np.concatenate(([0.0], cont_time))

    max_omega = float(np.nanmax(omega)) if omega.size else math.nan
    max_lambda = float(np.nanmax(lam)) if lam.size else math.nan
    overshoot = np.maximum(deviation, 0.0)
    time_to_horizon = _safe_interp(OMEGA_HORIZON, omega_time, cont_time_with_init)
    reached_horizon = time_to_horizon is not None

    return {
        "key": str(spec["key"]),
        "label": str(spec["label"]),
        "color": str(spec["color"]),
        "path": str(path),
        "init": init,
        "final": final,
        "omega": omega,
        "lambda": lam,
        "cont_time": cont_time,
        "omega_time": omega_time,
        "lambda_time": lambda_time,
        "cont_time_with_init": cont_time_with_init,
        "micro_idx": np.arange(1, len(steps) + 1, dtype=np.float64),
        "linear_iterations": linear,
        "rel_after": rel_after,
        "domega_used": domega_used,
        "correction_norm": correction,
        "deviation": deviation,
        "rmse_to_secant": float(np.sqrt(np.mean(deviation**2))) if deviation.size else math.nan,
        "max_abs_dev": float(np.max(np.abs(deviation))) if deviation.size else math.nan,
        "max_pos_dev": float(np.max(overshoot)) if overshoot.size else 0.0,
        "max_omega": max_omega,
        "max_lambda": max_lambda,
        "linear_total": int(np.nansum(linear)),
        "zero_domega_steps": int(np.sum(np.abs(domega_used) <= 1.0e-12)),
        "reached_horizon": reached_horizon,
        "time_to_horizon": time_to_horizon,
        "linear_not_converged_count": int(
            sum(bool(step.get("linear_any_not_converged", False)) for step in steps if "linear_any_not_converged" in step)
        ),
        "linear_hit_max_count": int(
            sum(bool(step.get("linear_any_hit_max_iterations", False)) for step in steps if "linear_any_hit_max_iterations" in step)
        ),
        "settings_summary": _settings_summary(init),
    }


def _plot_lambda_vs_omega(secant: dict[str, object], cases: list[dict[str, object]]) -> str:
    fig = plt.figure(figsize=(9.2, 6.0), dpi=180)
    plt.plot(secant["omega"] / 1.0e6, secant["lambda"], color="black", linewidth=2.2, label="Original Secant")
    for case in cases:
        plt.plot(case["omega"] / 1.0e6, case["lambda"], marker="o", markersize=3.0, linewidth=1.4, color=case["color"], label=case["label"])
    plt.xlabel(r"$\omega$ [$10^6$]")
    plt.ylabel(r"$\lambda$")
    plt.title(r"Micro-Walk Variants Against the Original Secant Branch")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    return _save_plot(fig, "lambda_vs_omega.png")


def _plot_time_curve(
    secant: dict[str, object],
    cases: list[dict[str, object]],
    *,
    y_key: str,
    ylabel: str,
    title: str,
    filename: str,
) -> str:
    fig = plt.figure(figsize=(9.2, 6.0), dpi=180)
    plt.plot(secant["time"], secant[y_key], color="black", linewidth=2.2, marker="o", markersize=3.5, label="Original Secant")
    for case in cases:
        plt.plot(case["cont_time_with_init"], case[y_key + "_time" if y_key in {"omega", "lambda"} else y_key], marker="o", markersize=3.0, linewidth=1.4, color=case["color"], label=case["label"])
    plt.xlabel("Continuation Wall Time After Init [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    return _save_plot(fig, filename)


def _plot_lambda_deviation(secant: dict[str, object], cases: list[dict[str, object]]) -> str:
    fig = plt.figure(figsize=(9.2, 6.0), dpi=180)
    for case in cases:
        plt.plot(case["omega"] / 1.0e6, case["deviation"], marker="o", markersize=3.0, linewidth=1.4, color=case["color"], label=case["label"])
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel(r"$\omega$ [$10^6$]")
    plt.ylabel(r"$\lambda - \lambda_{\mathrm{secant}}(\omega)$")
    plt.title("Deviation from the Secant Branch at the Same Omega")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    return _save_plot(fig, "lambda_deviation_vs_omega.png")


def _plot_micro_series(
    cases: list[dict[str, object]],
    *,
    y_key: str,
    ylabel: str,
    title: str,
    filename: str,
    logy: bool = False,
    hlines: list[tuple[float, str, str]] | None = None,
) -> str:
    fig = plt.figure(figsize=(9.2, 6.0), dpi=180)
    for case in cases:
        x = np.asarray(case["micro_idx"], dtype=np.float64)
        y = np.asarray(case[y_key], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            continue
        plt.plot(x[finite], y[finite], marker="o", markersize=3.0, linewidth=1.4, color=case["color"], label=case["label"])
    if hlines:
        seen: set[str] = set()
        for y, label, style in hlines:
            shown = label if label not in seen else "_nolegend_"
            plt.axhline(float(y), color="0.25", linestyle=style, linewidth=1.0, alpha=0.8, label=shown)
            seen.add(label)
    if logy:
        plt.yscale("log")
    plt.xlabel("Micro-Step Index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    return _save_plot(fig, filename)


def _build_summary(secant: dict[str, object], cases: list[dict[str, object]]) -> dict[str, object]:
    return {
        "secant_reference": {
            "path": secant["path"],
            "omega_max": float(np.max(secant["omega"])),
            "lambda_final": float(secant["lambda"][-1]),
            "time_to_6p7e6": secant["time_to_horizon"],
        },
        "cases": [
            {
                "key": case["key"],
                "label": case["label"],
                "path": case["path"],
                "max_omega": case["max_omega"],
                "max_lambda": case["max_lambda"],
                "rmse_to_secant": case["rmse_to_secant"],
                "max_pos_dev": case["max_pos_dev"],
                "max_abs_dev": case["max_abs_dev"],
                "linear_total": case["linear_total"],
                "zero_domega_steps": case["zero_domega_steps"],
                "reached_horizon": case["reached_horizon"],
                "time_to_6p7e6": case["time_to_horizon"],
                "linear_not_converged_count": case["linear_not_converged_count"],
                "linear_hit_max_count": case["linear_hit_max_count"],
                "settings_summary": case["settings_summary"],
                "final": case["final"],
            }
            for case in cases
        ],
    }


def main() -> None:
    _ensure_dirs()
    secant = _secant_curve()
    cases = [_load_micro_case(spec, secant) for spec in CASES]

    plots = {
        "lambda_vs_omega": _plot_lambda_vs_omega(secant, cases),
        "omega_vs_time": _plot_time_curve(
            secant,
            cases,
            y_key="omega",
            ylabel=r"$\omega$",
            title=r"Omega vs Continuation Time After Init",
            filename="omega_vs_time.png",
        ),
        "lambda_vs_time": _plot_time_curve(
            secant,
            cases,
            y_key="lambda",
            ylabel=r"$\lambda$",
            title=r"Lambda vs Continuation Time After Init",
            filename="lambda_vs_time.png",
        ),
        "lambda_deviation": _plot_lambda_deviation(secant, cases),
        "rel_after": _plot_micro_series(
            cases,
            y_key="rel_after",
            ylabel="Relative Residual After",
            title="Micro-Step Residual After Each Outer Step",
            filename="rel_after_vs_step.png",
            logy=True,
            hlines=[(1.0e-2, "1e-2", ":"), (1.0e-1, "1e-1", "--"), (1.0, "1", "-.")],
        ),
        "domega_used": _plot_micro_series(
            cases,
            y_key="domega_used",
            ylabel=r"$d\omega$ used",
            title=r"Used $\Delta\omega$ by Micro-Step",
            filename="domega_used_vs_step.png",
        ),
        "correction_norm": _plot_micro_series(
            cases,
            y_key="correction_norm",
            ylabel=r"$\|\Delta U\|_{\mathrm{free}}$",
            title="Absolute Free-DOF Correction Norm",
            filename="correction_norm_vs_step.png",
            hlines=[(1.0, "absolute gate 1.0", "--")],
        ),
    }

    summary = _build_summary(secant, cases)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cases_by_quality = sorted(
        cases,
        key=lambda c: (
            0 if c["reached_horizon"] else 1,
            -(c["time_to_horizon"] or -1.0),
            -float(c["max_omega"]),
            float(c["rmse_to_secant"]),
        ),
    )

    lines: list[str] = []
    lines.append("# Capped Absolute-Gate Variants vs Secant\n\n")
    lines.append("This report compares the capped absolute-gate reruns, the conservative `roll10` family, and the true secant continuation.\n\n")
    lines.append("Time plots use continuation-only wall time: each curve starts at the common post-init state `(lambda=1.1)` with its own init wall subtracted away.\n\n")
    lines.append("## Main Result\n\n")
    lines.append("- The earlier bug-fixed `cap 2.0 + absolute gate` run fixed the no-progress growth bug, but then decayed into tiny restart steps near `omega ≈ 6.4537e6`.\n")
    lines.append("- The new rerun changes only the controller percentages: growth `x1.35`, no-progress shrink `x0.9`, and a global floor `d_omega >= d_omega_initial`.\n")
    lines.append("- In the stored result, that new rerun does push slightly farther than the previous cap2.0 absolute-gate run (`6.4664e6` vs `6.4537e6`), but it still does not beat the older conservative `roll10` run on coverage or cost.\n")
    lines.append("- The new conservative `roll10` rerun with stall shrink `x0.9` is a negative result: it stalls earlier than the stored `x0.8` version and collapses into tiny restart steps around `omega ≈ 6.3823e6`.\n")
    lines.append("- Adding a floor `d_omega >= d_omega_initial` improves that `x0.9` conservative run, but it still settles into a fixed-omega repair loop around `omega ≈ 6.4365e6` and does not recover the original `x0.8` coverage.\n")
    lines.append("- The comparison below focuses only on the capped/controlled variants, as requested, with secant shown as the reference branch.\n\n")
    lines.append("Time-to-horizon in the table below is only a trajectory metric. The micro-walk variants are not fully converged accepted-step continuations, so those times are not directly comparable to the fully converged secant wall time.\n\n")
    lines.append("## Comparison Table\n\n")
    lines.append("| Case | Max omega | Reached 6.7e6 | Time to 6.7e6 [s] | RMSE to secant | Max +lambda overshoot | Linear total | Zero-domega steps | Linear nonconv. | Settings |\n")
    lines.append("| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
    for case in cases:
        time_to_h = case["time_to_horizon"]
        lines.append(
            "| "
            + f"{case['label']} | {case['max_omega'] / 1.0e6:.6f}e6 | "
            + ("yes" if case["reached_horizon"] else "no")
            + " | "
            + (f"{time_to_h:.1f}" if time_to_h is not None else "n/a")
            + f" | {case['rmse_to_secant']:.4f} | {case['max_pos_dev']:.4f} | {case['linear_total']} | "
            + f"{case['zero_domega_steps']} | {case['linear_not_converged_count']} | {case['settings_summary']} |\n"
        )
    lines.append(
        "| Original Secant | "
        + f"{float(np.max(secant['omega'])) / 1.0e6:.6f}e6 | "
        + ("yes" if secant["reached_horizon"] else "no")
        + " | "
        + (f"{secant['time_to_horizon']:.1f}" if secant["time_to_horizon"] is not None else "n/a")
        + " | 0.0000 | 0.0000 | n/a | n/a | n/a | production reference |\n"
    )
    lines.append("\n")
    lines.append("## What Worked Best\n\n")
    best_cov = max(cases, key=lambda c: float(c["max_omega"]))
    best_fit = min(cases, key=lambda c: float(c["rmse_to_secant"]))
    lines.append(f"- Best coverage: `{best_cov['label']}` reached `omega = {best_cov['max_omega']:.6f}`.\n")
    lines.append(f"- Best secant similarity: `{best_fit['label']}` with `RMSE = {best_fit['rmse_to_secant']:.4f}`.\n")
    lines.append("- The stored conservative `roll10` run with stall shrink `x0.8` still covers the most omega among the conservative/capped variants shown here, and it does so with far fewer linear iterations.\n")
    lines.append("- The conservative rerun with stall shrink `x0.9` is worse than the stored `x0.8` variant on coverage and linear work, but it is much closer to secant because it freezes earlier.\n")
    lines.append("- Adding the floor to the `x0.9` conservative run does improve coverage versus plain `x0.9`, but it still does not reach the stored `x0.8` conservative branch and still ends in a repair loop.\n")
    lines.append("- The previous cap2.0 absolute-gate run is the closest of the two cap2.0 runs to the secant curve, but it became too conservative once the restarted `d_omega` collapsed.\n")
    lines.append("- The new rerun prevents that collapse by flooring `d_omega` at the initial step size. That buys a small coverage gain, but it also keeps forcing floor-sized restarts in the hard tail, so the linear work goes up without reaching the horizon.\n\n")
    lines.append("## Linear Solver Diagnosis\n\n")
    lines.append("- The new rerun includes solver convergence logging.\n")
    lines.append(f"- Logged linear nonconvergence events: `{cases[-1]['linear_not_converged_count']}`.\n")
    lines.append(f"- Logged max-iteration hits: `{cases[-1]['linear_hit_max_count']}`.\n")
    lines.append("- So this rerun still does not point to a hidden Krylov failure. The remaining breakdown is controller-side: the restart/shrink logic is now safe, but the floor keeps the tail in a repeated repair-and-restart regime.\n\n")
    lines.append("## Plot Gallery\n\n")
    for key in [
        "lambda_vs_omega",
        "omega_vs_time",
        "lambda_vs_time",
        "lambda_deviation",
        "rel_after",
        "domega_used",
        "correction_norm",
    ]:
        lines.append(f"![{key}](plots/{plots[key]})\n\n")

    (OUT_DIR / "README.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
