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
OUT_DIR = MICRO_ROOT / "stored_vs_secant_report"
PLOTS_DIR = OUT_DIR / "plots"

SELECTED = [
    {
        "key": "full_rel_fixedomega_currentstate_omega7e6_corrected",
        "label": "Current-State Fixed-Omega, Corrected, Farthest Coverage",
        "color": "#1f77b4",
    },
    {
        "key": "full_rel_fixedomega_currentstate_omega7e6",
        "label": "Current-State Fixed-Omega, Earlier Variant",
        "color": "#17becf",
    },
    {
        "key": "full_rel_fixedomega_currentstate_omega7e6_growth1p25",
        "label": "Current-State Fixed-Omega, Growth x1.25",
        "color": "#9467bd",
    },
    {
        "key": "rank8_micro100_adaptive_cap652_negdl_rrb_roll10",
        "label": "Older Roll10 Short-Range Variant",
        "color": "#2ca02c",
    },
    {
        "key": "full_omega7e6_adaptive_negdl_rrb_roll10",
        "label": "Older Roll10 Overshoot Failure",
        "color": "#ff7f0e",
    },
    {
        "key": "full_rel_fixedomega_currentstate_omega7e6_cap1p0",
        "label": "Initial-Segment Cap Failure",
        "color": "#d62728",
    },
]


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_secant_curve() -> dict[str, object]:
    recs = [json.loads(line) for line in SECANT_PROGRESS.read_text(encoding="utf-8").splitlines() if line.strip()]
    init = next(r for r in recs if r.get("event") == "init_complete")
    accepted = [r for r in recs if r.get("event") == "step_accepted"]
    omega = [float(init["omega_hist"][0]), float(init["omega_hist"][1])]
    lam = [float(init["lambda_hist"][0]), float(init["lambda_hist"][1])]
    for step in accepted:
        omega.append(float(step["omega_value"]))
        lam.append(float(step["lambda_value"]))
    uniq_o: list[float] = []
    uniq_l: list[float] = []
    for o, l in zip(omega, lam):
        if not uniq_o or o > uniq_o[-1] + 1.0e-9:
            uniq_o.append(o)
            uniq_l.append(l)
    return {
        "omega": np.asarray(uniq_o, dtype=np.float64),
        "lambda": np.asarray(uniq_l, dtype=np.float64),
    }


def _settings_summary(key: str, init: dict[str, object]) -> str:
    parts: list[str] = []
    if "fixedomega_currentstate" in key:
        parts.append("fixed-omega=current-state")
    if "negdl_rrb" in key:
        parts.append("negdl+rollback")
    if str(init.get("mode", "")):
        parts.append(str(init["mode"]))
    roll = init.get("rolling_basis_size")
    comp = init.get("basis_compression")
    if roll is not None:
        comp_tag = "" if not comp or str(comp) == "none" else str(comp)
        parts.append(f"roll{int(roll)}{comp_tag}")
    if bool(init.get("disable_deflation_basis", False)):
        parts.append("no-defl")
    if bool(init.get("stall_recovery", False)):
        parts.append("stall-recovery")
    gate = init.get("correction_gate_mode")
    if gate:
        thr = init.get("correction_rel_threshold") if str(gate) == "relative" else init.get("correction_abs_threshold")
        if thr is not None:
            parts.append(f"{gate}-gate={thr}")
    alpha_thr = init.get("omega_freeze_alpha_threshold")
    if alpha_thr is not None:
        parts.append(f"alpha-freeze<{alpha_thr}")
    growth = init.get("omega_growth_factor")
    if growth is not None and not math.isclose(float(growth), 1.5):
        parts.append(f"growthx{float(growth):.2f}")
    step_cap = init.get("step_length_cap_mode")
    if step_cap and str(step_cap) != "none":
        parts.append(f"{step_cap}:{float(init.get('step_length_cap_factor', 1.0)):.3g}")
    return ", ".join(parts)


def _load_micro_run(path: Path, secant: dict[str, object]) -> dict[str, object]:
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        init = next(r for r in recs if r.get("event") == "init_complete")
    except StopIteration as exc:
        raise ValueError(f"missing init_complete in {path}") from exc
    steps = [r for r in recs if r.get("event") == "micro_step_complete"]
    final = next((r for r in reversed(recs) if r.get("event") == "run_complete"), None)
    if final is None:
        last = steps[-1]
        final = {
            "event": "run_complete_partial",
            "stop_reason": "partial_log_no_run_complete",
            "final_omega": float(last.get("committed_omega_after", last.get("omega_after", math.nan))),
            "final_lambda": float(last.get("committed_lambda_after", last.get("lambda_after", math.nan))),
            "runtime_seconds": float(last.get("total_wall_time", math.nan)),
            "micro_steps_completed": int(len(steps)),
        }
    x = np.asarray([float(s.get("committed_omega_after", s.get("omega_after", math.nan))) for s in steps], dtype=np.float64)
    y = np.asarray([float(s.get("committed_lambda_after", s.get("lambda_after", math.nan))) for s in steps], dtype=np.float64)
    rel = np.asarray([float(s.get("rel_residual_after", math.nan)) for s in steps], dtype=np.float64)
    domega = np.asarray([float(s.get("domega_used", 0.0)) for s in steps], dtype=np.float64)
    total_wall = np.asarray([float(s.get("total_wall_time", math.nan)) for s in steps], dtype=np.float64)
    linear = np.asarray([float(s.get("linear_iterations", 0.0)) for s in steps], dtype=np.float64)
    sec_o = np.asarray(secant["omega"], dtype=np.float64)
    sec_l = np.asarray(secant["lambda"], dtype=np.float64)
    sec_ref = np.interp(x, sec_o, sec_l)
    dev = y - sec_ref
    coverage = float((np.nanmax(x) - sec_o[1]) / max(sec_o[-1] - sec_o[1], 1.0e-12))
    zero_mask = np.abs(domega) <= 1.0e-12
    first_zero = int(np.where(zero_mask)[0][0] + 1) if np.any(zero_mask) else None
    i_max = int(np.nanargmax(y))
    return {
        "key": path.parent.parent.name,
        "path": str(path),
        "init": init,
        "steps": steps,
        "final": final,
        "micro_idx": np.arange(1, len(steps) + 1, dtype=np.float64),
        "omega": x,
        "lambda": y,
        "secant_lambda_at_omega": sec_ref,
        "lambda_dev": dev,
        "rel_after": rel,
        "domega_used": domega,
        "total_wall": total_wall,
        "linear_iterations": linear,
        "coverage": coverage,
        "max_omega": float(np.nanmax(x)),
        "max_lambda": float(np.nanmax(y)),
        "linear_total": int(np.nansum(linear)),
        "zero_domega_steps": int(np.sum(zero_mask)),
        "rmse_all": float(np.sqrt(np.mean(dev**2))),
        "mae_all": float(np.mean(np.abs(dev))),
        "maxerr_all": float(np.max(np.abs(dev))),
        "first_zero_step": first_zero,
        "max_lambda_step": int(i_max + 1),
        "max_lambda_dev": float(dev[i_max]),
        "max_lambda_rel_after": float(rel[i_max]),
        "settings_summary": _settings_summary(path.parent.parent.name, init),
        "linear_not_converged_count": int(
            sum(bool(s.get("linear_any_not_converged", False)) for s in steps if "linear_any_not_converged" in s)
        ),
    }


def _save_plot(fig: plt.Figure, filename: str) -> str:
    path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return filename


def _plot_selected_trajectories(secant: dict[str, object], cases: list[dict[str, object]]) -> str:
    fig = plt.figure(figsize=(9.0, 6.0), dpi=180)
    plt.plot(
        np.asarray(secant["omega"], dtype=np.float64) / 1.0e6,
        np.asarray(secant["lambda"], dtype=np.float64),
        color="black",
        linewidth=2.0,
        label="Original Secant Reference",
    )
    for case in cases:
        plt.plot(
            np.asarray(case["omega"], dtype=np.float64) / 1.0e6,
            np.asarray(case["lambda"], dtype=np.float64),
            marker="o",
            markersize=3.0,
            linewidth=1.4,
            color=str(case["color"]),
            label=str(case["label"]),
        )
    plt.xlabel(r"$\omega$ [$10^6$]")
    plt.ylabel(r"$\lambda$")
    plt.title(r"Stored Micro-Walk Variants Against the Original Secant Branch")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    return _save_plot(fig, "selected_lambda_vs_omega.png")


def _plot_selected_deviation(cases: list[dict[str, object]]) -> str:
    fig = plt.figure(figsize=(9.0, 6.0), dpi=180)
    for case in cases:
        plt.plot(
            np.asarray(case["micro_idx"], dtype=np.float64),
            np.asarray(case["lambda_dev"], dtype=np.float64),
            marker="o",
            markersize=3.0,
            linewidth=1.4,
            color=str(case["color"]),
            label=str(case["label"]),
        )
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("Micro-Step Index")
    plt.ylabel(r"$\lambda - \lambda_{\mathrm{secant}}(\omega)$")
    plt.title("Deviation from the Secant Reference at the Same Omega")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    return _save_plot(fig, "selected_lambda_deviation_vs_step.png")


def _plot_selected_residual(cases: list[dict[str, object]]) -> str:
    fig = plt.figure(figsize=(9.0, 6.0), dpi=180)
    for case in cases:
        plt.plot(
            np.asarray(case["micro_idx"], dtype=np.float64),
            np.asarray(case["rel_after"], dtype=np.float64),
            marker="o",
            markersize=3.0,
            linewidth=1.4,
            color=str(case["color"]),
            label=str(case["label"]),
        )
    plt.axhline(1.0e-1, color="0.25", linestyle=":", linewidth=1.0, alpha=0.8, label="1e-1")
    plt.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0, alpha=0.8, label="1")
    plt.yscale("log")
    plt.xlabel("Micro-Step Index")
    plt.ylabel("Relative Residual After")
    plt.title("Residual Growth and Tail Degradation")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    return _save_plot(fig, "selected_rel_after_vs_step.png")


def _plot_all_scatter(all_cases: list[dict[str, object]], selected_keys: set[str]) -> str:
    fig = plt.figure(figsize=(9.0, 6.0), dpi=180)
    for case in all_cases:
        is_selected = case["key"] in selected_keys
        plt.scatter(
            100.0 * float(case["coverage"]),
            float(case["rmse_all"]),
            s=60 if is_selected else 24,
            color=case.get("color", "#7f7f7f") if is_selected else "#bdbdbd",
            edgecolor="black" if is_selected else "none",
            alpha=0.9 if is_selected else 0.55,
        )
        if is_selected:
            plt.annotate(
                case["key"].replace("full_", "").replace("rank8_micro100_", ""),
                (100.0 * float(case["coverage"]), float(case["rmse_all"])),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
            )
    plt.xlabel("Coverage to 7e6 Horizon [%]")
    plt.ylabel("RMSE to Secant Branch over Stored Points")
    plt.title("All Stored Micro-Walk Runs: Coverage vs Secant Similarity")
    plt.grid(True, alpha=0.3)
    return _save_plot(fig, "all_runs_coverage_vs_rmse.png")


def _plot_all_curves(secant: dict[str, object], all_cases: list[dict[str, object]]) -> str:
    fig = plt.figure(figsize=(9.0, 6.0), dpi=180)
    plt.plot(
        np.asarray(secant["omega"], dtype=np.float64) / 1.0e6,
        np.asarray(secant["lambda"], dtype=np.float64),
        color="black",
        linewidth=2.0,
        label="Original Secant Reference",
    )
    for case in all_cases:
        plt.plot(
            np.asarray(case["omega"], dtype=np.float64) / 1.0e6,
            np.asarray(case["lambda"], dtype=np.float64),
            linewidth=0.9,
            alpha=0.35,
            color="#1f77b4" if float(case["coverage"]) >= 0.5 else "#7f7f7f",
        )
    plt.xlabel(r"$\omega$ [$10^6$]")
    plt.ylabel(r"$\lambda$")
    plt.title("All Stored Micro-Walk Trajectories vs Secant")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    return _save_plot(fig, "all_runs_lambda_vs_omega.png")


def _build_summary(secant: dict[str, object], all_cases: list[dict[str, object]], selected: list[dict[str, object]]) -> dict[str, object]:
    return {
        "secant_reference": {
            "path": str(SECANT_PROGRESS),
            "omega_max": float(np.max(np.asarray(secant["omega"], dtype=np.float64))),
            "lambda_final": float(np.asarray(secant["lambda"], dtype=np.float64)[-1]),
        },
        "selected_cases": [
            {
                "key": case["key"],
                "label": case["label"],
                "path": case["path"],
                "coverage": case["coverage"],
                "max_omega": case["max_omega"],
                "max_lambda": case["max_lambda"],
                "rmse_all": case["rmse_all"],
                "mae_all": case["mae_all"],
                "maxerr_all": case["maxerr_all"],
                "linear_total": case["linear_total"],
                "zero_domega_steps": case["zero_domega_steps"],
                "first_zero_step": case["first_zero_step"],
                "max_lambda_step": case["max_lambda_step"],
                "max_lambda_dev": case["max_lambda_dev"],
                "settings_summary": case["settings_summary"],
            }
            for case in selected
        ],
        "all_cases_ranked_by_coverage": [
            {
                "key": case["key"],
                "path": case["path"],
                "coverage": case["coverage"],
                "max_omega": case["max_omega"],
                "rmse_all": case["rmse_all"],
            }
            for case in sorted(all_cases, key=lambda c: (-float(c["max_omega"]), float(c["rmse_all"])))
        ],
    }


def main() -> None:
    _ensure_dirs()
    secant = _load_secant_curve()
    label_lookup = {spec["key"]: spec for spec in SELECTED}
    all_cases: list[dict[str, object]] = []
    for path in sorted(MICRO_ROOT.rglob("progress.jsonl")):
        try:
            case = _load_micro_run(path, secant)
        except ValueError:
            continue
        spec = label_lookup.get(case["key"])
        if spec is not None:
            case["label"] = spec["label"]
            case["color"] = spec["color"]
        else:
            case["label"] = case["key"]
            case["color"] = "#7f7f7f"
        all_cases.append(case)

    selected_keys = {spec["key"] for spec in SELECTED}
    selected_cases = [next(case for case in all_cases if case["key"] == spec["key"]) for spec in SELECTED]

    plots = {
        "selected_lambda_vs_omega": _plot_selected_trajectories(secant, selected_cases),
        "selected_lambda_deviation": _plot_selected_deviation(selected_cases),
        "selected_rel_after": _plot_selected_residual(selected_cases),
        "all_runs_scatter": _plot_all_scatter(all_cases, selected_keys),
        "all_runs_lambda_vs_omega": _plot_all_curves(secant, all_cases),
    }

    summary = _build_summary(secant, all_cases, selected_cases)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Stored Micro-Walk Runs vs Original Secant\n")
    lines.append("This report uses only stored artifacts. The original secant continuation to `omega=7e6` is the reference branch. Micro-walk variants are compared by both coverage and lambda-vs-omega similarity to that secant branch.\n")
    lines.append("## Reference\n")
    lines.append(f"- Original secant artifact: `artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_rank8_step12/data/progress.jsonl`\n")
    lines.append(f"- Secant final point: `omega = {float(np.max(secant['omega'])):.6f}`, `lambda = {float(np.asarray(secant['lambda'])[-1]):.6f}`\n")
    lines.append("\n")
    lines.append("## Selected Variants\n")
    lines.append("| Variant | Max omega | Coverage | RMSE to secant | Max lambda | First zero-domega step | Settings |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |\n")
    for case in selected_cases:
        lines.append(
            "| "
            + f"{case['label']} | {case['max_omega'] / 1.0e6:.6f}e6 | {100.0 * case['coverage']:.1f}% | "
            + f"{case['rmse_all']:.4f} | {case['max_lambda']:.6f} | {case['first_zero_step']} | {case['settings_summary']} |\n"
        )
    lines.append("\n")
    lines.append("## What Was Better In Older Runs\n")
    lines.append("- The best stored full-coverage micro-walk run is still `full_rel_fixedomega_currentstate_omega7e6_corrected`: it reaches `6.760e6`, the farthest of all stored micro-walk variants.\n")
    lines.append("- The earlier `full_rel_fixedomega_currentstate_omega7e6` run is slightly more faithful to secant at the same omega (`RMSE 0.0868` vs `0.1047`) but does not push past the `6.7145e6` barrier.\n")
    lines.append("- The older short-range `rank8_micro100_adaptive_cap652_negdl_rrb_roll10` run is the cleanest near `6.48e6`: it stays very close to secant (`RMSE 0.0212`) and never develops the large lambda overshoot seen in later capped runs.\n")
    lines.append("- The newer `growth x1.25` tweak reduces the lambda deviation substantially (`RMSE 0.0363`) compared with the farthest-coverage runs, but it pays for that by stalling much earlier at `6.5411e6`.\n")
    lines.append("\n")
    lines.append("## How The Failures Happened\n")
    for case in selected_cases:
        lines.append(
            "- "
            + f"{case['label']}: first zero-omega step at `{case['first_zero_step']}`, max lambda at step `{case['max_lambda_step']}` with deviation "
            + f"`{case['max_lambda_dev']:+.4f}` from secant and residual `{case['max_lambda_rel_after']:.3e}`."
            + "\n"
        )
    lines.append("- The two distinct failure patterns in stored runs are:\n")
    lines.append("  1. overshoot at fixed omega: lambda keeps moving away from secant after omega has already stopped, often followed by very large residuals.\n")
    lines.append("  2. repair stall: omega freezes, lambda relaxes back toward secant, and the run spends many steps with zero `domega` but does not recover outward progression.\n")
    lines.append("- The cap-at-initial-segment run is the clearest overshoot example: by the time omega has frozen at `6.4806e6`, lambda has drifted to `2.0644`, over `0.52` above secant at that same omega.\n")
    lines.append("- The corrected current-state run is the clearest repair-stall example: omega freezes near `6.7145e6`, lambda peaks about `0.172` above secant, then relaxes downward while the residual decreases, but outward progress never resumes.\n")
    lines.append("\n")
    lines.append("## Plots\n")
    lines.append(f"![selected trajectories](plots/{plots['selected_lambda_vs_omega']})\n")
    lines.append(f"![selected deviation](plots/{plots['selected_lambda_deviation']})\n")
    lines.append(f"![selected residual](plots/{plots['selected_rel_after']})\n")
    lines.append(f"![coverage vs rmse](plots/{plots['all_runs_scatter']})\n")
    lines.append(f"![all trajectories](plots/{plots['all_runs_lambda_vs_omega']})\n")

    (OUT_DIR / "README.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
