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
OUT_DIR = ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk/absgate_vs_secant_report"
PLOTS_DIR = OUT_DIR / "plots"
SECANT_PATH = ROOT / "artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_rank8_step12/data/progress.jsonl"

CASES = [
    {
        "key": "full_rel_fixedomega_currentstate_omega7e6_corrected_absgate",
        "label": "No Cap, Absolute Gate",
        "path": ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk/full_rel_fixedomega_currentstate_omega7e6_corrected_absgate/data/progress.jsonl",
        "color": "#d62728",
    },
    {
        "key": "full_rel_fixedomega_currentstate_omega7e6_growth1p25_absgate",
        "label": "Growth x1.25, Absolute Gate",
        "path": ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk/full_rel_fixedomega_currentstate_omega7e6_growth1p25_absgate/data/progress.jsonl",
        "color": "#9467bd",
    },
    {
        "key": "full_rel_fixedomega_currentstate_omega7e6_capunit_absgate",
        "label": "Unit Cap, Absolute Gate",
        "path": ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk/full_rel_fixedomega_currentstate_omega7e6_capunit_absgate/data/progress.jsonl",
        "color": "#2ca02c",
    },
    {
        "key": "full_rel_fixedomega_currentstate_omega7e6_cap1p0_absgate",
        "label": "Initial-Segment Cap, Absolute Gate",
        "path": ROOT / "artifacts/p4_l1_fixed_domega_micro_newton_walk/full_rel_fixedomega_currentstate_omega7e6_cap1p0_absgate/data/progress.jsonl",
        "color": "#1f77b4",
    },
]


def _ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_secant() -> dict[str, object]:
    recs = [json.loads(line) for line in SECANT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    init = next(r for r in recs if r.get("event") == "init_complete")
    steps = [r for r in recs if r.get("event") == "step_accepted"]
    omega = [float(init["omega_hist"][0]), float(init["omega_hist"][1])]
    lam = [float(init["lambda_hist"][0]), float(init["lambda_hist"][1])]
    for step in steps:
        omega.append(float(step["omega_value"]))
        lam.append(float(step["lambda_value"]))
    uniq_o: list[float] = []
    uniq_l: list[float] = []
    for o, l in zip(omega, lam):
        if not uniq_o or o > uniq_o[-1] + 1.0e-9:
            uniq_o.append(o)
            uniq_l.append(l)
    return {"omega": np.asarray(uniq_o, dtype=np.float64), "lambda": np.asarray(uniq_l, dtype=np.float64)}


def _load_case(spec: dict[str, object], secant: dict[str, object]) -> dict[str, object]:
    path = Path(spec["path"])
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    init = next(r for r in recs if r.get("event") == "init_complete")
    steps = [r for r in recs if r.get("event") == "micro_step_complete"]
    x = np.asarray([float(r.get("committed_omega_after", r.get("omega_after", math.nan))) for r in steps], dtype=np.float64)
    y = np.asarray([float(r.get("committed_lambda_after", r.get("lambda_after", math.nan))) for r in steps], dtype=np.float64)
    rel = np.asarray([float(r.get("rel_residual_after", math.nan)) for r in steps], dtype=np.float64)
    domega = np.asarray([float(r.get("domega_used", 0.0)) for r in steps], dtype=np.float64)
    wall = np.asarray([float(r.get("total_wall_time", math.nan)) for r in steps], dtype=np.float64)
    linear = np.asarray([float(r.get("linear_iterations", 0.0)) for r in steps], dtype=np.float64)
    sec_o = np.asarray(secant["omega"], dtype=np.float64)
    sec_l = np.asarray(secant["lambda"], dtype=np.float64)
    sec_interp = np.interp(x, sec_o, sec_l)
    dev = y - sec_interp
    return {
        "key": str(spec["key"]),
        "label": str(spec["label"]),
        "color": str(spec["color"]),
        "path": str(path),
        "init": init,
        "steps": steps,
        "micro_idx": np.arange(1, len(steps) + 1, dtype=np.float64),
        "omega": x,
        "lambda": y,
        "secant_lambda": sec_interp,
        "lambda_dev": dev,
        "rel_after": rel,
        "domega_used": domega,
        "total_wall": wall,
        "linear_iterations": linear,
        "max_omega": float(np.nanmax(x)),
        "max_lambda": float(np.nanmax(y)),
        "final_lambda": float(y[-1]),
        "final_rel": float(rel[-1]),
        "linear_total": int(np.nansum(linear)),
        "zero_domega_steps": int(np.sum(np.abs(domega) <= 1.0e-12)),
        "rmse": float(np.sqrt(np.mean(dev**2))),
        "mae": float(np.mean(np.abs(dev))),
        "first_zero_step": int(np.where(np.abs(domega) <= 1.0e-12)[0][0] + 1) if np.any(np.abs(domega) <= 1.0e-12) else None,
        "not_converged_count": int(sum(bool(r.get("linear_any_not_converged", False)) for r in steps)),
    }


def _save(fig: plt.Figure, filename: str) -> str:
    out = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return filename


def _plot_trajectories(secant: dict[str, object], cases: list[dict[str, object]]) -> str:
    fig = plt.figure(figsize=(9.0, 6.0), dpi=180)
    plt.plot(np.asarray(secant["omega"]) / 1.0e6, np.asarray(secant["lambda"]), color="black", linewidth=2.0, label="True Secant Reference")
    for case in cases:
        plt.plot(np.asarray(case["omega"]) / 1.0e6, np.asarray(case["lambda"]), marker="o", markersize=3.0, linewidth=1.5, color=str(case["color"]), label=str(case["label"]))
    plt.xlabel(r"$\omega$ [$10^6$]")
    plt.ylabel(r"$\lambda$")
    plt.title(r"Absolute-Gate Reruns vs True Secant Branch")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    return _save(fig, "lambda_vs_omega.png")


def _plot_deviation(cases: list[dict[str, object]]) -> str:
    fig = plt.figure(figsize=(9.0, 6.0), dpi=180)
    for case in cases:
        plt.plot(np.asarray(case["micro_idx"]), np.asarray(case["lambda_dev"]), marker="o", markersize=3.0, linewidth=1.5, color=str(case["color"]), label=str(case["label"]))
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("Micro-Step Index")
    plt.ylabel(r"$\lambda - \lambda_{\mathrm{secant}}(\omega)$")
    plt.title("Deviation from Secant at the Same Omega")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    return _save(fig, "lambda_deviation_vs_step.png")


def _plot_residual(cases: list[dict[str, object]]) -> str:
    fig = plt.figure(figsize=(9.0, 6.0), dpi=180)
    for case in cases:
        plt.plot(np.asarray(case["micro_idx"]), np.asarray(case["rel_after"]), marker="o", markersize=3.0, linewidth=1.5, color=str(case["color"]), label=str(case["label"]))
    plt.axhline(1.0e-1, color="0.25", linestyle=":", linewidth=1.0, alpha=0.8, label="1e-1")
    plt.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0, alpha=0.8, label="1")
    plt.yscale("log")
    plt.xlabel("Micro-Step Index")
    plt.ylabel("Relative Residual After")
    plt.title("Residual Behavior Under the Absolute Gate")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    return _save(fig, "rel_after_vs_step.png")


def _plot_domega(cases: list[dict[str, object]]) -> str:
    fig = plt.figure(figsize=(9.0, 6.0), dpi=180)
    for case in cases:
        plt.plot(np.asarray(case["micro_idx"]), np.asarray(case["domega_used"]), marker="o", markersize=3.0, linewidth=1.5, color=str(case["color"]), label=str(case["label"]))
    plt.xlabel("Micro-Step Index")
    plt.ylabel(r"$d\omega$ used")
    plt.title("Used Omega Increments")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    return _save(fig, "domega_used_vs_step.png")


def _plot_wall(secant: dict[str, object], cases: list[dict[str, object]]) -> str:
    fig = plt.figure(figsize=(9.0, 6.0), dpi=180)
    for case in cases:
        plt.plot(np.asarray(case["omega"]) / 1.0e6, np.asarray(case["total_wall"]), marker="o", markersize=3.0, linewidth=1.5, color=str(case["color"]), label=str(case["label"]))
    plt.xlabel(r"Committed $\omega$ [$10^6$]")
    plt.ylabel("Cumulative Wall Time [s]")
    plt.title("Cumulative Wall Along the Absolute-Gate Reruns")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    return _save(fig, "cumulative_wall_vs_omega.png")


def main() -> None:
    _ensure_dirs()
    secant = _load_secant()
    cases = [_load_case(spec, secant) for spec in CASES]
    plots = {
        "traj": _plot_trajectories(secant, cases),
        "dev": _plot_deviation(cases),
        "res": _plot_residual(cases),
        "domega": _plot_domega(cases),
        "wall": _plot_wall(secant, cases),
    }
    summary = {
        "secant_reference": {"path": str(SECANT_PATH), "final_omega": float(np.max(secant["omega"])), "final_lambda": float(np.asarray(secant["lambda"])[-1])},
        "cases": [
            {
                "key": case["key"],
                "label": case["label"],
                "path": case["path"],
                "max_omega": case["max_omega"],
                "max_lambda": case["max_lambda"],
                "final_lambda": case["final_lambda"],
                "final_rel": case["final_rel"],
                "rmse": case["rmse"],
                "mae": case["mae"],
                "linear_total": case["linear_total"],
                "zero_domega_steps": case["zero_domega_steps"],
                "first_zero_step": case["first_zero_step"],
                "not_converged_count": case["not_converged_count"],
            }
            for case in cases
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Absolute-Gate Reruns vs True Secant\n")
    lines.append("This report compares only the new absolute-gate reruns against the original secant continuation branch.\n")
    lines.append("## Summary\n")
    lines.append("| Run | Max omega | Max lambda | Final lambda | RMSE to secant | Zero-domega steps | First zero-domega | Linear nonconverged |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for case in cases:
        lines.append(
            "| "
            + f"{case['label']} | {case['max_omega'] / 1.0e6:.6f}e6 | {case['max_lambda']:.6f} | {case['final_lambda']:.6f} | "
            + f"{case['rmse']:.4f} | {case['zero_domega_steps']} | {case['first_zero_step']} | {case['not_converged_count']} |\n"
        )
    lines.append("\n")
    lines.append("## Interpretation\n")
    lines.append("- `No Cap, Absolute Gate` is the worst behaviorally: it freezes at step 7 and then the restarted step blows the residual up catastrophically.\n")
    lines.append("- `Growth x1.25, Absolute Gate` is the best of these reruns overall. It reaches about `6.438e6` while staying close to secant (`RMSE 0.0105`) and without any logged linear nonconvergence.\n")
    lines.append("- `Unit Cap, Absolute Gate` reaches slightly farther, about `6.447e6`, but with larger secant deviation and repeated repair cycles.\n")
    lines.append("- `Initial-Segment Cap, Absolute Gate` is the most faithful of the capped reruns by RMSE, but it stalls earliest among the stable variants, around `6.422e6`.\n")
    lines.append("- None of these absolute-gate reruns beat the stored relative-gate current-state runs on omega coverage.\n")
    lines.append("\n")
    lines.append("## Plots\n")
    lines.append(f"![trajectories](plots/{plots['traj']})\n")
    lines.append(f"![deviation](plots/{plots['dev']})\n")
    lines.append(f"![residual](plots/{plots['res']})\n")
    lines.append(f"![domega](plots/{plots['domega']})\n")
    lines.append(f"![wall](plots/{plots['wall']})\n")

    (OUT_DIR / "README.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
