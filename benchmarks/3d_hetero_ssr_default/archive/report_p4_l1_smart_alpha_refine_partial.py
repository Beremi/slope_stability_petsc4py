from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = ROOT / "artifacts/pmg_rank8_p2_levels_p4_omega7e6/p4_l1_smart_controller_v2_rank8_step100/data"
ALPHA_DIR = ROOT / "artifacts/p4_l1_smart_alpha_refine_full/rank8_step100/data"
OUT_DIR = ROOT / "artifacts/p4_l1_smart_alpha_refine_full/report"
PLOTS_DIR = OUT_DIR / "plots"


@dataclass
class RunData:
    label: str
    progress_lines: list[dict[str, Any]]
    init_complete: dict[str, Any]
    progress_latest: dict[str, Any] | None
    run_info: dict[str, Any] | None
    step_accepted: dict[int, dict[str, Any]]
    attempt_complete: dict[int, dict[str, Any]]
    newton_iterations: dict[int, list[dict[str, Any]]]
    secant_alpha: dict[int, float]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_run(data_dir: Path, label: str) -> RunData:
    progress_path = data_dir / "progress.jsonl"
    lines = [json.loads(line) for line in progress_path.read_text().splitlines()]
    init_complete = next(line for line in lines if line.get("event") == "init_complete")
    progress_latest = _load_json(data_dir / "progress_latest.json")
    run_info = _load_json(data_dir / "run_info.json")

    step_accepted = {
        line["accepted_step"]: line for line in lines if line.get("event") == "step_accepted"
    }
    attempt_complete = {
        line["target_step"]: line for line in lines if line.get("event") == "attempt_complete"
    }
    newton_iterations: dict[int, list[dict[str, Any]]] = {}
    for line in lines:
        if line.get("event") == "newton_iteration":
            newton_iterations.setdefault(line["target_step"], []).append(line)

    omega_hist = {1: init_complete["omega_hist"][0], 2: init_complete["omega_hist"][1]}
    for step, row in step_accepted.items():
        omega_hist[step] = row["omega_value"]
    secant_alpha = {
        step: (omega_hist[step] - omega_hist[step - 1]) / (omega_hist[step - 1] - omega_hist[step - 2])
        for step in range(3, max(omega_hist) + 1)
    }

    return RunData(
        label=label,
        progress_lines=lines,
        init_complete=init_complete,
        progress_latest=progress_latest,
        run_info=run_info,
        step_accepted=step_accepted,
        attempt_complete=attempt_complete,
        newton_iterations=newton_iterations,
        secant_alpha=secant_alpha,
    )


def cumulative_step_sums(run: RunData, max_step: int) -> dict[str, float]:
    rows = [row for step, row in run.step_accepted.items() if step <= max_step]
    return {
        "newton": sum(row["step_newton_iterations"] for row in rows),
        "linear": sum(row["step_linear_iterations"] for row in rows),
        "solve": sum(row["step_linear_solve_time"] for row in rows),
        "pc": sum(row["step_linear_preconditioner_time"] for row in rows),
        "orth": sum(row["step_linear_orthogonalization_time"] for row in rows),
        "wall": run.step_accepted[max_step]["total_wall_time"],
    }


def interp_baseline_time_for_omega(run: RunData, omega_target: float) -> float:
    rows = [run.step_accepted[step] for step in sorted(run.step_accepted)]
    for prev, cur in zip(rows, rows[1:]):
        if prev["omega_value"] <= omega_target <= cur["omega_value"]:
            omega_span = cur["omega_value"] - prev["omega_value"]
            frac = 0.0 if omega_span == 0.0 else (omega_target - prev["omega_value"]) / omega_span
            return prev["total_wall_time"] + frac * (cur["total_wall_time"] - prev["total_wall_time"])
    return rows[-1]["total_wall_time"]


def _dt(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def partial_step18_summary(run: RunData) -> dict[str, Any]:
    latest = run.progress_latest
    assert latest is not None
    step = latest["target_step"]
    iters = run.newton_iterations.get(step, [])
    last_accepted_step = max(run.step_accepted)
    last_accepted = run.step_accepted[last_accepted_step]
    return {
        "target_step": step,
        "iterations": len(iters),
        "sum_iteration_wall": sum(item["iteration_wall_time"] for item in iters),
        "sum_linear_iterations": sum(item["linear_iterations"] for item in iters),
        "sum_linear_solve": sum(item["linear_solve_time"] for item in iters),
        "sum_linear_pc": sum(item["linear_preconditioner_time"] for item in iters),
        "sum_linear_orth": sum(item["linear_orthogonalization_time"] for item in iters),
        "last_rel_residual": latest["rel_residual"],
        "last_criterion": latest["criterion"],
        "last_line_search_alpha": latest["alpha"],
        "last_lambda_value": latest["lambda_value"],
        "wall_after_last_accept_by_timestamp": (_dt(latest["timestamp"]) - _dt(last_accepted["timestamp"])).total_seconds(),
    }


def build_step_rows(alpha_run: RunData, baseline_run: RunData, start_step: int, end_step: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in range(start_step, end_step + 1):
        alpha_step = alpha_run.step_accepted[step]
        base_step = baseline_run.step_accepted[step]
        alpha_newton = alpha_run.newton_iterations.get(step, [])
        base_newton = baseline_run.newton_iterations.get(step, [])
        alpha_alphas = [item["alpha"] for item in alpha_newton if item.get("alpha") is not None]
        base_alphas = [item["alpha"] for item in base_newton if item.get("alpha") is not None]
        rows.append(
            {
                "step": step,
                "omega_alpha": alpha_step["omega_value"],
                "omega_base": base_step["omega_value"],
                "lambda_alpha": alpha_step["lambda_value"],
                "lambda_base": base_step["lambda_value"],
                "wall_alpha": alpha_step["step_wall_time"],
                "wall_base": base_step["step_wall_time"],
                "newton_alpha": alpha_step["step_newton_iterations"],
                "newton_base": base_step["step_newton_iterations"],
                "linear_alpha": alpha_step["step_linear_iterations"],
                "linear_base": base_step["step_linear_iterations"],
                "secant_alpha_alpha": alpha_run.secant_alpha[step],
                "secant_alpha_base": baseline_run.secant_alpha[step],
                "newton_ls_first_alpha": alpha_alphas[0] if alpha_alphas else None,
                "newton_ls_min_alpha": min(alpha_alphas) if alpha_alphas else None,
                "baseline_ls_first_alpha": base_alphas[0] if base_alphas else None,
                "baseline_ls_min_alpha": min(base_alphas) if base_alphas else None,
            }
        )
    return rows


def make_plots(alpha_run: RunData, baseline_run: RunData, rows: list[dict[str, Any]]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    alpha_steps = sorted(alpha_run.step_accepted)
    base_steps = sorted(baseline_run.step_accepted)

    plt.figure(figsize=(8, 5))
    plt.plot(
        [baseline_run.step_accepted[s]["omega_value"] for s in base_steps],
        [baseline_run.step_accepted[s]["lambda_value"] for s in base_steps],
        marker="o",
        ms=3,
        label="Smart Secant Baseline",
    )
    plt.plot(
        [alpha_run.step_accepted[s]["omega_value"] for s in alpha_steps],
        [alpha_run.step_accepted[s]["lambda_value"] for s in alpha_steps],
        marker="o",
        ms=3,
        label="Secant -> Alpha Refine (Interrupted)",
    )
    plt.xlabel("Omega")
    plt.ylabel("Lambda")
    plt.title("P4(L1) Continuation Prefix")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "lambda_omega_prefix.png", dpi=180)
    plt.close()

    steps = [row["step"] for row in rows]
    plt.figure(figsize=(9, 5))
    plt.plot(steps, [row["wall_base"] for row in rows], marker="o", label="Baseline")
    plt.plot(steps, [row["wall_alpha"] for row in rows], marker="o", label="Alpha Refine")
    plt.xlabel("Accepted Step")
    plt.ylabel("Step Wall Time [s]")
    plt.title("Accepted-Step Wall Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "step_wall_time.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(steps, [row["newton_base"] for row in rows], marker="o", label="Baseline")
    axes[0].plot(steps, [row["newton_alpha"] for row in rows], marker="o", label="Alpha Refine")
    axes[0].set_ylabel("Newton Iterations")
    axes[0].legend()
    axes[0].set_title("Accepted-Step Newton / Linear Iterations")
    axes[1].plot(steps, [row["linear_base"] for row in rows], marker="o", label="Baseline")
    axes[1].plot(steps, [row["linear_alpha"] for row in rows], marker="o", label="Alpha Refine")
    axes[1].set_ylabel("Linear Iterations")
    axes[1].set_xlabel("Accepted Step")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "step_iterations.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(steps, [row["secant_alpha_base"] for row in rows], marker="o", label="Baseline secant alpha")
    axes[0].plot(steps, [row["secant_alpha_alpha"] for row in rows], marker="o", label="Alpha-run secant alpha")
    axes[0].set_ylabel("Secant Alpha")
    axes[0].legend()
    axes[0].set_title("Secant Suggestion and Newton Line-Search Damping")
    axes[1].plot(steps, [row["baseline_ls_min_alpha"] for row in rows], marker="o", label="Baseline min Newton alpha")
    axes[1].plot(steps, [row["newton_ls_min_alpha"] for row in rows], marker="o", label="Alpha-run min Newton alpha")
    axes[1].set_ylabel("Min Newton Alpha")
    axes[1].set_xlabel("Accepted Step")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "alpha_diagnostics.png", dpi=180)
    plt.close()


def fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def write_report(alpha_run: RunData, baseline_run: RunData) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    max_alpha_step = max(alpha_run.step_accepted)
    rows = build_step_rows(alpha_run, baseline_run, 8, max_alpha_step)
    make_plots(alpha_run, baseline_run, rows)

    alpha_step12 = cumulative_step_sums(alpha_run, 12)
    base_step12 = cumulative_step_sums(baseline_run, 12)
    alpha_step17 = cumulative_step_sums(alpha_run, 17)
    base_step17 = cumulative_step_sums(baseline_run, 17)
    same_omega_baseline_time = interp_baseline_time_for_omega(
        baseline_run, alpha_run.step_accepted[17]["omega_value"]
    )
    partial18 = partial_step18_summary(alpha_run)

    summary = {
        "baseline_full_runtime_seconds": baseline_run.run_info["run_info"]["runtime_seconds"] if baseline_run.run_info else None,
        "baseline_full_step_count": baseline_run.run_info["run_info"]["step_count"] if baseline_run.run_info else None,
        "alpha_last_accepted_step": 17,
        "alpha_last_accepted_omega": alpha_run.step_accepted[17]["omega_value"],
        "alpha_last_accepted_lambda": alpha_run.step_accepted[17]["lambda_value"],
        "alpha_prefix_wall_step12": alpha_step12["wall"],
        "baseline_prefix_wall_step12": base_step12["wall"],
        "alpha_prefix_wall_step17": alpha_step17["wall"],
        "baseline_prefix_wall_step17": base_step17["wall"],
        "baseline_interp_wall_same_omega_as_alpha_step17": same_omega_baseline_time,
        "partial_step18": partial18,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    early_rows = [row for row in rows if 8 <= row["step"] <= 12]
    late_rows = [row for row in rows if 13 <= row["step"] <= 17]

    def table(step_rows: list[dict[str, Any]]) -> str:
        header = (
            "| Step | Omega A | Omega B | Wall A [s] | Wall B [s] | dWall [s] | "
            "Newt A | Newt B | Lin A | Lin B | Secant α A | Secant α B | "
            "Min Newton α A | Min Newton α B |\n"
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        lines = [header]
        for row in step_rows:
            lines.append(
                "| {step} | {omega_alpha:.0f} | {omega_base:.0f} | {wall_alpha:.3f} | {wall_base:.3f} | {dw:.3f} | "
                "{newton_alpha} | {newton_base} | {linear_alpha} | {linear_base} | {sa_a:.3f} | {sa_b:.3f} | "
                "{min_a:.5f} | {min_b:.5f} |".format(
                    step=row["step"],
                    omega_alpha=row["omega_alpha"],
                    omega_base=row["omega_base"],
                    wall_alpha=row["wall_alpha"],
                    wall_base=row["wall_base"],
                    dw=row["wall_alpha"] - row["wall_base"],
                    newton_alpha=row["newton_alpha"],
                    newton_base=row["newton_base"],
                    linear_alpha=row["linear_alpha"],
                    linear_base=row["linear_base"],
                    sa_a=row["secant_alpha_alpha"],
                    sa_b=row["secant_alpha_base"],
                    min_a=row["newton_ls_min_alpha"],
                    min_b=row["baseline_ls_min_alpha"],
                )
            )
        return "\n".join(lines)

    report = f"""# P4(L1) Smart-`d_omega` Secant vs Alpha-Refine Partial Comparison

## Scope

This report compares:

- completed smart-secant baseline: [run_info.json](../../pmg_rank8_p2_levels_p4_omega7e6/p4_l1_smart_controller_v2_rank8_step100/data/run_info.json)
- interrupted `secant -> secant_energy_alpha` run: [progress.jsonl](../rank8_step100/data/progress.jsonl)

The interrupted run used:

- plain secant through accepted step `7`
- `secant_energy_alpha` from target step `8` onward
- the same smart `d_omega` controller and rank-8 `P4(L1)` PMG-shell setup as the baseline

The alpha-refine run was manually stopped during target step `18`, so the fair direct comparison is the matched accepted prefix through step `17`.

## Headline

- Through accepted step `12`, alpha-refine was ahead:
  - wall time: `{alpha_step12["wall"]:.3f} s` vs `{base_step12["wall"]:.3f} s`
  - continuation Newton: `{alpha_step12["newton"]}` vs `{base_step12["newton"]}`
  - continuation linear: `{alpha_step12["linear"]}` vs `{base_step12["linear"]}`
- By accepted step `17`, alpha-refine was clearly behind:
  - wall time: `{alpha_step17["wall"]:.3f} s` vs `{base_step17["wall"]:.3f} s`
  - continuation Newton: `{alpha_step17["newton"]}` vs `{base_step17["newton"]}`
  - continuation linear: `{alpha_step17["linear"]}` vs `{base_step17["linear"]}`
- At the same reached omega (`{alpha_run.step_accepted[17]["omega_value"]:.0f}`), the baseline is still faster by interpolation:
  - alpha-refine: `{alpha_step17["wall"]:.3f} s`
  - baseline interpolated to same omega: `{same_omega_baseline_time:.3f} s`

The earlier short-step win did not survive the late plastic tail. Once the run entered steps `13-17`, the adaptive controller shrank `d_omega`, the alpha-refine path took more expensive accepted steps, and the cumulative gap reversed.

## Full / Partial Status

| Run | Status | Accepted Steps | Final Omega | Final Lambda | Wall Time [s] |
| --- | --- | ---: | ---: | ---: | ---: |
| Smart Secant Baseline | finished at `omega_max_stop` | {baseline_run.run_info["run_info"]["step_count"]} | {baseline_run.step_accepted[max(baseline_run.step_accepted)]["omega_value"]:.0f} | {baseline_run.step_accepted[max(baseline_run.step_accepted)]["lambda_value"]:.9f} | {baseline_run.run_info["run_info"]["runtime_seconds"]:.3f} |
| Secant -> Alpha Refine | interrupted during target step `18` | 17 accepted continuation steps after init | {alpha_run.step_accepted[17]["omega_value"]:.0f} | {alpha_run.step_accepted[17]["lambda_value"]:.9f} | {alpha_step17["wall"]:.3f} to last accepted step |

## Prefix Summary

| Prefix | Metric | Alpha Refine | Baseline |
| --- | --- | ---: | ---: |
| Through step 12 | Total wall [s] | {alpha_step12["wall"]:.3f} | {base_step12["wall"]:.3f} |
| Through step 12 | Continuation Newton | {alpha_step12["newton"]} | {base_step12["newton"]} |
| Through step 12 | Continuation Linear | {alpha_step12["linear"]} | {base_step12["linear"]} |
| Through step 12 | Linear solve [s] | {alpha_step12["solve"]:.3f} | {base_step12["solve"]:.3f} |
| Through step 12 | PC apply [s] | {alpha_step12["pc"]:.3f} | {base_step12["pc"]:.3f} |
| Through step 12 | Orthogonalization [s] | {alpha_step12["orth"]:.3f} | {base_step12["orth"]:.3f} |
| Through step 17 | Total wall [s] | {alpha_step17["wall"]:.3f} | {base_step17["wall"]:.3f} |
| Through step 17 | Continuation Newton | {alpha_step17["newton"]} | {base_step17["newton"]} |
| Through step 17 | Continuation Linear | {alpha_step17["linear"]} | {base_step17["linear"]} |
| Through step 17 | Linear solve [s] | {alpha_step17["solve"]:.3f} | {base_step17["solve"]:.3f} |
| Through step 17 | PC apply [s] | {alpha_step17["pc"]:.3f} | {base_step17["pc"]:.3f} |
| Through step 17 | Orthogonalization [s] | {alpha_step17["orth"]:.3f} | {base_step17["orth"]:.3f} |

## Interrupted Step 18 Snapshot

The alpha-refine run was stopped during target step `18`. At the stop point it had already spent:

- Newton iterations completed in step `18`: `{partial18["iterations"]}`
- summed iteration wall time: `{partial18["sum_iteration_wall"]:.3f} s`
- summed linear iterations: `{partial18["sum_linear_iterations"]}`
- summed linear solve / PC / orth times: `{partial18["sum_linear_solve"]:.3f} / {partial18["sum_linear_pc"]:.3f} / {partial18["sum_linear_orth"]:.3f} s`
- latest nonlinear criterion / relative residual: `{partial18["last_criterion"]:.3f}` / `{partial18["last_rel_residual"]:.3e}`
- latest Newton line-search alpha: `{partial18["last_line_search_alpha"]}`

For context, the completed baseline step `18` cost `170.722 s`, with `11` Newton iterations and `199` linear iterations. So by the time this run was killed, step `18` was already on track to be at least comparable to the baseline late-tail cost, not better.

## Accepted-Step Comparison: Early Window

{table(early_rows)}

## Accepted-Step Comparison: Late Window

{table(late_rows)}

## What We Can and Cannot Recover

- The interrupted artifact **does** preserve accepted-step wall time, Newton iterations, linear iterations, `lambda`, `omega`, and per-Newton line-search alpha.
- The interrupted artifact **does not** preserve the searched predictor alpha from `secant_energy_alpha`; only the standard secant suggestion and the actual Newton damping alphas are recoverable from `progress.jsonl`.
- Because the baseline full run predates the saved-step predictor diagnostics, there is no apples-to-apples `u_init -> u_newton` mismatch integral for the baseline in this report.

## Interpretation

- The local alpha-refine idea still looks useful in the transition zone around steps `11-12`.
- It did not stay useful once the run entered the hard late plastic-flow regime.
- The strongest symptom is not outright Newton failure; it is that accepted steps `13-17` became much more expensive than on the baseline.
- The minimum Newton line-search alpha also became very small on the alpha-refine run in the late window:
  - step `13`: `0.03125`
  - step `14`: `0.0625`
  - step `15`: `0.03125`
  - step `16`: `0.00390625`
  - step `17`: `0.03125`

That pattern is consistent with the alpha-refined predictor steering the solver into a less favorable basin in the late tail, even though it helped on one shorter prefix test.

## Plots

![Lambda Omega Prefix](plots/lambda_omega_prefix.png)

![Step Wall Time](plots/step_wall_time.png)

![Step Iterations](plots/step_iterations.png)

![Alpha Diagnostics](plots/alpha_diagnostics.png)
"""

    (OUT_DIR / "README.md").write_text(report)


def main() -> None:
    alpha_run = load_run(ALPHA_DIR, "alpha_refine")
    baseline_run = load_run(BASELINE_DIR, "baseline")
    write_report(alpha_run, baseline_run)


if __name__ == "__main__":
    main()
