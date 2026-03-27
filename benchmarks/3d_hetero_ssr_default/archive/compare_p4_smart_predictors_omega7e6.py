from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "archive" else SCRIPT_DIR
ROOT = BENCHMARK_DIR.parents[1]

DEFAULT_BASELINE_DIR = ROOT / "artifacts" / "pmg_rank8_p2_levels_p4_omega7e6" / "p4_l1_smart_controller_v2_rank8_step100"
DEFAULT_CANDIDATE_DIR = ROOT / "artifacts" / "pmg_rank8_p2_levels_p4_omega7e6" / "p4_l1_smart_controller_two_step_rank8_step100"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "p4_l1_smart_predictor_compare_omega7e6"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _relpath(from_path: Path, to_path: Path) -> str:
    return os.path.relpath(to_path.resolve(), from_path.parent.resolve())


def _safe_array(payload, key: str, dtype=float) -> np.ndarray:
    if key not in payload:
        return np.asarray([], dtype=dtype)
    return np.asarray(payload[key], dtype=dtype)


def _load_case(case_dir: Path) -> dict[str, object]:
    data_dir = case_dir / "data"
    npz = np.load(data_dir / "petsc_run.npz", allow_pickle=False)
    run_info = json.loads((data_dir / "run_info.json").read_text(encoding="utf-8"))
    linear = run_info.get("timings", {}).get("linear", {})
    predictor = run_info.get("predictor_diagnostics", {})

    lambda_hist = np.asarray(npz["lambda_hist"], dtype=np.float64)
    omega_hist = np.asarray(npz["omega_hist"], dtype=np.float64)
    step_index = _safe_array(npz, "stats_step_index", dtype=np.int64)
    step_omega = _safe_array(npz, "stats_step_omega", dtype=np.float64)
    step_lambda = _safe_array(npz, "stats_step_lambda", dtype=np.float64)
    step_wall = _safe_array(npz, "stats_step_wall_time", dtype=np.float64)
    step_newton_total = _safe_array(npz, "stats_step_newton_iterations_total", dtype=np.int64)
    step_linear_total = _safe_array(npz, "stats_step_linear_iterations", dtype=np.int64)
    step_linear_per_newton = np.divide(
        step_linear_total.astype(np.float64),
        np.maximum(step_newton_total.astype(np.float64), 1.0),
    )
    step_d_omega = _safe_array(npz, "stats_step_d_omega", dtype=np.float64)
    step_next_d_omega = _safe_array(npz, "stats_step_next_d_omega", dtype=np.float64)
    step_scale = _safe_array(npz, "stats_step_d_omega_scale", dtype=np.float64)
    step_u_diff = _safe_array(npz, "stats_step_initial_guess_displacement_diff_volume_integral", dtype=np.float64)
    step_dev_diff = _safe_array(npz, "stats_step_initial_guess_deviatoric_strain_diff_volume_integral", dtype=np.float64)
    step_lambda_guess = _safe_array(npz, "stats_step_lambda_initial_guess", dtype=np.float64)
    step_predictor_kind = [str(v) for v in _safe_array(npz, "stats_step_predictor_kind", dtype=str).tolist()]
    step_predictor_basis_dim = _safe_array(npz, "stats_step_predictor_basis_dim", dtype=np.float64)
    step_predictor_wall = _safe_array(npz, "stats_step_predictor_wall_time", dtype=np.float64)
    if step_u_diff.size == 0:
        step_u_diff = np.full(step_lambda.shape, np.nan, dtype=np.float64)
    if step_dev_diff.size == 0:
        step_dev_diff = np.full(step_lambda.shape, np.nan, dtype=np.float64)
    if step_lambda_guess.size == 0:
        step_lambda_guess = np.full(step_lambda.shape, np.nan, dtype=np.float64)
    if step_scale.size == 0:
        step_scale = np.full(step_lambda.shape, np.nan, dtype=np.float64)
    if step_d_omega.size == 0:
        step_d_omega = np.full(step_lambda.shape, np.nan, dtype=np.float64)
    if not step_predictor_kind:
        step_predictor_kind = ["unknown"] * int(step_lambda.size)
    lambda_guess_abs_error = np.abs(step_lambda - step_lambda_guess)
    cumulative_step_wall = np.cumsum(step_wall, dtype=np.float64)

    summary = {
        "label": str(run_info.get("params", {}).get("continuation_predictor", "unknown")),
        "runtime_seconds": float(run_info.get("run_info", {}).get("runtime_seconds", np.nan)),
        "step_count": int(run_info.get("run_info", {}).get("step_count", len(lambda_hist))),
        "accepted_continuation_steps": int(max(len(lambda_hist) - 2, 0)),
        "final_lambda": float(lambda_hist[-1]) if lambda_hist.size else np.nan,
        "final_omega": float(omega_hist[-1]) if omega_hist.size else np.nan,
        "continuation_newton_total": int(np.sum(step_newton_total, dtype=np.int64)),
        "continuation_linear_total": int(np.sum(step_linear_total, dtype=np.int64)),
        "pc_apply_total": float(linear.get("preconditioner_apply_time_total", np.nan)),
        "pc_setup_total": float(linear.get("preconditioner_setup_time_total", np.nan)),
        "attempt_u_diff_total": float(predictor.get("attempt_initial_guess_displacement_diff_volume_integral_total", np.nan)),
        "attempt_dev_diff_total": float(predictor.get("attempt_initial_guess_deviatoric_strain_diff_volume_integral_total", np.nan)),
        "step_u_diff_total": float(predictor.get("step_initial_guess_displacement_diff_volume_integral_total", np.nan)),
        "step_dev_diff_total": float(predictor.get("step_initial_guess_deviatoric_strain_diff_volume_integral_total", np.nan)),
        "step_u_diff_last": float(predictor.get("step_initial_guess_displacement_diff_volume_integral_last", np.nan)),
        "step_dev_diff_last": float(predictor.get("step_initial_guess_deviatoric_strain_diff_volume_integral_last", np.nan)),
        "predictor_wall_total": float(predictor.get("step_predictor_wall_time_total", np.nan)),
        "predictor_wall_last": float(predictor.get("step_predictor_wall_time_last", np.nan)),
        "predictor_basis_dim_last": float(predictor.get("step_predictor_basis_dim_last", np.nan)),
        "predictor_basis_dim_max": float(predictor.get("step_predictor_basis_dim_max", np.nan)),
    }

    return {
        "case_dir": case_dir,
        "run_info": run_info,
        "summary": summary,
        "lambda_hist": lambda_hist,
        "omega_hist": omega_hist,
        "step_index": step_index,
        "step_omega": step_omega,
        "step_lambda": step_lambda,
        "step_wall": step_wall,
        "step_newton_total": step_newton_total,
        "step_linear_total": step_linear_total,
        "step_linear_per_newton": step_linear_per_newton,
        "step_d_omega": step_d_omega,
        "step_next_d_omega": step_next_d_omega,
        "step_scale": step_scale,
        "step_u_diff": step_u_diff,
        "step_dev_diff": step_dev_diff,
        "step_lambda_guess": step_lambda_guess,
        "lambda_guess_abs_error": lambda_guess_abs_error,
        "step_predictor_kind": step_predictor_kind,
        "step_predictor_basis_dim": step_predictor_basis_dim,
        "step_predictor_wall": step_predictor_wall,
        "cumulative_step_wall": cumulative_step_wall,
    }


def _plot_lambda_omega(cases: list[dict[str, object]], plot_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=160)
    for case in cases:
        label = str(case["summary"]["label"])
        ax.plot(case["omega_hist"], case["lambda_hist"], marker="o", linewidth=1.4, markersize=3.0, label=label)
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(r"P4(L1) smart controller: $\lambda$-$\omega$ trajectory")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    path = plot_dir / "lambda_omega.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_step_metrics(cases: list[dict[str, object]], plot_dir: Path) -> tuple[Path, Path, Path]:
    fig1, ax1 = plt.subplots(figsize=(8.5, 6.5), dpi=160)
    fig2, ax2 = plt.subplots(figsize=(8.5, 6.5), dpi=160)
    fig3, ax3 = plt.subplots(figsize=(8.5, 6.5), dpi=160)
    for case in cases:
        label = str(case["summary"]["label"])
        ax1.plot(case["step_omega"], case["step_wall"], marker="o", linewidth=1.2, markersize=3.0, label=label)
        ax2.plot(case["step_omega"], case["step_newton_total"], marker="o", linewidth=1.2, markersize=3.0, label=f"{label} Newton")
        ax2.plot(case["step_omega"], case["step_linear_total"], marker="s", linewidth=1.2, markersize=3.0, linestyle="--", label=f"{label} Linear")
        ax3.plot(case["step_omega"], case["step_linear_per_newton"], marker="o", linewidth=1.2, markersize=3.0, label=label)
    ax1.set_xlabel(r"$\omega$")
    ax1.set_ylabel("step wall time [s]")
    ax1.set_yscale("log")
    ax1.set_title("Accepted-step wall time")
    ax1.grid(True, alpha=0.35)
    ax1.legend()
    fig1.tight_layout()
    path1 = plot_dir / "step_wall_time.png"
    fig1.savefig(path1)
    plt.close(fig1)

    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel("iterations")
    ax2.set_yscale("log")
    ax2.set_title("Accepted-step Newton and linear work")
    ax2.grid(True, alpha=0.35)
    ax2.legend()
    fig2.tight_layout()
    path2 = plot_dir / "step_iterations.png"
    fig2.savefig(path2)
    plt.close(fig2)

    ax3.set_xlabel(r"$\omega$")
    ax3.set_ylabel("linear iterations / Newton iteration")
    ax3.set_title("Linear work per Newton step")
    ax3.grid(True, alpha=0.35)
    ax3.legend()
    fig3.tight_layout()
    path3 = plot_dir / "linear_per_newton.png"
    fig3.savefig(path3)
    plt.close(fig3)
    return path1, path2, path3


def _plot_predictor_metrics(cases: list[dict[str, object]], plot_dir: Path) -> tuple[Path, Path, Path]:
    fig1, ax1 = plt.subplots(figsize=(8.5, 6.5), dpi=160)
    fig2, ax2 = plt.subplots(figsize=(8.5, 6.5), dpi=160)
    fig3, ax3 = plt.subplots(figsize=(8.5, 6.5), dpi=160)
    for case in cases:
        label = str(case["summary"]["label"])
        ax1.plot(case["step_omega"], case["step_u_diff"], marker="o", linewidth=1.2, markersize=3.0, label=label)
        ax2.plot(case["step_omega"], case["step_dev_diff"], marker="o", linewidth=1.2, markersize=3.0, label=label)
        ax3.plot(case["step_omega"], case["lambda_guess_abs_error"], marker="o", linewidth=1.2, markersize=3.0, label=label)
    ax1.set_xlabel(r"$\omega$")
    ax1.set_ylabel(r"$\int \|\Delta u\|\, dV$")
    ax1.set_yscale("log")
    ax1.set_title("Predictor displacement mismatch")
    ax1.grid(True, alpha=0.35)
    ax1.legend()
    fig1.tight_layout()
    path1 = plot_dir / "predictor_displacement_mismatch.png"
    fig1.savefig(path1)
    plt.close(fig1)

    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel(r"$\int \|dev(\Delta \varepsilon)\|\, dV$")
    ax2.set_yscale("log")
    ax2.set_title("Predictor deviatoric-strain mismatch")
    ax2.grid(True, alpha=0.35)
    ax2.legend()
    fig2.tight_layout()
    path2 = plot_dir / "predictor_deviatoric_mismatch.png"
    fig2.savefig(path2)
    plt.close(fig2)

    ax3.set_xlabel(r"$\omega$")
    ax3.set_ylabel(r"$|\lambda_{\mathrm{guess}}-\lambda_{\mathrm{final}}|$")
    ax3.set_yscale("log")
    ax3.set_title("Predictor lambda guess error")
    ax3.grid(True, alpha=0.35)
    ax3.legend()
    fig3.tight_layout()
    path3 = plot_dir / "predictor_lambda_error.png"
    fig3.savefig(path3)
    plt.close(fig3)
    return path1, path2, path3


def _plot_step_control(cases: list[dict[str, object]], plot_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.5), dpi=160, sharex=True)
    ax1, ax2 = axes
    for case in cases:
        label = str(case["summary"]["label"])
        ax1.plot(case["step_omega"], case["step_d_omega"], marker="o", linewidth=1.2, markersize=3.0, label=label)
        ax2.plot(case["step_omega"], case["step_scale"], marker="o", linewidth=1.2, markersize=3.0, label=label)
    ax1.set_ylabel(r"accepted $d\omega$")
    ax1.set_yscale("log")
    ax1.set_title("Omega step control")
    ax1.grid(True, alpha=0.35)
    ax1.legend()
    ax2.set_xlabel(r"$\omega$")
    ax2.set_ylabel("next-step scale")
    ax2.grid(True, alpha=0.35)
    ax2.legend()
    fig.tight_layout()
    path = plot_dir / "step_control.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _summary_rows(cases: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Case | Predictor | Runtime [s] | Accepted continuation steps | Final omega | Final lambda | Continuation Newton | Continuation linear | PC apply [s] | PC setup [s] | Predictor wall [s] | Basis dim max | Predictor disp total | Predictor dev total |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        summary = case["summary"]
        label = case["run_info"].get("params", {}).get("continuation_predictor", summary["label"])
        lines.append(
            "| {case_label} | `{predictor}` | {runtime:.3f} | {steps:d} | {omega:.6e} | {lam:.9f} | {newton:d} | {linear:d} | {pc_apply:.3f} | {pc_setup:.3f} | {pred_wall:.3f} | {basis_dim:.0f} | {u_diff:.6e} | {dev_diff:.6e} |".format(
                case_label=str(summary["label"]),
                predictor=str(label),
                runtime=float(summary["runtime_seconds"]),
                steps=int(summary["accepted_continuation_steps"]),
                omega=float(summary["final_omega"]),
                lam=float(summary["final_lambda"]),
                newton=int(summary["continuation_newton_total"]),
                linear=int(summary["continuation_linear_total"]),
                pc_apply=float(summary["pc_apply_total"]),
                pc_setup=float(summary["pc_setup_total"]),
                pred_wall=float(summary["predictor_wall_total"]),
                basis_dim=float(summary["predictor_basis_dim_max"]),
                u_diff=float(summary["step_u_diff_total"]),
                dev_diff=float(summary["step_dev_diff_total"]),
            )
        )
    return lines


def _comparison_rows(baseline: dict[str, object], candidate: dict[str, object]) -> list[str]:
    b = baseline["summary"]
    c = candidate["summary"]
    runtime_ratio = float(c["runtime_seconds"]) / float(b["runtime_seconds"])
    newton_ratio = float(c["continuation_newton_total"]) / max(float(b["continuation_newton_total"]), 1.0)
    linear_ratio = float(c["continuation_linear_total"]) / max(float(b["continuation_linear_total"]), 1.0)
    pc_apply_ratio = float(c["pc_apply_total"]) / max(float(b["pc_apply_total"]), 1.0e-12)
    lambda_delta = float(c["final_lambda"]) - float(b["final_lambda"])
    b_u = float(b["step_u_diff_total"])
    c_u = float(c["step_u_diff_total"])
    b_dev = float(b["step_dev_diff_total"])
    c_dev = float(c["step_dev_diff_total"])
    u_ratio = "n/a" if not np.isfinite(b_u) or abs(b_u) < 1.0e-12 else f"{c_u / b_u:.6f}"
    dev_ratio = "n/a" if not np.isfinite(b_dev) or abs(b_dev) < 1.0e-12 else f"{c_dev / b_dev:.6f}"
    pred_wall_ratio = float(c["predictor_wall_total"]) / max(float(b["predictor_wall_total"]), 1.0e-12)
    return [
        f"| Metric | {candidate['summary']['label']} / {baseline['summary']['label']} |",
        "| --- | ---: |",
        f"| Runtime ratio | {runtime_ratio:.6f} |",
        f"| Continuation Newton ratio | {newton_ratio:.6f} |",
        f"| Continuation linear ratio | {linear_ratio:.6f} |",
        f"| PC apply ratio | {pc_apply_ratio:.6f} |",
        f"| Predictor wall ratio | {pred_wall_ratio:.6f} |",
        f"| Final lambda delta | {lambda_delta:.9e} |",
        f"| Final omega delta | {float(c['final_omega']) - float(b['final_omega']):.9e} |",
        f"| Predictor displacement total ratio | {u_ratio} |",
        f"| Predictor deviatoric total ratio | {dev_ratio} |",
    ]


def _step_table(case: dict[str, object]) -> list[str]:
    lines = [
        f"### {case['summary']['label']}",
        "",
        "| Accepted step | Omega | Lambda | dOmega | Step wall [s] | Newton total | Linear total | Linear/Newton | Predictor | Basis dim | Predictor wall [s] | Lambda guess | |lambda err| | Disp mismatch | Dev mismatch |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx in range(int(case["step_index"].size)):
        lines.append(
            "| {step:d} | {omega:.6e} | {lam:.9f} | {domega:.6e} | {wall:.3f} | {newton:d} | {linear:d} | {lin_per_newton:.3f} | `{kind}` | {basis_dim:.0f} | {pred_wall:.3f} | {lam_guess:.9f} | {lam_err:.6e} | {u_diff:.6e} | {dev_diff:.6e} |".format(
                step=int(case["step_index"][idx]),
                omega=float(case["step_omega"][idx]),
                lam=float(case["step_lambda"][idx]),
                domega=float(case["step_d_omega"][idx]) if idx < case["step_d_omega"].size else np.nan,
                wall=float(case["step_wall"][idx]),
                newton=int(case["step_newton_total"][idx]),
                linear=int(case["step_linear_total"][idx]),
                lin_per_newton=float(case["step_linear_per_newton"][idx]),
                kind=str(case["step_predictor_kind"][idx]) if idx < len(case["step_predictor_kind"]) else "unknown",
                basis_dim=float(case["step_predictor_basis_dim"][idx]) if idx < case["step_predictor_basis_dim"].size else np.nan,
                pred_wall=float(case["step_predictor_wall"][idx]) if idx < case["step_predictor_wall"].size else np.nan,
                lam_guess=float(case["step_lambda_guess"][idx]) if idx < case["step_lambda_guess"].size else np.nan,
                lam_err=float(case["lambda_guess_abs_error"][idx]) if idx < case["lambda_guess_abs_error"].size else np.nan,
                u_diff=float(case["step_u_diff"][idx]) if idx < case["step_u_diff"].size else np.nan,
                dev_diff=float(case["step_dev_diff"][idx]) if idx < case["step_dev_diff"].size else np.nan,
            )
        )
    lines.append("")
    return lines


def generate_report(*, baseline_dir: Path, candidate_dir: Path, out_dir: Path) -> dict[str, object]:
    out_dir = _ensure_dir(out_dir)
    plot_dir = _ensure_dir(out_dir / "plots")
    baseline = _load_case(baseline_dir)
    candidate = _load_case(candidate_dir)
    baseline["summary"]["label"] = f"smart controller + {baseline['run_info'].get('params', {}).get('continuation_predictor', 'secant')}"
    candidate["summary"]["label"] = f"smart controller + {candidate['run_info'].get('params', {}).get('continuation_predictor', 'candidate')}"
    cases = [baseline, candidate]

    lambda_omega = _plot_lambda_omega(cases, plot_dir)
    step_wall, step_iters, linear_per_newton = _plot_step_metrics(cases, plot_dir)
    pred_u, pred_dev, pred_lambda = _plot_predictor_metrics(cases, plot_dir)
    step_control = _plot_step_control(cases, plot_dir)

    summary = {
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "baseline": baseline["summary"],
        "candidate": candidate["summary"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = out_dir / "README.md"
    lines: list[str] = [
        "# P4(L1) Smart Controller Predictor Comparison",
        "",
        "Comparison of P4(L1) smart-controller continuation runs with different predictor choices.",
        "",
        "## Summary",
        "",
        *_summary_rows(cases),
        "",
        (
            "Note: the secant baseline artifact predates the predictor-mismatch instrumentation "
            "(`u`/deviatoric mismatch and lambda-guess error), so those baseline fields are shown as `nan`/`unknown` "
            "rather than recomputed retroactively."
        ),
        "",
        "## Relative Change",
        "",
        *_comparison_rows(baseline, candidate),
        "",
        "## Plots",
        "",
        f"![Lambda Omega]({_relpath(report_path, lambda_omega)})",
        "",
        f"![Step Wall Time]({_relpath(report_path, step_wall)})",
        "",
        f"![Step Iterations]({_relpath(report_path, step_iters)})",
        "",
        f"![Linear Per Newton]({_relpath(report_path, linear_per_newton)})",
        "",
        f"![Predictor Displacement Mismatch]({_relpath(report_path, pred_u)})",
        "",
        f"![Predictor Deviatoric Mismatch]({_relpath(report_path, pred_dev)})",
        "",
        f"![Predictor Lambda Error]({_relpath(report_path, pred_lambda)})",
        "",
        f"![Step Control]({_relpath(report_path, step_control)})",
        "",
        "## Accepted-Step Tables",
        "",
        *_step_table(baseline),
        *_step_table(candidate),
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "report": str(report_path),
        "summary": str(out_dir / "summary.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare secant vs two-step predictor for the P4(L1) smart omega controller.")
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    result = generate_report(
        baseline_dir=Path(args.baseline_dir),
        candidate_dir=Path(args.candidate_dir),
        out_dir=Path(args.out_dir),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
