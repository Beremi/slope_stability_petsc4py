#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
BENCHMARK_DIR = SCRIPT_DIR if (SCRIPT_DIR / "case.toml").exists() else SCRIPT_DIR.parent
ROOT = BENCHMARK_DIR.parents[1]
CASE_TOML = BENCHMARK_DIR / "case.toml"
DEFAULT_ARTIFACT_ROOT = (
    ROOT / "artifacts" / "comparisons" / "slope_stability_3D_hetero_SSR_default" / "step8_newton_default_vs_less_precise_x100"
)
DEFAULT_REPORT_PATH = SCRIPT_DIR / "step8_newton_comparison.md"
PYTHON = ROOT / ".venv" / "bin" / "python"
SRC_DIR = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import notebook_support as nb


@dataclass(frozen=True)
class ReplayCase:
    key: str
    label: str
    tol: float
    out_dir: Path
    config_path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay step 8 for default and less-precise-x100 Newton tolerances and plot Newton traces."
    )
    parser.add_argument(
        "--omega-target",
        type=float,
        default=6.54e6,
        help="Stop target chosen to terminate immediately after accepted step 8.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Output root for configs, runs, plots, and extracted traces.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown report to write.",
    )
    parser.add_argument("--force", action="store_true", help="Delete existing replay artifacts and rerun.")
    parser.add_argument("--report-only", action="store_true", help="Build report from existing replay artifacts.")
    return parser.parse_args()


def _rel(path: Path, report_path: Path) -> str:
    return os.path.relpath(path, start=report_path.parent).replace(os.sep, "/")


def _format_float(value: float, digits: int = 6) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _format_sci(value: float) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.3e}"


def _case_specs(base_tol: float, artifact_root: Path) -> list[ReplayCase]:
    return [
        ReplayCase(
            key="default",
            label="Default",
            tol=float(base_tol),
            out_dir=artifact_root / "runs" / "default",
            config_path=artifact_root / "configs" / "default.toml",
        ),
        ReplayCase(
            key="less_precise_x100",
            label="Less precise x100",
            tol=float(base_tol * 100.0),
            out_dir=artifact_root / "runs" / "less_precise_x100",
            config_path=artifact_root / "configs" / "less_precise_x100.toml",
        ),
    ]


def _write_config(
    *,
    case: ReplayCase,
    base_sections: dict[str, dict[str, Any]],
    materials: list[dict[str, Any]],
    omega_target: float,
) -> None:
    sections = {name: copy.deepcopy(values) for name, values in base_sections.items()}
    sections.setdefault("continuation", {})
    sections.setdefault("newton", {})
    sections.setdefault("export", {})
    sections["continuation"]["method"] = "indirect"
    sections["continuation"]["predictor"] = "secant"
    sections["continuation"]["omega_max"] = float(omega_target)
    sections["newton"]["tol"] = float(case.tol)
    sections["export"]["write_custom_debug_bundle"] = False
    case.config_path.parent.mkdir(parents=True, exist_ok=True)
    case.config_path.write_text(nb.render_case_toml(sections, materials), encoding="utf-8")


def _run_case(case: ReplayCase, *, force: bool) -> None:
    if force and case.out_dir.exists():
        shutil.rmtree(case.out_dir)
    case.out_dir.mkdir(parents=True, exist_ok=True)
    history_path = case.out_dir / "exports" / "continuation_history.json"
    if not force and history_path.exists():
        obj = json.loads(history_path.read_text())
        events = obj.get("progress_events", [])
        if any(
            e.get("event") == "newton_iteration"
            and e.get("target_step") == 8
            and "accepted_relative_correction_norm" in e
            for e in events
        ):
            print(f"[reuse] {case.label}: {case.out_dir}")
            return
    cmd = [
        str(PYTHON),
        "-m",
        "slope_stability.cli.run_case_from_config",
        str(case.config_path),
        "--out_dir",
        str(case.out_dir),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_DIR)
    print(f"[run] {case.label}: {' '.join(cmd)}", flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{case.key}] {line.rstrip()}", flush=True)
    return_code = int(process.wait())
    if return_code != 0:
        raise RuntimeError(f"{case.label} replay failed with exit code {return_code}")


def _extract_step8_trace(case: ReplayCase) -> dict[str, Any]:
    history_path = case.out_dir / "exports" / "continuation_history.json"
    obj = json.loads(history_path.read_text())
    events = obj.get("progress_events", [])
    step8_events = [
        e
        for e in events
        if e.get("event") == "newton_iteration" and int(e.get("target_step", -1)) == 8 and int(e.get("attempt_in_step", 1)) == 1
    ]
    if not step8_events:
        raise RuntimeError(f"No step-8 Newton events found for {case.label}")
    step_accept = next(
        (
            e
            for e in events
            if e.get("event") == "step_accepted" and int(e.get("accepted_step", -1)) == 8
        ),
        None,
    )
    attempt_complete = next(
        (
            e
            for e in events
            if e.get("event") == "attempt_complete" and int(e.get("target_step", -1)) == 8 and int(e.get("attempt_in_step", 1)) == 1
        ),
        None,
    )
    trace = {
        "label": case.label,
        "key": case.key,
        "tol": case.tol,
        "iterations": [int(e.get("iteration", 0)) for e in step8_events],
        "criterion": [float(e.get("criterion", np.nan)) for e in step8_events],
        "rel_residual": [float(e.get("rel_residual", np.nan)) for e in step8_events],
        "lambda": [float(e.get("lambda_value", np.nan)) for e in step8_events],
        "accepted_correction_norm": [
            float(e["accepted_correction_norm"]) if e.get("accepted_correction_norm") is not None else np.nan
            for e in step8_events
        ],
        "iterate_free_norm": [
            float(e["iterate_free_norm"]) if e.get("iterate_free_norm") is not None else np.nan
            for e in step8_events
        ],
        "accepted_relative_correction_norm": [
            float(e["accepted_relative_correction_norm"]) if e.get("accepted_relative_correction_norm") is not None else np.nan
            for e in step8_events
        ],
        "alpha": [float(e.get("alpha", np.nan)) if e.get("alpha") is not None else np.nan for e in step8_events],
        "accepted_delta_lambda": [
            float(e.get("accepted_delta_lambda", np.nan)) if e.get("accepted_delta_lambda") is not None else np.nan
            for e in step8_events
        ],
        "omega_target": float(step8_events[0].get("omega_target", np.nan)),
        "lambda_before": float(step8_events[0].get("lambda_before", np.nan)),
        "newton_iterations": None if attempt_complete is None else int(attempt_complete.get("newton_iterations", 0)),
        "newton_relres_end": None
        if attempt_complete is None or attempt_complete.get("newton_relres_end") is None
        else float(attempt_complete.get("newton_relres_end")),
        "final_lambda": None if step_accept is None else float(step_accept.get("lambda_value", np.nan)),
        "final_omega": None if step_accept is None else float(step_accept.get("omega_value", np.nan)),
    }
    return trace


def _plot_pair(
    traces: list[dict[str, Any]],
    *,
    plots_dir: Path,
    key: str,
    ylabel: str,
    title: str,
    value_getter,
    logy: bool = False,
) -> Path:
    colors = {
        "default": "#0b63a3",
        "less_precise_x100": "#9b2226",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    for trace in traces:
        x = np.asarray(trace["iterations"], dtype=np.int64)
        y = np.asarray(value_getter(trace), dtype=np.float64)
        ax.plot(x, y, marker="o", linewidth=1.6, label=trace["label"], color=colors[trace["key"]])
    ax.set_xlabel("Newton iteration in continuation step 8")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = plots_dir / f"{key}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _plot_xy_pair(
    traces: list[dict[str, Any]],
    *,
    plots_dir: Path,
    key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    x_getter,
    y_getter,
    logx: bool = False,
    logy: bool = False,
) -> Path:
    colors = {
        "default": "#0b63a3",
        "less_precise_x100": "#9b2226",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    for trace in traces:
        x = np.asarray(x_getter(trace), dtype=np.float64)
        y = np.asarray(y_getter(trace), dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        ax.plot(
            x[mask],
            y[mask],
            marker="o",
            linewidth=1.6,
            label=trace["label"],
            color=colors[trace["key"]],
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = plots_dir / f"{key}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _write_report(
    *,
    report_path: Path,
    artifact_root: Path,
    omega_target: float,
    traces: list[dict[str, Any]],
    plot_paths: dict[str, Path],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Step 8 Newton Comparison")
    lines.append("")
    lines.append(
        "This replays the `slope_stability_3D_hetero_SSR_default` secant case only up to just beyond accepted continuation step 8 and compares the step-8 Newton solver trace for `Default` vs `Less precise x100`."
    )
    lines.append("")
    lines.append(f"- Replay `omega_max`: `{omega_target:.3e}`")
    lines.append("- `delta U` is plotted as the accepted free-DOF Newton correction norm `||alpha ΔU||` per Newton iteration.")
    lines.append(
        "- `delta U / U` is plotted as the relative accepted free-DOF Newton correction norm `||alpha ΔU|| / ||U||`, where `||U||` is the current Newton iterate free-DOF norm."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Case | Newton tol | Step-8 target omega | Step-8 final lambda | Step-8 final omega | Step-8 Newton iterations | Step-8 final relres |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for trace in traces:
        lines.append(
            "| "
            + " | ".join(
                [
                    trace["label"],
                    f"`{trace['tol']:.1e}`",
                    _format_float(trace["omega_target"], 1),
                    _format_float(trace["final_lambda"]),
                    _format_float(trace["final_omega"], 1),
                    str(trace["newton_iterations"]),
                    _format_sci(trace["newton_relres_end"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for title, key in [
        ("Criterion", "criterion"),
        ("Lambda", "lambda"),
        ("Delta U", "delta_u"),
        ("Delta U / U", "delta_u_over_u"),
        ("Newton Correction Norm vs Lambda", "correction_norm_vs_lambda"),
        ("Newton Correction Norm vs Criterion", "correction_norm_vs_criterion"),
        ("Relative Increment vs Lambda", "relative_increment_vs_lambda"),
        ("Relative Increment vs Criterion", "relative_increment_vs_criterion"),
        ("Lambda vs Criterion", "lambda_vs_criterion"),
    ]:
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}]({_rel(plot_paths[key], report_path)})")
        lines.append("")
    lines.append("## Step-8 Newton Data")
    lines.append("")
    lines.append(
        "| Iteration | Default criterion | Less precise x100 criterion | Default lambda | Less precise x100 lambda | Default `||alpha ΔU||` | Less precise x100 `||alpha ΔU||` | Default `||alpha ΔU|| / ||U||` | Less precise x100 `||alpha ΔU|| / ||U||` |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    by_key = {trace["key"]: trace for trace in traces}
    max_iter = max(len(trace["iterations"]) for trace in traces)
    for i in range(max_iter):
        row = [str(i + 1)]
        for field in ("criterion", "lambda", "accepted_correction_norm", "accepted_relative_correction_norm"):
            for key in ("default", "less_precise_x100"):
                vals = by_key[key][field]
                value = vals[i] if i < len(vals) else np.nan
                row.append(_format_sci(value) if field in ("criterion", "accepted_relative_correction_norm") else _format_float(value))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for trace in traces:
        run_dir = artifact_root / "runs" / trace["key"]
        config_path = artifact_root / "configs" / f"{trace['key']}.toml"
        lines.append(f"- {trace['label']}: config `{_rel(config_path, report_path)}`, artifact `{_rel(run_dir, report_path)}`")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    artifact_root = args.artifact_root.resolve()
    report_path = args.report_path.resolve()
    base_sections = nb.load_case_sections(CASE_TOML)
    materials = nb.load_case_materials(CASE_TOML)
    base_tol = float(base_sections.get("newton", {}).get("tol", 1.0e-4))
    cases = _case_specs(base_tol, artifact_root)
    for case in cases:
        _write_config(case=case, base_sections=base_sections, materials=materials, omega_target=float(args.omega_target))

    if not args.report_only:
        for case in cases:
            _run_case(case, force=bool(args.force))

    traces = [_extract_step8_trace(case) for case in cases]
    plots_dir = artifact_root / "report" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {
        "criterion": _plot_pair(
            traces,
            plots_dir=plots_dir,
            key="criterion",
            ylabel="Criterion",
            title="Step 8 Newton Criterion",
            value_getter=lambda trace: trace["criterion"],
            logy=True,
        ),
        "lambda": _plot_pair(
            traces,
            plots_dir=plots_dir,
            key="lambda",
            ylabel=r"$\lambda$",
            title="Step 8 Newton Lambda",
            value_getter=lambda trace: trace["lambda"],
            logy=False,
        ),
        "delta_u": _plot_pair(
            traces,
            plots_dir=plots_dir,
            key="delta_u",
            ylabel=r"$||\alpha \Delta U||$ (free DOFs)",
            title=r"Step 8 Accepted Newton Correction Norm",
            value_getter=lambda trace: trace["accepted_correction_norm"],
            logy=True,
        ),
        "delta_u_over_u": _plot_pair(
            traces,
            plots_dir=plots_dir,
            key="delta_u_over_u",
            ylabel=r"$||\alpha \Delta U|| / ||U||$",
            title=r"Step 8 Relative Newton Correction",
            value_getter=lambda trace: trace["accepted_relative_correction_norm"],
            logy=True,
        ),
        "correction_norm_vs_lambda": _plot_xy_pair(
            traces,
            plots_dir=plots_dir,
            key="correction_norm_vs_lambda",
            xlabel=r"$\lambda$",
            ylabel=r"$||\alpha \Delta U||$ (free DOFs)",
            title=r"Step 8 Newton Correction Norm vs $\lambda$",
            x_getter=lambda trace: trace["lambda"],
            y_getter=lambda trace: trace["accepted_correction_norm"],
            logy=True,
        ),
        "correction_norm_vs_criterion": _plot_xy_pair(
            traces,
            plots_dir=plots_dir,
            key="correction_norm_vs_criterion",
            xlabel="Criterion",
            ylabel=r"$||\alpha \Delta U||$ (free DOFs)",
            title=r"Step 8 Newton Correction Norm vs Criterion",
            x_getter=lambda trace: trace["criterion"],
            y_getter=lambda trace: trace["accepted_correction_norm"],
            logx=True,
            logy=True,
        ),
        "relative_increment_vs_lambda": _plot_xy_pair(
            traces,
            plots_dir=plots_dir,
            key="relative_increment_vs_lambda",
            xlabel=r"$\lambda$",
            ylabel=r"$||\alpha \Delta U|| / ||U||$",
            title=r"Step 8 Relative Increment vs $\lambda$",
            x_getter=lambda trace: trace["lambda"],
            y_getter=lambda trace: trace["accepted_relative_correction_norm"],
            logy=True,
        ),
        "relative_increment_vs_criterion": _plot_xy_pair(
            traces,
            plots_dir=plots_dir,
            key="relative_increment_vs_criterion",
            xlabel="Criterion",
            ylabel=r"$||\alpha \Delta U|| / ||U||$",
            title=r"Step 8 Relative Increment vs Criterion",
            x_getter=lambda trace: trace["criterion"],
            y_getter=lambda trace: trace["accepted_relative_correction_norm"],
            logx=True,
            logy=True,
        ),
        "lambda_vs_criterion": _plot_xy_pair(
            traces,
            plots_dir=plots_dir,
            key="lambda_vs_criterion",
            xlabel="Criterion",
            ylabel=r"$\lambda$",
            title=r"Step 8 Newton $\lambda$ vs Criterion",
            x_getter=lambda trace: trace["criterion"],
            y_getter=lambda trace: trace["lambda"],
            logx=True,
        ),
    }
    summary_path = artifact_root / "report" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "omega_target": float(args.omega_target),
                "cases": traces,
                "plots": {key: str(path) for key, path in plot_paths.items()},
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    _write_report(
        report_path=report_path,
        artifact_root=artifact_root,
        omega_target=float(args.omega_target),
        traces=traces,
        plot_paths=plot_paths,
    )
    print(f"[done] report: {report_path}")
    print(f"[done] summary: {summary_path}")


if __name__ == "__main__":
    main()
