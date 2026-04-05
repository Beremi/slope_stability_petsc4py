#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
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
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "comparisons" / "slope_stability_3D_hetero_SSR_default" / "p4_l1_pmg_newton_stops_omega6p7e6"
DEFAULT_REPORT_PATH = SCRIPT_DIR / "comparisons_p4_l1_pmg_partial.md"


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


COLOR_DEFAULT = "#0b63a3"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a live partial markdown report for the in-progress P4(L1) PMG sweep."
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--report-title",
        type=str,
        default="P4(L1) PMG Newton-Stop Comparison (Partial Live)",
    )
    return parser.parse_args()


def _rel(path: Path, report_path: Path) -> str:
    return os.path.relpath(path, start=report_path.parent).replace(os.sep, "/")


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _format_sci(value: float | None) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.3e}"


def _load_events(progress_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _latest_timestamp(events: list[dict[str, Any]]) -> str:
    if not events:
        return "n/a"
    return str(events[-1].get("timestamp", "n/a"))


def _accepted_step_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (
            event
            for event in events
            if str(event.get("event", "")) == "step_accepted" and _coerce_int(event.get("accepted_step"), -1) >= 1
        ),
        key=lambda event: _coerce_int(event.get("accepted_step"), -1),
    )


def _runtime_so_far(events: list[dict[str, Any]]) -> float:
    if not events:
        return float("nan")
    indexed_totals = [
        (idx, _coerce_float(event.get("total_wall_time")))
        for idx, event in enumerate(events)
        if np.isfinite(_coerce_float(event.get("total_wall_time")))
    ]
    if not indexed_totals:
        return float(
            np.nansum(
                [
                    _coerce_float(event.get("iteration_wall_time"))
                    for event in events
                    if str(event.get("event", "")) == "newton_iteration"
                ]
            )
        )
    last_idx, total_wall = max(indexed_totals, key=lambda item: item[0])
    tail_newton_time = float(
        np.nansum(
            [
                _coerce_float(event.get("iteration_wall_time"))
                for event in events[last_idx + 1 :]
                if str(event.get("event", "")) == "newton_iteration"
            ]
        )
    )
    return float(total_wall + tail_newton_time)


def _current_timing_components(events: list[dict[str, Any]]) -> dict[str, float]:
    init_event = next((event for event in events if str(event.get("event", "")) == "init_complete"), None)
    init_linear = _coerce_float(None if init_event is None else init_event.get("init_linear_solve_time"), 0.0)
    init_pc = _coerce_float(None if init_event is None else init_event.get("init_linear_preconditioner_time"), 0.0)
    init_orth = _coerce_float(None if init_event is None else init_event.get("init_linear_orthogonalization_time"), 0.0)

    step_accepts = _accepted_step_events(events)
    continuation_linear = float(np.nansum([_coerce_float(event.get("step_linear_solve_time"), 0.0) for event in step_accepts]))
    continuation_pc = float(np.nansum([_coerce_float(event.get("step_linear_preconditioner_time"), 0.0) for event in step_accepts]))
    continuation_orth = float(np.nansum([_coerce_float(event.get("step_linear_orthogonalization_time"), 0.0) for event in step_accepts]))

    indexed_totals = [
        (idx, _coerce_float(event.get("total_wall_time")))
        for idx, event in enumerate(events)
        if np.isfinite(_coerce_float(event.get("total_wall_time")))
    ]
    after_total = events[(max(indexed_totals, key=lambda item: item[0])[0] + 1) :] if indexed_totals else events
    tail_linear = float(
        np.nansum(
            [
                _coerce_float(event.get("linear_solve_time"))
                for event in after_total
                if str(event.get("event", "")) == "newton_iteration"
            ]
        )
    )
    tail_pc = float(
        np.nansum(
            [
                _coerce_float(event.get("linear_preconditioner_time"))
                for event in after_total
                if str(event.get("event", "")) == "newton_iteration"
            ]
        )
    )
    tail_orth = float(
        np.nansum(
            [
                _coerce_float(event.get("linear_orthogonalization_time"))
                for event in after_total
                if str(event.get("event", "")) == "newton_iteration"
            ]
        )
    )

    linear = init_linear + continuation_linear + tail_linear
    pc_apply = init_pc + continuation_pc + tail_pc
    orth = init_orth + continuation_orth + tail_orth
    runtime = _runtime_so_far(events)
    other = max(runtime - linear - pc_apply - orth, 0.0) if np.isfinite(runtime) else float("nan")
    return {
        "linear_solve": float(linear),
        "pc_apply": float(pc_apply),
        "orthogonalization": float(orth),
        "other": float(other),
        "runtime": float(runtime),
    }


def _build_live_case(events: list[dict[str, Any]]) -> dict[str, Any]:
    init_event = next((event for event in events if str(event.get("event", "")) == "init_complete"), None)
    if init_event is None:
        raise RuntimeError("No init_complete event found in progress.jsonl")

    accepted_steps = _accepted_step_events(events)
    lambda_hist = [_coerce_float(value) for value in list(init_event.get("lambda_hist", []))]
    omega_hist = [_coerce_float(value) for value in list(init_event.get("omega_hist", []))]
    for event in accepted_steps:
        lambda_hist.append(_coerce_float(event.get("lambda_value")))
        omega_hist.append(_coerce_float(event.get("omega_value")))

    latest = events[-1]
    current_target_step = _coerce_int(latest.get("target_step"), -1)
    current_attempt = _coerce_int(latest.get("attempt_in_step"), 1)

    step_indices = [_coerce_int(event.get("accepted_step"), -1) for event in accepted_steps]
    step_lambda = [_coerce_float(event.get("lambda_value")) for event in accepted_steps]
    step_omega = [_coerce_float(event.get("omega_value")) for event in accepted_steps]
    step_newton = [_coerce_float(event.get("step_newton_iterations")) for event in accepted_steps]
    step_linear = [_coerce_float(event.get("step_linear_iterations")) for event in accepted_steps]
    step_wall = [_coerce_float(event.get("step_wall_time")) for event in accepted_steps]
    step_relres = [_coerce_float(event.get("step_newton_relres_end")) for event in accepted_steps]
    step_relcorr = [_coerce_float(event.get("step_newton_relcorr_end")) for event in accepted_steps]

    latest_lambda = _coerce_float(latest.get("lambda_value"), lambda_hist[-1] if lambda_hist else float("nan"))
    latest_omega = _coerce_float(latest.get("omega_value"), omega_hist[-1] if omega_hist else float("nan"))
    latest_relres = _coerce_float(latest.get("rel_residual"))
    latest_relcorr = _coerce_float(latest.get("accepted_relative_correction_norm"))

    accepted_step_max = max(step_indices) if step_indices else 2
    live_status = "accepted" if current_target_step <= accepted_step_max else "running"

    init_newton_total = int(np.nansum(np.asarray(init_event.get("init_newton_iterations", []), dtype=np.float64)))
    init_linear_total = _coerce_int(init_event.get("init_linear_iterations"), 0)
    continuation_newton_total = int(np.nansum(np.asarray(step_newton, dtype=np.float64)))
    continuation_linear_total = int(np.nansum(np.asarray(step_linear, dtype=np.float64)))
    total_linear_per_newton = (
        float(continuation_linear_total) / float(continuation_newton_total)
        if continuation_newton_total > 0
        else float("nan")
    )

    timing = _current_timing_components(events)

    return {
        "key": "default",
        "label": "Default",
        "tol": 1.0e-4,
        "stopping_criterion": "relative_residual",
        "stopping_tol": 1.0e-4,
        "status": live_status,
        "timestamp": _latest_timestamp(events),
        "runtime_seconds": _runtime_so_far(events),
        "accepted_states": len(lambda_hist),
        "continuation_steps": len(step_indices),
        "final_lambda": lambda_hist[-1] if lambda_hist else float("nan"),
        "final_omega": omega_hist[-1] if omega_hist else float("nan"),
        "current_target_step": current_target_step,
        "current_attempt": current_attempt,
        "current_lambda": latest_lambda,
        "current_omega": latest_omega,
        "current_relres": latest_relres,
        "current_relcorr": latest_relcorr,
        "init_newton_total": init_newton_total,
        "continuation_newton_total": continuation_newton_total,
        "init_linear_total": init_linear_total,
        "continuation_linear_total": continuation_linear_total,
        "total_linear_per_newton": total_linear_per_newton,
        "lambda_hist": lambda_hist,
        "omega_hist": omega_hist,
        "step_indices": step_indices,
        "step_lambda": step_lambda,
        "step_omega": step_omega,
        "step_newton_iterations": step_newton,
        "step_linear_iterations": step_linear,
        "step_wall_time": step_wall,
        "step_relres": step_relres,
        "step_relcorr": step_relcorr,
        "timing": timing,
    }


def _extract_traces(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    step_accept_by_step: dict[int, dict[str, Any]] = {}
    success_attempt_by_step: dict[int, dict[str, Any]] = {}
    newton_events_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)

    for event in events:
        kind = str(event.get("event", ""))
        if kind == "step_accepted":
            step = _coerce_int(event.get("accepted_step"), -1)
            if step >= 1:
                step_accept_by_step[step] = event
            continue
        if kind == "attempt_complete":
            step = _coerce_int(event.get("target_step"), -1)
            if step >= 1 and bool(event.get("success", False)):
                success_attempt_by_step[step] = event
            continue
        if kind != "newton_iteration":
            continue
        step = _coerce_int(event.get("target_step"), -1)
        attempt = _coerce_int(event.get("attempt_in_step"), 1)
        if step >= 1:
            newton_events_by_key[(step, attempt)].append(event)

    traces: list[dict[str, Any]] = []
    for step in sorted(step_accept_by_step):
        step_accept = step_accept_by_step[step]
        attempt_complete = success_attempt_by_step.get(step)
        attempt = _coerce_int(None if attempt_complete is None else attempt_complete.get("attempt_in_step"), 1)
        if (step, attempt) not in newton_events_by_key:
            available_attempts = sorted(a for s, a in newton_events_by_key if s == step)
            if not available_attempts:
                continue
            attempt = int(available_attempts[-1])
        step_events = sorted(newton_events_by_key.get((step, attempt), []), key=lambda event: _coerce_int(event.get("iteration"), 0))
        if not step_events:
            continue
        traces.append(
            {
                "step": int(step),
                "label": "Default",
                "key": "default",
                "attempt_in_step": int(attempt),
                "iterations": [_coerce_int(event.get("iteration"), 0) for event in step_events],
                "criterion": [_coerce_float(event.get("criterion")) for event in step_events],
                "lambda": [_coerce_float(event.get("lambda_value")) for event in step_events],
                "delta_lambda": [_coerce_float(event.get("delta_lambda")) for event in step_events],
                "accepted_correction_norm": [_coerce_float(event.get("accepted_correction_norm")) for event in step_events],
                "accepted_relative_correction_norm": [
                    _coerce_float(event.get("accepted_relative_correction_norm")) for event in step_events
                ],
                "final_lambda": _coerce_float(step_accept.get("lambda_value")),
                "final_omega": _coerce_float(step_accept.get("omega_value")),
                "step_wall_time": _coerce_float(step_accept.get("step_wall_time")),
                "newton_iterations": _coerce_int(step_accept.get("step_newton_iterations"), len(step_events)),
                "final_relres": _coerce_float(step_accept.get("step_newton_relres_end")),
                "final_relcorr": _coerce_float(step_accept.get("step_newton_relcorr_end")),
                "accepted": True,
                "status": "accepted",
            }
        )

    latest_newton = next(
        (event for event in reversed(events) if str(event.get("event", "")) == "newton_iteration"),
        None,
    )
    if latest_newton is not None:
        current_step = _coerce_int(latest_newton.get("target_step"), -1)
        if current_step >= 1 and current_step not in step_accept_by_step:
            attempt = _coerce_int(latest_newton.get("attempt_in_step"), 1)
            step_events = sorted(
                newton_events_by_key.get((current_step, attempt), []),
                key=lambda event: _coerce_int(event.get("iteration"), 0),
            )
            if step_events:
                traces.append(
                    {
                        "step": int(current_step),
                        "label": "Default",
                        "key": "default",
                        "attempt_in_step": int(attempt),
                        "iterations": [_coerce_int(event.get("iteration"), 0) for event in step_events],
                        "criterion": [_coerce_float(event.get("criterion")) for event in step_events],
                        "lambda": [_coerce_float(event.get("lambda_value")) for event in step_events],
                        "delta_lambda": [_coerce_float(event.get("delta_lambda")) for event in step_events],
                        "accepted_correction_norm": [_coerce_float(event.get("accepted_correction_norm")) for event in step_events],
                        "accepted_relative_correction_norm": [
                            _coerce_float(event.get("accepted_relative_correction_norm")) for event in step_events
                        ],
                        "final_lambda": _coerce_float(step_events[-1].get("lambda_value")),
                        "final_omega": _coerce_float(step_events[-1].get("omega_value")),
                        "step_wall_time": float(
                            np.nansum([_coerce_float(event.get("iteration_wall_time")) for event in step_events])
                        ),
                        "newton_iterations": len(step_events),
                        "final_relres": _coerce_float(step_events[-1].get("rel_residual")),
                        "final_relcorr": _coerce_float(step_events[-1].get("accepted_relative_correction_norm")),
                        "accepted": False,
                        "status": "in_progress",
                    }
                )

    return sorted(traces, key=lambda trace: int(trace["step"]))


def _plot_single_series(
    *,
    x: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    logy: bool = False,
    current_point: tuple[float, float] | None = None,
    current_label: str = "Current Newton iterate",
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.any(mask):
        ax.plot(x[mask], y[mask], marker="o", linewidth=1.5, color=COLOR_DEFAULT, label="Accepted states")
    if current_point is not None and np.all(np.isfinite(np.asarray(current_point, dtype=np.float64))):
        ax.scatter([current_point[0]], [current_point[1]], marker="x", s=80, color="#9b2226", label=current_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_step_series(
    *,
    x: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    ylabel: str,
    title: str,
    logy: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    mask = np.isfinite(x) & np.isfinite(y)
    if logy:
        mask &= y > 0.0
    ax.plot(x[mask], y[mask], marker="o", linewidth=1.5, color=COLOR_DEFAULT, label="Default")
    ax.set_xlabel("Accepted continuation step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_runtime(runtime_seconds: float, *, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    bars = ax.bar(["Default (live)"], [runtime_seconds], color=[COLOR_DEFAULT])
    ax.set_ylabel("Seconds")
    ax.set_title("Runtime by Newton Stopping Case")
    for bar in bars:
        value = float(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_timing_breakdown(timing: dict[str, float], *, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)
    labels = ["Default (live)"]
    stacks = {
        "Linear solve": np.asarray([timing["linear_solve"]], dtype=np.float64),
        "PC apply": np.asarray([timing["pc_apply"]], dtype=np.float64),
        "Orthogonalization": np.asarray([timing["orthogonalization"]], dtype=np.float64),
        "Other": np.asarray([timing["other"]], dtype=np.float64),
    }
    palette = {
        "Linear solve": "#cc7000",
        "PC apply": "#73a857",
        "Orthogonalization": "#7b6db4",
        "Other": "#7f7f7f",
    }
    bottom = np.zeros(1, dtype=np.float64)
    for name, values in stacks.items():
        ax.bar(labels, values, bottom=bottom, label=name, color=palette[name])
        bottom += values
    ax.set_ylabel("Seconds")
    ax.set_title("Runtime Breakdown")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_step_iteration_series(
    trace: dict[str, Any],
    *,
    out_path: Path,
    ylabel: str,
    title: str,
    value_getter,
    logy: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    x = np.asarray(trace["iterations"], dtype=np.int64)
    y = np.asarray(value_getter(trace), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if logy:
        mask &= y > 0.0
    ax.plot(x[mask], y[mask], marker="o", linewidth=1.5, color=COLOR_DEFAULT, label=trace["label"])
    ax.set_xlabel(f"Newton iteration in continuation step {int(trace['step'])}")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _plot_step_xy(
    trace: dict[str, Any],
    *,
    out_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    x_getter,
    y_getter,
    logx: bool = False,
    logy: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    x = np.asarray(x_getter(trace), dtype=np.float64)
    y = np.asarray(y_getter(trace), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if logx:
        mask &= x > 0.0
    if logy:
        mask &= y > 0.0
    ax.plot(x[mask], y[mask], marker="o", linewidth=1.5, color=COLOR_DEFAULT, label=trace["label"])
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _build_step_plot_grid(*, report_path: Path, cells: list[tuple[str, Path]], columns: int) -> list[str]:
    lines: list[str] = []
    for start in range(0, len(cells), columns):
        chunk = list(cells[start : start + columns])
        while len(chunk) < columns:
            chunk.append(("", Path(".")))
        titles = [title for title, _ in chunk]
        images = [("" if not title else f"![{title}]({_rel(path, report_path)})") for title, path in chunk]
        lines.append("| " + " | ".join(titles) + " |")
        lines.append("| " + " | ".join(["---" for _ in chunk]) + " |")
        lines.append("| " + " | ".join(images) + " |")
        lines.append("")
    return lines


def _build_plots(case: dict[str, Any], traces: list[dict[str, Any]], *, artifact_root: Path) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    plots_dir = artifact_root / "report_partial_live" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    comparison_plots: dict[str, Path] = {}

    lambda_omega_path = plots_dir / "lambda_omega_overlay.png"
    _plot_single_series(
        x=np.asarray(case["omega_hist"], dtype=np.float64) / 1.0e6,
        y=np.asarray(case["lambda_hist"], dtype=np.float64),
        current_point=(float(case["current_omega"]) / 1.0e6, float(case["current_lambda"]))
        if case["status"] == "running"
        else None,
        current_label="Current step iterate",
        out_path=lambda_omega_path,
        xlabel=r"$\omega$ [$10^6$]",
        ylabel=r"$\lambda$",
        title=r"Continuation Curve to $\omega = 6.7 \times 10^6$",
    )
    comparison_plots["lambda_omega_overlay.png"] = lambda_omega_path

    lambda_state_path = plots_dir / "lambda_vs_state.png"
    _plot_single_series(
        x=np.arange(1, len(case["lambda_hist"]) + 1, dtype=np.int64),
        y=np.asarray(case["lambda_hist"], dtype=np.float64),
        out_path=lambda_state_path,
        xlabel="Accepted state",
        ylabel=r"$\lambda$",
        title="Lambda by Accepted State",
    )
    comparison_plots["lambda_vs_state.png"] = lambda_state_path

    omega_state_path = plots_dir / "omega_vs_state.png"
    _plot_single_series(
        x=np.arange(1, len(case["omega_hist"]) + 1, dtype=np.int64),
        y=np.asarray(case["omega_hist"], dtype=np.float64),
        out_path=omega_state_path,
        xlabel="Accepted state",
        ylabel=r"$\omega$",
        title="Omega by Accepted State",
    )
    comparison_plots["omega_vs_state.png"] = omega_state_path

    runtime_path = plots_dir / "runtime_by_case.png"
    _plot_runtime(float(case["runtime_seconds"]), out_path=runtime_path)
    comparison_plots["runtime_by_case.png"] = runtime_path

    timing_path = plots_dir / "timing_breakdown_stacked.png"
    _plot_timing_breakdown(case["timing"], out_path=timing_path)
    comparison_plots["timing_breakdown_stacked.png"] = timing_path

    step_indices = np.asarray(case["step_indices"], dtype=np.float64)
    step_newton = np.asarray(case["step_newton_iterations"], dtype=np.float64)
    step_linear = np.asarray(case["step_linear_iterations"], dtype=np.float64)
    step_wall = np.asarray(case["step_wall_time"], dtype=np.float64)
    step_relres = np.asarray(case["step_relres"], dtype=np.float64)
    step_relcorr = np.asarray(case["step_relcorr"], dtype=np.float64)

    for filename, values, ylabel, title, logy in [
        ("step_newton_iterations.png", step_newton, "Newton iterations", "Newton Iterations per Accepted Step", False),
        ("step_linear_iterations.png", step_linear, "Linear iterations", "Linear Iterations per Accepted Step", False),
        ("step_wall_time.png", step_wall, "Seconds", "Wall Time per Accepted Step", False),
        ("step_relres_end.png", step_relres, "Relative residual", "Final Newton Relative Residual per Accepted Step", True),
        (
            "step_relcorr_end.png",
            step_relcorr,
            r"$||\alpha \Delta U|| / ||U||$",
            "Final Newton Relative Correction per Accepted Step",
            True,
        ),
    ]:
        path = plots_dir / filename
        _plot_step_series(x=step_indices, y=np.asarray(values, dtype=np.float64), out_path=path, ylabel=ylabel, title=title, logy=logy)
        comparison_plots[filename] = path

    linear_per_newton = np.divide(step_linear, step_newton, out=np.full_like(step_linear, np.nan), where=np.abs(step_newton) > 0.0)
    linear_per_newton_path = plots_dir / "step_linear_per_newton.png"
    _plot_step_series(
        x=step_indices,
        y=linear_per_newton,
        out_path=linear_per_newton_path,
        ylabel="Linear / Newton",
        title="Linear Iterations per Newton Iteration",
    )
    comparison_plots["step_linear_per_newton.png"] = linear_per_newton_path

    step_sections: list[dict[str, Any]] = []
    step_plots_root = plots_dir / "newton_by_step"
    for trace in traces:
        step_dir = step_plots_root / f"step_{int(trace['step']):02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        plots = {
            "criterion.png": step_dir / "criterion.png",
            "lambda.png": step_dir / "lambda.png",
            "delta_lambda.png": step_dir / "delta_lambda.png",
            "delta_u.png": step_dir / "delta_u.png",
            "delta_u_over_u.png": step_dir / "delta_u_over_u.png",
            "correction_norm_vs_lambda.png": step_dir / "correction_norm_vs_lambda.png",
            "correction_norm_vs_criterion.png": step_dir / "correction_norm_vs_criterion.png",
            "lambda_vs_criterion.png": step_dir / "lambda_vs_criterion.png",
            "relative_increment_vs_lambda.png": step_dir / "relative_increment_vs_lambda.png",
            "relative_increment_vs_criterion.png": step_dir / "relative_increment_vs_criterion.png",
        }
        _plot_step_iteration_series(
            trace,
            out_path=plots["criterion.png"],
            ylabel="Criterion",
            title=f"Step {int(trace['step'])}: Criterion",
            value_getter=lambda current: current["criterion"],
            logy=True,
        )
        _plot_step_iteration_series(
            trace,
            out_path=plots["lambda.png"],
            ylabel=r"$\lambda$",
            title=f"Step {int(trace['step'])}: Lambda",
            value_getter=lambda current: current["lambda"],
        )
        _plot_step_iteration_series(
            trace,
            out_path=plots["delta_lambda.png"],
            ylabel=r"$|\Delta \lambda|$",
            title=f"Step {int(trace['step'])}: Absolute Delta Lambda",
            value_getter=lambda current: np.abs(np.asarray(current["delta_lambda"], dtype=np.float64)),
            logy=True,
        )
        _plot_step_iteration_series(
            trace,
            out_path=plots["delta_u.png"],
            ylabel=r"$||\alpha \Delta U||$",
            title=f"Step {int(trace['step'])}: Newton Correction Norm",
            value_getter=lambda current: current["accepted_correction_norm"],
            logy=True,
        )
        _plot_step_iteration_series(
            trace,
            out_path=plots["delta_u_over_u.png"],
            ylabel=r"$||\alpha \Delta U|| / ||U||$",
            title=f"Step {int(trace['step'])}: Relative Newton Correction",
            value_getter=lambda current: current["accepted_relative_correction_norm"],
            logy=True,
        )
        _plot_step_xy(
            trace,
            out_path=plots["correction_norm_vs_lambda.png"],
            xlabel=r"$\lambda$",
            ylabel=r"$||\alpha \Delta U||$",
            title=f"Step {int(trace['step'])}: Correction Norm vs Lambda",
            x_getter=lambda current: current["lambda"],
            y_getter=lambda current: current["accepted_correction_norm"],
            logy=True,
        )
        _plot_step_xy(
            trace,
            out_path=plots["correction_norm_vs_criterion.png"],
            xlabel="Criterion",
            ylabel=r"$||\alpha \Delta U||$",
            title=f"Step {int(trace['step'])}: Correction Norm vs Criterion",
            x_getter=lambda current: current["criterion"],
            y_getter=lambda current: current["accepted_correction_norm"],
            logx=True,
            logy=True,
        )
        _plot_step_xy(
            trace,
            out_path=plots["lambda_vs_criterion.png"],
            xlabel="Criterion",
            ylabel=r"$\lambda$",
            title=f"Step {int(trace['step'])}: Lambda vs Criterion",
            x_getter=lambda current: current["criterion"],
            y_getter=lambda current: current["lambda"],
            logx=True,
        )
        _plot_step_xy(
            trace,
            out_path=plots["relative_increment_vs_lambda.png"],
            xlabel=r"$\lambda$",
            ylabel=r"$||\alpha \Delta U|| / ||U||$",
            title=f"Step {int(trace['step'])}: Relative Correction vs Lambda",
            x_getter=lambda current: current["lambda"],
            y_getter=lambda current: current["accepted_relative_correction_norm"],
            logy=True,
        )
        _plot_step_xy(
            trace,
            out_path=plots["relative_increment_vs_criterion.png"],
            xlabel="Criterion",
            ylabel=r"$||\alpha \Delta U|| / ||U||$",
            title=f"Step {int(trace['step'])}: Relative Correction vs Criterion",
            x_getter=lambda current: current["criterion"],
            y_getter=lambda current: current["accepted_relative_correction_norm"],
            logx=True,
            logy=True,
        )
        step_sections.append({"step": int(trace["step"]), "trace": trace, "plots": plots})

    return comparison_plots, step_sections


def _write_report(
    *,
    report_path: Path,
    artifact_root: Path,
    progress_path: Path,
    case: dict[str, Any],
    comparison_plots: dict[str, Path],
    step_sections: list[dict[str, Any]],
) -> None:
    lines: list[str] = [
        f"# {args.report_title}",
        "",
        "This partial report is built from the live `progress.jsonl` stream of the current PMG `P4(L1)` rerun. "
        "No PMG less-precise variant has finished yet, so this report currently contains the in-progress `default` case only.",
        "",
        f"- Artifact root: `{artifact_root}`",
        f"- Progress source: `{progress_path}`",
        "- Solver family: `P4(L1)`, `secant`, `pc_backend = \"pmg_shell\"`",
        "- Requested comparison order: default, 100x less precision, relative correction `1e-2`, `|Δlambda| < 1e-2`, `|Δlambda| < 1e-3`, `|Δlambda| < 1e-3` + initial-segment step-length cap",
        f"- Last progress event: `{case['timestamp']}`",
        "",
        "## Live Status",
        "",
        "| Case | Status | Runtime so far [s] | Accepted states | Accepted continuation steps | Last accepted lambda | Last accepted omega | Current target step | Current lambda | Current omega | Current relres | Current `ΔU/U` |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| "
        + " | ".join(
            [
                case["label"],
                str(case["status"]),
                _format_float(case["runtime_seconds"], 3),
                str(case["accepted_states"]),
                str(case["continuation_steps"]),
                _format_float(case["final_lambda"]),
                _format_float(case["final_omega"], 1),
                str(case["current_target_step"]),
                _format_float(case["current_lambda"]),
                _format_float(case["current_omega"], 1),
                _format_sci(case["current_relres"]),
                _format_sci(case["current_relcorr"]),
            ]
        )
        + " |",
        "",
        "## Accepted-Step Summary",
        "",
        "| Case | Residual tol | Stop criterion | Stop tol | Init Newton | Accepted continuation Newton | Init linear | Accepted continuation linear | Linear / Newton |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| "
        + " | ".join(
            [
                case["label"],
                f"`{case['tol']:.1e}`",
                "relative residual",
                f"`{case['stopping_tol']:.1e}`",
                str(case["init_newton_total"]),
                str(case["continuation_newton_total"]),
                str(case["init_linear_total"]),
                str(case["continuation_linear_total"]),
                _format_float(case["total_linear_per_newton"], 3),
            ]
        )
        + " |",
        "",
        "### Accepted-Step Lambda",
        "",
        "| Step | Lambda |",
        "| --- | ---: |",
    ]
    for step, value in zip(case["step_indices"], case["step_lambda"], strict=True):
        lines.append(f"| {int(step)} | {_format_float(value)} |")
    lines.extend(
        [
            "",
            "### Accepted-Step Omega",
            "",
            "| Step | Omega |",
            "| --- | ---: |",
        ]
    )
    for step, value in zip(case["step_indices"], case["step_omega"], strict=True):
        lines.append(f"| {int(step)} | {_format_float(value, 1)} |")
    lines.extend(
        [
            "",
            "### Accepted-Step Newton Iterations",
            "",
            "| Step | Newton | Linear | Step wall [s] | Final relres | Final `ΔU/U` |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for step, newton, linear, wall, relres, relcorr in zip(
        case["step_indices"],
        case["step_newton_iterations"],
        case["step_linear_iterations"],
        case["step_wall_time"],
        case["step_relres"],
        case["step_relcorr"],
        strict=True,
    ):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(step)),
                    str(int(newton)),
                    str(int(linear)),
                    _format_float(wall, 3),
                    _format_sci(relres),
                    _format_sci(relcorr),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Comparison Plots", ""])
    for filename, title in [
        ("lambda_omega_overlay.png", "Lambda vs omega"),
        ("lambda_vs_state.png", "Lambda by accepted state"),
        ("omega_vs_state.png", "Omega by accepted state"),
        ("runtime_by_case.png", "Runtime by case"),
        ("timing_breakdown_stacked.png", "Timing breakdown"),
        ("step_newton_iterations.png", "Newton iterations per step"),
        ("step_linear_iterations.png", "Linear iterations per step"),
        ("step_linear_per_newton.png", "Linear per Newton"),
        ("step_wall_time.png", "Wall time per step"),
        ("step_relres_end.png", "Final relative residual per step"),
        ("step_relcorr_end.png", "Final relative correction per step"),
    ]:
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}]({_rel(comparison_plots[filename], report_path)})")
        lines.append("")

    lines.extend(
        [
            "## Per-Run Plot Availability",
            "",
            "The usual final per-run images are not available yet because the PMG rerun has not finished writing `petsc_run.npz` and the final plot bundle.",
            "",
            f"- Available now: [Continuation curve]({_rel(comparison_plots['lambda_omega_overlay.png'], report_path)})",
            "- Not available until completion: `petsc_displacements_3D.png`, `petsc_deviatoric_strain_3D.png`, `petsc_step_displacement.png`",
            "",
            "## Newton Solves",
            "",
            "Accepted steps below use the successful Newton solve that produced the accepted continuation step. The last section is the current live Newton trace of the in-progress step.",
            "",
        ]
    )

    for section in step_sections:
        trace = dict(section["trace"])
        plots = dict(section["plots"])
        heading = (
            f"### Continuation Step {int(trace['step'])} (In Progress)"
            if not bool(trace["accepted"])
            else f"### Accepted Continuation Step {int(trace['step'])}"
        )
        lines.append(heading)
        lines.append("")
        lines.append("| Case | Attempt in step | Newton iterations | Step wall [s] | Lambda | Omega | Relres | `ΔU/U` | Status |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        lines.append(
            "| "
            + " | ".join(
                [
                    trace["label"],
                    str(trace["attempt_in_step"]),
                    str(trace["newton_iterations"]),
                    _format_float(trace["step_wall_time"], 3),
                    _format_float(trace["final_lambda"]),
                    _format_float(trace["final_omega"], 1),
                    _format_sci(trace["final_relres"]),
                    _format_sci(trace["final_relcorr"]),
                    str(trace["status"]),
                ]
            )
            + " |"
        )
        lines.append("")
        lines.extend(
            _build_step_plot_grid(
                report_path=report_path,
                columns=2,
                cells=[
                    ("Criterion", plots["criterion.png"]),
                    ("Lambda", plots["lambda.png"]),
                    ("Abs Delta Lambda", plots["delta_lambda.png"]),
                    ("Delta U", plots["delta_u.png"]),
                    ("Delta U / U", plots["delta_u_over_u.png"]),
                ],
            )
        )
        lines.extend(
            _build_step_plot_grid(
                report_path=report_path,
                columns=3,
                cells=[
                    ("Delta U vs Lambda", plots["correction_norm_vs_lambda.png"]),
                    ("Delta U vs Criterion", plots["correction_norm_vs_criterion.png"]),
                    ("Lambda vs Criterion", plots["lambda_vs_criterion.png"]),
                    ("Delta U / U vs Lambda", plots["relative_increment_vs_lambda.png"]),
                    ("Delta U / U vs Criterion", plots["relative_increment_vs_criterion.png"]),
                ],
            )
        )

    lines.extend(
        [
            "## Artifacts",
            "",
            f"- Live default run: `{artifact_root / 'runs' / 'default'}`",
            f"- Live progress file: `{progress_path}`",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global args
    args = _parse_args()
    artifact_root = args.artifact_root.resolve()
    report_path = args.report_path.resolve()
    progress_path = artifact_root / "runs" / "default" / "data" / "progress.jsonl"
    if not progress_path.exists():
        raise FileNotFoundError(f"Missing progress file: {progress_path}")

    events = _load_events(progress_path)
    if not events:
        raise RuntimeError(f"No readable progress events found in {progress_path}")

    case = _build_live_case(events)
    traces = _extract_traces(events)
    comparison_plots, step_sections = _build_plots(case, traces, artifact_root=artifact_root)
    _write_report(
        report_path=report_path,
        artifact_root=artifact_root,
        progress_path=progress_path,
        case=case,
        comparison_plots=comparison_plots,
        step_sections=step_sections,
    )
    print(f"[done] {report_path}")


if __name__ == "__main__":
    main()
