"""Progress logging with compact real-time console rendering."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import TextIO

import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_num(value) -> str:
    if value is None:
        return "-"
    x = float(value)
    if not np.isfinite(x):
        return str(x)
    ax = abs(x)
    if ax == 0.0:
        return "0"
    if ax >= 1.0e4 or ax < 1.0e-3:
        return f"{x:.3e}"
    if ax >= 100.0:
        return f"{x:.1f}"
    if ax >= 10.0:
        return f"{x:.2f}"
    return f"{x:.4f}"


def _fmt_time(value) -> str:
    if value is None:
        return "-"
    t = float(value)
    if not np.isfinite(t):
        return str(t)
    if t >= 100.0:
        return f"{t:.0f}s"
    if t >= 10.0:
        return f"{t:.1f}s"
    if t >= 1.0:
        return f"{t:.2f}s"
    return f"{t:.3f}s"


def _int_value(payload: dict, key: str, default: int = 0) -> int:
    value = payload.get(key, default)
    if value is None:
        return int(default)
    return int(value)


def _load_label(kind: str) -> str:
    return "t" if kind == "ll_indirect" else "lambda"


def _delta_label(kind: str) -> str:
    return "dt" if kind == "ll_indirect" else "dlam"


class _ConsoleProgressRenderer:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self._attempt_key: tuple[object, ...] | None = None

    def _write_line(self, line: str) -> None:
        self.stream.write(line + "\n")
        self.stream.flush()

    def _attempt_context_key(self, payload: dict) -> tuple[object, ...]:
        return (
            payload.get("continuation_kind"),
            payload.get("phase"),
            payload.get("target_step"),
            payload.get("attempt_in_step"),
        )

    def _render_attempt_header(self, payload: dict) -> None:
        phase = str(payload.get("phase", "continuation"))
        kind = str(payload.get("continuation_kind", ""))
        if phase == "init":
            load_label = _load_label(kind)
            self._write_line(
                f"[init newton] {load_label}0={_fmt_num(payload.get('lambda_before'))} "
                f"omega={_fmt_num(payload.get('omega_target'))}"
            )
            return

        target_step = _int_value(payload, "target_step", 0)
        attempt = _int_value(payload, "attempt_in_step", 1)
        load_label = _load_label(kind)
        self._write_line(
            f"[step {target_step:02d} try {attempt}] "
            f"omega={_fmt_num(payload.get('omega_target'))} "
            f"{load_label}0={_fmt_num(payload.get('lambda_before'))}"
        )

    def _render_init_complete(self, payload: dict) -> None:
        lambda_hist = list(payload.get("lambda_hist", []))
        omega_hist = list(payload.get("omega_hist", []))
        newton_hist = list(payload.get("init_newton_iterations", []))
        parts = [
            "[init]",
            f"pts={len(lambda_hist)}",
        ]
        if lambda_hist:
            parts.append(f"lambda=[{_fmt_num(lambda_hist[0])}, {_fmt_num(lambda_hist[-1])}]")
        if omega_hist:
            parts.append(f"omega=[{_fmt_num(omega_hist[0])}, {_fmt_num(omega_hist[-1])}]")
        if newton_hist:
            parts.append(f"newton={newton_hist}")
        parts.append(f"lin={_int_value(payload, 'init_linear_iterations', 0)}")
        parts.append(f"t={_fmt_time(payload.get('total_wall_time', 0.0))}")
        self._write_line(" | ".join(parts))

    def _render_newton_iteration(self, payload: dict) -> None:
        attempt_key = self._attempt_context_key(payload)
        if attempt_key != self._attempt_key:
            self._render_attempt_header(payload)
            self._attempt_key = attempt_key

        kind = str(payload.get("continuation_kind", ""))
        load_label = _load_label(kind)
        delta_label = _delta_label(kind)
        iteration = _int_value(payload, "iteration", 0)
        status = str(payload.get("status", "iterate"))
        tag = f"N{iteration:02d}"
        if status == "converged":
            tag = f"{tag} conv"

        parts = [
            f"  {tag}",
            f"rel={_fmt_num(payload.get('rel_residual'))}",
            f"crit={_fmt_num(payload.get('criterion'))}",
        ]

        alpha = payload.get("alpha")
        if alpha is not None:
            parts.append(f"alpha={_fmt_num(alpha)}")

        parts.append(f"r={_fmt_num(payload.get('r'))}")

        lambda_value = payload.get("lambda_value")
        if lambda_value is not None:
            parts.append(f"{load_label}={_fmt_num(lambda_value)}")

        accepted_delta = payload.get("accepted_delta_lambda")
        if accepted_delta is not None:
            parts.append(f"{delta_label}={_fmt_num(accepted_delta)}")

        parts.append(f"lin={_int_value(payload, 'linear_iterations', 0)}")
        parts.append(f"solve={_fmt_time(payload.get('linear_solve_time', 0.0))}")

        pc_time = float(payload.get("linear_preconditioner_time", 0.0) or 0.0)
        if pc_time > 0.0:
            parts.append(f"pc={_fmt_time(pc_time)}")

        orth_time = float(payload.get("linear_orthogonalization_time", 0.0) or 0.0)
        if orth_time > 0.0:
            parts.append(f"orth={_fmt_time(orth_time)}")

        parts.append(f"iter={_fmt_time(payload.get('iteration_wall_time', 0.0))}")
        self._write_line(" | ".join(parts))

    def _render_attempt_complete(self, payload: dict) -> None:
        if bool(payload.get("success", False)):
            self._attempt_key = None
            return

        parts = [
            "  reject",
            f"newton={_int_value(payload, 'newton_iterations', 0)}",
            f"rel={_fmt_num(payload.get('newton_relres_end'))}",
            f"lin={_int_value(payload, 'linear_iterations', 0)}",
            f"t={_fmt_time(payload.get('attempt_wall_time', 0.0))}",
        ]
        self._write_line(" | ".join(parts))
        self._attempt_key = None

    def _render_step_accepted(self, payload: dict) -> None:
        kind = str(payload.get("continuation_kind", ""))
        load_label = _load_label(kind)
        delta_label = _delta_label(kind)
        step = _int_value(payload, "accepted_step", _int_value(payload, "accepted_steps", 0))
        attempts = payload.get("step_attempt_count", payload.get("attempt_count"))
        newton_total = payload.get("step_newton_iterations_total", payload.get("newton_iterations_total"))
        rel = payload.get("step_newton_relres_end", payload.get("newton_relres_end"))
        lin = payload.get("step_linear_iterations", payload.get("linear_iterations"))
        wall = payload.get("step_wall_time", 0.0)
        parts = [
            f"[step {step:02d} ok]",
            f"{load_label}={_fmt_num(payload.get('lambda_value'))}",
            f"{delta_label}={_fmt_num(payload.get('d_lambda'))}",
        ]
        if payload.get("d_lambda_diff_scaled") is not None:
            parts.append(f"scaled={_fmt_num(payload.get('d_lambda_diff_scaled'))}")
        parts.append(f"omega={_fmt_num(payload.get('omega_value'))}")
        if payload.get("d_omega") is not None:
            parts.append(f"domega={_fmt_num(payload.get('d_omega'))}")
        if payload.get("u_max") is not None:
            parts.append(f"umax={_fmt_num(payload.get('u_max'))}")
        if attempts is not None:
            parts.append(f"attempts={int(attempts)}")
        if newton_total is not None:
            parts.append(f"newton={int(newton_total)}")
        if rel is not None:
            parts.append(f"rel={_fmt_num(rel)}")
        if lin is not None:
            parts.append(f"lin={int(lin)}")
        parts.append(f"t={_fmt_time(wall)}")
        self._write_line(" | ".join(parts))
        self._attempt_key = None

    def _render_finished(self, payload: dict) -> None:
        kind = str(payload.get("continuation_kind", ""))
        load_label = _load_label(kind)
        parts = [
            "[done]",
            f"steps={_int_value(payload, 'accepted_steps', 0)}",
            f"{load_label}={_fmt_num(payload.get('lambda_last'))}",
            f"omega={_fmt_num(payload.get('omega_last'))}",
            f"t={_fmt_time(payload.get('total_wall_time', 0.0))}",
        ]
        stop_reason = payload.get("stop_reason")
        if stop_reason:
            parts.append(f"reason={stop_reason}")
        self._write_line(" | ".join(parts))
        self._attempt_key = None

    def render(self, payload: dict) -> None:
        event = str(payload.get("event", ""))
        if event == "init_complete":
            self._render_init_complete(payload)
            return
        if event == "newton_iteration":
            self._render_newton_iteration(payload)
            return
        if event == "attempt_complete":
            self._render_attempt_complete(payload)
            return
        if event == "step_accepted":
            self._render_step_accepted(payload)
            return
        if event == "finished":
            self._render_finished(payload)


def make_progress_logger(progress_dir: Path, *, console: TextIO | None = None):
    progress_jsonl = progress_dir / "progress.jsonl"
    progress_latest = progress_dir / "progress_latest.json"
    renderer = _ConsoleProgressRenderer(console if console is not None else sys.stdout)

    def _write(event: dict) -> None:
        payload = {"timestamp": _utc_now(), **event}
        with progress_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
            handle.flush()
        progress_latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        renderer.render(payload)

    return _write
