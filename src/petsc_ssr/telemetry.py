from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


HUGE = 1.7976931348623157e308


def _flag(value: bool) -> str:
    return str(value).lower()


@dataclass(slots=True)
class NewtonStats:
    converged: bool = False
    final_rel: float = HUGE
    final_rel_correction: float = HUGE
    newton_its: int = 0
    total_linear_its: int = 0
    line_search_its: int = 0
    assembly_time: float = 0.0
    solve_time: float = 0.0
    wall_time: float = 0.0
    phase_times: dict[str, float] = field(default_factory=dict)
    phase_calls: dict[str, int] = field(default_factory=dict)

    def add_step(self, result: dict[str, Any]) -> None:
        self.assembly_time += float(result["assembly_time"])
        self.solve_time += float(result["solve_time"])
        self.total_linear_its += int(result["linear_its"])
        self.line_search_its += int(result["line_search_its"])
        self.newton_its += int(result["newton_its"])
        self.final_rel = float(result["rel_residual"])
        self.final_rel_correction = float(result["rel_correction"])

    def record_phase(self, name: str, seconds: float = 0.0, calls: int = 1) -> None:
        self.phase_times[name] = self.phase_times.get(name, 0.0) + seconds
        self.phase_calls[name] = self.phase_calls.get(name, 0) + calls


@dataclass(slots=True)
class ContinuationTotals:
    newton_its: int = 0
    linear_its: int = 0
    line_search_its: int = 0
    final_rel: float = HUGE
    final_rel_correction: float = HUGE

    def add(self, stats: NewtonStats) -> None:
        self.newton_its += stats.newton_its
        self.linear_its += stats.total_linear_its
        self.line_search_its += stats.line_search_its
        self.final_rel = stats.final_rel
        self.final_rel_correction = stats.final_rel_correction


@dataclass(slots=True)
class ContinuationCurve:
    rows: list[dict[str, str]]
    summary: dict[str, Any]
    csv_path: Path

    @property
    def accepted_steps(self) -> int:
        return len(self.rows)

    @property
    def lambda_last(self) -> float:
        if self.summary.get("lambda_last") is not None:
            return float(self.summary["lambda_last"])
        return float(self.rows[-1]["lambda"]) if self.rows else 0.0

    @property
    def omega_last(self) -> float:
        if self.summary.get("omega_last") is not None:
            return float(self.summary["omega_last"])
        return float(self.rows[-1]["omega"]) if self.rows else 0.0

    def write_csv(self, path: str | Path) -> None:
        write_curve_csv(Path(path), self.rows)

    def linear_iterations_per_step(self) -> list[tuple[int, int]]:
        return [(int(row["step"]), int(row["linear_iterations"])) for row in self.rows]


class RankReporter:
    def __init__(self, rank: int) -> None:
        self.rank = rank

    @property
    def enabled(self) -> bool:
        return self.rank == 0

    def emit(self, message: str) -> None:
        if self.enabled:
            print(message, flush=True)

    def init_attempt(self, phase: str, lambda_value: float, d_lambda: float, basis_snapshot: int) -> None:
        self.emit(
            f"PY_SSR_INIT phase={phase} lambda={lambda_value:.8e} "
            f"d_lambda={d_lambda:.8e} basis_snapshot={basis_snapshot}"
        )

    def tiny_omega_shift(self, omega_prev: float, omega_cur: float) -> None:
        self.emit(
            "PY_SSR_INIT phase=advance tiny_omega_increment=true "
            f"omega_prev={omega_prev:.8e} omega_cur={omega_cur:.8e} shifting_seed=true"
        )

    def fixed_newton_step(self, lambda_value: float, iteration: int, result: dict[str, Any]) -> None:
        self.emit(
            "PY_INIT_NEWTON "
            f"lambda={lambda_value:.8e} it={iteration} rel_res={result['rel_residual']:.6e} "
            f"rel_corr={result['rel_correction']:.6e} alpha={result['alpha']:.6e} "
            f"linear_its={result['linear_its']} stop={_flag(bool(result['stop']))}"
        )

    def fixed_newton_summary(self, lambda_value: float, stats: NewtonStats) -> None:
        self.emit(
            "PY_INIT_NEWTON_SUMMARY "
            f"lambda={lambda_value:.8e} converged={_flag(stats.converged)} "
            f"final_rel={stats.final_rel:.6e} final_rel_correction={stats.final_rel_correction:.6e} "
            f"newton_its={stats.newton_its} linear_its={stats.total_linear_its} "
            f"wall_time={stats.wall_time:.6g}"
        )

    def indirect_newton_step(self, omega_target: float, iteration: int, result: dict[str, Any]) -> None:
        self.emit(
            "PY_NEWTON "
            f"omega={omega_target:.8e} it={iteration} lambda={result['lambda_out']:.8e} "
            f"rel_res={result['rel_residual']:.6e} rel_corr={result['rel_correction']:.6e} "
            f"alpha={result['alpha']:.6e} d_lambda={result['delta_lambda']:.6e} "
            f"linear_w={result['linear_its_w']} linear_v={result['linear_its_v']} "
            f"linear_its={result['linear_its']} stop={_flag(bool(result['stop']))}"
        )

    def indirect_newton_summary(self, omega_target: float, lambda_value: float, stats: NewtonStats) -> None:
        self.emit(
            "PY_NEWTON_SUMMARY "
            f"omega={omega_target:.8e} converged={_flag(stats.converged)} lambda={lambda_value:.8e} "
            f"final_rel={stats.final_rel:.6e} final_rel_correction={stats.final_rel_correction:.6e} "
            f"newton_its={stats.newton_its} linear_its={stats.total_linear_its} "
            f"line_search_its={stats.line_search_its} wall_time={stats.wall_time:.6g}"
        )

    def limit_load_newton_step(self, omega_target: float, iteration: int, result: dict[str, Any]) -> None:
        self.emit(
            "PY_LL_NEWTON "
            f"omega={omega_target:.8e} it={iteration} t={result['lambda_out']:.8e} "
            f"rel_res={result['rel_residual']:.6e} rel_corr={result['rel_correction']:.6e} "
            f"alpha={result['alpha']:.6e} d_t={result['delta_lambda']:.6e} "
            f"linear_w={result['linear_its_w']} linear_v={result['linear_its_v']} "
            f"linear_its={result['linear_its']} stop={_flag(bool(result['stop']))}"
        )

    def limit_load_newton_summary(self, omega_target: float, load_t: float, stats: NewtonStats) -> None:
        self.emit(
            "PY_LL_NEWTON_SUMMARY "
            f"omega={omega_target:.8e} converged={_flag(stats.converged)} t={load_t:.8e} "
            f"final_rel={stats.final_rel:.6e} final_rel_correction={stats.final_rel_correction:.6e} "
            f"newton_its={stats.newton_its} linear_its={stats.total_linear_its} "
            f"line_search_its={stats.line_search_its} wall_time={stats.wall_time:.6g}"
        )

    def accepted_init_step(self, step: int, omega: float, lambda_value: float, stats: NewtonStats) -> None:
        self.emit(
            f"PY_SSR_STEP step={step} phase=init omega={omega:.8e} lambda={lambda_value:.8e} "
            f"accepted=true newton_its={stats.newton_its} linear_its={stats.total_linear_its}"
        )

    def accepted_advance_step(self, step: int, omega: float, lambda_value: float, d_omega: float, d_lambda: float, stats: NewtonStats) -> None:
        self.emit(
            f"PY_SSR_STEP step={step} phase=init omega={omega:.8e} lambda={lambda_value:.8e} "
            f"d_omega={d_omega:.8e} d_lambda={d_lambda:.8e} accepted=true "
            f"newton_its={stats.newton_its} linear_its={stats.total_linear_its}"
        )

    def continuation_attempt(self, step: int, target: float, d_omega: float, lambda_predict: float, basis_snapshot: int) -> None:
        self.emit(
            f"PY_SSR_ATTEMPT step={step} target_omega={target:.8e} d_omega={d_omega:.8e} "
            f"lambda_predict={lambda_predict:.8e} basis_snapshot={basis_snapshot}"
        )

    def rejected_attempt(self, step: int, reductions: int, next_d_omega: float) -> None:
        self.emit(
            f"PY_SSR_ATTEMPT step={step} accepted=false reductions={reductions} "
            f"next_d_omega={next_d_omega:.8e}"
        )

    def accepted_continuation_step(self, step: int, omega: float, lambda_value: float, d_omega: float, d_lambda: float, stats: NewtonStats) -> None:
        self.emit(
            f"PY_SSR_STEP step={step} phase=cont omega={omega:.8e} lambda={lambda_value:.8e} "
            f"d_omega={d_omega:.8e} d_lambda={d_lambda:.8e} accepted=true "
            f"newton_its={stats.newton_its} linear_its={stats.total_linear_its} "
            f"rel_res={stats.final_rel:.6e}"
        )

    def result(self, omega: float, lambda_value: float, accepted_steps: int, totals: ContinuationTotals, wall_time: float, stop_reason: str, curve_csv: Path) -> None:
        self.emit(
            "PY_SSR_RESULT "
            f"omega_last={omega:.8e} lambda_last={lambda_value:.8e} accepted_steps={accepted_steps} "
            f"total_newton_iterations={totals.newton_its} total_linear_iterations={totals.linear_its} "
            f"total_line_search_iterations={totals.line_search_its} wall_time={wall_time:.6g} "
            f"stop_reason={stop_reason} curve_csv={curve_csv}"
        )


class CurveRecorder:
    def __init__(self, engine: Any, csv_path: Path) -> None:
        self.engine = engine
        self.csv_path = csv_path
        self.rows: list[dict[str, str]] = []

    def add(
        self,
        *,
        step: int,
        phase: str,
        omega: float,
        lambda_value: float,
        d_omega: float,
        d_lambda: float,
        slot: int,
        attempts: int,
        stats: NewtonStats,
        reason: str = "accepted",
    ) -> None:
        self.rows.append(
            {
                "step": str(step),
                "phase": phase,
                "omega": f"{omega:.16e}",
                "lambda": f"{lambda_value:.16e}",
                "d_omega": f"{d_omega:.16e}",
                "d_lambda": f"{d_lambda:.16e}",
                "u_max": f"{self.engine.displacement_max(slot):.16e}",
                "attempts": str(attempts),
                "newton_iterations": str(stats.newton_its),
                "linear_iterations": str(stats.total_linear_its),
                "line_search_iterations": str(stats.line_search_its),
                "rel_residual": f"{stats.final_rel:.16e}",
                "rel_correction": f"{stats.final_rel_correction:.16e}",
                "step_wall_time": f"{stats.wall_time:.16e}",
                "stop_reason": reason,
            }
        )

    def write(self) -> None:
        write_curve_csv(self.csv_path, self.rows)


def write_curve_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
