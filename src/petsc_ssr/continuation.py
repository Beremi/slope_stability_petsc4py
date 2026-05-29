from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

from .context import EngineRunResult, SsrContext
from .newton import solve_fixed_lambda_newton, solve_indirect_newton
from .operations import EngineOps, Slot
from .telemetry import ContinuationCurve, ContinuationTotals, CurveRecorder, NewtonStats, RankReporter, write_summary_json


@dataclass(slots=True)
class ContinuationState:
    step: int = 0
    attempts: int = 0
    omega_reductions: int = 0
    stop_reason: str = "running"
    lambda_prev: float = 0.0
    lambda_cur: float = 0.0
    omega_prev: float = 0.0
    omega_cur: float = 0.0
    d_omega: float = 0.0
    seed_stats: NewtonStats | None = None
    advance_stats: NewtonStats | None = None

    @property
    def running(self) -> bool:
        return self.stop_reason == "running"


def _solve_seed_point(ctx: SsrContext, ops: EngineOps, reporter: RankReporter, state: ContinuationState) -> None:
    opts = ctx.options
    U_old = ops.displacement_vectors.U_old
    lambda_seed = opts.lambda_init
    d_lambda = opts.d_lambda_init

    while True:
        basis_snapshot = ops.basis_cols()
        U_old.zero()
        reporter.init_attempt("seed", lambda_seed, d_lambda, basis_snapshot)
        stats = solve_fixed_lambda_newton(
            ops,
            opts,
            reporter,
            U_it=U_old,
            lambda_value=lambda_seed,
            criterion=opts.init_newton_stopping_criterion,
            stop_tol=opts.init_newton_stopping_tol,
        )
        state.attempts += 1
        if stats.converged:
            state.lambda_prev = lambda_seed
            state.omega_prev = U_old.omega()
            U_old.append_to_deflation_basis("SSR init seed")
            state.seed_stats = stats
            return

        ops.truncate_basis(basis_snapshot)
        lambda_seed *= 0.5
        d_lambda *= 0.5
        if d_lambda < opts.d_lambda_min:
            raise RuntimeError(f"SSR initialization failed before d_lambda_min {opts.d_lambda_min:.6e}")


def _solve_second_seed_point(ctx: SsrContext, ops: EngineOps, reporter: RankReporter, state: ContinuationState, init_basis_base: int) -> None:
    opts = ctx.options
    vectors = ops.displacement_vectors
    U_old = vectors.U_old
    U = vectors.U
    d_lambda = opts.d_lambda_init

    while True:
        basis_snapshot = ops.basis_cols()
        lambda_candidate = state.lambda_prev + d_lambda
        U.copy_from(U_old)
        reporter.init_attempt("advance", lambda_candidate, d_lambda, basis_snapshot)
        stats = solve_fixed_lambda_newton(
            ops,
            opts,
            reporter,
            U_it=U,
            lambda_value=lambda_candidate,
            criterion=opts.init_newton_stopping_criterion,
            stop_tol=opts.init_newton_stopping_tol,
        )
        state.attempts += 1

        if not stats.converged:
            ops.truncate_basis(basis_snapshot)
            d_lambda *= 0.5
            if d_lambda < opts.d_lambda_min:
                raise RuntimeError(f"SSR second initialization point failed before d_lambda_min {opts.d_lambda_min:.6e}")
            continue

        omega_candidate = U.omega()
        if (omega_candidate - state.omega_prev) / max(1.0, abs(state.omega_prev)) < 1.0e-5:
            reporter.tiny_omega_shift(state.omega_prev, omega_candidate)
            U_old.copy_from(U)
            state.lambda_prev = lambda_candidate
            state.omega_prev = omega_candidate
            continue

        ops.truncate_basis(init_basis_base)
        U_old.append_to_deflation_basis("SSR accepted init previous")
        state.lambda_cur = lambda_candidate
        state.omega_cur = omega_candidate
        state.d_omega = state.omega_cur - state.omega_prev
        state.advance_stats = stats
        if state.d_omega <= 0.0:
            raise RuntimeError(f"SSR initialization did not produce increasing omega: {state.omega_prev:.8e} -> {state.omega_cur:.8e}")
        return


def _record_seed_points(
    reporter: RankReporter,
    recorder: CurveRecorder,
    totals: ContinuationTotals,
    state: ContinuationState,
) -> None:
    if state.seed_stats is None or state.advance_stats is None:
        raise RuntimeError("SSR initialization was not completed before recording seed points")
    seed_stats = state.seed_stats
    recorder.add(
        step=state.step,
        phase="init",
        omega=state.omega_prev,
        lambda_value=state.lambda_prev,
        d_omega=0.0,
        d_lambda=0.0,
        slot=Slot.PREV,
        attempts=state.attempts,
        stats=seed_stats,
    )
    reporter.accepted_init_step(state.step, state.omega_prev, state.lambda_prev, seed_stats)
    totals.add(seed_stats)
    state.step += 1

    advance_stats = state.advance_stats
    recorder.add(
        step=state.step,
        phase="init",
        omega=state.omega_cur,
        lambda_value=state.lambda_cur,
        d_omega=state.d_omega,
        d_lambda=state.lambda_cur - state.lambda_prev,
        slot=Slot.CUR,
        attempts=state.attempts,
        stats=advance_stats,
    )
    reporter.accepted_advance_step(state.step, state.omega_cur, state.lambda_cur, state.d_omega, state.lambda_cur - state.lambda_prev, advance_stats)
    totals.add(advance_stats)
    state.step += 1


def _update_stop_reason(ctx: SsrContext, state: ContinuationState, *, check_slope: bool = True) -> None:
    opts = ctx.options
    if state.omega_cur >= opts.omega_max:
        state.stop_reason = "omega_max"
    elif state.step >= opts.continuation_step_max:
        state.stop_reason = "step_max"
    elif check_slope and opts.d_lambda_diff_scaled_min > 0.0:
        slope_scaled = abs((state.lambda_cur - state.lambda_prev) / max(state.omega_cur - state.omega_prev, 1.0e-30)) * max(state.omega_cur, 1.0)
        if slope_scaled <= opts.d_lambda_diff_scaled_min:
            state.stop_reason = "d_lambda_diff_scaled_min"


def _summary(ctx: SsrContext, ops: EngineOps, state: ContinuationState, totals: ContinuationTotals, continuation_wall: float, curve_csv: Path) -> dict[str, object]:
    info = ops.info()
    return {
        "engine": "petsc_ssr_python_loop",
        "variant": ctx.options.pc_variant,
        "ranks": info["ranks"],
        "partitioner": ctx.options.partitioner,
        "global_dofs": info["global_dofs"],
        "accepted_steps": state.step,
        "total_newton_its": totals.newton_its,
        "total_linear_its": totals.linear_its,
        "total_line_search_its": totals.line_search_its,
        "elastic_assembly_time": info["elastic_assembly_time"],
        "continuation_wall_time": continuation_wall,
        "wall_time": continuation_wall + info["create_time"],
        "omega_last": state.omega_cur,
        "lambda_last": state.lambda_cur,
        "final_rel": totals.final_rel,
        "final_rel_correction": totals.final_rel_correction,
        "stop_reason": state.stop_reason,
        "deflation": ctx.options.linear.deflation,
        "deflation_solver": ctx.options.linear.deflation_solver,
        "deflation_basis_cols": info["basis_cols"],
        "deflation_orthogonalization_time": info["deflation_orthogonalization_time"],
        "deflation_coarse_initial_time": info["deflation_coarse_initial_time"],
        "deflation_coarse_initial_calls": info["deflation_coarse_initial_calls"],
        "deflation_pc_apply_time": info["deflation_pc_apply_time"],
        "deflation_projector_time": info["deflation_projector_time"],
        "deflation_projected_pc_calls": info["deflation_projected_pc_calls"],
        "curve_csv": str(curve_csv),
    }


def run_indirect_ssr(ctx: SsrContext, options: object | None = None) -> ContinuationCurve:
    if options is not None:
        ctx.options = options  # type: ignore[assignment]

    ops = ctx.debug_engine_ops()
    reporter = RankReporter(ctx.rank)
    data_dir = ctx.output_dir / "data"
    curve_csv = data_dir / "continuation_curve.csv"
    summary_json = data_dir / "summary.json"
    recorder = CurveRecorder(ops, curve_csv)
    totals = ContinuationTotals()
    state = ContinuationState()
    vectors = ops.displacement_vectors
    U_old = vectors.U_old
    U = vectors.U
    U_ini = vectors.U_ini
    t_start = perf_counter()

    init_basis_base = ops.basis_cols()
    _solve_seed_point(ctx, ops, reporter, state)
    _solve_second_seed_point(ctx, ops, reporter, state, init_basis_base)
    _record_seed_points(reporter, recorder, totals, state)
    _update_stop_reason(ctx, state, check_slope=False)

    while state.running and state.step < ctx.options.continuation_step_max and state.omega_cur < ctx.options.omega_max:
        basis_snapshot = ops.basis_cols()

        omega_it = min(state.omega_cur + state.d_omega, ctx.options.omega_max)
        d_omega = omega_it - state.omega_cur
        secant_alpha = d_omega / (state.omega_cur - state.omega_prev)

        U_ini.secant_predict_from(U_old, U, alpha=secant_alpha, work=vectors.work)
        lambda_it = state.lambda_cur + secant_alpha * (state.lambda_cur - state.lambda_prev)

        reporter.continuation_attempt(state.step, omega_it, d_omega, lambda_it, basis_snapshot)
        lambda_it, stats = solve_indirect_newton(ops, ctx.options, reporter, U_it=U_ini, lambda_start=lambda_it, omega=omega_it)
        state.attempts += 1

        if not stats.converged:
            ops.truncate_basis(basis_snapshot)
            state.d_omega *= 0.5
            state.omega_reductions += 1
            reporter.rejected_attempt(state.step, state.omega_reductions, state.d_omega)
            if state.omega_reductions >= 5:
                state.stop_reason = "omega_reduction_limit"
            continue

        ops.truncate_basis(basis_snapshot)

        old_lambda = state.lambda_cur
        old_omega = state.omega_cur
        branch_double = (lambda_it - state.lambda_cur) < 0.9 * (state.lambda_cur - state.lambda_prev)

        U_old.copy_from(U)
        U.copy_from(U_ini)
        U.append_to_deflation_basis("SSR accepted continuation state")

        state.lambda_prev = old_lambda
        state.omega_prev = old_omega
        state.lambda_cur = lambda_it
        state.omega_cur = omega_it
        state.omega_reductions = 0

        recorder.add(
            step=state.step,
            phase="cont",
            omega=state.omega_cur,
            lambda_value=state.lambda_cur,
            d_omega=state.omega_cur - old_omega,
            d_lambda=state.lambda_cur - old_lambda,
            slot=Slot.CUR,
            attempts=state.attempts,
            stats=stats,
        )
        reporter.accepted_continuation_step(state.step, state.omega_cur, state.lambda_cur, state.omega_cur - old_omega, state.lambda_cur - old_lambda, stats)
        totals.add(stats)
        state.step += 1

        if branch_double:
            state.d_omega *= 2.0
        _update_stop_reason(ctx, state)

    _update_stop_reason(ctx, state)
    continuation_wall = perf_counter() - t_start
    summary = _summary(ctx, ops, state, totals, continuation_wall, curve_csv)

    if reporter.enabled:
        recorder.write()
        write_summary_json(summary_json, summary)
    reporter.result(state.omega_cur, state.lambda_cur, len(recorder.rows), totals, continuation_wall, state.stop_reason, curve_csv)

    result = EngineRunResult(ctx.output_dir, curve_csv, summary_json, float(summary["wall_time"]), summary)
    ctx.last_result = result
    return ContinuationCurve(rows=recorder.rows if reporter.enabled else [], summary=summary, csv_path=curve_csv)


def table_rows(curves: Iterable[ContinuationCurve]) -> list[dict[str, str]]:
    return [
        {
            "csv": str(curve.csv_path),
            "steps": str(curve.accepted_steps),
            "omega_last": f"{curve.omega_last:.8e}",
            "lambda_last": f"{curve.lambda_last:.8e}",
            "linear_iterations": str(curve.summary.get("total_linear_its", "")),
            "newton_iterations": str(curve.summary.get("total_newton_its", "")),
        }
        for curve in curves
    ]
