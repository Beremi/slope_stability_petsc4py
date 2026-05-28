from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .context import EngineRunResult, SsrContext
from .newton import solve_limit_load_newton
from .operations import EngineOps, Slot
from .telemetry import ContinuationCurve, ContinuationTotals, CurveRecorder, NewtonStats, RankReporter, write_summary_json


@dataclass(slots=True)
class LimitLoadState:
    step: int = 0
    attempts: int = 0
    omega_reductions: int = 0
    stop_reason: str = "running"
    omega_prev: float = 0.0
    omega_cur: float = 0.0
    load_t: float = 0.0
    d_omega: float = 0.0
    last_d_t: float = 0.0

    @property
    def running(self) -> bool:
        return self.stop_reason == "running"


def _summary(ctx: SsrContext, ops: EngineOps, state: LimitLoadState, totals: ContinuationTotals, continuation_wall: float, curve_csv: Path) -> dict[str, object]:
    info = ops.info()
    return {
        "engine": "petsc_ssr_python_loop",
        "analysis": "ll",
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
        "lambda_last": state.load_t,
        "final_rel": totals.final_rel,
        "final_rel_correction": totals.final_rel_correction,
        "stop_reason": state.stop_reason,
        "lambda_ell": ctx.options.lambda_ell,
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
        "solution_binary": str(ctx.output_dir / "data" / "final_displacement.petscbin"),
        "solution_points_csv": str(ctx.output_dir / "data" / "final_displacement_points.csv"),
        "solution_vtk": str(ctx.output_dir / "exports" / "final_solution.vtu"),
    }


def run_limit_load_continuation(ctx: SsrContext) -> ContinuationCurve:
    opts = ctx.options
    ops = EngineOps(ctx.create_engine())
    reporter = RankReporter(ctx.rank)
    data_dir = ctx.output_dir / "data"
    curve_csv = data_dir / "continuation_curve.csv"
    summary_json = data_dir / "summary.json"
    recorder = CurveRecorder(ops, curve_csv)
    totals = ContinuationTotals()
    state = LimitLoadState()
    vectors = ops.displacement_vectors
    U_old = vectors.U_old
    U = vectors.U
    U_ini = vectors.U_ini
    work = vectors.work
    t_start = perf_counter()

    elastic = ops.solve_elastic_initial_guess(Slot.CUR, opts.d_omega_ini_scale)
    ops.truncate_basis(0)
    seed_stats = NewtonStats(converged=True, total_linear_its=int(elastic["linear_its"]), solve_time=float(elastic["solve_time"]))
    state.d_omega = U.omega()
    if state.d_omega <= 0.0:
        raise RuntimeError(f"Limit-load elastic predictor produced non-positive omega increment {state.d_omega:.8e}")
    U_old.copy_from(U)

    recorder.add(
        step=state.step,
        phase="init",
        omega=0.0,
        lambda_value=0.0,
        d_omega=0.0,
        d_lambda=0.0,
        slot=Slot.CUR,
        attempts=0,
        stats=seed_stats,
    )
    totals.add(seed_stats)
    state.step += 1

    while state.running and state.step < opts.continuation_step_max and state.omega_cur < opts.omega_max:
        basis_snapshot = ops.basis_cols()
        omega_target = min(state.omega_cur + state.d_omega, opts.omega_max)
        if state.step <= 1:
            U_ini.copy_from(U)
        else:
            denom = max(state.omega_cur - state.omega_prev, 1.0e-30)
            alpha = (omega_target - state.omega_cur) / denom
            U_ini.secant_predict_from(U_old, U, alpha=alpha, work=work)

        reporter.continuation_attempt(state.step, omega_target, omega_target - state.omega_cur, state.load_t, basis_snapshot)
        load_t, stats = solve_limit_load_newton(ops, opts, reporter, U_it=U_ini, load_start=state.load_t, omega=omega_target)
        state.attempts += 1
        ops.truncate_basis(basis_snapshot)

        if not stats.converged:
            state.d_omega *= 0.5
            state.omega_reductions += 1
            reporter.rejected_attempt(state.step, state.omega_reductions, state.d_omega)
            if state.omega_reductions >= 5:
                state.stop_reason = "omega_reduction_limit"
            continue

        old_omega = state.omega_cur
        old_t = state.load_t
        U_old.copy_from(U)
        U.copy_from(U_ini)
        U.append_to_deflation_basis("LL accepted continuation state")

        state.omega_prev = old_omega
        state.omega_cur = omega_target
        state.load_t = load_t
        state.omega_reductions = 0

        recorder.add(
            step=state.step,
            phase="cont",
            omega=state.omega_cur,
            lambda_value=state.load_t,
            d_omega=state.omega_cur - old_omega,
            d_lambda=state.load_t - old_t,
            slot=Slot.CUR,
            attempts=state.attempts,
            stats=stats,
        )
        reporter.accepted_continuation_step(state.step, state.omega_cur, state.load_t, state.omega_cur - old_omega, state.load_t - old_t, stats)
        totals.add(stats)
        state.step += 1

        d_t_step = state.load_t - old_t
        if d_t_step < opts.d_t_min:
            state.stop_reason = "d_t_min"
        elif state.omega_cur >= opts.omega_max:
            state.stop_reason = "omega_max"
        elif state.step >= opts.continuation_step_max:
            state.stop_reason = "step_max"
        elif stats.newton_its < 20 and state.step > 2 and state.last_d_t > 0.0 and d_t_step < 0.9 * state.last_d_t:
            state.d_omega *= 2.0
        state.last_d_t = d_t_step

    if state.running:
        state.stop_reason = "omega_max" if state.omega_cur >= opts.omega_max else "step_max"
    continuation_wall = perf_counter() - t_start
    summary = _summary(ctx, ops, state, totals, continuation_wall, curve_csv)
    ops.write_solution(Slot.CUR)

    if reporter.enabled:
        recorder.write()
        write_summary_json(summary_json, summary)
    reporter.result(state.omega_cur, state.load_t, len(recorder.rows), totals, continuation_wall, state.stop_reason, curve_csv)

    result = EngineRunResult(ctx.output_dir, curve_csv, summary_json, float(summary["wall_time"]), summary)
    ctx.last_result = result
    return ContinuationCurve(rows=recorder.rows if reporter.enabled else [], summary=summary, csv_path=curve_csv)
