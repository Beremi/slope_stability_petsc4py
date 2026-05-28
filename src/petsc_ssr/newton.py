from __future__ import annotations

from time import perf_counter
from typing import Any

from .operations import DisplacementVector, EngineOps
from .telemetry import HUGE, NewtonStats, RankReporter


def _stopping_by_criterion(
    criterion: str,
    *,
    rel_residual: float,
    rel_correction: float,
    abs_delta_lambda: float,
    tol: float,
    have_correction: bool,
) -> bool:
    criterion = criterion.lower()
    if criterion == "relative_residual":
        return rel_residual <= tol
    if criterion == "relative_correction":
        return have_correction and rel_correction <= tol
    if criterion == "absolute_delta_lambda":
        return have_correction and abs_delta_lambda <= tol
    return False


def _fixed_report(rel: float, rel_corr: float, line_search: dict[str, Any] | None, solve: dict[str, Any] | None, *, stop: bool, failed: bool = False) -> dict[str, Any]:
    return {
        "rel_residual": rel,
        "rel_correction": rel_corr,
        "alpha": 0.0 if line_search is None else line_search["alpha"],
        "linear_its": 0 if solve is None else solve["linear_its"],
        "line_search_its": 0 if line_search is None else line_search["line_search_its"],
        "stop": stop,
        "failed": failed,
    }


def _indirect_report(
    rel: float,
    rel_corr: float,
    lambda_value: float,
    update: dict[str, Any] | None,
    line_search: dict[str, Any] | None,
    solve: dict[str, Any] | None,
    *,
    stop: bool,
    failed: bool = False,
) -> dict[str, Any]:
    return {
        "lambda_out": lambda_value,
        "rel_residual": rel,
        "rel_correction": rel_corr,
        "alpha": 0.0 if line_search is None else line_search["alpha"],
        "delta_lambda": 0.0 if update is None else update["delta_lambda"],
        "linear_its_w": 0 if solve is None else solve["linear_its_w"],
        "linear_its_v": 0 if solve is None else solve["linear_its_v"],
        "linear_its": 0 if solve is None else solve["linear_its"],
        "line_search_its": 0 if line_search is None else line_search["line_search_its"],
        "stop": stop,
        "failed": failed,
    }


def _limit_load_report(
    rel: float,
    rel_corr: float,
    load_t: float,
    update: dict[str, Any] | None,
    line_search: dict[str, Any] | None,
    solve: dict[str, Any] | None,
    *,
    stop: bool,
    failed: bool = False,
) -> dict[str, Any]:
    return {
        "lambda_out": load_t,
        "rel_residual": rel,
        "rel_correction": rel_corr,
        "alpha": 0.0 if line_search is None else line_search["alpha"],
        "delta_lambda": 0.0 if update is None else update["delta_lambda"],
        "linear_its_w": 0 if solve is None else solve["linear_its_w"],
        "linear_its_v": 0 if solve is None else solve["linear_its_v"],
        "linear_its": 0 if solve is None else solve["linear_its"],
        "line_search_its": 0 if line_search is None else line_search["line_search_its"],
        "stop": stop,
        "failed": failed,
    }


def solve_fixed_lambda_newton(
    ops: EngineOps,
    opts: Any,
    reporter: RankReporter,
    *,
    U_it: DisplacementVector,
    lambda_value: float,
    criterion: str,
    stop_tol: float,
) -> NewtonStats:
    stats = NewtonStats()
    algorithm = ops.algorithm_objects(stats)
    vectors = algorithm.vectors
    operators = algorithm.operators
    constitutive_matrix_builder = algorithm.constitutive_matrix_builder
    rhs_builder = algorithm.rhs_builder
    linear_solver = algorithm.linear_solver
    damping = algorithm.damping
    update = algorithm.update

    F = vectors.F
    K_elast = operators.K_elast
    K_tangent = operators.K_tangent
    du = vectors.du

    rel_corr = HUGE
    r = opts.r_min
    t0 = perf_counter()

    for iteration in range(opts.newton_max_it):
        response = constitutive_matrix_builder.build_F_K_tangent_all(lambda_value, U_it)
        rel = float(response["rel_residual"])
        stats.final_rel = rel

        stop = _stopping_by_criterion(
            criterion,
            rel_residual=rel,
            rel_correction=rel_corr,
            abs_delta_lambda=HUGE,
            tol=stop_tol,
            have_correction=False,
        )
        if stop or rel <= opts.newton_rtol:
            stats.converged = True
            reporter.fixed_newton_step(lambda_value, iteration, _fixed_report(rel, rel_corr, None, None, stop=True))
            break

        K_r = operators.regularized_tangent(r=r, K_elast=K_elast, K_tangent=K_tangent)
        rhs = rhs_builder.fixed_lambda_rhs(F)

        linear_solver.setup_preconditioner(K_r)
        linear_solver.A_orthogonalize(K_r, label="SSR fixed-lambda Newton correction")
        solve = linear_solver.solve(K_r, rhs, du, label="SSR fixed-lambda Newton correction")

        line_search = damping.fixed_lambda_directional(U_it, lambda_value, du, F)
        rel_corr = float(line_search["rel_correction"])
        stop = _stopping_by_criterion(
            criterion,
            rel_residual=rel,
            rel_correction=rel_corr,
            abs_delta_lambda=HUGE,
            tol=stop_tol,
            have_correction=bool(line_search["alpha"] > 0.0),
        )
        accepted = update.apply_fixed_lambda(U_it, float(line_search["alpha"]), r, update_basis=not stop)
        r = float(accepted["r_out"])
        failed = bool(accepted["failed"])
        stats.final_rel = rel
        stats.final_rel_correction = rel_corr
        reporter.fixed_newton_step(lambda_value, iteration, _fixed_report(rel, rel_corr, line_search, solve, stop=stop, failed=failed))

        if stop:
            stats.converged = True
            break
        if failed:
            break

    stats.wall_time = perf_counter() - t0
    reporter.fixed_newton_summary(lambda_value, stats)
    return stats


def solve_indirect_newton(
    ops: EngineOps,
    opts: Any,
    reporter: RankReporter,
    *,
    U_it: DisplacementVector,
    lambda_start: float,
    omega: float,
) -> tuple[float, NewtonStats]:
    stats = NewtonStats()
    algorithm = ops.algorithm_objects(stats)
    vectors = algorithm.vectors
    operators = algorithm.operators
    constitutive_matrix_builder = algorithm.constitutive_matrix_builder
    rhs_builder = algorithm.rhs_builder
    linear_solver = algorithm.linear_solver
    algebra = algorithm.algebra
    damping = algorithm.damping
    update = algorithm.update

    F = vectors.F
    G = vectors.G
    W = vectors.W
    V = vectors.V
    d_U = vectors.d_U
    K_elast = operators.K_elast
    K_tangent = operators.K_tangent

    rel_corr = HUGE
    abs_dlambda = HUGE
    r = opts.r_min
    compute_diffs = True
    lambda_it = lambda_start
    t0 = perf_counter()

    for iteration in range(opts.newton_max_it):
        response = constitutive_matrix_builder.build_F_K_tangent_all(lambda_it, U_it)
        rel = float(response["rel_residual"])
        stats.final_rel = rel

        stop = _stopping_by_criterion(
            opts.newton_stopping_criterion,
            rel_residual=rel,
            rel_correction=rel_corr,
            abs_delta_lambda=abs_dlambda,
            tol=opts.newton_stopping_tol,
            have_correction=False,
        )
        if stop or (rel <= opts.newton_rtol and iteration > 0):
            stats.converged = True
            reporter.indirect_newton_step(
                omega,
                iteration,
                _indirect_report(rel, rel_corr, lambda_it, None, None, None, stop=True),
            )
            break

        if compute_diffs:
            G = constitutive_matrix_builder.build_lambda_derivative(lambda_it, U_it, F)
        K_r = operators.regularized_tangent(r=r, K_elast=K_elast, K_tangent=K_tangent)
        rhs_W, rhs_V = rhs_builder.indirect_rhs(G, F)

        linear_solver.setup_preconditioner(K_r)
        linear_solver.A_orthogonalize(K_r, label="SSR indirect dW")
        solve_w = linear_solver.solve(K_r, rhs_W, W, label="SSR indirect dW")

        linear_solver.setup_preconditioner(K_r, force_reuse=True)
        linear_solver.A_orthogonalize(K_r, label="SSR indirect dV")
        solve_v = linear_solver.solve(K_r, rhs_V, V, label="SSR indirect dV")

        solve = {
            "linear_its_w": int(solve_w["linear_its_w"]),
            "linear_its_v": int(solve_v["linear_its_v"]),
            "linear_its": int(solve_w["linear_its_w"]) + int(solve_v["linear_its_v"]),
        }
        newton_increment = algebra.combine_indirect_directions(V, W)
        d_lambda = float(newton_increment["delta_lambda"])
        abs_dlambda = float(newton_increment["abs_delta_lambda"])

        line_search = damping.ALG5(U_it, lambda_it, d_U, d_lambda, omega, rel)
        rel_corr = float(line_search["rel_correction"])
        stop = _stopping_by_criterion(
            opts.newton_stopping_criterion,
            rel_residual=rel,
            rel_correction=rel_corr,
            abs_delta_lambda=abs_dlambda,
            tol=opts.newton_stopping_tol,
            have_correction=bool(line_search["alpha"] > 0.0),
        )

        accepted = update.accept_indirect(U_it, lambda_it, omega, float(line_search["alpha"]), d_lambda, r, update_basis=not stop)
        if line_search["alpha"] > 0.0:
            lambda_it = float(accepted["lambda_out"])
            rel = float(line_search["trial_rel"])
        r = float(accepted["r_out"])
        compute_diffs = bool(accepted["compute_diffs_out"])
        failed = bool(accepted["failed"])
        stats.final_rel = rel
        stats.final_rel_correction = rel_corr
        reporter.indirect_newton_step(
            omega,
            iteration,
            _indirect_report(rel, rel_corr, lambda_it, newton_increment, line_search, solve, stop=stop, failed=failed),
        )

        if stop:
            stats.converged = True
            break
        if failed:
            break

    if stats.converged:
        stats.final_rel = ops.residual_rel(U_it.slot, lambda_it)
    elif stats.final_rel <= 10.0 * opts.newton_rtol:
        stats.converged = True

    stats.wall_time = perf_counter() - t0
    reporter.indirect_newton_summary(omega, lambda_it, stats)
    return lambda_it, stats


def solve_limit_load_newton(
    ops: EngineOps,
    opts: Any,
    reporter: RankReporter,
    *,
    U_it: DisplacementVector,
    load_start: float,
    omega: float,
) -> tuple[float, NewtonStats]:
    stats = NewtonStats()
    algorithm = ops.algorithm_objects(stats)
    linear_solver = algorithm.linear_solver
    measured = ops.collecting(stats)
    vectors = algorithm.vectors
    W = vectors.W
    V = vectors.V

    load_t = float(load_start)
    rel_corr = HUGE
    r = opts.r_min
    t0 = perf_counter()

    for iteration in range(opts.newton_max_it):
        assembled = measured.assemble_limit_load(U_it.slot, opts.lambda_ell, load_t, r)
        rel = float(assembled["rel_residual"])
        stats.final_rel = rel
        stop = rel <= opts.newton_rtol and iteration > 0
        if stop:
            stats.converged = True
            reporter.limit_load_newton_step(
                omega,
                iteration,
                _limit_load_report(rel, rel_corr, load_t, None, None, None, stop=True),
            )
            break

        measured.build_limit_load_rhs(load_t)
        linear_solver.setup_preconditioner(algorithm.operators.K_r)
        linear_solver.A_orthogonalize(algorithm.operators.K_r, label="LL indirect dW")
        solve_w = linear_solver.solve(algorithm.operators.K_r, vectors.rhs_W, W, label="LL indirect dW")

        linear_solver.setup_preconditioner(algorithm.operators.K_r, force_reuse=True)
        linear_solver.A_orthogonalize(algorithm.operators.K_r, label="LL indirect dV")
        solve_v = linear_solver.solve(algorithm.operators.K_r, vectors.rhs_V, V, label="LL indirect dV")
        solve = {
            "linear_its_w": int(solve_w["linear_its_w"]),
            "linear_its_v": int(solve_v["linear_its_v"]),
            "linear_its": int(solve_w["linear_its_w"]) + int(solve_v["linear_its_v"]),
        }

        newton_increment = measured.form_limit_load_update()
        d_t = float(newton_increment["delta_lambda"])
        line_search = measured.limit_load_line_search(U_it.slot, opts.lambda_ell, load_t)
        rel_corr = float(line_search["rel_correction"])
        accepted = measured.accept_limit_load_update(
            U_it.slot,
            load_t,
            omega,
            float(line_search["alpha"]),
            d_t,
            r,
            update_basis=True,
        )
        if line_search["alpha"] > 0.0:
            load_t = float(accepted["lambda_out"])
        r = float(accepted["r_out"])
        failed = bool(accepted["failed"])
        stats.final_rel = rel
        stats.final_rel_correction = rel_corr
        reporter.limit_load_newton_step(
            omega,
            iteration,
            _limit_load_report(rel, rel_corr, load_t, newton_increment, line_search, solve, stop=False, failed=failed),
        )
        if failed:
            break

    stats.wall_time = perf_counter() - t0
    reporter.limit_load_newton_summary(omega, load_t, stats)
    return load_t, stats
