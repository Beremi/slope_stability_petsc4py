from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "src" / "petsc_ssr" / "native"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_stats_api_exposes_petcs_event_timers() -> None:
    header = _read(NATIVE / "include" / "petsc_ssr_stats.h")
    impl = _read(NATIVE / "profiling" / "stats.c.inc")

    assert "typedef struct {" in header
    assert "SsrProfileTimer" in header
    assert "SsrProfileTimerBegin" in header
    assert "SsrProfileTimerEnd" in header
    assert "SsrStage" in header
    assert "SsrProfilerStagePush" in header
    assert "SsrProfilerStagePop" in header
    assert "SSR_PROFILE_STAGE_PUSH" in header
    assert "SSR_PROFILE_STAGE_POP" in header
    assert "SSR_PROFILE_TIMER_BEGIN" in header
    assert "SSR_PROFILE_TIMER_END" in header
    assert "PetscLogEventBegin" in impl
    assert "PetscLogEventEnd" in impl
    assert "PetscLogStageRegister" in impl
    assert "PetscLogStagePush(ssr_log_stages[stage_id])" in impl
    assert "PetscLogStagePop()" in impl
    assert "PetscTime(&timer->start_time)" in impl
    assert "end_time - timer->start_time" in impl
    assert "SSR_EVENT_ENGINE_CREATE" in header
    assert "SSR_EVENT_ENGINE_RUN" in header
    assert "SSR_EVENT_ASSEMBLE_NEUMANN" in header
    assert "SSR_EVENT_PMG_FINE_SMOOTH" in header
    assert "SSR_EVENT_PMG_P2_SMOOTH" in header
    assert "SSR_EVENT_PMG_COARSE_SOLVE" in header
    assert "SSR_EVENT_PMG_TRANSFER" in header
    assert "SSR_EVENT_PMG_RESIDUAL" in header
    assert "SSR_EVENT_PMG_OPERATOR_UPDATE" in header
    assert "SSR_EVENT_PMG_GALERKIN_PRODUCT" in header
    assert "SSR_EVENT_PMG_REDISTRIBUTE" in header
    assert "SSR_EVENT_PMG_SUBMATRIX" in header
    assert "SSR_EVENT_PMG_CONCATENATE" in header
    assert "SSR_EVENT_DEFLATION_COARSE" in header
    assert "SSR_EVENT_DEFLATION_PC_APPLY" in header
    assert "SSR_EVENT_OPERATOR_BUILD" in header
    assert "SSR_EVENT_BUILD_RHS" in header
    assert "SSR_EVENT_KSP_SETUP" in header
    assert "SSR_EVENT_LINE_SEARCH" in header
    assert "SSR_EVENT_CONTINUATION_RUN" in header
    assert "SSR_EVENT_NEWTON_SOLVE" in header
    assert "SSR_EVENT_HYDRO_RUN" in header
    assert "SSR_EVENT_HYDRO_ASSEMBLE" in header
    assert "SSR_EVENT_HYDRO_LINEAR_SOLVE" in header
    assert "SSR PMG Fine Smooth" in impl
    assert "SSR Engine Create" in impl
    assert "SSR Engine Run" in impl
    assert "SSR PMG Transfer" in impl
    assert "SSR PMG Operator Update" in impl
    assert "SSR PMG Galerkin Product" in impl
    assert "SSR Deflation Coarse" in impl
    assert "SSR Deflation PC Apply" in impl
    assert "SSR Operator Build" in impl
    assert "SSR Build RHS" in impl
    assert "SSR KSP Setup" in impl
    assert "SSR Line Search" in impl
    assert "SSR Continuation Run" in impl
    assert "SSR Newton Solve" in impl
    assert "SSR Hydro Run" in impl
    assert "SSR Hydro Assemble" in impl
    assert "SSR Hydro Linear Solve" in impl
    assert "SSR Assemble Neumann" in impl
    assert "SSR_STAGE_DEFLATION_ORTHOGONALIZE" in header
    assert "SSR_STAGE_PMG_SHELL_FINE_SMOOTH" in header
    assert "SSR_STAGE_PMG_SHELL_TRANSFER" in header
    assert "deflation_orthogonalize" in impl
    assert "pmg_shell_fine_smooth" in impl
    assert "pmg_shell_transfer" in impl
    assert "SsrNewtonStepStats" in header
    assert "SsrContinuationStats" in header
    assert "SsrHydroStats" in header
    assert "SsrNeumannStats" in header
    assert "SsrStatsAddNewtonStepAssembly" in header
    assert "SsrStatsAddNewtonStepLinearSolve" in header
    assert "SsrStatsAddNewtonStepIteration" in header
    assert "SsrStatsAddNewtonStepLineSearch" in header
    assert "SsrStatsAddHydroAssembly" in header
    assert "SsrStatsAddHydroLinearSolve" in header
    assert "SsrStatsAddNeumannAssembly" in header
    assert "SsrStatsAccumulateElapsed" in header
    assert "const SsrNewtonStepStats *step_stats" in header
    assert "const void *step_result" not in header
    assert "if (counter) *counter += elapsed" in impl
    assert "stats->assembly_time += elapsed" in impl
    assert "stats->solve_time += elapsed" in impl
    assert "stats->total_linear_its += its" in impl
    assert "stats->line_search_its += its" in impl
    assert "stats->accepted_steps++" in impl
    assert "stats->total_newton_its += step_stats->newton_its" in impl
    assert "stats->linear_solves += 1" in impl
    assert "stats->quadrature_points += quadrature_points" in impl


def test_native_neumann_assembly_uses_shared_stats_timer_api() -> None:
    header = _read(NATIVE / "assembly" / "assembly.h")
    source = _read(NATIVE / "assembly" / "neumann.c")

    assert '#include "petsc_ssr_stats.h"' in header
    assert '#include "petsc_ssr_stats.h"' in source
    assert "SsrNeumannStats neumann_stats" in header
    assert "SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ASSEMBLE_NEUMANN, dm, rhs, &profile_timer)" in source
    assert "SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed)" in source
    assert "SsrStatsAddNeumannAssembly(&ctx->neumann_stats" in source
    assert "assembly_time=%.6g" in source


def test_log_stage_registration_is_centralized_in_stats_api() -> None:
    context = _read(NATIVE / "core" / "context.c.inc")
    fixed = _read(NATIVE / "linear" / "deflation_krylov.c.inc")
    pmg = _read(NATIVE / "linear" / "pmg_shell.c.inc")

    assert "PetscLogStageRegister" not in context
    assert "static PetscLogStage log_stage_" not in context
    assert "PetscCall(SsrProfilerRegister(NULL));" in context
    assert "PetscLogStagePush(" not in fixed
    assert "PetscLogStagePop(" not in fixed
    assert "PetscLogStagePush(" not in pmg
    assert "PetscLogStagePop(" not in pmg
    assert "SSR_PROFILE_STAGE_PUSH(SSR_STAGE_DEFLATION_ORTHOGONALIZE)" in fixed
    assert "SSR_PROFILE_STAGE_PUSH(SSR_STAGE_DEFLATION_INITIAL_GUESS)" in fixed
    assert "SSR_PROFILE_STAGE_PUSH(SSR_STAGE_DEFLATION_PROJECTOR)" in fixed
    assert "SSR_PROFILE_STAGE_PUSH(SSR_STAGE_PMG_SHELL_FINE_SMOOTH)" in pmg
    assert "SSR_PROFILE_STAGE_PUSH(SSR_STAGE_PMG_SHELL_TRANSFER)" in pmg
    assert "SSR_PROFILE_STAGE_PUSH(SSR_STAGE_PMG_SHELL_P1)" in pmg
    assert "SSR_PROFILE_STAGE_PUSH(SSR_STAGE_PMG_SHELL_P2)" in pmg


def test_newton_hot_path_uses_stats_timer_api_for_assembly_and_solves() -> None:
    fixed = _read(NATIVE / "nonlinear" / "newton_fixed_load.c.inc")
    indirect = _read(NATIVE / "nonlinear" / "newton_indirect_ssr.c.inc")

    assert "SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ASSEMBLE_TANGENT" in fixed
    assert "const SsrEvent assembly_event" in indirect
    assert "SSR_EVENT_ASSEMBLE_RESIDUAL" in indirect
    assert "SSR_EVENT_ASSEMBLE_TANGENT" in indirect
    assert "SSR_PROFILE_TIMER_BEGIN(NULL, assembly_event" in indirect

    for source in (fixed, indirect):
        assert "SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_NEWTON_SOLVE" in source
        assert "SSR_PROFILE_TIMER_END(NULL, &run_timer, &wall_time)" in source
        assert "SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_KSP_SOLVE" in source
        assert "SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_LINE_SEARCH" in source
        assert "SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed)" in source
        assert "SsrStatsAddNewtonStepAssembly(stats, elapsed)" in source
        assert "SsrStatsAddNewtonStepLinearSolve(stats," in source
        assert "SsrStatsAddNewtonStepIteration(stats)" in source
        assert "SsrStatsAddNewtonStepLineSearch(stats, ls_its)" in source
        assert "stats->wall_time             = wall_time" in source
        assert "PetscTime(" not in source
        assert "assembly_time += elapsed" not in source
        assert "solve_time += elapsed" not in source
        assert "total_linear_its +=" not in source
        assert "line_search_its += ls_its" not in source
        assert "assembly_time += t1 - t0" not in source
        assert "solve_time += t1 - t0" not in source
        assert "stats->wall_time             = t1 - t_start" not in source


def test_cli_elastic_setup_uses_stats_timer_api() -> None:
    source = _read(NATIVE / "core" / "cli_runner.c.inc")

    assert "SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ENGINE_RUN" in source
    assert "SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ASSEMBLE_ELASTIC" in source
    assert "SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_APPLY_DIRICHLET" in source
    assert "SSR_PROFILE_TIMER_END(NULL, &run_timer, &wall_time)" in source
    assert "SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elastic_assembly_time)" in source
    assert "SSR_PROFILE_TIMER_END(NULL, &profile_timer, &dirichlet_time)" in source
    assert "dirichlet_time" in source
    assert "PetscTime(" not in source


def test_cython_bridge_assembly_and_solve_helpers_use_stats_timer_api() -> None:
    source = _read(NATIVE / "cython" / "cython_api.c.inc")

    assert "SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ENGINE_CREATE" in source
    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_NEWTON_SOLVE") >= 2
    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ASSEMBLE_ELASTIC") == 1
    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ASSEMBLE_TANGENT") >= 4
    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, assembly_event") >= 2
    assert "SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_OPERATOR_BUILD" in source
    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_BUILD_RHS") >= 3
    assert "SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_KSP_SETUP" in source
    assert "SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_DEFLATION_ORTHO" in source
    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_KSP_SOLVE") >= 6
    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_LINE_SEARCH") >= 5
    assert "SSR_PROFILE_BEGIN(NULL, SSR_EVENT_KSP_SOLVE" not in source
    assert "SSR_PROFILE_END(NULL, SSR_EVENT_KSP_SOLVE" not in source
    assert "ctx->elastic_assembly_time = t1 - t0" not in source
    assert "out->assembly_time = t1 - t0" not in source
    assert "out->solve_time = t1 - t0" not in source
    assert "out->wall_time          = t1 - t0" not in source
    assert "out->wall_time       = t1 - t0" not in source
    assert "out->wall_time        = t1 - t0" not in source
    assert "PetscTime(" not in source


def test_deflation_hot_path_uses_stats_timer_api() -> None:
    source = _read(NATIVE / "linear" / "deflation_krylov.c.inc")

    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_DEFLATION_ORTHO") >= 2
    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_DEFLATION_COARSE") >= 2
    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_DEFLATION_PC_APPLY") >= 2
    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_DEFLATION_PROJECT") >= 2
    assert source.count("SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed)") >= 2
    assert "SsrStatsAccumulateElapsed(&solver->deflation_orthogonalization_time, elapsed)" in source
    assert "SsrStatsAccumulateElapsed(&solver->deflation_coarse_time, elapsed)" in source
    assert "SsrStatsAccumulateElapsed(&solver->deflation_coarse_time, coarse_time)" in source
    assert "SsrStatsAccumulateElapsed(&solver->deflation_pc_apply_time, pc_time)" in source
    assert "SsrStatsAccumulateElapsed(&solver->deflation_projector_time, projector_time)" in source
    assert "PetscTime(" not in source
    assert "deflation_orthogonalization_time += elapsed" not in source
    assert "deflation_coarse_time += elapsed" not in source
    assert "deflation_pc_apply_time += pc_time" not in source
    assert "deflation_projector_time += projector_time" not in source
    assert "deflation_orthogonalization_time += t1 - t0" not in source
    assert "deflation_coarse_time += t1 - t0" not in source
    assert "deflation_projector_time += t" not in source


def test_pmg_shell_setup_and_apply_emit_shared_profiler_events() -> None:
    source = _read(NATIVE / "linear" / "pmg_shell.c.inc")

    assert "SSR_PROFILE_BEGIN(NULL, SSR_EVENT_PMG_SETUP, pc, A)" in source
    assert "SSR_PROFILE_END(NULL, SSR_EVENT_PMG_SETUP, pc, A)" in source
    assert "SSR_PROFILE_BEGIN(NULL, SSR_EVENT_PMG_APPLY, pc, x)" in source
    assert "SSR_PROFILE_END(NULL, SSR_EVENT_PMG_APPLY, pc, x)" in source


def test_pmg_shell_apply_subphases_use_stats_timer_api() -> None:
    source = _read(NATIVE / "linear" / "pmg_shell.c.inc")

    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_FINE_SMOOTH") == 2
    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_P2_SMOOTH") == 2
    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_COARSE_SOLVE") == 1
    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_TRANSFER") == 4
    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_RESIDUAL") == 1
    assert source.count("SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed)") >= 10
    assert "SsrStatsAccumulateElapsed(&ctx->fine_smooth_time, elapsed)" in source
    assert "SsrStatsAccumulateElapsed(&ctx->p2_smooth_time, elapsed)" in source
    assert "SsrStatsAccumulateElapsed(&ctx->coarse_solve_time, elapsed)" in source
    assert "SsrStatsAccumulateElapsed(&ctx->restrict_time, elapsed)" in source
    assert "SsrStatsAccumulateElapsed(&ctx->prolong_time, elapsed)" in source
    assert "SsrStatsAccumulateElapsed(accum_time, elapsed)" in source
    assert "fine_smooth_time += elapsed" not in source
    assert "p2_smooth_time += elapsed" not in source
    assert "coarse_solve_time += elapsed" not in source
    assert "restrict_time += elapsed" not in source
    assert "prolong_time += elapsed" not in source
    assert "residual_time += t1 - t0" not in source
    assert "fine_smooth_time += t1 - t0" not in source
    assert "p2_smooth_time += t1 - t0" not in source
    assert "coarse_solve_time += t1 - t0" not in source
    assert "restrict_time += t1 - t0" not in source
    assert "prolong_time += t1 - t0" not in source


def test_pmg_shell_operator_update_submetrics_use_stats_timer_api() -> None:
    source = _read(NATIVE / "linear" / "pmg_shell.c.inc")

    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_OPERATOR_UPDATE") == 1
    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_GALERKIN_PRODUCT") == 2
    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_REDISTRIBUTE") == 2
    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_SUBMATRIX") == 1
    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_CONCATENATE") == 1
    assert "SsrStatsAccumulateElapsed(&ctx->operator_update_time, update_elapsed)" in source
    assert "PetscTime(" not in source
    assert "ctx->operator_update_time += update_elapsed" not in source
    assert "operator_update_time += t" not in source
    assert "submatrix_time   = t" not in source
    assert "concatenate_time = t" not in source


def test_hydro_solver_uses_shared_stats_timer_api() -> None:
    source = _read(NATIVE / "mesh" / "hydro_seepage.c")

    assert '#include "petsc_ssr_stats.h"' in source
    assert "SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_HYDRO_RUN" in source
    assert "SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_HYDRO_ASSEMBLE" in source
    assert "SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_HYDRO_LINEAR_SOLVE" in source
    assert "SSR_PROFILE_TIMER_END(NULL, &run_timer, &wall_time)" in source
    assert "SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed)" in source
    assert "SsrHydroStats  stats" in source
    assert "SsrStatsAddHydroAssembly(&ctx->stats, elapsed)" in source
    assert "SsrStatsAddHydroLinearSolve(stats, *its, elapsed, reason)" in source
    assert "const SsrHydroStats *stats" in source
    assert "ctx.stats.total_linear_its" in source
    assert "ctx.stats.assembly_time" in source
    assert "ctx.stats.solve_time" in source
    assert "PetscTime(" not in source
    assert "ctx->assembly_time += elapsed" not in source
    assert "solve_time += st" not in source
    assert "*time = elapsed" not in source
    assert "ctx->assembly_time += t1 - t0" not in source
    assert "*time = t1 - t0" not in source


def test_continuation_acceptance_uses_stats_checkpoint_api() -> None:
    indirect = _read(NATIVE / "continuation" / "continuation_indirect_ssr.c.inc")
    direct = _read(NATIVE / "continuation" / "continuation_direct_ssr.c.inc")

    for source in (indirect, direct):
        assert "SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_CONTINUATION_RUN" in source
        assert "SSR_PROFILE_TIMER_END(NULL, &run_timer, &wall_time)" in source
        assert source.count("SsrStatsAcceptContinuationStep(stats, &nstats)") >= 3
        assert "stats->wall_time = wall_time" in source
        assert "PetscTime(" not in source
        assert "stats->accepted_steps++" not in source
        assert "stats->total_newton_its += nstats.newton_its" not in source
        assert "stats->total_linear_its += nstats.total_linear_its" not in source
        assert "stats->total_line_search_its += nstats.line_search_its" not in source
        assert "stats->wall_time = t_end - t_start" not in source


def test_replay_debug_assembly_checks_use_stats_timer_api() -> None:
    source = _read(NATIVE / "replay" / "replay_debug.c.inc")

    assert source.count("SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ASSEMBLE_TANGENT") >= 3
    assert source.count("SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed)") >= 3
    assert "PetscTime(" not in source
    assert "t1 - t0" not in source
