#include "petsc_ssr_stats.h"

struct _n_SsrStats {
  PetscInt newton_iterations;
  PetscInt line_search_iterations;
  PetscInt linear_solves;
  PetscInt linear_iterations;
};

struct _n_SsrProfiler {
  PetscBool registered;
};

static PetscLogEvent ssr_log_events[SSR_EVENT_COUNT];
static PetscBool     ssr_log_events_registered = PETSC_FALSE;
static PetscLogStage ssr_log_stages[SSR_STAGE_COUNT];
static PetscBool     ssr_log_stages_registered = PETSC_FALSE;

static const char *const ssr_event_names[SSR_EVENT_COUNT] = {
  "SSR Assemble Elastic",
  "SSR Assemble Neumann",
  "SSR Engine Create",
  "SSR Engine Run",
  "SSR Assemble Tangent",
  "SSR Assemble Residual",
  "SSR Apply Dirichlet",
  "SSR Operator Build",
  "SSR Build RHS",
  "SSR KSP Setup",
  "SSR KSP Solve",
  "SSR PMG Setup",
  "SSR PMG Apply",
  "SSR PMG Fine Smooth",
  "SSR PMG P2 Smooth",
  "SSR PMG Coarse Solve",
  "SSR PMG Transfer",
  "SSR PMG Residual",
  "SSR PMG Operator Update",
  "SSR PMG Galerkin Product",
  "SSR PMG Redistribute",
  "SSR PMG Submatrix",
  "SSR PMG Concatenate",
  "SSR Deflation Orthogonalize",
  "SSR Deflation Coarse",
  "SSR Deflation PC Apply",
  "SSR Deflation Project",
  "SSR Linear Operator MatMult",
  "SSR Krylov Orthogonalize",
  "SSR Krylov Least Squares",
  "SSR Krylov Solution Update",
  "SSR Line Search",
  "SSR Output Write",
  "SSR Continuation Run",
  "SSR Newton Solve",
  "SSR Hydro Run",
  "SSR Hydro Assemble",
  "SSR Hydro Linear Solve",
};

static const char *const ssr_stage_names[SSR_STAGE_COUNT] = {
  "deflation_orthogonalize",
  "deflation_initial_guess",
  "deflation_projector",
  "pmg_shell_fine_smooth",
  "pmg_shell_residual",
  "pmg_shell_transfer",
  "pmg_shell_p2",
  "pmg_shell_p1",
};

PetscErrorCode SsrProfilerRegister(SsrProfiler *profiler)
{
  (void)profiler;
  PetscFunctionBeginUser;
  if (!ssr_log_events_registered) {
    for (PetscInt i = 0; i < (PetscInt)SSR_EVENT_COUNT; ++i) {
      PetscCall(PetscLogEventRegister(ssr_event_names[i], 0, &ssr_log_events[i]));
    }
    ssr_log_events_registered = PETSC_TRUE;
  }
  if (!ssr_log_stages_registered) {
    for (PetscInt i = 0; i < (PetscInt)SSR_STAGE_COUNT; ++i) {
      PetscCall(PetscLogStageRegister(ssr_stage_names[i], &ssr_log_stages[i]));
    }
    ssr_log_stages_registered = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrProfilerBegin(SsrProfiler *profiler, SsrEvent event_id, PetscObject a, PetscObject b)
{
  (void)profiler;
  PetscFunctionBeginUser;
  PetscCheck(event_id >= 0 && event_id < SSR_EVENT_COUNT, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid SSR profiler event id %d", (int)event_id);
  PetscCall(SsrProfilerRegister(NULL));
  PetscCall(PetscLogEventBegin(ssr_log_events[event_id], a, b, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrProfilerEnd(SsrProfiler *profiler, SsrEvent event_id, PetscObject a, PetscObject b)
{
  (void)profiler;
  PetscFunctionBeginUser;
  PetscCheck(event_id >= 0 && event_id < SSR_EVENT_COUNT, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid SSR profiler event id %d", (int)event_id);
  PetscCall(SsrProfilerRegister(NULL));
  PetscCall(PetscLogEventEnd(ssr_log_events[event_id], a, b, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrProfilerStagePush(SsrStage stage_id)
{
  PetscFunctionBeginUser;
  PetscCheck(stage_id >= 0 && stage_id < SSR_STAGE_COUNT, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid SSR profiler stage id %d", (int)stage_id);
  PetscCall(SsrProfilerRegister(NULL));
  PetscCall(PetscLogStagePush(ssr_log_stages[stage_id]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrProfilerStagePop(SsrStage stage_id)
{
  PetscFunctionBeginUser;
  PetscCheck(stage_id >= 0 && stage_id < SSR_STAGE_COUNT, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid SSR profiler stage id %d", (int)stage_id);
  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrProfileTimerBegin(SsrProfiler *profiler, SsrEvent event_id, PetscObject a, PetscObject b, SsrProfileTimer *timer)
{
  PetscFunctionBeginUser;
  PetscCheck(timer, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrProfileTimerBegin requires a timer");
  timer->event_id  = event_id;
  timer->object_a  = a;
  timer->object_b  = b;
  timer->active    = PETSC_TRUE;
  timer->start_time = 0.0;
  PetscCall(SsrProfilerBegin(profiler, event_id, a, b));
  PetscCall(PetscTime(&timer->start_time));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrProfileTimerEnd(SsrProfiler *profiler, SsrProfileTimer *timer, PetscLogDouble *elapsed)
{
  PetscLogDouble end_time;

  PetscFunctionBeginUser;
  PetscCheck(timer, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrProfileTimerEnd requires a timer");
  PetscCheck(timer->active, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "SsrProfileTimerEnd called for an inactive timer");
  if (elapsed) *elapsed = 0.0;
  PetscCall(PetscTime(&end_time));
  if (elapsed) *elapsed = end_time - timer->start_time;
  PetscCall(SsrProfilerEnd(profiler, timer->event_id, timer->object_a, timer->object_b));
  timer->active = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrStatsAccumulateElapsed(PetscLogDouble *counter, PetscLogDouble elapsed)
{
  PetscFunctionBeginUser;
  if (counter) *counter += elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrStatsAddNewtonIteration(SsrStats *stats, const char phase[])
{
  (void)phase;
  PetscFunctionBeginUser;
  if (stats) stats->newton_iterations += 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrStatsAddLinearSolve(SsrStats *stats, PetscInt its, KSPConvergedReason reason)
{
  (void)reason;
  PetscFunctionBeginUser;
  if (stats) {
    stats->linear_solves += 1;
    stats->linear_iterations += its;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrStatsAddLineSearchIteration(SsrStats *stats)
{
  PetscFunctionBeginUser;
  if (stats) stats->line_search_iterations += 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrStatsAddNewtonStepAssembly(SsrNewtonStepStats *stats, PetscLogDouble elapsed)
{
  PetscFunctionBeginUser;
  if (stats) stats->assembly_time += elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrStatsAddNewtonStepTangentAssembly(SsrNewtonStepStats *stats, PetscLogDouble elapsed)
{
  PetscFunctionBeginUser;
  if (stats) stats->tangent_assembly_time += elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrStatsAddNewtonStepResidualAssembly(SsrNewtonStepStats *stats, PetscLogDouble elapsed)
{
  PetscFunctionBeginUser;
  if (stats) stats->residual_assembly_time += elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrStatsAddNewtonStepOperatorBuild(SsrNewtonStepStats *stats, PetscLogDouble elapsed)
{
  PetscFunctionBeginUser;
  if (stats) stats->operator_build_time += elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrStatsAddNewtonStepLinearSolve(SsrNewtonStepStats *stats, PetscInt its, PetscLogDouble elapsed)
{
  PetscFunctionBeginUser;
  if (stats) {
    stats->solve_time += elapsed;
    stats->total_linear_its += its;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrStatsAddNewtonStepIteration(SsrNewtonStepStats *stats)
{
  PetscFunctionBeginUser;
  if (stats) stats->newton_its += 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrStatsAddNewtonStepLineSearch(SsrNewtonStepStats *stats, PetscInt its)
{
  PetscFunctionBeginUser;
  if (stats) stats->line_search_its += its;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrStatsAddNewtonStepLineSearchTime(SsrNewtonStepStats *stats, PetscLogDouble elapsed)
{
  PetscFunctionBeginUser;
  if (stats) stats->line_search_time += elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrStatsAcceptContinuationStep(SsrContinuationStats *stats, const SsrNewtonStepStats *step_stats)
{
  PetscFunctionBeginUser;
  if (stats && step_stats) {
    stats->accepted_steps++;
    stats->total_newton_its += step_stats->newton_its;
    stats->total_linear_its += step_stats->total_linear_its;
    stats->total_line_search_its += step_stats->line_search_its;
    stats->final_rel            = step_stats->final_rel;
    stats->final_rel_correction = step_stats->final_rel_correction;
    stats->assembly_time += step_stats->assembly_time;
    stats->tangent_assembly_time += step_stats->tangent_assembly_time;
    stats->residual_assembly_time += step_stats->residual_assembly_time;
    stats->operator_build_time += step_stats->operator_build_time;
    stats->linear_solve_time += step_stats->solve_time;
    stats->line_search_time += step_stats->line_search_time;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrStatsAddHydroAssembly(SsrHydroStats *stats, PetscLogDouble elapsed)
{
  PetscFunctionBeginUser;
  if (stats) stats->assembly_time += elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrStatsAddHydroLinearSolve(SsrHydroStats *stats, PetscInt its, PetscLogDouble elapsed, KSPConvergedReason reason)
{
  (void)reason;
  PetscFunctionBeginUser;
  if (stats) {
    stats->linear_solves += 1;
    stats->total_linear_its += its;
    stats->solve_time += elapsed;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrStatsAddNeumannAssembly(SsrNeumannStats *stats, PetscInt rules, PetscInt faces, PetscInt quadrature_points, PetscLogDouble elapsed)
{
  PetscFunctionBeginUser;
  if (stats) {
    stats->rules += rules;
    stats->faces += faces;
    stats->quadrature_points += quadrature_points;
    stats->assembly_time += elapsed;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
