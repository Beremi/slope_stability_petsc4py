#ifndef PETSC_SSR_STATS_H
#define PETSC_SSR_STATS_H

#include <petscksp.h>

typedef struct _n_SsrStats SsrStats;
typedef struct _n_SsrProfiler SsrProfiler;

typedef struct {
  PetscReal      final_rel;
  PetscReal      final_rel_correction;
  PetscInt       newton_its;
  PetscInt       total_linear_its;
  PetscInt       line_search_its;
  PetscLogDouble assembly_time;
  PetscLogDouble solve_time;
  PetscLogDouble wall_time;
  PetscBool      converged;
} SsrNewtonStepStats;

typedef struct {
  PetscInt       accepted_steps;
  PetscInt       total_newton_its;
  PetscInt       total_linear_its;
  PetscInt       total_line_search_its;
  PetscReal      omega_last;
  PetscReal      lambda_last;
  PetscReal      final_rel;
  PetscReal      final_rel_correction;
  PetscLogDouble wall_time;
  char           stop_reason[64];
} SsrContinuationStats;

typedef SsrContinuationStats SSRStats;

typedef struct {
  PetscInt       linear_solves;
  PetscInt       total_linear_its;
  PetscLogDouble assembly_time;
  PetscLogDouble solve_time;
} SsrHydroStats;

typedef struct {
  PetscInt       rules;
  PetscInt       faces;
  PetscInt       quadrature_points;
  PetscLogDouble assembly_time;
} SsrNeumannStats;

typedef enum {
  SSR_EVENT_ASSEMBLE_ELASTIC = 0,
  SSR_EVENT_ASSEMBLE_NEUMANN,
  SSR_EVENT_ENGINE_CREATE,
  SSR_EVENT_ENGINE_RUN,
  SSR_EVENT_ASSEMBLE_TANGENT,
  SSR_EVENT_ASSEMBLE_RESIDUAL,
  SSR_EVENT_APPLY_DIRICHLET,
  SSR_EVENT_OPERATOR_BUILD,
  SSR_EVENT_BUILD_RHS,
  SSR_EVENT_KSP_SETUP,
  SSR_EVENT_KSP_SOLVE,
  SSR_EVENT_PMG_SETUP,
  SSR_EVENT_PMG_APPLY,
  SSR_EVENT_PMG_FINE_SMOOTH,
  SSR_EVENT_PMG_P2_SMOOTH,
  SSR_EVENT_PMG_COARSE_SOLVE,
  SSR_EVENT_PMG_TRANSFER,
  SSR_EVENT_PMG_RESIDUAL,
  SSR_EVENT_PMG_OPERATOR_UPDATE,
  SSR_EVENT_PMG_GALERKIN_PRODUCT,
  SSR_EVENT_PMG_REDISTRIBUTE,
  SSR_EVENT_PMG_SUBMATRIX,
  SSR_EVENT_PMG_CONCATENATE,
  SSR_EVENT_DEFLATION_ORTHO,
  SSR_EVENT_DEFLATION_COARSE,
  SSR_EVENT_DEFLATION_PC_APPLY,
  SSR_EVENT_DEFLATION_PROJECT,
  SSR_EVENT_LINE_SEARCH,
  SSR_EVENT_OUTPUT_WRITE,
  SSR_EVENT_CONTINUATION_RUN,
  SSR_EVENT_NEWTON_SOLVE,
  SSR_EVENT_HYDRO_RUN,
  SSR_EVENT_HYDRO_ASSEMBLE,
  SSR_EVENT_HYDRO_LINEAR_SOLVE,
  SSR_EVENT_COUNT
} SsrEvent;

typedef enum {
  SSR_STAGE_DEFLATION_ORTHOGONALIZE = 0,
  SSR_STAGE_DEFLATION_INITIAL_GUESS,
  SSR_STAGE_DEFLATION_PROJECTOR,
  SSR_STAGE_PMG_SHELL_FINE_SMOOTH,
  SSR_STAGE_PMG_SHELL_RESIDUAL,
  SSR_STAGE_PMG_SHELL_TRANSFER,
  SSR_STAGE_PMG_SHELL_P2,
  SSR_STAGE_PMG_SHELL_P1,
  SSR_STAGE_COUNT
} SsrStage;

typedef struct {
  SsrEvent       event_id;
  PetscLogDouble start_time;
  PetscObject    object_a;
  PetscObject    object_b;
  PetscBool      active;
} SsrProfileTimer;

PetscErrorCode SsrProfilerRegister(SsrProfiler *profiler);
PetscErrorCode SsrProfilerBegin(SsrProfiler *profiler, SsrEvent event_id, PetscObject a, PetscObject b);
PetscErrorCode SsrProfilerEnd(SsrProfiler *profiler, SsrEvent event_id, PetscObject a, PetscObject b);
PetscErrorCode SsrProfilerStagePush(SsrStage stage_id);
PetscErrorCode SsrProfilerStagePop(SsrStage stage_id);
PetscErrorCode SsrProfileTimerBegin(SsrProfiler *profiler, SsrEvent event_id, PetscObject a, PetscObject b, SsrProfileTimer *timer);
PetscErrorCode SsrProfileTimerEnd(SsrProfiler *profiler, SsrProfileTimer *timer, PetscLogDouble *elapsed);

PetscErrorCode SsrStatsAccumulateElapsed(PetscLogDouble *counter, PetscLogDouble elapsed);
PetscErrorCode SsrStatsAddNewtonIteration(SsrStats *stats, const char phase[]);
PetscErrorCode SsrStatsAddLinearSolve(SsrStats *stats, PetscInt its, KSPConvergedReason reason);
PetscErrorCode SsrStatsAddLineSearchIteration(SsrStats *stats);
PetscErrorCode SsrStatsAddNewtonStepAssembly(SsrNewtonStepStats *stats, PetscLogDouble elapsed);
PetscErrorCode SsrStatsAddNewtonStepLinearSolve(SsrNewtonStepStats *stats, PetscInt its, PetscLogDouble elapsed);
PetscErrorCode SsrStatsAddNewtonStepIteration(SsrNewtonStepStats *stats);
PetscErrorCode SsrStatsAddNewtonStepLineSearch(SsrNewtonStepStats *stats, PetscInt its);
PetscErrorCode SsrStatsAcceptContinuationStep(SsrContinuationStats *stats, const SsrNewtonStepStats *step_stats);
PetscErrorCode SsrStatsAddHydroAssembly(SsrHydroStats *stats, PetscLogDouble elapsed);
PetscErrorCode SsrStatsAddHydroLinearSolve(SsrHydroStats *stats, PetscInt its, PetscLogDouble elapsed, KSPConvergedReason reason);
PetscErrorCode SsrStatsAddNeumannAssembly(SsrNeumannStats *stats, PetscInt rules, PetscInt faces, PetscInt quadrature_points, PetscLogDouble elapsed);

#define SSR_PROFILE_BEGIN(profiler, event_id, a, b) PetscCall(SsrProfilerBegin((profiler), (event_id), (PetscObject)(a), (PetscObject)(b)))
#define SSR_PROFILE_END(profiler, event_id, a, b) PetscCall(SsrProfilerEnd((profiler), (event_id), (PetscObject)(a), (PetscObject)(b)))
#define SSR_PROFILE_STAGE_PUSH(stage_id) PetscCall(SsrProfilerStagePush((stage_id)))
#define SSR_PROFILE_STAGE_POP(stage_id) PetscCall(SsrProfilerStagePop((stage_id)))
#define SSR_PROFILE_TIMER_BEGIN(profiler, event_id, a, b, timer) PetscCall(SsrProfileTimerBegin((profiler), (event_id), (PetscObject)(a), (PetscObject)(b), (timer)))
#define SSR_PROFILE_TIMER_END(profiler, timer, elapsed) PetscCall(SsrProfileTimerEnd((profiler), (timer), (elapsed)))

#endif
