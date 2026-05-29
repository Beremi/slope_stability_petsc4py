#ifndef PETSC_SSR_ENGINE_H
#define PETSC_SSR_ENGINE_H

#include "petsc_ssr_algorithms.h"
#include "petsc_ssr_problem.h"
#include "petsc_ssr_profile.h"
#include "petsc_ssr_stats.h"

#include <petscsys.h>

/*
   Public native engine facade.

   The production build still uses core/engine_main.c as a unity translation
   unit to preserve the current hot-path behavior. This header names the stable
   native engine surface for benchmark launches and keeps the fine-grained
   Cython debug bridge in engine_api.h out of the public case/solver contract.
*/

typedef struct {
  PetscBool      converged;
  PetscInt       accepted_steps;
  PetscInt       total_newton_its;
  PetscInt       total_linear_its;
  PetscInt       total_line_search_its;
  PetscReal      lambda_last;
  PetscReal      omega_last;
  PetscReal      final_rel;
  PetscLogDouble wall_time;
  char           summary_json[PETSC_MAX_PATH_LEN];
  char           curve_csv[PETSC_MAX_PATH_LEN];
} SsrRunResult;

PetscErrorCode SsrEngineCreate(MPI_Comm comm, const SsrRuntimeProfile *profile, SsrEngine *engine);
PetscErrorCode SsrEngineSetContinuation(SsrEngine engine, const char name[]);
PetscErrorCode SsrEngineSetNewton(SsrEngine engine, const char name[]);
PetscErrorCode SsrEngineSetLinearSolver(SsrEngine engine, const char name[]);
PetscErrorCode SsrEngineRun(SsrEngine engine, SsrRunResult *result);
PetscErrorCode SsrEngineDestroy(SsrEngine *engine);

PetscErrorCode P4IndirectSSRRunOptionsString(const char options[]);

/* Compatibility alias for callers that want the target PETSc-first name while
   the implementation symbol remains the proven maintained engine entry point. */
#define PetscSsrEngineRunOptionsString P4IndirectSSRRunOptionsString

#endif
