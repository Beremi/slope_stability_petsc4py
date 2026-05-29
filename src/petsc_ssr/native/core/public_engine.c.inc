#include "petsc_ssr_engine.h"

struct _p_SsrEngine {
  MPI_Comm          comm;
  SsrRuntimeProfile profile;
  char              continuation_algorithm[32];
  char              newton_algorithm[32];
  char              linear_algorithm[32];
};

static PetscErrorCode SsrEngineCopyAlgorithmName(MPI_Comm comm, char dest[], size_t n, const char source[], const char fallback[])
{
  const char *value = (source && source[0]) ? source : fallback;

  PetscFunctionBeginUser;
  PetscCheck(value && value[0], comm, PETSC_ERR_ARG_WRONG, "SSR engine algorithm selector must be non-empty");
  PetscCall(PetscStrncpy(dest, value, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrEngineCreate(MPI_Comm comm, const SsrRuntimeProfile *profile, SsrEngine *engine)
{
  SsrEngine ctx = NULL;

  PetscFunctionBeginUser;
  PetscCheck(engine, comm, PETSC_ERR_ARG_NULL, "SsrEngine output pointer is NULL");
  *engine = NULL;
  PetscCall(PetscNew(&ctx));
  ctx->comm = comm;
  if (profile) ctx->profile = *profile;
  PetscCall(SsrEngineCopyAlgorithmName(comm, ctx->continuation_algorithm, sizeof(ctx->continuation_algorithm),
                                       profile ? profile->continuation.algorithm : NULL, "indirect"));
  PetscCall(SsrEngineCopyAlgorithmName(comm, ctx->newton_algorithm, sizeof(ctx->newton_algorithm), profile ? profile->newton.algorithm : NULL,
                                       "indirect-ssr"));
  PetscCall(SsrEngineCopyAlgorithmName(comm, ctx->linear_algorithm, sizeof(ctx->linear_algorithm), profile ? profile->linear.algorithm : NULL,
                                       "pmg-deflated"));
  *engine = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrEngineSetContinuation(SsrEngine engine, const char name[])
{
  const SsrContinuationOps *ops = NULL;

  PetscFunctionBeginUser;
  PetscCheck(engine, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrEngine is NULL");
  PetscCall(SsrContinuationRegistryFind(name, &ops));
  PetscCall(SsrEngineCopyAlgorithmName(engine->comm, engine->continuation_algorithm, sizeof(engine->continuation_algorithm), name, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrEngineSetNewton(SsrEngine engine, const char name[])
{
  const SsrNewtonOps *ops = NULL;

  PetscFunctionBeginUser;
  PetscCheck(engine, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrEngine is NULL");
  PetscCall(SsrNewtonRegistryFind(name, &ops));
  PetscCall(SsrEngineCopyAlgorithmName(engine->comm, engine->newton_algorithm, sizeof(engine->newton_algorithm), name, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrEngineSetLinearSolver(SsrEngine engine, const char name[])
{
  const SsrLinearOps *ops = NULL;

  PetscFunctionBeginUser;
  PetscCheck(engine, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrEngine is NULL");
  PetscCall(SsrLinearRegistryFind(name, &ops));
  PetscCall(SsrEngineCopyAlgorithmName(engine->comm, engine->linear_algorithm, sizeof(engine->linear_algorithm), name, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrEngineRun(SsrEngine engine, SsrRunResult *result)
{
  PetscFunctionBeginUser;
  PetscCheck(engine, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrEngine is NULL");
  PetscCheck(result, engine->comm, PETSC_ERR_ARG_NULL, "SsrRunResult pointer is NULL");
  PetscCall(PetscMemzero(result, sizeof(*result)));
  SETERRQ(engine->comm, PETSC_ERR_SUP,
          "SsrEngineRun object lifecycle is a public native API scaffold; use PetscSsrEngineRunOptionsString until it is wired to the maintained unity backend");
}

PetscErrorCode SsrEngineDestroy(SsrEngine *engine)
{
  PetscFunctionBeginUser;
  if (!engine || !*engine) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFree(*engine));
  PetscFunctionReturn(PETSC_SUCCESS);
}
