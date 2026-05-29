#include "petsc_ssr_algorithms.h"
#include "material_mc.h"

typedef struct {
  const char *name;
  PetscErrorCode (*solve)(DM, AssemblyCtx *, AppCtx *, LinearSolverCtx *, Mat, Mat, Mat, Vec, PetscReal, Vec, SSRStats *);
} SsrNativeContinuationOps;

typedef struct {
  DM               dm;
  AssemblyCtx     *actx;
  AppCtx          *app;
  LinearSolverCtx *solver;
  Mat              Areg;
  Mat              Kelastic;
  Mat              Ktangent;
  Vec              f_ext;
  Vec              u;
  PetscReal        lambda;
  PetscReal       *lambda_ref;
  PetscReal        omega_target;
  PetscReal        rhs_norm;
  PetscReal        initial_r;
  const char      *criterion;
  PetscReal        stop_tol;
} SsrNativeNewtonInput;

typedef struct {
  const char *name;
  PetscErrorCode (*solve)(const SsrNativeNewtonInput *, NewtonStats *);
} SsrNativeNewtonOps;

typedef struct {
  LinearSolverCtx *solver;
  KSP              ksp;
  Vec              rhs;
  Vec              x;
  const char      *label;
} SsrNativeDeflationInput;

typedef struct {
  PetscInt  iterations;
  PetscReal reported_rel;
} SsrNativeDeflationResult;

typedef struct {
  const char          *name;
  DeflationSolverType type;
  PetscErrorCode (*solve)(const SsrNativeDeflationInput *, SsrNativeDeflationResult *);
} SsrNativeDeflationOps;

struct _p_SsrLinearCtx {
  LinearSolverCtx        *solver;
  const SsrLinearOptions *options;
};

struct _p_SsrLinearResult {
  PetscInt iterations;
};

typedef struct {
  SsrLinearCtx     linear;
  Vec              rhs;
  Vec              x;
  Vec              current_u;
  const char      *label;
  PetscBool        nonlinear_tangent;
} SsrNativeLinearInput;

typedef struct {
  PetscInt iterations;
} SsrNativeLinearResult;

typedef struct {
  const char *name;
  PetscErrorCode (*solve)(const SsrNativeLinearInput *, SsrNativeLinearResult *);
} SsrNativeLinearOps;

static PetscErrorCode SSRContinuationSolve(DM, AssemblyCtx *, AppCtx *, LinearSolverCtx *, Mat, Mat, Mat, Vec, PetscReal, Vec, SSRStats *);
static PetscErrorCode SSRDirectContinuationSolve(DM, AssemblyCtx *, AppCtx *, LinearSolverCtx *, Mat, Mat, Mat, Vec, PetscReal, Vec, SSRStats *);
static PetscErrorCode FixedLambdaNewtonSolve(DM, AssemblyCtx *, AppCtx *, LinearSolverCtx *, Mat, Mat, Mat, Vec, Vec, PetscReal, PetscReal, const char[], PetscReal, NewtonStats *);
static PetscErrorCode IndirectNewtonSolve(DM, AssemblyCtx *, AppCtx *, LinearSolverCtx *, Mat, Mat, Mat, Vec, Vec, PetscReal *, PetscReal, PetscReal, PetscReal, const char[], PetscReal, NewtonStats *);
static PetscErrorCode SolveLinearSystem(LinearSolverCtx *, Vec, Vec, const char *, Vec, PetscBool, PetscInt *);
static PetscErrorCode DeflatedFGMRESSolve(LinearSolverCtx *, KSP, Vec, Vec, const char[], PetscInt *, PetscReal *);
static PetscErrorCode DeflatedMatlabDFGMRESSolve(LinearSolverCtx *, KSP, Vec, Vec, const char[], PetscInt *, PetscReal *);
static PetscErrorCode DeflatedCGSolve(LinearSolverCtx *, KSP, Vec, Vec, const char[], PetscInt *, PetscReal *);
PetscErrorCode        SsrNeumannConstantTractionEvaluate(SsrNeumannValueCtx, const SsrNeumannValueInput *, SsrNeumannValueResult *);

static PetscErrorCode SsrMaterialMohrCoulombEvaluate(SsrMaterialCtx ctx, const SsrMaterialPointInput *input, SsrMaterialPointResult *result)
{
  MaterialMC mat;

  PetscFunctionBeginUser;
  (void)ctx;
  PetscCheck(input, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Mohr-Coulomb material evaluation requires input");
  PetscCheck(result, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Mohr-Coulomb material evaluation requires result");
  PetscCheck(input->strain, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Mohr-Coulomb material evaluation requires strain input");
  PetscCheck(result->stress, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Mohr-Coulomb material evaluation requires stress output");
  PetscCheck(input->dim == 2 || input->dim == 3, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
             "Mohr-Coulomb material evaluation supports dim 2 or 3, got %" PetscInt_FMT, input->dim);
  PetscCall(MaterialMCFromRegion(input->region, &mat));
  if (input->dim == 2) {
    if (input->plastic) MaterialMCPlasticStressTangent2D(&mat, input->lambda, input->strain, result->stress, result->tangent);
    else MaterialMCElasticStressTangent2D(&mat, input->strain, result->stress, result->tangent);
  } else {
    if (input->plastic) MaterialMCPlasticStressTangent(&mat, input->lambda, input->strain, result->stress, result->tangent);
    else MaterialMCElasticStressTangent(&mat, input->strain, result->stress, result->tangent);
  }
  result->gamma_sat   = mat.gamma_sat;
  result->gamma_unsat = mat.gamma_unsat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SsrNativeFixedLoadNewtonSolve(const SsrNativeNewtonInput *input, NewtonStats *stats)
{
  PetscFunctionBeginUser;
  PetscCheck(input, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrNativeFixedLoadNewtonSolve requires input");
  PetscCheck(input->dm, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Fixed-load Newton requires a DM");
  PetscCheck(stats, PetscObjectComm((PetscObject)input->dm), PETSC_ERR_ARG_NULL, "Fixed-load Newton requires stats output");
  PetscCheck(input->u, PetscObjectComm((PetscObject)input->dm), PETSC_ERR_ARG_NULL, "Fixed-load Newton requires a displacement vector");
  PetscCall(FixedLambdaNewtonSolve(input->dm, input->actx, input->app, input->solver, input->Areg, input->Kelastic,
                                   input->Ktangent, input->f_ext, input->u, input->lambda, input->rhs_norm,
                                   input->criterion, input->stop_tol, stats));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SsrNativeIndirectNewtonSolve(const SsrNativeNewtonInput *input, NewtonStats *stats)
{
  PetscFunctionBeginUser;
  PetscCheck(input, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrNativeIndirectNewtonSolve requires input");
  PetscCheck(input->dm, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Indirect Newton requires a DM");
  PetscCheck(stats, PetscObjectComm((PetscObject)input->dm), PETSC_ERR_ARG_NULL, "Indirect Newton requires stats output");
  PetscCheck(input->lambda_ref, PetscObjectComm((PetscObject)input->dm), PETSC_ERR_ARG_NULL, "Indirect Newton requires a lambda reference");
  PetscCall(IndirectNewtonSolve(input->dm, input->actx, input->app, input->solver, input->Areg, input->Kelastic,
                                input->Ktangent, input->f_ext, input->u, input->lambda_ref, input->omega_target,
                                input->rhs_norm, input->initial_r, input->criterion, input->stop_tol, stats));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SsrNativeDeflatedFGMRESSolve(const SsrNativeDeflationInput *input, SsrNativeDeflationResult *result)
{
  PetscFunctionBeginUser;
  PetscCheck(input, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrNativeDeflatedFGMRESSolve requires input");
  PetscCheck(result, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrNativeDeflatedFGMRESSolve requires a result");
  result->iterations  = 0;
  result->reported_rel = PETSC_MAX_REAL;
  PetscCall(DeflatedFGMRESSolve(input->solver, input->ksp, input->rhs, input->x, input->label, &result->iterations, &result->reported_rel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SsrNativeDeflatedMatlabDFGMRESSolve(const SsrNativeDeflationInput *input, SsrNativeDeflationResult *result)
{
  PetscFunctionBeginUser;
  PetscCheck(input, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrNativeDeflatedMatlabDFGMRESSolve requires input");
  PetscCheck(result, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrNativeDeflatedMatlabDFGMRESSolve requires a result");
  result->iterations  = 0;
  result->reported_rel = PETSC_MAX_REAL;
  PetscCall(DeflatedMatlabDFGMRESSolve(input->solver, input->ksp, input->rhs, input->x, input->label, &result->iterations, &result->reported_rel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SsrNativeDeflatedCGSolve(const SsrNativeDeflationInput *input, SsrNativeDeflationResult *result)
{
  PetscFunctionBeginUser;
  PetscCheck(input, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrNativeDeflatedCGSolve requires input");
  PetscCheck(result, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrNativeDeflatedCGSolve requires a result");
  result->iterations  = 0;
  result->reported_rel = PETSC_MAX_REAL;
  PetscCall(DeflatedCGSolve(input->solver, input->ksp, input->rhs, input->x, input->label, &result->iterations, &result->reported_rel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SsrNativeDeflationSolve(const SsrNativeDeflationOps *ops, LinearSolverCtx *solver, KSP ksp, Vec rhs, Vec x, const char label[], PetscInt *its, PetscReal *reported_rel)
{
  SsrNativeDeflationInput  input = {0};
  SsrNativeDeflationResult result;

  PetscFunctionBeginUser;
  PetscCheck(ops, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native deflation solve requires resolved operations");
  PetscCheck(ops->solve, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native deflation operations for %s do not define solve", ops->name ? ops->name : "(unknown)");
  PetscCheck(solver, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native deflation solve requires a solver context");
  PetscCheck(ksp, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native deflation solve requires a KSP");
  PetscCheck(rhs, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native deflation solve requires an RHS vector");
  PetscCheck(x, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native deflation solve requires a solution vector");
  PetscCheck(its, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native deflation solve requires iteration output");
  PetscCheck(reported_rel, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native deflation solve requires relative residual output");
  input.solver       = solver;
  input.ksp          = ksp;
  input.rhs          = rhs;
  input.x            = x;
  input.label        = label;
  result.iterations  = 0;
  result.reported_rel = PETSC_MAX_REAL;
  PetscCall(ops->solve(&input, &result));
  *its          = result.iterations;
  *reported_rel = result.reported_rel;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SsrNativeLinearCtxBind(LinearSolverCtx *solver, const SsrLinearOptions *options, SsrLinearCtx linear)
{
  PetscFunctionBeginUser;
  PetscCheck(linear, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native linear context bind requires a context");
  PetscCheck(solver, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native linear context bind requires a solver context");
  linear->solver  = solver;
  linear->options = options;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SsrNativeKspLinearSolve(const SsrNativeLinearInput *input, SsrNativeLinearResult *result)
{
  PetscFunctionBeginUser;
  PetscCheck(input, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrNativeKspLinearSolve requires input");
  PetscCheck(input->linear, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native linear solve requires a typed linear context");
  PetscCheck(input->linear->solver, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native linear solve requires a solver context");
  PetscCheck(input->rhs, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native linear solve requires an RHS vector");
  PetscCheck(input->x, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native linear solve requires a solution vector");
  PetscCheck(result, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native linear solve requires a result");
  result->iterations = 0;
  PetscCall(SolveLinearSystem(input->linear->solver, input->rhs, input->x, input->label, input->current_u,
                              input->nonlinear_tangent, &result->iterations));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SsrNativeLinearSolve(const SsrNativeLinearOps *ops, SsrLinearCtx linear, Vec rhs, Vec x, const char label[], Vec current_u, PetscBool nonlinear_tangent, PetscInt *its)
{
  SsrNativeLinearInput  input = {0};
  SsrNativeLinearResult result;
  PetscBool             profile_match = PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCheck(ops, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native linear solve requires resolved operations");
  PetscCheck(ops->solve, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native linear operations for %s do not define solve", ops->name ? ops->name : "(unknown)");
  PetscCheck(linear, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native linear solve requires a typed linear context");
  PetscCheck(linear->solver, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native linear solve requires a solver context");
  PetscCheck(its, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Native linear solve requires iteration output");
  if (linear->options && linear->options->algorithm[0]) {
    PetscCall(PetscStrcasecmp(linear->options->algorithm, ops->name, &profile_match));
  }
  PetscCheck(profile_match, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
             "Resolved linear context algorithm %s does not match native linear operations %s",
             linear->options ? linear->options->algorithm : "(unset)", ops->name ? ops->name : "(unknown)");
  input.linear             = linear;
  input.rhs                = rhs;
  input.x                  = x;
  input.current_u          = current_u;
  input.label              = label;
  input.nonlinear_tangent  = nonlinear_tangent;
  result.iterations        = 0;
  PetscCall(ops->solve(&input, &result));
  *its = result.iterations;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static const SsrContinuationOps ssr_continuation_registry[] = {
  {"indirect", NULL, NULL, NULL},
  {"direct", NULL, NULL, NULL},
};

static const SsrNativeContinuationOps ssr_native_continuation_registry[] = {
  {"indirect", SSRContinuationSolve},
  {"direct", SSRDirectContinuationSolve},
};

static const SsrNativeNewtonOps ssr_native_newton_registry[] = {
  {"fixed-load", SsrNativeFixedLoadNewtonSolve},
  {"indirect-ssr", SsrNativeIndirectNewtonSolve},
};

static const SsrNativeDeflationOps ssr_native_deflation_registry[] = {
  {"fgmres", DEFLATION_SOLVER_FGMRES, SsrNativeDeflatedFGMRESSolve},
  {"matlab_dfgmres", DEFLATION_SOLVER_MATLAB_DFGMRES, SsrNativeDeflatedMatlabDFGMRESSolve},
  {"dfgmres", DEFLATION_SOLVER_MATLAB_DFGMRES, SsrNativeDeflatedMatlabDFGMRESSolve},
  {"cg", DEFLATION_SOLVER_CG, SsrNativeDeflatedCGSolve},
};

static const SsrNativeLinearOps ssr_native_linear_registry[] = {
  {"pmg-deflated", SsrNativeKspLinearSolve},
  {"pmg", SsrNativeKspLinearSolve},
  {"gamg", SsrNativeKspLinearSolve},
  {"bddc", SsrNativeKspLinearSolve},
  {"fetidp", SsrNativeKspLinearSolve},
  {"none", SsrNativeKspLinearSolve},
  {"debug-direct", SsrNativeKspLinearSolve},
};

static const SsrNewtonOps ssr_newton_registry[] = {
  {"fixed-load", NULL, NULL, NULL},
  {"indirect-ssr", NULL, NULL, NULL},
};

static const SsrLinearOps ssr_linear_registry[] = {
  {"pmg-deflated", NULL, NULL, NULL, NULL},
  {"pmg", NULL, NULL, NULL, NULL},
  {"gamg", NULL, NULL, NULL, NULL},
  {"bddc", NULL, NULL, NULL, NULL},
  {"fetidp", NULL, NULL, NULL, NULL},
  {"none", NULL, NULL, NULL, NULL},
  {"debug-direct", NULL, NULL, NULL, NULL},
};

static const SsrMaterialOps ssr_material_registry[] = {
  {"mohr_coulomb", NULL, SsrMaterialMohrCoulombEvaluate, NULL},
  {"mohr-coulomb", NULL, SsrMaterialMohrCoulombEvaluate, NULL},
  {"mohr_coulomb_ssr", NULL, SsrMaterialMohrCoulombEvaluate, NULL},
  {"mohr_coulomb_limit_load", NULL, SsrMaterialMohrCoulombEvaluate, NULL},
};

static const SsrNeumannValueOps ssr_neumann_value_registry[] = {
  {"constant", NULL, NULL, NULL},
  {"constant-traction", NULL, SsrNeumannConstantTractionEvaluate, NULL},
  {"normal-pressure", NULL, NULL, NULL},
  {"hydrostatic-pressure", NULL, NULL, NULL},
  {"piecewise-linear-head", NULL, NULL, NULL},
  {"table-on-boundary", NULL, NULL, NULL},
  {"function-pointer-debug", NULL, NULL, NULL},
};

PetscErrorCode SsrContinuationRegistryFind(const char name[], const SsrContinuationOps **ops)
{
  PetscFunctionBeginUser;
  PetscCheck(ops, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrContinuationRegistryFind requires an output ops pointer");
  PetscCheck(name && name[0], PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SSR continuation algorithm name is required");
  *ops = NULL;
  for (PetscInt i = 0; i < (PetscInt)(sizeof(ssr_continuation_registry) / sizeof(ssr_continuation_registry[0])); ++i) {
    PetscBool match = PETSC_FALSE;

    PetscCall(PetscStrcasecmp(name, ssr_continuation_registry[i].name, &match));
    if (match) {
      *ops = &ssr_continuation_registry[i];
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscCheck(*ops, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown SSR continuation algorithm %s", name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SsrNativeContinuationRegistryFind(const char name[], const SsrNativeContinuationOps **ops)
{
  PetscFunctionBeginUser;
  PetscCheck(ops, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrNativeContinuationRegistryFind requires an output ops pointer");
  PetscCheck(name && name[0], PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SSR native continuation algorithm name is required");
  *ops = NULL;
  for (PetscInt i = 0; i < (PetscInt)(sizeof(ssr_native_continuation_registry) / sizeof(ssr_native_continuation_registry[0])); ++i) {
    PetscBool match = PETSC_FALSE;

    PetscCall(PetscStrcasecmp(name, ssr_native_continuation_registry[i].name, &match));
    if (match) {
      *ops = &ssr_native_continuation_registry[i];
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscCheck(*ops, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown native SSR continuation algorithm %s", name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SsrNativeNewtonRegistryFind(const char name[], const SsrNativeNewtonOps **ops)
{
  PetscFunctionBeginUser;
  PetscCheck(ops, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrNativeNewtonRegistryFind requires an output ops pointer");
  PetscCheck(name && name[0], PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SSR native Newton algorithm name is required");
  *ops = NULL;
  for (PetscInt i = 0; i < (PetscInt)(sizeof(ssr_native_newton_registry) / sizeof(ssr_native_newton_registry[0])); ++i) {
    PetscBool match = PETSC_FALSE;

    PetscCall(PetscStrcasecmp(name, ssr_native_newton_registry[i].name, &match));
    if (match) {
      *ops = &ssr_native_newton_registry[i];
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscCheck(*ops, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown native SSR Newton algorithm %s", name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SsrNativeLinearRegistryFind(const char name[], const SsrNativeLinearOps **ops)
{
  PetscFunctionBeginUser;
  PetscCheck(ops, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrNativeLinearRegistryFind requires an output ops pointer");
  PetscCheck(name && name[0], PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SSR native linear algorithm name is required");
  *ops = NULL;
  for (PetscInt i = 0; i < (PetscInt)(sizeof(ssr_native_linear_registry) / sizeof(ssr_native_linear_registry[0])); ++i) {
    PetscBool match = PETSC_FALSE;

    PetscCall(PetscStrcasecmp(name, ssr_native_linear_registry[i].name, &match));
    if (match) {
      *ops = &ssr_native_linear_registry[i];
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscCheck(*ops, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown native SSR linear algorithm %s", name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SsrNativeDeflationRegistryFind(DeflationSolverType type, const SsrNativeDeflationOps **ops)
{
  PetscFunctionBeginUser;
  PetscCheck(ops, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrNativeDeflationRegistryFind requires an output ops pointer");
  *ops = NULL;
  for (PetscInt i = 0; i < (PetscInt)(sizeof(ssr_native_deflation_registry) / sizeof(ssr_native_deflation_registry[0])); ++i) {
    if (type == ssr_native_deflation_registry[i].type) {
      *ops = &ssr_native_deflation_registry[i];
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscCheck(*ops, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown native SSR deflation solver type %d", (int)type);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrNewtonRegistryFind(const char name[], const SsrNewtonOps **ops)
{
  PetscFunctionBeginUser;
  PetscCheck(ops, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrNewtonRegistryFind requires an output ops pointer");
  PetscCheck(name && name[0], PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SSR Newton algorithm name is required");
  *ops = NULL;
  for (PetscInt i = 0; i < (PetscInt)(sizeof(ssr_newton_registry) / sizeof(ssr_newton_registry[0])); ++i) {
    PetscBool match = PETSC_FALSE;

    PetscCall(PetscStrcasecmp(name, ssr_newton_registry[i].name, &match));
    if (match) {
      *ops = &ssr_newton_registry[i];
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscCheck(*ops, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown SSR Newton algorithm %s", name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrLinearRegistryFind(const char name[], const SsrLinearOps **ops)
{
  PetscFunctionBeginUser;
  PetscCheck(ops, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrLinearRegistryFind requires an output ops pointer");
  PetscCheck(name && name[0], PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SSR linear algorithm name is required");
  *ops = NULL;
  for (PetscInt i = 0; i < (PetscInt)(sizeof(ssr_linear_registry) / sizeof(ssr_linear_registry[0])); ++i) {
    PetscBool match = PETSC_FALSE;

    PetscCall(PetscStrcasecmp(name, ssr_linear_registry[i].name, &match));
    if (match) {
      *ops = &ssr_linear_registry[i];
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscCheck(*ops, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown SSR linear algorithm %s", name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrMaterialRegistryFind(const char name[], const SsrMaterialOps **ops)
{
  PetscFunctionBeginUser;
  PetscCheck(ops, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrMaterialRegistryFind requires an output ops pointer");
  PetscCheck(name && name[0], PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SSR material model name is required");
  *ops = NULL;
  for (PetscInt i = 0; i < (PetscInt)(sizeof(ssr_material_registry) / sizeof(ssr_material_registry[0])); ++i) {
    PetscBool match = PETSC_FALSE;

    PetscCall(PetscStrcasecmp(name, ssr_material_registry[i].name, &match));
    if (match) {
      *ops = &ssr_material_registry[i];
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscCheck(*ops, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown SSR material model %s", name);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrNeumannValueRegistryFind(const char name[], const SsrNeumannValueOps **ops)
{
  PetscFunctionBeginUser;
  PetscCheck(ops, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SsrNeumannValueRegistryFind requires an output ops pointer");
  PetscCheck(name && name[0], PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SSR Neumann value model name is required");
  *ops = NULL;
  for (PetscInt i = 0; i < (PetscInt)(sizeof(ssr_neumann_value_registry) / sizeof(ssr_neumann_value_registry[0])); ++i) {
    PetscBool match = PETSC_FALSE;

    PetscCall(PetscStrcasecmp(name, ssr_neumann_value_registry[i].name, &match));
    if (match) {
      *ops = &ssr_neumann_value_registry[i];
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscCheck(*ops, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown SSR Neumann value model %s", name);
  PetscFunctionReturn(PETSC_SUCCESS);
}
