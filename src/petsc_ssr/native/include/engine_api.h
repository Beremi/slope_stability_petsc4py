#ifndef P4_SSR_ENGINE_H
#define P4_SSR_ENGINE_H

#include <petscsys.h>

typedef struct P4SsrEngine P4SsrEngine;

typedef struct {
  PetscInt       ranks;
  PetscInt       global_dofs;
  PetscInt       basis_cols;
  PetscReal      rhs_norm;
  PetscLogDouble elastic_assembly_time;
  PetscLogDouble create_time;
  PetscLogDouble deflation_orthogonalization_time;
  PetscLogDouble deflation_coarse_initial_time;
  PetscLogDouble deflation_pc_apply_time;
  PetscLogDouble deflation_projector_time;
  PetscInt       deflation_coarse_initial_calls;
  PetscInt       deflation_projected_pc_calls;
} P4SsrEngineInfo;

typedef struct {
  PetscBool      converged;
  PetscBool      solved;
  PetscBool      failed;
  PetscBool      stop;
  PetscBool      compute_diffs_out;
  PetscReal      rel_residual;
  PetscReal      rel_correction;
  PetscReal      alpha;
  PetscReal      r_out;
  PetscReal      lambda_out;
  PetscReal      delta_lambda;
  PetscReal      trial_rel;
  PetscReal      abs_delta_lambda;
  PetscReal      initial_decrease;
  PetscInt       linear_its;
  PetscInt       linear_its_w;
  PetscInt       linear_its_v;
  PetscInt       line_search_its;
  PetscInt       newton_its;
  PetscLogDouble assembly_time;
  PetscLogDouble solve_time;
  PetscLogDouble wall_time;
} P4SsrStepResult;

PetscErrorCode P4IndirectSSRRunOptionsString(const char options[]);

PetscErrorCode P4SsrEngineCreateOptionsString(const char options[], P4SsrEngine **out);
PetscErrorCode P4SsrEngineDestroy(P4SsrEngine **ctxp);
PetscErrorCode P4SsrEngineGetInfo(P4SsrEngine *ctx, P4SsrEngineInfo *info);
PetscErrorCode P4SsrEngineBasisCols(P4SsrEngine *ctx, PetscInt *cols);
PetscErrorCode P4SsrEngineTruncateBasis(P4SsrEngine *ctx, PetscInt n_keep);
PetscErrorCode P4SsrEngineAppendBasisFromSlot(P4SsrEngine *ctx, PetscInt slot, const char label[]);

PetscErrorCode P4SsrEngineVecZero(P4SsrEngine *ctx, PetscInt slot);
PetscErrorCode P4SsrEngineVecCopy(P4SsrEngine *ctx, PetscInt src, PetscInt dst);
PetscErrorCode P4SsrEngineVecWAXPY(P4SsrEngine *ctx, PetscInt dst, PetscReal alpha, PetscInt x, PetscInt y);
PetscErrorCode P4SsrEngineVecAXPY(P4SsrEngine *ctx, PetscInt y, PetscReal alpha, PetscInt x);
PetscErrorCode P4SsrEngineDotOmega(P4SsrEngine *ctx, PetscInt slot, PetscReal *omega);
PetscErrorCode P4SsrEngineScaleToOmega(P4SsrEngine *ctx, PetscInt slot, PetscReal omega);
PetscErrorCode P4SsrEngineDisplacementMax(P4SsrEngine *ctx, PetscInt slot, PetscReal *u_max);
PetscErrorCode P4SsrEngineWriteSolutionFromSlot(P4SsrEngine *ctx, PetscInt slot);

PetscErrorCode P4SsrEngineAssembleResidualJacobian(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineComputeLambdaDerivative(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineBuildRegularizedOperator(P4SsrEngine *ctx, PetscReal r, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineBuildFixedCorrectionRHS(P4SsrEngine *ctx, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineBuildIndirectRHS(P4SsrEngine *ctx, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineKSPSetup(P4SsrEngine *ctx, PetscBool force_reuse_preconditioner, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineAOrthogonalize(P4SsrEngine *ctx, const char label[], P4SsrStepResult *out);
PetscErrorCode P4SsrEngineKSPSolveFixedCorrection(P4SsrEngine *ctx, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineKSPSolveIndirectW(P4SsrEngine *ctx, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineKSPSolveIndirectV(P4SsrEngine *ctx, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineFixedLineSearch(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineApplyFixedCorrection(P4SsrEngine *ctx, PetscInt slot, PetscReal alpha, PetscReal r_in, PetscBool update_basis, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineFormIndirectUpdate(P4SsrEngine *ctx, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineIndirectLineSearch(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda, PetscReal omega_target, PetscReal current_rel, PetscReal d_lambda, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineAcceptIndirectUpdate(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda_in, PetscReal omega_target, PetscReal alpha, PetscReal d_lambda, PetscReal r_in, PetscBool update_basis, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineResidualRel(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda, PetscReal *rel);
PetscErrorCode P4SsrEngineSolveElasticInitialGuess(P4SsrEngine *ctx, PetscInt slot, PetscReal scale, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineAssembleLimitLoad(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda_ell, PetscReal load_t, PetscReal r, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineBuildLimitLoadRHS(P4SsrEngine *ctx, PetscReal load_t, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineFormLimitLoadUpdate(P4SsrEngine *ctx, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineLimitLoadLineSearch(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda_ell, PetscReal load_t, P4SsrStepResult *out);
PetscErrorCode P4SsrEngineAcceptLimitLoadUpdate(P4SsrEngine *ctx, PetscInt slot, PetscReal load_t, PetscReal omega_target, PetscReal alpha, PetscReal d_t, PetscReal r_in, PetscBool update_basis, P4SsrStepResult *out);

#endif
