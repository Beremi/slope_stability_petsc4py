#ifndef PETSC_SSR_ALGORITHMS_H
#define PETSC_SSR_ALGORITHMS_H

#include <petscksp.h>

typedef struct _p_SsrEngine *SsrEngine;
typedef struct _p_SsrContinuationCtx *SsrContinuationCtx;
typedef struct _p_SsrNewtonCtx *SsrNewtonCtx;
typedef struct _p_SsrLinearCtx *SsrLinearCtx;
typedef struct _p_SsrMaterialCtx *SsrMaterialCtx;
typedef struct _p_SsrNeumannValueCtx *SsrNeumannValueCtx;
typedef struct _p_SsrContinuationResult *SsrContinuationResult;
typedef struct _p_SsrNewtonInput *SsrNewtonInput;
typedef struct _p_SsrNewtonResult *SsrNewtonResult;
typedef struct _p_SsrLinearResult *SsrLinearResult;

typedef struct {
  PetscInt         dim;
  PetscInt         region;
  PetscBool        plastic;
  PetscReal        lambda;
  const PetscReal *strain;
} SsrMaterialPointInput;

typedef struct {
  PetscReal *stress;
  PetscReal *tangent;
  PetscReal  gamma_sat;
  PetscReal  gamma_unsat;
} SsrMaterialPointResult;

typedef struct {
  PetscInt         dim;
  PetscReal        time;
  const PetscReal *point;
  const PetscReal *normal;
} SsrNeumannValueInput;

typedef struct {
  PetscReal *traction;
} SsrNeumannValueResult;

typedef struct {
  const char *name;
  PetscErrorCode (*setup)(SsrEngine, SsrContinuationCtx *);
  PetscErrorCode (*run)(SsrContinuationCtx, SsrContinuationResult);
  PetscErrorCode (*destroy)(SsrContinuationCtx *);
} SsrContinuationOps;

typedef struct {
  const char *name;
  PetscErrorCode (*setup)(SsrEngine, SsrNewtonCtx *);
  PetscErrorCode (*solve)(SsrNewtonCtx, SsrNewtonInput, SsrNewtonResult);
  PetscErrorCode (*destroy)(SsrNewtonCtx *);
} SsrNewtonOps;

typedef struct {
  const char *name;
  PetscErrorCode (*setup)(SsrEngine, SsrLinearCtx *);
  PetscErrorCode (*solve)(SsrLinearCtx, Mat A, Vec b, Vec x, SsrLinearResult);
  PetscErrorCode (*recycle)(SsrLinearCtx, Vec update);
  PetscErrorCode (*destroy)(SsrLinearCtx *);
} SsrLinearOps;

typedef struct {
  const char *name;
  PetscErrorCode (*setup)(SsrEngine, SsrMaterialCtx *);
  PetscErrorCode (*evaluate)(SsrMaterialCtx, const SsrMaterialPointInput *, SsrMaterialPointResult *);
  PetscErrorCode (*destroy)(SsrMaterialCtx *);
} SsrMaterialOps;

typedef struct {
  const char *name;
  PetscErrorCode (*setup)(SsrEngine, SsrNeumannValueCtx *);
  PetscErrorCode (*evaluate)(SsrNeumannValueCtx, const SsrNeumannValueInput *, SsrNeumannValueResult *);
  PetscErrorCode (*destroy)(SsrNeumannValueCtx *);
} SsrNeumannValueOps;

PetscErrorCode SsrContinuationRegistryFind(const char name[], const SsrContinuationOps **ops);
PetscErrorCode SsrNewtonRegistryFind(const char name[], const SsrNewtonOps **ops);
PetscErrorCode SsrLinearRegistryFind(const char name[], const SsrLinearOps **ops);
PetscErrorCode SsrMaterialRegistryFind(const char name[], const SsrMaterialOps **ops);
PetscErrorCode SsrNeumannValueRegistryFind(const char name[], const SsrNeumannValueOps **ops);

#endif
