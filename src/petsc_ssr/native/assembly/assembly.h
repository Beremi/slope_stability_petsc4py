#pragma once

#include "p4_basis.h"
#include <petscksp.h>

typedef struct {
  DM       dm;
  P4Basis *basis;
  DMLabel  cell_sets;
  IS       constrained_is;
  IS       constrained_all_is;
  PetscInt n_constrained_local;
  PetscInt n_constrained_all;
  PetscInt *constrained_all;
  PetscInt cell_dofs;
  PetscBool seepage_enabled;
  PetscInt  seepage_npoints;
  void      *seepage_points; /* private sorted pressure point table */
  PetscReal *basis_ref_points;
  PetscReal seepage_grho;
  PetscReal seepage_tol;
} AssemblyCtx;

PetscErrorCode AssemblyCtxCreate(DM dm, P4Basis *basis, AssemblyCtx *ctx);
PetscErrorCode AssemblyCtxDestroy(AssemblyCtx *ctx);
PetscErrorCode AssemblyCtxLoadSeepagePressureCSV(AssemblyCtx *ctx, const char path[], PetscReal grho);
PetscErrorCode AssemblyCtxLoadCoordinateConstraintsCSV(AssemblyCtx *ctx, const char path[]);
PetscErrorCode BuildConstrainedGlobalIS(DM dm, IS *is, PetscInt *nlocal);
PetscErrorCode AssembleElasticProblem(AssemblyCtx *ctx, Mat A, Vec f_ext);
PetscErrorCode AssemblePlasticResidualJacobian(AssemblyCtx *ctx, PetscReal lambda, Vec u, Vec f_ext, Mat A, Vec residual, PetscBool assemble_jacobian);
PetscErrorCode ZeroConstrainedVector(IS is, Vec v);
PetscErrorCode ApplyZeroDirichlet(IS is, Mat A, Vec rhs);
PetscErrorCode AssemblyWriteDisplacementPointCSV(AssemblyCtx *ctx, Vec u, const char path[]);
