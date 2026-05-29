#pragma once

#include "p4_basis.h"
#include "petsc_ssr_stats.h"
#include <petscksp.h>

typedef struct {
  char     support_kind[32];
  char     support_name[128];
  char     dm_label[64];
  PetscInt tag;
  char     kind[32];
  char     geometry[64];
  PetscInt geometry_order;
  char     value_model[1024];
  char     value_model_name[64];
  char     native_status[64];
  PetscInt matched_points;
} AssemblyNeumannRule;

typedef struct {
  char     field[16];
  char     support_kind[32];
  char     support_name[128];
  char     dm_label[64];
  PetscInt tag;
  char     kind[32];
  char     geometry[64];
  PetscInt geometry_order;
  char     value_model[1024];
  char     native_status[64];
  PetscInt matched_points;
} AssemblySeepageBoundaryRule;

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
  PetscInt  neumann_rule_count;
  AssemblyNeumannRule *neumann_rules;
  SsrNeumannStats neumann_stats;
  PetscInt  seepage_boundary_rule_count;
  AssemblySeepageBoundaryRule *seepage_boundary_rules;
} AssemblyCtx;

typedef struct {
  PetscInt rows;
  PetscInt matched_points;
  PetscInt raw_constrained_rows;
  PetscInt owned_constraints;
  PetscInt global_constraints;
} AssemblyLabelConstraintStats;

typedef struct {
  PetscInt rows;
  PetscInt affine_rows;
  PetscInt curved_rows;
  PetscInt matched_points;
  PetscInt last_geometry_order;
  char     last_kind[32];
  char     last_geometry[64];
  char     last_value_model[64];
  char     last_native_status[64];
} AssemblyNeumannLabelStats;

typedef struct {
  PetscInt rows;
  PetscInt head_rows;
  PetscInt flux_rows;
  PetscInt matched_points;
} AssemblySeepageBoundaryLabelStats;

PetscErrorCode AssemblyCtxCreate(DM dm, P4Basis *basis, AssemblyCtx *ctx);
PetscErrorCode AssemblyCtxDestroy(AssemblyCtx *ctx);
PetscErrorCode AssemblyCtxLoadSeepagePressureCSV(AssemblyCtx *ctx, const char path[], PetscReal grho);
PetscErrorCode AssemblyCtxLoadLabelConstraintsCSV(AssemblyCtx *ctx, const char path[], PetscBool *loaded, AssemblyLabelConstraintStats *stats);
PetscErrorCode AssemblyCtxLoadNeumannLabelsCSV(AssemblyCtx *ctx, const char path[], AssemblyNeumannLabelStats *stats);
PetscErrorCode AssemblyCtxValidateNeumannLabelsCSV(AssemblyCtx *ctx, const char path[], PetscInt expected_rows);
PetscErrorCode AssemblyCtxAssembleNeumannResidual(AssemblyCtx *ctx, Vec rhs);
PetscErrorCode AssemblyCtxValidateSeepageBoundaryLabelsCSV(AssemblyCtx *ctx, const char path[], PetscInt expected_head_rows, PetscInt expected_flux_rows);
PetscErrorCode AssemblyCtxLoadCoordinateConstraintsCSV(AssemblyCtx *ctx, const char path[]);
PetscErrorCode BuildConstrainedGlobalIS(DM dm, IS *is, PetscInt *nlocal);
PetscErrorCode AssembleElasticProblem(AssemblyCtx *ctx, Mat A, Vec f_ext);
PetscErrorCode AssemblePlasticResidualJacobian(AssemblyCtx *ctx, PetscReal lambda, Vec u, Vec f_ext, Mat A, Vec residual, PetscBool assemble_jacobian);
PetscErrorCode ZeroConstrainedVector(IS is, Vec v);
PetscErrorCode ApplyZeroDirichlet(IS is, Mat A, Vec rhs);
PetscErrorCode AssemblyWriteDisplacementPointCSV(AssemblyCtx *ctx, Vec u, const char path[]);
