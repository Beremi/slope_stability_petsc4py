#ifndef PETSC_SSR_PROFILE_H
#define PETSC_SSR_PROFILE_H

#include <petscsys.h>

typedef struct {
  char     mesh[PETSC_MAX_PATH_LEN];
  PetscInt dim;
  PetscInt element_degree;
  PetscInt refine_levels;
  char     partitioner[32];
} SsrMeshOptions;

typedef struct {
  char      analysis[16];
  char      material_model[32];
  char      davis_type[8];
  PetscBool seepage_enabled;
} SsrPhysicsOptions;

typedef struct {
  char      algorithm[32];
  char      method[32];
  char      predictor[32];
  char      step_controller[32];
  PetscReal omega_max;
  PetscInt  step_max;
} SsrContinuationOptions;

typedef struct {
  char      algorithm[32];
  char      method[32];
  char      stopping_criterion[32];
  PetscReal stopping_tol;
  PetscInt  max_it;
  PetscInt  damp_max;
} SsrNewtonOptions;

typedef struct {
  char      apply_backend[32];
  char      shell_subcomm_type[32];
  char      smoother_ksp_type[32];
  char      smoother_pc_type[32];
  char      coarse_pc_type[32];
  char      coarse_ksp_type[32];
  char      coarse_inner_pc_type[32];
  char      coarse_telescope_subcomm_type[32];
  char      p2_telescope_subcomm_type[32];
  char      p2_telescope_ksp_type[32];
  char      p2_telescope_pc_type[32];
  PetscInt  p2_active_ranks;
  PetscInt  p1_active_ranks;
  PetscInt  lag_preconditioner;
  PetscInt  smoother_max_it;
  PetscInt  coarse_lu_max_dofs;
  PetscInt  coarse_telescope_active_ranks;
  PetscReal coarse_ksp_rtol;
  PetscInt  coarse_ksp_max_it;
  PetscInt  p2_telescope_active_ranks;
  PetscReal p2_telescope_ksp_rtol;
  PetscInt  p2_telescope_ksp_max_it;
  PetscInt  coarse_redundant_group_size;
  PetscBool coarse_gamg_aggressive_square_graph;
  PetscBool check_coarse_transfers;
} SsrPmgOptions;

typedef struct {
  PetscBool enabled;
  char      solver[32];
  char      projector[32];
  PetscReal basis_tol;
  PetscInt  max_it;
  PetscInt  max_vectors;
  PetscBool intra_newton_recycle;
  PetscBool krylov_persistent;
} SsrDeflationOptions;

typedef struct {
  char      profile[64];
  char      algorithm[32];
  char      pc_variant[32];
  char      requested_pc_variant[32];
  char      pc_variant_fallback_reason[64];
  char      ksp_type[32];
  PetscReal rtol;
  PetscInt  max_it;
  PetscBool reuse_preconditioner;
  PetscInt  pmg_p2_active_ranks;
  PetscInt  pmg_p1_active_ranks;
  SsrPmgOptions       pmg;
  SsrDeflationOptions deflation;
} SsrLinearOptions;

typedef struct {
  char      output_dir[PETSC_MAX_PATH_LEN];
  char      native_problem_manifest[PETSC_MAX_PATH_LEN];
  char      mechanics_bc_labels_csv[PETSC_MAX_PATH_LEN];
  char      mechanics_neumann_labels_csv[PETSC_MAX_PATH_LEN];
  char      seepage_boundary_labels_csv[PETSC_MAX_PATH_LEN];
  char      curve_csv[PETSC_MAX_PATH_LEN];
  char      summary_json[PETSC_MAX_PATH_LEN];
  PetscBool write_solution;
  PetscBool write_log_view;
} SsrOutputOptions;

typedef struct {
  SsrMeshOptions         mesh;
  SsrPhysicsOptions      physics;
  SsrContinuationOptions continuation;
  SsrNewtonOptions       newton;
  SsrLinearOptions       linear;
  SsrOutputOptions       output;
} SsrRuntimeProfile;

#endif
