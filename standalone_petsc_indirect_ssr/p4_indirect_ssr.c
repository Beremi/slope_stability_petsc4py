#include "assembly.h"
#include "p4_basis.h"

#include <petscblaslapack.h>
#include <petscdmplex.h>
#include <petscksp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static PetscLogStage log_stage_deflation_orthogonalize = -1;
static PetscLogStage log_stage_deflation_initial_guess = -1;
static PetscLogStage log_stage_deflation_projector     = -1;
static PetscLogStage log_stage_pmg_shell_fine_smooth   = -1;
static PetscLogStage log_stage_pmg_shell_residual      = -1;
static PetscLogStage log_stage_pmg_shell_transfer      = -1;
static PetscLogStage log_stage_pmg_shell_p2            = -1;
static PetscLogStage log_stage_pmg_shell_p1            = -1;

static PetscErrorCode RegisterLogStages(void)
{
  PetscFunctionBeginUser;
  PetscCall(PetscLogStageRegister("deflation_orthogonalize", &log_stage_deflation_orthogonalize));
  PetscCall(PetscLogStageRegister("deflation_initial_guess", &log_stage_deflation_initial_guess));
  PetscCall(PetscLogStageRegister("deflation_projector", &log_stage_deflation_projector));
  PetscCall(PetscLogStageRegister("pmg_shell_fine_smooth", &log_stage_pmg_shell_fine_smooth));
  PetscCall(PetscLogStageRegister("pmg_shell_residual", &log_stage_pmg_shell_residual));
  PetscCall(PetscLogStageRegister("pmg_shell_transfer", &log_stage_pmg_shell_transfer));
  PetscCall(PetscLogStageRegister("pmg_shell_p2", &log_stage_pmg_shell_p2));
  PetscCall(PetscLogStageRegister("pmg_shell_p1", &log_stage_pmg_shell_p1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef enum {
  VARIANT_GAMG,
  VARIANT_BDDC,
  VARIANT_FETIDP,
  VARIANT_PMG,
  VARIANT_NONE
} PCVariant;

typedef enum {
  DEFLATION_SOLVER_FGMRES,
  DEFLATION_SOLVER_MATLAB_DFGMRES,
  DEFLATION_SOLVER_CG
} DeflationSolverType;

typedef enum {
  DEFLATION_PROJECTOR_A_ORTHONORMAL,
  DEFLATION_PROJECTOR_BIORTHOGONAL
} DeflationProjectorType;

typedef struct {
  char      mesh[PETSC_MAX_PATH_LEN];
  PetscInt  refine_levels;
  PetscReal newton_rtol;
  PetscInt  newton_max_it;
  PetscReal omega_max;
  PetscReal lambda_init;
  PetscReal d_lambda_init;
  PetscReal d_lambda_min;
  PetscReal d_lambda_diff_scaled_min;
  PetscInt  continuation_step_max;
  PetscReal newton_stopping_tol;
  PetscReal init_newton_stopping_tol;
  PetscInt  it_damp_max;
  PetscReal r_min;
  char      newton_stopping_criterion[32];
  char      init_newton_stopping_criterion[32];
  char      continuation_predictor[32];
  char      omega_step_controller[32];
  char      curve_csv[PETSC_MAX_PATH_LEN];
  PetscReal ksp_rtol;
  PetscReal damping_min;
  PetscBool line_search;
  PetscBool use_box_mesh;
  PCVariant variant;
  char      variant_name[32];
  char      mesh_bc_mode[32];
  char      pmg_coarse_pc_type[32];
  char      pmg_smoother_ksp_type[32];
  char      pmg_smoother_pc_type[32];
  char      pmg_coarse_telescope_subcomm_type[32];
  char      pmg_coarse_telescope_ksp_type[32];
  char      pmg_coarse_telescope_pc_type[32];
  char      pmg_p2_telescope_subcomm_type[32];
  char      pmg_p2_telescope_ksp_type[32];
  char      pmg_p2_telescope_pc_type[32];
  char      pmg_apply_backend[32];
  char      pmg_shell_subcomm_type[32];
  PetscReal pmg_coarse_telescope_ksp_rtol;
  PetscReal pmg_p2_telescope_ksp_rtol;
  PetscInt  pmg_coarse_lu_max_dofs;
  PetscInt  pmg_smoother_max_it;
  PetscInt  pmg_coarse_redundant_group_size;
  PetscInt  pmg_coarse_telescope_active_ranks;
  PetscInt  pmg_coarse_telescope_ksp_max_it;
  PetscInt  pmg_p2_telescope_active_ranks;
  PetscInt  pmg_p2_telescope_ksp_max_it;
  PetscInt  pmg_shell_p2_active_ranks;
  PetscInt  pmg_shell_p1_active_ranks;
  PetscInt  pmg_lag_preconditioner;
  PetscBool pmg_coarse_gamg_aggressive_square_graph;
  PetscBool pmg_check_coarse_transfers;
  char      bddc_graph[32];
  char      bddc_coordinates[32];
  PetscBool bddc_collapse_shared;
  PetscBool bddc_local_solver_auto;
  PetscBool bddc_use_local_dirichlet;
  PetscInt  bddc_exact_local_max_dofs;
  PetscBool debug_bddc_dirichlet_rows;
  PetscBool inspect_partition;
  PetscBool reuse_linear_solver;
  PetscBool use_deflation;
  PetscBool indirect_newton_pair_freeze_matrix;
  char      deflation_solver_name[32];
  DeflationSolverType deflation_solver;
  char      deflation_projector_name[32];
  DeflationProjectorType deflation_projector;
  PetscReal deflation_basis_tol;
  PetscReal deflation_biorthogonal_pivot_tol;
  PetscInt  deflation_max_it;
  PetscInt  deflation_max_vectors;
  PetscBool deflation_monitor;
  PetscBool deflation_intra_newton_recycle;
  PetscInt  deflation_recycle_max_vectors;
  PetscReal deflation_recycle_basis_tol;
  PetscBool deflation_krylov_persistent;
  PetscReal deflation_krylov_basis_tol;
  PetscBool deflation_check_orthonormality;
  PetscReal deflation_orthonormality_warn_tol;
  PetscInt  deflation_reorthogonalize_sweeps;
  char      linear_replay_dir[PETSC_MAX_PATH_LEN];
  PetscBool linear_replay_use_exported_rhs;
  PetscBool linear_replay_check_pc_probe;
  char      step_replay_dir[PETSC_MAX_PATH_LEN];
  char      init_replay_dir[PETSC_MAX_PATH_LEN];
  PetscBool init_replay_use_exported_matrix;
  PetscBool init_replay_use_exported_rhs;
  PetscBool init_replay_use_exported_u;
  PetscBool init_replay_check_damping;
} AppCtx;

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
} NewtonStats;

typedef struct {
  DM           dm;
  AssemblyCtx *actx;
  AppCtx      *app;
  Mat          A;
  KSP          ksp;
  PetscBool    reuse;
  Vec         *raw_basis;
  PetscReal   *raw_basis_tol;
  Vec         *orth_basis;
  Vec         *left_basis;
  Vec         *Aorth_basis;
  PetscInt     n_raw_basis;
  PetscInt     raw_basis_cap;
  PetscInt     n_orth_basis;
  PetscInt     orth_basis_cap;
  PetscLogDouble deflation_orthogonalization_time;
  PetscLogDouble deflation_coarse_time;
  PetscLogDouble deflation_pc_apply_time;
  PetscLogDouble deflation_projector_time;
  PetscInt       deflation_coarse_calls;
  PetscInt       deflation_projected_pc_calls;
  Vec           *recycle_basis;
  PetscInt       n_recycle_basis;
  PetscInt       recycle_basis_cap;
  PetscBool      capture_recycle_basis;
  PetscBool      recycle_temp_basis_active;
  PetscInt       recycle_temp_start_raw;
  PetscBool      force_reuse_preconditioner;
  PetscInt       deflation_krylov_persistent_added;
  PetscInt       pmg_lag_solve_index;
  PetscBool      pmg_hierarchy_initialized;
  PetscBool      pmg_p1_basis_created;
  PetscBool      pmg_p2_basis_created;
  P4Basis        pmg_p1_basis;
  P4Basis        pmg_p2_basis;
  DM             pmg_dm_p1;
  DM             pmg_dm_p2;
  Mat            pmg_inject_p4_to_p2;
  Mat            pmg_inject_p2_to_p1;
  PetscBool      pmg_inject_p4_to_p2_transpose;
  PetscBool      pmg_inject_p2_to_p1_transpose;
  Vec            pmg_u_p1;
  Vec            pmg_u_p2;
} LinearSolverCtx;

static PetscErrorCode PMGApplyBackendIsShell(const AppCtx *app, PetscBool *is_shell)
{
  PetscFunctionBeginUser;
  PetscCall(PetscStrcasecmp(app->pmg_apply_backend, "shell_vcycle", is_shell));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RigidTx(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  (void)dim; (void)time; (void)x; (void)Nc; (void)ctx;
  u[0] = 1.0; u[1] = 0.0; u[2] = 0.0;
  return PETSC_SUCCESS;
}
static PetscErrorCode RigidTy(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  (void)dim; (void)time; (void)x; (void)Nc; (void)ctx;
  u[0] = 0.0; u[1] = 1.0; u[2] = 0.0;
  return PETSC_SUCCESS;
}
static PetscErrorCode RigidTz(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  (void)dim; (void)time; (void)x; (void)Nc; (void)ctx;
  u[0] = 0.0; u[1] = 0.0; u[2] = 1.0;
  return PETSC_SUCCESS;
}
static PetscErrorCode RigidRx(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  (void)dim; (void)time; (void)Nc; (void)ctx;
  u[0] = 0.0; u[1] = -x[2]; u[2] = x[1];
  return PETSC_SUCCESS;
}
static PetscErrorCode RigidRy(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  (void)dim; (void)time; (void)Nc; (void)ctx;
  u[0] = x[2]; u[1] = 0.0; u[2] = -x[0];
  return PETSC_SUCCESS;
}
static PetscErrorCode RigidRz(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  (void)dim; (void)time; (void)Nc; (void)ctx;
  u[0] = -x[1]; u[1] = x[0]; u[2] = 0.0;
  return PETSC_SUCCESS;
}

static PetscErrorCode ZeroDisplacement(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  (void)dim; (void)time; (void)x; (void)ctx;
  for (PetscInt c = 0; c < Nc; ++c) u[c] = 0.0;
  return PETSC_SUCCESS;
}

static PetscErrorCode ParseOptions(MPI_Comm comm, AppCtx *app)
{
  PetscBool flg;

  PetscFunctionBeginUser;
  PetscCall(PetscStrncpy(app->mesh, "data/adaptive_family_a_l1.msh", sizeof(app->mesh)));
  app->refine_levels  = 0;
  app->newton_rtol    = 1.0e-4;
  app->newton_max_it  = 200;
  app->omega_max      = 6.7e6;
  app->lambda_init    = 1.0;
  app->d_lambda_init  = 0.1;
  app->d_lambda_min   = 1.0e-3;
  app->d_lambda_diff_scaled_min = 1.0e-3;
  app->continuation_step_max    = 100;
  app->newton_stopping_tol      = 1.0e-4;
  app->init_newton_stopping_tol = 1.0e-3;
  app->it_damp_max   = 10;
  app->r_min         = 1.0e-4;
  PetscCall(PetscStrncpy(app->newton_stopping_criterion, "absolute_delta_lambda", sizeof(app->newton_stopping_criterion)));
  PetscCall(PetscStrncpy(app->init_newton_stopping_criterion, "relative_correction", sizeof(app->init_newton_stopping_criterion)));
  PetscCall(PetscStrncpy(app->continuation_predictor, "secant", sizeof(app->continuation_predictor)));
  PetscCall(PetscStrncpy(app->omega_step_controller, "legacy", sizeof(app->omega_step_controller)));
  PetscCall(PetscStrncpy(app->curve_csv, "continuation_curve.csv", sizeof(app->curve_csv)));
  app->ksp_rtol       = 1.0e-1;
  app->damping_min    = 1.0e-3;
  app->line_search    = PETSC_TRUE;
  app->use_box_mesh    = PETSC_FALSE;
  app->variant        = VARIANT_PMG;
  PetscCall(PetscStrncpy(app->variant_name, "pmg", sizeof(app->variant_name)));
  PetscCall(PetscStrncpy(app->mesh_bc_mode, "rollers", sizeof(app->mesh_bc_mode)));
  PetscCall(PetscStrncpy(app->pmg_coarse_pc_type, "auto", sizeof(app->pmg_coarse_pc_type)));
  PetscCall(PetscStrncpy(app->pmg_smoother_ksp_type, "chebyshev", sizeof(app->pmg_smoother_ksp_type)));
  PetscCall(PetscStrncpy(app->pmg_smoother_pc_type, "jacobi", sizeof(app->pmg_smoother_pc_type)));
  PetscCall(PetscStrncpy(app->pmg_coarse_telescope_subcomm_type, "interlaced", sizeof(app->pmg_coarse_telescope_subcomm_type)));
  PetscCall(PetscStrncpy(app->pmg_coarse_telescope_ksp_type, "fgmres", sizeof(app->pmg_coarse_telescope_ksp_type)));
  PetscCall(PetscStrncpy(app->pmg_coarse_telescope_pc_type, "gamg", sizeof(app->pmg_coarse_telescope_pc_type)));
  PetscCall(PetscStrncpy(app->pmg_p2_telescope_subcomm_type, "interlaced", sizeof(app->pmg_p2_telescope_subcomm_type)));
  PetscCall(PetscStrncpy(app->pmg_p2_telescope_ksp_type, "fgmres", sizeof(app->pmg_p2_telescope_ksp_type)));
  PetscCall(PetscStrncpy(app->pmg_p2_telescope_pc_type, "jacobi", sizeof(app->pmg_p2_telescope_pc_type)));
  PetscCall(PetscStrncpy(app->pmg_apply_backend, "shell_vcycle", sizeof(app->pmg_apply_backend)));
  PetscCall(PetscStrncpy(app->pmg_shell_subcomm_type, "interlaced", sizeof(app->pmg_shell_subcomm_type)));
  app->pmg_coarse_telescope_ksp_rtol   = 1.0e-3;
  app->pmg_p2_telescope_ksp_rtol       = 1.0e-3;
  app->pmg_coarse_lu_max_dofs = 50000;
  app->pmg_smoother_max_it    = 2;
  app->pmg_coarse_redundant_group_size         = 16;
  app->pmg_coarse_telescope_active_ranks       = 0;
  app->pmg_coarse_telescope_ksp_max_it         = 100;
  app->pmg_p2_telescope_active_ranks           = 0;
  app->pmg_p2_telescope_ksp_max_it             = 50;
  app->pmg_shell_p2_active_ranks                = 64;
  app->pmg_shell_p1_active_ranks                = 32;
  app->pmg_lag_preconditioner                  = 1;
  app->pmg_coarse_gamg_aggressive_square_graph = PETSC_FALSE;
  app->pmg_check_coarse_transfers              = PETSC_FALSE;
  PetscCall(PetscStrncpy(app->bddc_graph, "petsc", sizeof(app->bddc_graph)));
  PetscCall(PetscStrncpy(app->bddc_coordinates, "scalar", sizeof(app->bddc_coordinates)));
  app->bddc_collapse_shared    = PETSC_FALSE;
  app->bddc_local_solver_auto    = PETSC_FALSE;
  app->bddc_use_local_dirichlet  = PETSC_FALSE;
  app->bddc_exact_local_max_dofs = 8000;
  app->debug_bddc_dirichlet_rows = PETSC_FALSE;
  app->inspect_partition         = PETSC_FALSE;
  app->reuse_linear_solver       = PETSC_TRUE;
  app->use_deflation             = PETSC_TRUE;
  app->indirect_newton_pair_freeze_matrix = PETSC_FALSE;
  PetscCall(PetscStrncpy(app->deflation_solver_name, "fgmres", sizeof(app->deflation_solver_name)));
  app->deflation_solver      = DEFLATION_SOLVER_FGMRES;
  PetscCall(PetscStrncpy(app->deflation_projector_name, "a_orthonormal", sizeof(app->deflation_projector_name)));
  app->deflation_projector   = DEFLATION_PROJECTOR_A_ORTHONORMAL;
  app->deflation_basis_tol   = 1.0e-3;
  app->deflation_biorthogonal_pivot_tol = 1.0e-10;
  app->deflation_max_it      = 0;
  app->deflation_max_vectors = 0;
  app->deflation_monitor     = PETSC_FALSE;
  app->deflation_intra_newton_recycle = PETSC_FALSE;
  app->deflation_recycle_max_vectors  = 0;
  app->deflation_recycle_basis_tol     = 1.0e-30;
  app->deflation_krylov_persistent     = PETSC_FALSE;
  app->deflation_krylov_basis_tol      = 0.0;
  app->deflation_check_orthonormality  = PETSC_FALSE;
  app->deflation_orthonormality_warn_tol = 1.0e-8;
  app->deflation_reorthogonalize_sweeps = 0;
  app->linear_replay_dir[0]  = '\0';
  app->linear_replay_use_exported_rhs = PETSC_TRUE;
  app->linear_replay_check_pc_probe   = PETSC_TRUE;
  app->step_replay_dir[0]            = '\0';
  app->init_replay_dir[0]            = '\0';
  app->init_replay_use_exported_matrix = PETSC_TRUE;
  app->init_replay_use_exported_rhs    = PETSC_TRUE;
  app->init_replay_use_exported_u      = PETSC_TRUE;
  app->init_replay_check_damping       = PETSC_TRUE;

  PetscOptionsBegin(comm, NULL, "Standalone P4 indirect SSR options", NULL);
  PetscCall(PetscOptionsString("-mesh", "Gmsh mesh path", NULL, app->mesh, app->mesh, sizeof(app->mesh), NULL));
  PetscCall(PetscOptionsInt("-refine_levels", "Uniform DMPlex refinement levels", NULL, app->refine_levels, &app->refine_levels, NULL));
  PetscCall(PetscOptionsReal("-omega_max", "Maximum continuation work omega=f_ext^T u", NULL, app->omega_max, &app->omega_max, NULL));
  PetscCall(PetscOptionsReal("-lambda_init", "Initial strength reduction factor seed", NULL, app->lambda_init, &app->lambda_init, NULL));
  PetscCall(PetscOptionsReal("-d_lambda_init", "Initial lambda increment for continuation seeding", NULL, app->d_lambda_init, &app->d_lambda_init, NULL));
  PetscCall(PetscOptionsReal("-d_lambda_min", "Minimum lambda increment during initialization backoff", NULL, app->d_lambda_min, &app->d_lambda_min, NULL));
  PetscCall(PetscOptionsReal("-d_lambda_diff_scaled_min", "Stop when scaled lambda slope difference drops below this value", NULL, app->d_lambda_diff_scaled_min, &app->d_lambda_diff_scaled_min, NULL));
  PetscCall(PetscOptionsInt("-continuation_step_max", "Maximum accepted continuation points including initialization", NULL, app->continuation_step_max, &app->continuation_step_max, NULL));
  PetscCall(PetscOptionsString("-continuation_predictor", "Continuation predictor, currently only secant", NULL, app->continuation_predictor, app->continuation_predictor, sizeof(app->continuation_predictor), NULL));
  PetscCall(PetscOptionsString("-omega_step_controller", "Omega step controller, currently only legacy", NULL, app->omega_step_controller, app->omega_step_controller, sizeof(app->omega_step_controller), NULL));
  PetscCall(PetscOptionsString("-curve_csv", "Rank-0 CSV output path for accepted continuation curve", NULL, app->curve_csv, app->curve_csv, sizeof(app->curve_csv), NULL));
  PetscCall(PetscOptionsReal("-newton_rtol", "Relative residual tolerance used for nonlinear residual safeguards", NULL, app->newton_rtol, &app->newton_rtol, NULL));
  PetscCall(PetscOptionsInt("-newton_max_it", "Maximum fixed-lambda or indirect Newton iterations", NULL, app->newton_max_it, &app->newton_max_it, NULL));
  PetscCall(PetscOptionsString("-newton_stopping_criterion", "Indirect Newton stop: absolute_delta_lambda|relative_residual|relative_correction", NULL, app->newton_stopping_criterion, app->newton_stopping_criterion, sizeof(app->newton_stopping_criterion), NULL));
  PetscCall(PetscOptionsReal("-newton_stopping_tol", "Indirect Newton stopping tolerance for the selected criterion", NULL, app->newton_stopping_tol, &app->newton_stopping_tol, NULL));
  PetscCall(PetscOptionsString("-init_newton_stopping_criterion", "Initialization Newton stop: absolute_delta_lambda|relative_residual|relative_correction", NULL, app->init_newton_stopping_criterion, app->init_newton_stopping_criterion, sizeof(app->init_newton_stopping_criterion), NULL));
  PetscCall(PetscOptionsReal("-init_newton_stopping_tol", "Initialization Newton stopping tolerance for the selected criterion", NULL, app->init_newton_stopping_tol, &app->init_newton_stopping_tol, NULL));
  PetscCall(PetscOptionsInt("-it_damp_max", "Maximum ALG5/backtracking halvings per Newton iteration", NULL, app->it_damp_max, &app->it_damp_max, NULL));
  PetscCall(PetscOptionsReal("-r_min", "Minimum regularization weight in K_r=r*K_elastic+(1-r)*K_tangent", NULL, app->r_min, &app->r_min, NULL));
  PetscCall(PetscOptionsReal("-linear_rtol", "Default KSP relative tolerance", NULL, app->ksp_rtol, &app->ksp_rtol, NULL));
  PetscCall(PetscOptionsBool("-line_search", "Use residual backtracking", NULL, app->line_search, &app->line_search, NULL));
  PetscCall(PetscOptionsBool("-use_box_mesh", "Use a tiny generated unit-box tetra mesh for smoke tests", NULL, app->use_box_mesh, &app->use_box_mesh, NULL));
  PetscCall(PetscOptionsString("-mesh_bc_mode", "Boundary mode for imported meshes: rollers|base_only|full_sides", NULL, app->mesh_bc_mode, app->mesh_bc_mode, sizeof(app->mesh_bc_mode), NULL));
  PetscCall(PetscOptionsReal("-damping_min", "Minimum backtracking damping", NULL, app->damping_min, &app->damping_min, NULL));
  PetscCall(PetscOptionsString("-pc_variant", "gamg|bddc|fetidp|pmg|none", NULL, app->variant_name, app->variant_name, sizeof(app->variant_name), NULL));
  PetscCall(PetscOptionsString("-pmg_coarse_pc_type", "auto|hypre|gamg|lu", NULL, app->pmg_coarse_pc_type, app->pmg_coarse_pc_type, sizeof(app->pmg_coarse_pc_type), NULL));
  PetscCall(PetscOptionsInt("-pmg_coarse_lu_max_dofs", "Maximum P1 coarse-grid DOFs allowed for LU", NULL, app->pmg_coarse_lu_max_dofs, &app->pmg_coarse_lu_max_dofs, NULL));
  PetscCall(PetscOptionsInt("-pmg_coarse_redundant_group_size", "Use PCREDUNDANT for the PMG P1 coarse solve on groups of this many ranks; 0 disables", NULL, app->pmg_coarse_redundant_group_size, &app->pmg_coarse_redundant_group_size, NULL));
  PetscCall(PetscOptionsBool("-pmg_coarse_gamg_aggressive_square_graph", "Use PETSc GAMG square-graph aggressive coarsening on PMG P1 coarse solves", NULL, app->pmg_coarse_gamg_aggressive_square_graph, &app->pmg_coarse_gamg_aggressive_square_graph, NULL));
  PetscCall(PetscOptionsInt("-pmg_coarse_telescope_active_ranks", "If positive, use PCTELESCOPE for the PMG P1 solve when ranks are an integer multiple larger than this", NULL, app->pmg_coarse_telescope_active_ranks, &app->pmg_coarse_telescope_active_ranks, NULL));
  PetscCall(PetscOptionsString("-pmg_coarse_telescope_subcomm_type", "PCTELESCOPE subcomm type, usually interlaced or contiguous", NULL, app->pmg_coarse_telescope_subcomm_type, app->pmg_coarse_telescope_subcomm_type, sizeof(app->pmg_coarse_telescope_subcomm_type), NULL));
  PetscCall(PetscOptionsString("-pmg_coarse_telescope_ksp_type", "KSP type inside PMG P1 PCTELESCOPE", NULL, app->pmg_coarse_telescope_ksp_type, app->pmg_coarse_telescope_ksp_type, sizeof(app->pmg_coarse_telescope_ksp_type), NULL));
  PetscCall(PetscOptionsReal("-pmg_coarse_telescope_ksp_rtol", "KSP relative tolerance inside PMG P1 PCTELESCOPE", NULL, app->pmg_coarse_telescope_ksp_rtol, &app->pmg_coarse_telescope_ksp_rtol, NULL));
  PetscCall(PetscOptionsInt("-pmg_coarse_telescope_ksp_max_it", "KSP max iterations inside PMG P1 PCTELESCOPE", NULL, app->pmg_coarse_telescope_ksp_max_it, &app->pmg_coarse_telescope_ksp_max_it, NULL));
  PetscCall(PetscOptionsString("-pmg_coarse_telescope_pc_type", "PC type inside PMG P1 PCTELESCOPE", NULL, app->pmg_coarse_telescope_pc_type, app->pmg_coarse_telescope_pc_type, sizeof(app->pmg_coarse_telescope_pc_type), NULL));
  PetscCall(PetscOptionsInt("-pmg_p2_telescope_active_ranks", "If positive, use PCTELESCOPE for the PMG P2-level smoother PC when ranks are an integer multiple larger than this", NULL, app->pmg_p2_telescope_active_ranks, &app->pmg_p2_telescope_active_ranks, NULL));
  PetscCall(PetscOptionsString("-pmg_p2_telescope_subcomm_type", "PCTELESCOPE subcomm type for the PMG P2-level smoother PC", NULL, app->pmg_p2_telescope_subcomm_type, app->pmg_p2_telescope_subcomm_type, sizeof(app->pmg_p2_telescope_subcomm_type), NULL));
  PetscCall(PetscOptionsString("-pmg_p2_telescope_ksp_type", "KSP type inside PMG P2-level PCTELESCOPE", NULL, app->pmg_p2_telescope_ksp_type, app->pmg_p2_telescope_ksp_type, sizeof(app->pmg_p2_telescope_ksp_type), NULL));
  PetscCall(PetscOptionsReal("-pmg_p2_telescope_ksp_rtol", "KSP relative tolerance inside PMG P2-level PCTELESCOPE", NULL, app->pmg_p2_telescope_ksp_rtol, &app->pmg_p2_telescope_ksp_rtol, NULL));
  PetscCall(PetscOptionsInt("-pmg_p2_telescope_ksp_max_it", "KSP max iterations inside PMG P2-level PCTELESCOPE", NULL, app->pmg_p2_telescope_ksp_max_it, &app->pmg_p2_telescope_ksp_max_it, NULL));
  PetscCall(PetscOptionsString("-pmg_p2_telescope_pc_type", "PC type inside PMG P2-level PCTELESCOPE", NULL, app->pmg_p2_telescope_pc_type, app->pmg_p2_telescope_pc_type, sizeof(app->pmg_p2_telescope_pc_type), NULL));
  PetscCall(PetscOptionsString("-pmg_smoother_ksp_type", "PMG smoother KSP type", NULL, app->pmg_smoother_ksp_type, app->pmg_smoother_ksp_type, sizeof(app->pmg_smoother_ksp_type), NULL));
  PetscCall(PetscOptionsString("-pmg_smoother_pc_type", "PMG smoother PC type", NULL, app->pmg_smoother_pc_type, app->pmg_smoother_pc_type, sizeof(app->pmg_smoother_pc_type), NULL));
  PetscCall(PetscOptionsInt("-pmg_smoother_max_it", "PMG smoother iterations per V-cycle", NULL, app->pmg_smoother_max_it, &app->pmg_smoother_max_it, NULL));
  PetscCall(PetscOptionsString("-pmg_apply_backend", "PMG apply backend: pcmg|shell_vcycle", NULL, app->pmg_apply_backend, app->pmg_apply_backend, sizeof(app->pmg_apply_backend), NULL));
  PetscCall(PetscOptionsInt("-pmg_shell_p2_active_ranks", "Active MPI ranks for the shell V-cycle P2 layout; 0 or >= ranks keeps all ranks active", NULL, app->pmg_shell_p2_active_ranks, &app->pmg_shell_p2_active_ranks, NULL));
  PetscCall(PetscOptionsInt("-pmg_shell_p1_active_ranks", "Active MPI ranks for the shell V-cycle P1 layout; 0 or >= ranks keeps all ranks active", NULL, app->pmg_shell_p1_active_ranks, &app->pmg_shell_p1_active_ranks, NULL));
  PetscCall(PetscOptionsString("-pmg_shell_subcomm_type", "Shell V-cycle active-rank layout: interlaced|contiguous", NULL, app->pmg_shell_subcomm_type, app->pmg_shell_subcomm_type, sizeof(app->pmg_shell_subcomm_type), NULL));
  PetscCall(PetscOptionsBool("-pmg_check_coarse_transfers", "Check PMG P4->P2 and P2->P1 transfer matrices on exact polynomial fields", NULL, app->pmg_check_coarse_transfers, &app->pmg_check_coarse_transfers, NULL));
  PetscCall(PetscOptionsInt("-pmg_lag_preconditioner", "Rebuild persistent PMG preconditioner every N Newton linear solves; 1 rebuilds every solve", NULL, app->pmg_lag_preconditioner, &app->pmg_lag_preconditioner, NULL));
  PetscCall(PetscOptionsString("-bddc_graph", "topology|petsc", NULL, app->bddc_graph, app->bddc_graph, sizeof(app->bddc_graph), NULL));
  PetscCall(PetscOptionsString("-bddc_coordinates", "scalar|blocked|none", NULL, app->bddc_coordinates, app->bddc_coordinates, sizeof(app->bddc_coordinates), NULL));
  PetscCall(PetscOptionsBool("-bddc_collapse_shared", "With -bddc_graph topology, connect local DOFs sharing the same neighboring rank set", NULL, app->bddc_collapse_shared, &app->bddc_collapse_shared, NULL));
  PetscCall(PetscOptionsBool("-bddc_local_solver_auto", "Choose scalable BDDC local/coarse solvers for large subdomains", NULL, app->bddc_local_solver_auto, &app->bddc_local_solver_auto, NULL));
  PetscCall(PetscOptionsBool("-bddc_use_local_dirichlet", "Pass local constrained unit rows to PCBDDCSetDirichletBoundariesLocal", NULL, app->bddc_use_local_dirichlet, &app->bddc_use_local_dirichlet, NULL));
  PetscCall(PetscOptionsInt("-bddc_exact_local_max_dofs", "Maximum local MATIS rows before switching BDDC subsolves away from LU", NULL, app->bddc_exact_local_max_dofs, &app->bddc_exact_local_max_dofs, NULL));
  PetscCall(PetscOptionsBool("-debug_bddc_dirichlet_rows", "Check local MATIS rows marked as BDDC Dirichlet rows", NULL, app->debug_bddc_dirichlet_rows, &app->debug_bddc_dirichlet_rows, NULL));
  PetscCall(PetscOptionsBool("-inspect_partition", "Print DMPlex/MATIS partition diagnostics and exit before assembly", NULL, app->inspect_partition, &app->inspect_partition, NULL));
  PetscCall(PetscOptionsBool("-reuse_linear_solver", "Keep one KSP/PC hierarchy and refresh operators for repeated elastic/Newton solves", NULL, app->reuse_linear_solver, &app->reuse_linear_solver, NULL));
  PetscCall(PetscOptionsBool("-deflation", "Use collected linear solutions as an A-orthonormal deflation basis for Newton solves", NULL, app->use_deflation, &app->use_deflation, NULL));
  PetscCall(PetscOptionsBool("-indirect_newton_pair_freeze_matrix", "Chord-pair experiment: assemble a fresh indirect Newton tangent on even iterations, reuse it and its preconditioner on the following odd iteration", NULL, app->indirect_newton_pair_freeze_matrix, &app->indirect_newton_pair_freeze_matrix, NULL));
  PetscCall(PetscOptionsString("-deflation_solver", "Deflated outer Krylov method: fgmres|matlab_dfgmres|cg", NULL, app->deflation_solver_name, app->deflation_solver_name, sizeof(app->deflation_solver_name), NULL));
  PetscCall(PetscOptionsString("-deflation_projector", "Deflation projector: a_orthonormal|biorthogonal", NULL, app->deflation_projector_name, app->deflation_projector_name, sizeof(app->deflation_projector_name), NULL));
  PetscCall(PetscOptionsReal("-deflation_basis_tol", "Minimum A-norm squared for keeping a deflation vector during A-orthogonalization", NULL, app->deflation_basis_tol, &app->deflation_basis_tol, NULL));
  PetscCall(PetscOptionsReal("-deflation_biorthogonal_pivot_tol", "Relative pivot-angle cutoff for -deflation_projector biorthogonal", NULL, app->deflation_biorthogonal_pivot_tol, &app->deflation_biorthogonal_pivot_tol, NULL));
  PetscCall(PetscOptionsInt("-deflation_max_it", "Maximum iterations for the explicit deflated Krylov solve; 0 uses -ksp_max_it with a safe fallback", NULL, app->deflation_max_it, &app->deflation_max_it, NULL));
  PetscCall(PetscOptionsInt("-deflation_max_vectors", "Maximum collected deflation vectors; 0 keeps all Newton-step vectors", NULL, app->deflation_max_vectors, &app->deflation_max_vectors, NULL));
  PetscCall(PetscOptionsBool("-deflation_monitor", "Print explicit deflated Krylov residual history", NULL, app->deflation_monitor, &app->deflation_monitor, NULL));
  PetscCall(PetscOptionsBool("-deflation_intra_newton_recycle", "Temporarily recycle Krylov directions from the expected cheaper indirect Newton solve into the other solve with the same matrix", NULL, app->deflation_intra_newton_recycle, &app->deflation_intra_newton_recycle, NULL));
  PetscCall(PetscOptionsInt("-deflation_recycle_max_vectors", "Maximum temporary Krylov directions to recycle per indirect Newton pair; 0 keeps all directions", NULL, app->deflation_recycle_max_vectors, &app->deflation_recycle_max_vectors, NULL));
  PetscCall(PetscOptionsReal("-deflation_recycle_basis_tol", "Minimum A-norm squared for temporary intra-Newton Krylov recycle vectors; independent of -deflation_basis_tol", NULL, app->deflation_recycle_basis_tol, &app->deflation_recycle_basis_tol, NULL));
  PetscCall(PetscOptionsBool("-deflation_krylov_persistent", "Append every explicit FGMRES/DFGMRES preconditioned Krylov direction to the persistent deflation pool for the current Newton solve", NULL, app->deflation_krylov_persistent, &app->deflation_krylov_persistent, NULL));
  PetscCall(PetscOptionsReal("-deflation_krylov_basis_tol", "Minimum A-norm squared for persistent Krylov deflation directions; 0 disables scale cutoff", NULL, app->deflation_krylov_basis_tol, &app->deflation_krylov_basis_tol, NULL));
  PetscCall(PetscOptionsBool("-deflation_check_orthonormality", "After A-orthogonalizing the deflation basis, explicitly check B^T A B against identity", NULL, app->deflation_check_orthonormality, &app->deflation_check_orthonormality, NULL));
  PetscCall(PetscOptionsReal("-deflation_orthonormality_warn_tol", "Warning threshold for DEFLATION_GRAM_CHECK ok=false", NULL, app->deflation_orthonormality_warn_tol, &app->deflation_orthonormality_warn_tol, NULL));
  PetscCall(PetscOptionsInt("-deflation_reorthogonalize_sweeps", "Extra bidirectional A-reorthogonalization sweeps after the normal basis build and before each linear solve", NULL, app->deflation_reorthogonalize_sweeps, &app->deflation_reorthogonalize_sweeps, NULL));
  PetscCall(PetscOptionsString("-linear_replay_dir", "Replay one petsc4py-exported indirect Newton linear state from this sample directory", NULL, app->linear_replay_dir, app->linear_replay_dir, sizeof(app->linear_replay_dir), NULL));
  PetscCall(PetscOptionsBool("-linear_replay_use_exported_rhs", "Solve exported petsc4py RHS vectors instead of C-rebuilt RHS vectors", NULL, app->linear_replay_use_exported_rhs, &app->linear_replay_use_exported_rhs, NULL));
  PetscCall(PetscOptionsBool("-linear_replay_check_pc_probe", "Compare exported first PCApply/Arnoldi probe vectors in replay modes", NULL, app->linear_replay_check_pc_probe, &app->linear_replay_check_pc_probe, NULL));
  PetscCall(PetscOptionsString("-step_replay_dir", "Start a full indirect Newton solve from one petsc4py-exported indirect linear sample directory", NULL, app->step_replay_dir, app->step_replay_dir, sizeof(app->step_replay_dir), NULL));
  PetscCall(PetscOptionsString("-init_replay_dir", "Replay one petsc4py-exported fixed-lambda init Newton state from this sample directory", NULL, app->init_replay_dir, app->init_replay_dir, sizeof(app->init_replay_dir), NULL));
  PetscCall(PetscOptionsBool("-init_replay_use_exported_matrix", "Solve the init replay with the exported petsc4py regularized matrix", NULL, app->init_replay_use_exported_matrix, &app->init_replay_use_exported_matrix, NULL));
  PetscCall(PetscOptionsBool("-init_replay_use_exported_rhs", "Solve the init replay with the exported petsc4py RHS", NULL, app->init_replay_use_exported_rhs, &app->init_replay_use_exported_rhs, NULL));
  PetscCall(PetscOptionsBool("-init_replay_use_exported_u", "Assemble C init replay objects at the exported petsc4py displacement", NULL, app->init_replay_use_exported_u, &app->init_replay_use_exported_u, NULL));
  PetscCall(PetscOptionsBool("-init_replay_check_damping", "Compare C fixed-lambda damping against exported petsc4py damping metadata", NULL, app->init_replay_check_damping, &app->init_replay_check_damping, NULL));
  PetscOptionsEnd();

  PetscCall(PetscStrcasecmp(app->variant_name, "gamg", &flg));
  if (flg) app->variant = VARIANT_GAMG;
  else {
    PetscCall(PetscStrcasecmp(app->variant_name, "bddc", &flg));
    if (flg) app->variant = VARIANT_BDDC;
    else {
      PetscCall(PetscStrcasecmp(app->variant_name, "fetidp", &flg));
      if (flg) app->variant = VARIANT_FETIDP;
      else {
        PetscCall(PetscStrcasecmp(app->variant_name, "pmg", &flg));
        if (flg) app->variant = VARIANT_PMG;
        else {
          PetscCall(PetscStrcasecmp(app->variant_name, "none", &flg));
          PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-pc_variant must be gamg, bddc, fetidp, pmg, or none");
          app->variant = VARIANT_NONE;
        }
      }
    }
  }
  PetscCheck(app->omega_max > 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-omega_max must be positive");
  PetscCheck(app->lambda_init > 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-lambda_init must be positive");
  PetscCheck(app->d_lambda_init > 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-d_lambda_init must be positive");
  PetscCheck(app->d_lambda_min > 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-d_lambda_min must be positive");
  PetscCheck(app->d_lambda_diff_scaled_min >= 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-d_lambda_diff_scaled_min must be nonnegative");
  PetscCheck(app->continuation_step_max >= 2, comm, PETSC_ERR_ARG_OUTOFRANGE, "-continuation_step_max must be at least 2");
  PetscCheck(app->newton_max_it >= 1, comm, PETSC_ERR_ARG_OUTOFRANGE, "-newton_max_it must be positive");
  PetscCheck(app->newton_rtol > 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-newton_rtol must be positive");
  PetscCheck(app->newton_stopping_tol > 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-newton_stopping_tol must be positive");
  PetscCheck(app->init_newton_stopping_tol > 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-init_newton_stopping_tol must be positive");
  PetscCheck(app->it_damp_max >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-it_damp_max must be nonnegative");
  PetscCheck(app->r_min > 0.0 && app->r_min <= 1.0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-r_min must be in (0,1]");
  PetscCall(PetscStrcasecmp(app->continuation_predictor, "secant", &flg));
  PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-continuation_predictor currently supports only secant");
  PetscCall(PetscStrcasecmp(app->omega_step_controller, "legacy", &flg));
  PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-omega_step_controller currently supports only legacy");
  PetscCall(PetscStrcasecmp(app->newton_stopping_criterion, "absolute_delta_lambda", &flg));
  if (!flg) {
    PetscCall(PetscStrcasecmp(app->newton_stopping_criterion, "relative_residual", &flg));
    if (!flg) {
      PetscCall(PetscStrcasecmp(app->newton_stopping_criterion, "relative_correction", &flg));
      PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-newton_stopping_criterion must be absolute_delta_lambda, relative_residual, or relative_correction");
    }
  }
  PetscCall(PetscStrcasecmp(app->init_newton_stopping_criterion, "absolute_delta_lambda", &flg));
  if (!flg) {
    PetscCall(PetscStrcasecmp(app->init_newton_stopping_criterion, "relative_residual", &flg));
    if (!flg) {
      PetscCall(PetscStrcasecmp(app->init_newton_stopping_criterion, "relative_correction", &flg));
      PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-init_newton_stopping_criterion must be absolute_delta_lambda, relative_residual, or relative_correction");
    }
  }
  PetscCheck(app->pmg_coarse_redundant_group_size >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-pmg_coarse_redundant_group_size must be nonnegative");
  PetscCheck(app->pmg_coarse_telescope_active_ranks >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-pmg_coarse_telescope_active_ranks must be nonnegative");
  PetscCheck(app->pmg_coarse_telescope_ksp_max_it >= 1, comm, PETSC_ERR_ARG_OUTOFRANGE, "-pmg_coarse_telescope_ksp_max_it must be positive");
  PetscCheck(app->pmg_p2_telescope_active_ranks >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-pmg_p2_telescope_active_ranks must be nonnegative");
  PetscCheck(app->pmg_p2_telescope_ksp_max_it >= 1, comm, PETSC_ERR_ARG_OUTOFRANGE, "-pmg_p2_telescope_ksp_max_it must be positive");
  PetscCheck(app->pmg_shell_p2_active_ranks >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-pmg_shell_p2_active_ranks must be nonnegative");
  PetscCheck(app->pmg_shell_p1_active_ranks >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-pmg_shell_p1_active_ranks must be nonnegative");
  PetscCheck(app->pmg_lag_preconditioner >= 1, comm, PETSC_ERR_ARG_OUTOFRANGE, "-pmg_lag_preconditioner must be >= 1");
  PetscCall(PetscStrcasecmp(app->pmg_apply_backend, "pcmg", &flg));
  if (!flg) {
    PetscCall(PMGApplyBackendIsShell(app, &flg));
    PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-pmg_apply_backend must be pcmg or shell_vcycle");
  }
  PetscCall(PetscStrcasecmp(app->bddc_graph, "topology", &flg));
  if (!flg) {
    PetscCall(PetscStrcasecmp(app->bddc_graph, "petsc", &flg));
    PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-bddc_graph must be topology or petsc");
  }
  PetscCall(PetscStrcasecmp(app->bddc_coordinates, "scalar", &flg));
  if (!flg) {
    PetscCall(PetscStrcasecmp(app->bddc_coordinates, "blocked", &flg));
    if (!flg) {
      PetscCall(PetscStrcasecmp(app->bddc_coordinates, "none", &flg));
      PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-bddc_coordinates must be scalar, blocked, or none");
    }
  }
  PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "rollers", &flg));
  if (!flg) {
    PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "base_only", &flg));
    if (!flg) {
      PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "full_sides", &flg));
      PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-mesh_bc_mode must be rollers, base_only, or full_sides");
    }
  }
  PetscCheck(!app->bddc_use_local_dirichlet, comm, PETSC_ERR_SUP,
             "-bddc_use_local_dirichlet is disabled for the PETSc-constrained DMPlex plasticity driver; BDDC local Dirichlet rows must first be rebuilt in MATIS local row space");
  PetscCheck(app->deflation_basis_tol >= 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-deflation_basis_tol must be nonnegative");
  PetscCheck(app->deflation_biorthogonal_pivot_tol >= 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-deflation_biorthogonal_pivot_tol must be nonnegative");
  PetscCheck(app->deflation_recycle_basis_tol >= 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-deflation_recycle_basis_tol must be nonnegative");
  PetscCheck(app->deflation_krylov_basis_tol >= 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-deflation_krylov_basis_tol must be nonnegative");
  PetscCheck(app->deflation_orthonormality_warn_tol >= 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-deflation_orthonormality_warn_tol must be nonnegative");
  PetscCheck(app->deflation_reorthogonalize_sweeps >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-deflation_reorthogonalize_sweeps must be nonnegative");
  PetscCheck(app->deflation_max_it >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-deflation_max_it must be nonnegative");
  PetscCheck(app->deflation_max_vectors >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-deflation_max_vectors must be nonnegative");
  PetscCheck(app->deflation_recycle_max_vectors >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-deflation_recycle_max_vectors must be nonnegative");
  PetscCall(PetscStrcasecmp(app->deflation_solver_name, "fgmres", &flg));
  if (flg) app->deflation_solver = DEFLATION_SOLVER_FGMRES;
  else {
    PetscCall(PetscStrcasecmp(app->deflation_solver_name, "matlab_dfgmres", &flg));
    if (!flg) PetscCall(PetscStrcasecmp(app->deflation_solver_name, "dfgmres", &flg));
    if (flg) app->deflation_solver = DEFLATION_SOLVER_MATLAB_DFGMRES;
    else {
      PetscCall(PetscStrcasecmp(app->deflation_solver_name, "cg", &flg));
      PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-deflation_solver must be fgmres, matlab_dfgmres, dfgmres, or cg");
      app->deflation_solver = DEFLATION_SOLVER_CG;
    }
  }
  PetscCall(PetscStrcasecmp(app->deflation_projector_name, "a_orthonormal", &flg));
  if (flg) app->deflation_projector = DEFLATION_PROJECTOR_A_ORTHONORMAL;
  else {
    PetscCall(PetscStrcasecmp(app->deflation_projector_name, "biorthogonal", &flg));
    PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-deflation_projector must be a_orthonormal or biorthogonal");
    app->deflation_projector = DEFLATION_PROJECTOR_BIORTHOGONAL;
  }
  PetscCheck(app->deflation_projector != DEFLATION_PROJECTOR_BIORTHOGONAL || app->deflation_solver != DEFLATION_SOLVER_CG, comm, PETSC_ERR_SUP,
             "-deflation_projector biorthogonal is implemented for explicit FGMRES/DFGMRES, not CG");
  PetscCheck(!app->deflation_intra_newton_recycle || app->deflation_solver != DEFLATION_SOLVER_CG, comm, PETSC_ERR_SUP,
             "-deflation_intra_newton_recycle is implemented for explicit FGMRES/DFGMRES, not CG");
  PetscCheck(!app->deflation_krylov_persistent || app->deflation_solver != DEFLATION_SOLVER_CG, comm, PETSC_ERR_SUP,
             "-deflation_krylov_persistent is implemented for explicit FGMRES/DFGMRES, not CG");
  PetscCheck(!app->deflation_intra_newton_recycle || app->deflation_max_vectors == 0, comm, PETSC_ERR_SUP,
             "-deflation_intra_newton_recycle currently requires -deflation_max_vectors 0 so temporary vectors cannot evict permanent history");
  PetscCheck(!app->deflation_krylov_persistent || app->deflation_max_vectors == 0, comm, PETSC_ERR_SUP,
             "-deflation_krylov_persistent requires -deflation_max_vectors 0 so Krylov history is not evicted inside the Newton solve");
  PetscCheck(!app->use_deflation || app->variant != VARIANT_FETIDP, comm, PETSC_ERR_SUP,
             "Explicit deflation currently wraps PETSc PCs; KSPFETIDP is itself a KSP and is not supported by -deflation");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetDDPartitionerDefault(MPI_Comm comm, const AppCtx *app)
{
  PetscMPIInt size;
  PetscBool   is_dd = (PetscBool)(app->variant == VARIANT_BDDC || app->variant == VARIANT_FETIDP);
  PetscBool   user_set;
  const char *part_type = NULL;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (!is_dd || size <= 1) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscOptionsHasName(NULL, NULL, "-petscpartitioner_type", &user_set));
  if (user_set) PetscFunctionReturn(PETSC_SUCCESS);
#if defined(PETSC_HAVE_PARMETIS)
  part_type = "parmetis";
#elif defined(PETSC_HAVE_PTSCOTCH)
  part_type = "ptscotch";
#endif
  if (part_type) {
    PetscCall(PetscOptionsSetValue(NULL, "-petscpartitioner_type", part_type));
    PetscCall(PetscPrintf(comm, "BDDC/FETI-DP partitioner default: -petscpartitioner_type %s\n", part_type));
  } else {
    PetscCall(PetscPrintf(comm, "BDDC/FETI-DP partitioner warning: PETSc has no ParMETIS/PTScotch; using PETSc's available partitioner default\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RepairBoundaryFaceSets(DM dm)
{
  MPI_Comm      comm;
  DM            cdm;
  PetscSection  csec;
  Vec           coords;
  DMLabel       faceSets;
  PetscReal     local_min[3] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
  PetscReal     local_max[3] = {-PETSC_MAX_REAL, -PETSC_MAX_REAL, -PETSC_MAX_REAL};
  PetscReal     global_min[3], global_max[3], scale = 1.0, tol;
  PetscInt      vStart, vEnd, fStart, fEnd;

  PetscFunctionBeginUser;
  comm = PetscObjectComm((PetscObject)dm);
  PetscCall(DMGetLabel(dm, "Face Sets", &faceSets));
  if (faceSets) {
    PetscInt imported_local = 0, imported_global = 0;

    for (PetscInt tag = 1; tag <= 7; ++tag) {
      IS       points = NULL;
      PetscInt n = 0;

      PetscCall(DMLabelGetStratumIS(faceSets, tag, &points));
      if (points) PetscCall(ISGetLocalSize(points, &n));
      imported_local += n;
      PetscCall(ISDestroy(&points));
    }
    PetscCallMPI(MPI_Allreduce(&imported_local, &imported_global, 1, MPIU_INT, MPI_SUM, comm));
    if (imported_global > 0) PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinateSection(dm, &csec));
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCheck(coords, comm, PETSC_ERR_ARG_WRONGSTATE, "Mesh has no local coordinates");

  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    PetscScalar *xyz = NULL;
    PetscInt     size = 0;

    PetscCall(DMPlexVecGetClosure(cdm, csec, coords, v, &size, &xyz));
    if (size == 3) {
      for (PetscInt d = 0; d < 3; ++d) {
        const PetscReal xd = PetscRealPart(xyz[d]);
        local_min[d]       = PetscMin(local_min[d], xd);
        local_max[d]       = PetscMax(local_max[d], xd);
      }
    }
    PetscCall(DMPlexVecRestoreClosure(cdm, csec, coords, v, &size, &xyz));
  }
  PetscCallMPI(MPI_Allreduce(local_min, global_min, 3, MPIU_REAL, MPI_MIN, comm));
  PetscCallMPI(MPI_Allreduce(local_max, global_max, 3, MPIU_REAL, MPI_MAX, comm));
  for (PetscInt d = 0; d < 3; ++d) scale = PetscMax(scale, global_max[d] - global_min[d]);
  tol = 1.0e-9 * scale;

  PetscCall(DMCreateLabel(dm, "Face Sets"));
  PetscCall(DMGetLabel(dm, "Face Sets", &faceSets));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  for (PetscInt f = fStart; f < fEnd; ++f) {
    PetscReal vol, centroid[3], normal[3];
    PetscInt  support_size;

    PetscCall(DMPlexGetSupportSize(dm, f, &support_size));
    if (support_size != 1) continue;
    PetscCall(DMPlexComputeCellGeometryFVM(dm, f, &vol, centroid, normal));
    if (PetscAbsReal(centroid[0] - global_max[0]) <= tol) PetscCall(DMLabelSetValue(faceSets, f, 1));      /* x_max */
    else if (PetscAbsReal(centroid[0] - global_min[0]) <= tol) PetscCall(DMLabelSetValue(faceSets, f, 2)); /* x_min */
    else if (PetscAbsReal(centroid[2] - global_min[2]) <= tol) PetscCall(DMLabelSetValue(faceSets, f, 3)); /* z_min */
    else if (PetscAbsReal(centroid[2] - global_max[2]) <= tol) PetscCall(DMLabelSetValue(faceSets, f, 4)); /* z_max */
    else if (PetscAbsReal(centroid[1] - global_min[1]) <= tol) PetscCall(DMLabelSetValue(faceSets, f, 5)); /* base */
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildBoundaryMarkerFromFaceSets(DM dm)
{
  DMLabel faceSets = NULL, marker = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMGetLabel(dm, "Face Sets", &faceSets));
  PetscCheck(faceSets, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Mesh has no 'Face Sets' label for plasticity boundary conditions");
  PetscCall(DMCreateLabel(dm, "boundary_marker"));
  PetscCall(DMGetLabel(dm, "boundary_marker", &marker));
  for (PetscInt tag = 1; tag <= 7; ++tag) {
    IS              points = NULL;
    const PetscInt *idx;
    PetscInt        n;

    PetscCall(DMLabelGetStratumIS(faceSets, tag, &points));
    if (!points) continue;
    PetscCall(ISGetLocalSize(points, &n));
    PetscCall(ISGetIndices(points, &idx));
    for (PetscInt i = 0; i < n; ++i) PetscCall(DMLabelSetValue(marker, idx[i], tag));
    PetscCall(ISRestoreIndices(points, &idx));
    PetscCall(ISDestroy(&points));
  }
  PetscCall(DMPlexLabelComplete(dm, marker));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReportBoundaryCounts(DM dm, const AppCtx *app)
{
  MPI_Comm        comm = PetscObjectComm((PetscObject)dm);
  DMLabel         label = NULL;
  const char     *names[6] = {"x_max", "x_min", "z_min", "z_max", "base", "y_max"};
  const PetscInt  tags[6] = {1, 2, 3, 4, 5, 6};
  PetscInt        counts[6];

  PetscFunctionBeginUser;
  PetscCall(DMGetLabel(dm, "boundary_marker", &label));
  PetscCheck(label, comm, PETSC_ERR_ARG_WRONGSTATE, "Missing boundary_marker label");
  for (PetscInt k = 0; k < 6; ++k) {
    IS       points = NULL;
    PetscInt nloc = 0;

    PetscCall(DMLabelGetStratumIS(label, tags[k], &points));
    if (points) PetscCall(ISGetLocalSize(points, &nloc));
    PetscCallMPI(MPI_Allreduce(&nloc, &counts[k], 1, MPIU_INT, MPI_SUM, comm));
    PetscCall(PetscPrintf(comm, "BOUNDARY_COUNT name=%s tag=%" PetscInt_FMT " points=%" PetscInt_FMT "\n", names[k], tags[k], counts[k]));
    PetscCall(ISDestroy(&points));
  }
  PetscCheck(counts[4] > 0, comm, PETSC_ERR_ARG_WRONGSTATE, "Plasticity boundary label has no base points");
  if (!app->use_box_mesh) {
    PetscBool is_rollers, is_full_sides;

    PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "rollers", &is_rollers));
    PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "full_sides", &is_full_sides));
    if (is_rollers || is_full_sides) {
      PetscCheck(counts[0] > 0 && counts[1] > 0 && counts[2] > 0 && counts[3] > 0, comm, PETSC_ERR_ARG_WRONGSTATE,
                 "Plasticity boundary label is missing at least one side group: x_max=%" PetscInt_FMT " x_min=%" PetscInt_FMT " z_min=%" PetscInt_FMT " z_max=%" PetscInt_FMT,
                 counts[0], counts[1], counts[2], counts[3]);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AddPlasticityBoundaryConditions(DM dm, const AppCtx *app)
{
  DMLabel        label = NULL;
  const PetscInt components[3] = {0, 1, 2};
  const PetscInt base = 5, x_rollers[2] = {2, 1}, z_rollers[2] = {3, 4};
  const PetscInt x_comp[1] = {0}, y_comp[1] = {1}, z_comp[1] = {2};
  PetscBool      is_rollers, is_base_only, is_full_sides;

  PetscFunctionBeginUser;
  PetscCall(DMGetLabel(dm, "boundary_marker", &label));
  PetscCheck(label, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Missing boundary_marker label");
  PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "rollers", &is_rollers));
  PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "base_only", &is_base_only));
  PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "full_sides", &is_full_sides));
  if (is_full_sides) {
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "glued_base", label, 1, &base, 0, 3, components, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
  } else {
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "base_roller_y", label, 1, &base, 0, 1, y_comp, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
  }
  if (app->use_box_mesh || is_rollers) {
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "x_side_rollers", label, 2, x_rollers, 0, 1, x_comp, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "z_side_rollers", label, 2, z_rollers, 0, 1, z_comp, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
  } else if (is_full_sides) {
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "x_side_clamps", label, 2, x_rollers, 0, 3, components, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "z_side_clamps", label, 2, z_rollers, 0, 3, components, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
  } else {
    PetscCheck(is_base_only, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown -mesh_bc_mode %s", app->mesh_bc_mode);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *app, P4Basis *basis, DM *dm)
{
  DM cur;

  PetscFunctionBeginUser;
  if (app->use_box_mesh) {
    const PetscInt  cells[6][4] = {{0, 1, 2, 6}, {0, 2, 3, 6}, {0, 3, 7, 6}, {0, 7, 4, 6}, {0, 4, 5, 6}, {0, 5, 1, 6}};
    const PetscReal coords[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    PetscInt        fStart, fEnd;
    DMLabel         faceSets;
    PetscCall(DMPlexCreateFromCellListPetsc(comm, 3, 6, 8, 4, PETSC_TRUE, (const PetscInt *)cells, 3, (const PetscReal *)coords, &cur));
    PetscCall(DMCreateLabel(cur, "Cell Sets"));
    for (PetscInt c = 0; c < 6; ++c) PetscCall(DMSetLabelValue(cur, "Cell Sets", c, 1));
    PetscCall(DMCreateLabel(cur, "Face Sets"));
    PetscCall(DMGetLabel(cur, "Face Sets", &faceSets));
    PetscCall(DMPlexGetHeightStratum(cur, 1, &fStart, &fEnd));
    for (PetscInt f = fStart; f < fEnd; ++f) {
      PetscReal vol, centroid[3], normal[3];
      PetscInt  support_size;
      PetscCall(DMPlexGetSupportSize(cur, f, &support_size));
      if (support_size != 1) continue;
      PetscCall(DMPlexComputeCellGeometryFVM(cur, f, &vol, centroid, normal));
      if (PetscAbsReal(centroid[0]) < 1.0e-10) PetscCall(DMLabelSetValue(faceSets, f, 2));       /* x_min */
      else if (PetscAbsReal(centroid[0] - 1.0) < 1.0e-10) PetscCall(DMLabelSetValue(faceSets, f, 1)); /* x_max */
      else if (PetscAbsReal(centroid[1]) < 1.0e-10) PetscCall(DMLabelSetValue(faceSets, f, 5));       /* base */
      else if (PetscAbsReal(centroid[2]) < 1.0e-10) PetscCall(DMLabelSetValue(faceSets, f, 3));       /* z_min */
      else if (PetscAbsReal(centroid[2] - 1.0) < 1.0e-10) PetscCall(DMLabelSetValue(faceSets, f, 4)); /* z_max */
    }
  } else {
    PetscCall(DMPlexCreateFromFile(comm, app->mesh, NULL, PETSC_TRUE, &cur));
  }
  PetscCall(DMSetFromOptions(cur));
  for (PetscInt r = 0; r < app->refine_levels; ++r) {
    DM refined = NULL;

    PetscCall(DMPlexSetRefinementUniform(cur, PETSC_TRUE));
    PetscCall(DMRefine(cur, comm, &refined));
    PetscCheck(refined, comm, PETSC_ERR_SUP, "DMRefine did not produce a refined mesh at level %" PetscInt_FMT, r);
    PetscCall(PetscPrintf(comm, "UNIFORM_REFINE level=%" PetscInt_FMT " complete=true\n", r + 1));
    PetscCall(DMDestroy(&cur));
    cur = refined;
    PetscCall(DMSetFromOptions(cur));
  }
  PetscCall(RepairBoundaryFaceSets(cur));
  PetscCall(BuildBoundaryMarkerFromFaceSets(cur));
  PetscCall(ReportBoundaryCounts(cur, app));
  PetscCall(DMSetField(cur, 0, NULL, (PetscObject)basis->fe_vector));
  PetscCall(DMCreateDS(cur));
  PetscCall(AddPlasticityBoundaryConditions(cur, app));
  PetscCall(DMGetCoordinatesLocalSetUp(cur));
  if (app->variant == VARIANT_BDDC || app->variant == VARIANT_FETIDP) PetscCall(DMSetMatType(cur, MATIS));
  else PetscCall(DMSetMatType(cur, MATAIJ));
  *dm = cur;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReportPartitionDiagnostics(DM dm, Mat A, Vec u, const AssemblyCtx *actx, const AppCtx *app)
{
  MPI_Comm             comm = PetscObjectComm((PetscObject)dm);
  PetscMPIInt          size;
  PetscPartitioner     part = NULL;
  PetscPartitionerType part_type = NULL;
  PetscInt             cStart, cEnd, vStart, vEnd;
  PetscInt             local_cells, local_vertices, local_owned_rows, global_rows;
  PetscInt             cells_sum, cells_min, cells_max, vertices_sum, vertices_min, vertices_max, owned_min, owned_max;
  PetscInt             constraints_sum, constraints_min, constraints_max;
  PetscBool            ismatis = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMPlexGetPartitioner(dm, &part));
  if (part) PetscCall(PetscPartitionerGetType(part, &part_type));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  local_cells    = cEnd - cStart;
  local_vertices = vEnd - vStart;
  PetscCall(VecGetLocalSize(u, &local_owned_rows));
  PetscCall(VecGetSize(u, &global_rows));

  PetscCallMPI(MPI_Allreduce(&local_cells, &cells_sum, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&local_cells, &cells_min, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPI_Allreduce(&local_cells, &cells_max, 1, MPIU_INT, MPI_MAX, comm));
  PetscCallMPI(MPI_Allreduce(&local_vertices, &vertices_sum, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&local_vertices, &vertices_min, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPI_Allreduce(&local_vertices, &vertices_max, 1, MPIU_INT, MPI_MAX, comm));
  PetscCallMPI(MPI_Allreduce(&local_owned_rows, &owned_min, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPI_Allreduce(&local_owned_rows, &owned_max, 1, MPIU_INT, MPI_MAX, comm));
  PetscCallMPI(MPI_Allreduce(&actx->n_constrained_local, &constraints_sum, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&actx->n_constrained_local, &constraints_min, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPI_Allreduce(&actx->n_constrained_local, &constraints_max, 1, MPIU_INT, MPI_MAX, comm));

  PetscCall(PetscPrintf(comm,
                        "PARTITION_RESULT kind=dm partitioner=%s ranks=%d variant=%s mesh=%s%s refine_levels=%" PetscInt_FMT " cells_sum=%" PetscInt_FMT " cells_min=%" PetscInt_FMT " cells_max=%" PetscInt_FMT " cell_imbalance=%.6g vertices_sum=%" PetscInt_FMT " vertices_min=%" PetscInt_FMT " vertices_max=%" PetscInt_FMT " owned_rows_min=%" PetscInt_FMT " owned_rows_max=%" PetscInt_FMT " global_dofs=%" PetscInt_FMT " owned_constraints_sum=%" PetscInt_FMT " owned_constraints_min=%" PetscInt_FMT " owned_constraints_max=%" PetscInt_FMT " global_constraints=%" PetscInt_FMT "\n",
                        part_type ? part_type : "unknown", size, app->variant_name, app->use_box_mesh ? "generated-box:" : "", app->use_box_mesh ? "unit" : app->mesh, app->refine_levels,
                        cells_sum, cells_min, cells_max, cells_sum ? ((double)cells_max * (double)size) / (double)cells_sum : 0.0, vertices_sum, vertices_min, vertices_max, owned_min, owned_max, global_rows,
                        constraints_sum, constraints_min, constraints_max, actx->n_constrained_all));

  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
  if (ismatis) {
    Mat                    local_mat = NULL;
    ISLocalToGlobalMapping mapping = NULL;
    PetscInt               local_matis_rows, matis_rows_sum, matis_rows_min, matis_rows_max;
    PetscInt               map_rows, map_rows_min, map_rows_max, map_bs;
    PetscInt               n_neighs = 0, *neighs = NULL, *n_shared = NULL, **shared = NULL;
    PetscMPIInt            rank;
    PetscBool             *is_interface = NULL;
    PetscInt               interface_rows = 0, interface_rows_sum, interface_rows_min, interface_rows_max;
    PetscInt               remote_neighbors = 0, remote_neighbors_sum, remote_neighbors_min, remote_neighbors_max;
    PetscInt               shared_entries = 0, shared_entries_sum, shared_entries_min, shared_entries_max, max_shared_with_neighbor = 0, global_max_shared_with_neighbor;
    PetscInt               nblocks = 0, block_min = PETSC_MAX_INT, block_max = 0, block_sum = 0, block_sum_total, blocks_min, blocks_max, global_block_min, global_block_max;
    const PetscInt        *bsizes = NULL;

    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCall(MatISGetLocalMat(A, &local_mat));
    PetscCall(MatGetSize(local_mat, &local_matis_rows, NULL));
    PetscCall(MatGetVariableBlockSizes(local_mat, &nblocks, &bsizes));
    for (PetscInt b = 0; b < nblocks; ++b) {
      block_min = PetscMin(block_min, bsizes[b]);
      block_max = PetscMax(block_max, bsizes[b]);
      block_sum += bsizes[b];
    }
    if (!nblocks) block_min = 0;
    PetscCall(MatISRestoreLocalMat(A, &local_mat));
    PetscCall(MatISGetLocalToGlobalMapping(A, &mapping, NULL));
    PetscCall(ISLocalToGlobalMappingGetSize(mapping, &map_rows));
    PetscCall(ISLocalToGlobalMappingGetBlockSize(mapping, &map_bs));
    PetscCall(PetscCalloc1(map_rows, &is_interface));
    PetscCall(ISLocalToGlobalMappingGetInfo(mapping, &n_neighs, &neighs, &n_shared, &shared));
    for (PetscInt n = 0; n < n_neighs; ++n) {
      if (neighs[n] == rank) continue;
      ++remote_neighbors;
      shared_entries += n_shared[n];
      max_shared_with_neighbor = PetscMax(max_shared_with_neighbor, n_shared[n]);
      for (PetscInt j = 0; j < n_shared[n]; ++j) {
        const PetscInt row = shared[n][j];

        PetscCheck(row >= 0 && row < map_rows, comm, PETSC_ERR_PLIB, "Shared local row %" PetscInt_FMT " outside [0,%" PetscInt_FMT ")", row, map_rows);
        is_interface[row] = PETSC_TRUE;
      }
    }
    PetscCall(ISLocalToGlobalMappingRestoreInfo(mapping, &n_neighs, &neighs, &n_shared, &shared));
    for (PetscInt r = 0; r < map_rows; ++r)
      if (is_interface[r]) ++interface_rows;
    PetscCall(PetscFree(is_interface));

    PetscCallMPI(MPI_Allreduce(&local_matis_rows, &matis_rows_sum, 1, MPIU_INT, MPI_SUM, comm));
    PetscCallMPI(MPI_Allreduce(&local_matis_rows, &matis_rows_min, 1, MPIU_INT, MPI_MIN, comm));
    PetscCallMPI(MPI_Allreduce(&local_matis_rows, &matis_rows_max, 1, MPIU_INT, MPI_MAX, comm));
    PetscCallMPI(MPI_Allreduce(&map_rows, &map_rows_min, 1, MPIU_INT, MPI_MIN, comm));
    PetscCallMPI(MPI_Allreduce(&map_rows, &map_rows_max, 1, MPIU_INT, MPI_MAX, comm));
    PetscCallMPI(MPI_Allreduce(&interface_rows, &interface_rows_sum, 1, MPIU_INT, MPI_SUM, comm));
    PetscCallMPI(MPI_Allreduce(&interface_rows, &interface_rows_min, 1, MPIU_INT, MPI_MIN, comm));
    PetscCallMPI(MPI_Allreduce(&interface_rows, &interface_rows_max, 1, MPIU_INT, MPI_MAX, comm));
    PetscCallMPI(MPI_Allreduce(&remote_neighbors, &remote_neighbors_sum, 1, MPIU_INT, MPI_SUM, comm));
    PetscCallMPI(MPI_Allreduce(&remote_neighbors, &remote_neighbors_min, 1, MPIU_INT, MPI_MIN, comm));
    PetscCallMPI(MPI_Allreduce(&remote_neighbors, &remote_neighbors_max, 1, MPIU_INT, MPI_MAX, comm));
    PetscCallMPI(MPI_Allreduce(&shared_entries, &shared_entries_sum, 1, MPIU_INT, MPI_SUM, comm));
    PetscCallMPI(MPI_Allreduce(&shared_entries, &shared_entries_min, 1, MPIU_INT, MPI_MIN, comm));
    PetscCallMPI(MPI_Allreduce(&shared_entries, &shared_entries_max, 1, MPIU_INT, MPI_MAX, comm));
    PetscCallMPI(MPI_Allreduce(&max_shared_with_neighbor, &global_max_shared_with_neighbor, 1, MPIU_INT, MPI_MAX, comm));
    PetscCallMPI(MPI_Allreduce(&nblocks, &blocks_min, 1, MPIU_INT, MPI_MIN, comm));
    PetscCallMPI(MPI_Allreduce(&nblocks, &blocks_max, 1, MPIU_INT, MPI_MAX, comm));
    PetscCallMPI(MPI_Allreduce(&block_min, &global_block_min, 1, MPIU_INT, MPI_MIN, comm));
    PetscCallMPI(MPI_Allreduce(&block_max, &global_block_max, 1, MPIU_INT, MPI_MAX, comm));
    PetscCallMPI(MPI_Allreduce(&block_sum, &block_sum_total, 1, MPIU_INT, MPI_SUM, comm));

    PetscCall(PetscPrintf(comm,
                          "PARTITION_RESULT kind=matis partitioner=%s ranks=%d variant=%s matis_rows_sum=%" PetscInt_FMT " matis_rows_min=%" PetscInt_FMT " matis_rows_max=%" PetscInt_FMT " matis_row_imbalance=%.6g matis_duplication=%.6g map_rows_min=%" PetscInt_FMT " map_rows_max=%" PetscInt_FMT " map_ltog_bs=%" PetscInt_FMT " interface_rows_sum=%" PetscInt_FMT " interface_rows_min=%" PetscInt_FMT " interface_rows_max=%" PetscInt_FMT " remote_neighbors_sum=%" PetscInt_FMT " remote_neighbors_min=%" PetscInt_FMT " remote_neighbors_max=%" PetscInt_FMT " shared_entries_sum=%" PetscInt_FMT " shared_entries_min=%" PetscInt_FMT " shared_entries_max=%" PetscInt_FMT " max_shared_with_neighbor=%" PetscInt_FMT " variable_blocks_min=%" PetscInt_FMT " variable_blocks_max=%" PetscInt_FMT " variable_block_size_min=%" PetscInt_FMT " variable_block_size_max=%" PetscInt_FMT " variable_block_size_local_sum=%" PetscInt_FMT "\n",
                          part_type ? part_type : "unknown", size, app->variant_name, matis_rows_sum, matis_rows_min, matis_rows_max,
                          matis_rows_sum ? ((double)matis_rows_max * (double)size) / (double)matis_rows_sum : 0.0, global_rows ? (double)matis_rows_sum / (double)global_rows : 0.0, map_rows_min, map_rows_max,
                          map_bs, interface_rows_sum, interface_rows_min, interface_rows_max, remote_neighbors_sum, remote_neighbors_min, remote_neighbors_max, shared_entries_sum, shared_entries_min,
                          shared_entries_max, global_max_shared_with_neighbor, blocks_min, blocks_max, global_block_min, global_block_max, block_sum_total));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AttachNearNullspace(DM dm, IS constrained, Mat A)
{
  DM           subdm = NULL;
  MatNullSpace ns = NULL;
  PetscInt     field = 0;

  PetscFunctionBeginUser;
  (void)constrained;
  PetscCall(DMCreateSubDM(dm, 1, &field, NULL, &subdm));
  PetscCall(DMPlexCreateRigidBody(subdm, 0, &ns));
  PetscCall(MatSetNearNullSpace(A, ns));
  PetscCall(MatNullSpaceDestroy(&ns));
  PetscCall(DMDestroy(&subdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildOwnedBlockCoordinates(DM dm, P4Basis *basis, PetscInt *nblocks, PetscReal **block_coords)
{
  PetscDualSpace dual;
  PetscSection   lsec, gsec;
  Vec            v;
  PetscInt       lo, hi, cStart, cEnd;
  PetscReal     *owned_coords, *ref_points;

  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(dm, &v));
  PetscCall(VecGetOwnershipRange(v, &lo, &hi));
  PetscCall(VecDestroy(&v));
  PetscCheck(lo % 3 == 0 && hi % 3 == 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Expected vector ownership range divisible by 3");
  *nblocks = (hi - lo) / 3;
  PetscCall(PetscCalloc1(3 * (*nblocks), &owned_coords));

  PetscCall(PetscFEGetDualSpace(basis->fe_scalar, &dual));
  PetscCall(PetscMalloc1(3 * basis->n_basis, &ref_points));
  for (PetscInt b = 0; b < basis->n_basis; ++b) {
    PetscQuadrature q;
    PetscInt        dim, Nc, npoints;
    const PetscReal *points;

    PetscCall(PetscDualSpaceGetFunctional(dual, b, &q));
    PetscCall(PetscQuadratureGetData(q, &dim, &Nc, &npoints, &points, NULL));
    PetscCheck(dim == 3 && Nc == 1 && npoints >= 1, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Unexpected scalar dual functional shape");
    for (PetscInt d = 0; d < 3; ++d) ref_points[3 * b + d] = points[d];
  }

  PetscCall(DMGetLocalSection(dm, &lsec));
  PetscCall(DMGetGlobalSection(dm, &gsec));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
    PetscReal  v0[3], J[9], invJ[9], detJ;
    PetscInt   num_indices = 0, *indices = NULL;

    PetscCall(DMPlexComputeCellGeometryFEM(dm, cell, NULL, v0, J, invJ, &detJ));
    PetscCall(DMPlexGetClosureIndices(dm, lsec, gsec, cell, PETSC_TRUE, &num_indices, &indices, NULL, NULL));
    PetscCheck(num_indices == 3 * basis->n_basis, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB,
               "Unexpected coordinate closure size %" PetscInt_FMT " != %" PetscInt_FMT, num_indices, 3 * basis->n_basis);
    for (PetscInt b = 0; b < basis->n_basis; ++b) {
      const PetscReal *r = &ref_points[3 * b];
      PetscReal        x[3];

      for (PetscInt d = 0; d < 3; ++d) x[d] = v0[d] + J[d * 3 + 0] * (r[0] + 1.0) + J[d * 3 + 1] * (r[1] + 1.0) + J[d * 3 + 2] * (r[2] + 1.0);
      for (PetscInt comp = 0; comp < 3; ++comp) {
        const PetscInt row = indices[3 * b + comp];
        if (row >= lo && row < hi) {
          const PetscInt ib = (row - lo) / 3;
          owned_coords[3 * ib + 0] = x[0];
          owned_coords[3 * ib + 1] = x[1];
          owned_coords[3 * ib + 2] = x[2];
        }
      }
    }
    PetscCall(DMPlexRestoreClosureIndices(dm, lsec, gsec, cell, PETSC_TRUE, &num_indices, &indices, NULL, NULL));
  }
  PetscCall(PetscFree(ref_points));
  *block_coords = owned_coords;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildOwnedDofCoordinates(DM dm, P4Basis *basis, PetscInt *ndofs, PetscReal **dof_coords)
{
  PetscDualSpace dual;
  PetscSection   lsec, gsec;
  Vec            v;
  PetscInt       lo, hi, cStart, cEnd;
  PetscReal     *owned_coords, *ref_points;

  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(dm, &v));
  PetscCall(VecGetOwnershipRange(v, &lo, &hi));
  PetscCall(VecDestroy(&v));
  *ndofs = hi - lo;
  PetscCall(PetscCalloc1(3 * (*ndofs), &owned_coords));

  PetscCall(PetscFEGetDualSpace(basis->fe_scalar, &dual));
  PetscCall(PetscMalloc1(3 * basis->n_basis, &ref_points));
  for (PetscInt b = 0; b < basis->n_basis; ++b) {
    PetscQuadrature  q;
    PetscInt         dim, Nc, npoints;
    const PetscReal *points;

    PetscCall(PetscDualSpaceGetFunctional(dual, b, &q));
    PetscCall(PetscQuadratureGetData(q, &dim, &Nc, &npoints, &points, NULL));
    PetscCheck(dim == 3 && Nc == 1 && npoints >= 1, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Unexpected scalar dual functional shape");
    for (PetscInt d = 0; d < 3; ++d) ref_points[3 * b + d] = points[d];
  }

  PetscCall(DMGetLocalSection(dm, &lsec));
  PetscCall(DMGetGlobalSection(dm, &gsec));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
    PetscReal v0[3], J[9], invJ[9], detJ;
    PetscInt  num_indices = 0, *indices = NULL;

    PetscCall(DMPlexComputeCellGeometryFEM(dm, cell, NULL, v0, J, invJ, &detJ));
    PetscCall(DMPlexGetClosureIndices(dm, lsec, gsec, cell, PETSC_TRUE, &num_indices, &indices, NULL, NULL));
    PetscCheck(num_indices == 3 * basis->n_basis, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB,
               "Unexpected coordinate closure size %" PetscInt_FMT " != %" PetscInt_FMT, num_indices, 3 * basis->n_basis);
    for (PetscInt b = 0; b < basis->n_basis; ++b) {
      const PetscReal *r = &ref_points[3 * b];
      PetscReal        x[3];

      for (PetscInt d = 0; d < 3; ++d) x[d] = v0[d] + J[d * 3 + 0] * (r[0] + 1.0) + J[d * 3 + 1] * (r[1] + 1.0) + J[d * 3 + 2] * (r[2] + 1.0);
      for (PetscInt comp = 0; comp < 3; ++comp) {
        const PetscInt row = indices[3 * b + comp];

        if (row >= lo && row < hi) {
          const PetscInt i = row - lo;

          owned_coords[3 * i + 0] = x[0];
          owned_coords[3 * i + 1] = x[1];
          owned_coords[3 * i + 2] = x[2];
        }
      }
    }
    PetscCall(DMPlexRestoreClosureIndices(dm, lsec, gsec, cell, PETSC_TRUE, &num_indices, &indices, NULL, NULL));
  }
  PetscCall(PetscFree(ref_points));
  *dof_coords = owned_coords;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildOwnedDofCoordinatesComponents(DM dm, P4Basis *basis, PetscInt *ndofs, PetscReal **dof_coords, PetscInt **dof_components)
{
  PetscDualSpace dual;
  PetscSection   lsec, gsec;
  Vec            v;
  PetscInt       lo, hi, cStart, cEnd;
  PetscReal     *owned_coords, *ref_points;
  PetscInt      *owned_comp;

  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(dm, &v));
  PetscCall(VecGetOwnershipRange(v, &lo, &hi));
  PetscCall(VecDestroy(&v));
  *ndofs = hi - lo;
  PetscCall(PetscCalloc1(3 * (*ndofs), &owned_coords));
  PetscCall(PetscMalloc1(*ndofs, &owned_comp));
  for (PetscInt i = 0; i < *ndofs; ++i) owned_comp[i] = -1;

  PetscCall(PetscFEGetDualSpace(basis->fe_scalar, &dual));
  PetscCall(PetscMalloc1(3 * basis->n_basis, &ref_points));
  for (PetscInt b = 0; b < basis->n_basis; ++b) {
    PetscQuadrature  q;
    PetscInt         dim, Nc, npoints;
    const PetscReal *points;

    PetscCall(PetscDualSpaceGetFunctional(dual, b, &q));
    PetscCall(PetscQuadratureGetData(q, &dim, &Nc, &npoints, &points, NULL));
    PetscCheck(dim == 3 && Nc == 1 && npoints >= 1, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Unexpected scalar dual functional shape");
    for (PetscInt d = 0; d < 3; ++d) ref_points[3 * b + d] = points[d];
  }

  PetscCall(DMGetLocalSection(dm, &lsec));
  PetscCall(DMGetGlobalSection(dm, &gsec));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
    PetscReal v0[3], J[9], invJ[9], detJ;
    PetscInt  num_indices = 0, *indices = NULL;

    PetscCall(DMPlexComputeCellGeometryFEM(dm, cell, NULL, v0, J, invJ, &detJ));
    PetscCall(DMPlexGetClosureIndices(dm, lsec, gsec, cell, PETSC_TRUE, &num_indices, &indices, NULL, NULL));
    PetscCheck(num_indices == 3 * basis->n_basis, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB,
               "Unexpected coordinate closure size %" PetscInt_FMT " != %" PetscInt_FMT, num_indices, 3 * basis->n_basis);
    for (PetscInt b = 0; b < basis->n_basis; ++b) {
      const PetscReal *r = &ref_points[3 * b];
      PetscReal        x[3];

      for (PetscInt d = 0; d < 3; ++d) x[d] = v0[d] + J[d * 3 + 0] * (r[0] + 1.0) + J[d * 3 + 1] * (r[1] + 1.0) + J[d * 3 + 2] * (r[2] + 1.0);
      for (PetscInt comp = 0; comp < 3; ++comp) {
        const PetscInt row = indices[3 * b + comp];

        if (row >= lo && row < hi) {
          const PetscInt i = row - lo;

          owned_coords[3 * i + 0] = x[0];
          owned_coords[3 * i + 1] = x[1];
          owned_coords[3 * i + 2] = x[2];
          owned_comp[i]           = comp;
        }
      }
    }
    PetscCall(DMPlexRestoreClosureIndices(dm, lsec, gsec, cell, PETSC_TRUE, &num_indices, &indices, NULL, NULL));
  }
  PetscCall(PetscFree(ref_points));
  *dof_coords     = owned_coords;
  *dof_components = owned_comp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscBool IsConstrainedGlobalDofApp(const AssemblyCtx *actx, PetscInt idx)
{
  PetscInt lo = 0, hi = actx->n_constrained_all;

  if (idx < 0 || actx->n_constrained_all == 0) return PETSC_FALSE;
  while (lo < hi) {
    const PetscInt mid = lo + (hi - lo) / 2;
    if (actx->constrained_all[mid] == idx) return PETSC_TRUE;
    if (actx->constrained_all[mid] < idx) lo = mid + 1;
    else hi = mid;
  }
  return PETSC_FALSE;
}

static PetscErrorCode BuildLocalConstrainedIS(DM dm, AssemblyCtx *actx, IS *local_is)
{
  PetscSection lsec, gsec;
  PetscInt     pStart, pEnd, nidx = 0, cap = 0, *idx = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalSection(dm, &lsec));
  PetscCall(DMGetGlobalSection(dm, &gsec));
  PetscCall(PetscSectionGetChart(lsec, &pStart, &pEnd));
  for (PetscInt p = pStart; p < pEnd; ++p) {
    PetscInt ldof, gdof, loff, goff;

    PetscCall(PetscSectionGetDof(lsec, p, &ldof));
    PetscCall(PetscSectionGetDof(gsec, p, &gdof));
    if (ldof <= 0 || gdof <= 0) continue;
    PetscCheck(ldof == gdof, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Local/global dof mismatch on point %" PetscInt_FMT, p);
    PetscCall(PetscSectionGetOffset(lsec, p, &loff));
    PetscCall(PetscSectionGetOffset(gsec, p, &goff));
    if (goff < 0) goff = -(goff + 1);
    for (PetscInt d = 0; d < gdof; ++d) {
      if (!IsConstrainedGlobalDofApp(actx, goff + d)) continue;
      if (nidx == cap) {
        cap = cap ? 2 * cap : 1024;
        PetscCall(PetscRealloc(cap * sizeof(PetscInt), &idx));
      }
      idx[nidx++] = loff + d;
    }
  }
  PetscCall(PetscSortRemoveDupsInt(&nidx, idx));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), nidx, idx, PETSC_OWN_POINTER, local_is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AttachLocalNearNullspace(DM dm, P4Basis *basis, Mat A)
{
  PetscBool ismatis = PETSC_FALSE;
  Mat       local_mat = NULL;
  Vec       local_coords = NULL;
  PetscInt  nloc;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
  if (!ismatis) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(MatISGetLocalMat(A, &local_mat));
  PetscCall(MatCreateVecs(local_mat, &local_coords, NULL));
  PetscCall(VecGetLocalSize(local_coords, &nloc));
  PetscCheck(nloc % 3 == 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Expected local MATIS size divisible by displacement block size 3");
  PetscCall(VecSetBlockSize(local_coords, 3));
  PetscCall(VecZeroEntries(local_coords));

  {
    PetscDualSpace dual;
    PetscSection   lsec;
    PetscReal     *ref_points;
    PetscScalar   *coords;
    PetscInt       cStart, cEnd;

    PetscCall(PetscFEGetDualSpace(basis->fe_scalar, &dual));
    PetscCall(PetscMalloc1(3 * basis->n_basis, &ref_points));
    for (PetscInt b = 0; b < basis->n_basis; ++b) {
      PetscQuadrature  q;
      PetscInt         dim, Nc, npoints;
      const PetscReal *points;

      PetscCall(PetscDualSpaceGetFunctional(dual, b, &q));
      PetscCall(PetscQuadratureGetData(q, &dim, &Nc, &npoints, &points, NULL));
      PetscCheck(dim == 3 && Nc == 1 && npoints >= 1, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Unexpected scalar dual functional shape");
      for (PetscInt d = 0; d < 3; ++d) ref_points[3 * b + d] = points[d];
    }

    PetscCall(DMGetLocalSection(dm, &lsec));
    PetscCall(VecGetArray(local_coords, &coords));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
      PetscReal v0[3], J[9], invJ[9], detJ;
      PetscInt  num_indices = 0, *indices = NULL;

      PetscCall(DMPlexComputeCellGeometryFEM(dm, cell, NULL, v0, J, invJ, &detJ));
      PetscCall(DMPlexGetClosureIndices(dm, lsec, lsec, cell, PETSC_TRUE, &num_indices, &indices, NULL, NULL));
      PetscCheck(num_indices == 3 * basis->n_basis, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB,
                 "Unexpected local coordinate closure size %" PetscInt_FMT " != %" PetscInt_FMT, num_indices, 3 * basis->n_basis);
      for (PetscInt b = 0; b < basis->n_basis; ++b) {
        const PetscReal *r = &ref_points[3 * b];
        PetscReal        x[3];

        for (PetscInt d = 0; d < 3; ++d) x[d] = v0[d] + J[0 * 3 + d] * r[0] + J[1 * 3 + d] * r[1] + J[2 * 3 + d] * r[2];
        for (PetscInt comp = 0; comp < 3; ++comp) {
          const PetscInt row = indices[3 * b + comp];

          PetscCheck(row >= 0 && row < nloc, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB,
                     "Unexpected local coordinate row %" PetscInt_FMT " outside [0,%" PetscInt_FMT ")", row, nloc);
          coords[row] = x[comp];
        }
      }
      PetscCall(DMPlexRestoreClosureIndices(dm, lsec, lsec, cell, PETSC_TRUE, &num_indices, &indices, NULL, NULL));
    }
    PetscCall(VecRestoreArray(local_coords, &coords));
    PetscCall(PetscFree(ref_points));
  }

  {
    MatNullSpace ns;

    PetscCall(MatNullSpaceCreateRigidBody(local_coords, &ns));
    PetscCall(MatSetNearNullSpace(local_mat, ns));
    PetscCall(MatNullSpaceDestroy(&ns));
  }
  PetscCall(VecDestroy(&local_coords));
  PetscCall(MatISRestoreLocalMat(A, &local_mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetDefaultOption(const char option[], const char value[])
{
  PetscBool set;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsHasName(NULL, NULL, option, &set));
  if (!set) PetscCall(PetscOptionsSetValue(NULL, option, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetPrefixedDefault(const char prefix[], const char suffix[], const char value[])
{
  char option[256];

  PetscFunctionBeginUser;
  PetscCall(PetscSNPrintf(option, sizeof(option), "-%s%s", prefix, suffix));
  PetscCall(SetDefaultOption(option, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetBDDCConstraintDefaults(AppCtx *app, const char prefix[])
{
  PetscBool use_topology_graph;

  PetscFunctionBeginUser;
  PetscCall(SetPrefixedDefault(prefix, "use_vertices", "true"));
  PetscCall(SetPrefixedDefault(prefix, "use_edges", "true"));
  PetscCall(SetPrefixedDefault(prefix, "use_faces", "false"));
  PetscCall(SetPrefixedDefault(prefix, "use_change_of_basis", "true"));
  PetscCall(PetscStrcasecmp(app->bddc_graph, "topology", &use_topology_graph));
  if (use_topology_graph) PetscCall(SetPrefixedDefault(prefix, "use_local_mat_graph", "false"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetFETIDPDefaults(PetscReal solver_rtol)
{
  PetscBool max_it_set;
  PetscInt  max_it;
  char      value[64];

  PetscFunctionBeginUser;
  PetscCall(SetDefaultOption("-ksp_fetidp_fullyredundant", "false"));
  PetscCall(SetDefaultOption("-fetidp_ksp_type", "gmres"));
  PetscCall(PetscSNPrintf(value, sizeof(value), "%.16g", (double)solver_rtol));
  PetscCall(SetDefaultOption("-fetidp_ksp_rtol", value));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ksp_max_it", &max_it, &max_it_set));
  if (max_it_set) {
    PetscCall(PetscSNPrintf(value, sizeof(value), "%" PetscInt_FMT, max_it));
    PetscCall(SetDefaultOption("-fetidp_ksp_max_it", value));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetMATISLocalRows(Mat A, PetscInt *nloc)
{
  PetscBool ismatis = PETSC_FALSE;
  Mat       local_mat = NULL;

  PetscFunctionBeginUser;
  *nloc = 0;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
  if (!ismatis) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatISGetLocalMat(A, &local_mat));
  PetscCall(MatGetSize(local_mat, nloc, NULL));
  PetscCall(MatISRestoreLocalMat(A, &local_mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigureBDDCAutoSolvers(AppCtx *app, Mat A, const char prefix[])
{
  MPI_Comm comm = PetscObjectComm((PetscObject)A);
  PetscInt nloc = 0, max_nloc = 0;
  const char *pc_type;

  PetscFunctionBeginUser;
  if (!app->bddc_local_solver_auto) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(GetMATISLocalRows(A, &nloc));
  PetscCallMPI(MPI_Allreduce(&nloc, &max_nloc, 1, MPIU_INT, MPI_MAX, comm));
  if (max_nloc <= app->bddc_exact_local_max_dofs) PetscFunctionReturn(PETSC_SUCCESS);

#if defined(PETSC_HAVE_HYPRE)
  pc_type = "hypre";
#else
  pc_type = "gamg";
#endif
  PetscCall(SetPrefixedDefault(prefix, "dirichlet_ksp_type", "preonly"));
  PetscCall(SetPrefixedDefault(prefix, "dirichlet_pc_type", pc_type));
  PetscCall(SetPrefixedDefault(prefix, "dirichlet_approximate", "true"));
  PetscCall(SetPrefixedDefault(prefix, "neumann_ksp_type", "preonly"));
  PetscCall(SetPrefixedDefault(prefix, "neumann_pc_type", pc_type));
  PetscCall(SetPrefixedDefault(prefix, "neumann_approximate", "true"));
  PetscCall(SetPrefixedDefault(prefix, "coarse_ksp_type", "preonly"));
  PetscCall(SetPrefixedDefault(prefix, "coarse_pc_type", pc_type));
  {
    PetscBool outer_bddc;

    PetscCall(PetscStrcmp(prefix, "pc_bddc_", &outer_bddc));
    if (outer_bddc) PetscCall(SetDefaultOption("-ksp_type", "fgmres"));
  }
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(SetPrefixedDefault(prefix, "dirichlet_pc_hypre_type", "boomeramg"));
  PetscCall(SetPrefixedDefault(prefix, "dirichlet_pc_hypre_boomeramg_max_iter", "1"));
  PetscCall(SetPrefixedDefault(prefix, "dirichlet_pc_hypre_boomeramg_tol", "0.0"));
  PetscCall(SetPrefixedDefault(prefix, "neumann_pc_hypre_type", "boomeramg"));
  PetscCall(SetPrefixedDefault(prefix, "neumann_pc_hypre_boomeramg_max_iter", "1"));
  PetscCall(SetPrefixedDefault(prefix, "neumann_pc_hypre_boomeramg_tol", "0.0"));
  PetscCall(SetPrefixedDefault(prefix, "coarse_pc_hypre_type", "boomeramg"));
  PetscCall(SetPrefixedDefault(prefix, "coarse_pc_hypre_boomeramg_max_iter", "1"));
  PetscCall(SetPrefixedDefault(prefix, "coarse_pc_hypre_boomeramg_tol", "0.0"));
#endif
  PetscCall(PetscPrintf(comm, "BDDC auto solvers: max_local_rows=%" PetscInt_FMT " threshold=%" PetscInt_FMT " pc=%s prefix=%s\n", max_nloc, app->bddc_exact_local_max_dofs, pc_type, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildGlobalComponentMap(DM dm, PetscInt *ngids, PetscInt **gids, PetscInt **comps)
{
  PetscSection lsec, gsec;
  PetscInt     pStart, pEnd, n = 0, cap = 0, *gid = NULL, *comp = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalSection(dm, &lsec));
  PetscCall(DMGetGlobalSection(dm, &gsec));
  PetscCall(PetscSectionGetChart(lsec, &pStart, &pEnd));
  for (PetscInt p = pStart; p < pEnd; ++p) {
    const PetscInt *cdofs = NULL;
    PetscInt        dof, cdof, off, cind = 0;

    PetscCall(PetscSectionGetDof(lsec, p, &dof));
    if (!dof) continue;
    PetscCheck(dof % 3 == 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Expected vector dofs in blocks of 3 on point %" PetscInt_FMT, p);
    PetscCall(PetscSectionGetConstraintDof(lsec, p, &cdof));
    PetscCall(PetscSectionGetConstraintIndices(lsec, p, &cdofs));
    PetscCall(PetscSectionGetOffset(gsec, p, &off));
    for (PetscInt c = 0; c < dof; ++c) {
      if (cind < cdof && cdofs && c == cdofs[cind]) {
        ++cind;
        continue;
      }
      if (n == cap) {
        cap = cap ? 2 * cap : 1024;
        PetscCall(PetscRealloc((size_t)cap * sizeof(PetscInt), &gid));
        PetscCall(PetscRealloc((size_t)cap * sizeof(PetscInt), &comp));
      }
      gid[n]    = (off < 0 ? -(off + 1) : off) + c - cind;
      comp[n++] = c % 3;
    }
  }

  PetscCall(PetscSortIntWithArray(n, gid, comp));
  if (n) {
    PetscInt w = 1;

    for (PetscInt r = 1; r < n; ++r) {
      if (gid[r] == gid[w - 1]) {
        PetscCheck(comp[r] == comp[w - 1], PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB,
                   "Global dof %" PetscInt_FMT " was assigned components %" PetscInt_FMT " and %" PetscInt_FMT, gid[r], comp[w - 1], comp[r]);
        continue;
      }
      gid[w]  = gid[r];
      comp[w] = comp[r];
      ++w;
    }
    n = w;
  }

  *ngids = n;
  *gids  = gid;
  *comps = comp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetBDDCDofSplittingLocal(PC pc, DM dm, Mat A)
{
  PetscBool              ismatis = PETSC_FALSE;
  Mat                    local_mat = NULL;
  ISLocalToGlobalMapping rmap;
  const PetscInt        *ridx;
  IS                     fields[3] = {NULL, NULL, NULL};
  PetscInt               nloc, nrows, ngids, *gids = NULL, *comps = NULL;
  PetscInt               nfield[3] = {0, 0, 0};
  PetscInt              *field_idx[3] = {NULL, NULL, NULL};

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
  if (!ismatis) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatISGetLocalMat(A, &local_mat));
  PetscCall(MatGetSize(local_mat, &nrows, NULL));
  PetscCall(MatISRestoreLocalMat(A, &local_mat));
  PetscCall(MatISGetLocalToGlobalMapping(A, &rmap, NULL));
  PetscCall(ISLocalToGlobalMappingGetSize(rmap, &nloc));
  PetscCheck(nloc == nrows, PetscObjectComm((PetscObject)pc), PETSC_ERR_PLIB, "MATIS local map size %" PetscInt_FMT " differs from local matrix rows %" PetscInt_FMT, nloc, nrows);

  PetscCall(BuildGlobalComponentMap(dm, &ngids, &gids, &comps));
  PetscCall(PetscMalloc3(nloc, &field_idx[0], nloc, &field_idx[1], nloc, &field_idx[2]));
  PetscCall(ISLocalToGlobalMappingGetIndices(rmap, &ridx));
  for (PetscInt i = 0; i < nloc; ++i) {
    PetscInt loc, comp;

    PetscCheck(ridx[i] >= 0, PetscObjectComm((PetscObject)pc), PETSC_ERR_PLIB, "MATIS cleaned local-to-global map contains negative index %" PetscInt_FMT, ridx[i]);
    PetscCall(PetscFindInt(ridx[i], ngids, gids, &loc));
    PetscCheck(loc >= 0, PetscObjectComm((PetscObject)pc), PETSC_ERR_PLIB, "Could not recover displacement component for MATIS local row %" PetscInt_FMT " global dof %" PetscInt_FMT, i, ridx[i]);
    comp = comps[loc];
    PetscCheck(comp >= 0 && comp < 3, PetscObjectComm((PetscObject)pc), PETSC_ERR_PLIB, "Invalid component %" PetscInt_FMT " for MATIS local row %" PetscInt_FMT, comp, i);
    field_idx[comp][nfield[comp]++] = i;
  }
  PetscCall(ISLocalToGlobalMappingRestoreIndices(rmap, &ridx));

  for (PetscInt comp = 0; comp < 3; ++comp) PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nfield[comp], field_idx[comp], PETSC_COPY_VALUES, &fields[comp]));
  PetscCall(PCBDDCSetDofsSplittingLocal(pc, 3, fields));
  for (PetscInt comp = 0; comp < 3; ++comp) PetscCall(ISDestroy(&fields[comp]));
  PetscCall(PetscFree3(field_idx[0], field_idx[1], field_idx[2]));
  PetscCall(PetscFree(gids));
  PetscCall(PetscFree(comps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckLocalDirichletRows(Mat A, IS dirichlet_local)
{
  MPI_Comm       comm = PetscObjectComm((PetscObject)A);
  PetscBool      ismatis = PETSC_FALSE;
  Mat            local_mat = NULL;
  const PetscInt *rows;
  PetscInt       nrows, global_nrows = 0, local_bad_diag = 0, local_bad_offdiag = 0, global_bad_diag = 0, global_bad_offdiag = 0;
  PetscReal      local_max_diag_error = 0.0, local_max_offdiag = 0.0, global_max_diag_error = 0.0, global_max_offdiag = 0.0;
  PetscReal      tol = 1.0e-9;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
  if (!ismatis) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatISGetLocalMat(A, &local_mat));
  PetscCall(ISGetLocalSize(dirichlet_local, &nrows));
  PetscCall(ISGetIndices(dirichlet_local, &rows));
  for (PetscInt i = 0; i < nrows; ++i) {
    const PetscInt    row = rows[i];
    PetscInt          ncols;
    const PetscInt   *cols;
    const PetscScalar *vals;
    PetscScalar       diag = 0.0;
    PetscReal         offdiag = 0.0;
    PetscBool         found_diag = PETSC_FALSE;

    PetscCall(MatGetRow(local_mat, row, &ncols, &cols, &vals));
    for (PetscInt j = 0; j < ncols; ++j) {
      if (cols[j] == row) {
        diag += vals[j];
        found_diag = PETSC_TRUE;
      } else {
        offdiag = PetscMax(offdiag, PetscAbsScalar(vals[j]));
      }
    }
    PetscCall(MatRestoreRow(local_mat, row, &ncols, &cols, &vals));
    {
      const PetscReal diag_error = found_diag ? PetscAbsScalar(diag - 1.0) : PETSC_MAX_REAL;

      local_max_diag_error = PetscMax(local_max_diag_error, diag_error);
      local_max_offdiag    = PetscMax(local_max_offdiag, offdiag);
      if (diag_error > tol) ++local_bad_diag;
      if (offdiag > tol) ++local_bad_offdiag;
    }
  }
  PetscCall(ISRestoreIndices(dirichlet_local, &rows));
  PetscCallMPI(MPI_Allreduce(&nrows, &global_nrows, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&local_bad_diag, &global_bad_diag, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&local_bad_offdiag, &global_bad_offdiag, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&local_max_diag_error, &global_max_diag_error, 1, MPIU_REAL, MPI_MAX, comm));
  PetscCallMPI(MPI_Allreduce(&local_max_offdiag, &global_max_offdiag, 1, MPIU_REAL, MPI_MAX, comm));
  PetscCall(PetscPrintf(comm,
                        "BDDC local Dirichlet row check: rows=%" PetscInt_FMT " bad_diag=%" PetscInt_FMT " bad_offdiag=%" PetscInt_FMT " max_diag_error=%.3e max_offdiag=%.3e\n",
                        global_nrows, global_bad_diag, global_bad_offdiag, (double)global_max_diag_error, (double)global_max_offdiag));
  PetscCall(MatISRestoreLocalMat(A, &local_mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscInt *cols;
  PetscInt  n;
  PetscInt  cap;
} AdjacencyRow;

static PetscErrorCode AdjacencyAdd(AdjacencyRow rows[], PetscInt nloc, PetscInt row, PetscInt col)
{
  PetscFunctionBeginUser;
  PetscCheck(row >= 0 && row < nloc && col >= 0 && col < nloc, PETSC_COMM_SELF, PETSC_ERR_PLIB,
             "Local graph entry (%" PetscInt_FMT ",%" PetscInt_FMT ") outside [0,%" PetscInt_FMT ")", row, col, nloc);
  if (rows[row].n == rows[row].cap) {
    rows[row].cap = rows[row].cap ? 2 * rows[row].cap : 8;
    PetscCall(PetscRealloc(rows[row].cap * sizeof(PetscInt), &rows[row].cols));
  }
  rows[row].cols[rows[row].n++] = col;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AddSymmetricAdjacency(AdjacencyRow rows[], PetscInt nloc, PetscInt a, PetscInt b)
{
  PetscFunctionBeginUser;
  PetscCall(AdjacencyAdd(rows, nloc, a, b));
  if (a != b) PetscCall(AdjacencyAdd(rows, nloc, b, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscInt row;
  PetscInt count;
  PetscInt hash1;
  PetscInt hash2;
} SharedSignature;

static int CompareSharedSignature(const void *a, const void *b)
{
  const SharedSignature *sa = (const SharedSignature *)a;
  const SharedSignature *sb = (const SharedSignature *)b;

  if (sa->count < sb->count) return -1;
  if (sa->count > sb->count) return 1;
  if (sa->hash1 < sb->hash1) return -1;
  if (sa->hash1 > sb->hash1) return 1;
  if (sa->hash2 < sb->hash2) return -1;
  if (sa->hash2 > sb->hash2) return 1;
  if (sa->row < sb->row) return -1;
  if (sa->row > sb->row) return 1;
  return 0;
}

static PetscBool SameSharedSignature(const SharedSignature *a, const SharedSignature *b)
{
  return (PetscBool)(a->count == b->count && a->hash1 == b->hash1 && a->hash2 == b->hash2);
}

static PetscErrorCode AddSharedSetCollapseAdjacency(Mat A, AdjacencyRow rows[], PetscInt nloc)
{
  ISLocalToGlobalMapping mapping = NULL;
  PetscMPIInt            rank;
  PetscInt               n_neighs, *neighs, *n_shared, **shared;
  SharedSignature       *sigs = NULL;
  const PetscInt         p1 = 2147483647, p2 = 2147483629;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
  PetscCall(MatISGetLocalToGlobalMapping(A, &mapping, NULL));
  PetscCall(PetscCalloc1(nloc, &sigs));
  for (PetscInt r = 0; r < nloc; ++r) sigs[r].row = r;
  PetscCall(ISLocalToGlobalMappingGetInfo(mapping, &n_neighs, &neighs, &n_shared, &shared));
  for (PetscInt n = 0; n < n_neighs; ++n) {
    if (neighs[n] == rank) continue;
    for (PetscInt j = 0; j < n_shared[n]; ++j) {
      const PetscInt row = shared[n][j];
      long long      h1, h2, nr;

      PetscCheck(row >= 0 && row < nloc, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB,
                 "Shared local row %" PetscInt_FMT " outside [0,%" PetscInt_FMT ")", row, nloc);
      nr              = (long long)neighs[n] + 1;
      h1              = ((long long)sigs[row].hash1 + (nr * 1000003LL)) % p1;
      h2              = ((long long)sigs[row].hash2 + (nr * (nr + 17LL) * 9176LL)) % p2;
      sigs[row].hash1 = (PetscInt)h1;
      sigs[row].hash2 = (PetscInt)h2;
      sigs[row].count++;
    }
  }
  PetscCall(ISLocalToGlobalMappingRestoreInfo(mapping, &n_neighs, &neighs, &n_shared, &shared));
  qsort(sigs, (size_t)nloc, sizeof(*sigs), CompareSharedSignature);
  {
    PetscInt local_shared = 0, local_groups = 0, local_max_group = 0;
    PetscInt global_shared = 0, global_groups = 0, global_max_group = 0;

    for (PetscInt i = 0; i < nloc;) {
      PetscInt j = i + 1;

      while (j < nloc && SameSharedSignature(&sigs[i], &sigs[j])) ++j;
      if (sigs[i].count) {
        local_shared += j - i;
        local_groups++;
        local_max_group = PetscMax(local_max_group, j - i);
      }
      i = j;
    }
    PetscCallMPI(MPI_Allreduce(&local_shared, &global_shared, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)A)));
    PetscCallMPI(MPI_Allreduce(&local_groups, &global_groups, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)A)));
    PetscCallMPI(MPI_Allreduce(&local_max_group, &global_max_group, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)A)));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A),
                          "BDDC shared-set graph collapse: shared_rows=%" PetscInt_FMT " groups=%" PetscInt_FMT " max_group=%" PetscInt_FMT "\n",
                          global_shared, global_groups, global_max_group));
  }
  for (PetscInt i = 1; i < nloc; ++i) {
    if (!sigs[i].count || !SameSharedSignature(&sigs[i - 1], &sigs[i])) continue;
    PetscCall(AddSymmetricAdjacency(rows, nloc, sigs[i - 1].row, sigs[i].row));
  }
  PetscCall(PetscFree(sigs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildBasisBarycentric(P4Basis *basis, PetscInt **bary_out)
{
  PetscDualSpace dual;
  PetscInt      *bary;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(4 * basis->n_basis, &bary));
  PetscCall(PetscFEGetDualSpace(basis->fe_scalar, &dual));
  for (PetscInt b = 0; b < basis->n_basis; ++b) {
    PetscQuadrature  q;
    PetscInt         dim, Nc, npoints;
    const PetscReal *points;
    PetscReal        lambda[4], sum = 0.0;

    PetscCall(PetscDualSpaceGetFunctional(dual, b, &q));
    PetscCall(PetscQuadratureGetData(q, &dim, &Nc, &npoints, &points, NULL));
    PetscCheck(dim == 3 && Nc == 1 && npoints >= 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected scalar dual functional shape");
    for (PetscInt d = 0; d < 3; ++d) {
      lambda[d] = 0.5 * (points[d] + 1.0);
      sum += lambda[d];
    }
    lambda[3] = 1.0 - sum;
    for (PetscInt d = 0; d < 4; ++d) bary[4 * b + d] = (PetscInt)PetscFloorReal(basis->degree * lambda[d] + 0.5);
  }
  *bary_out = bary;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscBool BarycentricNeighbors(const PetscInt a[], const PetscInt b[])
{
  PetscInt dist = 0;

  for (PetscInt d = 0; d < 4; ++d) dist += PetscAbsInt(a[d] - b[d]);
  return (PetscBool)(dist == 2);
}

static PetscErrorCode ConfigureBDDCTopologyGraph(PC pc, DM dm, P4Basis *basis, Mat A, PetscBool collapse_shared)
{
  PetscBool     ismatis = PETSC_FALSE;
  Mat           local_mat = NULL;
  PetscSection  lsec;
  PetscInt      nloc, cStart, cEnd;
  PetscInt     *bary = NULL, *xadj = NULL, *adjncy = NULL;
  AdjacencyRow *rows = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
  if (!ismatis) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatISGetLocalMat(A, &local_mat));
  PetscCall(MatGetSize(local_mat, &nloc, NULL));
  PetscCall(MatISRestoreLocalMat(A, &local_mat));
  PetscCheck(nloc >= 0, PetscObjectComm((PetscObject)pc), PETSC_ERR_PLIB, "Invalid local MATIS size");
  PetscCall(BuildBasisBarycentric(basis, &bary));
  PetscCall(PetscCalloc1(nloc, &rows));
  PetscCall(DMGetLocalSection(dm, &lsec));

  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
    PetscInt num_indices = 0, *indices = NULL;

    PetscCall(DMPlexGetClosureIndices(dm, lsec, lsec, cell, PETSC_TRUE, &num_indices, &indices, NULL, NULL));
    PetscCheck(num_indices == 3 * basis->n_basis, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB,
               "Unexpected local graph closure size %" PetscInt_FMT " != %" PetscInt_FMT " on cell %" PetscInt_FMT, num_indices, 3 * basis->n_basis, cell);

    for (PetscInt b = 0; b < basis->n_basis; ++b) {
      for (PetscInt c0 = 0; c0 < 3; ++c0) {
        const PetscInt rb = indices[3 * b + c0];

        PetscCall(AdjacencyAdd(rows, nloc, rb, rb));
        for (PetscInt c1 = 0; c1 < 3; ++c1) PetscCall(AdjacencyAdd(rows, nloc, rb, indices[3 * b + c1]));
      }
    }
    for (PetscInt b = 0; b < basis->n_basis; ++b) {
      for (PetscInt c = b + 1; c < basis->n_basis; ++c) {
        if (!BarycentricNeighbors(&bary[4 * b], &bary[4 * c])) continue;
        for (PetscInt comp = 0; comp < 3; ++comp) PetscCall(AddSymmetricAdjacency(rows, nloc, indices[3 * b + comp], indices[3 * c + comp]));
      }
    }
    PetscCall(DMPlexRestoreClosureIndices(dm, lsec, lsec, cell, PETSC_TRUE, &num_indices, &indices, NULL, NULL));
  }
  if (collapse_shared) PetscCall(AddSharedSetCollapseAdjacency(A, rows, nloc));

  PetscCall(PetscMalloc1(nloc + 1, &xadj));
  xadj[0] = 0;
  for (PetscInt r = 0; r < nloc; ++r) {
    if (!rows[r].n) PetscCall(AdjacencyAdd(rows, nloc, r, r));
    PetscCall(PetscSortRemoveDupsInt(&rows[r].n, rows[r].cols));
    xadj[r + 1] = xadj[r] + rows[r].n;
  }
  PetscCall(PetscMalloc1(xadj[nloc], &adjncy));
  for (PetscInt r = 0; r < nloc; ++r) PetscCall(PetscArraycpy(&adjncy[xadj[r]], rows[r].cols, rows[r].n));
  PetscCall(PCBDDCSetLocalAdjacencyGraph(pc, nloc, xadj, adjncy, PETSC_OWN_POINTER));

  for (PetscInt r = 0; r < nloc; ++r) PetscCall(PetscFree(rows[r].cols));
  PetscCall(PetscFree(rows));
  PetscCall(PetscFree(bary));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigureBDDCPrimalVertices(PC pc, DM dm, P4Basis *basis)
{
  PetscSection lsec;
  PetscInt     cStart, cEnd, *bary = NULL, *idx = NULL, nidx = 0, cap = 0;

  PetscFunctionBeginUser;
  PetscCall(BuildBasisBarycentric(basis, &bary));
  PetscCall(DMGetLocalSection(dm, &lsec));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
    PetscInt num_indices = 0, *indices = NULL;

    PetscCall(DMPlexGetClosureIndices(dm, lsec, lsec, cell, PETSC_TRUE, &num_indices, &indices, NULL, NULL));
    PetscCheck(num_indices == 3 * basis->n_basis, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB,
               "Unexpected local primal closure size %" PetscInt_FMT " != %" PetscInt_FMT, num_indices, 3 * basis->n_basis);
    for (PetscInt b = 0; b < basis->n_basis; ++b) {
      PetscBool is_vertex = PETSC_FALSE;

      for (PetscInt d = 0; d < 4; ++d) {
        if (bary[4 * b + d] == basis->degree) is_vertex = PETSC_TRUE;
      }
      if (!is_vertex) continue;
      for (PetscInt comp = 0; comp < 3; ++comp) {
        if (nidx == cap) {
          cap = cap ? 2 * cap : 64;
          PetscCall(PetscRealloc(cap * sizeof(PetscInt), &idx));
        }
        idx[nidx++] = indices[3 * b + comp];
      }
    }
    PetscCall(DMPlexRestoreClosureIndices(dm, lsec, lsec, cell, PETSC_TRUE, &num_indices, &indices, NULL, NULL));
  }
  PetscCall(PetscSortRemoveDupsInt(&nidx, idx));
  {
    IS primals;

    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc), nidx, idx, PETSC_OWN_POINTER, &primals));
    idx = NULL;
    PetscCall(PCBDDCSetPrimalVerticesLocalIS(pc, primals));
    PetscCall(ISDestroy(&primals));
  }
  PetscCall(PetscFree(idx));
  PetscCall(PetscFree(bary));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigureBDDC(PC pc, DM dm, AssemblyCtx *actx, AppCtx *app, Mat A)
{
  IS         dirichlet_local = NULL;
  PetscInt   ncoords;
  PetscReal *coords = NULL;
  PetscBool  use_topology_graph;

  PetscFunctionBeginUser;
  PetscCall(PCSetType(pc, PCBDDC));
  if (app->bddc_use_local_dirichlet || app->debug_bddc_dirichlet_rows) {
    PetscCall(BuildLocalConstrainedIS(dm, actx, &dirichlet_local));
    if (app->debug_bddc_dirichlet_rows) PetscCall(CheckLocalDirichletRows(A, dirichlet_local));
    if (app->bddc_use_local_dirichlet) PetscCall(PCBDDCSetDirichletBoundariesLocal(pc, dirichlet_local));
    PetscCall(ISDestroy(&dirichlet_local));
  }
  {
    PetscBool use_scalar, use_blocked, use_none;

    PetscCall(PetscStrcasecmp(app->bddc_coordinates, "scalar", &use_scalar));
    PetscCall(PetscStrcasecmp(app->bddc_coordinates, "blocked", &use_blocked));
    PetscCall(PetscStrcasecmp(app->bddc_coordinates, "none", &use_none));
    if (use_scalar) {
      /*
        PETSc 3.24 BDDC still has "TODO: support for blocked" in its coordinate
        import path and checks against the scalar local pmat size. GAMG uses
        blocked coordinates; scalar is the default BDDC/FETI-DP workaround here.
      */
      PetscCall(BuildOwnedDofCoordinates(dm, actx->basis, &ncoords, &coords));
      PetscCall(PCSetCoordinates(pc, 3, ncoords, coords));
      PetscCall(PetscFree(coords));
    } else if (use_blocked) {
      PetscCall(BuildOwnedBlockCoordinates(dm, actx->basis, &ncoords, &coords));
      PetscCall(PCSetCoordinates(pc, 3, ncoords, coords));
      PetscCall(PetscFree(coords));
    } else {
      PetscCheck(use_none, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown -bddc_coordinates value %s", app->bddc_coordinates);
    }
  }
  PetscCall(SetBDDCDofSplittingLocal(pc, dm, A));
  PetscCall(PetscStrcasecmp(app->bddc_graph, "topology", &use_topology_graph));
  if (use_topology_graph) {
    PetscCall(ConfigureBDDCTopologyGraph(pc, dm, actx->basis, A, app->bddc_collapse_shared));
    PetscCall(ConfigureBDDCPrimalVertices(pc, dm, actx->basis));
  }
  PetscCall(AttachNearNullspace(dm, actx->constrained_is, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrepareInnerBDDCFromOptions(PC pc, const char prefix[])
{
  const char *existing = NULL;

  PetscFunctionBeginUser;
  PetscCall(PCSetType(pc, PCBDDC));
  PetscCall(PCGetOptionsPrefix(pc, &existing));
  if (!existing || !existing[0]) PetscCall(PCSetOptionsPrefix(pc, prefix));
  PetscCall(PCSetFromOptions(pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildBasisReferencePoints(P4Basis *basis, PetscReal **points_out)
{
  PetscDualSpace dual;
  PetscReal     *points_out_local;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(3 * basis->n_basis, &points_out_local));
  PetscCall(PetscFEGetDualSpace(basis->fe_scalar, &dual));
  for (PetscInt b = 0; b < basis->n_basis; ++b) {
    PetscQuadrature  q;
    PetscInt         dim, Nc, npoints;
    const PetscReal *points;

    PetscCall(PetscDualSpaceGetFunctional(dual, b, &q));
    PetscCall(PetscQuadratureGetData(q, &dim, &Nc, &npoints, &points, NULL));
    PetscCheck(dim == 3 && Nc == 1 && npoints >= 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected scalar dual functional shape");
    for (PetscInt d = 0; d < 3; ++d) points_out_local[3 * b + d] = points[d];
  }
  *points_out = points_out_local;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSameMeshLevelDM(DM fine_dm, P4Basis *basis, const AppCtx *app, DM *level_dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMClone(fine_dm, level_dm));
  PetscCall(DMClearDS(*level_dm));
  PetscCall(DMClearFields(*level_dm));
  PetscCall(DMSetField(*level_dm, 0, NULL, (PetscObject)basis->fe_vector));
  PetscCall(DMCreateDS(*level_dm));
  PetscCall(AddPlasticityBoundaryConditions(*level_dm, app));
  PetscCall(DMGetCoordinatesLocalSetUp(*level_dm));
  PetscCall(DMSetMatType(*level_dm, MATAIJ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildInterpolationMatrixWithLayouts(DM fine_dm, P4Basis *fine_basis, DM coarse_dm, P4Basis *coarse_basis,
                                                          PetscInt mat_mlocal, PetscInt mat_nlocal, PetscInt mat_M, PetscInt mat_N, Mat *P)
{
  MPI_Comm        comm = PetscObjectComm((PetscObject)fine_dm);
  PetscSection    fine_lsec, fine_gsec, coarse_lsec, coarse_gsec;
  Vec             fine_vec = NULL;
  PetscInt        rlo, rhi, cStart, cEnd;
  PetscReal      *fine_points = NULL;
  PetscTabulation coarse_at_fine = NULL;
  const PetscReal *phi;
  Mat             mat;

  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(fine_dm, &fine_vec));
  PetscCall(VecGetOwnershipRange(fine_vec, &rlo, &rhi));
  PetscCall(VecDestroy(&fine_vec));

  PetscCall(BuildBasisReferencePoints(fine_basis, &fine_points));
  PetscCall(PetscFECreateTabulation(coarse_basis->fe_scalar, 1, fine_basis->n_basis, fine_points, 0, &coarse_at_fine));
  phi = coarse_at_fine->T[0];

  PetscCall(MatCreateAIJ(comm, mat_mlocal, mat_nlocal, mat_M, mat_N, coarse_basis->n_basis, NULL, coarse_basis->n_basis, NULL, &mat));
  PetscCall(MatSetOption(mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(DMGetLocalSection(fine_dm, &fine_lsec));
  PetscCall(DMGetGlobalSection(fine_dm, &fine_gsec));
  PetscCall(DMGetLocalSection(coarse_dm, &coarse_lsec));
  PetscCall(DMGetGlobalSection(coarse_dm, &coarse_gsec));
  PetscCall(DMPlexGetHeightStratum(fine_dm, 0, &cStart, &cEnd));
  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
    PetscInt fine_n = 0, coarse_n = 0, *fine_idx = NULL, *coarse_idx = NULL;

    PetscCall(DMPlexGetClosureIndices(fine_dm, fine_lsec, fine_gsec, cell, PETSC_TRUE, &fine_n, &fine_idx, NULL, NULL));
    PetscCall(DMPlexGetClosureIndices(coarse_dm, coarse_lsec, coarse_gsec, cell, PETSC_TRUE, &coarse_n, &coarse_idx, NULL, NULL));
    PetscCheck(fine_n == 3 * fine_basis->n_basis, comm, PETSC_ERR_PLIB, "Unexpected fine transfer closure size %" PetscInt_FMT, fine_n);
    PetscCheck(coarse_n == 3 * coarse_basis->n_basis, comm, PETSC_ERR_PLIB, "Unexpected coarse transfer closure size %" PetscInt_FMT, coarse_n);
    for (PetscInt fb = 0; fb < fine_basis->n_basis; ++fb) {
      for (PetscInt comp = 0; comp < 3; ++comp) {
        const PetscInt row = fine_idx[3 * fb + comp];

        if (row < 0) continue;
        if (row < rlo || row >= rhi) continue;
        for (PetscInt cb = 0; cb < coarse_basis->n_basis; ++cb) {
          const PetscScalar val = phi[fb * coarse_basis->n_basis + cb];
          const PetscInt    col = coarse_idx[3 * cb + comp];

          if (col < 0) continue;
          if (PetscAbsScalar(val) <= 1.0e-12) continue;
          PetscCall(MatSetValue(mat, row, col, val, INSERT_VALUES));
        }
      }
    }
    PetscCall(DMPlexRestoreClosureIndices(fine_dm, fine_lsec, fine_gsec, cell, PETSC_TRUE, &fine_n, &fine_idx, NULL, NULL));
    PetscCall(DMPlexRestoreClosureIndices(coarse_dm, coarse_lsec, coarse_gsec, cell, PETSC_TRUE, &coarse_n, &coarse_idx, NULL, NULL));
  }
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));

  PetscCall(PetscTabulationDestroy(&coarse_at_fine));
  PetscCall(PetscFree(fine_points));
  *P = mat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildInterpolationMatrix(DM fine_dm, P4Basis *fine_basis, DM coarse_dm, P4Basis *coarse_basis, Mat *P)
{
  Vec      fine_vec = NULL, coarse_vec = NULL;
  PetscInt mlocal, nlocal, M, N;

  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(fine_dm, &fine_vec));
  PetscCall(DMCreateGlobalVector(coarse_dm, &coarse_vec));
  PetscCall(VecGetLocalSize(fine_vec, &mlocal));
  PetscCall(VecGetLocalSize(coarse_vec, &nlocal));
  PetscCall(VecGetSize(fine_vec, &M));
  PetscCall(VecGetSize(coarse_vec, &N));
  PetscCall(VecDestroy(&fine_vec));
  PetscCall(VecDestroy(&coarse_vec));
  PetscCall(BuildInterpolationMatrixWithLayouts(fine_dm, fine_basis, coarse_dm, coarse_basis, mlocal, nlocal, M, N, P));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ChoosePMGCoarsePC(AppCtx *app, DM coarse_dm, char coarse_pc[], size_t coarse_pc_len)
{
  MPI_Comm  comm = PetscObjectComm((PetscObject)coarse_dm);
  Vec       v = NULL;
  PetscInt  coarse_dofs;
  PetscMPIInt size;
  PetscBool flg, direct_set = PETSC_FALSE;
  char      direct_pc[32] = "";

  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(coarse_dm, &v));
  PetscCall(VecGetSize(v, &coarse_dofs));
  PetscCall(VecDestroy(&v));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-mg_coarse_pc_type", direct_pc, sizeof(direct_pc), &direct_set));
  if (direct_set) {
    PetscCall(PetscStrcasecmp(direct_pc, "lu", &flg));
    PetscCheck(!flg || coarse_dofs <= app->pmg_coarse_lu_max_dofs, comm, PETSC_ERR_ARG_WRONG,
               "Refusing PMG coarse LU for %" PetscInt_FMT " coarse DOFs above -pmg_coarse_lu_max_dofs %" PetscInt_FMT, coarse_dofs, app->pmg_coarse_lu_max_dofs);
#if !defined(PETSC_HAVE_HYPRE)
    PetscCall(PetscStrcasecmp(direct_pc, "hypre", &flg));
    PetscCheck(!flg, comm, PETSC_ERR_SUP, "This PETSc build has no HYPRE support");
#endif
    PetscCall(PetscStrncpy(coarse_pc, direct_pc, coarse_pc_len));
    PetscCall(PetscPrintf(comm, "PMG coarse space: dofs=%" PetscInt_FMT " selected_pc=%s source=mg_coarse_pc_type lu_limit=%" PetscInt_FMT "\n", coarse_dofs, coarse_pc,
                          app->pmg_coarse_lu_max_dofs));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscStrcasecmp(app->pmg_coarse_pc_type, "auto", &flg));
  if (flg) {
    if (size == 1 && coarse_dofs <= app->pmg_coarse_lu_max_dofs) PetscCall(PetscStrncpy(coarse_pc, "lu", coarse_pc_len));
    else PetscCall(PetscStrncpy(coarse_pc, "gamg", coarse_pc_len));
  } else {
    PetscBool is_lu, is_hypre, is_gamg;

    PetscCall(PetscStrcasecmp(app->pmg_coarse_pc_type, "lu", &is_lu));
    PetscCall(PetscStrcasecmp(app->pmg_coarse_pc_type, "hypre", &is_hypre));
    PetscCall(PetscStrcasecmp(app->pmg_coarse_pc_type, "gamg", &is_gamg));
    PetscCheck(is_lu || is_hypre || is_gamg, comm, PETSC_ERR_ARG_WRONG, "-pmg_coarse_pc_type must be auto, hypre, gamg, or lu");
    PetscCheck(!is_lu || coarse_dofs <= app->pmg_coarse_lu_max_dofs, comm, PETSC_ERR_ARG_WRONG,
               "Refusing PMG coarse LU for %" PetscInt_FMT " coarse DOFs above -pmg_coarse_lu_max_dofs %" PetscInt_FMT, coarse_dofs, app->pmg_coarse_lu_max_dofs);
#if !defined(PETSC_HAVE_HYPRE)
    PetscCheck(!is_hypre, comm, PETSC_ERR_SUP, "This PETSc build has no HYPRE support");
#endif
    PetscCall(PetscStrncpy(coarse_pc, app->pmg_coarse_pc_type, coarse_pc_len));
  }
  PetscCall(PetscPrintf(comm, "PMG coarse space: dofs=%" PetscInt_FMT " selected_pc=%s lu_limit=%" PetscInt_FMT "\n", coarse_dofs, coarse_pc, app->pmg_coarse_lu_max_dofs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigurePMGBasePC(PC pc, const char pc_type[], PetscBool aggressive_square_graph)
{
  PetscBool is_gamg, is_hypre;

  PetscFunctionBeginUser;
  PetscCall(PCSetType(pc, pc_type));
  PetscCall(PetscStrcasecmp(pc_type, "gamg", &is_gamg));
  PetscCall(PetscStrcasecmp(pc_type, "hypre", &is_hypre));
  if (is_gamg) {
    PetscCall(PCGAMGSetType(pc, PCGAMGAGG));
    PetscCall(PCGAMGSetAggressiveSquareGraph(pc, aggressive_square_graph));
  }
#if defined(PETSC_HAVE_HYPRE)
  if (is_hypre) PetscCall(PCHYPRESetType(pc, "boomeramg"));
#else
  PetscCheck(!is_hypre, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "This PETSc build has no HYPRE support");
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReportPMGLevelDofs(DM dm, PetscInt level, PetscInt degree)
{
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);
  Vec      v = NULL;
  PetscInt local_dofs, global_dofs, local_min, local_max;

  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(dm, &v));
  PetscCall(VecGetLocalSize(v, &local_dofs));
  PetscCall(VecGetSize(v, &global_dofs));
  PetscCallMPI(MPI_Allreduce(&local_dofs, &local_min, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPI_Allreduce(&local_dofs, &local_max, 1, MPIU_INT, MPI_MAX, comm));
  PetscCall(PetscPrintf(comm,
                        "PMG_LEVEL_DOF level=%" PetscInt_FMT " degree=%" PetscInt_FMT " local_dofs_rank0=%" PetscInt_FMT " local_dofs_min=%" PetscInt_FMT " local_dofs_max=%" PetscInt_FMT " global_dofs=%" PetscInt_FMT "\n",
                        level, degree, local_dofs, local_min, local_max, global_dofs));
  PetscCall(VecDestroy(&v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReportPMGLevelSolver(MPI_Comm comm, PetscInt level, KSP ksp)
{
  KSPType   ksp_type = NULL;
  PC        pc = NULL;
  PCType    pc_type = NULL;
  PetscReal rtol, abstol, dtol;
  PetscInt  max_it;

  PetscFunctionBeginUser;
  PetscCall(KSPGetType(ksp, &ksp_type));
  PetscCall(KSPGetTolerances(ksp, &rtol, &abstol, &dtol, &max_it));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCGetType(pc, &pc_type));
  PetscCall(PetscPrintf(comm,
                        "PMG_LEVEL_SOLVER level=%" PetscInt_FMT " ksp=%s pc=%s rtol=%.6e abstol=%.6e dtol=%.6e max_it=%" PetscInt_FMT "\n",
                        level, ksp_type ? ksp_type : "unset", pc_type ? pc_type : "unset", (double)rtol, (double)abstol, (double)dtol, max_it));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReportPMGSolverChoices(KSP ksp, const AppCtx *app)
{
  MPI_Comm comm = PetscObjectComm((PetscObject)ksp);
  PC       pc = NULL;
  PCType   pc_type = NULL;
  KSP      level_ksp = NULL;
  PetscBool is_mg = PETSC_FALSE;

  PetscFunctionBeginUser;
  if (app->variant != VARIANT_PMG) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCGetType(pc, &pc_type));
  if (pc_type) PetscCall(PetscStrcmp(pc_type, PCMG, &is_mg));
  if (!is_mg) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PCMGGetCoarseSolve(pc, &level_ksp));
  PetscCall(ReportPMGLevelSolver(comm, 0, level_ksp));
  for (PetscInt level = 1; level < 3; ++level) {
    PetscCall(PCMGGetSmoother(pc, level, &level_ksp));
    PetscCall(ReportPMGLevelSolver(comm, level, level_ksp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetPMGTelescopeDefaults(AppCtx *app, MPI_Comm comm)
{
  PetscMPIInt ranks;
  PetscBool   pc_set;
  char        value[64];

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &ranks));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-mg_coarse_pc_type", &pc_set));
  if (app->pmg_coarse_telescope_active_ranks <= 0 || ranks <= app->pmg_coarse_telescope_active_ranks || pc_set) {
    PetscCall(PetscPrintf(comm,
                          "PMG_TELESCOPE_CONFIG level=p1 enabled=false option_override=%s active_ranks=%" PetscInt_FMT " reduction_factor=1 subcomm=%s inner_ksp=%s inner_pc=%s\n",
                          pc_set ? "true" : "false", app->pmg_coarse_telescope_active_ranks, app->pmg_coarse_telescope_subcomm_type,
                          app->pmg_coarse_telescope_ksp_type, app->pmg_coarse_telescope_pc_type));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(ranks % app->pmg_coarse_telescope_active_ranks == 0, comm, PETSC_ERR_ARG_WRONG,
             "-pmg_coarse_telescope_active_ranks %" PetscInt_FMT " must divide MPI ranks %d", app->pmg_coarse_telescope_active_ranks, ranks);

  PetscCall(SetDefaultOption("-mg_coarse_pc_type", "telescope"));
  PetscCall(PetscSNPrintf(value, sizeof(value), "%" PetscInt_FMT, (PetscInt)ranks / app->pmg_coarse_telescope_active_ranks));
  PetscCall(SetDefaultOption("-mg_coarse_pc_telescope_reduction_factor", value));
  PetscCall(SetDefaultOption("-mg_coarse_pc_telescope_subcomm_type", app->pmg_coarse_telescope_subcomm_type));
  PetscCall(SetDefaultOption("-mg_coarse_telescope_ksp_type", app->pmg_coarse_telescope_ksp_type));
  PetscCall(PetscSNPrintf(value, sizeof(value), "%.16g", (double)app->pmg_coarse_telescope_ksp_rtol));
  PetscCall(SetDefaultOption("-mg_coarse_telescope_ksp_rtol", value));
  PetscCall(PetscSNPrintf(value, sizeof(value), "%" PetscInt_FMT, app->pmg_coarse_telescope_ksp_max_it));
  PetscCall(SetDefaultOption("-mg_coarse_telescope_ksp_max_it", value));
  PetscCall(SetDefaultOption("-mg_coarse_telescope_pc_type", app->pmg_coarse_telescope_pc_type));
  PetscCall(PetscPrintf(comm,
                        "PMG_TELESCOPE_CONFIG level=p1 enabled=true option_override=false active_ranks=%" PetscInt_FMT " reduction_factor=%" PetscInt_FMT " subcomm=%s inner_ksp=%s inner_pc=%s\n",
                        app->pmg_coarse_telescope_active_ranks, (PetscInt)ranks / app->pmg_coarse_telescope_active_ranks, app->pmg_coarse_telescope_subcomm_type,
                        app->pmg_coarse_telescope_ksp_type, app->pmg_coarse_telescope_pc_type));
  PetscCall(PetscPrintf(comm,
                        "PMG_TELESCOPE active_ranks=%" PetscInt_FMT " reduction_factor=%" PetscInt_FMT " subcomm=%s inner_ksp=%s inner_pc=%s\n",
                        app->pmg_coarse_telescope_active_ranks, (PetscInt)ranks / app->pmg_coarse_telescope_active_ranks, app->pmg_coarse_telescope_subcomm_type,
                        app->pmg_coarse_telescope_ksp_type, app->pmg_coarse_telescope_pc_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetPMGP2TelescopeDefaults(AppCtx *app, MPI_Comm comm)
{
  PetscMPIInt ranks;
  PetscBool   pc_set;
  char        value[64];

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &ranks));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-mg_levels_1_pc_type", &pc_set));
  if (app->pmg_p2_telescope_active_ranks <= 0 || ranks <= app->pmg_p2_telescope_active_ranks || pc_set) {
    PetscCall(PetscPrintf(comm,
                          "PMG_TELESCOPE_CONFIG level=p2 enabled=false option_override=%s active_ranks=%" PetscInt_FMT " reduction_factor=1 subcomm=%s inner_ksp=%s inner_pc=%s\n",
                          pc_set ? "true" : "false", app->pmg_p2_telescope_active_ranks, app->pmg_p2_telescope_subcomm_type,
                          app->pmg_p2_telescope_ksp_type, app->pmg_p2_telescope_pc_type));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(ranks % app->pmg_p2_telescope_active_ranks == 0, comm, PETSC_ERR_ARG_WRONG,
             "-pmg_p2_telescope_active_ranks %" PetscInt_FMT " must divide MPI ranks %d", app->pmg_p2_telescope_active_ranks, ranks);

  PetscCall(SetDefaultOption("-mg_levels_1_pc_type", "telescope"));
  PetscCall(PetscSNPrintf(value, sizeof(value), "%" PetscInt_FMT, (PetscInt)ranks / app->pmg_p2_telescope_active_ranks));
  PetscCall(SetDefaultOption("-mg_levels_1_pc_telescope_reduction_factor", value));
  PetscCall(SetDefaultOption("-mg_levels_1_pc_telescope_subcomm_type", app->pmg_p2_telescope_subcomm_type));
  PetscCall(SetDefaultOption("-mg_levels_1_telescope_ksp_type", app->pmg_p2_telescope_ksp_type));
  PetscCall(PetscSNPrintf(value, sizeof(value), "%.16g", (double)app->pmg_p2_telescope_ksp_rtol));
  PetscCall(SetDefaultOption("-mg_levels_1_telescope_ksp_rtol", value));
  PetscCall(PetscSNPrintf(value, sizeof(value), "%" PetscInt_FMT, app->pmg_p2_telescope_ksp_max_it));
  PetscCall(SetDefaultOption("-mg_levels_1_telescope_ksp_max_it", value));
  PetscCall(SetDefaultOption("-mg_levels_1_telescope_pc_type", app->pmg_p2_telescope_pc_type));
  {
    PetscBool is_gamg = PETSC_FALSE;

    PetscCall(PetscStrcasecmp(app->pmg_p2_telescope_pc_type, "gamg", &is_gamg));
    if (is_gamg) PetscCall(SetDefaultOption("-mg_levels_1_telescope_pc_gamg_aggressive_square_graph", app->pmg_coarse_gamg_aggressive_square_graph ? "true" : "false"));
  }
  PetscCall(PetscPrintf(comm,
                        "PMG_TELESCOPE_CONFIG level=p2 enabled=true option_override=false active_ranks=%" PetscInt_FMT " reduction_factor=%" PetscInt_FMT " subcomm=%s inner_ksp=%s inner_pc=%s\n",
                        app->pmg_p2_telescope_active_ranks, (PetscInt)ranks / app->pmg_p2_telescope_active_ranks, app->pmg_p2_telescope_subcomm_type,
                        app->pmg_p2_telescope_ksp_type, app->pmg_p2_telescope_pc_type));
  PetscCall(PetscPrintf(comm,
                        "PMG_P2_TELESCOPE active_ranks=%" PetscInt_FMT " reduction_factor=%" PetscInt_FMT " subcomm=%s inner_ksp=%s inner_pc=%s\n",
                        app->pmg_p2_telescope_active_ranks, (PetscInt)ranks / app->pmg_p2_telescope_active_ranks, app->pmg_p2_telescope_subcomm_type,
                        app->pmg_p2_telescope_ksp_type, app->pmg_p2_telescope_pc_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGParseSubcommType(MPI_Comm comm, const char name[], PetscSubcommType *type)
{
  PetscBool is_interlaced = PETSC_FALSE, is_contiguous = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscStrcasecmp(name, "interlaced", &is_interlaced));
  PetscCall(PetscStrcasecmp(name, "contiguous", &is_contiguous));
  PetscCheck(is_interlaced || is_contiguous, comm, PETSC_ERR_ARG_WRONG, "PMG telescope subcomm type must be interlaced or contiguous, got %s", name);
  *type = is_interlaced ? PETSC_SUBCOMM_INTERLACED : PETSC_SUBCOMM_CONTIGUOUS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGResolveActiveRanks(MPI_Comm comm, PetscInt requested_active_ranks, PetscInt *active_ranks, PetscInt *reduction_factor)
{
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (requested_active_ranks <= 0 || requested_active_ranks >= (PetscInt)size) *active_ranks = (PetscInt)size;
  else *active_ranks = requested_active_ranks;
  PetscCheck(*active_ranks > 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "Active-rank count must be positive");
  PetscCheck((PetscInt)size % *active_ranks == 0, comm, PETSC_ERR_ARG_WRONG,
             "Active-rank count %" PetscInt_FMT " must divide MPI ranks %d", *active_ranks, size);
  *reduction_factor = (PetscInt)size / *active_ranks;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGIsActiveRank(MPI_Comm comm, PetscInt active_ranks, PetscInt reduction_factor, PetscSubcommType subcomm_type, PetscBool *active)
{
  PetscMPIInt rank, size;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (active_ranks == (PetscInt)size) *active = PETSC_TRUE;
  else if (subcomm_type == PETSC_SUBCOMM_CONTIGUOUS) *active = (PetscBool)(rank < active_ranks);
  else *active = (PetscBool)(rank % reduction_factor == 0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscBool       active;
  PetscInt        requested_active_ranks;
  PetscInt        active_ranks;
  PetscInt        inactive_ranks;
  PetscInt        reduction_factor;
  PetscInt        global_dofs;
  PetscInt        local_dofs;
  PetscInt        local_min;
  PetscInt        local_max;
  PetscInt        ownership_start;
  PetscInt        ownership_end;
  PetscInt        block_size;
  PetscSubcommType subcomm_type;
  char            subcomm_type_name[32];
  MPI_Comm        subcomm;
  IS              isrow;
  Vec             full_template;
  Vec             sub_template;
  VecScatter      original_to_active;
} PMGActiveLayout;

static PetscErrorCode PMGActiveLayoutCreate(MPI_Comm comm, Vec original_template, PetscInt requested_active_ranks, const char subcomm_type_name[], PMGActiveLayout *layout)
{
  PetscMPIInt rank, size;
  VecType     vec_type = NULL;
  PetscInt    local_for_min;
  PetscInt    original_local, original_start, original_end;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(layout, sizeof(*layout)));
  layout->subcomm = MPI_COMM_NULL;
  layout->requested_active_ranks = requested_active_ranks;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PMGParseSubcommType(comm, subcomm_type_name, &layout->subcomm_type));
  PetscCall(PetscStrncpy(layout->subcomm_type_name, subcomm_type_name, sizeof(layout->subcomm_type_name)));
  PetscCall(VecGetSize(original_template, &layout->global_dofs));
  PetscCall(VecGetLocalSize(original_template, &original_local));
  PetscCall(VecGetOwnershipRange(original_template, &original_start, &original_end));
  PetscCall(VecGetBlockSize(original_template, &layout->block_size));
  PetscCall(VecGetType(original_template, &vec_type));
  PetscCall(PMGResolveActiveRanks(comm, requested_active_ranks, &layout->active_ranks, &layout->reduction_factor));
  layout->inactive_ranks   = (PetscInt)size - layout->active_ranks;
  PetscCall(PMGIsActiveRank(comm, layout->active_ranks, layout->reduction_factor, layout->subcomm_type, &layout->active));

  PetscCallMPI(MPI_Comm_split(comm, layout->active ? 0 : MPI_UNDEFINED, rank, &layout->subcomm));
  if (layout->active) {
    PetscCall(VecCreate(layout->subcomm, &layout->sub_template));
    PetscCall(VecSetSizes(layout->sub_template, layout->active_ranks == (PetscInt)size ? original_local : PETSC_DECIDE, layout->global_dofs));
    PetscCall(VecSetBlockSize(layout->sub_template, layout->block_size));
    if (vec_type) PetscCall(VecSetType(layout->sub_template, vec_type));
    PetscCall(VecSetFromOptions(layout->sub_template));
    PetscCall(VecGetLocalSize(layout->sub_template, &layout->local_dofs));
    PetscCall(VecGetOwnershipRange(layout->sub_template, &layout->ownership_start, &layout->ownership_end));
    if (layout->active_ranks == (PetscInt)size) {
      PetscCheck(layout->local_dofs == original_local && layout->ownership_start == original_start && layout->ownership_end == original_end, comm, PETSC_ERR_PLIB,
                 "All-active PMG layout failed to preserve original ownership: original [%" PetscInt_FMT ",%" PetscInt_FMT ") local %" PetscInt_FMT
                 ", active [%" PetscInt_FMT ",%" PetscInt_FMT ") local %" PetscInt_FMT,
                 original_start, original_end, original_local, layout->ownership_start, layout->ownership_end, layout->local_dofs);
    }
  } else {
    layout->local_dofs      = 0;
    layout->ownership_start = 0;
    layout->ownership_end   = 0;
  }

  PetscCall(VecCreate(comm, &layout->full_template));
  PetscCall(VecSetSizes(layout->full_template, layout->local_dofs, layout->global_dofs));
  PetscCall(VecSetBlockSize(layout->full_template, layout->block_size));
  if (vec_type) PetscCall(VecSetType(layout->full_template, vec_type));
  PetscCall(VecSetFromOptions(layout->full_template));
  PetscCall(ISCreateStride(comm, layout->local_dofs, layout->ownership_start, 1, &layout->isrow));
  PetscCall(ISSetBlockSize(layout->isrow, layout->block_size));
  PetscCall(VecScatterCreate(original_template, layout->isrow, layout->full_template, NULL, &layout->original_to_active));

  local_for_min = layout->active ? layout->local_dofs : PETSC_MAX_INT;
  PetscCallMPI(MPI_Allreduce(&local_for_min, &layout->local_min, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPI_Allreduce(&layout->local_dofs, &layout->local_max, 1, MPIU_INT, MPI_MAX, comm));
  if (layout->local_min == PETSC_MAX_INT) layout->local_min = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGActiveLayoutDestroy(PMGActiveLayout *layout)
{
  PetscFunctionBeginUser;
  PetscCall(VecScatterDestroy(&layout->original_to_active));
  PetscCall(VecDestroy(&layout->full_template));
  PetscCall(VecDestroy(&layout->sub_template));
  PetscCall(ISDestroy(&layout->isrow));
  if (layout->subcomm != MPI_COMM_NULL) PetscCallMPI(MPI_Comm_free(&layout->subcomm));
  PetscCall(PetscMemzero(layout, sizeof(*layout)));
  layout->subcomm = MPI_COMM_NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGActiveLayoutDuplicateFull(PMGActiveLayout *layout, Vec *v)
{
  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(layout->full_template, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGActiveLayoutDuplicateSub(PMGActiveLayout *layout, Vec *v)
{
  PetscFunctionBeginUser;
  if (layout->active) PetscCall(VecDuplicate(layout->sub_template, v));
  else *v = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGActiveLayoutCopyFullToSub(PMGActiveLayout *layout, Vec full, Vec sub)
{
  PetscFunctionBeginUser;
  if (layout->active) {
    const PetscScalar *full_array;
    PetscScalar       *sub_array;
    PetscInt           nfull, nsub;

    PetscCall(VecGetLocalSize(full, &nfull));
    PetscCall(VecGetLocalSize(sub, &nsub));
    PetscCheck(nfull == nsub, layout->subcomm, PETSC_ERR_PLIB, "Active full/sub vector sizes differ: %" PetscInt_FMT " != %" PetscInt_FMT, nfull, nsub);
    PetscCall(VecGetArrayRead(full, &full_array));
    PetscCall(VecGetArray(sub, &sub_array));
    for (PetscInt i = 0; i < nsub; ++i) sub_array[i] = full_array[i];
    PetscCall(VecRestoreArray(sub, &sub_array));
    PetscCall(VecRestoreArrayRead(full, &full_array));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGActiveLayoutCopySubToFull(PMGActiveLayout *layout, Vec sub, Vec full)
{
  PetscFunctionBeginUser;
  if (layout->active) {
    const PetscScalar *sub_array;
    PetscScalar       *full_array;
    PetscInt           nfull, nsub;

    PetscCall(VecGetLocalSize(full, &nfull));
    PetscCall(VecGetLocalSize(sub, &nsub));
    PetscCheck(nfull == nsub, layout->subcomm, PETSC_ERR_PLIB, "Active full/sub vector sizes differ: %" PetscInt_FMT " != %" PetscInt_FMT, nfull, nsub);
    PetscCall(VecGetArrayRead(sub, &sub_array));
    PetscCall(VecGetArray(full, &full_array));
    for (PetscInt i = 0; i < nfull; ++i) full_array[i] = sub_array[i];
    PetscCall(VecRestoreArray(full, &full_array));
    PetscCall(VecRestoreArrayRead(sub, &sub_array));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGActiveLayoutScatterOriginalToFull(PMGActiveLayout *layout, Vec original, Vec active_full)
{
  PetscFunctionBeginUser;
  PetscCall(VecScatterBegin(layout->original_to_active, original, active_full, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(layout->original_to_active, original, active_full, INSERT_VALUES, SCATTER_FORWARD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellRedistributeActiveMatrix(MPI_Comm comm, PMGActiveLayout *layout, Mat B, MatReuse reuse, Mat *Bred,
                                                       PetscLogDouble *submatrix_time, PetscLogDouble *concatenate_time)
{
  Mat           *submats = NULL;
  Mat            Blocal = NULL;
  IS             iscol = NULL;
  PetscInt       nc, nr, bs;
  PetscLogDouble t0, t1, t2;

  PetscFunctionBeginUser;
  *submatrix_time   = 0.0;
  *concatenate_time = 0.0;
  PetscCall(MatGetSize(B, &nr, &nc));
  PetscCall(MatGetBlockSizes(B, NULL, &bs));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, nc, 0, 1, &iscol));
  PetscCall(ISSetIdentity(iscol));
  PetscCall(ISSetBlockSize(iscol, bs));
  PetscCall(MatSetOption(B, MAT_SUBMAT_SINGLEIS, PETSC_TRUE));
  PetscCall(PetscTime(&t0));
  PetscCall(MatCreateSubMatrices(B, 1, &layout->isrow, &iscol, MAT_INITIAL_MATRIX, &submats));
  PetscCall(PetscTime(&t1));
  Blocal = submats[0];
  PetscCall(PetscFree(submats));
  if (layout->active) {
    PetscInt mm;

    PetscCall(MatGetSize(Blocal, &mm, NULL));
    PetscCall(MatCreateMPIMatConcatenateSeqMat(layout->subcomm, Blocal, mm, reuse, Bred));
  }
  PetscCall(PetscTime(&t2));
  *submatrix_time   = t1 - t0;
  *concatenate_time = t2 - t1;
  PetscCall(MatDestroy(&Blocal));
  PetscCall(ISDestroy(&iscol));
  (void)comm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellAttachSubcommNearNullspace(DM dm, PMGActiveLayout *layout, Mat Ared)
{
  DM           subdm = NULL;
  MatNullSpace ns = NULL;
  PetscInt     field = 0;
  PetscBool    has_const = PETSC_FALSE;
  PetscInt     nvec = 0;
  const Vec   *vecs = NULL;
  Vec         *sub_vecs = NULL;
  Vec          active_tmp = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMCreateSubDM(dm, 1, &field, NULL, &subdm));
  PetscCall(DMPlexCreateRigidBody(subdm, 0, &ns));
  PetscCall(MatNullSpaceGetVecs(ns, &has_const, &nvec, &vecs));
  PetscCall(PMGActiveLayoutDuplicateFull(layout, &active_tmp));
  if (layout->active && nvec > 0) PetscCall(VecDuplicateVecs(layout->sub_template, nvec, &sub_vecs));
  for (PetscInt i = 0; i < nvec; ++i) {
    PetscCall(PMGActiveLayoutScatterOriginalToFull(layout, vecs[i], active_tmp));
    PetscCall(PMGActiveLayoutCopyFullToSub(layout, active_tmp, layout->active ? sub_vecs[i] : NULL));
  }
  if (layout->active) {
    MatNullSpace sub_ns = NULL;

    PetscCall(MatNullSpaceCreate(layout->subcomm, has_const, nvec, sub_vecs, &sub_ns));
    PetscCall(MatSetNearNullSpace(Ared, sub_ns));
    PetscCall(MatNullSpaceDestroy(&sub_ns));
    if (nvec > 0) PetscCall(VecDestroyVecs(nvec, &sub_vecs));
  }
  PetscCall(VecDestroy(&active_tmp));
  PetscCall(MatNullSpaceDestroy(&ns));
  PetscCall(DMDestroy(&subdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  AppCtx        *app;
  DM             dm;
  AssemblyCtx   *actx;
  Mat            A4;
  PetscBool      setup_done;
  PetscBool      p1_basis_created;
  PetscBool      p2_basis_created;
  P4Basis        p1_basis;
  P4Basis        p2_basis;
  DM             dm_p1;
  DM             dm_p2;
  PMGActiveLayout p2_layout;
  PMGActiveLayout p1_layout;
  Mat            P42;
  Mat            P21;
  Mat            A2_active;
  Mat            A1_active;
  Mat            A2_sub;
  Mat            A1_sub;
  KSP            smooth4;
  KSP            smooth2;
  KSP            coarse1;
  Vec            r4;
  Vec            tmp4;
  Vec            z4;
  Vec            p2_rhs_full;
  Vec            p2_x_full;
  Vec            p2_r_full;
  Vec            p2_corr_full;
  Vec            p1_rhs_full;
  Vec            p1_x_full;
  Vec            p2_rhs;
  Vec            p2_x;
  Vec            p2_r;
  Vec            p2_delta;
  Vec            p1_rhs;
  Vec            p1_x;
  PetscBool      debug_capture;
  Vec            debug_fine_pre;
  Vec            debug_fine_residual;
  Vec            debug_p2_rhs;
  Vec            debug_p2_pre;
  Vec            debug_p2_residual;
  Vec            debug_p1_rhs;
  Vec            debug_p1_x;
  Vec            debug_p2_post;
  PetscInt       apply_calls;
  PetscInt       operator_updates;
  PetscLogDouble fine_smooth_time;
  PetscLogDouble p2_smooth_time;
  PetscLogDouble restrict_time;
  PetscLogDouble prolong_time;
  PetscLogDouble coarse_solve_time;
  PetscLogDouble residual_time;
  PetscLogDouble operator_update_time;
} PMGShellVCycleCtx;

static PetscErrorCode LinearSolverDestroyPMGHierarchy(LinearSolverCtx *solver);
static PetscErrorCode CheckPMGCoarseTransfers(LinearSolverCtx *solver);
static PetscErrorCode LinearSolverSetupPMGTransferChecks(LinearSolverCtx *solver, DM dm, const AppCtx *app);

static PetscErrorCode PMGShellConfigureSmootherKSP(KSP ksp, AppCtx *app, const char prefix[], Mat A)
{
  PC pc = NULL;

  PetscFunctionBeginUser;
  PetscCall(KSPSetOptionsPrefix(ksp, prefix));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, app->pmg_smoother_ksp_type));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_NONE));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(KSPSetTolerances(ksp, 0.0, 0.0, PETSC_CURRENT, app->pmg_smoother_max_it));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, app->pmg_smoother_pc_type));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellConfigureCoarseKSP(KSP ksp, AppCtx *app, const char prefix[], Mat A)
{
  PC pc = NULL;

  PetscFunctionBeginUser;
  PetscCall(KSPSetOptionsPrefix(ksp, prefix));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, app->pmg_coarse_telescope_ksp_type));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(KSPSetTolerances(ksp, app->pmg_coarse_telescope_ksp_rtol, PETSC_DEFAULT, PETSC_DEFAULT, app->pmg_coarse_telescope_ksp_max_it));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(ConfigurePMGBasePC(pc, app->pmg_coarse_telescope_pc_type, app->pmg_coarse_gamg_aggressive_square_graph));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellReportFineLevel(DM dm)
{
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);
  Vec      v = NULL;
  PetscMPIInt size;
  PetscInt local_dofs, global_dofs, local_min, local_max;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMCreateGlobalVector(dm, &v));
  PetscCall(VecGetLocalSize(v, &local_dofs));
  PetscCall(VecGetSize(v, &global_dofs));
  PetscCallMPI(MPI_Allreduce(&local_dofs, &local_min, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPI_Allreduce(&local_dofs, &local_max, 1, MPIU_INT, MPI_MAX, comm));
  PetscCall(PetscPrintf(comm,
                        "PMG_SHELL_LEVEL level=2 degree=4 active_ranks=%d global_dofs=%" PetscInt_FMT " local_min=%" PetscInt_FMT " local_max=%" PetscInt_FMT " inactive_ranks=0\n",
                        size, global_dofs, local_min, local_max));
  PetscCall(VecDestroy(&v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellReportActiveLevel(MPI_Comm comm, PetscInt level, PetscInt degree, PMGActiveLayout *layout)
{
  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(comm,
                        "PMG_SHELL_LEVEL level=%" PetscInt_FMT " degree=%" PetscInt_FMT " active_ranks=%" PetscInt_FMT " global_dofs=%" PetscInt_FMT " local_min=%" PetscInt_FMT " local_max=%" PetscInt_FMT " inactive_ranks=%" PetscInt_FMT "\n",
                        level, degree, layout->active_ranks, layout->global_dofs, layout->local_min, layout->local_max, layout->inactive_ranks));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellCreateHierarchy(PC pc, PMGShellVCycleCtx *ctx, Mat A4)
{
  MPI_Comm  comm = PetscObjectComm((PetscObject)pc);
  Vec       fine_vec = NULL, p2_vec = NULL, p1_vec = NULL;
  PetscInt  fine_local, fine_global;
  char      prefix4[128];

  PetscFunctionBeginUser;
  PetscCall(P4BasisCreateDegree(PETSC_COMM_SELF, 1, &ctx->p1_basis));
  ctx->p1_basis_created = PETSC_TRUE;
  PetscCall(P4BasisCreateDegree(PETSC_COMM_SELF, 2, &ctx->p2_basis));
  ctx->p2_basis_created = PETSC_TRUE;
  PetscCall(CreateSameMeshLevelDM(ctx->dm, &ctx->p1_basis, ctx->app, &ctx->dm_p1));
  PetscCall(CreateSameMeshLevelDM(ctx->dm, &ctx->p2_basis, ctx->app, &ctx->dm_p2));

  PetscCall(ReportPMGLevelDofs(ctx->dm_p1, 0, 1));
  PetscCall(ReportPMGLevelDofs(ctx->dm_p2, 1, 2));
  PetscCall(ReportPMGLevelDofs(ctx->dm, 2, 4));

  PetscCall(DMCreateGlobalVector(ctx->dm, &fine_vec));
  PetscCall(DMCreateGlobalVector(ctx->dm_p2, &p2_vec));
  PetscCall(DMCreateGlobalVector(ctx->dm_p1, &p1_vec));
  PetscCall(PMGActiveLayoutCreate(comm, p2_vec, ctx->app->pmg_shell_p2_active_ranks, ctx->app->pmg_shell_subcomm_type, &ctx->p2_layout));
  PetscCall(PMGActiveLayoutCreate(comm, p1_vec, ctx->app->pmg_shell_p1_active_ranks, ctx->app->pmg_shell_subcomm_type, &ctx->p1_layout));
  PetscCall(PMGShellReportFineLevel(ctx->dm));
  PetscCall(PMGShellReportActiveLevel(comm, 1, 2, &ctx->p2_layout));
  PetscCall(PMGShellReportActiveLevel(comm, 0, 1, &ctx->p1_layout));

  PetscCall(VecGetLocalSize(fine_vec, &fine_local));
  PetscCall(VecGetSize(fine_vec, &fine_global));
  PetscCall(BuildInterpolationMatrixWithLayouts(ctx->dm_p2, &ctx->p2_basis, ctx->dm_p1, &ctx->p1_basis, ctx->p2_layout.local_dofs,
                                                ctx->p1_layout.local_dofs, ctx->p2_layout.global_dofs, ctx->p1_layout.global_dofs, &ctx->P21));
  PetscCall(BuildInterpolationMatrixWithLayouts(ctx->dm, ctx->actx->basis, ctx->dm_p2, &ctx->p2_basis, fine_local, ctx->p2_layout.local_dofs,
                                                fine_global, ctx->p2_layout.global_dofs, &ctx->P42));
  PetscCall(PetscPrintf(comm,
                        "PMG_COARSE_OPERATOR_CONFIG type=galerkin_shell_vcycle p1_quadrature_points=%" PetscInt_FMT " p2_quadrature_points=%" PetscInt_FMT "\n",
                        ctx->p1_basis.n_qp, ctx->p2_basis.n_qp));
  if (ctx->app->pmg_check_coarse_transfers) {
    LinearSolverCtx transfer_solver;

    PetscCall(PetscMemzero(&transfer_solver, sizeof(transfer_solver)));
    transfer_solver.dm   = ctx->dm;
    transfer_solver.actx = ctx->actx;
    transfer_solver.app  = ctx->app;
    PetscCall(LinearSolverSetupPMGTransferChecks(&transfer_solver, ctx->dm, ctx->app));
    PetscCall(LinearSolverDestroyPMGHierarchy(&transfer_solver));
  }

  PetscCall(MatCreateVecs(A4, &ctx->tmp4, NULL));
  PetscCall(VecDuplicate(ctx->tmp4, &ctx->r4));
  PetscCall(VecDuplicate(ctx->tmp4, &ctx->z4));
  PetscCall(PMGActiveLayoutDuplicateFull(&ctx->p2_layout, &ctx->p2_rhs_full));
  PetscCall(PMGActiveLayoutDuplicateFull(&ctx->p2_layout, &ctx->p2_x_full));
  PetscCall(PMGActiveLayoutDuplicateFull(&ctx->p2_layout, &ctx->p2_r_full));
  PetscCall(PMGActiveLayoutDuplicateFull(&ctx->p2_layout, &ctx->p2_corr_full));
  PetscCall(PMGActiveLayoutDuplicateFull(&ctx->p1_layout, &ctx->p1_rhs_full));
  PetscCall(PMGActiveLayoutDuplicateFull(&ctx->p1_layout, &ctx->p1_x_full));
  PetscCall(PMGActiveLayoutDuplicateSub(&ctx->p2_layout, &ctx->p2_rhs));
  PetscCall(PMGActiveLayoutDuplicateSub(&ctx->p2_layout, &ctx->p2_x));
  PetscCall(PMGActiveLayoutDuplicateSub(&ctx->p2_layout, &ctx->p2_r));
  PetscCall(PMGActiveLayoutDuplicateSub(&ctx->p2_layout, &ctx->p2_delta));
  PetscCall(PMGActiveLayoutDuplicateSub(&ctx->p1_layout, &ctx->p1_rhs));
  PetscCall(PMGActiveLayoutDuplicateSub(&ctx->p1_layout, &ctx->p1_x));

  PetscCall(KSPCreate(comm, &ctx->smooth4));
  PetscCall(PetscSNPrintf(prefix4, sizeof(prefix4), "pmg_shell_fine_"));
  PetscCall(PMGShellConfigureSmootherKSP(ctx->smooth4, ctx->app, prefix4, A4));
  PetscCall(ReportPMGLevelSolver(comm, 2, ctx->smooth4));
  if (ctx->p2_layout.active) {
    PetscCall(KSPCreate(ctx->p2_layout.subcomm, &ctx->smooth2));
  }
  if (ctx->p1_layout.active) {
    PetscCall(KSPCreate(ctx->p1_layout.subcomm, &ctx->coarse1));
  }

  PetscCall(VecDestroy(&fine_vec));
  PetscCall(VecDestroy(&p2_vec));
  PetscCall(VecDestroy(&p1_vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellUpdateOperators(PC pc, PMGShellVCycleCtx *ctx, Mat A4)
{
  MPI_Comm       comm = PetscObjectComm((PetscObject)pc);
  MatReuse       reuse = ctx->operator_updates ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX;
  const char    *reuse_label = ctx->operator_updates ? "reuse" : "initial";
  PetscLogDouble t0, t1, t2, t3, t4, t5, sub_time, cat_time, p2_ptap, p1_ptap, p2_redist, p1_redist;

  PetscFunctionBeginUser;
  ctx->A4 = A4;
  PetscCall(PetscTime(&t0));
  PetscCall(KSPSetOperators(ctx->smooth4, A4, A4));

  PetscCall(PetscTime(&t1));
  PetscCall(MatPtAP(A4, ctx->P42, reuse, PETSC_DETERMINE, &ctx->A2_active));
  PetscCall(PetscTime(&t2));
  p2_ptap = t2 - t1;
  PetscCall(PMGShellRedistributeActiveMatrix(comm, &ctx->p2_layout, ctx->A2_active, reuse, &ctx->A2_sub, &sub_time, &cat_time));
  PetscCall(PetscTime(&t3));
  p2_redist = t3 - t2;
  if (reuse == MAT_INITIAL_MATRIX) PetscCall(PMGShellAttachSubcommNearNullspace(ctx->dm_p2, &ctx->p2_layout, ctx->A2_sub));
  if (ctx->p2_layout.active) {
    if (!ctx->smooth2) PetscCall(KSPCreate(ctx->p2_layout.subcomm, &ctx->smooth2));
    if (reuse == MAT_INITIAL_MATRIX) {
      PetscCall(PMGShellConfigureSmootherKSP(ctx->smooth2, ctx->app, "pmg_shell_p2_", ctx->A2_sub));
      PetscCall(ReportPMGLevelSolver(ctx->p2_layout.subcomm, 1, ctx->smooth2));
    } else PetscCall(KSPSetOperators(ctx->smooth2, ctx->A2_sub, ctx->A2_sub));
  }
  PetscCall(PetscPrintf(comm, "PMG_SHELL_OPERATOR_UPDATE level=1 reuse=%s ptap_time=%.6g redistribute_time=%.6g submatrix_time=%.6g concatenate_time=%.6g\n",
                        reuse_label, (double)p2_ptap, (double)p2_redist, (double)sub_time, (double)cat_time));

  PetscCall(PetscTime(&t3));
  PetscCall(MatPtAP(ctx->A2_active, ctx->P21, reuse, PETSC_DETERMINE, &ctx->A1_active));
  PetscCall(PetscTime(&t4));
  p1_ptap = t4 - t3;
  PetscCall(PMGShellRedistributeActiveMatrix(comm, &ctx->p1_layout, ctx->A1_active, reuse, &ctx->A1_sub, &sub_time, &cat_time));
  PetscCall(PetscTime(&t5));
  p1_redist = t5 - t4;
  if (reuse == MAT_INITIAL_MATRIX) PetscCall(PMGShellAttachSubcommNearNullspace(ctx->dm_p1, &ctx->p1_layout, ctx->A1_sub));
  if (ctx->p1_layout.active) {
    if (!ctx->coarse1) PetscCall(KSPCreate(ctx->p1_layout.subcomm, &ctx->coarse1));
    if (reuse == MAT_INITIAL_MATRIX) {
      PetscCall(PMGShellConfigureCoarseKSP(ctx->coarse1, ctx->app, "pmg_shell_p1_", ctx->A1_sub));
      PetscCall(ReportPMGLevelSolver(ctx->p1_layout.subcomm, 0, ctx->coarse1));
    } else PetscCall(KSPSetOperators(ctx->coarse1, ctx->A1_sub, ctx->A1_sub));
  }
  PetscCall(PetscPrintf(comm, "PMG_SHELL_OPERATOR_UPDATE level=0 reuse=%s ptap_time=%.6g redistribute_time=%.6g submatrix_time=%.6g concatenate_time=%.6g\n",
                        reuse_label, (double)p1_ptap, (double)p1_redist, (double)sub_time, (double)cat_time));
  ctx->operator_updates++;
  ctx->operator_update_time += t5 - t0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellResidual(Mat A, Vec rhs, Vec x, Vec r, Vec tmp, PetscLogDouble *accum_time)
{
  PetscLogDouble t0, t1;

  PetscFunctionBeginUser;
  PetscCall(PetscLogStagePush(log_stage_pmg_shell_residual));
  PetscCall(PetscTime(&t0));
  PetscCall(MatMult(A, x, tmp));
  PetscCall(VecWAXPY(r, -1.0, tmp, rhs));
  PetscCall(PetscTime(&t1));
  *accum_time += t1 - t0;
  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellDebugCapture(PMGShellVCycleCtx *ctx, Vec src, Vec *dst)
{
  PetscFunctionBeginUser;
  if (!ctx->debug_capture || !src) PetscFunctionReturn(PETSC_SUCCESS);
  if (!*dst) PetscCall(VecDuplicate(src, dst));
  PetscCall(VecCopy(src, *dst));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellVCycleSetUp(PC pc)
{
  PMGShellVCycleCtx *ctx = NULL;
  Mat                A = NULL;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, (void **)&ctx));
  PetscCall(PCGetOperators(pc, NULL, &A));
  PetscCheck(A, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "PMG shell V-cycle requires a fine operator");
  if (!ctx->setup_done) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pc), "PMG_BACKEND backend=shell_vcycle\n"));
    PetscCall(PMGShellCreateHierarchy(pc, ctx, A));
    ctx->setup_done = PETSC_TRUE;
  }
  PetscCall(PMGShellUpdateOperators(pc, ctx, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellVCycleApply(PC pc, Vec x, Vec y)
{
  PMGShellVCycleCtx *ctx = NULL;
  PetscLogDouble     t0, t1;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, (void **)&ctx));
  PetscCall(VecZeroEntries(y));

  PetscCall(PetscLogStagePush(log_stage_pmg_shell_fine_smooth));
  PetscCall(PetscTime(&t0));
  PetscCall(KSPSolve(ctx->smooth4, x, y));
  PetscCall(PetscTime(&t1));
  ctx->fine_smooth_time += t1 - t0;
  PetscCall(PMGShellDebugCapture(ctx, y, &ctx->debug_fine_pre));
  PetscCall(PetscLogStagePop());

  PetscCall(PMGShellResidual(ctx->A4, x, y, ctx->r4, ctx->tmp4, &ctx->residual_time));
  PetscCall(PMGShellDebugCapture(ctx, ctx->r4, &ctx->debug_fine_residual));
  PetscCall(PetscLogStagePush(log_stage_pmg_shell_transfer));
  PetscCall(PetscTime(&t0));
  PetscCall(MatMultTranspose(ctx->P42, ctx->r4, ctx->p2_rhs_full));
  PetscCall(PetscTime(&t1));
  ctx->restrict_time += t1 - t0;
  PetscCall(PMGActiveLayoutCopyFullToSub(&ctx->p2_layout, ctx->p2_rhs_full, ctx->p2_rhs));
  PetscCall(PMGShellDebugCapture(ctx, ctx->p2_rhs, &ctx->debug_p2_rhs));
  PetscCall(PetscLogStagePop());

  if (ctx->p2_layout.active) {
    PetscCall(VecZeroEntries(ctx->p2_x));
    PetscCall(PetscLogStagePush(log_stage_pmg_shell_p2));
    PetscCall(PetscTime(&t0));
    PetscCall(KSPSolve(ctx->smooth2, ctx->p2_rhs, ctx->p2_x));
    PetscCall(PetscTime(&t1));
    ctx->p2_smooth_time += t1 - t0;
    PetscCall(PMGShellDebugCapture(ctx, ctx->p2_x, &ctx->debug_p2_pre));
    PetscCall(PetscLogStagePop());
    PetscCall(PMGShellResidual(ctx->A2_sub, ctx->p2_rhs, ctx->p2_x, ctx->p2_r, ctx->p2_delta, &ctx->residual_time));
    PetscCall(PMGShellDebugCapture(ctx, ctx->p2_r, &ctx->debug_p2_residual));
  }
  PetscCall(PetscLogStagePush(log_stage_pmg_shell_transfer));
  PetscCall(PMGActiveLayoutCopySubToFull(&ctx->p2_layout, ctx->p2_r, ctx->p2_r_full));
  PetscCall(PetscTime(&t0));
  PetscCall(MatMultTranspose(ctx->P21, ctx->p2_r_full, ctx->p1_rhs_full));
  PetscCall(PetscTime(&t1));
  ctx->restrict_time += t1 - t0;
  PetscCall(PMGActiveLayoutCopyFullToSub(&ctx->p1_layout, ctx->p1_rhs_full, ctx->p1_rhs));
  PetscCall(PMGShellDebugCapture(ctx, ctx->p1_rhs, &ctx->debug_p1_rhs));
  PetscCall(PetscLogStagePop());

  if (ctx->p1_layout.active) {
    PetscCall(VecZeroEntries(ctx->p1_x));
    PetscCall(PetscLogStagePush(log_stage_pmg_shell_p1));
    PetscCall(PetscTime(&t0));
    PetscCall(KSPSolve(ctx->coarse1, ctx->p1_rhs, ctx->p1_x));
    PetscCall(PetscTime(&t1));
    ctx->coarse_solve_time += t1 - t0;
    PetscCall(PMGShellDebugCapture(ctx, ctx->p1_x, &ctx->debug_p1_x));
    PetscCall(PetscLogStagePop());
  }
  PetscCall(PetscLogStagePush(log_stage_pmg_shell_transfer));
  PetscCall(PMGActiveLayoutCopySubToFull(&ctx->p1_layout, ctx->p1_x, ctx->p1_x_full));
  PetscCall(PetscTime(&t0));
  PetscCall(MatMult(ctx->P21, ctx->p1_x_full, ctx->p2_corr_full));
  PetscCall(PetscTime(&t1));
  ctx->prolong_time += t1 - t0;
  PetscCall(PMGActiveLayoutCopyFullToSub(&ctx->p2_layout, ctx->p2_corr_full, ctx->p2_delta));
  PetscCall(PetscLogStagePop());
  if (ctx->p2_layout.active) {
    PetscCall(VecAXPY(ctx->p2_x, 1.0, ctx->p2_delta));
    PetscCall(PetscLogStagePush(log_stage_pmg_shell_p2));
    PetscCall(PetscTime(&t0));
    PetscCall(KSPSolve(ctx->smooth2, ctx->p2_rhs, ctx->p2_x));
    PetscCall(PetscTime(&t1));
    ctx->p2_smooth_time += t1 - t0;
    PetscCall(PMGShellDebugCapture(ctx, ctx->p2_x, &ctx->debug_p2_post));
    PetscCall(PetscLogStagePop());
  }
  PetscCall(PetscLogStagePush(log_stage_pmg_shell_transfer));
  PetscCall(PMGActiveLayoutCopySubToFull(&ctx->p2_layout, ctx->p2_x, ctx->p2_corr_full));
  PetscCall(PetscTime(&t0));
  PetscCall(MatMult(ctx->P42, ctx->p2_corr_full, ctx->z4));
  PetscCall(VecAXPY(y, 1.0, ctx->z4));
  PetscCall(PetscTime(&t1));
  ctx->prolong_time += t1 - t0;
  PetscCall(PetscLogStagePop());

  PetscCall(PetscLogStagePush(log_stage_pmg_shell_fine_smooth));
  PetscCall(PetscTime(&t0));
  PetscCall(KSPSolve(ctx->smooth4, x, y));
  PetscCall(PetscTime(&t1));
  ctx->fine_smooth_time += t1 - t0;
  PetscCall(PetscLogStagePop());
  ctx->apply_calls++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellVCycleDestroy(PC pc)
{
  PMGShellVCycleCtx *ctx = NULL;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, (void **)&ctx));
  if (!ctx) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pc),
                        "PMG_SHELL_APPLY_SUMMARY apply_calls=%" PetscInt_FMT " operator_updates=%" PetscInt_FMT
                        " fine_smooth=%.6g p2_smooth=%.6g restrict=%.6g prolong=%.6g coarse_solve=%.6g residual=%.6g operator_update=%.6g\n",
                        ctx->apply_calls, ctx->operator_updates, (double)ctx->fine_smooth_time, (double)ctx->p2_smooth_time,
                        (double)ctx->restrict_time, (double)ctx->prolong_time, (double)ctx->coarse_solve_time, (double)ctx->residual_time,
                        (double)ctx->operator_update_time));
  PetscCall(KSPDestroy(&ctx->smooth4));
  PetscCall(KSPDestroy(&ctx->smooth2));
  PetscCall(KSPDestroy(&ctx->coarse1));
  PetscCall(MatDestroy(&ctx->P42));
  PetscCall(MatDestroy(&ctx->P21));
  PetscCall(MatDestroy(&ctx->A2_active));
  PetscCall(MatDestroy(&ctx->A1_active));
  PetscCall(MatDestroy(&ctx->A2_sub));
  PetscCall(MatDestroy(&ctx->A1_sub));
  PetscCall(VecDestroy(&ctx->r4));
  PetscCall(VecDestroy(&ctx->tmp4));
  PetscCall(VecDestroy(&ctx->z4));
  PetscCall(VecDestroy(&ctx->p2_rhs_full));
  PetscCall(VecDestroy(&ctx->p2_x_full));
  PetscCall(VecDestroy(&ctx->p2_r_full));
  PetscCall(VecDestroy(&ctx->p2_corr_full));
  PetscCall(VecDestroy(&ctx->p1_rhs_full));
  PetscCall(VecDestroy(&ctx->p1_x_full));
  PetscCall(VecDestroy(&ctx->p2_rhs));
  PetscCall(VecDestroy(&ctx->p2_x));
  PetscCall(VecDestroy(&ctx->p2_r));
  PetscCall(VecDestroy(&ctx->p2_delta));
  PetscCall(VecDestroy(&ctx->p1_rhs));
  PetscCall(VecDestroy(&ctx->p1_x));
  PetscCall(VecDestroy(&ctx->debug_fine_pre));
  PetscCall(VecDestroy(&ctx->debug_fine_residual));
  PetscCall(VecDestroy(&ctx->debug_p2_rhs));
  PetscCall(VecDestroy(&ctx->debug_p2_pre));
  PetscCall(VecDestroy(&ctx->debug_p2_residual));
  PetscCall(VecDestroy(&ctx->debug_p1_rhs));
  PetscCall(VecDestroy(&ctx->debug_p1_x));
  PetscCall(VecDestroy(&ctx->debug_p2_post));
  PetscCall(PMGActiveLayoutDestroy(&ctx->p2_layout));
  PetscCall(PMGActiveLayoutDestroy(&ctx->p1_layout));
  PetscCall(DMDestroy(&ctx->dm_p1));
  PetscCall(DMDestroy(&ctx->dm_p2));
  if (ctx->p1_basis_created) PetscCall(P4BasisDestroy(&ctx->p1_basis));
  if (ctx->p2_basis_created) PetscCall(P4BasisDestroy(&ctx->p2_basis));
  PetscCall(PetscFree(ctx));
  PetscCall(PCShellSetContext(pc, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigurePMGShellVCycle(PC pc, DM dm, AssemblyCtx *actx, AppCtx *app)
{
  PMGShellVCycleCtx *ctx = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&ctx));
  ctx->app  = app;
  ctx->dm   = dm;
  ctx->actx = actx;
  ctx->p2_layout.subcomm = MPI_COMM_NULL;
  ctx->p1_layout.subcomm = MPI_COMM_NULL;
  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetContext(pc, ctx));
  PetscCall(PCShellSetName(pc, "pmg_shell_vcycle"));
  PetscCall(PCShellSetSetUp(pc, PMGShellVCycleSetUp));
  PetscCall(PCShellSetApply(pc, PMGShellVCycleApply));
  PetscCall(PCShellSetDestroy(pc, PMGShellVCycleDestroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverDestroyPMGHierarchy(LinearSolverCtx *solver)
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&solver->pmg_u_p1));
  PetscCall(VecDestroy(&solver->pmg_u_p2));
  PetscCall(MatDestroy(&solver->pmg_inject_p4_to_p2));
  PetscCall(MatDestroy(&solver->pmg_inject_p2_to_p1));
  PetscCall(DMDestroy(&solver->pmg_dm_p1));
  PetscCall(DMDestroy(&solver->pmg_dm_p2));
  if (solver->pmg_p1_basis_created) {
    PetscCall(P4BasisDestroy(&solver->pmg_p1_basis));
    solver->pmg_p1_basis_created = PETSC_FALSE;
  }
  if (solver->pmg_p2_basis_created) {
    PetscCall(P4BasisDestroy(&solver->pmg_p2_basis));
    solver->pmg_p2_basis_created = PETSC_FALSE;
  }
  solver->pmg_hierarchy_initialized       = PETSC_FALSE;
  solver->pmg_inject_p4_to_p2_transpose   = PETSC_FALSE;
  solver->pmg_inject_p2_to_p1_transpose   = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckPMGTransferShape(const char name[], Mat mat, PetscInt coarse_size, PetscInt fine_size, PetscBool *use_transpose)
{
  MPI_Comm comm = PetscObjectComm((PetscObject)mat);
  PetscInt rows, cols;

  PetscFunctionBeginUser;
  PetscCall(MatGetSize(mat, &rows, &cols));
  if (rows == coarse_size && cols == fine_size) {
    *use_transpose = PETSC_FALSE;
  } else if (rows == fine_size && cols == coarse_size) {
    *use_transpose = PETSC_TRUE;
  } else {
    PetscCheck(PETSC_FALSE, comm, PETSC_ERR_PLIB,
               "%s coarse transfer has shape %" PetscInt_FMT "x%" PetscInt_FMT ", expected %" PetscInt_FMT "x%" PetscInt_FMT " or transpose",
               name, rows, cols, coarse_size, fine_size);
  }
  PetscCall(PetscPrintf(comm,
                        "PMG_COARSE_TRANSFER name=%s rows=%" PetscInt_FMT " cols=%" PetscInt_FMT " coarse_dofs=%" PetscInt_FMT " fine_dofs=%" PetscInt_FMT " use_transpose=%s\n",
                        name, rows, cols, coarse_size, fine_size, *use_transpose ? "true" : "false"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGTransferConstant(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  (void)dim; (void)time; (void)x; (void)ctx;
  PetscCheck(Nc == 3, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Expected three displacement components");
  u[0] = 1.25;
  u[1] = -2.0;
  u[2] = 0.5;
  return PETSC_SUCCESS;
}

static PetscErrorCode PMGTransferAffine(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  (void)dim; (void)time; (void)ctx;
  PetscCheck(Nc == 3, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Expected three displacement components");
  u[0] = 1.0 + 2.0 * x[0] - 0.5 * x[1] + 0.25 * x[2];
  u[1] = -0.25 + x[0] + 0.75 * x[1] - 0.1 * x[2];
  u[2] = 0.5 - 0.2 * x[0] + 0.4 * x[1] + 1.3 * x[2];
  return PETSC_SUCCESS;
}

static PetscErrorCode PMGTransferQuadratic(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  (void)dim; (void)time; (void)ctx;
  PetscCheck(Nc == 3, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Expected three displacement components");
  u[0] = 1.0 + x[0] * x[0] + 2.0 * x[1] * x[2];
  u[1] = -0.3 + x[0] * x[1] + x[2] * x[2];
  u[2] = x[0] * x[2] + x[1] * x[1];
  return PETSC_SUCCESS;
}

static PetscErrorCode ApplyPMGTransfer(Mat transfer, PetscBool use_transpose, Vec fine, Vec coarse)
{
  PetscFunctionBeginUser;
  if (use_transpose) {
    PetscCall(MatMultTranspose(transfer, fine, coarse));
  } else {
    PetscCall(MatMult(transfer, fine, coarse));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckOnePMGTransfer(const char name[], const char test[], DM fine_dm, DM coarse_dm, Mat transfer, PetscBool use_transpose,
                                          PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *))
{
  MPI_Comm comm = PetscObjectComm((PetscObject)fine_dm);
  PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {func};
  Vec       fine = NULL, coarse_exact = NULL, coarse_actual = NULL, diff = NULL;
  PetscReal exact_l2, diff_l2, diff_inf, rel_l2;

  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(fine_dm, &fine));
  PetscCall(DMCreateGlobalVector(coarse_dm, &coarse_exact));
  PetscCall(VecDuplicate(coarse_exact, &coarse_actual));
  PetscCall(VecDuplicate(coarse_exact, &diff));
  PetscCall(DMProjectFunction(fine_dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, fine));
  PetscCall(DMProjectFunction(coarse_dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, coarse_exact));
  PetscCall(ApplyPMGTransfer(transfer, use_transpose, fine, coarse_actual));
  PetscCall(VecCopy(coarse_actual, diff));
  PetscCall(VecAXPY(diff, -1.0, coarse_exact));
  PetscCall(VecNorm(coarse_exact, NORM_2, &exact_l2));
  PetscCall(VecNorm(diff, NORM_2, &diff_l2));
  PetscCall(VecNorm(diff, NORM_INFINITY, &diff_inf));
  rel_l2 = exact_l2 > 0.0 ? diff_l2 / exact_l2 : diff_l2;
  PetscCall(PetscPrintf(comm,
                        "PMG_COARSE_TRANSFER_CHECK name=%s test=%s abs_l2=%.6e rel_l2=%.6e abs_inf=%.6e\n",
                        name, test, (double)diff_l2, (double)rel_l2, (double)diff_inf));
  PetscCall(VecDestroy(&diff));
  PetscCall(VecDestroy(&coarse_actual));
  PetscCall(VecDestroy(&coarse_exact));
  PetscCall(VecDestroy(&fine));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckCompositePMGTransfer(LinearSolverCtx *solver, const char test[],
                                                PetscErrorCode (*func)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *))
{
  MPI_Comm comm = PetscObjectComm((PetscObject)solver->dm);
  PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {func};
  Vec       fine = NULL, coarse_exact = NULL, diff = NULL;
  PetscReal exact_l2, diff_l2, diff_inf, rel_l2;

  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(solver->dm, &fine));
  PetscCall(DMCreateGlobalVector(solver->pmg_dm_p1, &coarse_exact));
  PetscCall(VecDuplicate(coarse_exact, &diff));
  PetscCall(DMProjectFunction(solver->dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, fine));
  PetscCall(DMProjectFunction(solver->pmg_dm_p1, 0.0, funcs, NULL, INSERT_ALL_VALUES, coarse_exact));
  PetscCall(ApplyPMGTransfer(solver->pmg_inject_p4_to_p2, solver->pmg_inject_p4_to_p2_transpose, fine, solver->pmg_u_p2));
  PetscCall(ApplyPMGTransfer(solver->pmg_inject_p2_to_p1, solver->pmg_inject_p2_to_p1_transpose, solver->pmg_u_p2, solver->pmg_u_p1));
  PetscCall(VecCopy(solver->pmg_u_p1, diff));
  PetscCall(VecAXPY(diff, -1.0, coarse_exact));
  PetscCall(VecNorm(coarse_exact, NORM_2, &exact_l2));
  PetscCall(VecNorm(diff, NORM_2, &diff_l2));
  PetscCall(VecNorm(diff, NORM_INFINITY, &diff_inf));
  rel_l2 = exact_l2 > 0.0 ? diff_l2 / exact_l2 : diff_l2;
  PetscCall(PetscPrintf(comm,
                        "PMG_COARSE_TRANSFER_CHECK name=p4_to_p1_composite test=%s abs_l2=%.6e rel_l2=%.6e abs_inf=%.6e\n",
                        test, (double)diff_l2, (double)rel_l2, (double)diff_inf));
  PetscCall(VecDestroy(&diff));
  PetscCall(VecDestroy(&coarse_exact));
  PetscCall(VecDestroy(&fine));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckPMGCoarseTransfers(LinearSolverCtx *solver)
{
  PetscFunctionBeginUser;
  PetscCall(CheckOnePMGTransfer("p4_to_p2", "constant", solver->dm, solver->pmg_dm_p2, solver->pmg_inject_p4_to_p2, solver->pmg_inject_p4_to_p2_transpose, PMGTransferConstant));
  PetscCall(CheckOnePMGTransfer("p4_to_p2", "affine", solver->dm, solver->pmg_dm_p2, solver->pmg_inject_p4_to_p2, solver->pmg_inject_p4_to_p2_transpose, PMGTransferAffine));
  PetscCall(CheckOnePMGTransfer("p4_to_p2", "quadratic", solver->dm, solver->pmg_dm_p2, solver->pmg_inject_p4_to_p2, solver->pmg_inject_p4_to_p2_transpose, PMGTransferQuadratic));
  PetscCall(CheckOnePMGTransfer("p2_to_p1", "constant", solver->pmg_dm_p2, solver->pmg_dm_p1, solver->pmg_inject_p2_to_p1, solver->pmg_inject_p2_to_p1_transpose, PMGTransferConstant));
  PetscCall(CheckOnePMGTransfer("p2_to_p1", "affine", solver->pmg_dm_p2, solver->pmg_dm_p1, solver->pmg_inject_p2_to_p1, solver->pmg_inject_p2_to_p1_transpose, PMGTransferAffine));
  PetscCall(CheckCompositePMGTransfer(solver, "constant", PMGTransferConstant));
  PetscCall(CheckCompositePMGTransfer(solver, "affine", PMGTransferAffine));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverSetupPMGTransferChecks(LinearSolverCtx *solver, DM dm, const AppCtx *app)
{
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);
  Vec      v4 = NULL, v2 = NULL, v1 = NULL;
  PetscInt n4, n2, n1;

  PetscFunctionBeginUser;
  PetscCall(LinearSolverDestroyPMGHierarchy(solver));
  PetscCall(P4BasisCreateDegree(PETSC_COMM_SELF, 1, &solver->pmg_p1_basis));
  solver->pmg_p1_basis_created = PETSC_TRUE;
  PetscCall(P4BasisCreateDegree(PETSC_COMM_SELF, 2, &solver->pmg_p2_basis));
  solver->pmg_p2_basis_created = PETSC_TRUE;
  PetscCall(CreateSameMeshLevelDM(dm, &solver->pmg_p1_basis, app, &solver->pmg_dm_p1));
  PetscCall(CreateSameMeshLevelDM(dm, &solver->pmg_p2_basis, app, &solver->pmg_dm_p2));
  PetscCall(DMCreateGlobalVector(solver->pmg_dm_p1, &solver->pmg_u_p1));
  PetscCall(DMCreateGlobalVector(solver->pmg_dm_p2, &solver->pmg_u_p2));
  PetscCall(BuildInterpolationMatrix(solver->pmg_dm_p2, &solver->pmg_p2_basis, dm, solver->actx->basis, &solver->pmg_inject_p4_to_p2));
  PetscCall(BuildInterpolationMatrix(solver->pmg_dm_p1, &solver->pmg_p1_basis, solver->pmg_dm_p2, &solver->pmg_p2_basis, &solver->pmg_inject_p2_to_p1));

  PetscCall(DMCreateGlobalVector(dm, &v4));
  PetscCall(DMCreateGlobalVector(solver->pmg_dm_p2, &v2));
  PetscCall(DMCreateGlobalVector(solver->pmg_dm_p1, &v1));
  PetscCall(VecGetSize(v4, &n4));
  PetscCall(VecGetSize(v2, &n2));
  PetscCall(VecGetSize(v1, &n1));
  PetscCall(CheckPMGTransferShape("p4_to_p2", solver->pmg_inject_p4_to_p2, n2, n4, &solver->pmg_inject_p4_to_p2_transpose));
  PetscCall(CheckPMGTransferShape("p2_to_p1", solver->pmg_inject_p2_to_p1, n1, n2, &solver->pmg_inject_p2_to_p1_transpose));
  PetscCall(VecDestroy(&v4));
  PetscCall(VecDestroy(&v2));
  PetscCall(VecDestroy(&v1));

  solver->pmg_hierarchy_initialized = PETSC_TRUE;
  if (app->pmg_check_coarse_transfers) PetscCall(CheckPMGCoarseTransfers(solver));
  PetscCall(PetscPrintf(comm,
                        "PMG_COARSE_TRANSFER_CONFIG p1_quadrature_points=%" PetscInt_FMT " p2_quadrature_points=%" PetscInt_FMT " p1_global_dofs=%" PetscInt_FMT " p2_global_dofs=%" PetscInt_FMT "\n",
                        solver->pmg_p1_basis.n_qp, solver->pmg_p2_basis.n_qp, n1, n2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigurePMG(PC pc, DM dm, AssemblyCtx *actx, AppCtx *app, LinearSolverCtx *solver)
{
  P4Basis p1_basis, p2_basis;
  DM      dm_p1 = NULL, dm_p2 = NULL;
  Mat     P21 = NULL, P42 = NULL;
  KSP     coarse = NULL, smoother = NULL;
  PC      coarse_pc = NULL, smoother_pc = NULL;
  char    coarse_pc_type[32];
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);
  PetscMPIInt ranks;
  PetscBool coarse_pc_from_options = PETSC_FALSE, coarse_is_lu = PETSC_FALSE, coarse_is_telescope = PETSC_FALSE, coarse_is_shell = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &ranks));
  PetscCall(PetscPrintf(comm, "PMG_BACKEND backend=pcmg\n"));
  PetscCall(P4BasisCreateDegree(PETSC_COMM_SELF, 1, &p1_basis));
  PetscCall(P4BasisCreateDegree(PETSC_COMM_SELF, 2, &p2_basis));
  PetscCall(CreateSameMeshLevelDM(dm, &p1_basis, app, &dm_p1));
  PetscCall(CreateSameMeshLevelDM(dm, &p2_basis, app, &dm_p2));
  PetscCall(ReportPMGLevelDofs(dm_p1, 0, 1));
  PetscCall(ReportPMGLevelDofs(dm_p2, 1, 2));
  PetscCall(ReportPMGLevelDofs(dm, 2, 4));
  PetscCall(BuildInterpolationMatrix(dm_p2, &p2_basis, dm_p1, &p1_basis, &P21));
  PetscCall(BuildInterpolationMatrix(dm, actx->basis, dm_p2, &p2_basis, &P42));
  PetscCall(SetPMGTelescopeDefaults(app, comm));
  PetscCall(SetPMGP2TelescopeDefaults(app, comm));
  PetscCall(ChoosePMGCoarsePC(app, dm_p1, coarse_pc_type, sizeof(coarse_pc_type)));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-mg_coarse_pc_type", &coarse_pc_from_options));
  PetscCall(PetscStrcasecmp(coarse_pc_type, "lu", &coarse_is_lu));
  PetscCall(PetscStrcasecmp(coarse_pc_type, "telescope", &coarse_is_telescope));
  PetscCall(PetscStrcasecmp(coarse_pc_type, "shell", &coarse_is_shell));
  PetscCheck(!coarse_is_shell, comm, PETSC_ERR_ARG_WRONG,
             "The removed PMG coarse shell experiment is no longer supported; use -pmg_apply_backend shell_vcycle for the maintained shell V-cycle backend");

  PetscCall(PCSetType(pc, PCMG));
  PetscCall(PCMGSetLevels(pc, 3, NULL));
  PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
  PetscCall(PCMGSetCycleType(pc, PC_MG_CYCLE_V));
  PetscCall(PCMGSetInterpolation(pc, 1, P21));
  PetscCall(PCMGSetInterpolation(pc, 2, P42));
  PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));
  PetscCall(PetscPrintf(comm, "PMG_COARSE_OPERATOR_CONFIG type=galerkin p1_quadrature_points=%" PetscInt_FMT " p2_quadrature_points=%" PetscInt_FMT "\n",
                        p1_basis.n_qp, p2_basis.n_qp));
  if (app->pmg_check_coarse_transfers) PetscCall(LinearSolverSetupPMGTransferChecks(solver, dm, app));

  PetscCall(PCMGGetCoarseSolve(pc, &coarse));
  PetscCall(KSPGetPC(coarse, &coarse_pc));
  if (!coarse_pc_from_options && !coarse_is_lu && app->pmg_coarse_redundant_group_size > 0 && ranks > app->pmg_coarse_redundant_group_size) {
    KSP      inner_ksp;
    PC       inner_pc;
    PetscInt copies = ((PetscInt)ranks + app->pmg_coarse_redundant_group_size - 1) / app->pmg_coarse_redundant_group_size;

    PetscCall(KSPSetType(coarse, KSPPREONLY));
    PetscCall(PCSetType(coarse_pc, PCREDUNDANT));
    PetscCall(PCRedundantSetNumber(coarse_pc, copies));
    PetscCall(PCRedundantGetKSP(coarse_pc, &inner_ksp));
    PetscCall(KSPSetType(inner_ksp, KSPFGMRES));
    PetscCall(KSPSetTolerances(inner_ksp, 1.0e-3, PETSC_DEFAULT, PETSC_DEFAULT, 100));
    PetscCall(KSPGetPC(inner_ksp, &inner_pc));
    PetscCall(ConfigurePMGBasePC(inner_pc, coarse_pc_type, app->pmg_coarse_gamg_aggressive_square_graph));
    PetscCall(PetscPrintf(comm, "PMG_REDUNDANT_CONFIG enabled=true group_size=%" PetscInt_FMT " copies=%" PetscInt_FMT " inner_ksp=fgmres inner_pc=%s\n",
                          app->pmg_coarse_redundant_group_size, copies, coarse_pc_type));
    PetscCall(PetscPrintf(comm,
                          "PMG_COARSE_SOLVE type=redundant group_size=%" PetscInt_FMT " copies=%" PetscInt_FMT " inner_pc=%s aggressive_square_graph=%s\n",
                          app->pmg_coarse_redundant_group_size, copies, coarse_pc_type, app->pmg_coarse_gamg_aggressive_square_graph ? "true" : "false"));
  } else {
    PetscCall(PetscPrintf(comm, "PMG_REDUNDANT_CONFIG enabled=false group_size=%" PetscInt_FMT " copies=0 inner_ksp=none inner_pc=none\n",
                          app->pmg_coarse_redundant_group_size));
    if (coarse_is_lu || coarse_is_telescope) {
      PetscCall(KSPSetType(coarse, KSPPREONLY));
    } else {
      PetscCall(KSPSetType(coarse, KSPFGMRES));
      PetscCall(KSPSetTolerances(coarse, 1.0e-3, PETSC_DEFAULT, PETSC_DEFAULT, 100));
      PetscCall(KSPGMRESSetRestart(coarse, 100));
    }
    PetscCall(ConfigurePMGBasePC(coarse_pc, coarse_pc_type, app->pmg_coarse_gamg_aggressive_square_graph));
    PetscCall(PetscPrintf(comm, "PMG_COARSE_SOLVE type=%s group_size=%" PetscInt_FMT " aggressive_square_graph=%s\n", coarse_pc_type,
                          app->pmg_coarse_redundant_group_size, app->pmg_coarse_gamg_aggressive_square_graph ? "true" : "false"));
  }
  for (PetscInt level = 1; level < 3; ++level) {
    PetscCall(PCMGGetSmoother(pc, level, &smoother));
    PetscCall(KSPSetType(smoother, app->pmg_smoother_ksp_type));
    PetscCall(KSPSetTolerances(smoother, 0.0, 0.0, PETSC_CURRENT, app->pmg_smoother_max_it));
    PetscCall(KSPGetPC(smoother, &smoother_pc));
    PetscCall(PCSetType(smoother_pc, app->pmg_smoother_pc_type));
  }

  PetscCall(MatDestroy(&P21));
  PetscCall(MatDestroy(&P42));
  PetscCall(DMDestroy(&dm_p1));
  PetscCall(DMDestroy(&dm_p2));
  PetscCall(P4BasisDestroy(&p1_basis));
  PetscCall(P4BasisDestroy(&p2_basis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RefreshKSPOperators(KSP ksp, AppCtx *app, Mat A, PetscBool report_effective_tolerance, PetscBool reuse_preconditioner)
{
  PetscReal solver_rtol = app->ksp_rtol;

  PetscFunctionBeginUser;
  PetscCall(KSPSetOperators(ksp, A, A));
  if (app->variant == VARIANT_FETIDP) solver_rtol *= 1.0e-2;
  PetscCall(KSPSetTolerances(ksp, solver_rtol, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT));
  if (report_effective_tolerance && solver_rtol != app->ksp_rtol) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ksp), "FETI-DP effective multiplier-space rtol=%.6e requested_primal_rtol=%.6e\n", (double)solver_rtol, (double)app->ksp_rtol));
  }
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_FALSE));
  PetscCall(KSPSetReusePreconditioner(ksp, reuse_preconditioner));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigureKSP(KSP ksp, DM dm, AssemblyCtx *actx, AppCtx *app, Mat A, PetscBool nonlinear_tangent, LinearSolverCtx *solver)
{
  PC        pc;
  PetscReal solver_rtol = app->ksp_rtol;

  (void)nonlinear_tangent;
  PetscFunctionBeginUser;
  if (app->variant == VARIANT_FETIDP) solver_rtol *= 1.0e-2;
  PetscCall(RefreshKSPOperators(ksp, app, A, PETSC_TRUE, PETSC_FALSE));
  if (app->variant == VARIANT_BDDC) {
    PetscCall(SetBDDCConstraintDefaults(app, "pc_bddc_"));
    PetscCall(ConfigureBDDCAutoSolvers(app, A, "pc_bddc_"));
  } else if (app->variant == VARIANT_FETIDP) {
    PetscCall(SetBDDCConstraintDefaults(app, "fetidp_bddc_pc_bddc_"));
    PetscCall(ConfigureBDDCAutoSolvers(app, A, "fetidp_bddc_pc_bddc_"));
  }
  if (app->variant == VARIANT_FETIDP) {
    PetscCall(SetFETIDPDefaults(solver_rtol));
    PetscCall(KSPSetType(ksp, KSPFETIDP));
  } else {
    PetscCall(KSPSetType(ksp, (app->variant == VARIANT_PMG || app->variant == VARIANT_BDDC) ? KSPFGMRES : KSPCG));
    if (app->variant == VARIANT_GAMG) PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
    if (app->variant == VARIANT_BDDC) PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
    PetscCall(KSPGetPC(ksp, &pc));
    if (app->variant == VARIANT_GAMG) {
      PetscCall(PCSetType(pc, PCGAMG));
    } else if (app->variant == VARIANT_BDDC) {
      PetscCall(PCSetType(pc, PCBDDC));
    } else if (app->variant == VARIANT_PMG) {
      PetscBool use_shell_vcycle = PETSC_FALSE;

      PetscCall(PMGApplyBackendIsShell(app, &use_shell_vcycle));
      if (use_shell_vcycle) PetscCall(ConfigurePMGShellVCycle(pc, dm, actx, app));
      else PetscCall(ConfigurePMG(pc, dm, actx, app, solver));
    } else {
      PetscCall(PCSetType(pc, PCNONE));
    }
  }
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(ReportPMGSolverChoices(ksp, app));
  if (app->variant == VARIANT_FETIDP) {
    PC inner_bddc = NULL;

    PetscCall(KSPFETIDPGetInnerBDDC(ksp, &inner_bddc));
    if (!inner_bddc) {
      PetscCall(PCCreate(PetscObjectComm((PetscObject)ksp), &inner_bddc));
      PetscCall(PrepareInnerBDDCFromOptions(inner_bddc, "fetidp_bddc_"));
      PetscCall(ConfigureBDDC(inner_bddc, dm, actx, app, A));
      PetscCall(KSPFETIDPSetInnerBDDC(ksp, inner_bddc));
      PetscCall(PCDestroy(&inner_bddc));
    } else {
      PetscCall(PrepareInnerBDDCFromOptions(inner_bddc, "fetidp_bddc_"));
      PetscCall(ConfigureBDDC(inner_bddc, dm, actx, app, A));
    }
  } else {
    PCType pctype = NULL;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCGetType(pc, &pctype));
    if (pctype) {
      PetscBool is_gamg, is_bddc;
      PetscCall(PetscStrcmp(pctype, PCGAMG, &is_gamg));
      PetscCall(PetscStrcmp(pctype, PCBDDC, &is_bddc));
      if (is_gamg) {
        PetscCall(AttachNearNullspace(dm, actx->constrained_is, A));
      } else if (is_bddc) {
        PetscCall(ConfigureBDDC(pc, dm, actx, app, A));
      } else if (app->variant == VARIANT_PMG) {
        PetscCall(AttachNearNullspace(dm, actx->constrained_is, A));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ResidualNormFree(AssemblyCtx *actx, Vec residual, PetscReal rhs_norm, PetscReal *rel)
{
  PetscReal norm;

  PetscFunctionBeginUser;
  PetscCall(ZeroConstrainedVector(actx->constrained_is, residual));
  PetscCall(VecNorm(residual, NORM_2, &norm));
  *rel = norm / rhs_norm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverClearOrthBasis(LinearSolverCtx *solver);
static PetscErrorCode LinearSolverClearRecycleBasis(LinearSolverCtx *solver);

static PetscErrorCode LinearSolverInit(LinearSolverCtx *solver, DM dm, AssemblyCtx *actx, AppCtx *app, Mat A)
{
  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(solver, sizeof(*solver)));
  solver->dm    = dm;
  solver->actx  = actx;
  solver->app   = app;
  solver->A     = A;
  solver->reuse = app->reuse_linear_solver;
  solver->recycle_temp_start_raw = -1;
  solver->force_reuse_preconditioner = PETSC_FALSE;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "LINEAR_SOLVER_REUSE enabled=%s\n", solver->reuse ? "true" : "false"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverDestroy(LinearSolverCtx *solver)
{
  PetscFunctionBeginUser;
  PetscCall(LinearSolverClearOrthBasis(solver));
  PetscCall(LinearSolverClearRecycleBasis(solver));
  for (PetscInt i = 0; i < solver->n_raw_basis; ++i) PetscCall(VecDestroy(&solver->raw_basis[i]));
  PetscCall(PetscFree(solver->raw_basis));
  PetscCall(PetscFree(solver->raw_basis_tol));
  PetscCall(PetscFree(solver->orth_basis));
  PetscCall(PetscFree(solver->left_basis));
  PetscCall(PetscFree(solver->Aorth_basis));
  PetscCall(PetscFree(solver->recycle_basis));
  PetscCall(KSPDestroy(&solver->ksp));
  PetscCall(LinearSolverDestroyPMGHierarchy(solver));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckLinearSolve(DM dm, AssemblyCtx *actx, AppCtx *app, Mat A, Vec rhs, Vec x, const char *label, KSP ksp, PetscInt *its)
{
  Vec                check = NULL;
  PetscReal          rhs_norm, true_norm, true_rel, true_limit;
  KSPConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(KSPGetIterationNumber(ksp, its));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "%s KSP iterations=%" PetscInt_FMT " reason=%D\n", label, *its, (PetscInt)reason));
  PetscCheck(reason > 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_NOT_CONVERGED, "%s KSP did not converge, reason %D", label, (PetscInt)reason);
  PetscCall(VecDuplicate(rhs, &check));
  PetscCall(MatMult(A, x, check));
  PetscCall(VecAXPY(check, -1.0, rhs));
  PetscCall(VecNorm(rhs, NORM_2, &rhs_norm));
  PetscCall(VecNorm(check, NORM_2, &true_norm));
  PetscCall(VecDestroy(&check));
  true_rel   = rhs_norm > 0.0 ? true_norm / rhs_norm : true_norm;
  true_limit = PetscMax(10.0 * app->ksp_rtol, 1.0e-10);
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "%s true_residual_rel=%.6e limit=%.6e\n", label, (double)true_rel, (double)true_limit));
  PetscCheck(true_rel <= true_limit, PetscObjectComm((PetscObject)dm), PETSC_ERR_NOT_CONVERGED,
             "%s true residual %.6e exceeds verification limit %.6e despite KSP reason %D", label, (double)true_rel, (double)true_limit, (PetscInt)reason);
  (void)actx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckLinearSolutionExplicit(DM dm, AppCtx *app, Mat A, Vec rhs, Vec x, const char *label, PetscInt its, PetscReal reported_rel)
{
  Vec       check = NULL;
  PetscReal rhs_norm, true_norm, true_rel, true_limit;

  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "%s explicit iterations=%" PetscInt_FMT " reported_rel=%.6e\n", label, its, (double)reported_rel));
  PetscCall(VecDuplicate(rhs, &check));
  PetscCall(MatMult(A, x, check));
  PetscCall(VecAXPY(check, -1.0, rhs));
  PetscCall(VecNorm(rhs, NORM_2, &rhs_norm));
  PetscCall(VecNorm(check, NORM_2, &true_norm));
  PetscCall(VecDestroy(&check));
  true_rel   = rhs_norm > 0.0 ? true_norm / rhs_norm : true_norm;
  true_limit = PetscMax(10.0 * app->ksp_rtol, 1.0e-10);
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "%s true_residual_rel=%.6e limit=%.6e\n", label, (double)true_rel, (double)true_limit));
  PetscCheck(true_rel <= true_limit, PetscObjectComm((PetscObject)dm), PETSC_ERR_NOT_CONVERGED,
             "%s true residual %.6e exceeds verification limit %.6e in explicit deflated solve", label, (double)true_rel, (double)true_limit);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverClearOrthBasis(LinearSolverCtx *solver)
{
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
    PetscCall(VecDestroy(&solver->orth_basis[i]));
    PetscCall(VecDestroy(&solver->left_basis[i]));
    PetscCall(VecDestroy(&solver->Aorth_basis[i]));
  }
  solver->n_orth_basis = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static Vec LinearSolverTestBasis(const LinearSolverCtx *solver, PetscInt i)
{
  if (solver->app->deflation_projector == DEFLATION_PROJECTOR_BIORTHOGONAL && solver->left_basis && solver->left_basis[i]) return solver->left_basis[i];
  return solver->orth_basis[i];
}

static PetscErrorCode LinearSolverAppendRawBasisWithTol(LinearSolverCtx *solver, Vec v, const char label[], PetscReal basis_tol)
{
  Vec copy = NULL;

  PetscFunctionBeginUser;
  if (!solver->app->use_deflation) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecDuplicate(v, &copy));
  PetscCall(VecCopy(v, copy));
  if (solver->raw_basis_cap == solver->n_raw_basis) {
    const PetscInt new_cap = solver->raw_basis_cap ? 2 * solver->raw_basis_cap : 8;

    PetscCall(PetscRealloc((size_t)new_cap * sizeof(Vec), &solver->raw_basis));
    PetscCall(PetscRealloc((size_t)new_cap * sizeof(PetscReal), &solver->raw_basis_tol));
    for (PetscInt i = solver->raw_basis_cap; i < new_cap; ++i) solver->raw_basis[i] = NULL;
    solver->raw_basis_cap = new_cap;
  }
  solver->raw_basis[solver->n_raw_basis]     = copy;
  solver->raw_basis_tol[solver->n_raw_basis] = basis_tol;
  solver->n_raw_basis++;
  copy                                = NULL;
  if (solver->app->deflation_max_vectors > 0 && solver->n_raw_basis > solver->app->deflation_max_vectors) {
    PetscCall(VecDestroy(&solver->raw_basis[0]));
    for (PetscInt i = 1; i < solver->n_raw_basis; ++i) {
      solver->raw_basis[i - 1]     = solver->raw_basis[i];
      solver->raw_basis_tol[i - 1] = solver->raw_basis_tol[i];
    }
    solver->raw_basis[--solver->n_raw_basis] = NULL;
  }
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_BASIS_ADD label=\"%s\" raw_cols=%" PetscInt_FMT " max_cols=%" PetscInt_FMT " basis_tol=%.6e\n",
                        label, solver->n_raw_basis, solver->app->deflation_max_vectors, (double)basis_tol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverAppendRawBasis(LinearSolverCtx *solver, Vec v, const char label[])
{
  PetscFunctionBeginUser;
  PetscCall(LinearSolverAppendRawBasisWithTol(solver, v, label, solver->app->deflation_basis_tol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverTruncateRawBasis(LinearSolverCtx *solver, PetscInt n_keep)
{
  PetscFunctionBeginUser;
  if (n_keep < 0) n_keep = 0;
  if (n_keep >= solver->n_raw_basis) PetscFunctionReturn(PETSC_SUCCESS);
  for (PetscInt i = n_keep; i < solver->n_raw_basis; ++i) PetscCall(VecDestroy(&solver->raw_basis[i]));
  solver->n_raw_basis = n_keep;
  for (PetscInt i = n_keep; i < solver->raw_basis_cap; ++i) {
    if (solver->raw_basis) solver->raw_basis[i] = NULL;
  }
  solver->recycle_temp_basis_active = PETSC_FALSE;
  solver->recycle_temp_start_raw    = -1;
  PetscCall(LinearSolverClearOrthBasis(solver));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_BASIS_RESTORE raw_cols=%" PetscInt_FMT "\n", solver->n_raw_basis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverClearRecycleBasis(LinearSolverCtx *solver)
{
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < solver->n_recycle_basis; ++i) PetscCall(VecDestroy(&solver->recycle_basis[i]));
  solver->n_recycle_basis = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverCaptureRecycleVector(LinearSolverCtx *solver, Vec v)
{
  Vec copy = NULL;

  PetscFunctionBeginUser;
  if (!solver->capture_recycle_basis) PetscFunctionReturn(PETSC_SUCCESS);
  if (solver->app->deflation_recycle_max_vectors > 0 && solver->n_recycle_basis >= solver->app->deflation_recycle_max_vectors) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecDuplicate(v, &copy));
  PetscCall(VecCopy(v, copy));
  if (solver->recycle_basis_cap == solver->n_recycle_basis) {
    const PetscInt new_cap = solver->recycle_basis_cap ? 2 * solver->recycle_basis_cap : 8;

    PetscCall(PetscRealloc((size_t)new_cap * sizeof(Vec), &solver->recycle_basis));
    for (PetscInt i = solver->recycle_basis_cap; i < new_cap; ++i) solver->recycle_basis[i] = NULL;
    solver->recycle_basis_cap = new_cap;
  }
  solver->recycle_basis[solver->n_recycle_basis++] = copy;
  copy = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverAppendPersistentKrylovVector(LinearSolverCtx *solver, Vec v, const char solve_label[], PetscInt krylov_it)
{
  char label[256];

  PetscFunctionBeginUser;
  if (!solver->app->deflation_krylov_persistent) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSNPrintf(label, sizeof(label), "%s persistent Krylov direction %" PetscInt_FMT, solve_label, krylov_it + 1));
  PetscCall(LinearSolverAppendRawBasisWithTol(solver, v, label, solver->app->deflation_krylov_basis_tol));
  solver->deflation_krylov_persistent_added++;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_KRYLOV_PERSISTENT_ADD solve=\"%s\" krylov_it=%" PetscInt_FMT " raw_cols=%" PetscInt_FMT " basis_tol=%.6e total_added=%" PetscInt_FMT "\n",
                        solve_label, krylov_it + 1, solver->n_raw_basis, (double)solver->app->deflation_krylov_basis_tol,
                        solver->deflation_krylov_persistent_added));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverBeginRecycleCapture(LinearSolverCtx *solver)
{
  PetscFunctionBeginUser;
  PetscCall(LinearSolverClearRecycleBasis(solver));
  solver->capture_recycle_basis = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverEndRecycleCapture(LinearSolverCtx *solver)
{
  PetscFunctionBeginUser;
  solver->capture_recycle_basis = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverAppendTemporaryRecycleBasis(LinearSolverCtx *solver, const char label[], PetscInt *added)
{
  PetscFunctionBeginUser;
  *added = 0;
  solver->recycle_temp_start_raw    = solver->n_raw_basis;
  solver->recycle_temp_basis_active = (PetscBool)(solver->n_recycle_basis > 0);
  for (PetscInt i = 0; i < solver->n_recycle_basis; ++i) {
    PetscCall(LinearSolverAppendRawBasisWithTol(solver, solver->recycle_basis[i], label, solver->app->deflation_recycle_basis_tol));
    ++(*added);
  }
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_INTRANEWTON_TEMP label=\"%s\" captured=%" PetscInt_FMT " added=%" PetscInt_FMT " max=%" PetscInt_FMT " temp_start_raw=%" PetscInt_FMT " recycle_tol=%.6e\n",
                        label, solver->n_recycle_basis, *added, solver->app->deflation_recycle_max_vectors,
                        solver->recycle_temp_start_raw, (double)solver->app->deflation_recycle_basis_tol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverStoreOrthVector(LinearSolverCtx *solver, Vec v, Vec Av, Vec w)
{
  Vec copy = NULL, Acopy = NULL, wcopy = NULL;

  PetscFunctionBeginUser;
  if (solver->orth_basis_cap == solver->n_orth_basis) {
    const PetscInt new_cap = solver->orth_basis_cap ? 2 * solver->orth_basis_cap : PetscMax(1, solver->n_raw_basis);

    PetscCall(PetscRealloc((size_t)new_cap * sizeof(Vec), &solver->orth_basis));
    PetscCall(PetscRealloc((size_t)new_cap * sizeof(Vec), &solver->left_basis));
    PetscCall(PetscRealloc((size_t)new_cap * sizeof(Vec), &solver->Aorth_basis));
    for (PetscInt i = solver->orth_basis_cap; i < new_cap; ++i) {
      solver->orth_basis[i]  = NULL;
      solver->left_basis[i]  = NULL;
      solver->Aorth_basis[i] = NULL;
    }
    solver->orth_basis_cap = new_cap;
  }
  PetscCall(VecDuplicate(v, &copy));
  PetscCall(VecCopy(v, copy));
  if (w) {
    PetscCall(VecDuplicate(w, &wcopy));
    PetscCall(VecCopy(w, wcopy));
  }
  PetscCall(VecDuplicate(Av, &Acopy));
  PetscCall(VecCopy(Av, Acopy));
  solver->orth_basis[solver->n_orth_basis]  = copy;
  solver->left_basis[solver->n_orth_basis]   = wcopy;
  solver->Aorth_basis[solver->n_orth_basis] = Acopy;
  solver->n_orth_basis++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverDropOrthVector(LinearSolverCtx *solver, PetscInt idx)
{
  PetscFunctionBeginUser;
  PetscCheck(idx >= 0 && idx < solver->n_orth_basis, PetscObjectComm((PetscObject)solver->dm), PETSC_ERR_ARG_OUTOFRANGE,
             "Invalid orthogonal basis index %" PetscInt_FMT " for %" PetscInt_FMT " columns", idx, solver->n_orth_basis);
  PetscCall(VecDestroy(&solver->orth_basis[idx]));
  PetscCall(VecDestroy(&solver->left_basis[idx]));
  PetscCall(VecDestroy(&solver->Aorth_basis[idx]));
  for (PetscInt i = idx + 1; i < solver->n_orth_basis; ++i) {
    solver->orth_basis[i - 1]  = solver->orth_basis[i];
    solver->left_basis[i - 1]  = solver->left_basis[i];
    solver->Aorth_basis[i - 1] = solver->Aorth_basis[i];
  }
  solver->n_orth_basis--;
  solver->orth_basis[solver->n_orth_basis]  = NULL;
  solver->left_basis[solver->n_orth_basis]  = NULL;
  solver->Aorth_basis[solver->n_orth_basis] = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverRenormalizeOrthVector(LinearSolverCtx *solver, PetscInt idx, PetscInt *dropped)
{
  PetscScalar norm_scalar;
  PetscReal   norm_a;

  PetscFunctionBeginUser;
  *dropped = 0;
  PetscCall(VecDot(solver->orth_basis[idx], solver->Aorth_basis[idx], &norm_scalar));
  norm_a = PetscRealPart(norm_scalar);
  if (norm_a <= 0.0) {
    PetscCall(LinearSolverDropOrthVector(solver, idx));
    *dropped = 1;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  {
    const PetscReal inv_norm = 1.0 / PetscSqrtReal(norm_a);

    PetscCall(VecScale(solver->orth_basis[idx], inv_norm));
    PetscCall(VecScale(solver->Aorth_basis[idx], inv_norm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverReorthogonalizeBasis(LinearSolverCtx *solver, const char label[])
{
  PetscInt dropped_total = 0;

  PetscFunctionBeginUser;
  if (solver->app->deflation_reorthogonalize_sweeps <= 0 || solver->n_orth_basis <= 1) PetscFunctionReturn(PETSC_SUCCESS);
  for (PetscInt sweep = 0; sweep < solver->app->deflation_reorthogonalize_sweeps; ++sweep) {
    PetscInt dropped_sweep = 0;

    for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
      for (PetscInt j = 0; j < i; ++j) {
        PetscScalar coeff;

        PetscCall(VecDot(solver->orth_basis[j], solver->Aorth_basis[i], &coeff));
        PetscCall(VecAXPY(solver->orth_basis[i], -coeff, solver->orth_basis[j]));
        PetscCall(VecAXPY(solver->Aorth_basis[i], -coeff, solver->Aorth_basis[j]));
      }
      {
        PetscInt dropped = 0;

        PetscCall(LinearSolverRenormalizeOrthVector(solver, i, &dropped));
        if (dropped) {
          ++dropped_sweep;
          --i;
        }
      }
    }
    for (PetscInt i = solver->n_orth_basis; i-- > 0;) {
      for (PetscInt j = solver->n_orth_basis; j-- > i + 1;) {
        PetscScalar coeff;

        PetscCall(VecDot(solver->orth_basis[j], solver->Aorth_basis[i], &coeff));
        PetscCall(VecAXPY(solver->orth_basis[i], -coeff, solver->orth_basis[j]));
        PetscCall(VecAXPY(solver->Aorth_basis[i], -coeff, solver->Aorth_basis[j]));
      }
      {
        PetscInt dropped = 0;

        PetscCall(LinearSolverRenormalizeOrthVector(solver, i, &dropped));
        if (dropped) ++dropped_sweep;
      }
    }
    dropped_total += dropped_sweep;
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                          "DEFLATION_REORTHO_SWEEP label=\"%s\" sweep=%" PetscInt_FMT " cols=%" PetscInt_FMT " dropped=%" PetscInt_FMT "\n",
                          label, sweep + 1, solver->n_orth_basis, dropped_sweep));
  }
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_REORTHO_SUMMARY label=\"%s\" sweeps=%" PetscInt_FMT " cols=%" PetscInt_FMT " dropped=%" PetscInt_FMT "\n",
                        label, solver->app->deflation_reorthogonalize_sweeps, solver->n_orth_basis, dropped_total));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverCheckAOrthonormality(LinearSolverCtx *solver, const char label[])
{
  PetscReal max_diag_err = 0.0, max_offdiag = 0.0, max_lower = 0.0, max_upper = 0.0, max_asym = 0.0;
  PetscReal min_diag = PETSC_MAX_REAL, max_diag = -PETSC_MAX_REAL;
  PetscInt  max_diag_i = -1, max_off_i = -1, max_off_j = -1, max_asym_i = -1, max_asym_j = -1;
  const char *mode = solver->app->deflation_projector == DEFLATION_PROJECTOR_BIORTHOGONAL ? "biorthogonal" : "a_orthonormal";

  PetscFunctionBeginUser;
  if (!solver->app->deflation_check_orthonormality) PetscFunctionReturn(PETSC_SUCCESS);
  for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
    PetscScalar gii;
    PetscReal   diag, diag_err;

    PetscCall(VecDot(LinearSolverTestBasis(solver, i), solver->Aorth_basis[i], &gii));
    diag     = PetscRealPart(gii);
    diag_err = PetscAbsScalar(gii - 1.0);
    if (diag < min_diag) min_diag = diag;
    if (diag > max_diag) max_diag = diag;
    if (diag_err > max_diag_err) {
      max_diag_err = diag_err;
      max_diag_i   = i;
    }
    for (PetscInt j = i + 1; j < solver->n_orth_basis; ++j) {
      PetscScalar gij, gji;
      PetscReal   abs_ij, abs_ji, asym;

      PetscCall(VecDot(LinearSolverTestBasis(solver, i), solver->Aorth_basis[j], &gij));
      PetscCall(VecDot(LinearSolverTestBasis(solver, j), solver->Aorth_basis[i], &gji));
      abs_ij = PetscAbsScalar(gij);
      abs_ji = PetscAbsScalar(gji);
      asym   = PetscAbsScalar(gij - gji);
      if (abs_ij > max_upper) max_upper = abs_ij;
      if (abs_ji > max_lower) max_lower = abs_ji;
      if (abs_ij > max_offdiag) {
        max_offdiag = abs_ij;
        max_off_i   = i;
        max_off_j   = j;
      }
      if (abs_ji > max_offdiag) {
        max_offdiag = abs_ji;
        max_off_i   = j;
        max_off_j   = i;
      }
      if (asym > max_asym) {
        max_asym   = asym;
        max_asym_i = i;
        max_asym_j = j;
      }
    }
  }
  if (!solver->n_orth_basis) {
    min_diag = 0.0;
    max_diag = 0.0;
  }
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_GRAM_CHECK label=\"%s\" mode=%s cols=%" PetscInt_FMT " max_diag_err=%.6e max_offdiag=%.6e max_lower=%.6e max_upper=%.6e max_asym=%.6e min_diag=%.6e max_diag=%.6e max_diag_i=%" PetscInt_FMT " max_off_i=%" PetscInt_FMT " max_off_j=%" PetscInt_FMT " max_asym_i=%" PetscInt_FMT " max_asym_j=%" PetscInt_FMT " warn_tol=%.6e ok=%s\n",
                        label, mode, solver->n_orth_basis, (double)max_diag_err, (double)max_offdiag, (double)max_lower,
                        (double)max_upper, (double)max_asym, (double)min_diag, (double)max_diag, max_diag_i, max_off_i,
                        max_off_j, max_asym_i, max_asym_j, (double)solver->app->deflation_orthonormality_warn_tol,
                        (max_diag_err <= solver->app->deflation_orthonormality_warn_tol &&
                         max_offdiag <= solver->app->deflation_orthonormality_warn_tol) ? "true" : "false"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverBiorthogonalizeBasis(LinearSolverCtx *solver, Mat A, const char label[])
{
  Vec            v = NULL, w = NULL, Av = NULL;
  PetscInt       skipped_small = 0, skipped_pivot = 0;
  PetscLogDouble t0, t1;
  const PetscInt passes = PetscMax((PetscInt)1, solver->app->deflation_reorthogonalize_sweeps + 1);

  PetscFunctionBeginUser;
  PetscCall(PetscLogStagePush(log_stage_deflation_orthogonalize));
  PetscCall(PetscTime(&t0));
  PetscCall(LinearSolverClearOrthBasis(solver));
  if (!solver->n_raw_basis) {
    PetscCall(PetscTime(&t1));
    solver->deflation_orthogonalization_time += t1 - t0;
    PetscCall(PetscLogStagePop());
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecDuplicate(solver->raw_basis[0], &v));
  PetscCall(VecDuplicate(solver->raw_basis[0], &w));
  PetscCall(VecDuplicate(solver->raw_basis[0], &Av));

  /*
    Build an oblique projector basis V,W with W^T A V = I.  V is the
    correction space used in the coarse component; W is only the test space
    used for coefficients.  As in the one-sided path, recent raw vectors get
    priority when near-dependent candidates are skipped.
  */
  for (PetscInt src = solver->n_raw_basis; src-- > 0;) {
    PetscScalar gamma;
    PetscReal   abs_gamma, basis_tol, w_norm, Av_norm, pivot_tol, pivot_threshold;

    PetscCall(VecCopy(solver->raw_basis[src], v));
    PetscCall(VecCopy(solver->raw_basis[src], w));
    PetscCall(MatMult(A, v, Av));

    for (PetscInt pass = 0; pass < passes; ++pass) {
      for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
        PetscScalar alpha, beta;

        PetscCall(VecDot(LinearSolverTestBasis(solver, i), Av, &alpha));
        PetscCall(VecAXPY(v, -alpha, solver->orth_basis[i]));
        PetscCall(VecAXPY(Av, -alpha, solver->Aorth_basis[i]));
        PetscCall(VecDot(w, solver->Aorth_basis[i], &beta));
        PetscCall(VecAXPY(w, -beta, LinearSolverTestBasis(solver, i)));
      }
    }

    PetscCall(VecDot(w, Av, &gamma));
    PetscCall(VecNorm(w, NORM_2, &w_norm));
    PetscCall(VecNorm(Av, NORM_2, &Av_norm));
    abs_gamma = PetscAbsScalar(gamma);
    basis_tol = solver->raw_basis_tol ? solver->raw_basis_tol[src] : solver->app->deflation_basis_tol;
    pivot_tol = solver->app->deflation_biorthogonal_pivot_tol * w_norm * Av_norm;
    pivot_threshold = PetscMax(PetscMax(basis_tol, pivot_tol), 1.0e-30);
    if (PetscIsInfOrNanReal(abs_gamma) || abs_gamma <= pivot_threshold) {
      ++skipped_small;
      if (abs_gamma <= PetscMax(pivot_tol, 1.0e-30)) ++skipped_pivot;
      continue;
    }
    {
      const PetscReal sign = PetscRealPart(gamma) < 0.0 ? -1.0 : 1.0;
      const PetscReal scale_right = sign / PetscSqrtReal(abs_gamma);
      const PetscReal scale_left  = 1.0 / PetscSqrtReal(abs_gamma);

      PetscCall(VecScale(v, scale_right));
      PetscCall(VecScale(Av, scale_right));
      PetscCall(VecScale(w, scale_left));
    }
    PetscCall(LinearSolverStoreOrthVector(solver, v, Av, w));
  }
  for (PetscInt i = 0; i < solver->n_orth_basis / 2; ++i) {
    Vec tmp = solver->orth_basis[i];
    Vec wtmp = solver->left_basis[i];
    Vec Atmp = solver->Aorth_basis[i];

    solver->orth_basis[i] = solver->orth_basis[solver->n_orth_basis - 1 - i];
    solver->orth_basis[solver->n_orth_basis - 1 - i] = tmp;
    solver->left_basis[i] = solver->left_basis[solver->n_orth_basis - 1 - i];
    solver->left_basis[solver->n_orth_basis - 1 - i] = wtmp;
    solver->Aorth_basis[i] = solver->Aorth_basis[solver->n_orth_basis - 1 - i];
    solver->Aorth_basis[solver->n_orth_basis - 1 - i] = Atmp;
  }
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&w));
  PetscCall(VecDestroy(&Av));
  PetscCall(PetscTime(&t1));
  solver->deflation_orthogonalization_time += t1 - t0;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_BIORTHO label=\"%s\" raw_cols=%" PetscInt_FMT " orth_cols=%" PetscInt_FMT " skipped_small=%" PetscInt_FMT " skipped_pivot=%" PetscInt_FMT " passes=%" PetscInt_FMT " default_tol=%.6e pivot_tol=%.6e recycle_active=%s recycle_start=%" PetscInt_FMT " recycle_tol=%.6e krylov_tol=%.6e time=%.6g\n",
                        label, solver->n_raw_basis, solver->n_orth_basis, skipped_small, skipped_pivot, passes,
                        (double)solver->app->deflation_basis_tol, (double)solver->app->deflation_biorthogonal_pivot_tol,
                        solver->recycle_temp_basis_active ? "true" : "false", solver->recycle_temp_start_raw, (double)solver->app->deflation_recycle_basis_tol,
                        (double)solver->app->deflation_krylov_basis_tol, (double)(t1 - t0)));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_CACHE_CONFIG orth_cols=%" PetscInt_FMT " cached_A_cols=%" PetscInt_FMT " left_cols=%" PetscInt_FMT " projector=biorthogonal\n",
                        solver->n_orth_basis, solver->n_orth_basis, solver->n_orth_basis));
  PetscCall(LinearSolverCheckAOrthonormality(solver, label));
  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverAOrthogonalizeBasis(LinearSolverCtx *solver, Mat A, const char label[])
{
  Vec            v = NULL, Av = NULL;
  PetscInt       skipped_small = 0, skipped_nonpositive = 0;
  PetscLogDouble t0, t1;

  PetscFunctionBeginUser;
  if (solver->app->deflation_projector == DEFLATION_PROJECTOR_BIORTHOGONAL) {
    PetscCall(LinearSolverBiorthogonalizeBasis(solver, A, label));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscLogStagePush(log_stage_deflation_orthogonalize));
  PetscCall(PetscTime(&t0));
  PetscCall(LinearSolverClearOrthBasis(solver));
  if (!solver->n_raw_basis) {
    PetscCall(PetscTime(&t1));
    solver->deflation_orthogonalization_time += t1 - t0;
    PetscCall(PetscLogStagePop());
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecDuplicate(solver->raw_basis[0], &v));
  PetscCall(VecDuplicate(solver->raw_basis[0], &Av));

  /*
    Match the Python/MATLAB deflation path's preference for recent Newton
    corrections when near-dependent columns are dropped, then reverse the kept
    list back to chronological order for easier diagnostics.
  */
  for (PetscInt src = solver->n_raw_basis; src-- > 0;) {
    PetscScalar norm_scalar;
    PetscReal   norm_a;

    PetscCall(VecCopy(solver->raw_basis[src], v));
    PetscCall(MatMult(A, v, Av));
    for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
      PetscScalar coeff;

      PetscCall(VecDot(solver->orth_basis[i], Av, &coeff));
      PetscCall(VecAXPY(v, -coeff, solver->orth_basis[i]));
      PetscCall(VecAXPY(Av, -coeff, solver->Aorth_basis[i]));
    }
    PetscCall(VecDot(v, Av, &norm_scalar));
    norm_a = PetscRealPart(norm_scalar);
    if (norm_a <= 0.0) {
      ++skipped_nonpositive;
      continue;
    }
    {
      const PetscReal basis_tol = solver->raw_basis_tol ? solver->raw_basis_tol[src] : solver->app->deflation_basis_tol;

      if (norm_a <= basis_tol) {
        ++skipped_small;
        continue;
      }
    }
    {
      const PetscReal inv_norm = 1.0 / PetscSqrtReal(norm_a);

      PetscCall(VecScale(v, inv_norm));
      PetscCall(VecScale(Av, inv_norm));
    }
    PetscCall(LinearSolverStoreOrthVector(solver, v, Av, NULL));
  }
  for (PetscInt i = 0; i < solver->n_orth_basis / 2; ++i) {
    Vec tmp = solver->orth_basis[i];
    Vec Atmp = solver->Aorth_basis[i];

    solver->orth_basis[i] = solver->orth_basis[solver->n_orth_basis - 1 - i];
    solver->orth_basis[solver->n_orth_basis - 1 - i] = tmp;
    solver->Aorth_basis[i] = solver->Aorth_basis[solver->n_orth_basis - 1 - i];
    solver->Aorth_basis[solver->n_orth_basis - 1 - i] = Atmp;
  }
  PetscCall(LinearSolverReorthogonalizeBasis(solver, label));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&Av));
  PetscCall(PetscTime(&t1));
  solver->deflation_orthogonalization_time += t1 - t0;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_ORTHO label=\"%s\" raw_cols=%" PetscInt_FMT " orth_cols=%" PetscInt_FMT " skipped_small=%" PetscInt_FMT " skipped_nonpositive=%" PetscInt_FMT " default_tol=%.6e recycle_active=%s recycle_start=%" PetscInt_FMT " recycle_tol=%.6e krylov_tol=%.6e time=%.6g\n",
                        label, solver->n_raw_basis, solver->n_orth_basis, skipped_small, skipped_nonpositive,
                        (double)solver->app->deflation_basis_tol, solver->recycle_temp_basis_active ? "true" : "false",
                        solver->recycle_temp_start_raw, (double)solver->app->deflation_recycle_basis_tol,
                        (double)solver->app->deflation_krylov_basis_tol, (double)(t1 - t0)));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_CACHE_CONFIG orth_cols=%" PetscInt_FMT " cached_A_cols=%" PetscInt_FMT " left_cols=0 projector=a_orthonormal\n",
                        solver->n_orth_basis, solver->n_orth_basis));
  PetscCall(LinearSolverCheckAOrthonormality(solver, label));
  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DeflationCoarseInitialGuess(LinearSolverCtx *solver, Mat A, Vec rhs, Vec x, Vec r, Vec work)
{
  PetscLogDouble t0, t1;

  PetscFunctionBeginUser;
  PetscCall(PetscLogStagePush(log_stage_deflation_initial_guess));
  PetscCall(PetscTime(&t0));
  PetscCall(VecZeroEntries(x));
  PetscCall(VecCopy(rhs, r));
  for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
    PetscScalar coeff;

    PetscCall(VecDot(LinearSolverTestBasis(solver, i), rhs, &coeff));
    PetscCall(VecAXPY(x, coeff, solver->orth_basis[i]));
    PetscCall(VecAXPY(r, -coeff, solver->Aorth_basis[i]));
  }
  PetscCall(PetscTime(&t1));
  solver->deflation_coarse_time += t1 - t0;
  ++solver->deflation_coarse_calls;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_COARSE_INITIAL basis_cols=%" PetscInt_FMT " call=%" PetscInt_FMT " time=%.6g cumulative_time=%.6g\n",
                        solver->n_orth_basis, solver->deflation_coarse_calls, (double)(t1 - t0), (double)solver->deflation_coarse_time));
  PetscCall(PetscLogStagePop());
  (void)A;
  (void)work;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DeflationApplyProjectedPC(LinearSolverCtx *solver, Mat A, PC pc, Vec v, Vec z, Vec Az, PetscBool return_Az)
{
  PetscLogDouble t0, t1, t2, pc_time, projector_time = 0.0;

  PetscFunctionBeginUser;
  PetscCall(PetscTime(&t0));
  PetscCall(PCApply(pc, v, z));
  PetscCall(PetscTime(&t1));
  if (solver->n_orth_basis || return_Az) {
    PetscCall(PetscLogStagePush(log_stage_deflation_projector));
    PetscCall(MatMult(A, z, Az));
    for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
      PetscScalar coeff;

      PetscCall(VecDot(LinearSolverTestBasis(solver, i), Az, &coeff));
      PetscCall(VecAXPY(z, -coeff, solver->orth_basis[i]));
      PetscCall(VecAXPY(Az, -coeff, solver->Aorth_basis[i]));
    }
    PetscCall(PetscLogStagePop());
  }
  PetscCall(PetscTime(&t2));
  pc_time        = t1 - t0;
  projector_time = t2 - t1;
  solver->deflation_pc_apply_time += pc_time;
  solver->deflation_projector_time += projector_time;
  ++solver->deflation_projected_pc_calls;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DeflationApplyMatlabProjectedPC(LinearSolverCtx *solver, Mat A, PC pc, Vec v, Vec z, Vec work)
{
  PetscLogDouble t0, t1, t2, pc_time, projector_time = 0.0;

  PetscFunctionBeginUser;
  PetscCall(PetscTime(&t0));
  PetscCall(PCApply(pc, v, z));
  PetscCall(PetscTime(&t1));
  if (solver->n_orth_basis) {
    PetscCall(PetscLogStagePush(log_stage_deflation_projector));
    PetscCall(MatMult(A, z, work));
    for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
      PetscScalar coeff;

      PetscCall(VecDot(LinearSolverTestBasis(solver, i), work, &coeff));
      PetscCall(VecAXPY(z, -coeff, solver->orth_basis[i]));
    }
    PetscCall(PetscLogStagePop());
  }
  PetscCall(PetscTime(&t2));
  pc_time        = t1 - t0;
  projector_time = t2 - t1;
  solver->deflation_pc_apply_time += pc_time;
  solver->deflation_projector_time += projector_time;
  ++solver->deflation_projected_pc_calls;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DenseLeastSquaresNormal(PetscInt ldh, const PetscScalar H[], PetscReal beta, PetscInt rows, PetscInt cols, PetscScalar y[], PetscReal *residual)
{
  PetscScalar *G = NULL, *c = NULL;

  PetscFunctionBeginUser;
  if (!cols) {
    *residual = beta;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscCalloc2(cols * cols, &G, cols, &c));
  for (PetscInt a = 0; a < cols; ++a) {
    c[a] = beta * H[a];
    for (PetscInt b = 0; b < cols; ++b) {
      PetscScalar v = 0.0;

      for (PetscInt r = 0; r < rows; ++r) v += H[r * ldh + a] * H[r * ldh + b];
      G[a * cols + b] = v;
    }
  }

  for (PetscInt k = 0; k < cols; ++k) {
    PetscInt  piv = k;
    PetscReal best = PetscAbsScalar(G[k * cols + k]);

    for (PetscInt i = k + 1; i < cols; ++i) {
      PetscReal cand = PetscAbsScalar(G[i * cols + k]);
      if (cand > best) {
        best = cand;
        piv  = i;
      }
    }
    PetscCheck(best > 1.0e-30, PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "Singular small least-squares system in deflated FGMRES");
    if (piv != k) {
      for (PetscInt j = k; j < cols; ++j) {
        PetscScalar tmp        = G[k * cols + j];
        G[k * cols + j]       = G[piv * cols + j];
        G[piv * cols + j]     = tmp;
      }
      {
        PetscScalar tmp = c[k];
        c[k]            = c[piv];
        c[piv]          = tmp;
      }
    }
    for (PetscInt i = k + 1; i < cols; ++i) {
      PetscScalar factor = G[i * cols + k] / G[k * cols + k];

      G[i * cols + k] = 0.0;
      for (PetscInt j = k + 1; j < cols; ++j) G[i * cols + j] -= factor * G[k * cols + j];
      c[i] -= factor * c[k];
    }
  }
  for (PetscInt i = cols; i-- > 0;) {
    PetscScalar sum = c[i];

    for (PetscInt j = i + 1; j < cols; ++j) sum -= G[i * cols + j] * y[j];
    y[i] = sum / G[i * cols + i];
  }
  {
    PetscReal r2 = 0.0;

    for (PetscInt r = 0; r < rows; ++r) {
      PetscScalar hy = 0.0;
      PetscScalar g  = (r == 0) ? beta : 0.0;

      for (PetscInt a = 0; a < cols; ++a) hy += H[r * ldh + a] * y[a];
      r2 += PetscSqr(PetscAbsScalar(g - hy));
    }
    *residual = PetscSqrtReal(r2);
  }
  PetscCall(PetscFree2(G, c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DenseLeastSquaresLAPACK(PetscInt ldh, const PetscScalar H[], PetscReal beta, PetscInt rows, PetscInt cols, PetscScalar y[], PetscReal *residual)
{
  PetscScalar  *A = NULL, *B = NULL, *work = NULL;
  PetscBLASInt  m, n, nrhs = 1, lda, ldb, info, lwork = -1;
  PetscScalar   work_query;
  char          trans = 'N';

  PetscFunctionBeginUser;
  if (!cols) {
    *residual = beta;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscBLASIntCast(rows, &m));
  PetscCall(PetscBLASIntCast(cols, &n));
  PetscCall(PetscBLASIntCast(PetscMax(rows, (PetscInt)1), &lda));
  PetscCall(PetscBLASIntCast(PetscMax(rows, cols), &ldb));
  PetscCall(PetscCalloc2((size_t)lda * (size_t)n, &A, (size_t)ldb, &B));
  for (PetscInt c = 0; c < cols; ++c) {
    for (PetscInt r = 0; r < rows; ++r) A[r + c * rows] = H[r * ldh + c];
  }
  B[0] = beta;
  LAPACKgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, &work_query, &lwork, &info);
  PetscCheck(!info, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACKgels workspace query failed with info=%d", (int)info);
  PetscCall(PetscBLASIntCast(PetscMax((PetscInt)1, (PetscInt)PetscRealPart(work_query)), &lwork));
  PetscCall(PetscMalloc1((size_t)lwork, &work));
  LAPACKgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, &info);
  PetscCheck(!info, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACKgels least-squares solve failed with info=%d", (int)info);
  for (PetscInt i = 0; i < cols; ++i) y[i] = B[i];
  {
    PetscReal r2 = 0.0;

    for (PetscInt r = 0; r < rows; ++r) {
      PetscScalar hy = 0.0;
      PetscScalar g  = (r == 0) ? beta : 0.0;

      for (PetscInt c = 0; c < cols; ++c) hy += H[r * ldh + c] * y[c];
      r2 += PetscSqr(PetscAbsScalar(g - hy));
    }
    *residual = PetscSqrtReal(r2);
  }
  PetscCall(PetscFree(work));
  PetscCall(PetscFree2(A, B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetExplicitDeflationMaxIt(KSP ksp, const AppCtx *app, PetscInt *max_it)
{
  PetscReal rtol, abstol, dtol;
  PetscInt  ksp_max_it;

  PetscFunctionBeginUser;
  PetscCall(KSPGetTolerances(ksp, &rtol, &abstol, &dtol, &ksp_max_it));
  (void)rtol; (void)abstol; (void)dtol;
  if (app->deflation_max_it > 0) *max_it = app->deflation_max_it;
  else if (ksp_max_it > 0 && ksp_max_it <= 1000) *max_it = ksp_max_it;
  else *max_it = 200;
  PetscCheck(*max_it > 0, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_OUTOFRANGE, "Explicit deflation max iterations must be positive");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DeflatedFGMRESSolve(LinearSolverCtx *solver, KSP ksp, Vec rhs, Vec x, const char label[], PetscInt *its, PetscReal *reported_rel)
{
  Mat          A = solver->A;
  PC           pc;
  Vec         *V = NULL, *Z = NULL;
  Vec          r = NULL, Az = NULL;
  PetscScalar *H = NULL, *y = NULL;
  PetscReal    rtol, abstol, dtol, rhs_norm, beta, rel = PETSC_MAX_REAL;
  PetscInt     max_it, ksp_max_it, final_its = 0;
  PetscBool    converged = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(KSPGetTolerances(ksp, &rtol, &abstol, &dtol, &ksp_max_it));
  (void)abstol; (void)dtol; (void)ksp_max_it;
  if (rtol <= 0.0) rtol = solver->app->ksp_rtol;
  PetscCall(GetExplicitDeflationMaxIt(ksp, solver->app, &max_it));
  PetscCall(VecDuplicateVecs(rhs, max_it + 1, &V));
  PetscCall(VecDuplicateVecs(rhs, max_it, &Z));
  PetscCall(VecDuplicate(rhs, &r));
  PetscCall(VecDuplicate(rhs, &Az));
  PetscCall(PetscCalloc2((max_it + 1) * max_it, &H, max_it, &y));

  PetscCall(DeflationCoarseInitialGuess(solver, A, rhs, x, r, Az));
  PetscCall(VecNorm(rhs, NORM_2, &rhs_norm));
  if (rhs_norm == 0.0) rhs_norm = 1.0;
  PetscCall(VecNorm(r, NORM_2, &beta));
  rel = beta / rhs_norm;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATED_FGMRES_INITIAL label=\"%s\" basis_cols=%" PetscInt_FMT " rhs_norm=%.17e beta=%.17e rel=%.17e\n",
                        label, solver->n_orth_basis, (double)rhs_norm, (double)beta, (double)rel));
  if (rel <= rtol) {
    converged = PETSC_TRUE;
  } else {
    PetscCall(VecCopy(r, V[0]));
    PetscCall(VecScale(V[0], 1.0 / beta));
  }

  for (PetscInt j = 0; !converged && j < max_it; ++j) {
    PetscReal hnext, res_norm;

    PetscCall(DeflationApplyProjectedPC(solver, A, pc, V[j], Z[j], Az, PETSC_TRUE));
    PetscCall(LinearSolverCaptureRecycleVector(solver, Z[j]));
    PetscCall(LinearSolverAppendPersistentKrylovVector(solver, Z[j], label, j));
    for (PetscInt i = 0; i <= j; ++i) {
      PetscScalar hij;

      PetscCall(VecDot(V[i], Az, &hij));
      H[i * max_it + j] = hij;
      PetscCall(VecAXPY(Az, -hij, V[i]));
    }
    PetscCall(VecNorm(Az, NORM_2, &hnext));
    H[(j + 1) * max_it + j] = hnext;
    if (hnext > 1.0e-14 && j + 1 < max_it + 1) {
      PetscCall(VecCopy(Az, V[j + 1]));
      PetscCall(VecScale(V[j + 1], 1.0 / hnext));
    }
    PetscCall(DenseLeastSquaresNormal(max_it, H, beta, j + 2, j + 1, y, &res_norm));
    rel        = res_norm / rhs_norm;
    final_its  = j + 1;
    if (solver->app->deflation_monitor) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm), "DEFLATED_FGMRES label=\"%s\" it=%" PetscInt_FMT " rel=%.6e\n", label, final_its, (double)rel));
    }
    if (rel <= rtol) converged = PETSC_TRUE;
    if (hnext <= 1.0e-14) break;
  }
  for (PetscInt i = 0; i < final_its; ++i) PetscCall(VecAXPY(x, y[i], Z[i]));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATED_SOLVE label=\"%s\" method=fgmres basis_cols=%" PetscInt_FMT " iterations=%" PetscInt_FMT " reported_rel=%.6e converged=%s\n",
                        label, solver->n_orth_basis, final_its, (double)rel, converged ? "true" : "false"));
  *its         = final_its;
  *reported_rel = rel;

  PetscCall(PetscFree2(H, y));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&Az));
  PetscCall(VecDestroyVecs(max_it, &Z));
  PetscCall(VecDestroyVecs(max_it + 1, &V));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DeflatedMatlabDFGMRESSolve(LinearSolverCtx *solver, KSP ksp, Vec rhs, Vec x, const char label[], PetscInt *its, PetscReal *reported_rel)
{
  Mat          A = solver->A;
  PC           pc;
  Vec         *V = NULL, *Z = NULL;
  Vec          r = NULL, work = NULL;
  PetscScalar *H = NULL, *y = NULL;
  PetscReal    rtol, abstol, dtol, rhs_norm, beta, rel = PETSC_MAX_REAL;
  PetscInt     max_it, ksp_max_it, final_its = 0;
  PetscBool    converged = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(KSPGetTolerances(ksp, &rtol, &abstol, &dtol, &ksp_max_it));
  (void)abstol; (void)dtol; (void)ksp_max_it;
  if (rtol <= 0.0) rtol = solver->app->ksp_rtol;
  PetscCall(GetExplicitDeflationMaxIt(ksp, solver->app, &max_it));
  PetscCall(VecDuplicateVecs(rhs, max_it + 1, &V));
  PetscCall(VecDuplicateVecs(rhs, max_it, &Z));
  PetscCall(VecDuplicate(rhs, &r));
  PetscCall(VecDuplicate(rhs, &work));
  PetscCall(PetscCalloc2((max_it + 1) * max_it, &H, max_it, &y));

  PetscCall(PetscLogStagePush(log_stage_deflation_initial_guess));
  PetscCall(VecZeroEntries(x));
  for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
    PetscScalar coeff;

    PetscCall(VecDot(LinearSolverTestBasis(solver, i), rhs, &coeff));
    PetscCall(VecAXPY(x, coeff, solver->orth_basis[i]));
  }
  PetscCall(MatMult(A, x, work));
  PetscCall(VecWAXPY(r, -1.0, work, rhs));
  PetscCall(PetscLogStagePop());
  ++solver->deflation_coarse_calls;

  PetscCall(VecNorm(rhs, NORM_2, &rhs_norm));
  if (rhs_norm == 0.0) rhs_norm = 1.0;
  PetscCall(VecNorm(r, NORM_2, &beta));
  rel = beta / rhs_norm;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATED_MATLAB_DFGMRES_INITIAL label=\"%s\" basis_cols=%" PetscInt_FMT " rhs_norm=%.17e beta=%.17e rel=%.17e\n",
                        label, solver->n_orth_basis, (double)rhs_norm, (double)beta, (double)rel));
  if (rel <= rtol) {
    converged = PETSC_TRUE;
  } else {
    PetscCall(VecCopy(r, V[0]));
    PetscCall(VecScale(V[0], 1.0 / beta));
  }

  for (PetscInt j = 0; !converged && j < max_it; ++j) {
    PetscReal hnext, res_norm;

    PetscCall(DeflationApplyMatlabProjectedPC(solver, A, pc, V[j], Z[j], work));
    PetscCall(MatMult(A, Z[j], work));
    PetscCall(LinearSolverCaptureRecycleVector(solver, Z[j]));
    PetscCall(LinearSolverAppendPersistentKrylovVector(solver, Z[j], label, j));
    for (PetscInt i = 0; i <= j; ++i) {
      PetscScalar hij;

      PetscCall(VecDot(V[i], work, &hij));
      H[i * max_it + j] = hij;
      PetscCall(VecAXPY(work, -hij, V[i]));
    }
    PetscCall(VecNorm(work, NORM_2, &hnext));
    H[(j + 1) * max_it + j] = hnext;
    if (j == 0) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                            "DEFLATED_MATLAB_DFGMRES_SAMPLE label=\"%s\" it=1 h00=%.17e h10=%.17e\n",
                            label, (double)PetscRealPart(H[0]), (double)hnext));
    }
    final_its = j + 1;
    if (hnext <= 1.0e-14) {
      PetscCall(DenseLeastSquaresLAPACK(max_it, H, beta, final_its + 1, final_its, y, &res_norm));
      rel = res_norm / rhs_norm;
      break;
    }
    PetscCall(VecCopy(work, V[j + 1]));
    PetscCall(VecScale(V[j + 1], 1.0 / hnext));
    PetscCall(DenseLeastSquaresLAPACK(max_it, H, beta, j + 2, j + 1, y, &res_norm));
    rel = res_norm / rhs_norm;
    if (solver->app->deflation_monitor) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm), "DEFLATED_MATLAB_DFGMRES label=\"%s\" it=%" PetscInt_FMT " rel=%.6e\n", label, final_its, (double)rel));
    }
    if (rel <= rtol) converged = PETSC_TRUE;
  }
  for (PetscInt i = 0; i < final_its; ++i) PetscCall(VecAXPY(x, y[i], Z[i]));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATED_SOLVE label=\"%s\" method=matlab_dfgmres basis_cols=%" PetscInt_FMT " iterations=%" PetscInt_FMT " reported_rel=%.6e converged=%s\n",
                        label, solver->n_orth_basis, final_its, (double)rel, converged ? "true" : "false"));
  *its          = final_its;
  *reported_rel = rel;

  PetscCall(PetscFree2(H, y));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&work));
  PetscCall(VecDestroyVecs(max_it, &Z));
  PetscCall(VecDestroyVecs(max_it + 1, &V));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DeflatedCGSolve(LinearSolverCtx *solver, KSP ksp, Vec rhs, Vec x, const char label[], PetscInt *its, PetscReal *reported_rel)
{
  Mat       A = solver->A;
  PC        pc;
  Vec       r = NULL, z = NULL, p = NULL, Ap = NULL;
  PetscReal rtol, abstol, dtol, rhs_norm, rnorm, rel, rho_real = 0.0;
  PetscInt  max_it, ksp_max_it, final_its = 0;
  PetscBool converged = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(KSPGetTolerances(ksp, &rtol, &abstol, &dtol, &ksp_max_it));
  (void)abstol; (void)dtol; (void)ksp_max_it;
  if (rtol <= 0.0) rtol = solver->app->ksp_rtol;
  PetscCall(GetExplicitDeflationMaxIt(ksp, solver->app, &max_it));
  PetscCall(VecDuplicate(rhs, &r));
  PetscCall(VecDuplicate(rhs, &z));
  PetscCall(VecDuplicate(rhs, &p));
  PetscCall(VecDuplicate(rhs, &Ap));

  PetscCall(DeflationCoarseInitialGuess(solver, A, rhs, x, r, Ap));
  PetscCall(VecNorm(rhs, NORM_2, &rhs_norm));
  if (rhs_norm == 0.0) rhs_norm = 1.0;
  PetscCall(VecNorm(r, NORM_2, &rnorm));
  rel = rnorm / rhs_norm;
  if (rel <= rtol) converged = PETSC_TRUE;

  if (!converged) {
    PetscScalar rho;

    PetscCall(DeflationApplyProjectedPC(solver, A, pc, r, z, Ap, PETSC_FALSE));
    PetscCall(VecDot(r, z, &rho));
    rho_real = PetscRealPart(rho);
    PetscCheck(rho_real > 0.0, PetscObjectComm((PetscObject)solver->dm), PETSC_ERR_NOT_CONVERGED,
               "Deflated CG encountered nonpositive preconditioned residual norm %.6e", (double)rho_real);
    PetscCall(VecCopy(z, p));
  }

  for (PetscInt it = 0; !converged && it < max_it; ++it) {
    PetscScalar pAp_scalar, rho_new;
    PetscReal   pAp_real, alpha, beta_cg;

    PetscCall(MatMult(A, p, Ap));
    PetscCall(VecDot(p, Ap, &pAp_scalar));
    pAp_real = PetscRealPart(pAp_scalar);
    PetscCheck(pAp_real > 0.0, PetscObjectComm((PetscObject)solver->dm), PETSC_ERR_NOT_CONVERGED,
               "Deflated CG encountered nonpositive p^T A p %.6e", (double)pAp_real);
    alpha = rho_real / pAp_real;
    PetscCall(VecAXPY(x, alpha, p));
    PetscCall(VecAXPY(r, -alpha, Ap));
    PetscCall(VecNorm(r, NORM_2, &rnorm));
    rel       = rnorm / rhs_norm;
    final_its = it + 1;
    if (solver->app->deflation_monitor) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm), "DEFLATED_CG label=\"%s\" it=%" PetscInt_FMT " rel=%.6e\n", label, final_its, (double)rel));
    }
    if (rel <= rtol) {
      converged = PETSC_TRUE;
      break;
    }
    PetscCall(DeflationApplyProjectedPC(solver, A, pc, r, z, Ap, PETSC_FALSE));
    PetscCall(VecDot(r, z, &rho_new));
    PetscCheck(PetscRealPart(rho_new) > 0.0, PetscObjectComm((PetscObject)solver->dm), PETSC_ERR_NOT_CONVERGED,
               "Deflated CG encountered nonpositive updated preconditioned residual norm %.6e", (double)PetscRealPart(rho_new));
    beta_cg  = PetscRealPart(rho_new) / rho_real;
    rho_real = PetscRealPart(rho_new);
    PetscCall(VecAYPX(p, beta_cg, z));
  }

  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATED_SOLVE label=\"%s\" method=cg basis_cols=%" PetscInt_FMT " iterations=%" PetscInt_FMT " reported_rel=%.6e converged=%s\n",
                        label, solver->n_orth_basis, final_its, (double)rel, converged ? "true" : "false"));
  *its          = final_its;
  *reported_rel = rel;
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&z));
  PetscCall(VecDestroy(&p));
  PetscCall(VecDestroy(&Ap));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolveLinearSystem(LinearSolverCtx *solver, Vec rhs, Vec x, const char *label, Vec current_u, PetscBool nonlinear_tangent, PetscInt *its)
{
  KSP ksp = NULL;

  PetscFunctionBeginUser;
  (void)current_u;
  if (solver->reuse) {
    if (!solver->ksp) {
      PetscCall(KSPCreate(PetscObjectComm((PetscObject)solver->dm), &solver->ksp));
      PetscCall(ConfigureKSP(solver->ksp, solver->dm, solver->actx, solver->app, solver->A, nonlinear_tangent, solver));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm), "LINEAR_SOLVER_REUSE configured persistent KSP/PC hierarchy\n"));
    } else {
      PetscBool reuse_pc = PETSC_FALSE;

      if (solver->app->variant == VARIANT_PMG && nonlinear_tangent) {
        const PetscInt lag     = solver->app->pmg_lag_preconditioner;
        const PetscInt idx     = solver->pmg_lag_solve_index;
        const PetscBool rebuild = (PetscBool)(lag <= 1 || idx % lag == 0);

        reuse_pc = (PetscBool)!rebuild;
        if (solver->force_reuse_preconditioner) reuse_pc = PETSC_TRUE;
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                              "PMG_LAG_PRECONDITIONER solve_index=%" PetscInt_FMT " lag=%" PetscInt_FMT " rebuild=%s forced_reuse=%s reuse_preconditioner=%s\n",
                              idx, lag, rebuild ? "true" : "false", solver->force_reuse_preconditioner ? "true" : "false", reuse_pc ? "true" : "false"));
        solver->pmg_lag_solve_index++;
      }
      if (solver->force_reuse_preconditioner) reuse_pc = PETSC_TRUE;
      PetscCall(RefreshKSPOperators(solver->ksp, solver->app, solver->A, PETSC_FALSE, reuse_pc));
    }
    ksp = solver->ksp;
  } else {
    PetscCall(KSPCreate(PetscObjectComm((PetscObject)solver->dm), &ksp));
    PetscCall(ConfigureKSP(ksp, solver->dm, solver->actx, solver->app, solver->A, nonlinear_tangent, solver));
  }
  if (solver->app->use_deflation && nonlinear_tangent && solver->n_raw_basis > 0) {
    PetscReal reported_rel = PETSC_MAX_REAL;

    PetscCall(LinearSolverAOrthogonalizeBasis(solver, solver->A, label));
    PetscCall(VecZeroEntries(x));
    if (solver->app->deflation_solver == DEFLATION_SOLVER_FGMRES) {
      PetscCall(DeflatedFGMRESSolve(solver, ksp, rhs, x, label, its, &reported_rel));
    } else if (solver->app->deflation_solver == DEFLATION_SOLVER_MATLAB_DFGMRES) {
      PetscCall(DeflatedMatlabDFGMRESSolve(solver, ksp, rhs, x, label, its, &reported_rel));
    } else {
      PetscCall(DeflatedCGSolve(solver, ksp, rhs, x, label, its, &reported_rel));
    }
    PetscCall(CheckLinearSolutionExplicit(solver->dm, solver->app, solver->A, rhs, x, label, *its, reported_rel));
  } else {
    PetscCall(KSPSolve(ksp, rhs, x));
    PetscCall(CheckLinearSolve(solver->dm, solver->actx, solver->app, solver->A, rhs, x, label, ksp, its));
  }
  if (!solver->reuse) PetscCall(KSPDestroy(&ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscInt       step;
  char           phase[16];
  PetscReal      omega;
  PetscReal      lambda;
  PetscReal      d_omega;
  PetscReal      d_lambda;
  PetscReal      u_max;
  PetscInt       attempts;
  PetscInt       newton_iterations;
  PetscInt       linear_iterations;
  PetscInt       line_search_iterations;
  PetscReal      rel_residual;
  PetscReal      rel_correction;
  PetscLogDouble step_wall_time;
  char           stop_reason[64];
} SSRCurveRow;

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
} SSRStats;

static PetscErrorCode DotOmega(Vec f_ext, Vec u, PetscReal *omega)
{
  PetscScalar dot;

  PetscFunctionBeginUser;
  PetscCall(VecDot(f_ext, u, &dot));
  *omega = PetscRealPart(dot);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DisplacementMax(Vec u, PetscReal *u_max)
{
  PetscFunctionBeginUser;
  PetscCall(VecNorm(u, NORM_INFINITY, u_max));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ScaleToOmega(Vec f_ext, Vec u, PetscReal omega_target)
{
  PetscReal omega;

  PetscFunctionBeginUser;
  PetscCall(DotOmega(f_ext, u, &omega));
  PetscCheck(PetscAbsReal(omega) > 1.0e-30, PetscObjectComm((PetscObject)u), PETSC_ERR_NOT_CONVERGED,
             "Cannot rescale displacement to omega %.6e because f_ext^T u is %.6e", (double)omega_target, (double)omega);
  PetscCall(VecScale(u, omega_target / omega));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AssembleResidualRel(AssemblyCtx *actx, PetscReal lambda, Vec u, Vec f_ext, Vec residual, PetscReal rhs_norm, PetscReal *rel)
{
  PetscFunctionBeginUser;
  PetscCall(AssemblePlasticResidualJacobian(actx, lambda, u, f_ext, NULL, residual, PETSC_FALSE));
  PetscCall(ResidualNormFree(actx, residual, rhs_norm, rel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FixedLambdaDirectionalDamping(AssemblyCtx *actx, AppCtx *app, PetscReal lambda, Vec u, Vec du, Vec current_residual, Vec f_ext, Vec u_trial, Vec r_trial, PetscReal *alpha_out, PetscInt *ls_its, PetscReal *initial_decrease)
{
  PetscReal alpha = 1.0, alpha_min = 0.0, alpha_max = 1.0;
  PetscScalar dot;

  PetscFunctionBeginUser;
  *alpha_out       = 0.0;
  *ls_its          = 0;
  *initial_decrease = PETSC_MAX_REAL;

  PetscCall(VecDot(current_residual, du, &dot));
  *initial_decrease = PetscRealPart(dot);
  if (PetscIsInfOrNanReal(*initial_decrease) || *initial_decrease >= 0.0) PetscFunctionReturn(PETSC_SUCCESS);
  if (app->it_damp_max == 0) {
    *alpha_out = 1.0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  for (PetscInt damp = 0; damp < app->it_damp_max; ++damp) {
    PetscReal decrease;

    (*ls_its)++;
    PetscCall(VecWAXPY(u_trial, alpha, du, u));
    PetscCall(AssemblePlasticResidualJacobian(actx, lambda, u_trial, f_ext, NULL, r_trial, PETSC_FALSE));
    PetscCall(ZeroConstrainedVector(actx->constrained_is, r_trial));
    PetscCall(VecDot(r_trial, du, &dot));
    decrease = PetscRealPart(dot);
    if (!PetscIsInfOrNanReal(decrease) && decrease < 0.0) {
      if (alpha == 1.0) break;
      alpha_min = alpha;
    } else {
      alpha_max = alpha;
    }
    alpha = 0.5 * (alpha_min + alpha_max);
  }
  *alpha_out = alpha;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StoppingByCriterion(const char criterion[], PetscReal rel_residual, PetscReal rel_correction, PetscReal abs_delta_lambda, PetscReal tol, PetscBool have_correction, PetscBool *stop)
{
  PetscBool is_rel_res, is_rel_corr, is_abs_lambda;

  PetscFunctionBeginUser;
  *stop = PETSC_FALSE;
  PetscCall(PetscStrcasecmp(criterion, "relative_residual", &is_rel_res));
  PetscCall(PetscStrcasecmp(criterion, "relative_correction", &is_rel_corr));
  PetscCall(PetscStrcasecmp(criterion, "absolute_delta_lambda", &is_abs_lambda));
  if (is_rel_res) *stop = (PetscBool)(rel_residual <= tol);
  else if (is_rel_corr) *stop = (PetscBool)(have_correction && rel_correction <= tol);
  else if (is_abs_lambda) *stop = (PetscBool)(have_correction && abs_delta_lambda <= tol);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeLambdaDerivativeFD(AssemblyCtx *actx, AppCtx *app, PetscReal lambda, Vec u, Vec f_ext, Vec residual, Vec residual_eps, Vec G)
{
  PetscReal eps = PetscMax(1.0e-12, app->newton_rtol / 1000.0);

  PetscFunctionBeginUser;
  PetscCall(AssemblePlasticResidualJacobian(actx, lambda + eps, u, f_ext, NULL, residual_eps, PETSC_FALSE));
  PetscCall(VecWAXPY(G, -1.0, residual, residual_eps));
  PetscCall(VecScale(G, 1.0 / eps));
  PetscCall(ZeroConstrainedVector(actx->constrained_is, G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildRegularizedOperator(Mat Areg, Mat Kelastic, Mat Ktangent, PetscReal r)
{
  PetscFunctionBeginUser;
  PetscCall(MatSetOption(Areg, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatZeroEntries(Areg));
  PetscCall(MatAXPY(Areg, 1.0 - r, Ktangent, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatAXPY(Areg, r, Kelastic, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatAssemblyBegin(Areg, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Areg, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(Areg, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CurveWriteHeader(FILE *fp)
{
  PetscFunctionBeginUser;
  if (!fp) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(fprintf(fp, "step,phase,omega,lambda,d_omega,d_lambda,u_max,attempts,newton_iterations,linear_iterations,line_search_iterations,rel_residual,rel_correction,step_wall_time,stop_reason\n") > 0,
             PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Failed to write continuation CSV header");
  fflush(fp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CurveWriteRow(FILE *fp, const SSRCurveRow *row)
{
  PetscFunctionBeginUser;
  if (!fp) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(fprintf(fp,
                     "%" PetscInt_FMT ",%s,%.16e,%.16e,%.16e,%.16e,%.16e,%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ",%.16e,%.16e,%.16e,%s\n",
                     row->step, row->phase, (double)row->omega, (double)row->lambda, (double)row->d_omega, (double)row->d_lambda,
                     (double)row->u_max, row->attempts, row->newton_iterations, row->linear_iterations, row->line_search_iterations,
                     (double)row->rel_residual, (double)row->rel_correction, (double)row->step_wall_time, row->stop_reason) > 0,
             PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Failed to write continuation CSV row");
  fflush(fp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FillCurveRow(SSRCurveRow *row, PetscInt step, const char phase[], PetscReal omega, PetscReal lambda, PetscReal d_omega, PetscReal d_lambda, Vec u, PetscInt attempts, const NewtonStats *stats, const char reason[])
{
  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(row, sizeof(*row)));
  row->step     = step;
  row->omega    = omega;
  row->lambda   = lambda;
  row->d_omega  = d_omega;
  row->d_lambda = d_lambda;
  row->attempts = attempts;
  PetscCall(PetscStrncpy(row->phase, phase, sizeof(row->phase)));
  PetscCall(PetscStrncpy(row->stop_reason, reason, sizeof(row->stop_reason)));
  PetscCall(DisplacementMax(u, &row->u_max));
  if (stats) {
    row->newton_iterations     = stats->newton_its;
    row->linear_iterations     = stats->total_linear_its;
    row->line_search_iterations = stats->line_search_its;
    row->rel_residual          = stats->final_rel;
    row->rel_correction        = stats->final_rel_correction;
    row->step_wall_time        = stats->wall_time;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FixedLambdaNewtonSolve(DM dm, AssemblyCtx *actx, AppCtx *app, LinearSolverCtx *solver, Mat Areg, Mat Kelastic, Mat Ktangent, Vec f_ext, Vec u, PetscReal lambda, PetscReal rhs_norm, const char criterion[], PetscReal stop_tol, NewtonStats *stats)
{
  Vec            residual = NULL, rhs = NULL, du = NULL, u_trial = NULL, r_trial = NULL;
  PetscReal      rel = PETSC_MAX_REAL, rel_corr = PETSC_MAX_REAL, r = app->r_min;
  PetscInt       total_linear_its = 0, newton_its = 0, line_search_its = 0;
  PetscLogDouble t_start, t0, t1, assembly_time = 0.0, solve_time = 0.0;
  PetscBool      stop = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(stats, sizeof(*stats)));
  PetscCall(PetscTime(&t_start));
  PetscCall(VecDuplicate(f_ext, &residual));
  PetscCall(VecDuplicate(f_ext, &rhs));
  PetscCall(VecDuplicate(f_ext, &du));
  PetscCall(VecDuplicate(f_ext, &u_trial));
  PetscCall(VecDuplicate(f_ext, &r_trial));

  for (PetscInt it = 0; it < app->newton_max_it; ++it) {
    PetscInt  linear_its = 0;
    PetscReal alpha = 1.0, du_norm, u_norm, initial_decrease = PETSC_MAX_REAL;
    PetscInt  ls_its = 0;

    PetscCall(PetscTime(&t0));
    PetscCall(AssemblePlasticResidualJacobian(actx, lambda, u, f_ext, Ktangent, residual, PETSC_TRUE));
    PetscCall(BuildRegularizedOperator(Areg, Kelastic, Ktangent, r));
    PetscCall(PetscTime(&t1));
    assembly_time += t1 - t0;
    PetscCall(ResidualNormFree(actx, residual, rhs_norm, &rel));
    PetscCall(StoppingByCriterion(criterion, rel, rel_corr, PETSC_MAX_REAL, stop_tol, PETSC_FALSE, &stop));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "SSR_INIT_NEWTON lambda=%.8e it=%" PetscInt_FMT " rel_res=%.6e rel_corr=%.6e stop=%s\n",
                          (double)lambda, it, (double)rel, (double)rel_corr, stop ? "true" : "false"));
    if (stop || rel <= app->newton_rtol) {
      stats->converged = PETSC_TRUE;
      break;
    }

    PetscCall(VecCopy(residual, rhs));
    PetscCall(VecScale(rhs, -1.0));
    PetscCall(ApplyZeroDirichlet(actx->constrained_is, Areg, rhs));
    PetscCall(VecZeroEntries(du));
    solver->A = Areg;
    PetscCall(PetscTime(&t0));
    PetscCall(SolveLinearSystem(solver, rhs, du, "SSR fixed-lambda Newton correction", u, PETSC_TRUE, &linear_its));
    PetscCall(PetscTime(&t1));
    solve_time += t1 - t0;
    total_linear_its += linear_its;
    ++newton_its;

    PetscCall(VecNorm(du, NORM_2, &du_norm));
    PetscCall(VecNorm(u, NORM_2, &u_norm));
    PetscCall(FixedLambdaDirectionalDamping(actx, app, lambda, u, du, residual, f_ext, u_trial, r_trial, &alpha, &ls_its, &initial_decrease));
    line_search_its += ls_its;
    rel_corr = (alpha * du_norm) / PetscMax(u_norm, 1.0e-30);
    PetscCall(VecScale(du, alpha));
    PetscCall(VecAXPY(u, 1.0, du));
    PetscCall(StoppingByCriterion(criterion, rel, rel_corr, PETSC_MAX_REAL, stop_tol, (PetscBool)(alpha > 0.0), &stop));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "SSR_INIT_NEWTON_ACCEPT lambda=%.8e it=%" PetscInt_FMT " alpha=%.6e rel_res=%.6e rel_corr=%.6e r=%.6e directional=%.6e stop=%s\n",
                          (double)lambda, it, (double)alpha, (double)rel, (double)rel_corr, (double)r, (double)initial_decrease, stop ? "true" : "false"));
    if (stop) {
      stats->converged = PETSC_TRUE;
      break;
    }
    if (alpha < 0.1) {
      if (alpha == 0.0) r *= 2.0;
      else r *= PetscPowReal(2.0, 0.25);
    } else {
      PetscCall(LinearSolverAppendRawBasis(solver, du, "SSR fixed-lambda Newton correction"));
      if (alpha > 0.5) r = PetscMax(r / PetscSqrtReal(2.0), app->r_min);
    }
    if (alpha == 0.0 && r > 1.0) break;
  }
  PetscCall(PetscTime(&t1));
  stats->final_rel             = rel;
  stats->final_rel_correction  = rel_corr;
  stats->newton_its            = newton_its;
  stats->total_linear_its      = total_linear_its;
  stats->line_search_its       = line_search_its;
  stats->assembly_time         = assembly_time;
  stats->solve_time            = solve_time;
  stats->wall_time             = t1 - t_start;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                        "SSR_INIT_NEWTON_SUMMARY lambda=%.8e converged=%s final_rel=%.6e final_rel_correction=%.6e newton_its=%" PetscInt_FMT " linear_its=%" PetscInt_FMT " wall_time=%.6g\n",
                        (double)lambda, stats->converged ? "true" : "false", (double)stats->final_rel, (double)stats->final_rel_correction,
                        stats->newton_its, stats->total_linear_its, (double)stats->wall_time));

  PetscCall(VecDestroy(&residual));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&du));
  PetscCall(VecDestroy(&u_trial));
  PetscCall(VecDestroy(&r_trial));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IndirectNewtonLineSearch(AssemblyCtx *actx, AppCtx *app, Vec f_ext, Vec u, PetscReal lambda, Vec dU, PetscReal d_lambda, PetscReal omega_target, PetscReal current_rel, PetscReal rhs_norm, Vec u_trial, Vec r_trial, PetscReal *alpha_out, PetscReal *trial_rel_out, PetscInt *ls_its)
{
  PetscReal alpha = 1.0, trial_rel = PETSC_MAX_REAL, last_alpha = 0.0, last_trial_rel = PETSC_MAX_REAL;

  PetscFunctionBeginUser;
  (void)omega_target;
  *ls_its = 0;
  if (PetscIsInfOrNanReal(d_lambda)) {
    *alpha_out = 0.0;
    *trial_rel_out = PETSC_MAX_REAL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  for (PetscInt damp = 0; damp < PetscMax(1, app->it_damp_max); ++damp) {
    if (lambda + alpha * d_lambda <= 0.0) {
      alpha *= 0.5;
      (*ls_its)++;
      continue;
    }
    PetscCall(VecWAXPY(u_trial, alpha, dU, u));
    PetscCall(AssembleResidualRel(actx, lambda + alpha * d_lambda, u_trial, f_ext, r_trial, rhs_norm, &trial_rel));
    (*ls_its)++;
    if (!PetscIsInfOrNanReal(trial_rel)) {
      last_alpha     = alpha;
      last_trial_rel = trial_rel;
      if (trial_rel < current_rel) break;
    }
    alpha *= 0.5;
  }
  *alpha_out     = last_alpha;
  *trial_rel_out = last_trial_rel;
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscInt  iteration;
  PetscReal lambda;
  PetscReal r;
  PetscReal alpha;
  PetscReal delta_lambda;
  PetscReal accepted_delta_lambda;
  PetscReal rel_residual;
  PetscReal rel_correction;
  PetscReal stopping_value;
  PetscInt  dW_iterations;
  PetscInt  dV_iterations;
  PetscInt  linear_iterations;
  PetscInt  line_search_iterations;
  PetscInt  deflation_basis_dim_solve;
  PetscInt  deflation_basis_dim_end;
  char      status[32];
} StepReplayExpectedRow;

typedef struct {
  PetscBool              loaded;
  PetscInt               nrows;
  StepReplayExpectedRow *rows;
} StepReplayExpected;

static PetscErrorCode StepReplayExpectedLoad(MPI_Comm comm, const char dir[], StepReplayExpected *expected)
{
  char path[PETSC_MAX_PATH_LEN], line[1024];
  FILE *fh = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(expected, sizeof(*expected)));
  if (!dir || !dir[0]) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSNPrintf(path, sizeof(path), "%s/step_expected.csv", dir));
  fh = fopen(path, "r");
  if (!fh) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCalloc1(1024, &expected->rows));
  if (!fgets(line, sizeof(line), fh)) {
    fclose(fh);
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  while (fgets(line, sizeof(line), fh)) {
    StepReplayExpectedRow row;
    int                   nscan;
    long long             iteration, dW_iterations, dV_iterations, linear_iterations, line_search_iterations, basis_solve, basis_end;

    PetscCall(PetscMemzero(&row, sizeof(row)));
    nscan = sscanf(line,
                   "%lld,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lld,%lld,%lld,%lld,%lld,%lld,%31[^,\n]",
                   &iteration, &row.lambda, &row.r, &row.alpha, &row.delta_lambda, &row.accepted_delta_lambda,
                   &row.rel_residual, &row.rel_correction, &row.stopping_value, &dW_iterations, &dV_iterations,
                   &linear_iterations, &line_search_iterations, &basis_solve, &basis_end, row.status);
    if (nscan < 15) continue;
    row.iteration                   = (PetscInt)iteration;
    row.dW_iterations               = (PetscInt)dW_iterations;
    row.dV_iterations               = (PetscInt)dV_iterations;
    row.linear_iterations           = (PetscInt)linear_iterations;
    row.line_search_iterations      = (PetscInt)line_search_iterations;
    row.deflation_basis_dim_solve   = (PetscInt)basis_solve;
    row.deflation_basis_dim_end     = (PetscInt)basis_end;
    PetscCheck(expected->nrows < 1024, comm, PETSC_ERR_ARG_SIZ, "Too many step replay expected rows in %s", path);
    expected->rows[expected->nrows++] = row;
  }
  fclose(fh);
  expected->loaded = PETSC_TRUE;
  PetscCall(PetscPrintf(comm, "STEP_REPLAY_EXPECTED file=%s rows=%" PetscInt_FMT "\n", path, expected->nrows));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static const StepReplayExpectedRow *StepReplayExpectedFind(const StepReplayExpected *expected, PetscInt iteration)
{
  if (!expected || !expected->loaded) return NULL;
  for (PetscInt i = 0; i < expected->nrows; ++i) {
    if (expected->rows[i].iteration == iteration) return &expected->rows[i];
  }
  return NULL;
}

static PetscErrorCode StepReplayExpectedDestroy(StepReplayExpected *expected)
{
  PetscFunctionBeginUser;
  PetscCall(PetscFree(expected->rows));
  expected->loaded = PETSC_FALSE;
  expected->nrows  = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IndirectNewtonSolve(DM dm, AssemblyCtx *actx, AppCtx *app, LinearSolverCtx *solver, Mat Areg, Mat Kelastic, Mat Ktangent, Vec f_ext, Vec u, PetscReal *lambda, PetscReal omega_target, PetscReal rhs_norm, PetscReal initial_r, const char criterion[], PetscReal stop_tol, NewtonStats *stats)
{
  Vec            residual = NULL, residual_eps = NULL, G = NULL, rhsW = NULL, rhsV = NULL, dW = NULL, dV = NULL, dU = NULL, u_trial = NULL, r_trial = NULL;
  PetscReal      rel = PETSC_MAX_REAL, rel_corr = PETSC_MAX_REAL, r = PetscMax(initial_r, app->r_min), abs_dlambda = PETSC_MAX_REAL;
  PetscInt       total_linear_its = 0, newton_its = 0, line_search_its = 0;
  PetscInt       prev_itsW = -1, prev_itsV = -1;
  PetscLogDouble t_start, t0, t1, assembly_time = 0.0, solve_time = 0.0;
  PetscBool      stop = PETSC_FALSE, compute_diffs = PETSC_TRUE;
  PetscBool      have_pair_matrix = PETSC_FALSE;
  PetscInt       pair_matrix_start_it = -1;
  PetscReal      pair_matrix_r = r;
  const PetscInt krylov_persistent_start_added = solver->deflation_krylov_persistent_added;
  StepReplayExpected expected;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(stats, sizeof(*stats)));
  PetscCall(StepReplayExpectedLoad(PetscObjectComm((PetscObject)dm), app->step_replay_dir, &expected));
  PetscCall(PetscTime(&t_start));
  PetscCall(VecDuplicate(f_ext, &residual));
  PetscCall(VecDuplicate(f_ext, &residual_eps));
  PetscCall(VecDuplicate(f_ext, &G));
  PetscCall(VecDuplicate(f_ext, &rhsW));
  PetscCall(VecDuplicate(f_ext, &rhsV));
  PetscCall(VecDuplicate(f_ext, &dW));
  PetscCall(VecDuplicate(f_ext, &dV));
  PetscCall(VecDuplicate(f_ext, &dU));
  PetscCall(VecDuplicate(f_ext, &u_trial));
  PetscCall(VecDuplicate(f_ext, &r_trial));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                        "DEFLATION_KRYLOV_PERSISTENT_CONFIG enabled=%s basis_tol=%.6e raw_cols_start=%" PetscInt_FMT " total_added_start=%" PetscInt_FMT "\n",
                        app->deflation_krylov_persistent ? "true" : "false", (double)app->deflation_krylov_basis_tol,
                        solver->n_raw_basis, krylov_persistent_start_added));

  for (PetscInt it = 0; it < app->newton_max_it; ++it) {
    PetscInt  itsW = 0, itsV = 0, ls_its = 0;
    PetscReal denom, numer, alpha = 0.0, trial_rel = PETSC_MAX_REAL, dU_norm, u_norm;
    PetscScalar dot;
    const PetscReal lambda_start = *lambda;
    const PetscReal r_start = r;
    const PetscInt  basis_start = solver->n_raw_basis;

    PetscCall(PetscTime(&t0));
    const PetscBool freeze_pair_matrix =
      (PetscBool)(app->indirect_newton_pair_freeze_matrix && have_pair_matrix && (it % 2 == 1));

    if (freeze_pair_matrix) {
      PetscCall(AssemblePlasticResidualJacobian(actx, *lambda, u, f_ext, NULL, residual, PETSC_FALSE));
    } else {
      PetscCall(AssemblePlasticResidualJacobian(actx, *lambda, u, f_ext, Ktangent, residual, PETSC_TRUE));
    }
    PetscCall(ResidualNormFree(actx, residual, rhs_norm, &rel));
    if (compute_diffs) PetscCall(ComputeLambdaDerivativeFD(actx, app, *lambda, u, f_ext, residual, residual_eps, G));
    if (!freeze_pair_matrix) {
      PetscCall(BuildRegularizedOperator(Areg, Kelastic, Ktangent, r));
      have_pair_matrix    = PETSC_TRUE;
      pair_matrix_start_it = it;
      pair_matrix_r        = r;
    }
    PetscCall(PetscTime(&t1));
    assembly_time += t1 - t0;
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                          "SSR_PAIR_MATRIX omega=%.8e it=%" PetscInt_FMT " enabled=%s mode=%s pair_start_it=%" PetscInt_FMT " current_r=%.6e matrix_r=%.6e force_pc_reuse=%s\n",
                          (double)omega_target, it, app->indirect_newton_pair_freeze_matrix ? "true" : "false",
                          freeze_pair_matrix ? "frozen" : "fresh", pair_matrix_start_it, (double)r, (double)pair_matrix_r,
                          freeze_pair_matrix ? "true" : "false"));

    PetscCall(StoppingByCriterion(criterion, rel, rel_corr, abs_dlambda, stop_tol, PETSC_FALSE, &stop));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                          "SSR_NEWTON omega=%.8e it=%" PetscInt_FMT " lambda=%.8e rel_res=%.6e rel_corr=%.6e r=%.6e compute_diffs=%s stop=%s\n",
                          (double)omega_target, it, (double)*lambda, (double)rel, (double)rel_corr, (double)r, compute_diffs ? "true" : "false", stop ? "true" : "false"));
    if (stop || (rel <= app->newton_rtol && it > 0)) {
      stats->converged = PETSC_TRUE;
      break;
    }

    PetscCall(VecCopy(G, rhsW));
    PetscCall(VecScale(rhsW, -1.0));
    PetscCall(VecCopy(residual, rhsV));
    PetscCall(VecScale(rhsV, -1.0));
    PetscCall(ZeroConstrainedVector(actx->constrained_is, rhsW));
    PetscCall(ZeroConstrainedVector(actx->constrained_is, rhsV));
    solver->A = Areg;
    PetscCall(VecZeroEntries(dW));
    PetscCall(VecZeroEntries(dV));
    PetscCall(PetscTime(&t0));
    {
      const PetscBool saved_force_reuse = solver->force_reuse_preconditioner;

    if (app->deflation_intra_newton_recycle && app->use_deflation && solver->n_raw_basis > 0) {
      const PetscBool cheap_is_v = (PetscBool)(prev_itsW >= 0 && prev_itsV >= 0 && prev_itsV < prev_itsW);
      const PetscInt  raw_snapshot = solver->n_raw_basis;
      PetscInt        temp_added = 0;

      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                            "DEFLATION_INTRANEWTON_SELECT omega=%.8e it=%" PetscInt_FMT " enabled=true previous_w=%" PetscInt_FMT " previous_v=%" PetscInt_FMT " first=%s target=%s raw_cols=%" PetscInt_FMT "\n",
                            (double)omega_target, it, prev_itsW, prev_itsV, cheap_is_v ? "dV" : "dW", cheap_is_v ? "dW" : "dV",
                            raw_snapshot));
      if (cheap_is_v) {
        solver->force_reuse_preconditioner = freeze_pair_matrix;
        PetscCall(LinearSolverBeginRecycleCapture(solver));
        PetscCall(SolveLinearSystem(solver, rhsV, dV, "SSR indirect dV", u, PETSC_TRUE, &itsV));
        PetscCall(LinearSolverEndRecycleCapture(solver));
        PetscCall(LinearSolverAppendTemporaryRecycleBasis(solver, "SSR intra-newton dV-to-dW", &temp_added));
        solver->force_reuse_preconditioner = PETSC_TRUE;
        PetscCall(SolveLinearSystem(solver, rhsW, dW, "SSR indirect dW", u, PETSC_TRUE, &itsW));
      } else {
        solver->force_reuse_preconditioner = freeze_pair_matrix;
        PetscCall(LinearSolverBeginRecycleCapture(solver));
        PetscCall(SolveLinearSystem(solver, rhsW, dW, "SSR indirect dW", u, PETSC_TRUE, &itsW));
        PetscCall(LinearSolverEndRecycleCapture(solver));
        PetscCall(LinearSolverAppendTemporaryRecycleBasis(solver, "SSR intra-newton dW-to-dV", &temp_added));
        solver->force_reuse_preconditioner = PETSC_TRUE;
        PetscCall(SolveLinearSystem(solver, rhsV, dV, "SSR indirect dV", u, PETSC_TRUE, &itsV));
      }
      PetscCall(LinearSolverTruncateRawBasis(solver, raw_snapshot));
      PetscCall(LinearSolverClearRecycleBasis(solver));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                            "DEFLATION_INTRANEWTON_RESULT omega=%.8e it=%" PetscInt_FMT " first=%s temp_added=%" PetscInt_FMT " linear_w=%" PetscInt_FMT " linear_v=%" PetscInt_FMT "\n",
                            (double)omega_target, it, cheap_is_v ? "dV" : "dW", temp_added, itsW, itsV));
    } else {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                            "DEFLATION_INTRANEWTON_SELECT omega=%.8e it=%" PetscInt_FMT " enabled=false previous_w=%" PetscInt_FMT " previous_v=%" PetscInt_FMT " raw_cols=%" PetscInt_FMT "\n",
                            (double)omega_target, it, prev_itsW, prev_itsV, solver->n_raw_basis));
      solver->force_reuse_preconditioner = freeze_pair_matrix;
      PetscCall(SolveLinearSystem(solver, rhsW, dW, "SSR indirect dW", u, PETSC_TRUE, &itsW));
      solver->force_reuse_preconditioner = PETSC_TRUE;
      PetscCall(SolveLinearSystem(solver, rhsV, dV, "SSR indirect dV", u, PETSC_TRUE, &itsV));
    }
      solver->force_reuse_preconditioner = saved_force_reuse;
    }
    PetscCall(PetscTime(&t1));
    solve_time += t1 - t0;
    total_linear_its += itsW + itsV;
    prev_itsW = itsW;
    prev_itsV = itsV;
    ++newton_its;

    PetscCall(VecDot(f_ext, dW, &dot));
    denom = PetscRealPart(dot);
    PetscCall(VecDot(f_ext, dV, &dot));
    numer = PetscRealPart(dot);
    PetscCheck(PetscAbsReal(denom) > 1.0e-30, PetscObjectComm((PetscObject)dm), PETSC_ERR_NOT_CONVERGED,
               "Indirect Newton has singular constraint denominator f_ext^T dW=%.6e", (double)denom);
    abs_dlambda = PetscAbsReal(-numer / denom);
    PetscCall(VecCopy(dV, dU));
    PetscCall(VecAXPY(dU, -numer / denom, dW));
    PetscCall(IndirectNewtonLineSearch(actx, app, f_ext, u, *lambda, dU, -numer / denom, omega_target, rel, rhs_norm, u_trial, r_trial, &alpha, &trial_rel, &ls_its));
    line_search_its += ls_its;
    PetscCall(VecNorm(dU, NORM_2, &dU_norm));
    PetscCall(VecNorm(u, NORM_2, &u_norm));
    rel_corr = (alpha * dU_norm) / PetscMax(u_norm, 1.0);

    PetscCall(StoppingByCriterion(criterion, rel, rel_corr, abs_dlambda, stop_tol, (PetscBool)(alpha > 0.0), &stop));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                          "SSR_NEWTON_ACCEPT omega=%.8e it=%" PetscInt_FMT " alpha=%.6e lambda=%.8e d_lambda=%.6e rel_res=%.6e trial_rel=%.6e rel_corr=%.6e linear_w=%" PetscInt_FMT " linear_v=%" PetscInt_FMT " linear_its=%" PetscInt_FMT " stop=%s\n",
                          (double)omega_target, it, (double)alpha, (double)*lambda, (double)(-numer / denom), (double)rel, (double)trial_rel,
                          (double)rel_corr, itsW, itsV, itsW + itsV, stop ? "true" : "false"));
    {
      const StepReplayExpectedRow *erow = StepReplayExpectedFind(&expected, it + 1);

      if (erow) {
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                              "STEP_REPLAY_NEWTON_COMPARE it=%" PetscInt_FMT " expected_w=%" PetscInt_FMT " c_w=%" PetscInt_FMT " expected_v=%" PetscInt_FMT " c_v=%" PetscInt_FMT " expected_total=%" PetscInt_FMT " c_total=%" PetscInt_FMT " expected_alpha=%.17e c_alpha=%.17e expected_lambda=%.17e c_lambda=%.17e expected_r=%.17e c_r=%.17e expected_dlambda=%.17e c_dlambda=%.17e expected_rel_res=%.17e c_rel_res=%.17e expected_rel_corr=%.17e c_rel_corr=%.17e expected_ls=%" PetscInt_FMT " c_ls=%" PetscInt_FMT " expected_basis=%" PetscInt_FMT " c_basis=%" PetscInt_FMT " expected_status=%s c_stop=%s\n",
                              it + 1, erow->dW_iterations, itsW, erow->dV_iterations, itsV, erow->linear_iterations,
                              itsW + itsV, (double)erow->alpha, (double)alpha, (double)erow->lambda, (double)lambda_start,
                              (double)erow->r, (double)r_start, (double)erow->delta_lambda, (double)(-numer / denom),
                              (double)erow->rel_residual, (double)rel, (double)erow->rel_correction, (double)rel_corr,
                              erow->line_search_iterations, ls_its, erow->deflation_basis_dim_solve, basis_start,
                              erow->status, stop ? "true" : "false"));
      }
    }
    if (alpha > 0.0) {
      PetscCall(VecCopy(u_trial, u));
      PetscCall(ScaleToOmega(f_ext, u, omega_target));
      *lambda += alpha * (-numer / denom);
      rel = trial_rel;
    }
    if (stop) {
      stats->converged = PETSC_TRUE;
      break;
    }
    compute_diffs = PETSC_TRUE;
    {
      PetscBool appended_pair_solutions = PETSC_FALSE;

      if (app->indirect_newton_pair_freeze_matrix && alpha > 0.0) {
        PetscCall(LinearSolverAppendRawBasis(solver, dW, "SSR indirect pair dW"));
        PetscCall(LinearSolverAppendRawBasis(solver, dV, "SSR indirect pair dV"));
        appended_pair_solutions = PETSC_TRUE;
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                              "DEFLATION_PAIR_SOLUTIONS_ADD omega=%.8e it=%" PetscInt_FMT " raw_cols=%" PetscInt_FMT " alpha=%.6e\n",
                              (double)omega_target, it, solver->n_raw_basis, (double)alpha));
      }

    if (alpha < 0.1) {
      if (alpha == 0.0) {
        compute_diffs = PETSC_FALSE;
        r *= 2.0;
      } else {
        r *= PetscPowReal(2.0, 0.25);
      }
    } else if (alpha > 0.5) {
      if (!appended_pair_solutions) {
        PetscCall(LinearSolverAppendRawBasis(solver, dW, "SSR indirect dW"));
        PetscCall(LinearSolverAppendRawBasis(solver, dV, "SSR indirect dV"));
      }
      r = PetscMax(r / PetscSqrtReal(2.0), app->r_min);
    } else {
      if (!appended_pair_solutions) {
        PetscCall(LinearSolverAppendRawBasis(solver, dW, "SSR indirect dW"));
        PetscCall(LinearSolverAppendRawBasis(solver, dV, "SSR indirect dV"));
      }
    }
    }
    if (alpha == 0.0 && r > 1.0) break;
  }

  PetscCall(AssembleResidualRel(actx, *lambda, u, f_ext, residual, rhs_norm, &rel));
  PetscCall(PetscTime(&t1));
  stats->final_rel             = rel;
  stats->final_rel_correction  = rel_corr;
  stats->newton_its            = newton_its;
  stats->total_linear_its      = total_linear_its;
  stats->line_search_its       = line_search_its;
  stats->assembly_time         = assembly_time;
  stats->solve_time            = solve_time;
  stats->wall_time             = t1 - t_start;
  if (!stats->converged && rel <= 10.0 * app->newton_rtol) stats->converged = PETSC_TRUE;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                        "SSR_NEWTON_SUMMARY omega=%.8e converged=%s lambda=%.8e final_rel=%.6e final_rel_correction=%.6e newton_its=%" PetscInt_FMT " linear_its=%" PetscInt_FMT " line_search_its=%" PetscInt_FMT " assembly_time=%.6g solve_time=%.6g wall_time=%.6g\n",
                        (double)omega_target, stats->converged ? "true" : "false", (double)*lambda, (double)stats->final_rel,
                        (double)stats->final_rel_correction, stats->newton_its, stats->total_linear_its, stats->line_search_its,
                        (double)stats->assembly_time, (double)stats->solve_time, (double)stats->wall_time));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                        "DEFLATION_KRYLOV_PERSISTENT_SUMMARY enabled=%s added_this_newton=%" PetscInt_FMT " added_total=%" PetscInt_FMT " raw_cols_end=%" PetscInt_FMT "\n",
                        app->deflation_krylov_persistent ? "true" : "false",
                        solver->deflation_krylov_persistent_added - krylov_persistent_start_added,
                        solver->deflation_krylov_persistent_added, solver->n_raw_basis));
  PetscCall(StepReplayExpectedDestroy(&expected));

  PetscCall(VecDestroy(&residual));
  PetscCall(VecDestroy(&residual_eps));
  PetscCall(VecDestroy(&G));
  PetscCall(VecDestroy(&rhsW));
  PetscCall(VecDestroy(&rhsV));
  PetscCall(VecDestroy(&dW));
  PetscCall(VecDestroy(&dV));
  PetscCall(VecDestroy(&dU));
  PetscCall(VecDestroy(&u_trial));
  PetscCall(VecDestroy(&r_trial));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SSRContinuationSolve(DM dm, AssemblyCtx *actx, AppCtx *app, LinearSolverCtx *solver, Mat Areg, Mat Kelastic, Mat Ktangent, Vec f_ext, PetscReal rhs_norm, SSRStats *stats)
{
  MPI_Comm       comm = PetscObjectComm((PetscObject)dm);
  PetscMPIInt    rank;
  FILE          *csv = NULL;
  Vec            u_prev = NULL, u_cur = NULL, u_guess = NULL, u_tmp = NULL;
  PetscReal      lambda_prev, lambda_cur, lambda_guess, omega_prev, omega_cur, d_lambda = app->d_lambda_init, d_omega;
  PetscReal      lambda_init = app->lambda_init;
  PetscInt       init_basis_base;
  PetscInt       attempts = 0, step = 0, omega_reductions = 0;
  PetscLogDouble t_start, t_end;
  NewtonStats    nstats;
  SSRCurveRow    row;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(stats, sizeof(*stats)));
  PetscCall(PetscStrncpy(stats->stop_reason, "running", sizeof(stats->stop_reason)));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    csv = fopen(app->curve_csv, "w");
    PetscCheck(csv, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Failed to open continuation CSV %s", app->curve_csv);
  }
  PetscCall(CurveWriteHeader(csv));
  PetscCall(PetscTime(&t_start));
  init_basis_base = solver->n_raw_basis;
  PetscCall(VecDuplicate(f_ext, &u_prev));
  PetscCall(VecDuplicate(f_ext, &u_cur));
  PetscCall(VecDuplicate(f_ext, &u_guess));
  PetscCall(VecDuplicate(f_ext, &u_tmp));

  while (PETSC_TRUE) {
    const PetscInt basis_snapshot = solver->n_raw_basis;

    PetscCall(VecZeroEntries(u_prev));
    PetscCall(PetscPrintf(comm, "SSR_INIT phase=seed lambda=%.8e d_lambda=%.8e basis_snapshot=%" PetscInt_FMT "\n", (double)lambda_init, (double)d_lambda, basis_snapshot));
    PetscCall(FixedLambdaNewtonSolve(dm, actx, app, solver, Areg, Kelastic, Ktangent, f_ext, u_prev, lambda_init, rhs_norm, app->init_newton_stopping_criterion, app->init_newton_stopping_tol, &nstats));
    attempts++;
    if (nstats.converged) break;
    PetscCall(LinearSolverTruncateRawBasis(solver, basis_snapshot));
    lambda_init *= 0.5;
    d_lambda *= 0.5;
    PetscCheck(d_lambda >= app->d_lambda_min, comm, PETSC_ERR_NOT_CONVERGED, "SSR initialization failed before d_lambda_min %.6e", (double)app->d_lambda_min);
  }
  lambda_prev = lambda_init;
  PetscCall(DotOmega(f_ext, u_prev, &omega_prev));
  PetscCall(LinearSolverAppendRawBasis(solver, u_prev, "SSR init seed"));
  PetscCall(FillCurveRow(&row, step, "init", omega_prev, lambda_prev, 0.0, 0.0, u_prev, attempts, &nstats, "accepted"));
  PetscCall(CurveWriteRow(csv, &row));
  PetscCall(PetscPrintf(comm, "SSR_STEP step=%" PetscInt_FMT " phase=init omega=%.8e lambda=%.8e accepted=true newton_its=%" PetscInt_FMT " linear_its=%" PetscInt_FMT "\n",
                        step, (double)omega_prev, (double)lambda_prev, nstats.newton_its, nstats.total_linear_its));
  stats->accepted_steps++;
  stats->total_newton_its += nstats.newton_its;
  stats->total_linear_its += nstats.total_linear_its;
  stats->total_line_search_its += nstats.line_search_its;
  stats->final_rel            = nstats.final_rel;
  stats->final_rel_correction = nstats.final_rel_correction;
  step++;

  while (PETSC_TRUE) {
    const PetscInt basis_snapshot = solver->n_raw_basis;

    lambda_cur = lambda_prev + d_lambda;
    PetscCall(VecCopy(u_prev, u_cur));
    PetscCall(PetscPrintf(comm, "SSR_INIT phase=advance lambda=%.8e d_lambda=%.8e basis_snapshot=%" PetscInt_FMT "\n", (double)lambda_cur, (double)d_lambda, basis_snapshot));
    PetscCall(FixedLambdaNewtonSolve(dm, actx, app, solver, Areg, Kelastic, Ktangent, f_ext, u_cur, lambda_cur, rhs_norm, app->init_newton_stopping_criterion, app->init_newton_stopping_tol, &nstats));
    attempts++;
    if (!nstats.converged) {
      PetscCall(LinearSolverTruncateRawBasis(solver, basis_snapshot));
      d_lambda *= 0.5;
      PetscCheck(d_lambda >= app->d_lambda_min, comm, PETSC_ERR_NOT_CONVERGED, "SSR second initialization point failed before d_lambda_min %.6e", (double)app->d_lambda_min);
      continue;
    }
    PetscCall(DotOmega(f_ext, u_cur, &omega_cur));
    if ((omega_cur - omega_prev) / PetscMax(1.0, PetscAbsReal(omega_prev)) < 1.0e-5) {
      PetscCall(PetscPrintf(comm, "SSR_INIT phase=advance tiny_omega_increment=true omega_prev=%.8e omega_cur=%.8e shifting_seed=true\n", (double)omega_prev, (double)omega_cur));
      PetscCall(VecCopy(u_cur, u_prev));
      lambda_prev = lambda_cur;
      omega_prev  = omega_cur;
      continue;
    }
    break;
  }
  PetscCall(LinearSolverTruncateRawBasis(solver, init_basis_base));
  PetscCall(LinearSolverAppendRawBasis(solver, u_prev, "SSR accepted init previous"));
  PetscCall(FillCurveRow(&row, step, "init", omega_cur, lambda_cur, omega_cur - omega_prev, lambda_cur - lambda_prev, u_cur, attempts, &nstats, "accepted"));
  PetscCall(CurveWriteRow(csv, &row));
  PetscCall(PetscPrintf(comm, "SSR_STEP step=%" PetscInt_FMT " phase=init omega=%.8e lambda=%.8e d_omega=%.8e d_lambda=%.8e accepted=true newton_its=%" PetscInt_FMT " linear_its=%" PetscInt_FMT "\n",
                        step, (double)omega_cur, (double)lambda_cur, (double)(omega_cur - omega_prev), (double)(lambda_cur - lambda_prev), nstats.newton_its, nstats.total_linear_its));
  stats->accepted_steps++;
  stats->total_newton_its += nstats.newton_its;
  stats->total_linear_its += nstats.total_linear_its;
  stats->total_line_search_its += nstats.line_search_its;
  stats->final_rel            = nstats.final_rel;
  stats->final_rel_correction = nstats.final_rel_correction;
  step++;

  d_omega = omega_cur - omega_prev;
  PetscCheck(d_omega > 0.0, comm, PETSC_ERR_NOT_CONVERGED, "SSR initialization did not produce increasing omega: %.8e -> %.8e", (double)omega_prev, (double)omega_cur);
  if (omega_cur >= app->omega_max || step >= app->continuation_step_max) {
    PetscCall(PetscStrncpy(stats->stop_reason, omega_cur >= app->omega_max ? "omega_max" : "step_max", sizeof(stats->stop_reason)));
  }

  while (PETSC_TRUE) {
    PetscBool      reason_running;
    const PetscInt basis_snapshot = solver->n_raw_basis;
    PetscReal      target = PetscMin(omega_cur + d_omega, app->omega_max);
    PetscReal      alpha_sec = (target - omega_cur) / (omega_cur - omega_prev);
    PetscReal      omega_old = omega_cur;
    PetscBool      branch_double = PETSC_FALSE;

    PetscCall(PetscStrcmp(stats->stop_reason, "running", &reason_running));
    if (!reason_running || step >= app->continuation_step_max || omega_cur >= app->omega_max) break;
    PetscCall(VecCopy(u_cur, u_guess));
    PetscCall(VecWAXPY(u_tmp, -1.0, u_prev, u_cur));
    PetscCall(VecAXPY(u_guess, alpha_sec, u_tmp));
    lambda_guess = lambda_cur + alpha_sec * (lambda_cur - lambda_prev);
    PetscCall(PetscPrintf(comm, "SSR_ATTEMPT step=%" PetscInt_FMT " target_omega=%.8e d_omega=%.8e lambda_predict=%.8e basis_snapshot=%" PetscInt_FMT "\n",
                          step, (double)target, (double)d_omega, (double)lambda_guess, basis_snapshot));
    PetscCall(IndirectNewtonSolve(dm, actx, app, solver, Areg, Kelastic, Ktangent, f_ext, u_guess, &lambda_guess, target, rhs_norm, app->r_min, app->newton_stopping_criterion, app->newton_stopping_tol, &nstats));
    attempts++;
    if (!nstats.converged) {
      PetscCall(LinearSolverTruncateRawBasis(solver, basis_snapshot));
      d_omega *= 0.5;
      omega_reductions++;
      PetscCall(PetscPrintf(comm, "SSR_ATTEMPT step=%" PetscInt_FMT " accepted=false reductions=%" PetscInt_FMT " next_d_omega=%.8e\n", step, omega_reductions, (double)d_omega));
      if (omega_reductions >= 5) PetscCall(PetscStrncpy(stats->stop_reason, "omega_reduction_limit", sizeof(stats->stop_reason)));
      continue;
    }
    PetscCall(LinearSolverTruncateRawBasis(solver, basis_snapshot));

    branch_double = (PetscBool)((lambda_guess - lambda_cur) < 0.9 * (lambda_cur - lambda_prev));
    lambda_prev = lambda_cur;
    omega_prev  = omega_cur;
    PetscCall(VecCopy(u_cur, u_prev));
    lambda_cur = lambda_guess;
    omega_cur  = target;
    PetscCall(VecCopy(u_guess, u_cur));
    PetscCall(LinearSolverAppendRawBasis(solver, u_cur, "SSR accepted continuation state"));
    omega_reductions = 0;
    PetscCall(FillCurveRow(&row, step, "cont", omega_cur, lambda_cur, omega_cur - omega_prev, lambda_cur - lambda_prev, u_cur, attempts, &nstats, "accepted"));
    PetscCall(CurveWriteRow(csv, &row));
    PetscCall(PetscPrintf(comm,
                          "SSR_STEP step=%" PetscInt_FMT " phase=cont omega=%.8e lambda=%.8e d_omega=%.8e d_lambda=%.8e accepted=true newton_its=%" PetscInt_FMT " linear_its=%" PetscInt_FMT " rel_res=%.6e\n",
                          step, (double)omega_cur, (double)lambda_cur, (double)(omega_cur - omega_prev), (double)(lambda_cur - lambda_prev),
                          nstats.newton_its, nstats.total_linear_its, (double)nstats.final_rel));
    stats->accepted_steps++;
    stats->total_newton_its += nstats.newton_its;
    stats->total_linear_its += nstats.total_linear_its;
    stats->total_line_search_its += nstats.line_search_its;
    stats->final_rel            = nstats.final_rel;
    stats->final_rel_correction = nstats.final_rel_correction;
    step++;

    if (branch_double) d_omega *= 2.0;
    if (app->d_lambda_diff_scaled_min > 0.0) {
      const PetscReal slope_scaled = PetscAbsReal((lambda_cur - lambda_prev) / PetscMax(omega_cur - omega_prev, 1.0e-30)) * PetscMax(omega_cur, 1.0);
      if (slope_scaled <= app->d_lambda_diff_scaled_min) PetscCall(PetscStrncpy(stats->stop_reason, "d_lambda_diff_scaled_min", sizeof(stats->stop_reason)));
    }
    if (omega_cur >= app->omega_max) PetscCall(PetscStrncpy(stats->stop_reason, "omega_max", sizeof(stats->stop_reason)));
    (void)omega_old;
  }
  {
    PetscBool reason_running;

    PetscCall(PetscStrcmp(stats->stop_reason, "running", &reason_running));
    if (omega_cur >= app->omega_max && reason_running) PetscCall(PetscStrncpy(stats->stop_reason, "omega_max", sizeof(stats->stop_reason)));
    if (step >= app->continuation_step_max && reason_running) PetscCall(PetscStrncpy(stats->stop_reason, "step_max", sizeof(stats->stop_reason)));
  }
  stats->omega_last  = omega_cur;
  stats->lambda_last = lambda_cur;
  PetscCall(PetscTime(&t_end));
  stats->wall_time = t_end - t_start;
  PetscCall(PetscPrintf(comm,
                        "SSR_RESULT omega_last=%.8e lambda_last=%.8e accepted_steps=%" PetscInt_FMT " total_newton_iterations=%" PetscInt_FMT " total_linear_iterations=%" PetscInt_FMT " total_line_search_iterations=%" PetscInt_FMT " wall_time=%.6g stop_reason=%s curve_csv=%s\n",
                        (double)stats->omega_last, (double)stats->lambda_last, stats->accepted_steps, stats->total_newton_its,
                        stats->total_linear_its, stats->total_line_search_its, (double)stats->wall_time, stats->stop_reason, app->curve_csv));

  if (csv) fclose(csv);
  PetscCall(VecDestroy(&u_prev));
  PetscCall(VecDestroy(&u_cur));
  PetscCall(VecDestroy(&u_guess));
  PetscCall(VecDestroy(&u_tmp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  char      kind[64];
  PetscReal lambda;
  PetscReal omega;
  PetscReal r;
  PetscReal rel_residual;
  PetscInt  sample_id;
  PetscInt  newton_iteration;
  PetscInt  basis_cols;
  PetscInt  expected_dW_iterations;
  PetscInt  expected_dV_iterations;
  PetscReal expected_dW_reported_residual_final;
  PetscReal expected_dV_reported_residual_final;
  PetscInt  expected_iterations;
  PetscReal expected_reported_residual_final;
  PetscReal expected_true_residual_final;
  PetscReal expected_alpha;
  PetscInt  expected_line_search_iterations;
  PetscReal expected_initial_decrease;
  PetscReal expected_rel_correction;
  PetscBool expected_stop;
} LinearReplayMeta;

static PetscErrorCode LinearReplayReadMeta(MPI_Comm comm, const char dir[], LinearReplayMeta *meta)
{
  char path[PETSC_MAX_PATH_LEN];
  FILE *fh = NULL;
  int   rank;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(meta, sizeof(*meta)));
  PetscCall(PetscStrncpy(meta->kind, "indirect_linear", sizeof(meta->kind)));
  meta->r = 1.0e-4;
  meta->expected_dW_iterations = -1;
  meta->expected_dV_iterations = -1;
  meta->expected_dW_reported_residual_final = PETSC_MAX_REAL;
  meta->expected_dV_reported_residual_final = PETSC_MAX_REAL;
  meta->expected_iterations = -1;
  meta->expected_reported_residual_final = PETSC_MAX_REAL;
  meta->expected_true_residual_final = PETSC_MAX_REAL;
  meta->expected_alpha = PETSC_MAX_REAL;
  meta->expected_line_search_iterations = -1;
  meta->expected_initial_decrease = PETSC_MAX_REAL;
  meta->expected_rel_correction = PETSC_MAX_REAL;
  meta->expected_stop = PETSC_FALSE;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscSNPrintf(path, sizeof(path), "%s/meta.txt", dir));
  fh = fopen(path, "r");
  PetscCheck(fh, comm, PETSC_ERR_FILE_OPEN, "Failed to open replay metadata %s", path);
  while (!feof(fh)) {
    char key[128], value[256];
    if (fscanf(fh, "%127s %255s", key, value) != 2) break;
    if (!strcmp(key, "kind")) PetscCall(PetscStrncpy(meta->kind, value, sizeof(meta->kind)));
    else if (!strcmp(key, "sample_id")) meta->sample_id = (PetscInt)strtol(value, NULL, 10);
    else if (!strcmp(key, "omega")) meta->omega = (PetscReal)strtod(value, NULL);
    else if (!strcmp(key, "lambda")) meta->lambda = (PetscReal)strtod(value, NULL);
    else if (!strcmp(key, "r")) meta->r = (PetscReal)strtod(value, NULL);
    else if (!strcmp(key, "newton_iteration")) meta->newton_iteration = (PetscInt)strtol(value, NULL, 10);
    else if (!strcmp(key, "rel_residual")) meta->rel_residual = (PetscReal)strtod(value, NULL);
    else if (!strcmp(key, "basis_cols")) meta->basis_cols = (PetscInt)strtol(value, NULL, 10);
    else if (!strcmp(key, "expected_dW_iterations")) meta->expected_dW_iterations = (PetscInt)strtol(value, NULL, 10);
    else if (!strcmp(key, "expected_dV_iterations")) meta->expected_dV_iterations = (PetscInt)strtol(value, NULL, 10);
    else if (!strcmp(key, "expected_dW_reported_residual_final")) meta->expected_dW_reported_residual_final = (PetscReal)strtod(value, NULL);
    else if (!strcmp(key, "expected_dV_reported_residual_final")) meta->expected_dV_reported_residual_final = (PetscReal)strtod(value, NULL);
    else if (!strcmp(key, "expected_iterations")) meta->expected_iterations = (PetscInt)strtol(value, NULL, 10);
    else if (!strcmp(key, "expected_reported_residual_final")) meta->expected_reported_residual_final = (PetscReal)strtod(value, NULL);
    else if (!strcmp(key, "expected_true_residual_final")) meta->expected_true_residual_final = (PetscReal)strtod(value, NULL);
    else if (!strcmp(key, "expected_alpha")) meta->expected_alpha = (PetscReal)strtod(value, NULL);
    else if (!strcmp(key, "expected_line_search_iterations")) meta->expected_line_search_iterations = (PetscInt)strtol(value, NULL, 10);
    else if (!strcmp(key, "expected_initial_decrease")) meta->expected_initial_decrease = (PetscReal)strtod(value, NULL);
    else if (!strcmp(key, "expected_rel_correction")) meta->expected_rel_correction = (PetscReal)strtod(value, NULL);
    else if (!strcmp(key, "expected_stop")) meta->expected_stop = (PetscBool)(!strcmp(value, "true"));
  }
  fclose(fh);
  PetscCall(PetscPrintf(comm,
                        "REPLAY_META dir=%s kind=%s sample_id=%" PetscInt_FMT " omega=%.8e lambda=%.8e r=%.8e newton_iteration=%" PetscInt_FMT " basis_cols=%" PetscInt_FMT " expected_dW=%" PetscInt_FMT " expected_dV=%" PetscInt_FMT " expected_iterations=%" PetscInt_FMT "\n",
                        dir, meta->kind, meta->sample_id, (double)meta->omega, (double)meta->lambda, (double)meta->r, meta->newton_iteration,
                        meta->basis_cols, meta->expected_dW_iterations, meta->expected_dV_iterations, meta->expected_iterations));
  (void)rank;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayLoadVec(MPI_Comm comm, const char dir[], const char name[], Vec template_vec, Vec *v)
{
  char        path[PETSC_MAX_PATH_LEN];
  PetscViewer viewer = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscSNPrintf(path, sizeof(path), "%s/%s", dir, name));
  PetscCall(VecDuplicate(template_vec, v));
  PetscCall(PetscViewerBinaryOpen(comm, path, FILE_MODE_READ, &viewer));
  PetscCall(VecLoad(*v, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  long long ix, iy, iz;
  PetscReal x, y, z;
  PetscInt  comp;
  PetscInt  export_index;
} LinearReplayDofKey;

typedef struct {
  long long ix, iy, iz;
} LinearReplayCoordKey3;

typedef struct {
  LinearReplayCoordKey3 sorted_vertices[4];
  LinearReplayCoordKey3 ordered_vertices[4];
  LinearReplayCoordKey3 sorted_nodes[35];
  PetscReal             ordered_vertex_xyz[12];
  PetscReal             node_xyz[105];
  PetscInt              nodes[35];
} LinearReplayCellTopology;

static long long LinearReplayCoordKey(PetscReal x)
{
  return (long long)llround((double)x * 1.0e7);
}

static LinearReplayCoordKey3 LinearReplayMakeCoordKey3(PetscReal x, PetscReal y, PetscReal z)
{
  LinearReplayCoordKey3 key;

  key.ix = LinearReplayCoordKey(x);
  key.iy = LinearReplayCoordKey(y);
  key.iz = LinearReplayCoordKey(z);
  return key;
}

static int LinearReplayCoordKey3Compare(const void *a, const void *b)
{
  const LinearReplayCoordKey3 *ka = (const LinearReplayCoordKey3 *)a;
  const LinearReplayCoordKey3 *kb = (const LinearReplayCoordKey3 *)b;

  if (ka->ix < kb->ix) return -1;
  if (ka->ix > kb->ix) return 1;
  if (ka->iy < kb->iy) return -1;
  if (ka->iy > kb->iy) return 1;
  if (ka->iz < kb->iz) return -1;
  if (ka->iz > kb->iz) return 1;
  return 0;
}

static PetscBool LinearReplayCoordKey3Equal(LinearReplayCoordKey3 a, LinearReplayCoordKey3 b)
{
  return (PetscBool)(a.ix == b.ix && a.iy == b.iy && a.iz == b.iz);
}

static void LinearReplaySort4CoordKeys(LinearReplayCoordKey3 keys[4])
{
  qsort(keys, 4, sizeof(keys[0]), LinearReplayCoordKey3Compare);
}

static int LinearReplayCellTopologyCompare(const void *a, const void *b)
{
  const LinearReplayCellTopology *ca = (const LinearReplayCellTopology *)a;
  const LinearReplayCellTopology *cb = (const LinearReplayCellTopology *)b;

  for (PetscInt i = 0; i < 4; ++i) {
    int cmp = LinearReplayCoordKey3Compare(&ca->sorted_vertices[i], &cb->sorted_vertices[i]);
    if (cmp) return cmp;
  }
  return 0;
}

static int LinearReplayDofKeyCompare(const void *a, const void *b)
{
  const LinearReplayDofKey *ka = (const LinearReplayDofKey *)a;
  const LinearReplayDofKey *kb = (const LinearReplayDofKey *)b;

  if (ka->comp < kb->comp) return -1;
  if (ka->comp > kb->comp) return 1;
  if (ka->ix < kb->ix) return -1;
  if (ka->ix > kb->ix) return 1;
  if (ka->iy < kb->iy) return -1;
  if (ka->iy > kb->iy) return 1;
  if (ka->iz < kb->iz) return -1;
  if (ka->iz > kb->iz) return 1;
  if (ka->export_index < kb->export_index) return -1;
  if (ka->export_index > kb->export_index) return 1;
  return 0;
}

static int LinearReplayDofKeyComparePosition(const LinearReplayDofKey *ka, const LinearReplayDofKey *kb)
{
  if (ka->comp < kb->comp) return -1;
  if (ka->comp > kb->comp) return 1;
  if (ka->ix < kb->ix) return -1;
  if (ka->ix > kb->ix) return 1;
  if (ka->iy < kb->iy) return -1;
  if (ka->iy > kb->iy) return 1;
  if (ka->iz < kb->iz) return -1;
  if (ka->iz > kb->iz) return 1;
  return 0;
}

static PetscErrorCode LinearReplayFindDofKey(const LinearReplayDofKey keys[], PetscInt nkeys, const LinearReplayDofKey *needle, PetscInt *export_index, PetscInt *match_pos)
{
  PetscInt lo = 0, hi = nkeys;

  PetscFunctionBeginUser;
  *export_index = -1;
  if (match_pos) *match_pos = -1;
  while (lo < hi) {
    PetscInt mid = lo + (hi - lo) / 2;
    int      cmp = LinearReplayDofKeyComparePosition(needle, &keys[mid]);

    if (cmp == 0) {
      *export_index = keys[mid].export_index;
      if (match_pos) *match_pos = mid;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (cmp < 0) hi = mid;
    else lo = mid + 1;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscInt LinearReplayP4LocalIndexFromCounts(const PetscInt counts[4])
{
  const PetscInt edge_order[6][2] = {{0, 1}, {1, 2}, {0, 2}, {1, 3}, {2, 3}, {0, 3}};
  const PetscInt face_order[4][3] = {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};
  const PetscInt tri_counts[3][3] = {{2, 1, 1}, {1, 2, 1}, {1, 1, 2}};
  PetscInt       idx = 0;

  for (PetscInt v = 0; v < 4; ++v, ++idx) {
    PetscBool match = PETSC_TRUE;
    for (PetscInt i = 0; i < 4; ++i) match = (PetscBool)(match && counts[i] == (i == v ? 4 : 0));
    if (match) return idx;
  }
  for (PetscInt e = 0; e < 6; ++e) {
    for (PetscInt k = 1; k < 4; ++k, ++idx) {
      PetscInt c[4] = {0, 0, 0, 0};
      c[edge_order[e][0]] = 4 - k;
      c[edge_order[e][1]] = k;
      if (c[0] == counts[0] && c[1] == counts[1] && c[2] == counts[2] && c[3] == counts[3]) return idx;
    }
  }
  for (PetscInt f = 0; f < 4; ++f) {
    for (PetscInt t = 0; t < 3; ++t, ++idx) {
      PetscInt c[4] = {0, 0, 0, 0};
      for (PetscInt j = 0; j < 3; ++j) c[face_order[f][j]] = tri_counts[t][j];
      if (c[0] == counts[0] && c[1] == counts[1] && c[2] == counts[2] && c[3] == counts[3]) return idx;
    }
  }
  if (counts[0] == 1 && counts[1] == 1 && counts[2] == 1 && counts[3] == 1) return idx;
  return -1;
}

static PetscErrorCode LinearReplayBuildPetscBasisCounts(P4Basis *basis, PetscInt counts[35][4])
{
  PetscDualSpace dual;

  PetscFunctionBeginUser;
  PetscCheck(basis->degree == 4 && basis->n_basis == 35, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Replay topology mapper currently requires P4 basis with 35 nodes");
  PetscCall(PetscFEGetDualSpace(basis->fe_scalar, &dual));
  for (PetscInt b = 0; b < basis->n_basis; ++b) {
    PetscQuadrature  q;
    PetscInt         dim, Nc, npoints;
    const PetscReal *r;
    PetscReal        xi0, xi1, xi2;

    PetscCall(PetscDualSpaceGetFunctional(dual, b, &q));
    PetscCall(PetscQuadratureGetData(q, &dim, &Nc, &npoints, &r, NULL));
    PetscCheck(dim == 3 && Nc == 1 && npoints >= 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected scalar dual functional shape");
    xi0 = 0.5 * (r[0] + 1.0);
    xi1 = 0.5 * (r[1] + 1.0);
    xi2 = 0.5 * (r[2] + 1.0);
    counts[b][0] = 4 - (PetscInt)PetscFloorReal(4.0 * xi0 + 0.5) - (PetscInt)PetscFloorReal(4.0 * xi1 + 0.5) - (PetscInt)PetscFloorReal(4.0 * xi2 + 0.5);
    counts[b][2] = (PetscInt)PetscFloorReal(4.0 * xi0 + 0.5);
    counts[b][1] = (PetscInt)PetscFloorReal(4.0 * xi1 + 0.5);
    counts[b][3] = (PetscInt)PetscFloorReal(4.0 * xi2 + 0.5);
    PetscCheck(counts[b][0] >= 0 && counts[b][1] >= 0 && counts[b][2] >= 0 && counts[b][3] >= 0 &&
                 counts[b][0] + counts[b][1] + counts[b][2] + counts[b][3] == 4,
               PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid P4 replay basis counts for local basis %" PetscInt_FMT, b);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayFindCellTopology(const LinearReplayCellTopology cells[], PetscInt ncells, const LinearReplayCoordKey3 sorted_vertices[4], PetscInt *idx)
{
  LinearReplayCellTopology needle;
  PetscInt                 lo = 0, hi = ncells;

  PetscFunctionBeginUser;
  *idx = -1;
  PetscCall(PetscMemzero(&needle, sizeof(needle)));
  for (PetscInt i = 0; i < 4; ++i) needle.sorted_vertices[i] = sorted_vertices[i];
  while (lo < hi) {
    PetscInt mid = lo + (hi - lo) / 2;
    int      cmp = LinearReplayCellTopologyCompare(&needle, &cells[mid]);

    if (cmp == 0) {
      *idx = mid;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (cmp < 0) hi = mid;
    else lo = mid + 1;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayLoadCellTopology(MPI_Comm comm, const char dir[], LinearReplayCellTopology **cells_out, PetscInt *ncells_out)
{
  char                      path[PETSC_MAX_PATH_LEN], line[4096];
  FILE                     *fh = NULL;
  LinearReplayCellTopology *cells = NULL;
  PetscInt                  ncells = 0, cap = 0;

  PetscFunctionBeginUser;
  *cells_out = NULL;
  *ncells_out = 0;
  PetscCall(PetscSNPrintf(path, sizeof(path), "%s/cell_topology.csv", dir));
  fh = fopen(path, "r");
  if (!fh) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(fgets(line, sizeof(line), fh), comm, PETSC_ERR_FILE_UNEXPECTED, "Empty replay topology map %s", path);
  while (fgets(line, sizeof(line), fh)) {
    char  *tok  = strtok(line, ",\n\r");
    double xyz[12], pxyz[105];

    PetscCheck(tok, comm, PETSC_ERR_FILE_UNEXPECTED, "Malformed replay topology row in %s", path);
    if (ncells == cap) {
      cap = cap ? 2 * cap : 32768;
      PetscCall(PetscRealloc((size_t)cap * sizeof(*cells), &cells));
    }
    for (PetscInt i = 0; i < 12; ++i) {
      tok = strtok(NULL, ",\n\r");
      PetscCheck(tok, comm, PETSC_ERR_FILE_UNEXPECTED, "Malformed vertex columns in replay topology row in %s", path);
      xyz[i] = strtod(tok, NULL);
    }
    for (PetscInt v = 0; v < 4; ++v) {
      cells[ncells].ordered_vertices[v] = LinearReplayMakeCoordKey3((PetscReal)xyz[3 * v + 0], (PetscReal)xyz[3 * v + 1], (PetscReal)xyz[3 * v + 2]);
      cells[ncells].sorted_vertices[v]  = cells[ncells].ordered_vertices[v];
      for (PetscInt d = 0; d < 3; ++d) cells[ncells].ordered_vertex_xyz[3 * v + d] = (PetscReal)xyz[3 * v + d];
    }
    LinearReplaySort4CoordKeys(cells[ncells].sorted_vertices);
    for (PetscInt i = 0; i < 105; ++i) {
      tok = strtok(NULL, ",\n\r");
      PetscCheck(tok, comm, PETSC_ERR_FILE_UNEXPECTED, "Malformed point columns in replay topology row in %s", path);
      pxyz[i] = strtod(tok, NULL);
    }
    for (PetscInt p = 0; p < 35; ++p) {
      cells[ncells].sorted_nodes[p] = LinearReplayMakeCoordKey3((PetscReal)pxyz[3 * p + 0], (PetscReal)pxyz[3 * p + 1], (PetscReal)pxyz[3 * p + 2]);
      for (PetscInt d = 0; d < 3; ++d) cells[ncells].node_xyz[3 * p + d] = (PetscReal)pxyz[3 * p + d];
    }
    qsort(cells[ncells].sorted_nodes, 35, sizeof(cells[ncells].sorted_nodes[0]), LinearReplayCoordKey3Compare);
    for (PetscInt i = 0; i < 35; ++i) {
      tok = strtok(NULL, ",\n\r");
      PetscCheck(tok, comm, PETSC_ERR_FILE_UNEXPECTED, "Malformed node columns in replay topology row in %s", path);
      cells[ncells].nodes[i] = (PetscInt)strtol(tok, NULL, 10);
    }
    ++ncells;
  }
  fclose(fh);
  qsort(cells, (size_t)ncells, sizeof(*cells), LinearReplayCellTopologyCompare);
  *cells_out  = cells;
  *ncells_out = ncells;
  PetscCall(PetscPrintf(comm, "REPLAY_TOPOLOGY_MAP file=%s status=loaded cells=%" PetscInt_FMT "\n", path, ncells));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayTryBuildTopologyMap(DM dm, P4Basis *basis, const char dir[], Vec template_vec, const PetscInt full_to_export[], PetscInt full_to_export_size, PetscInt c_to_export[], PetscBool *complete)
{
  MPI_Comm                  comm = PetscObjectComm((PetscObject)dm);
  LinearReplayCellTopology *cells = NULL;
  PetscInt                  ncells = 0, basis_counts[35][4], lo, hi, cStart, cEnd, missing = 0, missing_global = 0, cell_misses = 0, cell_misses_global = 0;
  PetscInt                  coord_checked = 0, coord_checked_global = 0, local_index_disagreements = 0, local_index_disagreements_global = 0;
  PetscReal                 max_node_coord_error = 0.0, max_node_coord_error_global = 0.0;
  PetscInt                  short_coord_cells = 0, short_coord_cells_global = 0, first_ncoords = -1, printed_first_cell_miss = 0;
  PetscSection              lsec, gsec;
  Vec                       coordinates = NULL;

  PetscFunctionBeginUser;
  *complete = PETSC_FALSE;
  PetscCall(LinearReplayLoadCellTopology(comm, dir, &cells, &ncells));
  if (!cells || !ncells) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(LinearReplayBuildPetscBasisCounts(basis, basis_counts));
  PetscCall(VecGetOwnershipRange(template_vec, &lo, &hi));
  for (PetscInt i = 0; i < hi - lo; ++i) c_to_export[i] = -1;
  PetscCall(DMGetLocalSection(dm, &lsec));
  PetscCall(DMGetGlobalSection(dm, &gsec));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
    PetscInt              ncoords = 0, topo_idx = -1, vertex_to_py[4], num_indices = 0, *indices = NULL;
    LinearReplayCoordKey3 vertex_keys[4], sorted_vertices[4];
    const LinearReplayCellTopology *topo;
    PetscReal             v0[3], J[9], invJ[9], detJ;

    PetscCall(DMPlexComputeCellGeometryFEM(dm, cell, NULL, v0, J, invJ, &detJ));
    for (PetscInt cv = 0; cv < 4; ++cv) {
      PetscReal x[3] = {v0[0], v0[1], v0[2]};

      if (cv == 1) {
        for (PetscInt d = 0; d < 3; ++d) x[d] += 2.0 * J[d * 3 + 1];
      } else if (cv == 2) {
        for (PetscInt d = 0; d < 3; ++d) x[d] += 2.0 * J[d * 3 + 0];
      } else if (cv == 3) {
        for (PetscInt d = 0; d < 3; ++d) x[d] += 2.0 * J[d * 3 + 2];
      }
      vertex_keys[cv] = LinearReplayMakeCoordKey3(x[0], x[1], x[2]);
      sorted_vertices[cv] = vertex_keys[cv];
    }
    LinearReplaySort4CoordKeys(sorted_vertices);
    PetscCall(LinearReplayFindCellTopology(cells, ncells, sorted_vertices, &topo_idx));
    if (topo_idx < 0) {
      if (!printed_first_cell_miss) {
        printed_first_cell_miss = 1;
        PetscCall(PetscPrintf(comm,
                              "REPLAY_TOPOLOGY_CELL_MISS c0=(%lld,%lld,%lld) c1=(%lld,%lld,%lld) c2=(%lld,%lld,%lld) c3=(%lld,%lld,%lld) t0=(%lld,%lld,%lld) t1=(%lld,%lld,%lld) t2=(%lld,%lld,%lld) t3=(%lld,%lld,%lld)\n",
                              sorted_vertices[0].ix, sorted_vertices[0].iy, sorted_vertices[0].iz, sorted_vertices[1].ix,
                              sorted_vertices[1].iy, sorted_vertices[1].iz, sorted_vertices[2].ix, sorted_vertices[2].iy, sorted_vertices[2].iz,
                              sorted_vertices[3].ix, sorted_vertices[3].iy, sorted_vertices[3].iz,
                              cells[0].sorted_vertices[0].ix, cells[0].sorted_vertices[0].iy, cells[0].sorted_vertices[0].iz,
                              cells[0].sorted_vertices[1].ix, cells[0].sorted_vertices[1].iy, cells[0].sorted_vertices[1].iz,
                              cells[0].sorted_vertices[2].ix, cells[0].sorted_vertices[2].iy, cells[0].sorted_vertices[2].iz,
                              cells[0].sorted_vertices[3].ix, cells[0].sorted_vertices[3].iy, cells[0].sorted_vertices[3].iz));
      }
      ++cell_misses;
      continue;
    }
    topo = &cells[topo_idx];
    for (PetscInt cv = 0; cv < 4; ++cv) {
      vertex_to_py[cv] = -1;
      for (PetscInt pv = 0; pv < 4; ++pv) {
        if (LinearReplayCoordKey3Equal(vertex_keys[cv], topo->ordered_vertices[pv])) {
          vertex_to_py[cv] = pv;
          break;
        }
      }
      PetscCheck(vertex_to_py[cv] >= 0, comm, PETSC_ERR_PLIB, "Could not orient replay topology cell");
    }
    PetscCall(DMPlexGetClosureIndices(dm, lsec, gsec, cell, PETSC_TRUE, &num_indices, &indices, NULL, NULL));
    PetscCheck(num_indices == 3 * basis->n_basis, comm, PETSC_ERR_PLIB, "Unexpected replay closure size %" PetscInt_FMT, num_indices);
    for (PetscInt b = 0; b < basis->n_basis; ++b) {
      PetscInt counts_py[4] = {0, 0, 0, 0};
      PetscInt local_py_counts, local_py = -1, node;
      PetscReal x_py[3] = {0.0, 0.0, 0.0}, best_err = PETSC_MAX_REAL;

      for (PetscInt cv = 0; cv < 4; ++cv) counts_py[vertex_to_py[cv]] = basis_counts[b][cv];
      local_py_counts = LinearReplayP4LocalIndexFromCounts(counts_py);
      PetscCheck(local_py_counts >= 0 && local_py_counts < 35, comm, PETSC_ERR_PLIB, "Could not map replay P4 local basis counts");
      for (PetscInt pv = 0; pv < 4; ++pv) {
        const PetscReal w = ((PetscReal)counts_py[pv]) / 4.0;
        for (PetscInt d = 0; d < 3; ++d) x_py[d] += w * topo->ordered_vertex_xyz[3 * pv + d];
      }
      for (PetscInt p = 0; p < 35; ++p) {
        PetscReal err = 0.0;
        for (PetscInt d = 0; d < 3; ++d) err = PetscMax(err, PetscAbsReal(x_py[d] - topo->node_xyz[3 * p + d]));
        if (err < best_err) {
          best_err = err;
          local_py = p;
        }
      }
      PetscCheck(local_py >= 0 && local_py < 35 && best_err <= 1.0e-7, comm, PETSC_ERR_PLIB,
                 "Could not coordinate-match replay P4 local node for basis %" PetscInt_FMT " best_err=%.8e", b, (double)best_err);
      ++coord_checked;
      max_node_coord_error = PetscMax(max_node_coord_error, best_err);
      if (local_py != local_py_counts) ++local_index_disagreements;
      node = topo->nodes[local_py];
      for (PetscInt comp = 0; comp < 3; ++comp) {
        const PetscInt row = indices[3 * b + comp];
        const PetscInt full = 3 * node + comp;
        PetscInt       export_index = -1;

        if (row < lo || row >= hi) continue;
        if (full >= 0 && full < full_to_export_size) export_index = full_to_export[full];
        PetscCheck(export_index >= 0, comm, PETSC_ERR_PLIB, "Replay topology mapped to constrained/missing Python DOF full=%" PetscInt_FMT, full);
        if (c_to_export[row - lo] >= 0 && c_to_export[row - lo] != export_index) {
          SETERRQ(comm, PETSC_ERR_PLIB, "Conflicting replay topology map for local row %" PetscInt_FMT, row - lo);
        }
        c_to_export[row - lo] = export_index;
      }
    }
    PetscCall(DMPlexRestoreClosureIndices(dm, lsec, gsec, cell, PETSC_TRUE, &num_indices, &indices, NULL, NULL));
  }
  for (PetscInt i = 0; i < hi - lo; ++i) {
    if (c_to_export[i] < 0) ++missing;
  }
  PetscCallMPI(MPI_Allreduce(&missing, &missing_global, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&cell_misses, &cell_misses_global, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&short_coord_cells, &short_coord_cells_global, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&coord_checked, &coord_checked_global, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&local_index_disagreements, &local_index_disagreements_global, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&max_node_coord_error, &max_node_coord_error_global, 1, MPIU_REAL, MPI_MAX, comm));
  PetscCall(PetscPrintf(comm,
                        "REPLAY_TOPOLOGY_RESULT local_rows=%" PetscInt_FMT " missing_rows=%" PetscInt_FMT " missing_cells=%" PetscInt_FMT " short_coord_cells=%" PetscInt_FMT " first_short_ncoords=%" PetscInt_FMT " coordinate_checked=%" PetscInt_FMT " local_index_disagreements=%" PetscInt_FMT " max_node_coord_error=%.8e\n",
                        hi - lo, missing_global, cell_misses_global, short_coord_cells_global, first_ncoords, coord_checked_global,
                        local_index_disagreements_global, (double)max_node_coord_error_global));
  *complete = (PetscBool)(missing_global == 0);
  PetscCall(PetscFree(cells));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayBuildCToExportMap(DM dm, P4Basis *basis, const char dir[], Vec template_vec, PetscInt **c_to_export, PetscBool *mapped)
{
  MPI_Comm            comm = PetscObjectComm((PetscObject)dm);
  char                path[PETSC_MAX_PATH_LEN], line[512];
  FILE               *fh = NULL;
  LinearReplayDofKey *keys = NULL;
  PetscInt           *full_to_export = NULL;
  PetscInt            full_cap = 0, nkeys = 0, cap = 0, nloc, nowned, missing = 0, missing_global = 0, duplicates = 0, duplicates_global = 0;
  PetscReal           first_missing_x = 0.0, first_missing_y = 0.0, first_missing_z = 0.0;
  PetscInt            first_missing_comp = -1;
  PetscReal           max_coord_error = 0.0, max_coord_error_global = 0.0;
  PetscReal          *coords = NULL;
  PetscInt           *comps = NULL;

  PetscFunctionBeginUser;
  *c_to_export = NULL;
  *mapped      = PETSC_FALSE;
  PetscCall(PetscSNPrintf(path, sizeof(path), "%s/free_dof_map.csv", dir));
  fh = fopen(path, "r");
  if (!fh) {
    PetscCall(PetscPrintf(comm, "REPLAY_DOF_MAP file=%s status=absent mode=identity\n", path));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (!fgets(line, sizeof(line), fh)) {
    fclose(fh);
    SETERRQ(comm, PETSC_ERR_FILE_UNEXPECTED, "Empty replay DOF map %s", path);
  }
  while (fgets(line, sizeof(line), fh)) {
    long long export_index, full_index, node, comp;
    double    x, y, z;
    int       got = sscanf(line, "%lld,%lld,%lld,%lld,%lf,%lf,%lf", &export_index, &full_index, &node, &comp, &x, &y, &z);

    PetscCheck(got == 7, comm, PETSC_ERR_FILE_UNEXPECTED, "Malformed replay DOF map row in %s: %s", path, line);
    if (nkeys == cap) {
      cap = cap ? 2 * cap : 65536;
      PetscCall(PetscRealloc((size_t)cap * sizeof(*keys), &keys));
    }
    keys[nkeys].export_index = (PetscInt)export_index;
    keys[nkeys].comp         = (PetscInt)comp;
    keys[nkeys].x            = (PetscReal)x;
    keys[nkeys].y            = (PetscReal)y;
    keys[nkeys].z            = (PetscReal)z;
    keys[nkeys].ix           = LinearReplayCoordKey((PetscReal)x);
    keys[nkeys].iy           = LinearReplayCoordKey((PetscReal)y);
    keys[nkeys].iz           = LinearReplayCoordKey((PetscReal)z);
    if ((PetscInt)full_index >= full_cap) {
      const PetscInt old_cap = full_cap;
      full_cap = PetscMax((PetscInt)full_index + 1, full_cap ? 2 * full_cap : 1024);
      PetscCall(PetscRealloc((size_t)full_cap * sizeof(*full_to_export), &full_to_export));
      for (PetscInt i = old_cap; i < full_cap; ++i) full_to_export[i] = -1;
    }
    full_to_export[(PetscInt)full_index] = (PetscInt)export_index;
    ++nkeys;
    (void)full_index;
    (void)node;
  }
  fclose(fh);
  qsort(keys, (size_t)nkeys, sizeof(*keys), LinearReplayDofKeyCompare);
  for (PetscInt i = 1; i < nkeys; ++i) {
    if (LinearReplayDofKeyComparePosition(&keys[i - 1], &keys[i]) == 0) ++duplicates;
  }
  PetscCallMPI(MPI_Allreduce(&duplicates, &duplicates_global, 1, MPIU_INT, MPI_MAX, comm));

  PetscCall(VecGetLocalSize(template_vec, &nloc));
  PetscCall(PetscMalloc1(nloc, c_to_export));
  PetscCall(BuildOwnedDofCoordinatesComponents(dm, basis, &nowned, &coords, &comps));
  PetscCheck(nowned == nloc, comm, PETSC_ERR_PLIB, "Replay DOF coordinate count %" PetscInt_FMT " != local vector size %" PetscInt_FMT, nowned, nloc);
  for (PetscInt i = 0; i < nloc; ++i) {
    LinearReplayDofKey key;
    PetscInt           match_pos = -1;

    key.export_index = -1;
    key.comp         = comps[i];
    key.x            = coords[3 * i + 0];
    key.y            = coords[3 * i + 1];
    key.z            = coords[3 * i + 2];
    key.ix           = LinearReplayCoordKey(coords[3 * i + 0]);
    key.iy           = LinearReplayCoordKey(coords[3 * i + 1]);
    key.iz           = LinearReplayCoordKey(coords[3 * i + 2]);
    PetscCall(LinearReplayFindDofKey(keys, nkeys, &key, &(*c_to_export)[i], &match_pos));
    if ((*c_to_export)[i] < 0) {
      if (!missing) {
        first_missing_x    = coords[3 * i + 0];
        first_missing_y    = coords[3 * i + 1];
        first_missing_z    = coords[3 * i + 2];
        first_missing_comp = comps[i];
      }
      ++missing;
    } else if (match_pos >= 0) {
      const PetscReal dx = PetscAbsReal(key.x - keys[match_pos].x);
      const PetscReal dy = PetscAbsReal(key.y - keys[match_pos].y);
      const PetscReal dz = PetscAbsReal(key.z - keys[match_pos].z);
      max_coord_error = PetscMax(max_coord_error, PetscMax(dx, PetscMax(dy, dz)));
    }
  }
  PetscCallMPI(MPI_Allreduce(&missing, &missing_global, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&max_coord_error, &max_coord_error_global, 1, MPIU_REAL, MPI_MAX, comm));
  if (missing) {
    PetscCall(PetscPrintf(comm,
                          "REPLAY_DOF_MAP_MISS local_missing=%" PetscInt_FMT " first_comp=%" PetscInt_FMT " first_xyz=(%.17e,%.17e,%.17e) first_key=(%lld,%lld,%lld) first_export=(comp=%" PetscInt_FMT ",key=%lld,%lld,%lld,index=%" PetscInt_FMT ")\n",
                          missing, first_missing_comp, (double)first_missing_x, (double)first_missing_y, (double)first_missing_z,
                          LinearReplayCoordKey(first_missing_x), LinearReplayCoordKey(first_missing_y), LinearReplayCoordKey(first_missing_z),
                          nkeys ? keys[0].comp : -1, nkeys ? keys[0].ix : 0, nkeys ? keys[0].iy : 0, nkeys ? keys[0].iz : 0,
                          nkeys ? keys[0].export_index : -1));
  }
  PetscCall(PetscPrintf(comm,
                        "REPLAY_MAP_CHECK missing=%" PetscInt_FMT " duplicate=%" PetscInt_FMT " max_coord_error=%.8e local_rows=%" PetscInt_FMT " entries=%" PetscInt_FMT " scale=1e7\n",
                        missing_global, duplicates_global, (double)max_coord_error_global, nloc, nkeys));
  if (!missing_global && !duplicates_global) {
    *mapped = PETSC_TRUE;
    PetscCall(PetscPrintf(comm, "REPLAY_DOF_MAP file=%s status=loaded entries=%" PetscInt_FMT " local_rows=%" PetscInt_FMT " mode=coordinate_component scale=1e7\n",
                          path, nkeys, nloc));
    PetscCall(PetscFree(coords));
    PetscCall(PetscFree(comps));
    PetscCall(PetscFree(full_to_export));
    PetscCall(PetscFree(keys));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  {
    PetscBool topology_complete = PETSC_FALSE;

    PetscCall(LinearReplayTryBuildTopologyMap(dm, basis, dir, template_vec, full_to_export, full_cap, *c_to_export, &topology_complete));
    if (topology_complete) {
      *mapped = PETSC_TRUE;
      PetscCall(PetscPrintf(comm, "REPLAY_DOF_MAP file=%s status=loaded entries=%" PetscInt_FMT " local_rows=%" PetscInt_FMT " mode=cell_topology_fallback\n",
                            path, nkeys, nloc));
      PetscCall(PetscFree(coords));
      PetscCall(PetscFree(comps));
      PetscCall(PetscFree(full_to_export));
      PetscCall(PetscFree(keys));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscCheck(!missing_global, comm, PETSC_ERR_ARG_INCOMP, "Replay coordinate DOF map missed %" PetscInt_FMT " C-owned rows", missing_global);
  PetscCheck(!duplicates_global, comm, PETSC_ERR_ARG_INCOMP, "Replay coordinate DOF map has %" PetscInt_FMT " duplicate coordinate/component entries", duplicates_global);
  PetscCall(PetscFree(coords));
  PetscCall(PetscFree(comps));
  PetscCall(PetscFree(full_to_export));
  PetscCall(PetscFree(keys));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayLoadMappedVec(MPI_Comm comm, const char dir[], const char name[], Vec template_vec, const PetscInt c_to_export[], PetscBool mapped, Vec *v)
{
  Vec raw = NULL;

  PetscFunctionBeginUser;
  PetscCall(LinearReplayLoadVec(comm, dir, name, template_vec, &raw));
  if (!mapped) {
    *v = raw;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  {
    VecScatter     scatter = NULL;
    Vec            seq = NULL;
    const PetscScalar *seq_array = NULL;
    PetscScalar       *array = NULL;
    PetscInt           nloc;

    PetscCall(VecScatterCreateToAll(raw, &scatter, &seq));
    PetscCall(VecScatterBegin(scatter, raw, seq, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scatter, raw, seq, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecDuplicate(template_vec, v));
    PetscCall(VecGetLocalSize(*v, &nloc));
    PetscCall(VecGetArrayRead(seq, &seq_array));
    PetscCall(VecGetArray(*v, &array));
    for (PetscInt i = 0; i < nloc; ++i) array[i] = seq_array[c_to_export[i]];
    PetscCall(VecRestoreArray(*v, &array));
    PetscCall(VecRestoreArrayRead(seq, &seq_array));
    PetscCall(VecDestroy(&seq));
    PetscCall(VecScatterDestroy(&scatter));
    PetscCall(VecDestroy(&raw));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayLoadMappedVecRaw(MPI_Comm comm, const char dir[], const char name[], Vec template_vec, const PetscInt c_to_export[], PetscBool mapped, Vec *v)
{
  char              path[PETSC_MAX_PATH_LEN];
  FILE             *fh = NULL;
  long long         nll = 0;
  PetscInt          n = 0, nloc, lo = 0, hi = 0;
  PetscScalar      *values = NULL;
  PetscScalar      *array = NULL;
  PetscMPIInt       rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscSNPrintf(path, sizeof(path), "%s/%s", dir, name));
  if (rank == 0) {
    fh = fopen(path, "rb");
    PetscCheck(fh, comm, PETSC_ERR_FILE_OPEN, "Could not open raw replay vector %s", path);
    PetscCheck(fread(&nll, sizeof(nll), 1, fh) == 1, comm, PETSC_ERR_FILE_READ, "Could not read raw replay vector length from %s", path);
  }
  PetscCallMPI(MPI_Bcast(&nll, 1, MPI_LONG_LONG, 0, comm));
  PetscCheck(nll >= 0, comm, PETSC_ERR_FILE_UNEXPECTED, "Invalid raw replay vector length %lld in %s", nll, path);
  n = (PetscInt)nll;
  PetscCall(PetscMalloc1(n, &values));
  if (rank == 0) {
    PetscCheck(fread(values, sizeof(PetscScalar), (size_t)n, fh) == (size_t)n, comm, PETSC_ERR_FILE_READ, "Could not read raw replay vector payload from %s", path);
    fclose(fh);
  }
  PetscCallMPI(MPI_Bcast(values, n, MPIU_SCALAR, 0, comm));
  PetscCall(VecDuplicate(template_vec, v));
  PetscCall(VecGetLocalSize(*v, &nloc));
  PetscCall(VecGetOwnershipRange(*v, &lo, &hi));
  PetscCall(VecGetArray(*v, &array));
  if (mapped) {
    for (PetscInt i = 0; i < nloc; ++i) {
      PetscCheck(c_to_export[i] >= 0 && c_to_export[i] < n, comm, PETSC_ERR_ARG_OUTOFRANGE,
                 "Replay raw vector index %" PetscInt_FMT " outside length %" PetscInt_FMT, c_to_export[i], n);
      array[i] = values[c_to_export[i]];
    }
  } else {
    PetscCheck(hi <= n, comm, PETSC_ERR_ARG_SIZ, "Replay raw vector length %" PetscInt_FMT " shorter than local range end %" PetscInt_FMT, n, hi);
    for (PetscInt i = 0; i < nloc; ++i) array[i] = values[lo + i];
  }
  PetscCall(VecRestoreArray(*v, &array));
  PetscCall(PetscFree(values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayLoadMappedVecOptional(MPI_Comm comm, const char dir[], const char name[], Vec template_vec, const PetscInt c_to_export[], PetscBool mapped, Vec *v, PetscBool *present)
{
  char path[PETSC_MAX_PATH_LEN], raw_name[PETSC_MAX_PATH_LEN], *dot = NULL;
  FILE *fh = NULL;

  PetscFunctionBeginUser;
  *v       = NULL;
  *present = PETSC_FALSE;
  PetscCall(PetscSNPrintf(path, sizeof(path), "%s/%s", dir, name));
  fh = fopen(path, "rb");
  if (fh) {
    fclose(fh);
    *present = PETSC_TRUE;
    PetscCall(LinearReplayLoadMappedVec(comm, dir, name, template_vec, c_to_export, mapped, v));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscStrncpy(raw_name, name, sizeof(raw_name)));
  dot = strrchr(raw_name, '.');
  if (dot) PetscCall(PetscStrncpy(dot, ".raw", sizeof(raw_name) - (size_t)(dot - raw_name)));
  else PetscCall(PetscStrlcat(raw_name, ".raw", sizeof(raw_name)));
  PetscCall(PetscSNPrintf(path, sizeof(path), "%s/%s", dir, raw_name));
  fh = fopen(path, "rb");
  if (!fh) PetscFunctionReturn(PETSC_SUCCESS);
  fclose(fh);
  *present = PETSC_TRUE;
  PetscCall(LinearReplayLoadMappedVecRaw(comm, dir, raw_name, template_vec, c_to_export, mapped, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayMappedVecRoundtrip(MPI_Comm comm, const char dir[], const char name[], const char label[], Vec mapped_vec, Vec template_vec, const PetscInt c_to_export[], PetscBool mapped)
{
  Vec                  raw = NULL, back = NULL, diff = NULL;
  const PetscScalar   *mapped_array = NULL;
  PetscInt             nloc;
  PetscReal            norm_raw, norm_back, norm_diff;

  PetscFunctionBeginUser;
  if (!mapped) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(LinearReplayLoadVec(comm, dir, name, template_vec, &raw));
  PetscCall(VecDuplicate(raw, &back));
  PetscCall(VecZeroEntries(back));
  PetscCall(VecGetLocalSize(mapped_vec, &nloc));
  PetscCall(VecGetArrayRead(mapped_vec, &mapped_array));
  for (PetscInt i = 0; i < nloc; ++i) {
    PetscInt idx = c_to_export[i];
    PetscCall(VecSetValues(back, 1, &idx, &mapped_array[i], INSERT_VALUES));
  }
  PetscCall(VecRestoreArrayRead(mapped_vec, &mapped_array));
  PetscCall(VecAssemblyBegin(back));
  PetscCall(VecAssemblyEnd(back));
  PetscCall(VecDuplicate(raw, &diff));
  PetscCall(VecCopy(raw, diff));
  PetscCall(VecAXPY(diff, -1.0, back));
  PetscCall(VecNorm(raw, NORM_2, &norm_raw));
  PetscCall(VecNorm(back, NORM_2, &norm_back));
  PetscCall(VecNorm(diff, NORM_2, &norm_diff));
  PetscCall(PetscPrintf(comm,
                        "REPLAY_VEC_ROUNDTRIP label=%s norm_export=%.8e norm_roundtrip=%.8e diff=%.8e rel=%.8e\n",
                        label, (double)norm_raw, (double)norm_back, (double)norm_diff, (double)(norm_diff / PetscMax(norm_raw, 1.0e-300))));
  PetscCall(VecDestroy(&diff));
  PetscCall(VecDestroy(&back));
  PetscCall(VecDestroy(&raw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayLoadMappedMatNamed(MPI_Comm comm, const char dir[], const char filename[], Vec template_vec, const PetscInt c_to_export[], PetscBool mapped, Mat *A)
{
  char        path[PETSC_MAX_PATH_LEN];
  FILE       *fh = NULL;
  PetscViewer viewer = NULL;
  Mat         raw = NULL;
  IS          is = NULL;
  PetscInt    nloc;

  PetscFunctionBeginUser;
  *A = NULL;
  if (!mapped) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSNPrintf(path, sizeof(path), "%s/%s", dir, filename));
  fh = fopen(path, "rb");
  if (!fh) PetscFunctionReturn(PETSC_SUCCESS);
  fclose(fh);
  PetscCall(PetscViewerBinaryOpen(comm, path, FILE_MODE_READ, &viewer));
  PetscCall(MatCreate(comm, &raw));
  PetscCall(MatSetType(raw, MATAIJ));
  PetscCall(MatLoad(raw, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  /* The IS is local to the C replay layout: C row i pulls petsc4py free row c_to_export[i]. */
  PetscCall(VecGetLocalSize(template_vec, &nloc));
  PetscCall(ISCreateGeneral(comm, nloc, c_to_export, PETSC_COPY_VALUES, &is));
  PetscCall(MatCreateSubMatrix(raw, is, is, MAT_INITIAL_MATRIX, A));
  {
    PetscInt rawM, rawN, rawm, rawn, mapM, mapN, mapm, mapn;
    PetscCall(MatGetSize(raw, &rawM, &rawN));
    PetscCall(MatGetLocalSize(raw, &rawm, &rawn));
    PetscCall(MatGetSize(*A, &mapM, &mapN));
    PetscCall(MatGetLocalSize(*A, &mapm, &mapn));
    PetscCall(PetscPrintf(comm,
                          "REPLAY_MATRIX file=%s status=loaded_permuted raw_size=%" PetscInt_FMT "x%" PetscInt_FMT " raw_local=%" PetscInt_FMT "x%" PetscInt_FMT " mapped_size=%" PetscInt_FMT "x%" PetscInt_FMT " mapped_local=%" PetscInt_FMT "x%" PetscInt_FMT "\n",
                          path, rawM, rawN, rawm, rawn, mapM, mapN, mapm, mapn));
  }
  PetscCall(ISDestroy(&is));
  PetscCall(MatDestroy(&raw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayLoadMappedMat(MPI_Comm comm, const char dir[], Vec template_vec, const PetscInt c_to_export[], PetscBool mapped, Mat *A)
{
  PetscFunctionBeginUser;
  PetscCall(LinearReplayLoadMappedMatNamed(comm, dir, "A_free.mat", template_vec, c_to_export, mapped, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayMatResidual(MPI_Comm comm, const char label[], Mat A, Vec x, Vec rhs)
{
  Vec       r = NULL;
  PetscReal norm_rhs, norm_x, norm_r;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(rhs, &r));
  PetscCall(MatMult(A, x, r));
  PetscCall(VecAXPY(r, -1.0, rhs));
  PetscCall(VecNorm(rhs, NORM_2, &norm_rhs));
  PetscCall(VecNorm(x, NORM_2, &norm_x));
  PetscCall(VecNorm(r, NORM_2, &norm_r));
  PetscCall(PetscPrintf(comm,
                        "REPLAY_MATRIX_RESIDUAL label=%s norm_x=%.8e norm_rhs=%.8e residual=%.8e rel_to_rhs=%.8e\n",
                        label, (double)norm_x, (double)norm_rhs, (double)norm_r, (double)(norm_r / PetscMax(norm_rhs, 1.0e-300))));
  PetscCall(VecDestroy(&r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayVecDiff(MPI_Comm comm, const char label[], Vec a, Vec b)
{
  Vec       diff = NULL;
  PetscReal norm_a, norm_b, norm_diff;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(a, &diff));
  PetscCall(VecCopy(a, diff));
  PetscCall(VecAXPY(diff, -1.0, b));
  PetscCall(VecNorm(a, NORM_2, &norm_a));
  PetscCall(VecNorm(b, NORM_2, &norm_b));
  PetscCall(VecNorm(diff, NORM_2, &norm_diff));
  PetscCall(PetscPrintf(comm, "REPLAY_VEC_DIFF label=%s norm_a=%.8e norm_b=%.8e diff=%.8e rel_to_a=%.8e\n",
                        label, (double)norm_a, (double)norm_b, (double)norm_diff, (double)(norm_diff / PetscMax(norm_a, 1.0e-300))));
  PetscCall(VecDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitReplayVecDiff(MPI_Comm comm, const char label[], Vec a, Vec b)
{
  Vec       diff = NULL;
  PetscReal norm_a, norm_b, norm_diff;

  PetscFunctionBeginUser;
  if (!a || !b) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecDuplicate(a, &diff));
  PetscCall(VecCopy(a, diff));
  PetscCall(VecAXPY(diff, -1.0, b));
  PetscCall(VecNorm(a, NORM_2, &norm_a));
  PetscCall(VecNorm(b, NORM_2, &norm_b));
  PetscCall(VecNorm(diff, NORM_2, &norm_diff));
  PetscCall(PetscPrintf(comm, "INIT_REPLAY_VEC_DIFF label=%s norm_export=%.8e norm_C=%.8e diff=%.8e rel_to_export=%.8e\n",
                        label, (double)norm_a, (double)norm_b, (double)norm_diff, (double)(norm_diff / PetscMax(norm_a, 1.0e-300))));
  PetscCall(VecDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayMatrixActionDiff(MPI_Comm comm, const char label[], Mat A_exported, Mat A_c)
{
  Vec       x = NULL, y_exp = NULL, y_c = NULL, diff = NULL;
  PetscInt  lo, hi;
  PetscReal norm_exp, norm_c, norm_diff;
  PetscScalar *arr = NULL;

  PetscFunctionBeginUser;
  if (!A_exported || !A_c) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatCreateVecs(A_c, &x, &y_c));
  PetscCall(VecDuplicate(y_c, &y_exp));
  PetscCall(VecDuplicate(y_c, &diff));
  PetscCall(VecGetOwnershipRange(x, &lo, &hi));
  PetscCall(VecGetArray(x, &arr));
  for (PetscInt i = lo; i < hi; ++i) {
    const PetscReal v = PetscSinReal(0.131 * (PetscReal)(i + 1)) + 0.25 * PetscCosReal(0.017 * (PetscReal)(i + 3));
    arr[i - lo] = v;
  }
  PetscCall(VecRestoreArray(x, &arr));
  PetscCall(MatMult(A_exported, x, y_exp));
  PetscCall(MatMult(A_c, x, y_c));
  PetscCall(VecCopy(y_exp, diff));
  PetscCall(VecAXPY(diff, -1.0, y_c));
  PetscCall(VecNorm(y_exp, NORM_2, &norm_exp));
  PetscCall(VecNorm(y_c, NORM_2, &norm_c));
  PetscCall(VecNorm(diff, NORM_2, &norm_diff));
  PetscCall(PetscPrintf(comm,
                        "INIT_REPLAY_MATRIX_DIFF label=%s norm_export=%.8e norm_C=%.8e action_diff=%.8e rel_to_export=%.8e\n",
                        label, (double)norm_exp, (double)norm_c, (double)norm_diff, (double)(norm_diff / PetscMax(norm_exp, 1.0e-300))));
  PetscCall(VecDestroy(&diff));
  PetscCall(VecDestroy(&y_exp));
  PetscCall(VecDestroy(&y_c));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayProbeCompareStage(MPI_Comm comm, AppCtx *app, const char probe_label[], const char key[], Vec template_vec, Vec c_vec,
                                                    const PetscInt c_to_export[], PetscBool mapped)
{
  Vec       exp = NULL;
  PetscBool have = PETSC_FALSE;
  char      name[128], diff_label[160];

  PetscFunctionBeginUser;
  if (!template_vec || !c_vec) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSNPrintf(name, sizeof(name), "probe_%s_%s.vec", probe_label, key));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->linear_replay_dir, name, template_vec, c_to_export, mapped, &exp, &have));
  if (have) {
    PetscCall(PetscSNPrintf(diff_label, sizeof(diff_label), "probe_%s_%s_exported_minus_C", probe_label, key));
    PetscCall(LinearReplayVecDiff(comm, diff_label, exp, c_vec));
  }
  PetscCall(VecDestroy(&exp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayProbeOne(DM dm, AppCtx *app, LinearSolverCtx *solver, Mat A, PC pc, const char label[], Vec template_vec, PetscInt c_to_export[], PetscBool mapped)
{
  MPI_Comm  comm = PetscObjectComm((PetscObject)dm);
  Vec       v0_exp = NULL, pc_exp = NULL, z_exp = NULL, Az_exp = NULL, arn_exp = NULL, fine_pre_exp = NULL, fine_resid_exp = NULL;
  Vec       pc_c = NULL, z_c = NULL, Az_c = NULL, arn_c = NULL;
  PetscBool have_v0 = PETSC_FALSE, have_pc = PETSC_FALSE, have_z = PETSC_FALSE, have_Az = PETSC_FALSE, have_arn = PETSC_FALSE;
  PetscBool have_fine_pre = PETSC_FALSE, have_fine_resid = PETSC_FALSE, is_shell = PETSC_FALSE;
  PCType    pc_type = NULL;
  PMGShellVCycleCtx *shell_ctx = NULL;
  char      name[128], diff_label[128];

  PetscFunctionBeginUser;
  PetscCall(PetscSNPrintf(name, sizeof(name), "probe_%s_v0_local.vec", label));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->linear_replay_dir, name, template_vec, c_to_export, mapped, &v0_exp, &have_v0));
  if (!have_v0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSNPrintf(name, sizeof(name), "probe_%s_pc_v0_local.vec", label));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->linear_replay_dir, name, template_vec, c_to_export, mapped, &pc_exp, &have_pc));
  PetscCall(PetscSNPrintf(name, sizeof(name), "probe_%s_z0_local.vec", label));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->linear_replay_dir, name, template_vec, c_to_export, mapped, &z_exp, &have_z));
  PetscCall(PetscSNPrintf(name, sizeof(name), "probe_%s_Az0_local.vec", label));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->linear_replay_dir, name, template_vec, c_to_export, mapped, &Az_exp, &have_Az));
  PetscCall(PetscSNPrintf(name, sizeof(name), "probe_%s_arnoldi_residual0_local.vec", label));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->linear_replay_dir, name, template_vec, c_to_export, mapped, &arn_exp, &have_arn));
  PetscCall(PetscSNPrintf(name, sizeof(name), "probe_%s_mg_fine_pre_local.vec", label));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->linear_replay_dir, name, template_vec, c_to_export, mapped, &fine_pre_exp, &have_fine_pre));
  PetscCall(PetscSNPrintf(name, sizeof(name), "probe_%s_mg_fine_residual_local.vec", label));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->linear_replay_dir, name, template_vec, c_to_export, mapped, &fine_resid_exp, &have_fine_resid));

  PetscCall(PCGetType(pc, &pc_type));
  if (pc_type) PetscCall(PetscStrcmp(pc_type, PCSHELL, &is_shell));
  if (is_shell) PetscCall(PCShellGetContext(pc, (void **)&shell_ctx));

  PetscCall(VecDuplicate(template_vec, &pc_c));
  PetscCall(VecDuplicate(template_vec, &z_c));
  PetscCall(VecDuplicate(template_vec, &Az_c));
  PetscCall(VecDuplicate(template_vec, &arn_c));
  if (shell_ctx) shell_ctx->debug_capture = PETSC_TRUE;
  PetscCall(PCApply(pc, v0_exp, pc_c));
  if (shell_ctx) shell_ctx->debug_capture = PETSC_FALSE;
  if (shell_ctx && have_fine_pre && shell_ctx->debug_fine_pre) {
    PetscCall(PetscSNPrintf(diff_label, sizeof(diff_label), "probe_%s_mg_fine_pre_exported_minus_C", label));
    PetscCall(LinearReplayVecDiff(comm, diff_label, fine_pre_exp, shell_ctx->debug_fine_pre));
  }
  if (shell_ctx && have_fine_resid && shell_ctx->debug_fine_residual) {
    PetscCall(PetscSNPrintf(diff_label, sizeof(diff_label), "probe_%s_mg_fine_residual_exported_minus_C", label));
    PetscCall(LinearReplayVecDiff(comm, diff_label, fine_resid_exp, shell_ctx->debug_fine_residual));
  }
  if (shell_ctx) {
    PetscCall(LinearReplayProbeCompareStage(comm, app, label, "mg_p2_rhs_local", shell_ctx->p2_rhs, shell_ctx->debug_p2_rhs, NULL, PETSC_FALSE));
    PetscCall(LinearReplayProbeCompareStage(comm, app, label, "mg_p2_pre_local", shell_ctx->p2_rhs, shell_ctx->debug_p2_pre, NULL, PETSC_FALSE));
    PetscCall(LinearReplayProbeCompareStage(comm, app, label, "mg_p2_residual_local", shell_ctx->p2_rhs, shell_ctx->debug_p2_residual, NULL, PETSC_FALSE));
    PetscCall(LinearReplayProbeCompareStage(comm, app, label, "mg_p1_rhs_local", shell_ctx->p1_rhs, shell_ctx->debug_p1_rhs, NULL, PETSC_FALSE));
    PetscCall(LinearReplayProbeCompareStage(comm, app, label, "mg_p1_x_local", shell_ctx->p1_rhs, shell_ctx->debug_p1_x, NULL, PETSC_FALSE));
    PetscCall(LinearReplayProbeCompareStage(comm, app, label, "mg_p2_post_local", shell_ctx->p2_rhs, shell_ctx->debug_p2_post, NULL, PETSC_FALSE));
  }
  if (have_pc) {
    PetscCall(PetscSNPrintf(diff_label, sizeof(diff_label), "probe_%s_pc_v0_exported_minus_C", label));
    PetscCall(LinearReplayVecDiff(comm, diff_label, pc_exp, pc_c));
  }
  PetscCall(DeflationApplyMatlabProjectedPC(solver, A, pc, v0_exp, z_c, Az_c));
  if (have_z) {
    PetscCall(PetscSNPrintf(diff_label, sizeof(diff_label), "probe_%s_z0_exported_minus_C", label));
    PetscCall(LinearReplayVecDiff(comm, diff_label, z_exp, z_c));
  }
  PetscCall(MatMult(A, z_c, Az_c));
  if (have_Az) {
    PetscCall(PetscSNPrintf(diff_label, sizeof(diff_label), "probe_%s_Az0_exported_minus_C", label));
    PetscCall(LinearReplayVecDiff(comm, diff_label, Az_exp, Az_c));
  }
  {
    PetscScalar h00;
    PetscReal   h10;

    PetscCall(VecCopy(Az_c, arn_c));
    PetscCall(VecDot(v0_exp, arn_c, &h00));
    PetscCall(VecAXPY(arn_c, -h00, v0_exp));
    PetscCall(VecNorm(arn_c, NORM_2, &h10));
    if (have_arn) {
      PetscCall(PetscSNPrintf(diff_label, sizeof(diff_label), "probe_%s_arnoldi0_exported_minus_C", label));
      PetscCall(LinearReplayVecDiff(comm, diff_label, arn_exp, arn_c));
    }
    PetscCall(PetscPrintf(comm, "REPLAY_PC_PROBE label=%s h00=%.17e h10=%.17e\n", label, (double)PetscRealPart(h00), (double)h10));
  }

  PetscCall(VecDestroy(&v0_exp));
  PetscCall(VecDestroy(&pc_exp));
  PetscCall(VecDestroy(&z_exp));
  PetscCall(VecDestroy(&Az_exp));
  PetscCall(VecDestroy(&arn_exp));
  PetscCall(VecDestroy(&fine_pre_exp));
  PetscCall(VecDestroy(&fine_resid_exp));
  PetscCall(VecDestroy(&pc_c));
  PetscCall(VecDestroy(&z_c));
  PetscCall(VecDestroy(&Az_c));
  PetscCall(VecDestroy(&arn_c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayProbePC(DM dm, AppCtx *app, LinearSolverCtx *solver, Mat A, Vec template_vec, PetscInt c_to_export[], PetscBool mapped)
{
  KSP       ksp = NULL;
  PC        pc = NULL;
  PetscBool owned_ksp = PETSC_FALSE;

  PetscFunctionBeginUser;
  if (solver->n_raw_basis) PetscCall(LinearSolverAOrthogonalizeBasis(solver, A, "REPLAY PC probe"));
  if (solver->reuse) {
    if (!solver->ksp) {
      PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm), &solver->ksp));
      PetscCall(ConfigureKSP(solver->ksp, solver->dm, solver->actx, solver->app, A, PETSC_TRUE, solver));
    } else {
      PetscCall(RefreshKSPOperators(solver->ksp, solver->app, A, PETSC_FALSE, PETSC_FALSE));
    }
    ksp = solver->ksp;
  } else {
    owned_ksp = PETSC_TRUE;
    PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm), &ksp));
    PetscCall(ConfigureKSP(ksp, solver->dm, solver->actx, solver->app, A, PETSC_TRUE, solver));
  }
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(LinearReplayProbeOne(dm, app, solver, A, pc, "du", template_vec, c_to_export, mapped));
  PetscCall(LinearReplayProbeOne(dm, app, solver, A, pc, "dW", template_vec, c_to_export, mapped));
  PetscCall(LinearReplayProbeOne(dm, app, solver, A, pc, "dV", template_vec, c_to_export, mapped));
  if (owned_ksp) PetscCall(KSPDestroy(&ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayVecComponentSummary(MPI_Comm comm, const char dir[], const char label[], Vec v, const PetscInt c_to_export[], PetscBool mapped)
{
  char               path[PETSC_MAX_PATH_LEN], line[512];
  FILE              *fh = NULL;
  PetscInt          *export_to_comp = NULL;
  PetscInt           nloc, N;
  const PetscScalar *arr = NULL;
  PetscReal          norm2_local[3] = {0.0, 0.0, 0.0}, norm2_global[3] = {0.0, 0.0, 0.0};
  PetscReal          sum_local[3] = {0.0, 0.0, 0.0}, sum_global[3] = {0.0, 0.0, 0.0};
  PetscInt           count_local[3] = {0, 0, 0}, count_global[3] = {0, 0, 0};

  PetscFunctionBeginUser;
  if (!mapped) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetSize(v, &N));
  PetscCall(PetscCalloc1(N, &export_to_comp));
  for (PetscInt i = 0; i < N; ++i) export_to_comp[i] = -1;
  PetscCall(PetscSNPrintf(path, sizeof(path), "%s/free_dof_map.csv", dir));
  fh = fopen(path, "r");
  PetscCheck(fh, comm, PETSC_ERR_FILE_OPEN, "Could not open replay DOF map %s", path);
  PetscCheck(fgets(line, sizeof(line), fh), comm, PETSC_ERR_FILE_UNEXPECTED, "Empty replay DOF map %s", path);
  while (fgets(line, sizeof(line), fh)) {
    long long export_index, full_index, node, comp;
    double    x, y, z;
    int       got = sscanf(line, "%lld,%lld,%lld,%lld,%lf,%lf,%lf", &export_index, &full_index, &node, &comp, &x, &y, &z);
    PetscCheck(got == 7, comm, PETSC_ERR_FILE_UNEXPECTED, "Malformed replay DOF map row in %s: %s", path, line);
    if (export_index >= 0 && export_index < N) export_to_comp[(PetscInt)export_index] = (PetscInt)comp;
    (void)full_index;
    (void)node;
    (void)x;
    (void)y;
    (void)z;
  }
  fclose(fh);
  PetscCall(VecGetLocalSize(v, &nloc));
  PetscCall(VecGetArrayRead(v, &arr));
  for (PetscInt i = 0; i < nloc; ++i) {
    const PetscInt exp  = c_to_export[i];
    const PetscInt comp = (exp >= 0 && exp < N) ? export_to_comp[exp] : -1;
    const PetscReal val = PetscRealPart(arr[i]);
    if (comp >= 0 && comp < 3) {
      norm2_local[comp] += val * val;
      sum_local[comp] += val;
      count_local[comp]++;
    }
  }
  PetscCall(VecRestoreArrayRead(v, &arr));
  PetscCallMPI(MPI_Allreduce(norm2_local, norm2_global, 3, MPIU_REAL, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(sum_local, sum_global, 3, MPIU_REAL, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(count_local, count_global, 3, MPIU_INT, MPI_SUM, comm));
  PetscCall(PetscPrintf(comm,
                        "REPLAY_VEC_COMPONENT label=%s comp0_norm=%.8e comp1_norm=%.8e comp2_norm=%.8e comp0_sum=%.8e comp1_sum=%.8e comp2_sum=%.8e comp0_count=%" PetscInt_FMT " comp1_count=%" PetscInt_FMT " comp2_count=%" PetscInt_FMT "\n",
                        label, (double)PetscSqrtReal(norm2_global[0]), (double)PetscSqrtReal(norm2_global[1]), (double)PetscSqrtReal(norm2_global[2]),
                        (double)sum_global[0], (double)sum_global[1], (double)sum_global[2], count_global[0], count_global[1], count_global[2]));
  PetscCall(PetscFree(export_to_comp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearReplayRun(DM dm, AssemblyCtx *actx, AppCtx *app, LinearSolverCtx *solver, Mat Areg, Mat Kelastic, Mat Ktangent, Vec f_ext, PetscReal rhs_norm)
{
  MPI_Comm         comm = PetscObjectComm((PetscObject)dm);
  LinearReplayMeta meta;
  Vec              u = NULL, residual = NULL, residual_eps = NULL, G = NULL, F = NULL, F_eps = NULL;
  Vec              f_exp = NULL, F_exp = NULL, G_exp = NULL, F_eps_exp = NULL;
  Vec              rhsW = NULL, rhsV = NULL, rhsW_exp = NULL, rhsV_exp = NULL;
  Vec              dW = NULL, dV = NULL, dW_exp = NULL, dV_exp = NULL;
  Mat              replayA = NULL, solveA = NULL;
  PetscInt        *c_to_export = NULL;
  PetscBool        mapped = PETSC_FALSE;
  PetscBool        have_f = PETSC_FALSE, have_F = PETSC_FALSE, have_G = PETSC_FALSE, have_F_eps = PETSC_FALSE;
  PetscInt         itsW = 0, itsV = 0;
  PetscLogDouble   t0, t1;

  PetscFunctionBeginUser;
  PetscCall(LinearReplayReadMeta(comm, app->linear_replay_dir, &meta));
  PetscCall(LinearReplayBuildCToExportMap(dm, actx->basis, app->linear_replay_dir, f_ext, &c_to_export, &mapped));
  PetscCall(LinearReplayLoadMappedVec(comm, app->linear_replay_dir, "u.vec", f_ext, c_to_export, mapped, &u));
  PetscCall(LinearReplayMappedVecRoundtrip(comm, app->linear_replay_dir, "u.vec", "u", u, f_ext, c_to_export, mapped));
  PetscCall(VecDuplicate(f_ext, &residual));
  PetscCall(VecDuplicate(f_ext, &residual_eps));
  PetscCall(VecDuplicate(f_ext, &G));
  PetscCall(VecDuplicate(f_ext, &F));
  PetscCall(VecDuplicate(f_ext, &F_eps));
  PetscCall(VecDuplicate(f_ext, &rhsW));
  PetscCall(VecDuplicate(f_ext, &rhsV));
  PetscCall(VecDuplicate(f_ext, &dW));
  PetscCall(VecDuplicate(f_ext, &dV));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->linear_replay_dir, "f_free.vec", f_ext, c_to_export, mapped, &f_exp, &have_f));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->linear_replay_dir, "F_free.vec", f_ext, c_to_export, mapped, &F_exp, &have_F));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->linear_replay_dir, "G_free.vec", f_ext, c_to_export, mapped, &G_exp, &have_G));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->linear_replay_dir, "F_eps_free.vec", f_ext, c_to_export, mapped, &F_eps_exp, &have_F_eps));
  PetscCall(LinearReplayLoadMappedVec(comm, app->linear_replay_dir, "rhs_w.vec", f_ext, c_to_export, mapped, &rhsW_exp));
  PetscCall(LinearReplayLoadMappedVec(comm, app->linear_replay_dir, "rhs_v.vec", f_ext, c_to_export, mapped, &rhsV_exp));
  PetscCall(LinearReplayLoadMappedVec(comm, app->linear_replay_dir, "solution_w.vec", f_ext, c_to_export, mapped, &dW_exp));
  PetscCall(LinearReplayLoadMappedVec(comm, app->linear_replay_dir, "solution_v.vec", f_ext, c_to_export, mapped, &dV_exp));
  PetscCall(LinearReplayLoadMappedMat(comm, app->linear_replay_dir, f_ext, c_to_export, mapped, &replayA));
  if (replayA) {
    PetscCall(LinearReplayMatResidual(comm, "exported_A_solutionW_minus_rhsW", replayA, dW_exp, rhsW_exp));
    PetscCall(LinearReplayMatResidual(comm, "exported_A_solutionV_minus_rhsV", replayA, dV_exp, rhsV_exp));
  }

  PetscCall(PetscTime(&t0));
  PetscCall(AssemblePlasticResidualJacobian(actx, meta.lambda, u, f_ext, Ktangent, residual, PETSC_TRUE));
  PetscCall(ComputeLambdaDerivativeFD(actx, app, meta.lambda, u, f_ext, residual, residual_eps, G));
  PetscCall(VecWAXPY(F, 1.0, residual, f_ext));
  PetscCall(VecWAXPY(F_eps, 1.0, residual_eps, f_ext));
  PetscCall(BuildRegularizedOperator(Areg, Kelastic, Ktangent, meta.r));
  PetscCall(PetscTime(&t1));
  PetscCall(PetscPrintf(comm, "REPLAY_ASSEMBLY time=%.6g\n", (double)(t1 - t0)));

  if (have_f) PetscCall(LinearReplayVecDiff(comm, "f_free_exported_minus_C", f_exp, f_ext));
  if (have_f) {
    PetscCall(LinearReplayVecComponentSummary(comm, app->linear_replay_dir, "f_free_exported", f_exp, c_to_export, mapped));
    PetscCall(LinearReplayVecComponentSummary(comm, app->linear_replay_dir, "f_free_C", f_ext, c_to_export, mapped));
  }
  if (have_F) PetscCall(LinearReplayVecDiff(comm, "F_free_exported_minus_C", F_exp, F));
  if (have_G) PetscCall(LinearReplayVecDiff(comm, "G_free_exported_minus_C", G_exp, G));
  if (have_F_eps) PetscCall(LinearReplayVecDiff(comm, "F_eps_free_exported_minus_C", F_eps_exp, F_eps));

  PetscCall(VecCopy(G, rhsW));
  PetscCall(VecScale(rhsW, -1.0));
  PetscCall(VecCopy(residual, rhsV));
  PetscCall(VecScale(rhsV, -1.0));
  PetscCall(ZeroConstrainedVector(actx->constrained_is, rhsW));
  PetscCall(ZeroConstrainedVector(actx->constrained_is, rhsV));
  PetscCall(LinearReplayVecDiff(comm, "rhsW_exported_minus_C", rhsW_exp, rhsW));
  PetscCall(LinearReplayVecDiff(comm, "rhsV_exported_minus_C", rhsV_exp, rhsV));
  if (app->linear_replay_use_exported_rhs) {
    PetscCall(VecCopy(rhsW_exp, rhsW));
    PetscCall(VecCopy(rhsV_exp, rhsV));
  }

  PetscCall(PetscPrintf(comm, "REPLAY_BASIS_LOAD cols=%" PetscInt_FMT "\n", meta.basis_cols));
  for (PetscInt i = 0; i < meta.basis_cols; ++i) {
    char name[64];
    Vec  b = NULL;

    PetscCall(PetscSNPrintf(name, sizeof(name), "basis_%04" PetscInt_FMT ".vec", i));
    PetscCall(LinearReplayLoadMappedVec(comm, app->linear_replay_dir, name, f_ext, c_to_export, mapped, &b));
    PetscCall(LinearSolverAppendRawBasis(solver, b, "linear replay exported basis"));
    PetscCall(VecDestroy(&b));
  }

  solveA    = replayA ? replayA : Areg;
  solver->A = solveA;
  if (app->linear_replay_check_pc_probe) PetscCall(LinearReplayProbePC(dm, app, solver, solveA, f_ext, c_to_export, mapped));
  PetscCall(VecZeroEntries(dW));
  PetscCall(VecZeroEntries(dV));
  PetscCall(SolveLinearSystem(solver, rhsW, dW, "REPLAY dW", u, PETSC_TRUE, &itsW));
  PetscCall(SolveLinearSystem(solver, rhsV, dV, "REPLAY dV", u, PETSC_TRUE, &itsV));
  PetscCall(LinearReplayVecDiff(comm, "solutionW_exported_minus_C", dW_exp, dW));
  PetscCall(LinearReplayVecDiff(comm, "solutionV_exported_minus_C", dV_exp, dV));
  PetscCall(PetscPrintf(comm,
                        "REPLAY_RESULT dir=%s exported_rhs=%s expected_dW=%" PetscInt_FMT " c_dW=%" PetscInt_FMT " expected_dV=%" PetscInt_FMT " c_dV=%" PetscInt_FMT " expected_total=%" PetscInt_FMT " c_total=%" PetscInt_FMT " basis_cols=%" PetscInt_FMT "\n",
                        app->linear_replay_dir, app->linear_replay_use_exported_rhs ? "true" : "false", meta.expected_dW_iterations, itsW,
                        meta.expected_dV_iterations, itsV, meta.expected_dW_iterations + meta.expected_dV_iterations, itsW + itsV,
                        solver->n_raw_basis));
  (void)rhs_norm;

  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&residual));
  PetscCall(VecDestroy(&residual_eps));
  PetscCall(VecDestroy(&G));
  PetscCall(VecDestroy(&F));
  PetscCall(VecDestroy(&F_eps));
  PetscCall(VecDestroy(&f_exp));
  PetscCall(VecDestroy(&F_exp));
  PetscCall(VecDestroy(&G_exp));
  PetscCall(VecDestroy(&F_eps_exp));
  PetscCall(VecDestroy(&rhsW));
  PetscCall(VecDestroy(&rhsV));
  PetscCall(VecDestroy(&rhsW_exp));
  PetscCall(VecDestroy(&rhsV_exp));
  PetscCall(VecDestroy(&dW));
  PetscCall(VecDestroy(&dV));
  PetscCall(VecDestroy(&dW_exp));
  PetscCall(VecDestroy(&dV_exp));
  PetscCall(MatDestroy(&replayA));
  PetscCall(PetscFree(c_to_export));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StepReplayRun(DM dm, AssemblyCtx *actx, AppCtx *app, LinearSolverCtx *solver, Mat Areg, Mat Kelastic, Mat Ktangent, Vec f_ext, PetscReal rhs_norm, NewtonStats *stats)
{
  MPI_Comm       comm = PetscObjectComm((PetscObject)dm);
  LinearReplayMeta meta;
  PetscInt      *c_to_export = NULL;
  PetscBool      mapped = PETSC_FALSE;
  Vec            u = NULL, residual = NULL, residual_eps = NULL, G = NULL, F = NULL, F_eps = NULL, rhsW = NULL, rhsV = NULL;
  Vec            f_exp = NULL, F_exp = NULL, G_exp = NULL, F_eps_exp = NULL, rhsW_exp = NULL, rhsV_exp = NULL;
  Mat            replayA = NULL;
  PetscBool      have_f = PETSC_FALSE, have_F = PETSC_FALSE, have_G = PETSC_FALSE, have_F_eps = PETSC_FALSE, have_rhsW = PETSC_FALSE, have_rhsV = PETSC_FALSE;
  PetscReal      lambda;
  PetscLogDouble t0, t1;

  PetscFunctionBeginUser;
  PetscCall(LinearReplayReadMeta(comm, app->step_replay_dir, &meta));
  PetscCall(PetscPrintf(comm,
                        "STEP_REPLAY_META dir=%s kind=%s sample_id=%" PetscInt_FMT " omega=%.8e lambda_start=%.8e r_start=%.8e exported_newton_iteration=%" PetscInt_FMT " exported_basis_cols=%" PetscInt_FMT " exported_first_dW=%" PetscInt_FMT " exported_first_dV=%" PetscInt_FMT "\n",
                        app->step_replay_dir, meta.kind, meta.sample_id, (double)meta.omega, (double)meta.lambda, (double)meta.r,
                        meta.newton_iteration, meta.basis_cols, meta.expected_dW_iterations, meta.expected_dV_iterations));

  PetscCall(LinearReplayBuildCToExportMap(dm, actx->basis, app->step_replay_dir, f_ext, &c_to_export, &mapped));
  PetscCall(PetscPrintf(comm, "STEP_REPLAY_MAP_CHECK mapped=%s\n", mapped ? "true" : "false"));
  PetscCall(LinearReplayLoadMappedVec(comm, app->step_replay_dir, "u.vec", f_ext, c_to_export, mapped, &u));
  PetscCall(LinearReplayMappedVecRoundtrip(comm, app->step_replay_dir, "u.vec", "u", u, f_ext, c_to_export, mapped));
  PetscCall(VecDuplicate(f_ext, &residual));
  PetscCall(VecDuplicate(f_ext, &residual_eps));
  PetscCall(VecDuplicate(f_ext, &G));
  PetscCall(VecDuplicate(f_ext, &F));
  PetscCall(VecDuplicate(f_ext, &F_eps));
  PetscCall(VecDuplicate(f_ext, &rhsW));
  PetscCall(VecDuplicate(f_ext, &rhsV));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->step_replay_dir, "f_free.vec", f_ext, c_to_export, mapped, &f_exp, &have_f));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->step_replay_dir, "F_free.vec", f_ext, c_to_export, mapped, &F_exp, &have_F));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->step_replay_dir, "G_free.vec", f_ext, c_to_export, mapped, &G_exp, &have_G));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->step_replay_dir, "F_eps_free.vec", f_ext, c_to_export, mapped, &F_eps_exp, &have_F_eps));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->step_replay_dir, "rhs_w.vec", f_ext, c_to_export, mapped, &rhsW_exp, &have_rhsW));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->step_replay_dir, "rhs_v.vec", f_ext, c_to_export, mapped, &rhsV_exp, &have_rhsV));

  PetscCall(PetscTime(&t0));
  PetscCall(AssemblePlasticResidualJacobian(actx, meta.lambda, u, f_ext, Ktangent, residual, PETSC_TRUE));
  PetscCall(ComputeLambdaDerivativeFD(actx, app, meta.lambda, u, f_ext, residual, residual_eps, G));
  PetscCall(VecWAXPY(F, 1.0, residual, f_ext));
  PetscCall(VecWAXPY(F_eps, 1.0, residual_eps, f_ext));
  PetscCall(BuildRegularizedOperator(Areg, Kelastic, Ktangent, meta.r));
  PetscCall(PetscTime(&t1));
  PetscCall(PetscPrintf(comm, "STEP_REPLAY_ASSEMBLY_CHECK time=%.6g\n", (double)(t1 - t0)));

  PetscCall(VecCopy(G, rhsW));
  PetscCall(VecScale(rhsW, -1.0));
  PetscCall(VecCopy(residual, rhsV));
  PetscCall(VecScale(rhsV, -1.0));
  PetscCall(ZeroConstrainedVector(actx->constrained_is, rhsW));
  PetscCall(ZeroConstrainedVector(actx->constrained_is, rhsV));
  if (have_f) PetscCall(LinearReplayVecDiff(comm, "step_f_free_exported_minus_C", f_exp, f_ext));
  if (have_F) PetscCall(LinearReplayVecDiff(comm, "step_F_free_exported_minus_C", F_exp, F));
  if (have_G) PetscCall(LinearReplayVecDiff(comm, "step_G_free_exported_minus_C", G_exp, G));
  if (have_F_eps) PetscCall(LinearReplayVecDiff(comm, "step_F_eps_free_exported_minus_C", F_eps_exp, F_eps));
  if (have_rhsW) PetscCall(LinearReplayVecDiff(comm, "step_rhsW_exported_minus_C", rhsW_exp, rhsW));
  if (have_rhsV) PetscCall(LinearReplayVecDiff(comm, "step_rhsV_exported_minus_C", rhsV_exp, rhsV));
  PetscCall(LinearReplayLoadMappedMat(comm, app->step_replay_dir, f_ext, c_to_export, mapped, &replayA));
  if (replayA) PetscCall(LinearReplayMatrixActionDiff(comm, "step_Areg_exported_minus_C", replayA, Areg));

  PetscCall(PetscPrintf(comm, "STEP_REPLAY_BASIS_LOAD cols=%" PetscInt_FMT "\n", meta.basis_cols));
  for (PetscInt i = 0; i < meta.basis_cols; ++i) {
    char name[64];
    Vec  b = NULL;

    PetscCall(PetscSNPrintf(name, sizeof(name), "basis_%04" PetscInt_FMT ".vec", i));
    PetscCall(LinearReplayLoadMappedVec(comm, app->step_replay_dir, name, f_ext, c_to_export, mapped, &b));
    PetscCall(LinearSolverAppendRawBasis(solver, b, "step replay exported basis"));
    PetscCall(VecDestroy(&b));
  }
  {
    char saved_linear_replay_dir[PETSC_MAX_PATH_LEN];

    PetscCall(PetscStrncpy(saved_linear_replay_dir, app->linear_replay_dir, sizeof(saved_linear_replay_dir)));
    PetscCall(PetscStrncpy(app->linear_replay_dir, app->step_replay_dir, sizeof(app->linear_replay_dir)));
    solver->A = Areg;
    if (app->linear_replay_check_pc_probe) PetscCall(LinearReplayProbePC(dm, app, solver, Areg, f_ext, c_to_export, mapped));
    PetscCall(PetscStrncpy(app->linear_replay_dir, saved_linear_replay_dir, sizeof(app->linear_replay_dir)));
  }

  lambda = meta.lambda;
  PetscCall(IndirectNewtonSolve(dm, actx, app, solver, Areg, Kelastic, Ktangent, f_ext, u, &lambda, meta.omega, rhs_norm, meta.r, app->newton_stopping_criterion, app->newton_stopping_tol, stats));
  PetscCall(PetscPrintf(comm,
                        "STEP_REPLAY_RESULT dir=%s omega=%.8e lambda_start=%.8e lambda_end=%.8e converged=%s newton_its=%" PetscInt_FMT " linear_its=%" PetscInt_FMT " final_rel=%.6e final_rel_correction=%.6e basis_cols_end=%" PetscInt_FMT "\n",
                        app->step_replay_dir, (double)meta.omega, (double)meta.lambda, (double)lambda, stats->converged ? "true" : "false",
                        stats->newton_its, stats->total_linear_its, (double)stats->final_rel, (double)stats->final_rel_correction,
                        solver->n_raw_basis));

  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&residual));
  PetscCall(VecDestroy(&residual_eps));
  PetscCall(VecDestroy(&G));
  PetscCall(VecDestroy(&F));
  PetscCall(VecDestroy(&F_eps));
  PetscCall(VecDestroy(&rhsW));
  PetscCall(VecDestroy(&rhsV));
  PetscCall(VecDestroy(&f_exp));
  PetscCall(VecDestroy(&F_exp));
  PetscCall(VecDestroy(&G_exp));
  PetscCall(VecDestroy(&F_eps_exp));
  PetscCall(VecDestroy(&rhsW_exp));
  PetscCall(VecDestroy(&rhsV_exp));
  PetscCall(MatDestroy(&replayA));
  PetscCall(PetscFree(c_to_export));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitReplayRun(DM dm, AssemblyCtx *actx, AppCtx *app, LinearSolverCtx *solver, Mat Areg, Mat Kelastic, Mat Ktangent, Vec f_ext, PetscReal rhs_norm)
{
  MPI_Comm       comm = PetscObjectComm((PetscObject)dm);
  LinearReplayMeta meta;
  PetscInt      *c_to_export = NULL;
  PetscBool      mapped = PETSC_FALSE;
  Vec            u_exp = NULL, u = NULL, residual = NULL, F = NULL, rhs_c = NULL, rhs_exp = NULL, du_exp = NULL, du_c = NULL, u_after_exp = NULL;
  Vec            u_after_c = NULL, damping_trial = NULL;
  Mat            Areg_exp = NULL, Ktangent_exp = NULL, Kelastic_exp = NULL, solveA = NULL;
  PetscBool      have_F_exp = PETSC_FALSE, have_f_exp = PETSC_FALSE;
  Vec            F_exp = NULL, f_exp = NULL;
  PetscInt       its = 0;
  PetscReal      rel = PETSC_MAX_REAL, alpha_c = 0.0, initial_decrease_c = PETSC_MAX_REAL, du_norm, u_norm, rel_corr_c;
  PetscInt       ls_c = 0;
  PetscLogDouble t0, t1;

  PetscFunctionBeginUser;
  PetscCall(LinearReplayReadMeta(comm, app->init_replay_dir, &meta));
  PetscCall(PetscPrintf(comm,
                        "INIT_REPLAY_META dir=%s kind=%s sample_id=%" PetscInt_FMT " lambda=%.8e r=%.8e it=%" PetscInt_FMT " basis_cols=%" PetscInt_FMT "\n",
                        app->init_replay_dir, meta.kind, meta.sample_id, (double)meta.lambda, (double)meta.r, meta.newton_iteration,
                        meta.basis_cols));
  PetscCall(LinearReplayBuildCToExportMap(dm, actx->basis, app->init_replay_dir, f_ext, &c_to_export, &mapped));
  PetscCall(PetscPrintf(comm, "INIT_REPLAY_MAP_CHECK mapped=%s\n", mapped ? "true" : "false"));
  PetscCall(LinearReplayLoadMappedVec(comm, app->init_replay_dir, "u_before.vec", f_ext, c_to_export, mapped, &u_exp));
  PetscCall(LinearReplayMappedVecRoundtrip(comm, app->init_replay_dir, "u_before.vec", "u_before", u_exp, f_ext, c_to_export, mapped));
  PetscCall(VecDuplicate(f_ext, &u));
  if (app->init_replay_use_exported_u) PetscCall(VecCopy(u_exp, u));
  else PetscCall(VecZeroEntries(u));
  PetscCall(VecDuplicate(f_ext, &residual));
  PetscCall(VecDuplicate(f_ext, &F));
  PetscCall(VecDuplicate(f_ext, &rhs_c));
  PetscCall(VecDuplicate(f_ext, &du_c));
  PetscCall(VecDuplicate(f_ext, &u_after_c));
  PetscCall(VecDuplicate(f_ext, &damping_trial));
  PetscCall(LinearReplayLoadMappedVec(comm, app->init_replay_dir, "rhs.vec", f_ext, c_to_export, mapped, &rhs_exp));
  PetscCall(LinearReplayLoadMappedVec(comm, app->init_replay_dir, "du.vec", f_ext, c_to_export, mapped, &du_exp));
  PetscCall(LinearReplayLoadMappedVec(comm, app->init_replay_dir, "u_after.vec", f_ext, c_to_export, mapped, &u_after_exp));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->init_replay_dir, "F_free.vec", f_ext, c_to_export, mapped, &F_exp, &have_F_exp));
  PetscCall(LinearReplayLoadMappedVecOptional(comm, app->init_replay_dir, "f_free.vec", f_ext, c_to_export, mapped, &f_exp, &have_f_exp));
  PetscCall(LinearReplayLoadMappedMatNamed(comm, app->init_replay_dir, "Areg_free.mat", f_ext, c_to_export, mapped, &Areg_exp));
  PetscCall(LinearReplayLoadMappedMatNamed(comm, app->init_replay_dir, "Ktangent_free.mat", f_ext, c_to_export, mapped, &Ktangent_exp));
  PetscCall(LinearReplayLoadMappedMatNamed(comm, app->init_replay_dir, "Kelastic_free.mat", f_ext, c_to_export, mapped, &Kelastic_exp));

  PetscCall(PetscTime(&t0));
  PetscCall(AssemblePlasticResidualJacobian(actx, meta.lambda, u, f_ext, Ktangent, residual, PETSC_TRUE));
  PetscCall(BuildRegularizedOperator(Areg, Kelastic, Ktangent, meta.r));
  PetscCall(PetscTime(&t1));
  PetscCall(PetscPrintf(comm, "INIT_REPLAY_ASSEMBLY time=%.6g\n", (double)(t1 - t0)));
  PetscCall(ResidualNormFree(actx, residual, rhs_norm, &rel));
  PetscCall(VecWAXPY(F, 1.0, residual, f_ext));
  PetscCall(VecCopy(residual, rhs_c));
  PetscCall(VecScale(rhs_c, -1.0));
  PetscCall(ZeroConstrainedVector(actx->constrained_is, rhs_c));

  PetscCall(InitReplayVecDiff(comm, "u_before_exported_minus_C", u_exp, u));
  if (have_f_exp) PetscCall(InitReplayVecDiff(comm, "f_free_exported_minus_C", f_exp, f_ext));
  if (have_F_exp) PetscCall(InitReplayVecDiff(comm, "F_free_exported_minus_C", F_exp, F));
  PetscCall(InitReplayVecDiff(comm, "rhs_exported_minus_C", rhs_exp, rhs_c));
  if (Areg_exp) PetscCall(LinearReplayMatrixActionDiff(comm, "Areg_exported_minus_C", Areg_exp, Areg));
  if (Ktangent_exp) PetscCall(LinearReplayMatrixActionDiff(comm, "Ktangent_exported_minus_C", Ktangent_exp, Ktangent));
  if (Kelastic_exp) PetscCall(LinearReplayMatrixActionDiff(comm, "Kelastic_exported_minus_C", Kelastic_exp, Kelastic));

  PetscCall(PetscPrintf(comm, "INIT_REPLAY_BASIS_LOAD cols=%" PetscInt_FMT "\n", meta.basis_cols));
  for (PetscInt i = 0; i < meta.basis_cols; ++i) {
    char name[64];
    Vec  b = NULL;

    PetscCall(PetscSNPrintf(name, sizeof(name), "basis_%04" PetscInt_FMT ".vec", i));
    PetscCall(LinearReplayLoadMappedVec(comm, app->init_replay_dir, name, f_ext, c_to_export, mapped, &b));
    PetscCall(LinearSolverAppendRawBasis(solver, b, "init replay exported basis"));
    PetscCall(VecDestroy(&b));
  }
  PetscCall(PetscPrintf(comm, "INIT_REPLAY_BASIS_COMPARE exported_cols=%" PetscInt_FMT " C_raw_cols=%" PetscInt_FMT "\n", meta.basis_cols, solver->n_raw_basis));

  solveA = (app->init_replay_use_exported_matrix && Areg_exp) ? Areg_exp : Areg;
  solver->A = solveA;
  PetscCall(PetscStrncpy(app->linear_replay_dir, app->init_replay_dir, sizeof(app->linear_replay_dir)));
  if (app->linear_replay_check_pc_probe) PetscCall(LinearReplayProbePC(dm, app, solver, solveA, f_ext, c_to_export, mapped));
  PetscCall(VecZeroEntries(du_c));
  PetscCall(SolveLinearSystem(solver, app->init_replay_use_exported_rhs ? rhs_exp : rhs_c, du_c, "INIT REPLAY fixed-lambda Newton correction", u, PETSC_TRUE, &its));
  PetscCall(InitReplayVecDiff(comm, "du_exported_minus_C_solve", du_exp, du_c));
  PetscCall(PetscPrintf(comm,
                        "INIT_REPLAY_LINEAR_RESULT dir=%s exported_matrix=%s exported_rhs=%s expected_iterations=%" PetscInt_FMT " C_iterations=%" PetscInt_FMT " expected_reported_final=%.8e rel_residual_C_state=%.8e\n",
                        app->init_replay_dir, (app->init_replay_use_exported_matrix && Areg_exp) ? "true" : "false",
                        app->init_replay_use_exported_rhs ? "true" : "false", meta.expected_iterations, its,
                        (double)meta.expected_reported_residual_final, (double)rel));

  if (app->init_replay_check_damping) {
    PetscCall(FixedLambdaDirectionalDamping(actx, app, meta.lambda, u, du_exp, residual, f_ext, damping_trial, u_after_c,
                                            &alpha_c, &ls_c, &initial_decrease_c));
    PetscCall(VecNorm(du_exp, NORM_2, &du_norm));
    PetscCall(VecNorm(u, NORM_2, &u_norm));
    rel_corr_c = (alpha_c * du_norm) / PetscMax(u_norm, 1.0e-30);
    PetscCall(VecWAXPY(u_after_c, alpha_c, du_exp, u));
    PetscCall(InitReplayVecDiff(comm, "u_after_exported_minus_C_damping_on_exported_du", u_after_exp, u_after_c));
    PetscCall(PetscPrintf(comm,
                          "INIT_REPLAY_DAMPING_COMPARE expected_alpha=%.17e C_alpha=%.17e alpha_diff=%.8e expected_ls=%" PetscInt_FMT " C_ls=%" PetscInt_FMT " expected_initial_decrease=%.17e C_initial_decrease=%.17e expected_rel_correction=%.17e C_rel_correction=%.17e\n",
                          (double)meta.expected_alpha, (double)alpha_c, (double)PetscAbsReal(meta.expected_alpha - alpha_c),
                          meta.expected_line_search_iterations, ls_c, (double)meta.expected_initial_decrease, (double)initial_decrease_c,
                          (double)meta.expected_rel_correction, (double)rel_corr_c));
  }
  PetscCall(PetscPrintf(comm,
                        "INIT_REPLAY_UPDATE_COMPARE dir=%s use_exported_u=%s use_exported_matrix=%s use_exported_rhs=%s\n",
                        app->init_replay_dir, app->init_replay_use_exported_u ? "true" : "false",
                        (app->init_replay_use_exported_matrix && Areg_exp) ? "true" : "false",
                        app->init_replay_use_exported_rhs ? "true" : "false"));

  PetscCall(VecDestroy(&u_exp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&residual));
  PetscCall(VecDestroy(&F));
  PetscCall(VecDestroy(&rhs_c));
  PetscCall(VecDestroy(&rhs_exp));
  PetscCall(VecDestroy(&du_exp));
  PetscCall(VecDestroy(&du_c));
  PetscCall(VecDestroy(&u_after_exp));
  PetscCall(VecDestroy(&u_after_c));
  PetscCall(VecDestroy(&damping_trial));
  PetscCall(VecDestroy(&F_exp));
  PetscCall(VecDestroy(&f_exp));
  PetscCall(MatDestroy(&Areg_exp));
  PetscCall(MatDestroy(&Ktangent_exp));
  PetscCall(MatDestroy(&Kelastic_exp));
  PetscCall(PetscFree(c_to_export));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx          app;
  P4Basis         basis;
  DM              dm = NULL;
  AssemblyCtx     actx;
  LinearSolverCtx lsolver;
  SSRStats        ssr_stats;
  Mat             Areg = NULL, Kelastic = NULL, Ktangent = NULL;
  Vec             u = NULL, f_ext = NULL;
  PetscInt        cStart, cEnd, nStart, nEnd;
  PetscInt        local_dofs, global_dofs;
  PetscReal       rhs_norm;
  PetscLogDouble  t_start, t_end, t0, t1, elastic_assembly_time;

  PetscCall(PetscInitialize(&argc, &argv, NULL, "Standalone pure PETSc P4 indirect SSR case\n"));
  PetscCall(RegisterLogStages());
  PetscCall(PetscTime(&t_start));
  PetscCall(ParseOptions(PETSC_COMM_WORLD, &app));
  PetscCall(SetDDPartitionerDefault(PETSC_COMM_WORLD, &app));
  PetscCall(P4BasisCreate(PETSC_COMM_SELF, &basis));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &app, &basis, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &nStart, &nEnd));
  PetscCall(DMCreateMatrix(dm, &Areg));
  PetscCall(DMCreateMatrix(dm, &Kelastic));
  PetscCall(DMCreateMatrix(dm, &Ktangent));
  PetscCall(MatSetOption(Areg, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(MatSetOption(Kelastic, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(MatSetOption(Ktangent, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(DMCreateGlobalVector(dm, &f_ext));
  PetscCall(VecGetLocalSize(u, &local_dofs));
  PetscCall(VecGetSize(u, &global_dofs));
  PetscCall(AssemblyCtxCreate(dm, &basis, &actx));
  PetscCall(LinearSolverInit(&lsolver, dm, &actx, &app, Areg));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "mesh=%s%s refine_levels=%" PetscInt_FMT " local_cells=%" PetscInt_FMT " local_vertices=%" PetscInt_FMT " local_dofs=%" PetscInt_FMT " global_dofs=%" PetscInt_FMT " P4_basis=%" PetscInt_FMT " owned_constraints=%" PetscInt_FMT " global_constraints=%" PetscInt_FMT " pc_variant=%s pmg_backend=%s bc=%s omega_max=%.6g lambda_init=%.6g d_lambda_init=%.6g curve_csv=%s\n",
                        app.use_box_mesh ? "generated-box:" : "", app.use_box_mesh ? "unit" : app.mesh, app.refine_levels, cEnd - cStart, nEnd - nStart, local_dofs, global_dofs, basis.n_basis, actx.n_constrained_local, actx.n_constrained_all, app.variant_name, app.pmg_apply_backend, app.mesh_bc_mode, (double)app.omega_max, (double)app.lambda_init, (double)app.d_lambda_init, app.curve_csv));
  if (app.inspect_partition) {
    PetscCall(ReportPartitionDiagnostics(dm, Areg, u, &actx, &app));
    PetscCall(LinearSolverDestroy(&lsolver));
    PetscCall(AssemblyCtxDestroy(&actx));
    PetscCall(VecDestroy(&f_ext));
    PetscCall(VecDestroy(&u));
    PetscCall(MatDestroy(&Ktangent));
    PetscCall(MatDestroy(&Kelastic));
    PetscCall(MatDestroy(&Areg));
    PetscCall(DMDestroy(&dm));
    PetscCall(P4BasisDestroy(&basis));
    PetscCall(PetscFinalize());
    return 0;
  }

  PetscCall(PetscTime(&t0));
  PetscCall(AssembleElasticProblem(&actx, Kelastic, f_ext));
  PetscCall(PetscTime(&t1));
  elastic_assembly_time = t1 - t0;
  PetscCall(ApplyZeroDirichlet(actx.constrained_is, Kelastic, f_ext));
  PetscCall(VecNorm(f_ext, NORM_2, &rhs_norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "SSR_ELASTIC assembly_time=%.6g rhs_norm=%.6e\n", (double)elastic_assembly_time, (double)rhs_norm));
  {
    PetscBool check_symmetry = PETSC_FALSE;

    PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_matrix_symmetry", &check_symmetry, NULL));
    if (check_symmetry) {
      PetscBool symmetric;

      PetscCall(MatIsSymmetric(Kelastic, 1.0e-10, &symmetric));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "elastic matrix_symmetric=%s tol=1e-10\n", symmetric ? "true" : "false"));
    }
  }
  PetscCall(VecZeroEntries(u));
  if (app.init_replay_dir[0]) {
    PetscCall(InitReplayRun(dm, &actx, &app, &lsolver, Areg, Kelastic, Ktangent, f_ext, rhs_norm));
    PetscCall(PetscTime(&t_end));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "RESULT variant=%s init_replay=true global_dofs=%" PetscInt_FMT " wall_time=%.6g deflation=%s deflation_solver=%s init_replay_dir=%s\n",
                          app.variant_name, global_dofs, (double)(t_end - t_start), app.use_deflation ? "true" : "false",
                          app.deflation_solver_name, app.init_replay_dir));
    PetscCall(LinearSolverDestroy(&lsolver));
    PetscCall(AssemblyCtxDestroy(&actx));
    PetscCall(VecDestroy(&f_ext));
    PetscCall(VecDestroy(&u));
    PetscCall(MatDestroy(&Ktangent));
    PetscCall(MatDestroy(&Kelastic));
    PetscCall(MatDestroy(&Areg));
    PetscCall(DMDestroy(&dm));
    PetscCall(P4BasisDestroy(&basis));
    PetscCall(PetscFinalize());
    return 0;
  }
  if (app.step_replay_dir[0]) {
    NewtonStats step_stats;

    PetscCall(StepReplayRun(dm, &actx, &app, &lsolver, Areg, Kelastic, Ktangent, f_ext, rhs_norm, &step_stats));
    PetscCall(PetscTime(&t_end));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "DEFLATION_TIMING orthogonalization_time=%.6g coarse_initial_time=%.6g coarse_initial_calls=%" PetscInt_FMT " pc_apply_time=%.6g projector_time=%.6g projected_pc_calls=%" PetscInt_FMT "\n",
                          (double)lsolver.deflation_orthogonalization_time, (double)lsolver.deflation_coarse_time, lsolver.deflation_coarse_calls,
                          (double)lsolver.deflation_pc_apply_time, (double)lsolver.deflation_projector_time, lsolver.deflation_projected_pc_calls));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "RESULT variant=%s step_replay=true global_dofs=%" PetscInt_FMT " newton_its=%" PetscInt_FMT " linear_its=%" PetscInt_FMT " wall_time=%.6g deflation=%s deflation_solver=%s step_replay_dir=%s converged=%s final_rel=%.6e final_rel_correction=%.6e\n",
                          app.variant_name, global_dofs, step_stats.newton_its, step_stats.total_linear_its, (double)(t_end - t_start),
                          app.use_deflation ? "true" : "false", app.deflation_solver_name, app.step_replay_dir,
                          step_stats.converged ? "true" : "false", (double)step_stats.final_rel, (double)step_stats.final_rel_correction));
    PetscCall(LinearSolverDestroy(&lsolver));
    PetscCall(AssemblyCtxDestroy(&actx));
    PetscCall(VecDestroy(&f_ext));
    PetscCall(VecDestroy(&u));
    PetscCall(MatDestroy(&Ktangent));
    PetscCall(MatDestroy(&Kelastic));
    PetscCall(MatDestroy(&Areg));
    PetscCall(DMDestroy(&dm));
    PetscCall(P4BasisDestroy(&basis));
    PetscCall(PetscFinalize());
    return 0;
  }
  if (app.linear_replay_dir[0]) {
    PetscCall(LinearReplayRun(dm, &actx, &app, &lsolver, Areg, Kelastic, Ktangent, f_ext, rhs_norm));
    PetscCall(PetscTime(&t_end));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "RESULT variant=%s replay=true global_dofs=%" PetscInt_FMT " wall_time=%.6g deflation=%s deflation_solver=%s replay_dir=%s\n",
                          app.variant_name, global_dofs, (double)(t_end - t_start), app.use_deflation ? "true" : "false",
                          app.deflation_solver_name, app.linear_replay_dir));
    PetscCall(LinearSolverDestroy(&lsolver));
    PetscCall(AssemblyCtxDestroy(&actx));
    PetscCall(VecDestroy(&f_ext));
    PetscCall(VecDestroy(&u));
    PetscCall(MatDestroy(&Ktangent));
    PetscCall(MatDestroy(&Kelastic));
    PetscCall(MatDestroy(&Areg));
    PetscCall(DMDestroy(&dm));
    PetscCall(P4BasisDestroy(&basis));
    PetscCall(PetscFinalize());
    return 0;
  }
  PetscCall(SSRContinuationSolve(dm, &actx, &app, &lsolver, Areg, Kelastic, Ktangent, f_ext, rhs_norm, &ssr_stats));
  PetscCall(PetscTime(&t_end));
  {
    PetscMPIInt size;
    PetscPartitioner     part = NULL;
    PetscPartitionerType part_type = NULL;

    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCall(DMPlexGetPartitioner(dm, &part));
    if (part) PetscCall(PetscPartitionerGetType(part, &part_type));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "DEFLATION_TIMING orthogonalization_time=%.6g coarse_initial_time=%.6g coarse_initial_calls=%" PetscInt_FMT " pc_apply_time=%.6g projector_time=%.6g projected_pc_calls=%" PetscInt_FMT "\n",
                          (double)lsolver.deflation_orthogonalization_time, (double)lsolver.deflation_coarse_time, lsolver.deflation_coarse_calls,
                          (double)lsolver.deflation_pc_apply_time, (double)lsolver.deflation_projector_time, lsolver.deflation_projected_pc_calls));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "RESULT variant=%s ranks=%d partitioner=%s global_dofs=%" PetscInt_FMT " accepted_steps=%" PetscInt_FMT " total_newton_its=%" PetscInt_FMT " total_linear_its=%" PetscInt_FMT " total_line_search_its=%" PetscInt_FMT " elastic_assembly_time=%.6g continuation_wall_time=%.6g wall_time=%.6g omega_last=%.6e lambda_last=%.6e final_rel=%.6e final_rel_correction=%.6e stop_reason=%s deflation=%s deflation_solver=%s deflation_basis_cols=%" PetscInt_FMT " curve_csv=%s\n",
                          app.variant_name, size, part_type ? part_type : "unknown", global_dofs, ssr_stats.accepted_steps, ssr_stats.total_newton_its,
                          ssr_stats.total_linear_its, ssr_stats.total_line_search_its, (double)elastic_assembly_time, (double)ssr_stats.wall_time,
                          (double)(t_end - t_start), (double)ssr_stats.omega_last, (double)ssr_stats.lambda_last, (double)ssr_stats.final_rel,
                          (double)ssr_stats.final_rel_correction, ssr_stats.stop_reason, app.use_deflation ? "true" : "false", app.deflation_solver_name,
                          lsolver.n_raw_basis, app.curve_csv));
  }

  PetscCall(LinearSolverDestroy(&lsolver));
  PetscCall(AssemblyCtxDestroy(&actx));
  PetscCall(VecDestroy(&f_ext));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&Ktangent));
  PetscCall(MatDestroy(&Kelastic));
  PetscCall(MatDestroy(&Areg));
  PetscCall(DMDestroy(&dm));
  PetscCall(P4BasisDestroy(&basis));
  PetscCall(PetscFinalize());
  return 0;
}
