#include "assembly.h"
#include "material_mc.h"
#include "p4_basis.h"
#include "petsc_ssr_algorithms.h"
#include "petsc_ssr_engine.h"
#include "petsc_ssr_profile.h"
#include "petsc_ssr_stats.h"

#include <petscblaslapack.h>
#include <petscdmplex.h>
#include <petscksp.h>
#include <petscviewer.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static PetscErrorCode RegisterLogStages(void)
{
  PetscFunctionBeginUser;
  PetscCall(SsrProfilerRegister(NULL));
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
  SsrRuntimeProfile resolved_profile;
  char      mesh[PETSC_MAX_PATH_LEN];
  PetscInt  refine_levels;
  PetscReal newton_rtol;
  PetscInt  newton_max_it;
  PetscReal omega_max;
  char      analysis[16];
  PetscReal lambda_init;
  PetscReal d_lambda_init;
  PetscReal d_lambda_min;
  PetscReal d_lambda_diff_scaled_min;
  PetscReal lambda_ell;
  PetscReal d_t_min;
  PetscReal d_omega_ini_scale;
  PetscInt  continuation_step_max;
  PetscReal newton_stopping_tol;
  PetscReal init_newton_stopping_tol;
  PetscInt  it_damp_max;
  PetscReal r_min;
  char      newton_stopping_criterion[32];
  char      init_newton_stopping_criterion[32];
  char      continuation_algorithm[32];
  char      continuation_method[32];
  char      newton_algorithm[32];
  char      linear_algorithm[32];
  char      continuation_predictor[32];
  char      omega_step_controller[32];
  char      curve_csv[PETSC_MAX_PATH_LEN];
  char      summary_json[PETSC_MAX_PATH_LEN];
  char      solution_binary[PETSC_MAX_PATH_LEN];
  char      solution_vtk[PETSC_MAX_PATH_LEN];
  char      solution_points_csv[PETSC_MAX_PATH_LEN];
  char      native_problem_manifest[PETSC_MAX_PATH_LEN];
  char      mechanics_bc_labels_csv[PETSC_MAX_PATH_LEN];
  char      mechanics_neumann_labels_csv[PETSC_MAX_PATH_LEN];
  char      seepage_boundary_labels_csv[PETSC_MAX_PATH_LEN];
  char      seepage_pressure_source[64];
  char      seepage_pressure_csv[PETSC_MAX_PATH_LEN];
  char      mechanics_bc_nodes_csv[PETSC_MAX_PATH_LEN];
  PetscBool debug_coordinate_bc_table;
  PetscBool native_problem_manifest_loaded;
  PetscInt  native_manifest_mechanics_dirichlet;
  PetscInt  native_manifest_mechanics_neumann;
  PetscInt  native_manifest_seepage_head;
  PetscInt  native_manifest_seepage_flux;
  PetscReal seepage_grho;
  PetscInt  element_degree;
  PetscInt  mechanics_dim;
  PetscReal ksp_rtol;
  PetscReal damping_min;
  PetscBool line_search;
  PetscBool use_box_mesh;
  PCVariant variant;
  char      variant_name[32];
  char      requested_variant_name[32];
  char      variant_fallback_reason[64];
  char      mesh_bc_mode[32];
  PetscInt  bc_tag_base;
  PetscInt  bc_tag_x_min;
  PetscInt  bc_tag_x_max;
  PetscInt  bc_tag_z_min;
  PetscInt  bc_tag_z_max;
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

typedef SsrNewtonStepStats NewtonStats;

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
  PetscLogDouble ksp_setup_time;
  PetscLogDouble linear_operator_matvec_time;
  PetscLogDouble krylov_orthogonalization_time;
  PetscLogDouble krylov_least_squares_time;
  PetscLogDouble krylov_solution_update_time;
  PetscLogDouble pmg_operator_update_time;
  PetscLogDouble pmg_fine_smooth_time;
  PetscLogDouble pmg_p2_smooth_time;
  PetscLogDouble pmg_restrict_time;
  PetscLogDouble pmg_prolong_time;
  PetscLogDouble pmg_coarse_solve_time;
  PetscLogDouble pmg_residual_time;
  PetscInt       deflation_coarse_calls;
  PetscInt       deflation_projected_pc_calls;
  PetscInt       pmg_apply_calls;
  PetscInt       pmg_operator_updates;
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

static const char *SsrLinearAlgorithmNameForApp(const AppCtx *app)
{
  if (app->variant == VARIANT_PMG) return app->use_deflation ? "pmg-deflated" : "pmg";
  if (app->variant == VARIANT_GAMG) return "gamg";
  if (app->variant == VARIANT_BDDC) return "bddc";
  if (app->variant == VARIANT_FETIDP) return "fetidp";
  return "none";
}

static void SsrNormalizeAlgorithmName(char name[])
{
  if (!name) return;
  for (char *p = name; *p; ++p) {
    if (*p == '_') *p = '-';
  }
}

static PetscErrorCode AppCtxNormalizeAlgorithmSelectors(MPI_Comm comm, AppCtx *app)
{
  const char *expected_linear = NULL;
  const char *expected_newton = NULL;
  PetscBool   flg = PETSC_FALSE, indirect = PETSC_FALSE;

  PetscFunctionBeginUser;
  SsrNormalizeAlgorithmName(app->continuation_algorithm);
  SsrNormalizeAlgorithmName(app->newton_algorithm);
  SsrNormalizeAlgorithmName(app->linear_algorithm);

  PetscCall(PetscStrcasecmp(app->continuation_algorithm, "auto", &flg));
  if (flg) PetscCall(PetscStrncpy(app->continuation_algorithm, app->continuation_method, sizeof(app->continuation_algorithm)));

  PetscCall(PetscStrcasecmp(app->continuation_method, "indirect", &indirect));
  expected_newton = indirect ? "indirect-ssr" : "fixed-load";
  PetscCall(PetscStrcasecmp(app->newton_algorithm, "auto", &flg));
  if (flg) PetscCall(PetscStrncpy(app->newton_algorithm, expected_newton, sizeof(app->newton_algorithm)));

  expected_linear = SsrLinearAlgorithmNameForApp(app);
  PetscCall(PetscStrcasecmp(app->linear_algorithm, "auto", &flg));
  if (flg) PetscCall(PetscStrncpy(app->linear_algorithm, expected_linear, sizeof(app->linear_algorithm)));

  PetscCall(PetscStrcasecmp(app->continuation_algorithm, app->continuation_method, &flg));
  PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-continuation_algorithm must match -continuation_method for the current native continuation implementations");

  PetscCall(PetscStrcasecmp(app->newton_algorithm, expected_newton, &flg));
  PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-newton_algorithm must be %s for continuation method %s", expected_newton, app->continuation_method);

  PetscCall(PetscStrcasecmp(app->linear_algorithm, expected_linear, &flg));
  if (!flg && app->variant == VARIANT_NONE) {
    PetscCall(PetscStrcasecmp(app->linear_algorithm, "debug-direct", &flg));
  }
  PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-linear_algorithm must be compatible with -pc_variant and -deflation");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ValidateAlgorithmRegistrySelection(MPI_Comm comm, const AppCtx *app)
{
  const SsrContinuationOps *continuation_ops = NULL;
  const SsrNewtonOps       *newton_ops = NULL;
  const SsrLinearOps       *linear_ops = NULL;
  const SsrMaterialOps     *material_ops = NULL;
  PetscBool                 indirect = PETSC_FALSE;

  (void)comm;
  PetscFunctionBeginUser;
  PetscCall(SsrContinuationRegistryFind(app->continuation_algorithm, &continuation_ops));
  PetscCall(SsrNewtonRegistryFind("fixed-load", &newton_ops));
  PetscCall(PetscStrcasecmp(app->newton_algorithm, "indirect-ssr", &indirect));
  if (indirect) PetscCall(SsrNewtonRegistryFind(app->newton_algorithm, &newton_ops));
  PetscCall(SsrLinearRegistryFind(app->linear_algorithm, &linear_ops));
  PetscCall(SsrMaterialRegistryFind("mohr_coulomb", &material_ops));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppCtxSyncResolvedProfile(AppCtx *app)
{
  char      ksp_type[32] = "";
  PetscBool ksp_type_set = PETSC_FALSE;
  PetscBool ksp_max_it_set = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscStrncpy(app->resolved_profile.mesh.mesh, app->mesh, sizeof(app->resolved_profile.mesh.mesh)));
  app->resolved_profile.mesh.dim = app->mechanics_dim;
  app->resolved_profile.mesh.element_degree = app->element_degree;
  app->resolved_profile.mesh.refine_levels = app->refine_levels;
  PetscCall(PetscStrncpy(app->resolved_profile.mesh.partitioner, "petsc-options", sizeof(app->resolved_profile.mesh.partitioner)));

  PetscCall(PetscStrncpy(app->resolved_profile.physics.analysis, app->analysis, sizeof(app->resolved_profile.physics.analysis)));
  PetscCall(PetscStrncpy(app->resolved_profile.physics.material_model, "mohr_coulomb", sizeof(app->resolved_profile.physics.material_model)));
  PetscCall(PetscStrncpy(app->resolved_profile.physics.davis_type, "B", sizeof(app->resolved_profile.physics.davis_type)));
  app->resolved_profile.physics.seepage_enabled = (PetscBool)(app->seepage_pressure_csv[0] || app->seepage_boundary_labels_csv[0]);

  PetscCall(PetscStrncpy(app->resolved_profile.continuation.algorithm, app->continuation_algorithm, sizeof(app->resolved_profile.continuation.algorithm)));
  PetscCall(PetscStrncpy(app->resolved_profile.continuation.method, app->continuation_method, sizeof(app->resolved_profile.continuation.method)));
  PetscCall(PetscStrncpy(app->resolved_profile.continuation.predictor, app->continuation_predictor, sizeof(app->resolved_profile.continuation.predictor)));
  PetscCall(PetscStrncpy(app->resolved_profile.continuation.step_controller, app->omega_step_controller, sizeof(app->resolved_profile.continuation.step_controller)));
  app->resolved_profile.continuation.omega_max = app->omega_max;
  app->resolved_profile.continuation.step_max = app->continuation_step_max;

  PetscCall(PetscStrncpy(app->resolved_profile.newton.algorithm, app->newton_algorithm, sizeof(app->resolved_profile.newton.algorithm)));
  PetscCall(PetscStrncpy(app->resolved_profile.newton.method, "regularized", sizeof(app->resolved_profile.newton.method)));
  PetscCall(PetscStrncpy(app->resolved_profile.newton.stopping_criterion, app->newton_stopping_criterion, sizeof(app->resolved_profile.newton.stopping_criterion)));
  app->resolved_profile.newton.stopping_tol = app->newton_stopping_tol;
  app->resolved_profile.newton.max_it = app->newton_max_it;
  app->resolved_profile.newton.damp_max = app->it_damp_max;

  PetscCall(PetscStrncpy(app->resolved_profile.linear.profile, SsrLinearAlgorithmNameForApp(app), sizeof(app->resolved_profile.linear.profile)));
  PetscCall(PetscStrncpy(app->resolved_profile.linear.algorithm, app->linear_algorithm, sizeof(app->resolved_profile.linear.algorithm)));
  PetscCall(PetscStrncpy(app->resolved_profile.linear.pc_variant, app->variant_name, sizeof(app->resolved_profile.linear.pc_variant)));
  PetscCall(PetscStrncpy(app->resolved_profile.linear.requested_pc_variant, app->requested_variant_name, sizeof(app->resolved_profile.linear.requested_pc_variant)));
  PetscCall(PetscStrncpy(app->resolved_profile.linear.pc_variant_fallback_reason, app->variant_fallback_reason, sizeof(app->resolved_profile.linear.pc_variant_fallback_reason)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-ksp_type", ksp_type, sizeof(ksp_type), &ksp_type_set));
  PetscCall(PetscStrncpy(app->resolved_profile.linear.ksp_type, ksp_type_set ? ksp_type : "profile-default", sizeof(app->resolved_profile.linear.ksp_type)));
  app->resolved_profile.linear.rtol = app->ksp_rtol;
  app->resolved_profile.linear.max_it = 0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ksp_max_it", &app->resolved_profile.linear.max_it, &ksp_max_it_set));
  (void)ksp_max_it_set;
  app->resolved_profile.linear.reuse_preconditioner = app->reuse_linear_solver;
  app->resolved_profile.linear.pmg_p2_active_ranks = app->pmg_shell_p2_active_ranks;
  app->resolved_profile.linear.pmg_p1_active_ranks = app->pmg_shell_p1_active_ranks;
  PetscCall(PetscStrncpy(app->resolved_profile.linear.pmg.apply_backend, app->pmg_apply_backend, sizeof(app->resolved_profile.linear.pmg.apply_backend)));
  PetscCall(PetscStrncpy(app->resolved_profile.linear.pmg.shell_subcomm_type, app->pmg_shell_subcomm_type, sizeof(app->resolved_profile.linear.pmg.shell_subcomm_type)));
  PetscCall(PetscStrncpy(app->resolved_profile.linear.pmg.smoother_ksp_type, app->pmg_smoother_ksp_type, sizeof(app->resolved_profile.linear.pmg.smoother_ksp_type)));
  PetscCall(PetscStrncpy(app->resolved_profile.linear.pmg.smoother_pc_type, app->pmg_smoother_pc_type, sizeof(app->resolved_profile.linear.pmg.smoother_pc_type)));
  PetscCall(PetscStrncpy(app->resolved_profile.linear.pmg.coarse_pc_type, app->pmg_coarse_pc_type, sizeof(app->resolved_profile.linear.pmg.coarse_pc_type)));
  PetscCall(PetscStrncpy(app->resolved_profile.linear.pmg.coarse_ksp_type, app->pmg_coarse_telescope_ksp_type, sizeof(app->resolved_profile.linear.pmg.coarse_ksp_type)));
  PetscCall(PetscStrncpy(app->resolved_profile.linear.pmg.coarse_inner_pc_type, app->pmg_coarse_telescope_pc_type, sizeof(app->resolved_profile.linear.pmg.coarse_inner_pc_type)));
  PetscCall(PetscStrncpy(app->resolved_profile.linear.pmg.coarse_telescope_subcomm_type, app->pmg_coarse_telescope_subcomm_type, sizeof(app->resolved_profile.linear.pmg.coarse_telescope_subcomm_type)));
  PetscCall(PetscStrncpy(app->resolved_profile.linear.pmg.p2_telescope_subcomm_type, app->pmg_p2_telescope_subcomm_type, sizeof(app->resolved_profile.linear.pmg.p2_telescope_subcomm_type)));
  PetscCall(PetscStrncpy(app->resolved_profile.linear.pmg.p2_telescope_ksp_type, app->pmg_p2_telescope_ksp_type, sizeof(app->resolved_profile.linear.pmg.p2_telescope_ksp_type)));
  PetscCall(PetscStrncpy(app->resolved_profile.linear.pmg.p2_telescope_pc_type, app->pmg_p2_telescope_pc_type, sizeof(app->resolved_profile.linear.pmg.p2_telescope_pc_type)));
  app->resolved_profile.linear.pmg.p2_active_ranks = app->pmg_shell_p2_active_ranks;
  app->resolved_profile.linear.pmg.p1_active_ranks = app->pmg_shell_p1_active_ranks;
  app->resolved_profile.linear.pmg.lag_preconditioner = app->pmg_lag_preconditioner;
  app->resolved_profile.linear.pmg.smoother_max_it = app->pmg_smoother_max_it;
  app->resolved_profile.linear.pmg.coarse_lu_max_dofs = app->pmg_coarse_lu_max_dofs;
  app->resolved_profile.linear.pmg.coarse_telescope_active_ranks = app->pmg_coarse_telescope_active_ranks;
  app->resolved_profile.linear.pmg.coarse_ksp_rtol = app->pmg_coarse_telescope_ksp_rtol;
  app->resolved_profile.linear.pmg.coarse_ksp_max_it = app->pmg_coarse_telescope_ksp_max_it;
  app->resolved_profile.linear.pmg.p2_telescope_active_ranks = app->pmg_p2_telescope_active_ranks;
  app->resolved_profile.linear.pmg.p2_telescope_ksp_rtol = app->pmg_p2_telescope_ksp_rtol;
  app->resolved_profile.linear.pmg.p2_telescope_ksp_max_it = app->pmg_p2_telescope_ksp_max_it;
  app->resolved_profile.linear.pmg.coarse_redundant_group_size = app->pmg_coarse_redundant_group_size;
  app->resolved_profile.linear.pmg.coarse_gamg_aggressive_square_graph = app->pmg_coarse_gamg_aggressive_square_graph;
  app->resolved_profile.linear.pmg.check_coarse_transfers = app->pmg_check_coarse_transfers;
  app->resolved_profile.linear.deflation.enabled = app->use_deflation;
  PetscCall(PetscStrncpy(app->resolved_profile.linear.deflation.solver, app->deflation_solver_name, sizeof(app->resolved_profile.linear.deflation.solver)));
  PetscCall(PetscStrncpy(app->resolved_profile.linear.deflation.projector, app->deflation_projector_name, sizeof(app->resolved_profile.linear.deflation.projector)));
  app->resolved_profile.linear.deflation.basis_tol = app->deflation_basis_tol;
  app->resolved_profile.linear.deflation.max_it = app->deflation_max_it;
  app->resolved_profile.linear.deflation.max_vectors = app->deflation_max_vectors;
  app->resolved_profile.linear.deflation.intra_newton_recycle = app->deflation_intra_newton_recycle;
  app->resolved_profile.linear.deflation.krylov_persistent = app->deflation_krylov_persistent;

  app->resolved_profile.output.output_dir[0] = '\0';
  PetscCall(PetscStrncpy(app->resolved_profile.output.native_problem_manifest, app->native_problem_manifest, sizeof(app->resolved_profile.output.native_problem_manifest)));
  PetscCall(PetscStrncpy(app->resolved_profile.output.mechanics_bc_labels_csv, app->mechanics_bc_labels_csv, sizeof(app->resolved_profile.output.mechanics_bc_labels_csv)));
  PetscCall(PetscStrncpy(app->resolved_profile.output.mechanics_neumann_labels_csv, app->mechanics_neumann_labels_csv, sizeof(app->resolved_profile.output.mechanics_neumann_labels_csv)));
  PetscCall(PetscStrncpy(app->resolved_profile.output.seepage_boundary_labels_csv, app->seepage_boundary_labels_csv, sizeof(app->resolved_profile.output.seepage_boundary_labels_csv)));
  PetscCall(PetscStrncpy(app->resolved_profile.output.curve_csv, app->curve_csv, sizeof(app->resolved_profile.output.curve_csv)));
  PetscCall(PetscStrncpy(app->resolved_profile.output.summary_json, app->summary_json, sizeof(app->resolved_profile.output.summary_json)));
  app->resolved_profile.output.write_solution = (PetscBool)(app->solution_binary[0] || app->solution_vtk[0] || app->solution_points_csv[0]);
  PetscCall(PetscOptionsHasName(NULL, NULL, "-log_view", &app->resolved_profile.output.write_log_view));
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

static void ResolveDefaultPmgShellActiveRanks(PetscMPIInt comm_size, PetscInt *p2_active_ranks, PetscInt *p1_active_ranks)
{
  PetscInt world = (PetscInt)comm_size;
  PetscInt cap   = world < 64 ? world : 64;
  PetscInt half  = world > 1 ? world / 2 : 1;

  if (cap < 1) cap = 1;
  *p2_active_ranks = cap;
  *p1_active_ranks = half < cap ? half : cap;
  if (*p1_active_ranks < 1) *p1_active_ranks = 1;
}

static PetscErrorCode ParseOptions(MPI_Comm comm, AppCtx *app)
{
  PetscBool   flg, pmg_shell_p2_active_ranks_set = PETSC_FALSE, pmg_shell_p1_active_ranks_set = PETSC_FALSE;
  PetscMPIInt comm_size = 1;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &comm_size));
  PetscCall(PetscStrncpy(app->mesh, "meshes/3d_hetero_slope/adaptive_family_a_l1.msh", sizeof(app->mesh)));
  app->refine_levels  = 0;
  app->newton_rtol    = 1.0e-4;
  app->newton_max_it  = 200;
  app->omega_max      = 6.7e6;
  PetscCall(PetscStrncpy(app->analysis, "ssr", sizeof(app->analysis)));
  app->lambda_init    = 1.0;
  app->d_lambda_init  = 0.1;
  app->d_lambda_min   = 1.0e-3;
  app->d_lambda_diff_scaled_min = 1.0e-3;
  app->lambda_ell              = 1.0;
  app->d_t_min                 = 1.0e-3;
  app->d_omega_ini_scale       = 0.2;
  app->continuation_step_max    = 100;
  app->newton_stopping_tol      = 1.0e-4;
  app->init_newton_stopping_tol = 1.0e-3;
  app->it_damp_max   = 10;
  app->r_min         = 1.0e-4;
  PetscCall(PetscStrncpy(app->newton_stopping_criterion, "absolute_delta_lambda", sizeof(app->newton_stopping_criterion)));
  PetscCall(PetscStrncpy(app->init_newton_stopping_criterion, "relative_correction", sizeof(app->init_newton_stopping_criterion)));
  PetscCall(PetscStrncpy(app->continuation_algorithm, "auto", sizeof(app->continuation_algorithm)));
  PetscCall(PetscStrncpy(app->continuation_method, "indirect", sizeof(app->continuation_method)));
  PetscCall(PetscStrncpy(app->newton_algorithm, "auto", sizeof(app->newton_algorithm)));
  PetscCall(PetscStrncpy(app->linear_algorithm, "auto", sizeof(app->linear_algorithm)));
  PetscCall(PetscStrncpy(app->continuation_predictor, "secant", sizeof(app->continuation_predictor)));
  PetscCall(PetscStrncpy(app->omega_step_controller, "classic", sizeof(app->omega_step_controller)));
  PetscCall(PetscStrncpy(app->curve_csv, "continuation_curve.csv", sizeof(app->curve_csv)));
  app->summary_json[0] = '\0';
  app->solution_binary[0] = '\0';
  app->solution_vtk[0] = '\0';
  app->solution_points_csv[0] = '\0';
  app->native_problem_manifest[0] = '\0';
  app->mechanics_bc_labels_csv[0] = '\0';
  app->mechanics_neumann_labels_csv[0] = '\0';
  app->seepage_boundary_labels_csv[0] = '\0';
  app->seepage_pressure_source[0] = '\0';
  app->seepage_pressure_csv[0] = '\0';
  app->mechanics_bc_nodes_csv[0] = '\0';
  app->debug_coordinate_bc_table = PETSC_FALSE;
  app->native_problem_manifest_loaded = PETSC_FALSE;
  app->native_manifest_mechanics_dirichlet = 0;
  app->native_manifest_mechanics_neumann = 0;
  app->native_manifest_seepage_head = 0;
  app->native_manifest_seepage_flux = 0;
  app->seepage_grho = 9.81;
  app->element_degree = 4;
  app->mechanics_dim  = 3;
  app->ksp_rtol       = 1.0e-1;
  app->damping_min    = 1.0e-3;
  app->line_search    = PETSC_TRUE;
  app->use_box_mesh    = PETSC_FALSE;
  app->variant        = VARIANT_PMG;
  PetscCall(PetscStrncpy(app->variant_name, "pmg", sizeof(app->variant_name)));
  PetscCall(PetscStrncpy(app->requested_variant_name, app->variant_name, sizeof(app->requested_variant_name)));
  app->variant_fallback_reason[0] = '\0';
  PetscCall(PetscStrncpy(app->mesh_bc_mode, "rollers", sizeof(app->mesh_bc_mode)));
  app->bc_tag_base  = 5;
  app->bc_tag_x_min = 2;
  app->bc_tag_x_max = 1;
  app->bc_tag_z_min = 3;
  app->bc_tag_z_max = 4;
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
  app->pmg_shell_p2_active_ranks                = 0;
  app->pmg_shell_p1_active_ranks                = 0;
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
  PetscCall(PetscOptionsString("-analysis", "Mechanics analysis label: ssr|ll", NULL, app->analysis, app->analysis, sizeof(app->analysis), NULL));
  PetscCall(PetscOptionsReal("-omega_max", "Maximum continuation work omega=f_ext^T u", NULL, app->omega_max, &app->omega_max, NULL));
  PetscCall(PetscOptionsReal("-lambda_init", "Initial strength reduction factor seed", NULL, app->lambda_init, &app->lambda_init, NULL));
  PetscCall(PetscOptionsReal("-d_lambda_init", "Initial lambda increment for continuation seeding", NULL, app->d_lambda_init, &app->d_lambda_init, NULL));
  PetscCall(PetscOptionsReal("-d_lambda_min", "Minimum lambda increment during initialization backoff", NULL, app->d_lambda_min, &app->d_lambda_min, NULL));
  PetscCall(PetscOptionsReal("-d_lambda_diff_scaled_min", "Stop when scaled lambda slope difference drops below this value", NULL, app->d_lambda_diff_scaled_min, &app->d_lambda_diff_scaled_min, NULL));
  PetscCall(PetscOptionsReal("-lambda_ell", "Limit-load strength reduction fixed during load continuation", NULL, app->lambda_ell, &app->lambda_ell, NULL));
  PetscCall(PetscOptionsReal("-d_t_min", "Limit-load stop threshold for load-factor increment", NULL, app->d_t_min, &app->d_t_min, NULL));
  PetscCall(PetscOptionsReal("-d_omega_ini_scale", "Limit-load elastic omega predictor scale", NULL, app->d_omega_ini_scale, &app->d_omega_ini_scale, NULL));
  PetscCall(PetscOptionsInt("-continuation_step_max", "Maximum accepted continuation points including initialization", NULL, app->continuation_step_max, &app->continuation_step_max, NULL));
  PetscCall(PetscOptionsString("-continuation_algorithm", "SSR continuation algorithm selected by the resolved profile; auto follows -continuation_method", NULL, app->continuation_algorithm, app->continuation_algorithm, sizeof(app->continuation_algorithm), NULL));
  PetscCall(PetscOptionsString("-continuation_method", "SSR continuation method: indirect|direct", NULL, app->continuation_method, app->continuation_method, sizeof(app->continuation_method), NULL));
  PetscCall(PetscOptionsString("-continuation_predictor", "Continuation predictor, currently only secant", NULL, app->continuation_predictor, app->continuation_predictor, sizeof(app->continuation_predictor), NULL));
  PetscCall(PetscOptionsString("-omega_step_controller", "Omega step controller policy, currently classic", NULL, app->omega_step_controller, app->omega_step_controller, sizeof(app->omega_step_controller), NULL));
  PetscCall(PetscOptionsString("-curve_csv", "Rank-0 CSV output path for accepted continuation curve", NULL, app->curve_csv, app->curve_csv, sizeof(app->curve_csv), NULL));
  PetscCall(PetscOptionsString("-summary_json", "Rank-0 JSON output path for C-hotpath run summary", NULL, app->summary_json, app->summary_json, sizeof(app->summary_json), NULL));
  PetscCall(PetscOptionsString("-solution_binary", "PETSc binary output path for the final displacement Vec", NULL, app->solution_binary, app->solution_binary, sizeof(app->solution_binary), NULL));
  PetscCall(PetscOptionsString("-solution_vtk", "VTK/VTU output path for the final DMPlex displacement field", NULL, app->solution_vtk, app->solution_vtk, sizeof(app->solution_vtk), NULL));
  PetscCall(PetscOptionsString("-solution_points_csv", "Coordinate/displacement CSV for case-TOMLible postprocessing", NULL, app->solution_points_csv, app->solution_points_csv, sizeof(app->solution_points_csv), NULL));
  PetscCall(PetscOptionsString("-native_problem_manifest", "Resolved asset/support manifest for future DMPlex-native BC ingestion", NULL, app->native_problem_manifest, app->native_problem_manifest, sizeof(app->native_problem_manifest), NULL));
  PetscCall(PetscOptionsString("-mechanics_bc_labels_csv", "Coordinate-free DMPlex label/tag constraint table for future native BC ingestion", NULL, app->mechanics_bc_labels_csv, app->mechanics_bc_labels_csv, sizeof(app->mechanics_bc_labels_csv), NULL));
  PetscCall(PetscOptionsString("-mechanics_neumann_labels_csv", "Coordinate-free DMPlex label/tag mechanics Neumann table; currently fail-explicit pending face quadrature", NULL, app->mechanics_neumann_labels_csv, app->mechanics_neumann_labels_csv, sizeof(app->mechanics_neumann_labels_csv), NULL));
  PetscCall(PetscOptionsString("-seepage_boundary_labels_csv", "Coordinate-free DMPlex label/tag seepage boundary table; current mechanics coupling still uses pressure CSV", NULL, app->seepage_boundary_labels_csv, app->seepage_boundary_labels_csv, sizeof(app->seepage_boundary_labels_csv), NULL));
  PetscCall(PetscOptionsString("-seepage_pressure_source", "Source contract for -seepage_pressure_csv, currently hydro_prepass_coordinate_bridge", NULL, app->seepage_pressure_source, app->seepage_pressure_source, sizeof(app->seepage_pressure_source), NULL));
  PetscCall(PetscOptionsString("-seepage_pressure_csv", "Coordinate-keyed pressure CSV for seepage-coupled mechanics external load", NULL, app->seepage_pressure_csv, app->seepage_pressure_csv, sizeof(app->seepage_pressure_csv), NULL));
  PetscCall(PetscOptionsBool("-debug_coordinate_bc_table", "Allow debug coordinate/component mechanics constraint fallback", NULL, app->debug_coordinate_bc_table, &app->debug_coordinate_bc_table, NULL));
  PetscCall(PetscOptionsString("-mechanics_bc_nodes_csv", "Debug coordinate/component CSV for mechanics constraints; requires -debug_coordinate_bc_table true", NULL, app->mechanics_bc_nodes_csv, app->mechanics_bc_nodes_csv, sizeof(app->mechanics_bc_nodes_csv), NULL));
  PetscCall(PetscOptionsReal("-seepage_grho", "Water unit weight used for seepage saturation threshold", NULL, app->seepage_grho, &app->seepage_grho, NULL));
  PetscCall(PetscOptionsInt("-element_degree", "Mechanics FE degree: 1, 2, or 4", NULL, app->element_degree, &app->element_degree, NULL));
  PetscCall(PetscOptionsInt("-fe_degree", "Alias for -element_degree", NULL, app->element_degree, &app->element_degree, NULL));
  PetscCall(PetscOptionsInt("-mechanics_dim", "Mechanics spatial dimension: 2 triangles or 3 tetrahedra", NULL, app->mechanics_dim, &app->mechanics_dim, NULL));
  PetscCall(PetscOptionsInt("-spatial_dim", "Alias for -mechanics_dim", NULL, app->mechanics_dim, &app->mechanics_dim, NULL));
  PetscCall(PetscOptionsReal("-newton_rtol", "Relative residual tolerance used for nonlinear residual safeguards", NULL, app->newton_rtol, &app->newton_rtol, NULL));
  PetscCall(PetscOptionsInt("-newton_max_it", "Maximum fixed-lambda or indirect Newton iterations", NULL, app->newton_max_it, &app->newton_max_it, NULL));
  PetscCall(PetscOptionsString("-newton_algorithm", "SSR Newton algorithm selected by the resolved profile; auto follows the continuation method", NULL, app->newton_algorithm, app->newton_algorithm, sizeof(app->newton_algorithm), NULL));
  PetscCall(PetscOptionsString("-newton_stopping_criterion", "Indirect Newton stop: absolute_delta_lambda|relative_residual|relative_correction", NULL, app->newton_stopping_criterion, app->newton_stopping_criterion, sizeof(app->newton_stopping_criterion), NULL));
  PetscCall(PetscOptionsReal("-newton_stopping_tol", "Indirect Newton stopping tolerance for the selected criterion", NULL, app->newton_stopping_tol, &app->newton_stopping_tol, NULL));
  PetscCall(PetscOptionsString("-init_newton_stopping_criterion", "Initialization Newton stop: absolute_delta_lambda|relative_residual|relative_correction", NULL, app->init_newton_stopping_criterion, app->init_newton_stopping_criterion, sizeof(app->init_newton_stopping_criterion), NULL));
  PetscCall(PetscOptionsReal("-init_newton_stopping_tol", "Initialization Newton stopping tolerance for the selected criterion", NULL, app->init_newton_stopping_tol, &app->init_newton_stopping_tol, NULL));
  PetscCall(PetscOptionsInt("-it_damp_max", "Maximum ALG5/backtracking halvings per Newton iteration", NULL, app->it_damp_max, &app->it_damp_max, NULL));
  PetscCall(PetscOptionsReal("-r_min", "Minimum regularization weight in K_r=r*K_elastic+(1-r)*K_tangent", NULL, app->r_min, &app->r_min, NULL));
  PetscCall(PetscOptionsReal("-linear_rtol", "Default KSP relative tolerance", NULL, app->ksp_rtol, &app->ksp_rtol, NULL));
  PetscCall(PetscOptionsBool("-line_search", "Use residual backtracking", NULL, app->line_search, &app->line_search, NULL));
  PetscCall(PetscOptionsString("-linear_algorithm", "SSR linear algorithm selected by the resolved profile; auto follows -pc_variant and -deflation", NULL, app->linear_algorithm, app->linear_algorithm, sizeof(app->linear_algorithm), NULL));
  PetscCall(PetscOptionsBool("-use_box_mesh", "Use a tiny generated unit-box tetra mesh for smoke tests", NULL, app->use_box_mesh, &app->use_box_mesh, NULL));
  PetscCall(PetscOptionsString("-mesh_bc_mode", "Boundary mode for imported meshes: rollers|base_only|full_sides", NULL, app->mesh_bc_mode, app->mesh_bc_mode, sizeof(app->mesh_bc_mode), NULL));
  PetscCall(PetscOptionsInt("-bc_tag_base", "Face Sets/boundary_marker tag for base boundary", NULL, app->bc_tag_base, &app->bc_tag_base, NULL));
  PetscCall(PetscOptionsInt("-bc_tag_x_min", "Face Sets/boundary_marker tag for x-min roller boundary; <=0 disables", NULL, app->bc_tag_x_min, &app->bc_tag_x_min, NULL));
  PetscCall(PetscOptionsInt("-bc_tag_x_max", "Face Sets/boundary_marker tag for x-max roller boundary; <=0 disables", NULL, app->bc_tag_x_max, &app->bc_tag_x_max, NULL));
  PetscCall(PetscOptionsInt("-bc_tag_z_min", "Face Sets/boundary_marker tag for z-min roller boundary; <=0 disables", NULL, app->bc_tag_z_min, &app->bc_tag_z_min, NULL));
  PetscCall(PetscOptionsInt("-bc_tag_z_max", "Face Sets/boundary_marker tag for z-max roller boundary; <=0 disables", NULL, app->bc_tag_z_max, &app->bc_tag_z_max, NULL));
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
  PetscCall(PetscOptionsInt("-pmg_shell_p2_active_ranks", "Active MPI ranks for the shell V-cycle P2 layout; 0 or >= ranks keeps all ranks active", NULL, app->pmg_shell_p2_active_ranks, &app->pmg_shell_p2_active_ranks, &pmg_shell_p2_active_ranks_set));
  PetscCall(PetscOptionsInt("-pmg_shell_p1_active_ranks", "Active MPI ranks for the shell V-cycle P1 layout; 0 or >= ranks keeps all ranks active", NULL, app->pmg_shell_p1_active_ranks, &app->pmg_shell_p1_active_ranks, &pmg_shell_p1_active_ranks_set));
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

  if (!pmg_shell_p2_active_ranks_set || !pmg_shell_p1_active_ranks_set) {
    PetscInt default_p2_active_ranks = 0, default_p1_active_ranks = 0;
    ResolveDefaultPmgShellActiveRanks(comm_size, &default_p2_active_ranks, &default_p1_active_ranks);
    if (!pmg_shell_p2_active_ranks_set) app->pmg_shell_p2_active_ranks = default_p2_active_ranks;
    if (!pmg_shell_p1_active_ranks_set) app->pmg_shell_p1_active_ranks = default_p1_active_ranks;
  }

  PetscCall(PetscStrncpy(app->requested_variant_name, app->variant_name, sizeof(app->requested_variant_name)));
  app->variant_fallback_reason[0] = '\0';
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
  PetscCheck(app->element_degree == 1 || app->element_degree == 2 || app->element_degree == 4, comm, PETSC_ERR_ARG_OUTOFRANGE,
             "-element_degree must be 1, 2, or 4");
  PetscCheck(app->mechanics_dim == 2 || app->mechanics_dim == 3, comm, PETSC_ERR_ARG_OUTOFRANGE, "-mechanics_dim must be 2 or 3");
  PetscCheck(app->mechanics_dim == 3 || (app->variant != VARIANT_BDDC && app->variant != VARIANT_FETIDP), comm, PETSC_ERR_SUP,
             "2D mechanics currently supports pc_variant gamg, pmg, or none; BDDC/FETI-DP helpers remain 3D-only");
  if (app->element_degree == 1 && app->variant == VARIANT_PMG) {
    app->variant = VARIANT_GAMG;
    PetscCall(PetscStrncpy(app->variant_name, "gamg", sizeof(app->variant_name)));
    PetscCall(PetscStrncpy(app->variant_fallback_reason, "p1_has_no_p_hierarchy", sizeof(app->variant_fallback_reason)));
    PetscCall(PetscPrintf(comm, "FE_DEGREE_CONFIG degree=1 requested_pc_variant=pmg fallback_pc_variant=gamg reason=p1_has_no_p_hierarchy\n"));
  }
  PetscCheck(app->newton_rtol > 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-newton_rtol must be positive");
  PetscCheck(app->newton_stopping_tol > 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-newton_stopping_tol must be positive");
  PetscCheck(app->init_newton_stopping_tol > 0.0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-init_newton_stopping_tol must be positive");
  PetscCheck(app->it_damp_max >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-it_damp_max must be nonnegative");
  PetscCheck(app->r_min > 0.0 && app->r_min <= 1.0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-r_min must be in (0,1]");
  PetscCall(PetscStrcasecmp(app->continuation_method, "indirect", &flg));
  if (!flg) {
    PetscCall(PetscStrcasecmp(app->continuation_method, "direct", &flg));
    PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-continuation_method must be indirect or direct");
  }
  PetscCall(AppCtxNormalizeAlgorithmSelectors(comm, app));
  PetscCall(ValidateAlgorithmRegistrySelection(comm, app));
  PetscCall(PetscStrcasecmp(app->continuation_predictor, "secant", &flg));
  PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-continuation_predictor currently supports only secant");
  PetscCall(PetscStrcasecmp(app->omega_step_controller, "classic", &flg));
  if (!flg) {
    PetscCall(PetscStrcasecmp(app->omega_step_controller, "legacy", &flg));
    if (flg) PetscCall(PetscStrncpy(app->omega_step_controller, "classic", sizeof(app->omega_step_controller)));
    PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-omega_step_controller currently supports only classic");
  }
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
      if (!flg) {
        PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "fixed_base_rollers", &flg));
        if (!flg) PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "2d_rollers", &flg));
        PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-mesh_bc_mode must be rollers, base_only, full_sides, fixed_base_rollers, or 2d_rollers");
      }
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
  PetscCall(AppCtxSyncResolvedProfile(app));
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
  {
    PetscInt check_tags[5] = {app->bc_tag_base, app->bc_tag_x_min, app->bc_tag_x_max, app->bc_tag_z_min, app->bc_tag_z_max};
    PetscInt check_counts[5];

    for (PetscInt k = 0; k < 5; ++k) {
      IS       points = NULL;
      PetscInt nloc = 0;

      check_counts[k] = 0;
      if (check_tags[k] <= 0) continue;
      PetscCall(DMLabelGetStratumIS(label, check_tags[k], &points));
      if (points) PetscCall(ISGetLocalSize(points, &nloc));
      PetscCallMPI(MPI_Allreduce(&nloc, &check_counts[k], 1, MPIU_INT, MPI_SUM, comm));
      PetscCall(ISDestroy(&points));
    }
    PetscCall(PetscPrintf(comm,
                          "BC_TAG_CONFIG base=%" PetscInt_FMT " base_points=%" PetscInt_FMT " x_min=%" PetscInt_FMT " x_min_points=%" PetscInt_FMT " x_max=%" PetscInt_FMT " x_max_points=%" PetscInt_FMT " z_min=%" PetscInt_FMT " z_min_points=%" PetscInt_FMT " z_max=%" PetscInt_FMT " z_max_points=%" PetscInt_FMT "\n",
                          app->bc_tag_base, check_counts[0], app->bc_tag_x_min, check_counts[1], app->bc_tag_x_max, check_counts[2],
                          app->bc_tag_z_min, check_counts[3], app->bc_tag_z_max, check_counts[4]));
    PetscCheck(app->bc_tag_base > 0 && check_counts[0] > 0, comm, PETSC_ERR_ARG_WRONGSTATE,
               "Plasticity boundary label has no points for configured base tag %" PetscInt_FMT, app->bc_tag_base);
  }
  if (!app->use_box_mesh) {
    PetscBool is_rollers, is_full_sides, is_fixed_base_rollers;

    PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "rollers", &is_rollers));
    PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "full_sides", &is_full_sides));
    PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "fixed_base_rollers", &is_fixed_base_rollers));
    if (app->mechanics_dim == 3 && (is_rollers || is_full_sides || is_fixed_base_rollers) &&
        app->bc_tag_x_min == 2 && app->bc_tag_x_max == 1 && app->bc_tag_z_min == 3 && app->bc_tag_z_max == 4) {
      PetscCheck(counts[0] > 0 && counts[1] > 0 && counts[2] > 0 && counts[3] > 0, comm, PETSC_ERR_ARG_WRONGSTATE,
                 "Plasticity boundary label is missing at least one side group: x_max=%" PetscInt_FMT " x_min=%" PetscInt_FMT " z_min=%" PetscInt_FMT " z_max=%" PetscInt_FMT,
                 counts[0], counts[1], counts[2], counts[3]);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscBool MechanicsLabelConstraintTableActive(const AppCtx *app)
{
  if (!app || !app->mechanics_bc_labels_csv[0]) return PETSC_FALSE;
  if (app->native_problem_manifest_loaded && app->native_manifest_mechanics_dirichlet <= 0) return PETSC_FALSE;
  return PETSC_TRUE;
}

static PetscErrorCode AddPlasticityBoundaryConditions(DM dm, const AppCtx *app)
{
  DMLabel        label = NULL;
  const PetscInt components3[3] = {0, 1, 2};
  const PetscInt components2[2] = {0, 1};
  const PetscInt base = app->bc_tag_base;
  PetscInt       x_rollers[2], z_rollers[2], nx = 0, nz = 0;
  const PetscInt x_comp[1] = {0}, y_comp[1] = {1}, z_comp[1] = {2};
  PetscBool      is_rollers, is_base_only, is_full_sides, is_fixed_base_rollers, is_2d_rollers;
  PetscBool      label_table_active;

  PetscFunctionBeginUser;
  label_table_active = MechanicsLabelConstraintTableActive(app);
  if (label_table_active) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                          "MECHANICS_BC_MODE source=label_table path=%s legacy_mode=%s status=petsc_section_boundaries_enabled\n",
                          app->mechanics_bc_labels_csv, app->mesh_bc_mode));
    PetscCall(AssemblyApplyLabelBoundaryConditions(dm, app->mechanics_bc_labels_csv,
                                                   app->native_problem_manifest_loaded ? app->native_manifest_mechanics_dirichlet : -1));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(DMGetLabel(dm, "boundary_marker", &label));
  PetscCheck(label, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Missing boundary_marker label");
  PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "rollers", &is_rollers));
  PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "base_only", &is_base_only));
  PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "full_sides", &is_full_sides));
  PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "fixed_base_rollers", &is_fixed_base_rollers));
  PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "2d_rollers", &is_2d_rollers));
  if (app->bc_tag_x_min > 0) x_rollers[nx++] = app->bc_tag_x_min;
  if (app->bc_tag_x_max > 0) x_rollers[nx++] = app->bc_tag_x_max;
  if (app->bc_tag_z_min > 0) z_rollers[nz++] = app->bc_tag_z_min;
  if (app->bc_tag_z_max > 0) z_rollers[nz++] = app->bc_tag_z_max;
  if (app->mechanics_dim == 2) {
    if (is_full_sides || is_fixed_base_rollers || is_2d_rollers) {
      PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "fixed_base", label, 1, &base, 0, 2, components2, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
    } else {
      PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "base_roller_y", label, 1, &base, 0, 1, y_comp, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
    }
    if (is_rollers || is_fixed_base_rollers || is_2d_rollers || app->use_box_mesh) {
      if (nx > 0) PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "x_side_rollers", label, nx, x_rollers, 0, 1, x_comp, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
    } else if (is_full_sides) {
      if (nx > 0) PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "x_side_clamps", label, nx, x_rollers, 0, 2, components2, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
    } else {
      PetscCheck(is_base_only, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown -mesh_bc_mode %s for 2D", app->mesh_bc_mode);
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (is_full_sides || is_fixed_base_rollers) {
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "glued_base", label, 1, &base, 0, 3, components3, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
  } else {
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "base_roller_y", label, 1, &base, 0, 1, y_comp, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
  }
  if (app->use_box_mesh || is_rollers || is_fixed_base_rollers) {
    if (nx > 0) PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "x_side_rollers", label, nx, x_rollers, 0, 1, x_comp, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
    if (nz > 0) PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "z_side_rollers", label, nz, z_rollers, 0, 1, z_comp, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
  } else if (is_full_sides) {
    if (nx > 0) PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "x_side_clamps", label, nx, x_rollers, 0, 3, components3, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
    if (nz > 0) PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "z_side_clamps", label, nz, z_rollers, 0, 3, components3, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
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
    PetscInt        fStart, fEnd;
    DMLabel         faceSets;
    if (app->mechanics_dim == 2) {
      const PetscInt  cells[2][3] = {{0, 1, 2}, {0, 2, 3}};
      const PetscReal coords[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
      PetscCall(DMPlexCreateFromCellListPetsc(comm, 2, 2, 4, 3, PETSC_TRUE, (const PetscInt *)cells, 2, (const PetscReal *)coords, &cur));
    } else {
      const PetscInt  cells[6][4] = {{0, 1, 2, 6}, {0, 2, 3, 6}, {0, 3, 7, 6}, {0, 7, 4, 6}, {0, 4, 5, 6}, {0, 5, 1, 6}};
      const PetscReal coords[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
      PetscCall(DMPlexCreateFromCellListPetsc(comm, 3, 6, 8, 4, PETSC_TRUE, (const PetscInt *)cells, 3, (const PetscReal *)coords, &cur));
    }
    PetscCall(DMCreateLabel(cur, "Cell Sets"));
    {
      PetscInt cStart, cEnd;
      PetscCall(DMPlexGetHeightStratum(cur, 0, &cStart, &cEnd));
      for (PetscInt c = cStart; c < cEnd; ++c) PetscCall(DMSetLabelValue(cur, "Cell Sets", c, 1));
    }
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
      else if (app->mechanics_dim == 3 && PetscAbsReal(centroid[2]) < 1.0e-10) PetscCall(DMLabelSetValue(faceSets, f, 3));       /* z_min */
      else if (app->mechanics_dim == 3 && PetscAbsReal(centroid[2] - 1.0) < 1.0e-10) PetscCall(DMLabelSetValue(faceSets, f, 4)); /* z_max */
      else if (app->mechanics_dim == 2 && PetscAbsReal(centroid[1] - 1.0) < 1.0e-10) PetscCall(DMLabelSetValue(faceSets, f, 6)); /* y_max */
    }
  } else {
    PetscBool label_table_active = MechanicsLabelConstraintTableActive(app);
    if (label_table_active) {
      PetscBool user_set_mark_vertices = PETSC_FALSE;

      PetscCall(PetscOptionsHasName(NULL, NULL, "-dm_plex_gmsh_mark_vertices", &user_set_mark_vertices));
      if (!user_set_mark_vertices) PetscCall(PetscOptionsSetValue(NULL, "-dm_plex_gmsh_mark_vertices", "true"));
      PetscCall(PetscPrintf(comm,
                            "MECHANICS_BC_LABELS_VERTEX_MARKING enabled=true source=label_table option=dm_plex_gmsh_mark_vertices user_set=%s\n",
                            user_set_mark_vertices ? "true" : "false"));
    }
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
  {
    PetscInt mesh_dim;
    PetscCall(DMGetDimension(cur, &mesh_dim));
    PetscCheck(mesh_dim == app->mechanics_dim, comm, PETSC_ERR_ARG_INCOMP,
               "Mesh dimension %" PetscInt_FMT " does not match -mechanics_dim %" PetscInt_FMT, mesh_dim, app->mechanics_dim);
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
  PetscInt             constraints_local = 0, constraints_global = 0, section_constraints_local = 0, section_constraints_global = 0;
  PetscBool            section_constraints_active = PETSC_FALSE;
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
  PetscCall(AssemblyGetSectionConstraintStats(dm, &section_constraints_local, &section_constraints_global, &section_constraints_active));
  constraints_local = actx->n_constrained_local;
  constraints_global = actx->n_constrained_all;
  if (section_constraints_active && constraints_global == 0) {
    constraints_local = section_constraints_local;
    constraints_global = section_constraints_global;
  }

  PetscCallMPI(MPI_Allreduce(&local_cells, &cells_sum, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&local_cells, &cells_min, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPI_Allreduce(&local_cells, &cells_max, 1, MPIU_INT, MPI_MAX, comm));
  PetscCallMPI(MPI_Allreduce(&local_vertices, &vertices_sum, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&local_vertices, &vertices_min, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPI_Allreduce(&local_vertices, &vertices_max, 1, MPIU_INT, MPI_MAX, comm));
  PetscCallMPI(MPI_Allreduce(&local_owned_rows, &owned_min, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPI_Allreduce(&local_owned_rows, &owned_max, 1, MPIU_INT, MPI_MAX, comm));
  PetscCallMPI(MPI_Allreduce(&constraints_local, &constraints_sum, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&constraints_local, &constraints_min, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPI_Allreduce(&constraints_local, &constraints_max, 1, MPIU_INT, MPI_MAX, comm));

  PetscCall(PetscPrintf(comm,
                        "PARTITION_RESULT kind=dm partitioner=%s ranks=%d variant=%s mesh=%s%s refine_levels=%" PetscInt_FMT " cells_sum=%" PetscInt_FMT " cells_min=%" PetscInt_FMT " cells_max=%" PetscInt_FMT " cell_imbalance=%.6g vertices_sum=%" PetscInt_FMT " vertices_min=%" PetscInt_FMT " vertices_max=%" PetscInt_FMT " owned_rows_min=%" PetscInt_FMT " owned_rows_max=%" PetscInt_FMT " global_dofs=%" PetscInt_FMT " owned_constraints_sum=%" PetscInt_FMT " owned_constraints_min=%" PetscInt_FMT " owned_constraints_max=%" PetscInt_FMT " global_constraints=%" PetscInt_FMT "\n",
                        part_type ? part_type : "unknown", size, app->variant_name, app->use_box_mesh ? "generated-box:" : "", app->use_box_mesh ? "unit" : app->mesh, app->refine_levels,
                        cells_sum, cells_min, cells_max, cells_sum ? ((double)cells_max * (double)size) / (double)cells_sum : 0.0, vertices_sum, vertices_min, vertices_max, owned_min, owned_max, global_rows,
                        constraints_sum, constraints_min, constraints_max, constraints_global));

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
