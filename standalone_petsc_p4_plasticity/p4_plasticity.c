#include "assembly.h"
#include "p4_basis.h"

#include <petscdmplex.h>
#include <petscksp.h>
#include <stdlib.h>

typedef enum {
  VARIANT_GAMG,
  VARIANT_BDDC,
  VARIANT_FETIDP,
  VARIANT_PMG,
  VARIANT_NONE
} PCVariant;

typedef enum {
  DEFLATION_SOLVER_FGMRES,
  DEFLATION_SOLVER_CG
} DeflationSolverType;

typedef struct {
  char      mesh[PETSC_MAX_PATH_LEN];
  PetscReal lambda;
  PetscInt  refine_levels;
  PetscReal newton_rtol;
  PetscInt  newton_max_it;
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
  char      pmg_coarse_telescope_mode[32];
  PetscReal pmg_coarse_telescope_ksp_rtol;
  PetscReal pmg_p2_telescope_ksp_rtol;
  PetscInt  pmg_coarse_lu_max_dofs;
  PetscInt  pmg_smoother_max_it;
  PetscInt  pmg_coarse_redundant_group_size;
  PetscInt  pmg_coarse_telescope_active_ranks;
  PetscInt  pmg_coarse_telescope_ksp_max_it;
  PetscInt  pmg_p2_telescope_active_ranks;
  PetscInt  pmg_p2_telescope_ksp_max_it;
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
  char      deflation_solver_name[32];
  DeflationSolverType deflation_solver;
  PetscReal deflation_basis_tol;
  PetscInt  deflation_max_it;
  PetscInt  deflation_max_vectors;
  PetscBool deflation_monitor;
} AppCtx;

typedef struct {
  PetscReal      final_rel;
  PetscInt       newton_its;
  PetscInt       total_linear_its;
  PetscLogDouble assembly_time;
  PetscLogDouble solve_time;
} NewtonStats;

typedef struct {
  DM           dm;
  AssemblyCtx *actx;
  AppCtx      *app;
  Mat          A;
  KSP          ksp;
  PetscBool    reuse;
  Vec         *raw_basis;
  Vec         *orth_basis;
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
  app->lambda         = 1.2;
  app->refine_levels  = 0;
  app->newton_rtol    = 1.0e-4;
  app->newton_max_it  = 20;
  app->ksp_rtol       = 1.0e-8;
  app->damping_min    = 1.0e-3;
  app->line_search    = PETSC_TRUE;
  app->use_box_mesh    = PETSC_FALSE;
  app->variant        = VARIANT_GAMG;
  PetscCall(PetscStrncpy(app->variant_name, "gamg", sizeof(app->variant_name)));
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
  PetscCall(PetscStrncpy(app->pmg_coarse_telescope_mode, "default", sizeof(app->pmg_coarse_telescope_mode)));
  app->pmg_coarse_telescope_ksp_rtol   = 1.0e-3;
  app->pmg_p2_telescope_ksp_rtol       = 1.0e-3;
  app->pmg_coarse_lu_max_dofs = 50000;
  app->pmg_smoother_max_it    = 2;
  app->pmg_coarse_redundant_group_size         = 16;
  app->pmg_coarse_telescope_active_ranks       = 0;
  app->pmg_coarse_telescope_ksp_max_it         = 100;
  app->pmg_p2_telescope_active_ranks           = 0;
  app->pmg_p2_telescope_ksp_max_it             = 50;
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
  app->use_deflation             = PETSC_FALSE;
  PetscCall(PetscStrncpy(app->deflation_solver_name, "fgmres", sizeof(app->deflation_solver_name)));
  app->deflation_solver      = DEFLATION_SOLVER_FGMRES;
  app->deflation_basis_tol   = 1.0e-3;
  app->deflation_max_it      = 0;
  app->deflation_max_vectors = 0;
  app->deflation_monitor     = PETSC_FALSE;

  PetscOptionsBegin(comm, NULL, "Standalone P4 plasticity options", NULL);
  PetscCall(PetscOptionsString("-mesh", "Gmsh mesh path", NULL, app->mesh, app->mesh, sizeof(app->mesh), NULL));
  PetscCall(PetscOptionsReal("-lambda", "Fixed strength reduction factor", NULL, app->lambda, &app->lambda, NULL));
  PetscCall(PetscOptionsInt("-refine_levels", "Uniform DMPlex refinement levels", NULL, app->refine_levels, &app->refine_levels, NULL));
  PetscCall(PetscOptionsReal("-newton_rtol", "Relative residual tolerance", NULL, app->newton_rtol, &app->newton_rtol, NULL));
  PetscCall(PetscOptionsInt("-newton_max_it", "Maximum Newton iterations", NULL, app->newton_max_it, &app->newton_max_it, NULL));
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
  PetscCall(PetscOptionsString("-pmg_coarse_telescope_mode", "PMG P1 telescope implementation: default|coarse_dm", NULL, app->pmg_coarse_telescope_mode, app->pmg_coarse_telescope_mode, sizeof(app->pmg_coarse_telescope_mode), NULL));
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
  PetscCall(PetscOptionsString("-deflation_solver", "Deflated outer Krylov method: fgmres|cg", NULL, app->deflation_solver_name, app->deflation_solver_name, sizeof(app->deflation_solver_name), NULL));
  PetscCall(PetscOptionsReal("-deflation_basis_tol", "Minimum A-norm squared for keeping a deflation vector during A-orthogonalization", NULL, app->deflation_basis_tol, &app->deflation_basis_tol, NULL));
  PetscCall(PetscOptionsInt("-deflation_max_it", "Maximum iterations for the explicit deflated Krylov solve; 0 uses -ksp_max_it with a safe fallback", NULL, app->deflation_max_it, &app->deflation_max_it, NULL));
  PetscCall(PetscOptionsInt("-deflation_max_vectors", "Maximum collected deflation vectors; 0 keeps all Newton-step vectors", NULL, app->deflation_max_vectors, &app->deflation_max_vectors, NULL));
  PetscCall(PetscOptionsBool("-deflation_monitor", "Print explicit deflated Krylov residual history", NULL, app->deflation_monitor, &app->deflation_monitor, NULL));
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
  PetscCheck(app->pmg_coarse_redundant_group_size >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-pmg_coarse_redundant_group_size must be nonnegative");
  PetscCheck(app->pmg_coarse_telescope_active_ranks >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-pmg_coarse_telescope_active_ranks must be nonnegative");
  PetscCheck(app->pmg_coarse_telescope_ksp_max_it >= 1, comm, PETSC_ERR_ARG_OUTOFRANGE, "-pmg_coarse_telescope_ksp_max_it must be positive");
  PetscCheck(app->pmg_p2_telescope_active_ranks >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-pmg_p2_telescope_active_ranks must be nonnegative");
  PetscCheck(app->pmg_p2_telescope_ksp_max_it >= 1, comm, PETSC_ERR_ARG_OUTOFRANGE, "-pmg_p2_telescope_ksp_max_it must be positive");
  PetscCheck(app->pmg_lag_preconditioner >= 1, comm, PETSC_ERR_ARG_OUTOFRANGE, "-pmg_lag_preconditioner must be >= 1");
  PetscCall(PetscStrcasecmp(app->pmg_coarse_telescope_mode, "default", &flg));
  if (!flg) {
    PetscCall(PetscStrcasecmp(app->pmg_coarse_telescope_mode, "coarse_dm", &flg));
    PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-pmg_coarse_telescope_mode must be default or coarse_dm");
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
  PetscCheck(app->deflation_max_it >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-deflation_max_it must be nonnegative");
  PetscCheck(app->deflation_max_vectors >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-deflation_max_vectors must be nonnegative");
  PetscCall(PetscStrcasecmp(app->deflation_solver_name, "fgmres", &flg));
  if (flg) app->deflation_solver = DEFLATION_SOLVER_FGMRES;
  else {
    PetscCall(PetscStrcasecmp(app->deflation_solver_name, "cg", &flg));
    PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-deflation_solver must be fgmres or cg");
    app->deflation_solver = DEFLATION_SOLVER_CG;
  }
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
  const PetscInt x_comp[1] = {0}, z_comp[1] = {2};
  PetscBool      is_rollers, is_base_only, is_full_sides;

  PetscFunctionBeginUser;
  PetscCall(DMGetLabel(dm, "boundary_marker", &label));
  PetscCheck(label, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Missing boundary_marker label");
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "glued_base", label, 1, &base, 0, 3, components, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
  PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "rollers", &is_rollers));
  PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "base_only", &is_base_only));
  PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "full_sides", &is_full_sides));
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

      for (PetscInt d = 0; d < 3; ++d) x[d] = v0[d] + J[0 * 3 + d] * r[0] + J[1 * 3 + d] * r[1] + J[2 * 3 + d] * r[2];
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

      for (PetscInt d = 0; d < 3; ++d) x[d] = v0[d] + J[0 * 3 + d] * r[0] + J[1 * 3 + d] * r[1] + J[2 * 3 + d] * r[2];
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

static PetscErrorCode BuildInterpolationMatrix(DM fine_dm, P4Basis *fine_basis, DM coarse_dm, P4Basis *coarse_basis, Mat *P)
{
  MPI_Comm        comm = PetscObjectComm((PetscObject)fine_dm);
  PetscSection    fine_lsec, fine_gsec, coarse_lsec, coarse_gsec;
  Vec             fine_vec = NULL, coarse_vec = NULL;
  PetscInt        mlocal, nlocal, M, N, rlo, rhi, cStart, cEnd;
  PetscReal      *fine_points = NULL;
  PetscTabulation coarse_at_fine = NULL;
  const PetscReal *phi;
  Mat             mat;

  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(fine_dm, &fine_vec));
  PetscCall(DMCreateGlobalVector(coarse_dm, &coarse_vec));
  PetscCall(VecGetLocalSize(fine_vec, &mlocal));
  PetscCall(VecGetLocalSize(coarse_vec, &nlocal));
  PetscCall(VecGetSize(fine_vec, &M));
  PetscCall(VecGetSize(coarse_vec, &N));
  PetscCall(VecGetOwnershipRange(fine_vec, &rlo, &rhi));
  PetscCall(VecDestroy(&fine_vec));
  PetscCall(VecDestroy(&coarse_vec));

  PetscCall(BuildBasisReferencePoints(fine_basis, &fine_points));
  PetscCall(PetscFECreateTabulation(coarse_basis->fe_scalar, 1, fine_basis->n_basis, fine_points, 0, &coarse_at_fine));
  phi = coarse_at_fine->T[0];

  PetscCall(MatCreateAIJ(comm, mlocal, nlocal, M, N, coarse_basis->n_basis, NULL, coarse_basis->n_basis, NULL, &mat));
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
  PetscBool   use_coarse_dm = PETSC_FALSE;
  char        value[64];

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &ranks));
  PetscCall(PetscStrcasecmp(app->pmg_coarse_telescope_mode, "coarse_dm", &use_coarse_dm));
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

  PetscCall(SetDefaultOption("-mg_coarse_pc_type", use_coarse_dm ? "shell" : "telescope"));
  PetscCall(PetscSNPrintf(value, sizeof(value), "%" PetscInt_FMT, (PetscInt)ranks / app->pmg_coarse_telescope_active_ranks));
  if (!use_coarse_dm) {
    PetscCall(SetDefaultOption("-mg_coarse_pc_telescope_reduction_factor", value));
    PetscCall(SetDefaultOption("-mg_coarse_pc_telescope_subcomm_type", app->pmg_coarse_telescope_subcomm_type));
  }
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

typedef struct {
  AppCtx          *app;
  PetscBool       setup_done;
  PetscBool       active;
  PetscInt        active_ranks;
  PetscInt        reduction_factor;
  PetscInt        full_dofs;
  PetscInt        reduced_local_min;
  PetscInt        reduced_local_max;
  PetscSubcommType subcomm_type;
  char            subcomm_type_name[32];
  MPI_Comm        subcomm;
  IS              isrow;
  VecScatter      scatter;
  Vec             xred;
  Vec             yred;
  Vec             xtmp;
  Mat             Ared;
  KSP             subksp;
  PetscInt        apply_calls;
  PetscInt        operator_updates;
  PetscLogDouble  scatter_forward_time;
  PetscLogDouble  scatter_reverse_time;
  PetscLogDouble  inner_solve_time;
  PetscLogDouble  operator_update_time;
} PMGCoarseDMShellCtx;

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

static PetscErrorCode PMGCoarseDMShellScatterToReduced(PMGCoarseDMShellCtx *ctx, Vec full, Vec reduced)
{
  const PetscScalar *tmp_array;

  PetscFunctionBeginUser;
  PetscCall(VecScatterBegin(ctx->scatter, full, ctx->xtmp, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx->scatter, full, ctx->xtmp, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGetArrayRead(ctx->xtmp, &tmp_array));
  if (ctx->active) {
    PetscScalar *red_array;
    PetscInt     nred;

    PetscCall(VecGetLocalSize(reduced, &nred));
    PetscCall(VecGetArray(reduced, &red_array));
    for (PetscInt i = 0; i < nred; ++i) red_array[i] = tmp_array[i];
    PetscCall(VecRestoreArray(reduced, &red_array));
  }
  PetscCall(VecRestoreArrayRead(ctx->xtmp, &tmp_array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGCoarseDMShellScatterFromReduced(PMGCoarseDMShellCtx *ctx, Vec reduced, Vec full)
{
  PetscScalar *tmp_array;

  PetscFunctionBeginUser;
  PetscCall(VecGetArray(ctx->xtmp, &tmp_array));
  if (ctx->active) {
    const PetscScalar *red_array;
    PetscInt           nred;

    PetscCall(VecGetLocalSize(reduced, &nred));
    PetscCall(VecGetArrayRead(reduced, &red_array));
    for (PetscInt i = 0; i < nred; ++i) tmp_array[i] = red_array[i];
    PetscCall(VecRestoreArrayRead(reduced, &red_array));
  }
  PetscCall(VecRestoreArray(ctx->xtmp, &tmp_array));
  PetscCall(VecScatterBegin(ctx->scatter, ctx->xtmp, full, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(ctx->scatter, ctx->xtmp, full, INSERT_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGCoarseDMShellAttachNearNullspace(PMGCoarseDMShellCtx *ctx, Mat full_mat)
{
  MatNullSpace full_ns = NULL;
  PetscBool    has_const = PETSC_FALSE;
  PetscInt     nvec = 0;
  const Vec   *vecs = NULL;
  Vec         *sub_vecs = NULL;

  PetscFunctionBeginUser;
  PetscCall(MatGetNearNullSpace(full_mat, &full_ns));
  if (!full_ns) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatNullSpaceGetVecs(full_ns, &has_const, &nvec, &vecs));
  if (ctx->active && nvec > 0) PetscCall(VecDuplicateVecs(ctx->xred, nvec, &sub_vecs));
  for (PetscInt i = 0; i < nvec; ++i) PetscCall(PMGCoarseDMShellScatterToReduced(ctx, vecs[i], ctx->active ? sub_vecs[i] : NULL));
  if (ctx->active) {
    MatNullSpace sub_ns = NULL;

    PetscCall(MatNullSpaceCreate(ctx->subcomm, has_const, nvec, sub_vecs, &sub_ns));
    PetscCall(MatSetNearNullSpace(ctx->Ared, sub_ns));
    PetscCall(MatNullSpaceDestroy(&sub_ns));
    if (nvec > 0) PetscCall(VecDestroyVecs(nvec, &sub_vecs));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGCoarseDMShellCreateScatters(PC pc, PMGCoarseDMShellCtx *ctx, Mat B)
{
  MPI_Comm  comm = PetscObjectComm((PetscObject)pc);
  PetscMPIInt rank, size;
  PetscInt   M, m = 0, st, ed, bs;
  Vec        x = NULL;
  VecType    vectype = NULL;
  PetscInt   min_in, max_in;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(ctx->active_ranks > 0 && size > ctx->active_ranks, comm, PETSC_ERR_ARG_WRONG,
             "PMG coarse-DM shell requires 0 < active_ranks < ranks");
  PetscCheck(size % ctx->active_ranks == 0, comm, PETSC_ERR_ARG_WRONG,
             "-pmg_coarse_telescope_active_ranks %" PetscInt_FMT " must divide MPI ranks %d", ctx->active_ranks, size);
  ctx->reduction_factor = (PetscInt)size / ctx->active_ranks;
  if (ctx->subcomm_type == PETSC_SUBCOMM_CONTIGUOUS) ctx->active = (PetscBool)(rank < ctx->active_ranks);
  else ctx->active = (PetscBool)(rank % ctx->reduction_factor == 0);
  PetscCallMPI(MPI_Comm_split(comm, ctx->active ? 0 : MPI_UNDEFINED, rank, &ctx->subcomm));

  PetscCall(MatGetSize(B, &M, NULL));
  PetscCall(MatGetBlockSize(B, &bs));
  PetscCall(MatCreateVecs(B, &x, NULL));
  PetscCall(MatGetVecType(B, &vectype));
  ctx->full_dofs = M;
  if (ctx->active) {
    PetscCall(VecCreate(ctx->subcomm, &ctx->xred));
    PetscCall(VecSetSizes(ctx->xred, PETSC_DECIDE, M));
    PetscCall(VecSetBlockSize(ctx->xred, bs));
    if (vectype) PetscCall(VecSetType(ctx->xred, vectype));
    PetscCall(VecSetFromOptions(ctx->xred));
    PetscCall(VecDuplicate(ctx->xred, &ctx->yred));
    PetscCall(VecGetLocalSize(ctx->xred, &m));
    PetscCall(VecGetOwnershipRange(ctx->xred, &st, &ed));
    PetscCall(ISCreateStride(comm, ed - st, st, 1, &ctx->isrow));
  } else {
    PetscCall(VecGetOwnershipRange(x, &st, &ed));
    PetscCall(ISCreateStride(comm, 0, st, 1, &ctx->isrow));
  }
  PetscCall(ISSetBlockSize(ctx->isrow, bs));
  PetscCall(VecCreate(comm, &ctx->xtmp));
  PetscCall(VecSetSizes(ctx->xtmp, m, PETSC_DECIDE));
  PetscCall(VecSetBlockSize(ctx->xtmp, bs));
  if (vectype) PetscCall(VecSetType(ctx->xtmp, vectype));
  PetscCall(VecSetFromOptions(ctx->xtmp));
  PetscCall(VecScatterCreate(x, ctx->isrow, ctx->xtmp, NULL, &ctx->scatter));
  min_in = ctx->active ? m : PETSC_MAX_INT;
  max_in = ctx->active ? m : 0;
  PetscCallMPI(MPI_Allreduce(&min_in, &ctx->reduced_local_min, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPI_Allreduce(&max_in, &ctx->reduced_local_max, 1, MPIU_INT, MPI_MAX, comm));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGCoarseDMShellUpdateOperator(PC pc, PMGCoarseDMShellCtx *ctx, Mat B)
{
  MPI_Comm       comm = PetscObjectComm((PetscObject)pc);
  Mat           *submats = NULL;
  Mat            Blocal = NULL;
  IS             iscol = NULL;
  PetscInt       nr, nc, bs;
  PetscLogDouble t0, t1, t_sub0, t_sub1, t_cat0, t_cat1, submatrix_time, concatenate_time;
  const char    *reuse_label = ctx->setup_done ? "reuse" : "initial";
  MatReuse       reuse = ctx->setup_done ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX;

  PetscFunctionBeginUser;
  PetscCall(PetscTime(&t0));
  PetscCall(MatGetSize(B, &nr, &nc));
  PetscCall(MatGetBlockSizes(B, NULL, &bs));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, nc, 0, 1, &iscol));
  PetscCall(ISSetIdentity(iscol));
  PetscCall(ISSetBlockSize(iscol, bs));
  PetscCall(MatSetOption(B, MAT_SUBMAT_SINGLEIS, PETSC_TRUE));
  PetscCall(PetscTime(&t_sub0));
  PetscCall(MatCreateSubMatrices(B, 1, &ctx->isrow, &iscol, MAT_INITIAL_MATRIX, &submats));
  PetscCall(PetscTime(&t_sub1));
  Blocal = submats[0];
  PetscCall(PetscFree(submats));
  if (ctx->active) {
    PetscInt mm;

    PetscCall(MatGetSize(Blocal, &mm, NULL));
    PetscCall(PetscTime(&t_cat0));
    PetscCall(MatCreateMPIMatConcatenateSeqMat(ctx->subcomm, Blocal, mm, reuse, &ctx->Ared));
    PetscCall(PetscTime(&t_cat1));
    PetscCall(KSPSetOperators(ctx->subksp, ctx->Ared, ctx->Ared));
  } else {
    t_cat0 = t_cat1 = 0.0;
  }
  if (reuse == MAT_INITIAL_MATRIX) PetscCall(PMGCoarseDMShellAttachNearNullspace(ctx, B));
  PetscCall(MatDestroy(&Blocal));
  PetscCall(ISDestroy(&iscol));
  PetscCall(PetscTime(&t1));
  submatrix_time   = t_sub1 - t_sub0;
  concatenate_time = t_cat1 - t_cat0;
  ctx->operator_update_time += t1 - t0;
  ctx->operator_updates++;
  PetscCall(PetscPrintf(comm,
                        "PMG_COARSE_DM_OPERATOR_UPDATE reuse=%s matrix_reuse=%s time=%.6g submatrix_time=%.6g concatenate_time=%.6g attach_nullspace=%s\n",
                        reuse_label, reuse == MAT_INITIAL_MATRIX ? "initial" : "reuse", (double)(t1 - t0), (double)submatrix_time,
                        (double)concatenate_time, reuse == MAT_INITIAL_MATRIX ? "true" : "false"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGCoarseDMShellSetUp(PC pc)
{
  PMGCoarseDMShellCtx *ctx = NULL;
  Mat                  B = NULL;
  MPI_Comm             comm = PetscObjectComm((PetscObject)pc);

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, (void **)&ctx));
  PetscCall(PCGetOperators(pc, NULL, &B));
  PetscCheck(B, comm, PETSC_ERR_ARG_WRONGSTATE, "PMG coarse-DM shell requires a P1 coarse operator");
  if (!ctx->setup_done) {
    const char *prefix = NULL;
    PC          inner_pc = NULL;

    PetscCall(PMGCoarseDMShellCreateScatters(pc, ctx, B));
    if (ctx->active) {
      PetscCall(KSPCreate(ctx->subcomm, &ctx->subksp));
      PetscCall(KSPSetType(ctx->subksp, ctx->app->pmg_coarse_telescope_ksp_type));
      PetscCall(KSPSetTolerances(ctx->subksp, ctx->app->pmg_coarse_telescope_ksp_rtol, PETSC_DEFAULT, PETSC_DEFAULT, ctx->app->pmg_coarse_telescope_ksp_max_it));
      PetscCall(KSPGetPC(ctx->subksp, &inner_pc));
      PetscCall(ConfigurePMGBasePC(inner_pc, ctx->app->pmg_coarse_telescope_pc_type, ctx->app->pmg_coarse_gamg_aggressive_square_graph));
      PetscCall(PCGetOptionsPrefix(pc, &prefix));
      PetscCall(KSPSetOptionsPrefix(ctx->subksp, prefix));
      PetscCall(KSPAppendOptionsPrefix(ctx->subksp, "telescope_"));
      PetscCall(KSPSetFromOptions(ctx->subksp));
    }
    PetscCall(PetscPrintf(comm,
                          "PMG_COARSE_DM_CONFIG enabled=true mode=custom_shell_fallback active_ranks=%" PetscInt_FMT " subcomm=%s full_dofs=%" PetscInt_FMT " reduced_local_min=%" PetscInt_FMT " reduced_local_max=%" PetscInt_FMT "\n",
                          ctx->active_ranks, ctx->subcomm_type_name, ctx->full_dofs, ctx->reduced_local_min, ctx->reduced_local_max));
  }
  PetscCall(PMGCoarseDMShellUpdateOperator(pc, ctx, B));
  ctx->setup_done = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGCoarseDMShellApply(PC pc, Vec x, Vec y)
{
  PMGCoarseDMShellCtx *ctx = NULL;
  PetscLogDouble       t0, t1, t2, t3;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, (void **)&ctx));
  PetscCall(PetscTime(&t0));
  PetscCall(PMGCoarseDMShellScatterToReduced(ctx, x, ctx->xred));
  PetscCall(PetscTime(&t1));
  if (ctx->active) PetscCall(KSPSolve(ctx->subksp, ctx->xred, ctx->yred));
  PetscCall(PetscTime(&t2));
  PetscCall(PMGCoarseDMShellScatterFromReduced(ctx, ctx->yred, y));
  PetscCall(PetscTime(&t3));
  ctx->apply_calls++;
  ctx->scatter_forward_time += t1 - t0;
  ctx->inner_solve_time += t2 - t1;
  ctx->scatter_reverse_time += t3 - t2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGCoarseDMShellDestroy(PC pc)
{
  PMGCoarseDMShellCtx *ctx = NULL;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, (void **)&ctx));
  if (!ctx) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pc),
                        "PMG_COARSE_DM_SUMMARY apply_calls=%" PetscInt_FMT " operator_updates=%" PetscInt_FMT
                        " scatter_forward_time=%.6g inner_solve_time=%.6g scatter_reverse_time=%.6g operator_update_time=%.6g\n",
                        ctx->apply_calls, ctx->operator_updates, (double)ctx->scatter_forward_time, (double)ctx->inner_solve_time,
                        (double)ctx->scatter_reverse_time, (double)ctx->operator_update_time));
  PetscCall(KSPDestroy(&ctx->subksp));
  PetscCall(MatDestroy(&ctx->Ared));
  PetscCall(VecDestroy(&ctx->xred));
  PetscCall(VecDestroy(&ctx->yred));
  PetscCall(VecDestroy(&ctx->xtmp));
  PetscCall(VecScatterDestroy(&ctx->scatter));
  PetscCall(ISDestroy(&ctx->isrow));
  if (ctx->subcomm != MPI_COMM_NULL) PetscCallMPI(MPI_Comm_free(&ctx->subcomm));
  PetscCall(PetscFree(ctx));
  PetscCall(PCShellSetContext(pc, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigurePMGCoarseDMShell(PC pc, AppCtx *app, PetscInt active_ranks)
{
  PMGCoarseDMShellCtx *ctx = NULL;
  MPI_Comm             comm = PetscObjectComm((PetscObject)pc);

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&ctx));
  ctx->app          = app;
  ctx->active_ranks = active_ranks;
  ctx->subcomm      = MPI_COMM_NULL;
  PetscCall(PMGParseSubcommType(comm, app->pmg_coarse_telescope_subcomm_type, &ctx->subcomm_type));
  PetscCall(PetscStrncpy(ctx->subcomm_type_name, app->pmg_coarse_telescope_subcomm_type, sizeof(ctx->subcomm_type_name)));
  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetContext(pc, ctx));
  PetscCall(PCShellSetName(pc, "pmg_coarse_dm_shell"));
  PetscCall(PCShellSetSetUp(pc, PMGCoarseDMShellSetUp));
  PetscCall(PCShellSetApply(pc, PMGCoarseDMShellApply));
  PetscCall(PCShellSetDestroy(pc, PMGCoarseDMShellDestroy));
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
  PetscBool use_coarse_dm_mode = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &ranks));
  PetscCall(PetscStrcasecmp(app->pmg_coarse_telescope_mode, "coarse_dm", &use_coarse_dm_mode));
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
  PetscCheck(!coarse_is_shell || use_coarse_dm_mode, comm, PETSC_ERR_ARG_WRONG,
             "-mg_coarse_pc_type shell is reserved for -pmg_coarse_telescope_mode coarse_dm");
  if (!use_coarse_dm_mode || !coarse_is_shell) {
    Vec      p1_vec = NULL;
    PetscInt p1_global_dofs = 0;

    PetscCall(DMCreateGlobalVector(dm_p1, &p1_vec));
    PetscCall(VecGetSize(p1_vec, &p1_global_dofs));
    PetscCall(VecDestroy(&p1_vec));
    PetscCall(PetscPrintf(comm,
                          "PMG_COARSE_DM_CONFIG enabled=false mode=%s active_ranks=%" PetscInt_FMT " subcomm=%s full_dofs=%" PetscInt_FMT " reduced_local_min=0 reduced_local_max=0\n",
                          app->pmg_coarse_telescope_mode, app->pmg_coarse_telescope_active_ranks, app->pmg_coarse_telescope_subcomm_type, p1_global_dofs));
  }

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
    if (coarse_is_lu || coarse_is_telescope || coarse_is_shell) {
      PetscCall(KSPSetType(coarse, KSPPREONLY));
    } else {
      PetscCall(KSPSetType(coarse, KSPFGMRES));
      PetscCall(KSPSetTolerances(coarse, 1.0e-3, PETSC_DEFAULT, PETSC_DEFAULT, 100));
      PetscCall(KSPGMRESSetRestart(coarse, 100));
    }
    if (coarse_is_shell) {
      PetscCall(ConfigurePMGCoarseDMShell(coarse_pc, app, app->pmg_coarse_telescope_active_ranks));
    } else {
      PetscCall(ConfigurePMGBasePC(coarse_pc, coarse_pc_type, app->pmg_coarse_gamg_aggressive_square_graph));
    }
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
      PetscCall(ConfigurePMG(pc, dm, actx, app, solver));
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

static PetscErrorCode LinearSolverInit(LinearSolverCtx *solver, DM dm, AssemblyCtx *actx, AppCtx *app, Mat A)
{
  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(solver, sizeof(*solver)));
  solver->dm    = dm;
  solver->actx  = actx;
  solver->app   = app;
  solver->A     = A;
  solver->reuse = app->reuse_linear_solver;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "LINEAR_SOLVER_REUSE enabled=%s\n", solver->reuse ? "true" : "false"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverDestroy(LinearSolverCtx *solver)
{
  PetscFunctionBeginUser;
  PetscCall(LinearSolverClearOrthBasis(solver));
  for (PetscInt i = 0; i < solver->n_raw_basis; ++i) PetscCall(VecDestroy(&solver->raw_basis[i]));
  PetscCall(PetscFree(solver->raw_basis));
  PetscCall(PetscFree(solver->orth_basis));
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
  for (PetscInt i = 0; i < solver->n_orth_basis; ++i) PetscCall(VecDestroy(&solver->orth_basis[i]));
  solver->n_orth_basis = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverAppendRawBasis(LinearSolverCtx *solver, Vec v, const char label[])
{
  Vec copy = NULL;

  PetscFunctionBeginUser;
  if (!solver->app->use_deflation) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecDuplicate(v, &copy));
  PetscCall(VecCopy(v, copy));
  if (solver->raw_basis_cap == solver->n_raw_basis) {
    const PetscInt new_cap = solver->raw_basis_cap ? 2 * solver->raw_basis_cap : 8;

    PetscCall(PetscRealloc((size_t)new_cap * sizeof(Vec), &solver->raw_basis));
    for (PetscInt i = solver->raw_basis_cap; i < new_cap; ++i) solver->raw_basis[i] = NULL;
    solver->raw_basis_cap = new_cap;
  }
  solver->raw_basis[solver->n_raw_basis++] = copy;
  copy                                = NULL;
  if (solver->app->deflation_max_vectors > 0 && solver->n_raw_basis > solver->app->deflation_max_vectors) {
    PetscCall(VecDestroy(&solver->raw_basis[0]));
    for (PetscInt i = 1; i < solver->n_raw_basis; ++i) solver->raw_basis[i - 1] = solver->raw_basis[i];
    solver->raw_basis[--solver->n_raw_basis] = NULL;
  }
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_BASIS_ADD label=\"%s\" raw_cols=%" PetscInt_FMT " max_cols=%" PetscInt_FMT "\n",
                        label, solver->n_raw_basis, solver->app->deflation_max_vectors));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverStoreOrthVector(LinearSolverCtx *solver, Vec v)
{
  Vec copy = NULL;

  PetscFunctionBeginUser;
  if (solver->orth_basis_cap == solver->n_orth_basis) {
    const PetscInt new_cap = solver->orth_basis_cap ? 2 * solver->orth_basis_cap : PetscMax(1, solver->n_raw_basis);

    PetscCall(PetscRealloc((size_t)new_cap * sizeof(Vec), &solver->orth_basis));
    for (PetscInt i = solver->orth_basis_cap; i < new_cap; ++i) solver->orth_basis[i] = NULL;
    solver->orth_basis_cap = new_cap;
  }
  PetscCall(VecDuplicate(v, &copy));
  PetscCall(VecCopy(v, copy));
  solver->orth_basis[solver->n_orth_basis++] = copy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverAOrthogonalizeBasis(LinearSolverCtx *solver, Mat A, const char label[])
{
  Vec            v = NULL, Av = NULL;
  PetscInt       skipped_small = 0, skipped_nonpositive = 0;
  PetscLogDouble t0, t1;

  PetscFunctionBeginUser;
  PetscCall(PetscTime(&t0));
  PetscCall(LinearSolverClearOrthBasis(solver));
  if (!solver->n_raw_basis) PetscFunctionReturn(PETSC_SUCCESS);
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
    for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
      PetscScalar coeff;

      PetscCall(MatMult(A, v, Av));
      PetscCall(VecDot(solver->orth_basis[i], Av, &coeff));
      PetscCall(VecAXPY(v, -coeff, solver->orth_basis[i]));
    }
    PetscCall(MatMult(A, v, Av));
    PetscCall(VecDot(v, Av, &norm_scalar));
    norm_a = PetscRealPart(norm_scalar);
    if (norm_a <= 0.0) {
      ++skipped_nonpositive;
      continue;
    }
    if (norm_a <= solver->app->deflation_basis_tol) {
      ++skipped_small;
      continue;
    }
    PetscCall(VecScale(v, 1.0 / PetscSqrtReal(norm_a)));
    PetscCall(LinearSolverStoreOrthVector(solver, v));
  }
  for (PetscInt i = 0; i < solver->n_orth_basis / 2; ++i) {
    Vec tmp = solver->orth_basis[i];

    solver->orth_basis[i] = solver->orth_basis[solver->n_orth_basis - 1 - i];
    solver->orth_basis[solver->n_orth_basis - 1 - i] = tmp;
  }
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&Av));
  PetscCall(PetscTime(&t1));
  solver->deflation_orthogonalization_time += t1 - t0;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_ORTHO label=\"%s\" raw_cols=%" PetscInt_FMT " orth_cols=%" PetscInt_FMT " skipped_small=%" PetscInt_FMT " skipped_nonpositive=%" PetscInt_FMT " tol=%.6e time=%.6g\n",
                        label, solver->n_raw_basis, solver->n_orth_basis, skipped_small, skipped_nonpositive,
                        (double)solver->app->deflation_basis_tol, (double)(t1 - t0)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DeflationCoarseInitialGuess(LinearSolverCtx *solver, Mat A, Vec rhs, Vec x, Vec r, Vec work)
{
  PetscLogDouble t0, t1;

  PetscFunctionBeginUser;
  PetscCall(PetscTime(&t0));
  PetscCall(VecZeroEntries(x));
  for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
    PetscScalar coeff;

    PetscCall(VecDot(solver->orth_basis[i], rhs, &coeff));
    PetscCall(VecAXPY(x, coeff, solver->orth_basis[i]));
  }
  PetscCall(VecCopy(rhs, r));
  if (solver->n_orth_basis) {
    PetscCall(MatMult(A, x, work));
    PetscCall(VecAXPY(r, -1.0, work));
  }
  PetscCall(PetscTime(&t1));
  solver->deflation_coarse_time += t1 - t0;
  ++solver->deflation_coarse_calls;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_COARSE_INITIAL basis_cols=%" PetscInt_FMT " call=%" PetscInt_FMT " time=%.6g cumulative_time=%.6g\n",
                        solver->n_orth_basis, solver->deflation_coarse_calls, (double)(t1 - t0), (double)solver->deflation_coarse_time));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DeflationApplyProjectedPC(LinearSolverCtx *solver, Mat A, PC pc, Vec v, Vec z, Vec Az)
{
  PetscLogDouble t0, t1, t2, pc_time, projector_time = 0.0;

  PetscFunctionBeginUser;
  PetscCall(PetscTime(&t0));
  PetscCall(PCApply(pc, v, z));
  PetscCall(PetscTime(&t1));
  if (solver->n_orth_basis) {
    PetscCall(MatMult(A, z, Az));
    for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
      PetscScalar coeff;

      PetscCall(VecDot(solver->orth_basis[i], Az, &coeff));
      PetscCall(VecAXPY(z, -coeff, solver->orth_basis[i]));
    }
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
  if (rel <= rtol) {
    converged = PETSC_TRUE;
  } else {
    PetscCall(VecCopy(r, V[0]));
    PetscCall(VecScale(V[0], 1.0 / beta));
  }

  for (PetscInt j = 0; !converged && j < max_it; ++j) {
    PetscReal hnext, res_norm;

    PetscCall(DeflationApplyProjectedPC(solver, A, pc, V[j], Z[j], Az));
    PetscCall(MatMult(A, Z[j], Az));
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

    PetscCall(DeflationApplyProjectedPC(solver, A, pc, r, z, Ap));
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
    PetscCall(DeflationApplyProjectedPC(solver, A, pc, r, z, Ap));
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
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                              "PMG_LAG_PRECONDITIONER solve_index=%" PetscInt_FMT " lag=%" PetscInt_FMT " rebuild=%s reuse_preconditioner=%s\n",
                              idx, lag, rebuild ? "true" : "false", reuse_pc ? "true" : "false"));
        solver->pmg_lag_solve_index++;
      }
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

static PetscErrorCode NewtonSolve(DM dm, AssemblyCtx *actx, AppCtx *app, LinearSolverCtx *solver, Mat A, Vec f_ext, Vec u, PetscReal rhs_norm, NewtonStats *stats)
{
  Vec            residual, rhs, du, u_trial, r_trial;
  PetscReal      rel = -1.0, trial_rel;
  PetscInt       total_linear_its = 0, newton_its = 0;
  PetscLogDouble t0, t1, assembly_time = 0.0, solve_time = 0.0;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(stats, sizeof(*stats)));
  PetscCall(VecDuplicate(f_ext, &residual));
  PetscCall(VecDuplicate(f_ext, &rhs));
  PetscCall(VecDuplicate(f_ext, &du));
  PetscCall(VecDuplicate(f_ext, &u_trial));
  PetscCall(VecDuplicate(f_ext, &r_trial));

  for (PetscInt it = 0; it < app->newton_max_it; ++it) {
    PetscInt linear_its = 0;

    PetscCall(PetscTime(&t0));
    PetscCall(AssemblePlasticResidualJacobian(actx, app->lambda, u, f_ext, A, residual, PETSC_TRUE));
    PetscCall(PetscTime(&t1));
    assembly_time += t1 - t0;
    PetscCall(ResidualNormFree(actx, residual, rhs_norm, &rel));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "Newton it=%" PetscInt_FMT " rel_res=%10.4e\n", it, (double)rel));
    if (rel <= app->newton_rtol) break;

    PetscCall(VecCopy(residual, rhs));
    PetscCall(VecScale(rhs, -1.0));
    PetscCall(ApplyZeroDirichlet(actx->constrained_is, A, rhs));
    PetscCall(VecZeroEntries(du));
    PetscCall(PetscTime(&t0));
    PetscCall(SolveLinearSystem(solver, rhs, du, "Newton correction", u, PETSC_TRUE, &linear_its));
    PetscCall(PetscTime(&t1));
    solve_time += t1 - t0;
    total_linear_its += linear_its;
    ++newton_its;
    PetscCall(LinearSolverAppendRawBasis(solver, du, "Newton correction"));

    if (app->line_search) {
      PetscReal alpha = 1.0;
      while (PETSC_TRUE) {
        PetscCall(VecWAXPY(u_trial, alpha, du, u));
        PetscCall(AssemblePlasticResidualJacobian(actx, app->lambda, u_trial, f_ext, NULL, r_trial, PETSC_FALSE));
        PetscCall(ResidualNormFree(actx, r_trial, rhs_norm, &trial_rel));
        if (trial_rel < rel || alpha <= app->damping_min) {
          PetscCall(VecCopy(u_trial, u));
          rel = trial_rel;
          PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "  alpha=%8.3e trial_rel=%10.4e\n", (double)alpha, (double)trial_rel));
          break;
        }
        alpha *= 0.5;
      }
    } else {
      PetscCall(VecAXPY(u, 1.0, du));
      PetscCall(AssemblePlasticResidualJacobian(actx, app->lambda, u, f_ext, NULL, r_trial, PETSC_FALSE));
      PetscCall(ResidualNormFree(actx, r_trial, rhs_norm, &rel));
    }
  }
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                        "Newton summary: final_rel=%10.4e total_linear_its=%" PetscInt_FMT " assembly_time=%.6g solve_time=%.6g\n",
                        (double)rel, total_linear_its, (double)assembly_time, (double)solve_time));
  stats->final_rel        = rel;
  stats->newton_its       = newton_its;
  stats->total_linear_its = total_linear_its;
  stats->assembly_time    = assembly_time;
  stats->solve_time       = solve_time;

  PetscCall(VecDestroy(&residual));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&du));
  PetscCall(VecDestroy(&u_trial));
  PetscCall(VecDestroy(&r_trial));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx         app;
  P4Basis        basis;
  DM             dm = NULL;
  AssemblyCtx    actx;
  LinearSolverCtx lsolver;
  NewtonStats    newton_stats;
  Mat            A = NULL;
  Vec            u = NULL, f_ext = NULL;
  PetscInt       cStart, cEnd, nStart, nEnd, elastic_its;
  PetscInt       local_dofs, global_dofs;
  PetscReal      rhs_norm, u_norm;
  PetscLogDouble t_start, t_end, t0, t1, elastic_assembly_time, elastic_solve_time;

  PetscCall(PetscInitialize(&argc, &argv, NULL, "Standalone pure PETSc P4 plasticity case\n"));
  PetscCall(PetscTime(&t_start));
  PetscCall(ParseOptions(PETSC_COMM_WORLD, &app));
  PetscCall(SetDDPartitionerDefault(PETSC_COMM_WORLD, &app));
  PetscCall(P4BasisCreate(PETSC_COMM_SELF, &basis));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &app, &basis, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &nStart, &nEnd));
  PetscCall(DMCreateMatrix(dm, &A));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(DMCreateGlobalVector(dm, &f_ext));
  PetscCall(VecGetLocalSize(u, &local_dofs));
  PetscCall(VecGetSize(u, &global_dofs));
  PetscCall(AssemblyCtxCreate(dm, &basis, &actx));
  PetscCall(LinearSolverInit(&lsolver, dm, &actx, &app, A));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "mesh=%s%s refine_levels=%" PetscInt_FMT " local_cells=%" PetscInt_FMT " local_vertices=%" PetscInt_FMT " local_dofs=%" PetscInt_FMT " global_dofs=%" PetscInt_FMT " P4_basis=%" PetscInt_FMT " owned_constraints=%" PetscInt_FMT " global_constraints=%" PetscInt_FMT " pc_variant=%s bc=%s lambda=%.6g\n",
                        app.use_box_mesh ? "generated-box:" : "", app.use_box_mesh ? "unit" : app.mesh, app.refine_levels, cEnd - cStart, nEnd - nStart, local_dofs, global_dofs, basis.n_basis, actx.n_constrained_local, actx.n_constrained_all, app.variant_name, app.mesh_bc_mode, (double)app.lambda));
  if (app.inspect_partition) {
    PetscCall(ReportPartitionDiagnostics(dm, A, u, &actx, &app));
    PetscCall(LinearSolverDestroy(&lsolver));
    PetscCall(AssemblyCtxDestroy(&actx));
    PetscCall(VecDestroy(&f_ext));
    PetscCall(VecDestroy(&u));
    PetscCall(MatDestroy(&A));
    PetscCall(DMDestroy(&dm));
    PetscCall(P4BasisDestroy(&basis));
    PetscCall(PetscFinalize());
    return 0;
  }

  PetscCall(PetscTime(&t0));
  PetscCall(AssembleElasticProblem(&actx, A, f_ext));
  PetscCall(PetscTime(&t1));
  elastic_assembly_time = t1 - t0;
  PetscCall(VecNorm(f_ext, NORM_2, &rhs_norm));
  PetscCall(ApplyZeroDirichlet(actx.constrained_is, A, f_ext));
  PetscCall(VecNorm(f_ext, NORM_2, &rhs_norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "elastic assembly_time=%.6g rhs_norm=%.6e\n", (double)elastic_assembly_time, (double)rhs_norm));
  {
    PetscBool check_symmetry = PETSC_FALSE;

    PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_matrix_symmetry", &check_symmetry, NULL));
    if (check_symmetry) {
      PetscBool symmetric;

      PetscCall(MatIsSymmetric(A, 1.0e-10, &symmetric));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "elastic matrix_symmetric=%s tol=1e-10\n", symmetric ? "true" : "false"));
    }
  }
  PetscCall(VecZeroEntries(u));
  PetscCall(PetscTime(&t0));
  PetscCall(SolveLinearSystem(&lsolver, f_ext, u, "Elastic initial", NULL, PETSC_FALSE, &elastic_its));
  PetscCall(PetscTime(&t1));
  elastic_solve_time = t1 - t0;
  PetscCall(VecNorm(u, NORM_2, &u_norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "elastic solve_time=%.6g u_norm=%.6e\n", (double)elastic_solve_time, (double)u_norm));
  PetscCall(LinearSolverAppendRawBasis(&lsolver, u, "Elastic initial"));

  PetscCall(NewtonSolve(dm, &actx, &app, &lsolver, A, f_ext, u, rhs_norm, &newton_stats));
  PetscCall(VecNorm(u, NORM_2, &u_norm));
  PetscCall(PetscTime(&t_end));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "final displacement_norm=%.8e total_wall_time=%.6g\n", (double)u_norm, (double)(t_end - t_start)));
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
                          "RESULT variant=%s ranks=%d partitioner=%s global_dofs=%" PetscInt_FMT " elastic_its=%" PetscInt_FMT " newton_its=%" PetscInt_FMT " newton_linear_its=%" PetscInt_FMT " total_linear_its=%" PetscInt_FMT " elastic_assembly_time=%.6g elastic_solve_time=%.6g newton_assembly_time=%.6g newton_solve_time=%.6g wall_time=%.6g final_rel=%.6e deflation=%s deflation_solver=%s deflation_basis_cols=%" PetscInt_FMT " deflation_orthogonalization_time=%.6g deflation_coarse_initial_time=%.6g deflation_coarse_initial_calls=%" PetscInt_FMT " deflation_pc_apply_time=%.6g deflation_projector_time=%.6g deflation_projected_pc_calls=%" PetscInt_FMT "\n",
                          app.variant_name, size, part_type ? part_type : "unknown", global_dofs, elastic_its, newton_stats.newton_its, newton_stats.total_linear_its, elastic_its + newton_stats.total_linear_its,
                          (double)elastic_assembly_time, (double)elastic_solve_time, (double)newton_stats.assembly_time, (double)newton_stats.solve_time, (double)(t_end - t_start),
                          (double)newton_stats.final_rel, app.use_deflation ? "true" : "false", app.deflation_solver_name, lsolver.n_raw_basis, (double)lsolver.deflation_orthogonalization_time,
                          (double)lsolver.deflation_coarse_time, lsolver.deflation_coarse_calls, (double)lsolver.deflation_pc_apply_time,
                          (double)lsolver.deflation_projector_time, lsolver.deflation_projected_pc_calls));
  }

  PetscCall(LinearSolverDestroy(&lsolver));
  PetscCall(AssemblyCtxDestroy(&actx));
  PetscCall(VecDestroy(&f_ext));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&dm));
  PetscCall(P4BasisDestroy(&basis));
  PetscCall(PetscFinalize());
  return 0;
}
