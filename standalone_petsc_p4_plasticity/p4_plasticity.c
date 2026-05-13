#include "assembly.h"
#include "p4_basis.h"

#include <petscksp.h>

typedef enum {
  VARIANT_GAMG,
  VARIANT_BDDC,
  VARIANT_FETIDP,
  VARIANT_NONE
} PCVariant;

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
} AppCtx;

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

  PetscOptionsBegin(comm, NULL, "Standalone P4 plasticity options", NULL);
  PetscCall(PetscOptionsString("-mesh", "Gmsh mesh path", NULL, app->mesh, app->mesh, sizeof(app->mesh), NULL));
  PetscCall(PetscOptionsReal("-lambda", "Fixed strength reduction factor", NULL, app->lambda, &app->lambda, NULL));
  PetscCall(PetscOptionsInt("-refine_levels", "Uniform DMPlex refinement levels", NULL, app->refine_levels, &app->refine_levels, NULL));
  PetscCall(PetscOptionsReal("-newton_rtol", "Relative residual tolerance", NULL, app->newton_rtol, &app->newton_rtol, NULL));
  PetscCall(PetscOptionsInt("-newton_max_it", "Maximum Newton iterations", NULL, app->newton_max_it, &app->newton_max_it, NULL));
  PetscCall(PetscOptionsReal("-linear_rtol", "Default KSP relative tolerance", NULL, app->ksp_rtol, &app->ksp_rtol, NULL));
  PetscCall(PetscOptionsBool("-line_search", "Use residual backtracking", NULL, app->line_search, &app->line_search, NULL));
  PetscCall(PetscOptionsBool("-use_box_mesh", "Use a tiny generated unit-box tetra mesh for smoke tests", NULL, app->use_box_mesh, &app->use_box_mesh, NULL));
  PetscCall(PetscOptionsReal("-damping_min", "Minimum backtracking damping", NULL, app->damping_min, &app->damping_min, NULL));
  PetscCall(PetscOptionsString("-pc_variant", "gamg|bddc|fetidp|none", NULL, app->variant_name, app->variant_name, sizeof(app->variant_name), NULL));
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
        PetscCall(PetscStrcasecmp(app->variant_name, "none", &flg));
        PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-pc_variant must be gamg, bddc, fetidp, or none");
        app->variant = VARIANT_NONE;
      }
    }
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
    PetscCall(DMRefine(cur, comm, &refined));
    PetscCheck(refined, comm, PETSC_ERR_SUP, "DMRefine did not produce a refined mesh at level %" PetscInt_FMT, r);
    PetscCall(DMDestroy(&cur));
    cur = refined;
    PetscCall(DMSetFromOptions(cur));
  }
  PetscCall(DMSetField(cur, 0, NULL, (PetscObject)basis->fe_vector));
  PetscCall(DMCreateDS(cur));
  PetscCall(DMGetCoordinatesLocalSetUp(cur));
  if (app->variant == VARIANT_BDDC || app->variant == VARIANT_FETIDP) PetscCall(DMSetMatType(cur, MATIS));
  else PetscCall(DMSetMatType(cur, MATAIJ));
  *dm = cur;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AttachNearNullspace(DM dm, IS constrained, Mat A)
{
  PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *);
  PetscErrorCode (*modes_f[6])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *) = {RigidTx, RigidTy, RigidTz, RigidRx, RigidRy, RigidRz};
  Vec          modes[6], kept[6];
  MatNullSpace ns;
  PetscInt     nkept = 0;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < 6; ++i) {
    PetscCall(DMCreateGlobalVector(dm, &modes[i]));
    funcs[0] = modes_f[i];
    PetscCall(DMProjectFunction(dm, 0.0, funcs, NULL, INSERT_ALL_VALUES, modes[i]));
    PetscCall(ZeroConstrainedVector(constrained, modes[i]));
    for (PetscInt j = 0; j < nkept; ++j) {
      PetscScalar dot;
      PetscCall(VecDot(modes[i], kept[j], &dot));
      PetscCall(VecAXPY(modes[i], -dot, kept[j]));
    }
    PetscReal norm;
    PetscCall(VecNorm(modes[i], NORM_2, &norm));
    if (norm > 1.0e-12) {
      PetscCall(VecScale(modes[i], 1.0 / norm));
      kept[nkept++] = modes[i];
    } else {
      PetscCall(VecDestroy(&modes[i]));
    }
  }
  if (nkept > 0) {
    PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)A), PETSC_FALSE, nkept, kept, &ns));
    PetscCall(MatSetNearNullSpace(A, ns));
    PetscCall(MatNullSpaceDestroy(&ns));
  }
  for (PetscInt i = 0; i < nkept; ++i) PetscCall(VecDestroy(&kept[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigureKSP(KSP ksp, DM dm, AppCtx *app, Mat A, IS constrained)
{
  PC pc;

  PetscFunctionBeginUser;
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetTolerances(ksp, app->ksp_rtol, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT));
  if (app->variant == VARIANT_FETIDP) {
    PetscCall(KSPSetType(ksp, KSPFETIDP));
  } else {
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPGetPC(ksp, &pc));
    if (app->variant == VARIANT_GAMG) {
      PetscCall(PCSetType(pc, PCGAMG));
    } else if (app->variant == VARIANT_BDDC) {
      PetscCall(PCSetType(pc, PCBDDC));
    } else {
      PetscCall(PCSetType(pc, PCNONE));
    }
  }
  PetscCall(KSPSetFromOptions(ksp));
  if (app->variant != VARIANT_FETIDP) {
    PCType pctype = NULL;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCGetType(pc, &pctype));
    if (pctype) {
      PetscBool is_gamg;
      PetscCall(PetscStrcmp(pctype, PCGAMG, &is_gamg));
      if (is_gamg) PetscCall(AttachNearNullspace(dm, constrained, A));
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

static PetscErrorCode SolveWithFreshKSP(DM dm, AssemblyCtx *actx, AppCtx *app, Mat A, Vec rhs, Vec x, const char *label, PetscInt *its)
{
  KSP       ksp;
  PetscBool reuse_nonzero = PETSC_TRUE;
  KSPConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm), &ksp));
  PetscCall(ConfigureKSP(ksp, dm, app, A, actx->constrained_is));
  PetscCall(KSPSolve(ksp, rhs, x));
  PetscCall(KSPGetIterationNumber(ksp, its));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "%s KSP iterations=%" PetscInt_FMT " reason=%D\n", label, *its, (PetscInt)reason));
  PetscCheck(reason > 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_NOT_CONVERGED, "%s KSP did not converge, reason %D", label, (PetscInt)reason);
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-reuse_linear_solver", &reuse_nonzero, NULL));
  (void)reuse_nonzero;
  PetscCall(KSPDestroy(&ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NewtonSolve(DM dm, AssemblyCtx *actx, AppCtx *app, Mat A, Vec f_ext, Vec u, PetscReal rhs_norm)
{
  Vec            residual, rhs, du, u_trial, r_trial;
  PetscReal      rel = PETSC_MAX_REAL, trial_rel;
  PetscInt       total_linear_its = 0;
  PetscLogDouble t0, t1, assembly_time = 0.0, solve_time = 0.0;

  PetscFunctionBeginUser;
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
    PetscCall(SolveWithFreshKSP(dm, actx, app, A, rhs, du, "Newton correction", &linear_its));
    PetscCall(PetscTime(&t1));
    solve_time += t1 - t0;
    total_linear_its += linear_its;

    if (app->line_search) {
      PetscReal alpha = 1.0;
      while (PETSC_TRUE) {
        PetscCall(VecWAXPY(u_trial, alpha, du, u));
        PetscCall(AssemblePlasticResidualJacobian(actx, app->lambda, u_trial, f_ext, NULL, r_trial, PETSC_FALSE));
        PetscCall(ResidualNormFree(actx, r_trial, rhs_norm, &trial_rel));
        if (trial_rel < rel || alpha <= app->damping_min) {
          PetscCall(VecCopy(u_trial, u));
          PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "  alpha=%8.3e trial_rel=%10.4e\n", (double)alpha, (double)trial_rel));
          break;
        }
        alpha *= 0.5;
      }
    } else {
      PetscCall(VecAXPY(u, 1.0, du));
    }
  }
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                        "Newton summary: final_rel=%10.4e total_linear_its=%" PetscInt_FMT " assembly_time=%.6g solve_time=%.6g\n",
                        (double)rel, total_linear_its, (double)assembly_time, (double)solve_time));

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
  Mat            A = NULL;
  Vec            u = NULL, f_ext = NULL;
  PetscInt       cStart, cEnd, nStart, nEnd, elastic_its;
  PetscReal      rhs_norm, u_norm;
  PetscLogDouble t_start, t_end, t0, t1;

  PetscCall(PetscInitialize(&argc, &argv, NULL, "Standalone pure PETSc P4 plasticity case\n"));
  PetscCall(PetscTime(&t_start));
  PetscCall(ParseOptions(PETSC_COMM_WORLD, &app));
  PetscCall(P4BasisCreate(PETSC_COMM_SELF, &basis));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &app, &basis, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &nStart, &nEnd));
  PetscCall(DMCreateMatrix(dm, &A));
  PetscCall(MatSetBlockSize(A, 3));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(DMCreateGlobalVector(dm, &f_ext));
  PetscCall(AssemblyCtxCreate(dm, &basis, &actx));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "mesh=%s%s refine_levels=%" PetscInt_FMT " local_cells=%" PetscInt_FMT " local_vertices=%" PetscInt_FMT " P4_basis=%" PetscInt_FMT " local_constraints=%" PetscInt_FMT " pc_variant=%s lambda=%.6g\n",
                        app.use_box_mesh ? "generated-box:" : "", app.use_box_mesh ? "unit" : app.mesh, app.refine_levels, cEnd - cStart, nEnd - nStart, basis.n_basis, actx.n_constrained_local, app.variant_name, (double)app.lambda));

  PetscCall(PetscTime(&t0));
  PetscCall(AssembleElasticProblem(&actx, A, f_ext));
  PetscCall(PetscTime(&t1));
  PetscCall(VecNorm(f_ext, NORM_2, &rhs_norm));
  PetscCall(ApplyZeroDirichlet(actx.constrained_is, A, f_ext));
  PetscCall(VecNorm(f_ext, NORM_2, &rhs_norm));
  PetscCall(VecZeroEntries(u));
  PetscCall(SolveWithFreshKSP(dm, &actx, &app, A, f_ext, u, "Elastic initial", &elastic_its));
  PetscCall(VecNorm(u, NORM_2, &u_norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "elastic assembly_time=%.6g rhs_norm=%.6e u_norm=%.6e\n", (double)(t1 - t0), (double)rhs_norm, (double)u_norm));

  PetscCall(NewtonSolve(dm, &actx, &app, A, f_ext, u, rhs_norm));
  PetscCall(VecNorm(u, NORM_2, &u_norm));
  PetscCall(PetscTime(&t_end));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "final displacement_norm=%.8e total_wall_time=%.6g\n", (double)u_norm, (double)(t_end - t_start)));

  PetscCall(AssemblyCtxDestroy(&actx));
  PetscCall(VecDestroy(&f_ext));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&dm));
  PetscCall(P4BasisDestroy(&basis));
  PetscCall(PetscFinalize());
  return 0;
}
