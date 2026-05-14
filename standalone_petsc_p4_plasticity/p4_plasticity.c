#include "assembly.h"
#include "p4_basis.h"

#include <petscksp.h>

typedef enum {
  VARIANT_GAMG,
  VARIANT_BDDC,
  VARIANT_FETIDP,
  VARIANT_PMG,
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
  char      pmg_coarse_pc_type[32];
  char      pmg_smoother_ksp_type[32];
  char      pmg_smoother_pc_type[32];
  PetscInt  pmg_coarse_lu_max_dofs;
  PetscInt  pmg_smoother_max_it;
  char      bddc_graph[32];
  PetscBool bddc_local_solver_auto;
  PetscInt  bddc_exact_local_max_dofs;
} AppCtx;

typedef struct {
  PetscReal      final_rel;
  PetscInt       newton_its;
  PetscInt       total_linear_its;
  PetscLogDouble assembly_time;
  PetscLogDouble solve_time;
} NewtonStats;

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
  PetscCall(PetscStrncpy(app->pmg_coarse_pc_type, "auto", sizeof(app->pmg_coarse_pc_type)));
  PetscCall(PetscStrncpy(app->pmg_smoother_ksp_type, "chebyshev", sizeof(app->pmg_smoother_ksp_type)));
  PetscCall(PetscStrncpy(app->pmg_smoother_pc_type, "jacobi", sizeof(app->pmg_smoother_pc_type)));
  app->pmg_coarse_lu_max_dofs = 50000;
  app->pmg_smoother_max_it    = 2;
  PetscCall(PetscStrncpy(app->bddc_graph, "petsc", sizeof(app->bddc_graph)));
  app->bddc_local_solver_auto    = PETSC_TRUE;
  app->bddc_exact_local_max_dofs = 8000;

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
  PetscCall(PetscOptionsString("-pc_variant", "gamg|bddc|fetidp|pmg|none", NULL, app->variant_name, app->variant_name, sizeof(app->variant_name), NULL));
  PetscCall(PetscOptionsString("-pmg_coarse_pc_type", "auto|hypre|gamg|lu", NULL, app->pmg_coarse_pc_type, app->pmg_coarse_pc_type, sizeof(app->pmg_coarse_pc_type), NULL));
  PetscCall(PetscOptionsInt("-pmg_coarse_lu_max_dofs", "Maximum P1 coarse-grid DOFs allowed for LU", NULL, app->pmg_coarse_lu_max_dofs, &app->pmg_coarse_lu_max_dofs, NULL));
  PetscCall(PetscOptionsString("-pmg_smoother_ksp_type", "PMG smoother KSP type", NULL, app->pmg_smoother_ksp_type, app->pmg_smoother_ksp_type, sizeof(app->pmg_smoother_ksp_type), NULL));
  PetscCall(PetscOptionsString("-pmg_smoother_pc_type", "PMG smoother PC type", NULL, app->pmg_smoother_pc_type, app->pmg_smoother_pc_type, sizeof(app->pmg_smoother_pc_type), NULL));
  PetscCall(PetscOptionsInt("-pmg_smoother_max_it", "PMG smoother iterations per V-cycle", NULL, app->pmg_smoother_max_it, &app->pmg_smoother_max_it, NULL));
  PetscCall(PetscOptionsString("-bddc_graph", "topology|petsc", NULL, app->bddc_graph, app->bddc_graph, sizeof(app->bddc_graph), NULL));
  PetscCall(PetscOptionsBool("-bddc_local_solver_auto", "Choose scalable BDDC local/coarse solvers for large subdomains", NULL, app->bddc_local_solver_auto, &app->bddc_local_solver_auto, NULL));
  PetscCall(PetscOptionsInt("-bddc_exact_local_max_dofs", "Maximum local MATIS rows before switching BDDC subsolves away from LU", NULL, app->bddc_exact_local_max_dofs, &app->bddc_exact_local_max_dofs, NULL));
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
  PetscCall(PetscStrcasecmp(app->bddc_graph, "topology", &flg));
  if (!flg) {
    PetscCall(PetscStrcasecmp(app->bddc_graph, "petsc", &flg));
    PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-bddc_graph must be topology or petsc");
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
  PetscCall(RepairBoundaryFaceSets(cur));
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
        PetscCheck(row >= 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Unexpected negative coordinate closure index");
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

        PetscCheck(row >= 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Unexpected negative coordinate closure index");
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
  PetscCall(SetPrefixedDefault(prefix, "use_edges", "false"));
  PetscCall(SetPrefixedDefault(prefix, "use_faces", "false"));
  PetscCall(PetscStrcasecmp(app->bddc_graph, "topology", &use_topology_graph));
  if (use_topology_graph) PetscCall(SetPrefixedDefault(prefix, "use_local_mat_graph", "false"));
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
  PetscCall(SetPrefixedDefault(prefix, "neumann_ksp_type", "preonly"));
  PetscCall(SetPrefixedDefault(prefix, "neumann_pc_type", pc_type));
  PetscCall(SetPrefixedDefault(prefix, "coarse_ksp_type", "preonly"));
  PetscCall(SetPrefixedDefault(prefix, "coarse_pc_type", pc_type));
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

static PetscErrorCode SetBDDCDofSplittingLocal(PC pc, Mat A)
{
  PetscBool ismatis = PETSC_FALSE;
  Mat       local_mat = NULL;
  IS        fields[3] = {NULL, NULL, NULL};
  PetscInt  nloc;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
  if (!ismatis) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatISGetLocalMat(A, &local_mat));
  PetscCall(MatGetSize(local_mat, &nloc, NULL));
  PetscCheck(nloc % 3 == 0, PetscObjectComm((PetscObject)pc), PETSC_ERR_PLIB, "Expected MATIS local rows divisible by 3");
  for (PetscInt comp = 0; comp < 3; ++comp) PetscCall(ISCreateStride(PetscObjectComm((PetscObject)pc), nloc / 3, comp, 3, &fields[comp]));
  PetscCall(PCBDDCSetDofsSplittingLocal(pc, 3, fields));
  for (PetscInt comp = 0; comp < 3; ++comp) PetscCall(ISDestroy(&fields[comp]));
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

static PetscErrorCode ConfigureBDDCTopologyGraph(PC pc, DM dm, P4Basis *basis, Mat A)
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
  PetscCall(BuildLocalConstrainedIS(dm, actx, &dirichlet_local));
  PetscCall(PCBDDCSetDirichletBoundariesLocal(pc, dirichlet_local));
  PetscCall(ISDestroy(&dirichlet_local));
  /*
    PETSc 3.24 BDDC still has "TODO: support for blocked" in its coordinate
    import path and checks against the scalar local pmat size. GAMG uses
    blocked coordinates; BDDC/FETI-DP need scalar-equation coordinates here.
  */
  PetscCall(BuildOwnedDofCoordinates(dm, actx->basis, &ncoords, &coords));
  PetscCall(PCSetCoordinates(pc, 3, ncoords, coords));
  PetscCall(PetscFree(coords));
  PetscCall(SetBDDCDofSplittingLocal(pc, A));
  PetscCall(PetscStrcasecmp(app->bddc_graph, "topology", &use_topology_graph));
  if (use_topology_graph) {
    PetscCall(ConfigureBDDCTopologyGraph(pc, dm, actx->basis, A));
    PetscCall(ConfigureBDDCPrimalVertices(pc, dm, actx->basis));
  }
  PetscCall(AttachNearNullspace(dm, actx->constrained_is, A));
  PetscCall(AttachLocalNearNullspace(dm, actx->basis, A));
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

static PetscErrorCode CreateSameMeshLevelDM(DM fine_dm, P4Basis *basis, DM *level_dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMClone(fine_dm, level_dm));
  PetscCall(DMClearDS(*level_dm));
  PetscCall(DMClearFields(*level_dm));
  PetscCall(DMSetField(*level_dm, 0, NULL, (PetscObject)basis->fe_vector));
  PetscCall(DMCreateDS(*level_dm));
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

        if (row < rlo || row >= rhi) continue;
        for (PetscInt cb = 0; cb < coarse_basis->n_basis; ++cb) {
          const PetscScalar val = phi[fb * coarse_basis->n_basis + cb];
          const PetscInt    col = coarse_idx[3 * cb + comp];

          if (PetscAbsScalar(val) <= 1.0e-12) continue;
          PetscCheck(col >= 0, comm, PETSC_ERR_PLIB, "Unexpected negative coarse interpolation column");
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
  }

  PetscCall(PetscStrcasecmp(app->pmg_coarse_pc_type, "auto", &flg));
  if (flg) {
    if (size == 1 && coarse_dofs <= app->pmg_coarse_lu_max_dofs) PetscCall(PetscStrncpy(coarse_pc, "lu", coarse_pc_len));
#if defined(PETSC_HAVE_HYPRE)
    else PetscCall(PetscStrncpy(coarse_pc, "hypre", coarse_pc_len));
#else
    else PetscCall(PetscStrncpy(coarse_pc, "gamg", coarse_pc_len));
#endif
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

static PetscErrorCode ConfigurePMG(PC pc, DM dm, AssemblyCtx *actx, AppCtx *app)
{
  P4Basis p1_basis, p2_basis;
  DM      dm_p1 = NULL, dm_p2 = NULL;
  Mat     P21 = NULL, P42 = NULL;
  KSP     coarse = NULL, smoother = NULL;
  PC      coarse_pc = NULL, smoother_pc = NULL;
  char    coarse_pc_type[32];

  PetscFunctionBeginUser;
  PetscCall(P4BasisCreateDegree(PETSC_COMM_SELF, 1, &p1_basis));
  PetscCall(P4BasisCreateDegree(PETSC_COMM_SELF, 2, &p2_basis));
  PetscCall(CreateSameMeshLevelDM(dm, &p1_basis, &dm_p1));
  PetscCall(CreateSameMeshLevelDM(dm, &p2_basis, &dm_p2));
  PetscCall(BuildInterpolationMatrix(dm_p2, &p2_basis, dm_p1, &p1_basis, &P21));
  PetscCall(BuildInterpolationMatrix(dm, actx->basis, dm_p2, &p2_basis, &P42));
  PetscCall(ChoosePMGCoarsePC(app, dm_p1, coarse_pc_type, sizeof(coarse_pc_type)));

  PetscCall(PCSetType(pc, PCMG));
  PetscCall(PCMGSetLevels(pc, 3, NULL));
  PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
  PetscCall(PCMGSetCycleType(pc, PC_MG_CYCLE_V));
  PetscCall(PCMGSetInterpolation(pc, 1, P21));
  PetscCall(PCMGSetInterpolation(pc, 2, P42));
  PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));

  PetscCall(PCMGGetCoarseSolve(pc, &coarse));
  PetscCall(KSPSetType(coarse, KSPPREONLY));
  PetscCall(KSPGetPC(coarse, &coarse_pc));
  PetscCall(PCSetType(coarse_pc, coarse_pc_type));
#if defined(PETSC_HAVE_HYPRE)
  {
    PetscBool is_hypre;
    PetscCall(PetscStrcasecmp(coarse_pc_type, "hypre", &is_hypre));
    if (is_hypre) PetscCall(PCHYPRESetType(coarse_pc, "boomeramg"));
  }
#endif
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

static PetscErrorCode ConfigureKSP(KSP ksp, DM dm, AssemblyCtx *actx, AppCtx *app, Mat A)
{
  PC pc;

  PetscFunctionBeginUser;
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetTolerances(ksp, app->ksp_rtol, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT));
  if (app->variant == VARIANT_BDDC) {
    PetscCall(SetBDDCConstraintDefaults(app, "pc_bddc_"));
    PetscCall(ConfigureBDDCAutoSolvers(app, A, "pc_bddc_"));
  } else if (app->variant == VARIANT_FETIDP) {
    PetscCall(SetBDDCConstraintDefaults(app, "fetidp_bddc_pc_bddc_"));
    PetscCall(ConfigureBDDCAutoSolvers(app, A, "fetidp_bddc_pc_bddc_"));
  }
  if (app->variant == VARIANT_FETIDP) {
    PetscCall(KSPSetType(ksp, KSPFETIDP));
  } else {
    PetscCall(KSPSetType(ksp, app->variant == VARIANT_PMG ? KSPFGMRES : KSPCG));
    PetscCall(KSPGetPC(ksp, &pc));
    if (app->variant == VARIANT_GAMG) {
      PetscCall(PCSetType(pc, PCGAMG));
    } else if (app->variant == VARIANT_BDDC) {
      PetscCall(PCSetType(pc, PCBDDC));
    } else if (app->variant == VARIANT_PMG) {
      PetscCall(ConfigurePMG(pc, dm, actx, app));
    } else {
      PetscCall(PCSetType(pc, PCNONE));
    }
  }
  PetscCall(KSPSetFromOptions(ksp));
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
        PetscInt   ncoords;
        PetscReal *coords = NULL;

        PetscCall(BuildOwnedBlockCoordinates(dm, actx->basis, &ncoords, &coords));
        PetscCall(PCSetCoordinates(pc, 3, ncoords, coords));
        PetscCall(PetscFree(coords));
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

static PetscErrorCode SolveWithFreshKSP(DM dm, AssemblyCtx *actx, AppCtx *app, Mat A, Vec rhs, Vec x, const char *label, PetscInt *its)
{
  KSP       ksp;
  PetscBool reuse_nonzero = PETSC_TRUE;
  KSPConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm), &ksp));
  PetscCall(ConfigureKSP(ksp, dm, actx, app, A));
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

static PetscErrorCode NewtonSolve(DM dm, AssemblyCtx *actx, AppCtx *app, Mat A, Vec f_ext, Vec u, PetscReal rhs_norm, NewtonStats *stats)
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
    PetscCall(SolveWithFreshKSP(dm, actx, app, A, rhs, du, "Newton correction", &linear_its));
    PetscCall(PetscTime(&t1));
    solve_time += t1 - t0;
    total_linear_its += linear_its;
    ++newton_its;

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
  PetscCall(P4BasisCreate(PETSC_COMM_SELF, &basis));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &app, &basis, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &nStart, &nEnd));
  PetscCall(DMCreateMatrix(dm, &A));
  PetscCall(MatSetBlockSize(A, 3));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(DMCreateGlobalVector(dm, &f_ext));
  PetscCall(VecGetLocalSize(u, &local_dofs));
  PetscCall(VecGetSize(u, &global_dofs));
  PetscCall(AssemblyCtxCreate(dm, &basis, &actx));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "mesh=%s%s refine_levels=%" PetscInt_FMT " local_cells=%" PetscInt_FMT " local_vertices=%" PetscInt_FMT " local_dofs=%" PetscInt_FMT " global_dofs=%" PetscInt_FMT " P4_basis=%" PetscInt_FMT " owned_constraints=%" PetscInt_FMT " global_constraints=%" PetscInt_FMT " pc_variant=%s lambda=%.6g\n",
                        app.use_box_mesh ? "generated-box:" : "", app.use_box_mesh ? "unit" : app.mesh, app.refine_levels, cEnd - cStart, nEnd - nStart, local_dofs, global_dofs, basis.n_basis, actx.n_constrained_local, actx.n_constrained_all, app.variant_name, (double)app.lambda));

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
  PetscCall(SolveWithFreshKSP(dm, &actx, &app, A, f_ext, u, "Elastic initial", &elastic_its));
  PetscCall(PetscTime(&t1));
  elastic_solve_time = t1 - t0;
  PetscCall(VecNorm(u, NORM_2, &u_norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "elastic solve_time=%.6g u_norm=%.6e\n", (double)elastic_solve_time, (double)u_norm));

  PetscCall(NewtonSolve(dm, &actx, &app, A, f_ext, u, rhs_norm, &newton_stats));
  PetscCall(VecNorm(u, NORM_2, &u_norm));
  PetscCall(PetscTime(&t_end));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "final displacement_norm=%.8e total_wall_time=%.6g\n", (double)u_norm, (double)(t_end - t_start)));
  {
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "RESULT variant=%s ranks=%d global_dofs=%" PetscInt_FMT " elastic_its=%" PetscInt_FMT " newton_its=%" PetscInt_FMT " newton_linear_its=%" PetscInt_FMT " total_linear_its=%" PetscInt_FMT " elastic_assembly_time=%.6g elastic_solve_time=%.6g newton_assembly_time=%.6g newton_solve_time=%.6g wall_time=%.6g final_rel=%.6e\n",
                          app.variant_name, size, global_dofs, elastic_its, newton_stats.newton_its, newton_stats.total_linear_its, elastic_its + newton_stats.total_linear_its,
                          (double)elastic_assembly_time, (double)elastic_solve_time, (double)newton_stats.assembly_time, (double)newton_stats.solve_time, (double)(t_end - t_start),
                          (double)newton_stats.final_rel));
  }

  PetscCall(AssemblyCtxDestroy(&actx));
  PetscCall(VecDestroy(&f_ext));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&dm));
  PetscCall(P4BasisDestroy(&basis));
  PetscCall(PetscFinalize());
  return 0;
}
