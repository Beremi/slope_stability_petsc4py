#include "p4_elasticity_common.h"
#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscfe.h>
#include <petsclog.h>
#include <stdint.h>

typedef struct {
  PetscInt  faces[3];
  PetscInt  degree;
  PetscReal young;
  PetscReal poisson;
  PetscReal pressure;
  PetscReal gravity;
  char      variant[32];
  char      mesh_bc_mode[32];
  char      mesh[PETSC_MAX_PATH_LEN];
  char      case_name[32];
  P4ElasticityCaseKind case_kind;
  PetscBool use_mesh;
  PetscBool configure_bddc_metadata;
  PetscBool inspect_layout;
  PetscInt  pmg_coarse_redundant_group_size;
  PetscBool pmg_coarse_gamg_aggressive_square_graph;
  PetscReal matis_duplication_limit;
  PetscInt  material_sweep_count;
  PetscInt  material_sweep_seed;
  PetscReal material_sweep_young_min;
  PetscReal material_sweep_young_max;
  PetscReal material_sweep_poisson_min;
  PetscReal material_sweep_poisson_max;
  PetscBool material_sweep_reuse_pc;
  char      material_sweep_mode[32];
} AppCtx;

enum {
  MARKER_X_MAX = 1,
  MARKER_X_MIN = 2,
  MARKER_Z_MIN = 3,
  MARKER_Z_MAX = 4,
  MARKER_BASE  = 5,
  MARKER_Y_MAX = 6
};

static PetscReal g_mu = 0.3846153846153846, g_lambda = 0.5769230769230769, g_pressure = 1.0, g_gravity = 0.0;

static const PetscInt  boundaryValues[] = {MARKER_X_MAX, MARKER_X_MIN, MARKER_Z_MIN, MARKER_Z_MAX, MARKER_BASE, MARKER_Y_MAX};
static const char     *boundaryNames[]  = {"x_max", "x_min", "z_min", "z_max", "base", "y_max"};

static PetscErrorCode SetElasticityMaterial(PetscReal young, PetscReal poisson, PetscReal pressure, PetscReal gravity)
{
  PetscFunctionBeginUser;
  PetscCheck(young > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Young's modulus must be positive");
  PetscCheck(poisson > -1.0 && poisson < 0.5, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Poisson ratio must lie in (-1,0.5)");
  g_mu       = young / (2.0 * (1.0 + poisson));
  g_lambda   = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
  g_pressure = pressure;
  g_gravity  = gravity;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static uint64_t SplitMix64(uint64_t *state)
{
  uint64_t z;

  *state += UINT64_C(0x9e3779b97f4a7c15);
  z = *state;
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}

static PetscReal UnitRandom53(uint64_t value)
{
  return (PetscReal)((value >> 11) * (1.0 / 9007199254740992.0));
}

static PetscErrorCode MaterialSweepSample(const AppCtx *app, PetscInt sample, PetscReal *young, PetscReal *poisson)
{
  uint64_t  state;
  PetscReal uE, uNu, logE0, logE1;

  PetscFunctionBeginUser;
  if (app->material_sweep_count <= 1) {
    *young   = app->young;
    *poisson = app->poisson;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  state = ((uint64_t)(PetscMax(app->material_sweep_seed, 0) + 1)) ^ ((uint64_t)(sample + 1) * UINT64_C(0xd1b54a32d192ed03));
  uE    = UnitRandom53(SplitMix64(&state));
  uNu   = UnitRandom53(SplitMix64(&state));
  logE0 = PetscLogReal(app->material_sweep_young_min);
  logE1 = PetscLogReal(app->material_sweep_young_max);
  *young   = PetscExpReal(logE0 + uE * (logE1 - logE0));
  *poisson = app->material_sweep_poisson_min + uNu * (app->material_sweep_poisson_max - app->material_sweep_poisson_min);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IsDDVariant(const AppCtx *app, PetscBool *is_dd)
{
  PetscBool is_bddc = PETSC_FALSE, is_fetidp = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscStrcasecmp(app->variant, "bddc", &is_bddc));
  PetscCall(PetscStrcasecmp(app->variant, "fetidp", &is_fetidp));
  *is_dd = (PetscBool)(is_bddc || is_fetidp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IsPMGVariant(const AppCtx *app, PetscBool *is_pmg)
{
  PetscFunctionBeginUser;
  PetscCall(PetscStrcasecmp(app->variant, "pmg", is_pmg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IsSweepMode(const AppCtx *app, const char mode[], PetscBool *match)
{
  PetscFunctionBeginUser;
  PetscCall(PetscStrcasecmp(app->material_sweep_mode, mode, match));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static const char *BoundaryLabelName(const AppCtx *app)
{
  return app->use_mesh ? "boundary_marker" : "marker";
}

static PetscErrorCode ZeroDisplacement(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar u[], void *ctx)
{
  PetscFunctionBeginUser;
  for (PetscInt c = 0; c < Nc; ++c) u[c] = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void ResidualBody(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (PetscInt c = 0; c < dim; ++c) f0[c] = 0.0;
  if (dim > 1) f0[1] = -g_gravity;
}

static void ResidualStress(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscReal trace = 0.0;

  for (PetscInt i = 0; i < dim; ++i) trace += PetscRealPart(u_x[i * dim + i]);
  for (PetscInt i = 0; i < dim; ++i) {
    for (PetscInt j = 0; j < dim; ++j) f1[i * dim + j] = g_mu * (u_x[i * dim + j] + u_x[j * dim + i]);
    f1[i * dim + i] += g_lambda * trace;
  }
}

static void JacobianElasticity(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  for (PetscInt i = 0; i < dim; ++i) {
    for (PetscInt j = 0; j < dim; ++j) {
      for (PetscInt k = 0; k < dim; ++k) {
        for (PetscInt l = 0; l < dim; ++l) {
          g3[((i * dim + j) * dim + k) * dim + l] = (i == j && k == l ? g_lambda : 0.0) + (i == k && j == l ? g_mu : 0.0) + (i == l && j == k ? g_mu : 0.0);
        }
      }
    }
  }
}

static void TopTraction(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 0.0;
  f0[1] = 0.0;
  f0[2] = -g_pressure;
}

static void BoundaryNoFlux(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  for (PetscInt c = 0; c < dim * dim; ++c) f1[c] = 0.0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, const P4ElasticityCase *spec, AppCtx *app)
{
  PetscBool flg;
  PetscBool meshSet = PETSC_FALSE, pressureSet = PETSC_FALSE, gravitySet = PETSC_FALSE;
  PetscBool sweepModeSet = PETSC_FALSE, sweepReuseSet = PETSC_FALSE;
  PetscInt  nfaces  = 3;

  PetscFunctionBeginUser;
  PetscCheck(spec, comm, PETSC_ERR_ARG_NULL, "Missing elasticity case spec");
  app->faces[0] = 2;
  app->faces[1] = 2;
  app->faces[2] = 2;
  app->degree   = 4;
  app->young    = 1.0;
  app->poisson  = 0.30;
  app->pressure = spec->default_pressure;
  app->gravity  = spec->default_gravity;
  app->case_kind = spec->kind;
  app->use_mesh = (PetscBool)(spec->kind == P4_ELASTICITY_L1_MESH);
  app->configure_bddc_metadata = PETSC_FALSE;
  app->inspect_layout           = PETSC_FALSE;
  app->pmg_coarse_redundant_group_size        = 64;
  app->pmg_coarse_gamg_aggressive_square_graph = PETSC_FALSE;
  app->matis_duplication_limit  = 1.25;
  app->material_sweep_count       = 1;
  app->material_sweep_seed        = 1729;
  app->material_sweep_young_min   = 0.5;
  app->material_sweep_young_max   = 2.0;
  app->material_sweep_poisson_min = 0.20;
  app->material_sweep_poisson_max = 0.45;
  app->material_sweep_reuse_pc    = PETSC_TRUE;
  PetscCall(PetscStrncpy(app->material_sweep_mode, "reuse_pc", sizeof(app->material_sweep_mode)));
  PetscCall(PetscStrncpy(app->case_name, spec->name ? spec->name : "unknown", sizeof(app->case_name)));
  PetscCall(PetscStrncpy(app->mesh, spec->default_mesh ? spec->default_mesh : "", sizeof(app->mesh)));
  PetscCall(PetscStrncpy(app->variant, "gamg", sizeof(app->variant)));
  PetscCall(PetscStrncpy(app->mesh_bc_mode, spec->default_bc_mode ? spec->default_bc_mode : "rollers", sizeof(app->mesh_bc_mode)));

  PetscOptionsBegin(comm, NULL, "P4 elasticity options", NULL);
  if (app->use_mesh) {
    PetscCall(PetscOptionsString("-mesh", "Mesh file to read with DMPlexCreateFromFile", NULL, app->mesh, app->mesh, sizeof(app->mesh), &meshSet));
    PetscCall(PetscOptionsString("-mesh_bc_mode", "For mesh case: rollers|base_only|full_sides", NULL, app->mesh_bc_mode, app->mesh_bc_mode, sizeof(app->mesh_bc_mode), NULL));
  } else {
    PetscCall(PetscOptionsIntArray("-cube_faces", "Hex subdivisions in x,y,z before tetrahedralization", NULL, app->faces, &nfaces, &flg));
    PetscCheck(!flg || nfaces == 3, comm, PETSC_ERR_ARG_SIZ, "-cube_faces expects exactly three integers");
  }
  PetscCall(PetscOptionsInt("-degree", "Lagrange FE degree", NULL, app->degree, &app->degree, NULL));
  PetscCall(PetscOptionsReal("-young", "Young's modulus", NULL, app->young, &app->young, NULL));
  PetscCall(PetscOptionsReal("-poisson", "Poisson ratio", NULL, app->poisson, &app->poisson, NULL));
  PetscCall(PetscOptionsReal("-pressure", "Downward top traction magnitude for generated cube", NULL, app->pressure, &app->pressure, &pressureSet));
  PetscCall(PetscOptionsReal("-gravity", "Downward body force in the mesh vertical y-direction", NULL, app->gravity, &app->gravity, &gravitySet));
  PetscCall(PetscOptionsString("-pc_variant", "gamg|pmg|bddc|fetidp|none", NULL, app->variant, app->variant, sizeof(app->variant), NULL));
  PetscCall(PetscOptionsInt("-pmg_coarse_redundant_group_size", "Use PCREDUNDANT for PMG P1 coarse GAMG on groups of this many ranks; 0 disables", NULL, app->pmg_coarse_redundant_group_size, &app->pmg_coarse_redundant_group_size, NULL));
  PetscCall(PetscOptionsBool("-pmg_coarse_gamg_aggressive_square_graph", "Use PETSc GAMG square-graph aggressive coarsening on PMG P1 coarse solves", NULL, app->pmg_coarse_gamg_aggressive_square_graph, &app->pmg_coarse_gamg_aggressive_square_graph, NULL));
  PetscCall(PetscOptionsBool("-configure_bddc_metadata", "Experimental local Dirichlet/splitting metadata for PCBDDC/KSPFETIDP", NULL, app->configure_bddc_metadata, &app->configure_bddc_metadata, NULL));
  PetscCall(PetscOptionsBool("-inspect_layout", "Print constrained section/MATIS layout diagnostics and exit", NULL, app->inspect_layout, &app->inspect_layout, NULL));
  PetscCall(PetscOptionsReal("-matis_duplication_limit", "Abort L1 BDDC/FETI-DP solves when MATIS duplicated rows / global rows reaches this value", NULL, app->matis_duplication_limit, &app->matis_duplication_limit, NULL));
  PetscCall(PetscOptionsInt("-material_sweep_count", "Number of randomly sampled elastic material systems to solve on the same mesh", NULL, app->material_sweep_count, &app->material_sweep_count, NULL));
  PetscCall(PetscOptionsInt("-material_sweep_seed", "Deterministic seed for -material_sweep_count > 1", NULL, app->material_sweep_seed, &app->material_sweep_seed, NULL));
  PetscCall(PetscOptionsReal("-material_sweep_young_min", "Minimum Young's modulus for the material sweep", NULL, app->material_sweep_young_min, &app->material_sweep_young_min, NULL));
  PetscCall(PetscOptionsReal("-material_sweep_young_max", "Maximum Young's modulus for the material sweep", NULL, app->material_sweep_young_max, &app->material_sweep_young_max, NULL));
  PetscCall(PetscOptionsReal("-material_sweep_poisson_min", "Minimum Poisson ratio for the material sweep", NULL, app->material_sweep_poisson_min, &app->material_sweep_poisson_min, NULL));
  PetscCall(PetscOptionsReal("-material_sweep_poisson_max", "Maximum Poisson ratio for the material sweep", NULL, app->material_sweep_poisson_max, &app->material_sweep_poisson_max, NULL));
  PetscCall(PetscOptionsBool("-material_sweep_reuse_pc", "Compatibility shortcut: false maps to -material_sweep_mode refresh, true maps to reuse_pc unless -material_sweep_mode is set", NULL, app->material_sweep_reuse_pc, &app->material_sweep_reuse_pc, &sweepReuseSet));
  PetscCall(PetscOptionsString("-material_sweep_mode", "Material sweep setup mode: fresh|refresh|reuse_pc", NULL, app->material_sweep_mode, app->material_sweep_mode, sizeof(app->material_sweep_mode), &sweepModeSet));
  PetscOptionsEnd();

  if (!sweepModeSet && sweepReuseSet) PetscCall(PetscStrncpy(app->material_sweep_mode, app->material_sweep_reuse_pc ? "reuse_pc" : "refresh", sizeof(app->material_sweep_mode)));

  (void)meshSet;
  (void)pressureSet;
  (void)gravitySet;
  PetscCheck(!app->use_mesh || app->mesh[0], comm, PETSC_ERR_ARG_WRONG, "Mesh case requires a non-empty -mesh path");
  for (PetscInt d = 0; d < 3; ++d) PetscCheck(app->faces[d] > 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-cube_faces entries must be positive");
  PetscCheck(app->degree >= 1, comm, PETSC_ERR_ARG_OUTOFRANGE, "-degree must be >= 1");
  PetscCheck(app->pmg_coarse_redundant_group_size >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "-pmg_coarse_redundant_group_size must be nonnegative");
  PetscCheck(app->poisson > -1.0 && app->poisson < 0.5, comm, PETSC_ERR_ARG_OUTOFRANGE, "-poisson must lie in (-1,0.5)");
  PetscCheck(app->material_sweep_count >= 1, comm, PETSC_ERR_ARG_OUTOFRANGE, "-material_sweep_count must be positive");
  PetscCheck(app->material_sweep_young_min > 0.0 && app->material_sweep_young_max >= app->material_sweep_young_min, comm, PETSC_ERR_ARG_OUTOFRANGE,
             "-material_sweep_young_min/max must be positive and ordered");
  PetscCheck(app->material_sweep_poisson_min > -1.0 && app->material_sweep_poisson_max < 0.5 && app->material_sweep_poisson_max >= app->material_sweep_poisson_min, comm,
             PETSC_ERR_ARG_OUTOFRANGE, "-material_sweep_poisson_min/max must lie in (-1,0.5) and be ordered");
  {
    PetscBool sweepFresh, sweepRefresh, sweepReuse;

    PetscCall(IsSweepMode(app, "fresh", &sweepFresh));
    PetscCall(IsSweepMode(app, "refresh", &sweepRefresh));
    PetscCall(IsSweepMode(app, "reuse_pc", &sweepReuse));
    PetscCheck(sweepFresh || sweepRefresh || sweepReuse, comm, PETSC_ERR_ARG_WRONG, "-material_sweep_mode must be fresh, refresh, or reuse_pc");
    app->material_sweep_reuse_pc = sweepReuse;
  }
  if (app->use_mesh) {
    PetscBool isRollers, isBaseOnly, isFullSides;
    PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "rollers", &isRollers));
    PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "base_only", &isBaseOnly));
    PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "full_sides", &isFullSides));
    PetscCheck(isRollers || isBaseOnly || isFullSides, comm, PETSC_ERR_ARG_WRONG, "-mesh_bc_mode must be rollers, base_only, or full_sides");
  }
  PetscCall(SetElasticityMaterial(app->young, app->poisson, app->pressure, app->gravity));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetDDPartitionerDefault(MPI_Comm comm, const AppCtx *app)
{
  PetscBool   is_dd, user_set;
  PetscMPIInt ranks;
  const char *part_type = NULL;

  PetscFunctionBeginUser;
  PetscCall(IsDDVariant(app, &is_dd));
  PetscCallMPI(MPI_Comm_size(comm, &ranks));
  if (!app->use_mesh || !is_dd || ranks <= 1) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscOptionsHasName(NULL, NULL, "-petscpartitioner_type", &user_set));
  if (user_set) PetscFunctionReturn(PETSC_SUCCESS);
#if defined(PETSC_HAVE_PARMETIS)
  part_type = "parmetis";
#elif defined(PETSC_HAVE_PTSCOTCH)
  part_type = "ptscotch";
#endif
  if (part_type) {
    PetscCall(PetscOptionsSetValue(NULL, "-petscpartitioner_type", part_type));
    PetscCall(PetscPrintf(comm, "L1 BDDC/FETI-DP partitioner default: -petscpartitioner_type %s\n", part_type));
  } else {
    PetscCall(PetscPrintf(comm, "L1 BDDC/FETI-DP partitioner warning: PETSc has no ParMETIS/PTScotch; using available default\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscInt VertexIndex(PetscInt nx, PetscInt ny, PetscInt i, PetscInt j, PetscInt k)
{
  return (k * (ny + 1) + j) * (nx + 1) + i;
}

static PetscErrorCode CreateMesh(MPI_Comm comm, const AppCtx *app, DM *dm)
{
  const PetscInt  nx = app->faces[0], ny = app->faces[1], nz = app->faces[2];
  const PetscInt  numVertices = (nx + 1) * (ny + 1) * (nz + 1), numCells = 6 * nx * ny * nz;
  PetscInt       *cells;
  PetscReal      *coords;
  PetscPartitioner part;
  DM               dist = NULL;

  PetscFunctionBeginUser;
  if (app->use_mesh) {
    DM cur;

    PetscCall(DMPlexCreateFromFile(comm, app->mesh, NULL, PETSC_TRUE, &cur));
    PetscCall(DMSetFromOptions(cur));
    PetscCall(DMPlexGetPartitioner(cur, &part));
    PetscCall(PetscPartitionerSetFromOptions(part));
    PetscCall(DMPlexDistribute(cur, 0, NULL, &dist));
    if (dist) {
      PetscCall(DMDestroy(&cur));
      cur = dist;
      PetscCall(DMSetFromOptions(cur));
    }
    PetscCall(DMGetCoordinatesLocalSetUp(cur));
    *dm = cur;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscMalloc1((size_t)numCells * 4, &cells));
  PetscCall(PetscMalloc1((size_t)numVertices * 3, &coords));
  for (PetscInt k = 0, v = 0; k <= nz; ++k) {
    for (PetscInt j = 0; j <= ny; ++j) {
      for (PetscInt i = 0; i <= nx; ++i, ++v) {
        coords[3 * v + 0] = ((PetscReal)i) / ((PetscReal)nx);
        coords[3 * v + 1] = ((PetscReal)j) / ((PetscReal)ny);
        coords[3 * v + 2] = ((PetscReal)k) / ((PetscReal)nz);
      }
    }
  }
  for (PetscInt k = 0, c = 0; k < nz; ++k) {
    for (PetscInt j = 0; j < ny; ++j) {
      for (PetscInt i = 0; i < nx; ++i) {
        const PetscInt v000 = VertexIndex(nx, ny, i, j, k);
        const PetscInt v100 = VertexIndex(nx, ny, i + 1, j, k);
        const PetscInt v110 = VertexIndex(nx, ny, i + 1, j + 1, k);
        const PetscInt v010 = VertexIndex(nx, ny, i, j + 1, k);
        const PetscInt v001 = VertexIndex(nx, ny, i, j, k + 1);
        const PetscInt v101 = VertexIndex(nx, ny, i + 1, j, k + 1);
        const PetscInt v111 = VertexIndex(nx, ny, i + 1, j + 1, k + 1);
        const PetscInt v011 = VertexIndex(nx, ny, i, j + 1, k + 1);
        const PetscInt tet[6][4] = {{v000, v100, v110, v111}, {v000, v110, v010, v111}, {v000, v010, v011, v111}, {v000, v011, v001, v111}, {v000, v001, v101, v111}, {v000, v101, v100, v111}};

        for (PetscInt t = 0; t < 6; ++t, ++c) {
          for (PetscInt p = 0; p < 4; ++p) cells[4 * c + p] = tet[t][p];
        }
      }
    }
  }
  PetscCall(DMPlexCreateFromCellListPetsc(comm, 3, numCells, numVertices, 4, PETSC_TRUE, cells, 3, coords, dm));
  PetscCall(PetscFree(cells));
  PetscCall(PetscFree(coords));
  PetscCall(PetscObjectSetName((PetscObject)*dm, "tetra_cube"));
  PetscCall(DMPlexGetPartitioner(*dm, &part));
  PetscCall(PetscPartitionerSetFromOptions(part));
  PetscCall(DMPlexDistribute(*dm, 0, NULL, &dist));
  if (dist) {
    PetscCall(DMDestroy(dm));
    *dm = dist;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MarkBoundaryFaces(DM dm)
{
  MPI_Comm     comm;
  DM           cdm;
  PetscSection csec;
  Vec          coords;
  DMLabel      label;
  PetscReal    local_min[3] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
  PetscReal    local_max[3] = {-PETSC_MAX_REAL, -PETSC_MAX_REAL, -PETSC_MAX_REAL};
  PetscReal    global_min[3], global_max[3], scale = 1.0, tol;
  PetscInt     vStart, vEnd, fStart, fEnd;

  PetscFunctionBeginUser;
  comm = PetscObjectComm((PetscObject)dm);
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinateSection(dm, &csec));
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCheck(coords, comm, PETSC_ERR_ARG_WRONGSTATE, "Mesh has no local coordinates");

  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    PetscScalar *xyz  = NULL;
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

  PetscCall(DMCreateLabel(dm, "marker"));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  for (PetscInt f = fStart; f < fEnd; ++f) {
    PetscInt  supportSize;
    PetscReal vol, centroid[3], normal[3];

    PetscCall(DMPlexGetSupportSize(dm, f, &supportSize));
    if (supportSize != 1) continue;
    PetscCall(DMPlexComputeCellGeometryFVM(dm, f, &vol, centroid, normal));
    if (PetscAbsReal(centroid[0] - global_max[0]) <= tol) PetscCall(DMLabelSetValue(label, f, MARKER_X_MAX));
    else if (PetscAbsReal(centroid[0] - global_min[0]) <= tol) PetscCall(DMLabelSetValue(label, f, MARKER_X_MIN));
    else if (PetscAbsReal(centroid[2] - global_min[2]) <= tol) PetscCall(DMLabelSetValue(label, f, MARKER_Z_MIN));
    else if (PetscAbsReal(centroid[2] - global_max[2]) <= tol) PetscCall(DMLabelSetValue(label, f, MARKER_Z_MAX));
    else if (PetscAbsReal(centroid[1] - global_min[1]) <= tol) PetscCall(DMLabelSetValue(label, f, MARKER_BASE));
    else if (PetscAbsReal(centroid[1] - global_max[1]) <= tol) PetscCall(DMLabelSetValue(label, f, MARKER_Y_MAX));
  }
  PetscCall(DMPlexLabelComplete(dm, label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildBoundaryMarkerFromFaceSets(DM dm)
{
  DMLabel faceSets = NULL, marker = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMGetLabel(dm, "Face Sets", &faceSets));
  PetscCheck(faceSets, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE,
             "Gmsh mesh has no 'Face Sets' label. Run with -dm_view ::ascii_info_detail or check the .msh PhysicalNames/Entities.");
  PetscCall(DMCreateLabel(dm, "boundary_marker"));
  PetscCall(DMGetLabel(dm, "boundary_marker", &marker));
  for (PetscInt k = 0; k < (PetscInt)(sizeof(boundaryValues) / sizeof(boundaryValues[0])); ++k) {
    IS              points = NULL;
    const PetscInt *idx;
    PetscInt        n;

    PetscCall(DMLabelGetStratumIS(faceSets, boundaryValues[k], &points));
    if (!points) continue;
    PetscCall(ISGetLocalSize(points, &n));
    PetscCall(ISGetIndices(points, &idx));
    for (PetscInt i = 0; i < n; ++i) PetscCall(DMLabelSetValue(marker, idx[i], boundaryValues[k]));
    PetscCall(ISRestoreIndices(points, &idx));
    PetscCall(ISDestroy(&points));
  }
  PetscCall(DMPlexLabelComplete(dm, marker));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReportBoundaryCounts(DM dm, const AppCtx *app)
{
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);
  DMLabel  label = NULL;
  PetscInt counts[sizeof(boundaryValues) / sizeof(boundaryValues[0])];

  PetscFunctionBeginUser;
  PetscCall(DMGetLabel(dm, BoundaryLabelName(app), &label));
  PetscCheck(label, comm, PETSC_ERR_ARG_WRONGSTATE, "Missing boundary label %s", BoundaryLabelName(app));
  for (PetscInt k = 0; k < (PetscInt)(sizeof(boundaryValues) / sizeof(boundaryValues[0])); ++k) {
    IS       is = NULL;
    PetscInt nloc = 0;

    PetscCall(DMLabelGetStratumIS(label, boundaryValues[k], &is));
    if (is) PetscCall(ISGetLocalSize(is, &nloc));
    PetscCallMPI(MPI_Allreduce(&nloc, &counts[k], 1, MPIU_INT, MPI_SUM, comm));
    PetscCall(PetscPrintf(comm, "BOUNDARY_COUNT name=%s tag=%" PetscInt_FMT " points=%" PetscInt_FMT "\n", boundaryNames[k], boundaryValues[k], counts[k]));
    PetscCall(ISDestroy(&is));
  }

  if (app->use_mesh) {
    PetscBool isRollers, isFullSides;

    PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "rollers", &isRollers));
    PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "full_sides", &isFullSides));
    PetscCheck(counts[4] > 0, comm, PETSC_ERR_ARG_WRONGSTATE, "L1 boundary label has no base points");
    if (isRollers || isFullSides) {
      PetscCheck(counts[0] > 0 && counts[1] > 0 && counts[2] > 0 && counts[3] > 0, comm, PETSC_ERR_ARG_WRONGSTATE,
                 "L1 boundary label is missing at least one required side group: x_max=%" PetscInt_FMT " x_min=%" PetscInt_FMT " z_min=%" PetscInt_FMT " z_max=%" PetscInt_FMT,
                 counts[0], counts[1], counts[2], counts[3]);
    }
  } else {
    PetscCheck(counts[2] > 0, comm, PETSC_ERR_ARG_WRONGSTATE, "Cube boundary label has no clamped z_min points");
    PetscCheck(app->pressure == 0.0 || counts[3] > 0, comm, PETSC_ERR_ARG_WRONGSTATE, "Cube boundary label has no loaded z_max points");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupDiscretization(DM dm, const AppCtx *app)
{
  const PetscInt components[3] = {0, 1, 2};
  PetscFE        fe;
  PetscDS        ds;
  DMLabel        label;
  PetscWeakForm  wf;
  PetscInt       bd;

  PetscFunctionBeginUser;
  PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, 3, 3, PETSC_TRUE, app->degree, PETSC_DETERMINE, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "displacement"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, ResidualBody, ResidualStress));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, JacobianElasticity));

  PetscCall(DMGetLabel(dm, BoundaryLabelName(app), &label));
  PetscCheck(label, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Missing boundary label %s", BoundaryLabelName(app));
  if (app->use_mesh) {
    const PetscInt base = MARKER_BASE, xRollers[2] = {MARKER_X_MIN, MARKER_X_MAX}, zRollers[2] = {MARKER_Z_MIN, MARKER_Z_MAX};
    const PetscInt xComp[1] = {0}, zComp[1] = {2};
    PetscBool      isRollers, isBaseOnly, isFullSides;

    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "glued_base", label, 1, &base, 0, 3, components, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
    PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "rollers", &isRollers));
    PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "base_only", &isBaseOnly));
    PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "full_sides", &isFullSides));
    if (isRollers) {
      PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "x_side_rollers", label, 2, xRollers, 0, 1, xComp, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
      PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "z_side_rollers", label, 2, zRollers, 0, 1, zComp, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
    } else if (isFullSides) {
      PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "x_side_clamps", label, 2, xRollers, 0, 3, components, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
      PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "z_side_clamps", label, 2, zRollers, 0, 3, components, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
    } else {
      PetscCheck(isBaseOnly, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown -mesh_bc_mode %s", app->mesh_bc_mode);
    }
  } else {
    const PetscInt bottom = MARKER_Z_MIN, top = MARKER_Z_MAX;

    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "clamped_bottom", label, 1, &bottom, 0, 3, components, (PetscVoidFn *)ZeroDisplacement, NULL, NULL, NULL));
    if (app->pressure != 0.0) {
      PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "top_pressure", label, 1, &top, 0, 3, components, NULL, NULL, NULL, &bd));
      PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
      PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, top, 0, 0, 0, TopTraction, 0, BoundaryNoFlux));
    }
  }
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreatePMGLevelDM(DM templateDM, const AppCtx *app, PetscInt degree, const char name[], DM *levelDM)
{
  AppCtx levelApp = *app;

  PetscFunctionBeginUser;
  levelApp.degree = degree;
  PetscCall(DMClone(templateDM, levelDM));
  PetscCall(PetscObjectSetName((PetscObject)*levelDM, name));
  PetscCall(SetupDiscretization(*levelDM, &levelApp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupPMGHierarchy(DM fineDM, const AppCtx *app)
{
  DM       p2 = NULL, p1 = NULL;
  PetscInt n1, n2;

  PetscFunctionBeginUser;
  PetscCheck(app->degree == 4, PetscObjectComm((PetscObject)fineDM), PETSC_ERR_SUP, "-pc_variant pmg currently implements the P4 -> P2 -> P1 hierarchy, so use -degree 4");
  PetscCall(CreatePMGLevelDM(fineDM, app, 2, "pmg_p2", &p2));
  PetscCall(CreatePMGLevelDM(fineDM, app, 1, "pmg_p1", &p1));
  PetscCall(DMSetCoarseDM(fineDM, p2));
  PetscCall(DMSetCoarseDM(p2, p1));
  {
    Vec v1 = NULL, v2 = NULL;
    PetscInt n1Local, n2Local;

    PetscCall(DMCreateGlobalVector(p1, &v1));
    PetscCall(DMCreateGlobalVector(p2, &v2));
    PetscCall(VecGetLocalSize(v1, &n1Local));
    PetscCall(VecGetLocalSize(v2, &n2Local));
    PetscCallMPI(MPI_Allreduce(&n1Local, &n1, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)fineDM)));
    PetscCallMPI(MPI_Allreduce(&n2Local, &n2, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)fineDM)));
    PetscCall(VecDestroy(&v2));
    PetscCall(VecDestroy(&v1));
  }
  PetscCheck(n1 > 0 && n2 > n1, PetscObjectComm((PetscObject)fineDM), PETSC_ERR_ARG_SIZ,
             "PMG needs a nonempty hierarchy; got P2=%" PetscInt_FMT " P1=%" PetscInt_FMT " free dofs. Use a larger mesh for P4 -> P2 -> P1.", n2, n1);
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)fineDM), "PMG_HIERARCHY degrees=4,2,1 coarse_global_dofs=%" PetscInt_FMT ",%" PetscInt_FMT " coarse_pc=gamg\n", n2, n1));
  PetscCall(DMDestroy(&p2));
  PetscCall(DMDestroy(&p1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetPMGTransfer(PC pc, PetscInt level, DM coarse, DM fine)
{
  Mat P = NULL, R = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMCreateInterpolation(coarse, fine, &P, NULL));
  PetscCall(MatTranspose(P, MAT_INITIAL_MATRIX, &R));
  PetscCall(PCMGSetInterpolation(pc, level, P));
  PetscCall(PCMGSetRestriction(pc, level, R));
  PetscCall(PCMGSetInjection(pc, level, R));
  PetscCall(MatDestroy(&R));
  PetscCall(MatDestroy(&P));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigurePMG(PC pc, DM fineDM)
{
  DM p2 = NULL, p1 = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoarseDM(fineDM, &p2));
  PetscCheck(p2, PetscObjectComm((PetscObject)fineDM), PETSC_ERR_ARG_WRONGSTATE, "PMG fine DM has no P2 coarse DM");
  PetscCall(DMGetCoarseDM(p2, &p1));
  PetscCheck(p1, PetscObjectComm((PetscObject)fineDM), PETSC_ERR_ARG_WRONGSTATE, "PMG P2 DM has no P1 coarse DM");
  PetscCall(SetPMGTransfer(pc, 1, p1, p2));
  PetscCall(SetPMGTransfer(pc, 2, p2, fineDM));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigurePMGCoarseSolve(KSP coarse, const AppCtx *app)
{
  MPI_Comm    comm;
  KSP         innerKSP;
  PC          cpc, innerPC;
  PetscMPIInt ranks;
  PetscInt    groupSize, nredundant;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)coarse, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &ranks));
  groupSize = app->pmg_coarse_redundant_group_size;

  PetscCall(KSPSetType(coarse, KSPCG));
  PetscCall(KSPSetTolerances(coarse, 1.0e-3, PETSC_DEFAULT, PETSC_DEFAULT, 50));
  PetscCall(KSPGetPC(coarse, &cpc));

  if (groupSize > 0 && ranks > groupSize) {
    nredundant = ((PetscInt)ranks + groupSize - 1) / groupSize;
    PetscCall(KSPSetType(coarse, KSPPREONLY));
    PetscCall(PCSetType(cpc, PCREDUNDANT));
    PetscCall(PCRedundantSetNumber(cpc, nredundant));
    PetscCall(PCRedundantGetKSP(cpc, &innerKSP));
    PetscCall(KSPSetType(innerKSP, KSPCG));
    PetscCall(KSPSetTolerances(innerKSP, 1.0e-3, PETSC_DEFAULT, PETSC_DEFAULT, 50));
    PetscCall(KSPGetPC(innerKSP, &innerPC));
    PetscCall(PCSetType(innerPC, PCGAMG));
    PetscCall(PCGAMGSetType(innerPC, PCGAMGAGG));
    PetscCall(PCGAMGSetAggressiveSquareGraph(innerPC, app->pmg_coarse_gamg_aggressive_square_graph));
    PetscCall(PetscPrintf(comm,
                          "PMG_COARSE_SOLVE type=redundant group_size=%" PetscInt_FMT " copies=%" PetscInt_FMT " inner_pc=gamg aggressive_square_graph=%s\n",
                          groupSize, nredundant, app->pmg_coarse_gamg_aggressive_square_graph ? "true" : "false"));
  } else {
    PetscCall(PCSetType(cpc, PCGAMG));
    PetscCall(PCGAMGSetType(cpc, PCGAMGAGG));
    PetscCall(PCGAMGSetAggressiveSquareGraph(cpc, app->pmg_coarse_gamg_aggressive_square_graph));
    PetscCall(PetscPrintf(comm, "PMG_COARSE_SOLVE type=gamg group_size=%" PetscInt_FMT " copies=1 aggressive_square_graph=%s\n", groupSize,
                          app->pmg_coarse_gamg_aggressive_square_graph ? "true" : "false"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigureSolver(SNES snes, DM dm, Mat A, const AppCtx *app)
{
  KSP       ksp;
  PC        pc;
  PetscBool is_gamg, is_pmg, is_bddc, is_fetidp, is_none;

  PetscFunctionBeginUser;
  PetscCall(SNESSetType(snes, SNESKSPONLY));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPSetTolerances(ksp, 1.0e-8, PETSC_DEFAULT, PETSC_DEFAULT, 200));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PetscStrcasecmp(app->variant, "gamg", &is_gamg));
  PetscCall(PetscStrcasecmp(app->variant, "pmg", &is_pmg));
  PetscCall(PetscStrcasecmp(app->variant, "bddc", &is_bddc));
  PetscCall(PetscStrcasecmp(app->variant, "fetidp", &is_fetidp));
  PetscCall(PetscStrcasecmp(app->variant, "none", &is_none));
  PetscCheck(is_gamg || is_pmg || is_bddc || is_fetidp || is_none, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "-pc_variant must be gamg, pmg, bddc, fetidp, or none");

  if (is_gamg) {
    PetscCall(PCSetType(pc, PCGAMG));
    PetscCall(PCGAMGSetType(pc, PCGAMGAGG));
  } else if (is_pmg) {
    KSP coarse;

    PetscCall(PCSetType(pc, PCMG));
    PetscCall(PCMGSetLevels(pc, 3, NULL));
    PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
    PetscCall(PCMGSetCycleType(pc, PC_MG_CYCLE_V));
    PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));
    PetscCall(ConfigurePMG(pc, dm));
    PetscCall(PCMGGetCoarseSolve(pc, &coarse));
    PetscCall(ConfigurePMGCoarseSolve(coarse, app));
    for (PetscInt level = 1; level < 3; ++level) {
      KSP smooth;
      PC  spc;

      PetscCall(PCMGGetSmoother(pc, level, &smooth));
      PetscCall(KSPSetType(smooth, KSPCHEBYSHEV));
      PetscCall(KSPSetNormType(smooth, KSP_NORM_NONE));
      PetscCall(KSPSetTolerances(smooth, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 2));
      PetscCall(KSPGetPC(smooth, &spc));
      PetscCall(PCSetType(spc, PCJACOBI));
    }
  } else if (is_bddc) {
    PetscCall(PCSetType(pc, PCBDDC));
  } else if (is_fetidp) {
    PetscCall(KSPSetType(ksp, KSPFETIDP));
  } else {
    PetscCall(PCSetType(pc, PCNONE));
  }

  if (is_gamg || is_pmg || is_bddc || is_fetidp) {
    DM           subdm;
    MatNullSpace nearNullSpace;
    PetscInt     field = 0;
    PetscObject  displacement;

    PetscCall(DMCreateSubDM(dm, 1, &field, NULL, &subdm));
    PetscCall(DMPlexCreateRigidBody(subdm, 0, &nearNullSpace));
    PetscCall(DMGetField(dm, 0, NULL, &displacement));
    PetscCall(PetscObjectCompose(displacement, "nearnullspace", (PetscObject)nearNullSpace));
    PetscCall(MatSetNearNullSpace(A, nearNullSpace));
    PetscCall(MatNullSpaceDestroy(&nearNullSpace));
    PetscCall(DMDestroy(&subdm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppendConstrainedPoint(PetscSection section, PetscInt point, PetscInt ncomponents, const PetscInt components[], PetscInt *nidx, PetscInt *cap, PetscInt **idx)
{
  PetscInt dof, off;

  PetscFunctionBeginUser;
  PetscCall(PetscSectionGetDof(section, point, &dof));
  if (dof <= 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(dof % 3 == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected vector dofs in blocks of 3 on point %" PetscInt_FMT, point);
  PetscCall(PetscSectionGetOffset(section, point, &off));
  for (PetscInt b = 0; b < dof / 3; ++b) {
    for (PetscInt c = 0; c < ncomponents; ++c) {
      if (*nidx == *cap) {
        *cap = *cap ? 2 * *cap : 1024;
        PetscCall(PetscRealloc((size_t)*cap * sizeof(PetscInt), idx));
      }
      (*idx)[(*nidx)++] = off + 3 * b + components[c];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppendConstrainedStratum(DMLabel label, PetscSection section, PetscInt value, PetscInt ncomponents, const PetscInt components[], PetscInt *nidx, PetscInt *cap, PetscInt **idx)
{
  IS              points = NULL;
  const PetscInt *pidx;
  PetscInt        npoints;

  PetscFunctionBeginUser;
  PetscCall(DMLabelGetStratumIS(label, value, &points));
  if (!points) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(ISGetLocalSize(points, &npoints));
  PetscCall(ISGetIndices(points, &pidx));
  for (PetscInt i = 0; i < npoints; ++i) PetscCall(AppendConstrainedPoint(section, pidx[i], ncomponents, components, nidx, cap, idx));
  PetscCall(ISRestoreIndices(points, &pidx));
  PetscCall(ISDestroy(&points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildLocalDirichletIS(DM dm, const AppCtx *app, IS *dirichlet)
{
  const PetscInt allComponents[3] = {0, 1, 2};
  const PetscInt xComponent[1]   = {0};
  const PetscInt zComponent[1]   = {2};
  PetscSection   section;
  DMLabel        label;
  PetscInt       nidx = 0, cap = 0, *idx = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetLabel(dm, BoundaryLabelName(app), &label));
  PetscCheck(label, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Missing boundary label %s", BoundaryLabelName(app));

  if (app->use_mesh) {
    PetscBool isRollers, isBaseOnly, isFullSides;

    PetscCall(AppendConstrainedStratum(label, section, MARKER_BASE, 3, allComponents, &nidx, &cap, &idx));
    PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "rollers", &isRollers));
    PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "base_only", &isBaseOnly));
    PetscCall(PetscStrcasecmp(app->mesh_bc_mode, "full_sides", &isFullSides));
    if (isRollers) {
      PetscCall(AppendConstrainedStratum(label, section, MARKER_X_MIN, 1, xComponent, &nidx, &cap, &idx));
      PetscCall(AppendConstrainedStratum(label, section, MARKER_X_MAX, 1, xComponent, &nidx, &cap, &idx));
      PetscCall(AppendConstrainedStratum(label, section, MARKER_Z_MIN, 1, zComponent, &nidx, &cap, &idx));
      PetscCall(AppendConstrainedStratum(label, section, MARKER_Z_MAX, 1, zComponent, &nidx, &cap, &idx));
    } else if (isFullSides) {
      PetscCall(AppendConstrainedStratum(label, section, MARKER_X_MIN, 3, allComponents, &nidx, &cap, &idx));
      PetscCall(AppendConstrainedStratum(label, section, MARKER_X_MAX, 3, allComponents, &nidx, &cap, &idx));
      PetscCall(AppendConstrainedStratum(label, section, MARKER_Z_MIN, 3, allComponents, &nidx, &cap, &idx));
      PetscCall(AppendConstrainedStratum(label, section, MARKER_Z_MAX, 3, allComponents, &nidx, &cap, &idx));
    } else {
      PetscCheck(isBaseOnly, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown -mesh_bc_mode %s", app->mesh_bc_mode);
    }
  } else {
    PetscCall(AppendConstrainedStratum(label, section, MARKER_Z_MIN, 3, allComponents, &nidx, &cap, &idx));
  }

  PetscCall(PetscSortRemoveDupsInt(&nidx, idx));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), nidx, idx, PETSC_OWN_POINTER, dirichlet));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildGlobalComponentMap(DM dm, PetscInt *ngids, PetscInt **gids, PetscInt **comps)
{
  PetscSection section, sectionGlobal;
  PetscInt     pStart, pEnd, n = 0, cap = 0, *gid = NULL, *comp = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(DMGetGlobalSection(dm, &sectionGlobal));
  PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
  for (PetscInt p = pStart; p < pEnd; ++p) {
    const PetscInt *cdofs = NULL;
    PetscInt        dof, cdof, off, cind = 0;

    PetscCall(PetscSectionGetDof(section, p, &dof));
    if (!dof) continue;
    PetscCheck(dof % 3 == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected vector dofs in blocks of 3 on point %" PetscInt_FMT, p);
    PetscCall(PetscSectionGetConstraintDof(section, p, &cdof));
    PetscCall(PetscSectionGetConstraintIndices(section, p, &cdofs));
    PetscCall(PetscSectionGetOffset(sectionGlobal, p, &off));
    for (PetscInt c = 0; c < dof; ++c) {
      if (cind < cdof && c == cdofs[cind]) {
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
        PetscCheck(comp[r] == comp[w - 1], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Global dof %" PetscInt_FMT " was assigned components %" PetscInt_FMT " and %" PetscInt_FMT, gid[r], comp[w - 1], comp[r]);
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
  PetscBool                ismatis = PETSC_FALSE;
  Mat                      localMat = NULL;
  ISLocalToGlobalMapping   rmap;
  const PetscInt          *ridx;
  IS                       fields[3] = {NULL, NULL, NULL};
  PetscInt                 nloc, nrows, ngids, *gids = NULL, *comps = NULL;
  PetscInt                 nfield[3] = {0, 0, 0};
  PetscInt                *fieldIdx[3] = {NULL, NULL, NULL};

  PetscFunctionBeginUser;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
  if (!ismatis) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatISGetLocalMat(A, &localMat));
  PetscCall(MatGetSize(localMat, &nrows, NULL));
  PetscCall(MatISRestoreLocalMat(A, &localMat));
  PetscCall(MatISGetLocalToGlobalMapping(A, &rmap, NULL));
  PetscCall(ISLocalToGlobalMappingGetSize(rmap, &nloc));
  PetscCheck(nloc == nrows, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MATIS local map size %" PetscInt_FMT " differs from local matrix rows %" PetscInt_FMT, nloc, nrows);

  PetscCall(BuildGlobalComponentMap(dm, &ngids, &gids, &comps));
  PetscCall(PetscMalloc3(nloc, &fieldIdx[0], nloc, &fieldIdx[1], nloc, &fieldIdx[2]));
  PetscCall(ISLocalToGlobalMappingGetIndices(rmap, &ridx));
  for (PetscInt i = 0; i < nloc; ++i) {
    PetscInt loc, comp;

    PetscCheck(ridx[i] >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MATIS cleaned local-to-global map contains negative index %" PetscInt_FMT, ridx[i]);
    PetscCall(PetscFindInt(ridx[i], ngids, gids, &loc));
    PetscCheck(loc >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not recover displacement component for MATIS local row %" PetscInt_FMT " global dof %" PetscInt_FMT, i, ridx[i]);
    comp = comps[loc];
    PetscCheck(comp >= 0 && comp < 3, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid component %" PetscInt_FMT " for MATIS local row %" PetscInt_FMT, comp, i);
    fieldIdx[comp][nfield[comp]++] = i;
  }
  PetscCall(ISLocalToGlobalMappingRestoreIndices(rmap, &ridx));

  for (PetscInt comp = 0; comp < 3; ++comp) PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nfield[comp], fieldIdx[comp], PETSC_COPY_VALUES, &fields[comp]));
  PetscCall(PCBDDCSetDofsSplittingLocal(pc, 3, fields));
  for (PetscInt comp = 0; comp < 3; ++comp) PetscCall(ISDestroy(&fields[comp]));
  PetscCall(PetscFree3(fieldIdx[0], fieldIdx[1], fieldIdx[2]));
  PetscCall(PetscFree(gids));
  PetscCall(PetscFree(comps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigureBDDCMetadata(PC pc, DM dm, const AppCtx *app, Mat A)
{
  IS dirichlet = NULL;

  PetscFunctionBeginUser;
  PetscCall(PCSetType(pc, PCBDDC));
  PetscCall(BuildLocalDirichletIS(dm, app, &dirichlet));
  PetscCall(PCBDDCSetDirichletBoundariesLocal(pc, dirichlet));
  PetscCall(ISDestroy(&dirichlet));
  PetscCall(SetBDDCDofSplittingLocal(pc, dm, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigureBDDCComponentSplitting(PC pc, DM dm, Mat A)
{
  PetscFunctionBeginUser;
  PetscCall(SetBDDCDofSplittingLocal(pc, dm, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigureFETIDPMetadata(KSP ksp, DM dm, const AppCtx *app, Mat A)
{
  PC inner = NULL;

  PetscFunctionBeginUser;
  PetscCall(KSPFETIDPGetInnerBDDC(ksp, &inner));
  PetscCall(ConfigureBDDCMetadata(inner, dm, app, A));
  PetscCall(KSPFETIDPSetInnerBDDC(ksp, inner));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigureFETIDPComponentSplitting(KSP ksp, DM dm, Mat A)
{
  PC inner = NULL;

  PetscFunctionBeginUser;
  PetscCall(KSPFETIDPGetInnerBDDC(ksp, &inner));
  PetscCall(ConfigureBDDCComponentSplitting(inner, dm, A));
  PetscCall(KSPFETIDPSetInnerBDDC(ksp, inner));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReportLayoutStats(DM dm, Mat A, const AppCtx *app, PetscReal *matisDuplication)
{
  MPI_Comm                 comm = PetscObjectComm((PetscObject)dm);
  PetscSection             section;
  ISLocalToGlobalMapping   ltog;
  PetscBool                ismatis = PETSC_FALSE;
  PetscMPIInt              ranks;
  PetscInt                 pStart, pEnd, localStorage, localUnconstrained, matRowsLocal, matRowsGlobal;
  PetscInt                 dofPts = 0, partialPts = 0, fullConstraintPts = 0, cdofTotal = 0, ccomp[3] = {0, 0, 0};
  PetscInt                 sumLocalStorage, sumLocalUnconstrained, sumDofPts, sumPartialPts, sumFullConstraintPts, sumCdofTotal, sumCcomp[3];
  PetscInt                 minMatRows, maxMatRows, dmBlockSize;

  PetscFunctionBeginUser;
  if (matisDuplication) *matisDuplication = 0.0;
  PetscCallMPI(MPI_Comm_size(comm, &ranks));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
  PetscCall(PetscSectionGetStorageSize(section, &localStorage));
  PetscCall(PetscSectionGetConstrainedStorageSize(section, &localUnconstrained));
  PetscCall(MatGetLocalSize(A, &matRowsLocal, NULL));
  PetscCall(MatGetSize(A, &matRowsGlobal, NULL));
  PetscCall(DMGetLocalToGlobalMapping(dm, &ltog));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(ltog, &dmBlockSize));

  for (PetscInt p = pStart; p < pEnd; ++p) {
    const PetscInt *cdofs = NULL;
    PetscInt        dof, cdof;

    PetscCall(PetscSectionGetDof(section, p, &dof));
    if (!dof) continue;
    PetscCall(PetscSectionGetConstraintDof(section, p, &cdof));
    PetscCall(PetscSectionGetConstraintIndices(section, p, &cdofs));
    ++dofPts;
    cdofTotal += cdof;
    if (cdof && cdof < dof) ++partialPts;
    if (cdof == dof) ++fullConstraintPts;
    for (PetscInt i = 0; i < cdof; ++i) ccomp[cdofs[i] % 3]++;
  }

  PetscCallMPI(MPI_Allreduce(&localStorage, &sumLocalStorage, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&localUnconstrained, &sumLocalUnconstrained, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&dofPts, &sumDofPts, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&partialPts, &sumPartialPts, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&fullConstraintPts, &sumFullConstraintPts, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&cdofTotal, &sumCdofTotal, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(ccomp, sumCcomp, 3, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&matRowsLocal, &minMatRows, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPI_Allreduce(&matRowsLocal, &maxMatRows, 1, MPIU_INT, MPI_MAX, comm));
  PetscCall(PetscPrintf(comm,
                        "LAYOUT ranks=%d mesh=%s bc=%s degree=%" PetscInt_FMT " mat_global_rows=%" PetscInt_FMT " mat_local_rows_minmax=%" PetscInt_FMT ",%" PetscInt_FMT " dm_ltog_bs=%" PetscInt_FMT " local_storage_sum=%" PetscInt_FMT " local_unconstrained_sum=%" PetscInt_FMT " dof_points_sum=%" PetscInt_FMT " partial_constraint_points_sum=%" PetscInt_FMT " full_constraint_points_sum=%" PetscInt_FMT " constrained_dofs_sum=%" PetscInt_FMT " constrained_components_sum=%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT "\n",
                        ranks, app->use_mesh ? app->mesh : "generated_cube", app->use_mesh ? app->mesh_bc_mode : "cube_clamped_bottom", app->degree, matRowsGlobal, minMatRows, maxMatRows, dmBlockSize, sumLocalStorage,
                        sumLocalUnconstrained, sumDofPts, sumPartialPts, sumFullConstraintPts, sumCdofTotal, sumCcomp[0], sumCcomp[1], sumCcomp[2]));

  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &ismatis));
  if (ismatis) {
    ISLocalToGlobalMapping rmap;
    const PetscInt        *ridx;
    PetscInt               nloc, mapBlockSize, ngids, *gids = NULL, *comps = NULL;
    PetscInt               localCompRows[3] = {0, 0, 0}, sumCompRows[3], minMapRows, maxMapRows;
    PetscReal              duplication;

    PetscCall(MatISGetLocalToGlobalMapping(A, &rmap, NULL));
    PetscCall(ISLocalToGlobalMappingGetSize(rmap, &nloc));
    PetscCall(ISLocalToGlobalMappingGetBlockSize(rmap, &mapBlockSize));
    PetscCall(BuildGlobalComponentMap(dm, &ngids, &gids, &comps));
    PetscCall(ISLocalToGlobalMappingGetIndices(rmap, &ridx));
    for (PetscInt i = 0; i < nloc; ++i) {
      PetscInt loc;

      PetscCall(PetscFindInt(ridx[i], ngids, gids, &loc));
      PetscCheck(loc >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not recover component for MATIS local row %" PetscInt_FMT " global dof %" PetscInt_FMT, i, ridx[i]);
      localCompRows[comps[loc]]++;
    }
    PetscCall(ISLocalToGlobalMappingRestoreIndices(rmap, &ridx));
    PetscCallMPI(MPI_Allreduce(localCompRows, sumCompRows, 3, MPIU_INT, MPI_SUM, comm));
    PetscCallMPI(MPI_Allreduce(&nloc, &minMapRows, 1, MPIU_INT, MPI_MIN, comm));
    PetscCallMPI(MPI_Allreduce(&nloc, &maxMapRows, 1, MPIU_INT, MPI_MAX, comm));
    duplication = matRowsGlobal ? ((PetscReal)(sumCompRows[0] + sumCompRows[1] + sumCompRows[2])) / ((PetscReal)matRowsGlobal) : 0.0;
    if (matisDuplication) *matisDuplication = duplication;
    PetscCall(PetscPrintf(comm,
                          "MATIS_LAYOUT ranks=%d map_rows_minmax=%" PetscInt_FMT ",%" PetscInt_FMT " map_ltog_bs=%" PetscInt_FMT " component_rows_sum=%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT "\n",
                          ranks, minMapRows, maxMapRows, mapBlockSize, sumCompRows[0], sumCompRows[1], sumCompRows[2]));
    PetscCall(PetscPrintf(comm, "MATIS_DUPLICATION value=%.6g limit=%.6g\n", (double)duplication, (double)app->matis_duplication_limit));
    PetscCall(PetscFree(gids));
    PetscCall(PetscFree(comps));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateElasticityMatrix(DM dm, Mat *A)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreateMatrix(dm, A));
  PetscCall(MatSetOption(*A, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(*A, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  PetscCall(MatSetOption(*A, MAT_SPD, PETSC_TRUE));
  PetscCall(MatSetOption(*A, MAT_SPD_ETERNAL, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateConfiguredSNES(DM dm, Mat A, const AppCtx *app, PetscBool is_bddc, PetscBool is_fetidp, SNES *snes)
{
  KSP ksp;
  PC  pc;

  PetscFunctionBeginUser;
  PetscCall(SNESCreate(PETSC_COMM_WORLD, snes));
  PetscCall(SNESSetDM(*snes, dm));
  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, NULL));
  PetscCall(SNESSetJacobian(*snes, A, A, NULL, NULL));
  PetscCall(ConfigureSolver(*snes, dm, A, app));
  PetscCall(SNESSetFromOptions(*snes));
  PetscCall(SNESGetKSP(*snes, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  if (is_bddc) PetscCall(ConfigureBDDCComponentSplitting(pc, dm, A));
  if (is_fetidp) PetscCall(ConfigureFETIDPComponentSplitting(ksp, dm, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4ElasticityRun(int argc, char **argv, const P4ElasticityCase *spec, const char help[])
{
  AppCtx         app;
  DM             dm;
  SNES           snes;
  KSP            ksp;
  Mat            A;
  Vec            u, rhs;
  PetscInt       cStart, cEnd, localCells, globalCells, localSize, globalSize, minLocalSize, maxLocalSize, its;
  PetscMPIInt    ranks;
  PetscReal      solveTime, t0, t1, unorm;
  KSPConvergedReason reason;
  PetscBool      is_pmg, is_bddc, is_fetidp;
  PetscLogStage  sweepFirstStage = -1, sweepRepeatedStage = -1;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, spec, &app));
  PetscCall(SetDDPartitionerDefault(PETSC_COMM_WORLD, &app));
  PetscCheck(!app.configure_bddc_metadata, PETSC_COMM_WORLD, PETSC_ERR_SUP,
             "-configure_bddc_metadata is disabled for this constrained PetscDS/DMPlex FEM driver: current local Dirichlet candidates are full local-section offsets, not verified MATIS local matrix rows");
  PetscCall(IsPMGVariant(&app, &is_pmg));
  PetscCall(PetscStrcasecmp(app.variant, "bddc", &is_bddc));
  PetscCall(PetscStrcasecmp(app.variant, "fetidp", &is_fetidp));

  PetscCall(CreateMesh(PETSC_COMM_WORLD, &app, &dm));
  if (app.use_mesh) PetscCall(BuildBoundaryMarkerFromFaceSets(dm));
  else PetscCall(MarkBoundaryFaces(dm));
  PetscCall(ReportBoundaryCounts(dm, &app));
  if (is_pmg) PetscCall(SetupPMGHierarchy(dm, &app));
  PetscCall(SetupDiscretization(dm, &app));
  if (is_bddc || is_fetidp) {
    PetscCall(DMSetMatType(dm, MATIS));
    PetscCall(PetscOptionsSetValue(NULL, "-mat_is_localmat_type", "aij"));
  }
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(VecDuplicate(u, &rhs));
  PetscCall(CreateElasticityMatrix(dm, &A));
  if (app.inspect_layout) {
    PetscCall(ReportLayoutStats(dm, A, &app, NULL));
    PetscCall(VecDestroy(&rhs));
    PetscCall(VecDestroy(&u));
    PetscCall(MatDestroy(&A));
    PetscCall(DMDestroy(&dm));
    PetscCall(PetscFinalize());
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (app.use_mesh && (is_bddc || is_fetidp)) {
    PetscReal duplication;

    PetscCall(ReportLayoutStats(dm, A, &app, &duplication));
    PetscCheck(duplication < app.matis_duplication_limit, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "Bad MATIS duplication %.6g >= %.6g: do not run BDDC/FETI-DP on this L1 partition unless this diagnostic run intentionally raises -matis_duplication_limit",
               (double)duplication, (double)app.matis_duplication_limit);
  }

  PetscCall(CreateConfiguredSNES(dm, A, &app, is_bddc, is_fetidp, &snes));
  PetscCall(SNESGetKSP(snes, &ksp));

  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  localCells = cEnd - cStart;
  PetscCallMPI(MPI_Allreduce(&localCells, &globalCells, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCall(VecGetLocalSize(u, &localSize));
  PetscCall(VecGetSize(u, &globalSize));
  PetscCallMPI(MPI_Allreduce(&localSize, &minLocalSize, 1, MPIU_INT, MPI_MIN, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&localSize, &maxLocalSize, 1, MPIU_INT, MPI_MAX, PETSC_COMM_WORLD));
  if (app.use_mesh) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "mesh_tet_p%d mesh=%s ranks=%d global_cells=%" PetscInt_FMT " local_dofs_minmax=%" PetscInt_FMT ",%" PetscInt_FMT " global_dofs=%" PetscInt_FMT " variant=%s bc=%s gravity=%.6g E=%.6g nu=%.6g\n",
                          (int)app.degree, app.mesh, ranks, globalCells, minLocalSize, maxLocalSize, globalSize, app.variant, app.mesh_bc_mode, (double)app.gravity, (double)app.young, (double)app.poisson));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "cube_tet_p%d faces=%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT " ranks=%d global_cells=%" PetscInt_FMT " local_dofs_minmax=%" PetscInt_FMT ",%" PetscInt_FMT " global_dofs=%" PetscInt_FMT " variant=%s pressure=%.6g E=%.6g nu=%.6g\n",
                          (int)app.degree, app.faces[0], app.faces[1], app.faces[2], ranks, globalCells, minLocalSize, maxLocalSize, globalSize, app.variant, (double)app.pressure, (double)app.young, (double)app.poisson));
  }

  if (app.material_sweep_count > 1) {
    PetscReal firstSolveTime = 0.0, repeatedSolveTime = 0.0, minSolveTime = PETSC_MAX_REAL, maxSolveTime = 0.0;
    PetscInt  firstIts = 0, repeatedIts = 0, totalIts = 0, maxIts = 0;
    PetscInt  converged = 0;
    PetscBool sweepFresh, sweepReuse;

    PetscCall(IsSweepMode(&app, "fresh", &sweepFresh));
    PetscCall(IsSweepMode(&app, "reuse_pc", &sweepReuse));
    PetscCall(PetscLogStageRegister("material_sweep_first_setup", &sweepFirstStage));
    PetscCall(PetscLogStageRegister("material_sweep_repeated_solves", &sweepRepeatedStage));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "SWEEP_CONFIG count=%" PetscInt_FMT " seed=%" PetscInt_FMT " E_range=%.6g,%.6g nu_range=%.6g,%.6g mode=%s reuse_pc=%s\n",
                          app.material_sweep_count, app.material_sweep_seed, (double)app.material_sweep_young_min, (double)app.material_sweep_young_max,
                          (double)app.material_sweep_poisson_min, (double)app.material_sweep_poisson_max, app.material_sweep_mode, sweepReuse ? "true" : "false"));
    if (sweepFresh) {
      PetscCall(SNESDestroy(&snes));
      PetscCall(MatDestroy(&A));
    } else if (sweepReuse) {
      PetscCall(SNESSetLagPreconditioner(snes, -2));
      PetscCall(SNESSetLagPreconditionerPersists(snes, PETSC_TRUE));
    } else {
      PetscCall(SNESSetLagPreconditioner(snes, 1));
      PetscCall(SNESSetLagPreconditionerPersists(snes, PETSC_FALSE));
      PetscCall(KSPSetReusePreconditioner(ksp, PETSC_FALSE));
    }
    for (PetscInt sample = 0; sample < app.material_sweep_count; ++sample) {
      PetscReal sampleYoung, samplePoisson;

      PetscCall(MaterialSweepSample(&app, sample, &sampleYoung, &samplePoisson));
      PetscCall(SetElasticityMaterial(sampleYoung, samplePoisson, app.pressure, app.gravity));
      PetscCall(VecZeroEntries(u));
      PetscCall(VecZeroEntries(rhs));
      PetscCall(PetscLogStagePush(sample == 0 ? sweepFirstStage : sweepRepeatedStage));
      PetscCall(PetscTime(&t0));
      if (sweepFresh) {
        PetscCall(CreateElasticityMatrix(dm, &A));
        PetscCall(CreateConfiguredSNES(dm, A, &app, is_bddc, is_fetidp, &snes));
      }
      PetscCall(SNESSolve(snes, rhs, u));
      PetscCall(PetscTime(&t1));
      PetscCall(PetscLogStagePop());
      solveTime = t1 - t0;
      PetscCall(VecNorm(u, NORM_INFINITY, &unorm));
      PetscCall(SNESGetKSP(snes, &ksp));
      PetscCall(KSPGetIterationNumber(ksp, &its));
      PetscCall(KSPGetConvergedReason(ksp, &reason));
      if (sweepFresh) {
        PetscCall(SNESDestroy(&snes));
        PetscCall(MatDestroy(&A));
      }
      if (reason > 0) ++converged;
      totalIts += its;
      maxIts = PetscMax(maxIts, its);
      minSolveTime = PetscMin(minSolveTime, solveTime);
      maxSolveTime = PetscMax(maxSolveTime, solveTime);
      if (sample == 0) {
        firstSolveTime = solveTime;
        firstIts       = its;
      } else {
        repeatedSolveTime += solveTime;
        repeatedIts += its;
      }
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                            "SWEEP_RESULT sample=%" PetscInt_FMT " variant=%s ranks=%d degree=%" PetscInt_FMT " E=%.12g nu=%.12g lambda=%.12g mu=%.12g mode=%s reuse_pc_active=%s ksp_its=%" PetscInt_FMT " ksp_reason=%d solve_time=%.6g max_abs_u=%.6e\n",
                            sample, app.variant, ranks, app.degree, (double)sampleYoung, (double)samplePoisson, (double)g_lambda, (double)g_mu,
                            app.material_sweep_mode, (sample > 0 && sweepReuse) ? "true" : "false", its, (int)reason, (double)solveTime, (double)unorm));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "RESULT variant=%s ranks=%d degree=%" PetscInt_FMT " mesh=%s bc=%s faces=%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT " global_dofs=%" PetscInt_FMT " sweep_count=%" PetscInt_FMT " sweep_mode=%s sweep_reuse_pc=%s converged=%" PetscInt_FMT " total_ksp_its=%" PetscInt_FMT " first_ksp_its=%" PetscInt_FMT " repeated_ksp_its=%" PetscInt_FMT " max_ksp_its=%" PetscInt_FMT " first_solve_time=%.6g repeated_avg_solve_time=%.6g min_solve_time=%.6g max_solve_time=%.6g total_solve_time=%.6g\n",
                          app.variant, ranks, app.degree, app.use_mesh ? app.mesh : "generated_cube", app.use_mesh ? app.mesh_bc_mode : "cube_clamped_bottom", app.faces[0], app.faces[1], app.faces[2], globalSize,
                          app.material_sweep_count, app.material_sweep_mode, sweepReuse ? "true" : "false", converged, totalIts, firstIts, repeatedIts, maxIts, (double)firstSolveTime,
                          (double)(repeatedSolveTime / (PetscReal)(app.material_sweep_count - 1)), (double)minSolveTime, (double)maxSolveTime, (double)(firstSolveTime + repeatedSolveTime)));
  } else {
    PetscCall(VecZeroEntries(u));
    PetscCall(VecZeroEntries(rhs));
    PetscCall(PetscTime(&t0));
    PetscCall(SNESSolve(snes, rhs, u));
    PetscCall(PetscTime(&t1));
    solveTime = t1 - t0;
    PetscCall(VecNorm(u, NORM_INFINITY, &unorm));
    PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(KSPGetIterationNumber(ksp, &its));
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "RESULT variant=%s ranks=%d degree=%" PetscInt_FMT " mesh=%s bc=%s faces=%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT " global_dofs=%" PetscInt_FMT " ksp_its=%" PetscInt_FMT " ksp_reason=%d solve_time=%.6g max_abs_u=%.6e\n",
                          app.variant, ranks, app.degree, app.use_mesh ? app.mesh : "generated_cube", app.use_mesh ? app.mesh_bc_mode : "cube_clamped_bottom", app.faces[0], app.faces[1], app.faces[2], globalSize, its, (int)reason, (double)solveTime, (double)unorm));
  }

  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&A));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  PetscFunctionReturn(PETSC_SUCCESS);
}
