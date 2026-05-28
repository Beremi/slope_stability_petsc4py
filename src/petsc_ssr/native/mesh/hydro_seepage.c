#include "hydro_seepage.h"
#include "p4_basis.h"

#include <petscdmplex.h>
#include <petscksp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  char      mesh[PETSC_MAX_PATH_LEN];
  char      elem_type[8];
  char      pc_variant[32];
  char      pmg_coarse_pc_type[32];
  char      head_mode[32];
  char      summary_json[PETSC_MAX_PATH_LEN];
  char      pressure_binary[PETSC_MAX_PATH_LEN];
  char      dof_coords_csv[PETSC_MAX_PATH_LEN];
  PetscInt  dim;
  PetscInt  degree;
  PetscInt  newton_max_it;
  PetscInt  ksp_max_it;
  PetscInt  pmg_smoother_max_it;
  PetscInt  pmg_coarse_max_it;
  PetscReal newton_tol;
  PetscReal ksp_rtol;
  PetscReal grho;
  PetscReal head_x0, head_y0, head_x1, head_y1;
  PetscBool log_view;
} HydroOptions;

typedef struct {
  PetscInt   n;
  PetscReal *xyz;
} HydroCoordSet;

typedef struct {
  HydroCoordSet dry;
  HydroCoordSet porous;
  HydroCoordSet free_head;
  HydroCoordSet support;
} HydroHeadSets;

typedef struct {
  DM             dm;
  P4Basis       *basis;
  IS             constrained_is;
  Vec            prescribed;
  PetscBool     *target_dry;
  PetscBool     *target_porous;
  PetscBool     *target_free;
  PetscBool     *target_support;
  PetscLogDouble assembly_time;
  PetscInt       n_constrained_local;
  PetscInt       n_constrained_global;
  HydroOptions   opt;
} HydroCtx;

static PetscErrorCode HydroOptionsSetDefaults(HydroOptions *opt)
{
  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(opt, sizeof(*opt)));
  PetscCall(PetscStrncpy(opt->mesh, "meshes/3d_hetero_seepage_transition/transition_default.msh", sizeof(opt->mesh)));
  PetscCall(PetscStrncpy(opt->elem_type, "P2", sizeof(opt->elem_type)));
  PetscCall(PetscStrncpy(opt->pc_variant, "pmg", sizeof(opt->pc_variant)));
  PetscCall(PetscStrncpy(opt->pmg_coarse_pc_type, "gamg", sizeof(opt->pmg_coarse_pc_type)));
  PetscCall(PetscStrncpy(opt->head_mode, "comsol3d", sizeof(opt->head_mode)));
  PetscCall(PetscStrncpy(opt->summary_json, ".local/tmp/comsol_seepage_petsc/hydro_summary.json", sizeof(opt->summary_json)));
  opt->pressure_binary[0] = '\0';
  opt->dof_coords_csv[0]  = '\0';
  opt->dim                 = 3;
  opt->degree              = 2;
  opt->newton_max_it       = 50;
  opt->ksp_max_it          = 500;
  opt->pmg_smoother_max_it = 10;
  opt->pmg_coarse_max_it   = 5;
  opt->newton_tol          = 1.0e-10;
  opt->ksp_rtol            = 1.0e-10;
  opt->grho                = 9.81;
  opt->head_x0             = 0.0;
  opt->head_y0             = 0.0;
  opt->head_x1             = 1.0;
  opt->head_y1             = 0.0;
  opt->log_view            = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroParseOptions(MPI_Comm comm, HydroOptions *opt)
{
  PetscBool flg;

  PetscFunctionBeginUser;
  PetscCall(HydroOptionsSetDefaults(opt));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-hydro_mesh", opt->mesh, sizeof(opt->mesh), NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-hydro_elem_type", opt->elem_type, sizeof(opt->elem_type), NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-hydro_pc_variant", opt->pc_variant, sizeof(opt->pc_variant), NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-hydro_pmg_coarse_pc_type", opt->pmg_coarse_pc_type, sizeof(opt->pmg_coarse_pc_type), NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-hydro_head_mode", opt->head_mode, sizeof(opt->head_mode), NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-hydro_summary_json", opt->summary_json, sizeof(opt->summary_json), NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-hydro_pressure_binary", opt->pressure_binary, sizeof(opt->pressure_binary), NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-hydro_dof_coords_csv", opt->dof_coords_csv, sizeof(opt->dof_coords_csv), NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-hydro_dim", &opt->dim, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-hydro_newton_max_it", &opt->newton_max_it, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-hydro_ksp_max_it", &opt->ksp_max_it, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-hydro_pmg_smoother_max_it", &opt->pmg_smoother_max_it, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-hydro_pmg_coarse_max_it", &opt->pmg_coarse_max_it, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-hydro_newton_tol", &opt->newton_tol, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-hydro_ksp_rtol", &opt->ksp_rtol, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-hydro_grho", &opt->grho, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-hydro_head_x0", &opt->head_x0, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-hydro_head_y0", &opt->head_y0, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-hydro_head_x1", &opt->head_x1, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-hydro_head_y1", &opt->head_y1, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-hydro_log_view", &opt->log_view, NULL));
  PetscCheck(opt->dim == 2 || opt->dim == 3, comm, PETSC_ERR_ARG_WRONG, "-hydro_dim must be 2 or 3");
  PetscCall(PetscStrcasecmp(opt->elem_type, "P1", &flg));
  if (flg) opt->degree = 1;
  else {
    PetscCall(PetscStrcasecmp(opt->elem_type, "P2", &flg));
    if (flg) opt->degree = 2;
    else {
      PetscCall(PetscStrcasecmp(opt->elem_type, "P4", &flg));
      PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "-hydro_elem_type must be P1, P2, or P4");
      opt->degree = 4;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void HydroLogicalName(char name[])
{
  const char *prefixes[] = {"nodeset:", "nodeset_", "boundary:", "boundary_", "region:", "region_"};
  for (size_t i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); ++i) {
    const size_t n = strlen(prefixes[i]);
    if (strncmp(name, prefixes[i], n) == 0) {
      memmove(name, name + n, strlen(name + n) + 1);
      return;
    }
  }
}

static PetscErrorCode HydroCoordSetAppend(HydroCoordSet *set, const PetscReal x[3])
{
  PetscFunctionBeginUser;
  PetscCall(PetscRealloc(sizeof(PetscReal) * 3 * (set->n + 1), &set->xyz));
  set->xyz[3 * set->n + 0] = x[0];
  set->xyz[3 * set->n + 1] = x[1];
  set->xyz[3 * set->n + 2] = x[2];
  set->n++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroNodeCoordAppend(PetscInt **tags, PetscReal **xyz, PetscInt *n, PetscInt tag, const PetscReal x[3])
{
  PetscFunctionBeginUser;
  PetscCall(PetscRealloc(sizeof(PetscInt) * (*n + 1), tags));
  PetscCall(PetscRealloc(sizeof(PetscReal) * 3 * (*n + 1), xyz));
  (*tags)[*n]           = tag;
  (*xyz)[3 * (*n) + 0] = x[0];
  (*xyz)[3 * (*n) + 1] = x[1];
  (*xyz)[3 * (*n) + 2] = x[2];
  (*n)++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscBool HydroFindNodeCoord(const PetscInt tags[], const PetscReal xyz[], PetscInt n, PetscInt tag, PetscReal x[3])
{
  for (PetscInt i = 0; i < n; ++i) {
    if (tags[i] == tag) {
      x[0] = xyz[3 * i + 0];
      x[1] = xyz[3 * i + 1];
      x[2] = xyz[3 * i + 2];
      return PETSC_TRUE;
    }
  }
  return PETSC_FALSE;
}

static PetscInt HydroParseEntityFirstPhysical(const char line[], PetscInt ncoords)
{
  const char *p = line;
  char       *end = NULL;
  long        nphys;

  (void)strtol(p, &end, 10);
  if (end == p) return -1;
  p = end;
  for (PetscInt i = 0; i < ncoords; ++i) {
    (void)strtod(p, &end);
    if (end == p) return -1;
    p = end;
  }
  nphys = strtol(p, &end, 10);
  if (end == p || nphys <= 0) return -1;
  p = end;
  return (PetscInt)strtol(p, NULL, 10);
}

static PetscInt HydroParseElementNodes(const char line[], PetscInt nodes[], PetscInt maxnodes)
{
  const char *p = line;
  char       *end = NULL;
  PetscInt    n = 0;

  while (*p && n < maxnodes) {
    long value = strtol(p, &end, 10);
    if (end == p) break;
    nodes[n++] = (PetscInt)value;
    p = end;
  }
  return n;
}

static PetscErrorCode HydroAppendNamedHeadSet(HydroHeadSets *sets, const char name[], const PetscReal xyz[3])
{
  PetscFunctionBeginUser;
  if (strcmp(name, "head_dry") == 0) PetscCall(HydroCoordSetAppend(&sets->dry, xyz));
  else if (strcmp(name, "head_porous") == 0) PetscCall(HydroCoordSetAppend(&sets->porous, xyz));
  else if (strcmp(name, "head_free") == 0) PetscCall(HydroCoordSetAppend(&sets->free_head, xyz));
  else if (strcmp(name, "head_support") == 0) PetscCall(HydroCoordSetAppend(&sets->support, xyz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroHeadSetsDestroy(HydroHeadSets *sets)
{
  PetscFunctionBeginUser;
  PetscCall(PetscFree(sets->dry.xyz));
  PetscCall(PetscFree(sets->porous.xyz));
  PetscCall(PetscFree(sets->free_head.xyz));
  PetscCall(PetscFree(sets->support.xyz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroParseGmshHeadSets(MPI_Comm comm, const char mesh[], HydroHeadSets *sets)
{
  FILE     *fh = NULL;
  char      line[8192], physical[4][1024][128];
  PetscInt  entity_phys[4][4096], max_entity = 4095;
  PetscInt *node_tags = NULL, n_node_tags = 0;
  PetscReal *node_xyz = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(sets, sizeof(*sets)));
  PetscCall(PetscMemzero(physical, sizeof(physical)));
  for (PetscInt d = 0; d < 4; ++d) {
    for (PetscInt i = 0; i <= max_entity; ++i) entity_phys[d][i] = -1;
  }
  fh = fopen(mesh, "r");
  PetscCheck(fh, comm, PETSC_ERR_FILE_OPEN, "Cannot open hydro mesh %s", mesh);
  while (fgets(line, sizeof(line), fh)) {
    if (strncmp(line, "$PhysicalNames", 14) == 0) {
      int n = 0;
      PetscCheck(fgets(line, sizeof(line), fh), comm, PETSC_ERR_FILE_READ, "Unexpected EOF in $PhysicalNames");
      sscanf(line, "%d", &n);
      for (int i = 0; i < n; ++i) {
        int  dim = 0, tag = 0;
        char name[128] = "";
        PetscCheck(fgets(line, sizeof(line), fh), comm, PETSC_ERR_FILE_READ, "Unexpected EOF in physical names");
        if (sscanf(line, "%d %d \"%127[^\"]\"", &dim, &tag, name) == 3 && dim >= 0 && dim <= 3 && tag >= 0 && tag < 1024) {
          HydroLogicalName(name);
          PetscCall(PetscStrncpy(physical[dim][tag], name, sizeof(physical[dim][tag])));
        }
      }
    } else if (strncmp(line, "$Entities", 9) == 0) {
      int np = 0, nc = 0, ns = 0, nv = 0;
      PetscCheck(fgets(line, sizeof(line), fh), comm, PETSC_ERR_FILE_READ, "Unexpected EOF in $Entities");
      sscanf(line, "%d %d %d %d", &np, &nc, &ns, &nv);
      for (int i = 0; i < np; ++i) {
        int    tag = 0;
        double x, y, z;
        PetscCheck(fgets(line, sizeof(line), fh), comm, PETSC_ERR_FILE_READ, "Unexpected EOF in point entities");
        sscanf(line, "%d %lf %lf %lf", &tag, &x, &y, &z);
        if (tag >= 0 && tag <= max_entity) entity_phys[0][tag] = HydroParseEntityFirstPhysical(line, 3);
      }
      for (int i = 0; i < nc; ++i) {
        int tag = 0;
        PetscCheck(fgets(line, sizeof(line), fh), comm, PETSC_ERR_FILE_READ, "Unexpected EOF in curve entities");
        sscanf(line, "%d", &tag);
        if (tag >= 0 && tag <= max_entity) entity_phys[1][tag] = HydroParseEntityFirstPhysical(line, 6);
      }
      for (int i = 0; i < ns; ++i) {
        int tag = 0;
        PetscCheck(fgets(line, sizeof(line), fh), comm, PETSC_ERR_FILE_READ, "Unexpected EOF in surface entities");
        sscanf(line, "%d", &tag);
        if (tag >= 0 && tag <= max_entity) entity_phys[2][tag] = HydroParseEntityFirstPhysical(line, 6);
      }
      for (int i = 0; i < nv; ++i) {
        int tag = 0;
        PetscCheck(fgets(line, sizeof(line), fh), comm, PETSC_ERR_FILE_READ, "Unexpected EOF in volume entities");
        sscanf(line, "%d", &tag);
        if (tag >= 0 && tag <= max_entity) entity_phys[3][tag] = HydroParseEntityFirstPhysical(line, 6);
      }
    } else if (strncmp(line, "$Nodes", 6) == 0) {
      int nblocks = 0;
      PetscCheck(fgets(line, sizeof(line), fh), comm, PETSC_ERR_FILE_READ, "Unexpected EOF in $Nodes");
      sscanf(line, "%d", &nblocks);
      for (int b = 0; b < nblocks; ++b) {
        int dim = 0, entity = 0, parametric = 0, nnode = 0, phys = -1;
        char name[128] = "";
        PetscCheck(fgets(line, sizeof(line), fh), comm, PETSC_ERR_FILE_READ, "Unexpected EOF in node block header");
        sscanf(line, "%d %d %d %d", &dim, &entity, &parametric, &nnode);
        if (dim >= 0 && dim <= 3 && entity >= 0 && entity <= max_entity) phys = entity_phys[dim][entity];
        if (phys >= 0 && phys < 1024) PetscCall(PetscStrncpy(name, physical[dim][phys], sizeof(name)));
        PetscInt *tags = NULL;
        if (nnode > 0) PetscCall(PetscMalloc1(nnode, &tags));
        for (int i = 0; i < nnode; ++i) {
          int tag_tmp = 0;
          PetscCheck(fgets(line, sizeof(line), fh), comm, PETSC_ERR_FILE_READ, "Unexpected EOF in node tags");
          sscanf(line, "%d", &tag_tmp);
          tags[i] = (PetscInt)tag_tmp;
        }
        for (int i = 0; i < nnode; ++i) {
          PetscReal xyz[3];
          PetscCheck(fgets(line, sizeof(line), fh), comm, PETSC_ERR_FILE_READ, "Unexpected EOF in node coordinates");
          sscanf(line, "%lf %lf %lf", (double *)&xyz[0], (double *)&xyz[1], (double *)&xyz[2]);
          PetscCall(HydroNodeCoordAppend(&node_tags, &node_xyz, &n_node_tags, tags[i], xyz));
          PetscCall(HydroAppendNamedHeadSet(sets, name, xyz));
        }
        PetscCall(PetscFree(tags));
      }
    } else if (strncmp(line, "$Elements", 9) == 0) {
      int nblocks = 0;
      PetscCheck(fgets(line, sizeof(line), fh), comm, PETSC_ERR_FILE_READ, "Unexpected EOF in $Elements");
      sscanf(line, "%d", &nblocks);
      for (int b = 0; b < nblocks; ++b) {
        int dim = 0, entity = 0, type = 0, nelem = 0, phys = -1;
        char name[128] = "";
        PetscCheck(fgets(line, sizeof(line), fh), comm, PETSC_ERR_FILE_READ, "Unexpected EOF in element block header");
        sscanf(line, "%d %d %d %d", &dim, &entity, &type, &nelem);
        if (dim >= 0 && dim <= 3 && entity >= 0 && entity <= max_entity) phys = entity_phys[dim][entity];
        if (phys >= 0 && phys < 1024) PetscCall(PetscStrncpy(name, physical[dim][phys], sizeof(name)));
        for (int i = 0; i < nelem; ++i) {
          PetscCheck(fgets(line, sizeof(line), fh), comm, PETSC_ERR_FILE_READ, "Unexpected EOF in element rows");
          if (name[0]) {
            PetscInt nodes[64], nnodes = HydroParseElementNodes(line, nodes, 64);
            for (PetscInt k = 1; k < nnodes; ++k) {
              PetscReal xyz[3];
              if (HydroFindNodeCoord(node_tags, node_xyz, n_node_tags, nodes[k], xyz)) PetscCall(HydroAppendNamedHeadSet(sets, name, xyz));
            }
          }
        }
      }
    }
  }
  fclose(fh);
  PetscCall(PetscFree(node_tags));
  PetscCall(PetscFree(node_xyz));
  PetscCheck((sets->dry.n > 0 && sets->porous.n > 0 && sets->free_head.n > 0) || sets->support.n > 0, comm, PETSC_ERR_ARG_WRONGSTATE,
             "Failed to parse hydro head nodesets from %s: dry=%" PetscInt_FMT " porous=%" PetscInt_FMT " free=%" PetscInt_FMT " support=%" PetscInt_FMT,
             mesh, sets->dry.n, sets->porous.n, sets->free_head.n, sets->support.n);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscBool HydroCoordInSet(const HydroCoordSet *set, const PetscReal x[3])
{
  const PetscReal tol = 1.0e-9;
  for (PetscInt i = 0; i < set->n; ++i) {
    const PetscReal *y = &set->xyz[3 * i];
    if (PetscAbsReal(x[0] - y[0]) <= tol && PetscAbsReal(x[1] - y[1]) <= tol && PetscAbsReal(x[2] - y[2]) <= tol) return PETSC_TRUE;
  }
  return PETSC_FALSE;
}

static PetscErrorCode HydroCreateScalarDM(MPI_Comm comm, const HydroOptions *opt, P4Basis *basis, DM *dm)
{
  DM cur = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMPlexCreateFromFile(comm, opt->mesh, NULL, PETSC_TRUE, &cur));
  PetscCall(DMSetFromOptions(cur));
  PetscCall(DMSetField(cur, 0, NULL, (PetscObject)basis->fe_scalar));
  PetscCall(DMCreateDS(cur));
  PetscCall(DMGetCoordinatesLocalSetUp(cur));
  PetscCall(DMSetMatType(cur, MATAIJ));
  *dm = cur;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroPointCentroid(DM dm, PetscInt point, PetscReal x[3])
{
  DM             cdm = NULL;
  Vec            coords = NULL;
  PetscSection   cs = NULL;
  PetscInt       nclosure = 0, *closure = NULL, nverts = 0;

  PetscFunctionBeginUser;
  x[0] = x[1] = x[2] = 0.0;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCall(DMGetLocalSection(cdm, &cs));
  PetscCall(DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &nclosure, &closure));
  for (PetscInt i = 0; i < nclosure; ++i) {
    PetscInt p = closure[2 * i], dof = 0;
    PetscCall(PetscSectionGetDof(cs, p, &dof));
    if (dof == 2 || dof == 3) {
      PetscScalar *vals = NULL;
      PetscInt     n = 0;
      PetscCall(DMPlexVecGetClosure(cdm, cs, coords, p, &n, &vals));
      if (n == dof) {
        x[0] += PetscRealPart(vals[0]);
        x[1] += PetscRealPart(vals[1]);
        if (dof == 3) x[2] += PetscRealPart(vals[2]);
        nverts++;
      }
      PetscCall(DMPlexVecRestoreClosure(cdm, cs, coords, p, &n, &vals));
    }
  }
  PetscCall(DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &nclosure, &closure));
  PetscCheck(nverts > 0, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Could not compute centroid for mesh point %" PetscInt_FMT, point);
  x[0] /= (PetscReal)nverts;
  x[1] /= (PetscReal)nverts;
  x[2] /= (PetscReal)nverts;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroMarkVertexTargets(DM dm, const HydroHeadSets *sets, HydroCtx *ctx)
{
  PetscInt vStart, vEnd;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(PetscCalloc4(vEnd, &ctx->target_dry, vEnd, &ctx->target_porous, vEnd, &ctx->target_free, vEnd, &ctx->target_support));
  for (PetscInt v = vStart; v < vEnd; ++v) {
    PetscReal x[3];
    PetscCall(HydroPointCentroid(dm, v, x));
    ctx->target_dry[v]    = HydroCoordInSet(&sets->dry, x);
    ctx->target_porous[v] = HydroCoordInSet(&sets->porous, x);
    ctx->target_free[v]   = HydroCoordInSet(&sets->free_head, x);
    ctx->target_support[v] = HydroCoordInSet(&sets->support, x);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroPointAllVerticesInTarget(DM dm, PetscInt point, const PetscBool target[], PetscBool *selected)
{
  PetscInt nclosure = 0, *closure = NULL, nverts = 0;

  PetscFunctionBeginUser;
  *selected = PETSC_FALSE;
  PetscCall(DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &nclosure, &closure));
  *selected = PETSC_TRUE;
  for (PetscInt i = 0; i < nclosure; ++i) {
    PetscInt p = closure[2 * i], depth;
    PetscCall(DMPlexGetPointDepth(dm, p, &depth));
    if (depth == 0) {
      nverts++;
      if (!target[p]) *selected = PETSC_FALSE;
    }
  }
  PetscCall(DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &nclosure, &closure));
  if (!nverts) *selected = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroPointTouchesBoundaryFace(DM dm, PetscInt point, PetscBool *touches)
{
  PetscInt nstar = 0, *star = NULL;

  PetscFunctionBeginUser;
  *touches = PETSC_FALSE;
  PetscCall(DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE, &nstar, &star));
  for (PetscInt i = 0; i < nstar; ++i) {
    PetscInt p = star[2 * i], height, support_size;
    PetscCall(DMPlexGetPointHeight(dm, p, &height));
    if (height != 1) continue;
    PetscCall(DMPlexGetSupportSize(dm, p, &support_size));
    if (support_size == 1) {
      *touches = PETSC_TRUE;
      break;
    }
  }
  PetscCall(DMPlexRestoreTransitiveClosure(dm, point, PETSC_FALSE, &nstar, &star));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscReal HydroPiecewiseHead(const HydroOptions *opt, const PetscReal x[3])
{
  PetscReal t;

  if (PetscAbsReal(opt->head_x1 - opt->head_x0) <= 1.0e-30) return opt->head_y0;
  t = (x[0] - opt->head_x0) / (opt->head_x1 - opt->head_x0);
  if (t < 0.0) t = 0.0;
  if (t > 1.0) t = 1.0;
  return opt->head_y0 + t * (opt->head_y1 - opt->head_y0);
}

static PetscReal HydroPrescribedValue(const HydroCtx *ctx, const PetscReal x[3], PetscBool porous, PetscBool freeh, PetscBool support)
{
  PetscReal prescribed = 0.0;

  if (support) prescribed = PetscMax(prescribed, ctx->opt.grho * PetscMax(HydroPiecewiseHead(&ctx->opt, x) - x[1], 0.0));
  if (porous) prescribed = PetscMax(prescribed, ctx->opt.grho * PetscMax(55.0 - x[1], 0.0));
  if (freeh) prescribed = PetscMax(prescribed, ctx->opt.grho * PetscMax(35.0 - x[1], 0.0));
  return prescribed;
}

static PetscErrorCode HydroBuildConstraints(HydroCtx *ctx)
{
  DM            dm = ctx->dm;
  MPI_Comm      comm = PetscObjectComm((PetscObject)dm);
  PetscSection  gsec = NULL;
  PetscInt      pStart, pEnd, cap = 0, nidx = 0, *idx = NULL;
  PetscScalar   value;

  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(dm, &ctx->prescribed));
  PetscCall(VecZeroEntries(ctx->prescribed));
  PetscCall(DMGetGlobalSection(dm, &gsec));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  cap = pEnd - pStart;
  PetscCall(PetscMalloc1(cap, &idx));
  for (PetscInt p = pStart; p < pEnd; ++p) {
    PetscInt  dof = 0, off = 0;
    PetscBool dry = PETSC_FALSE, porous = PETSC_FALSE, freeh = PETSC_FALSE, support = PETSC_FALSE, boundary = PETSC_TRUE;
    PetscReal x[3], prescribed = 0.0;
    PetscInt  depth;

    PetscCall(PetscSectionGetDof(gsec, p, &dof));
    if (dof <= 0) continue;
    PetscCall(DMPlexGetPointDepth(dm, p, &depth));
    if (depth > 0) {
      PetscCall(HydroPointTouchesBoundaryFace(dm, p, &boundary));
      if (!boundary) continue;
    }
    PetscCall(HydroPointAllVerticesInTarget(dm, p, ctx->target_dry, &dry));
    PetscCall(HydroPointAllVerticesInTarget(dm, p, ctx->target_porous, &porous));
    PetscCall(HydroPointAllVerticesInTarget(dm, p, ctx->target_free, &freeh));
    PetscCall(HydroPointAllVerticesInTarget(dm, p, ctx->target_support, &support));
    PetscCall(PetscSectionGetOffset(gsec, p, &off));
    if (off < 0) continue;
    PetscCall(HydroPointCentroid(dm, p, x));
    prescribed = HydroPrescribedValue(ctx, x, porous, freeh, (PetscBool)(strcmp(ctx->opt.head_mode, "support_piecewise") == 0 || strcmp(ctx->opt.head_mode, "support") == 0));
    value = prescribed;
    for (PetscInt d = 0; d < dof; ++d) {
      PetscCall(VecSetValue(ctx->prescribed, off + d, value, INSERT_VALUES));
      if (dry || porous || freeh || support) idx[nidx++] = off + d;
    }
  }
  PetscCall(VecAssemblyBegin(ctx->prescribed));
  PetscCall(VecAssemblyEnd(ctx->prescribed));
  PetscCall(ISCreateGeneral(comm, nidx, idx, PETSC_OWN_POINTER, &ctx->constrained_is));
  ctx->n_constrained_local = nidx;
  PetscCallMPI(MPI_Allreduce(&ctx->n_constrained_local, &ctx->n_constrained_global, 1, MPIU_INT, MPI_SUM, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroCtxCreate(DM dm, P4Basis *basis, const HydroHeadSets *sets, const HydroOptions *opt, HydroCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(ctx, sizeof(*ctx)));
  ctx->dm    = dm;
  ctx->basis = basis;
  ctx->opt   = *opt;
  PetscCall(HydroMarkVertexTargets(dm, sets, ctx));
  PetscCall(HydroBuildConstraints(ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroCtxDestroy(HydroCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(ISDestroy(&ctx->constrained_is));
  PetscCall(VecDestroy(&ctx->prescribed));
  PetscCall(PetscFree4(ctx->target_dry, ctx->target_porous, ctx->target_free, ctx->target_support));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void HydroGradPhys(const P4Basis *basis, PetscInt q, PetscInt b, const PetscReal invJ[9], PetscReal grad[3])
{
  const PetscReal *dref = &basis->basis_der[(q * basis->n_basis + b) * basis->dim];

  for (PetscInt d = 0; d < 3; ++d) grad[d] = 0.0;
  for (PetscInt d = 0; d < basis->dim; ++d) {
    for (PetscInt r = 0; r < basis->dim; ++r) grad[d] += invJ[r * basis->dim + d] * dref[r];
  }
}

static PetscErrorCode HydroPenaltyEps(DM dm, PetscInt cell, PetscReal *eps)
{
  PetscInt  nclosure = 0, *closure = NULL, nverts = 0;
  PetscReal pts[4][3], l12, l13, l23, l14, l24, l34;
  PetscInt  dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &nclosure, &closure));
  for (PetscInt i = 0; i < nclosure && nverts < 4; ++i) {
    PetscInt p = closure[2 * i], depth;
    PetscCall(DMPlexGetPointDepth(dm, p, &depth));
    if (depth == 0) {
      PetscCall(HydroPointCentroid(dm, p, pts[nverts]));
      nverts++;
    }
  }
  PetscCall(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &nclosure, &closure));
  PetscCheck((dim == 2 && nverts == 3) || (dim == 3 && nverts == 4), PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB,
             "Unexpected vertex count for hydro cell: dim=%" PetscInt_FMT " nverts=%" PetscInt_FMT, dim, nverts);
  l12 = PetscSqrtReal(PetscSqr(pts[0][0] - pts[1][0]) + PetscSqr(pts[0][1] - pts[1][1]) + PetscSqr(pts[0][2] - pts[1][2]));
  l13 = PetscSqrtReal(PetscSqr(pts[0][0] - pts[2][0]) + PetscSqr(pts[0][1] - pts[2][1]) + PetscSqr(pts[0][2] - pts[2][2]));
  l23 = PetscSqrtReal(PetscSqr(pts[1][0] - pts[2][0]) + PetscSqr(pts[1][1] - pts[2][1]) + PetscSqr(pts[1][2] - pts[2][2]));
  if (dim == 2) {
    *eps = 9.81 * PetscMin(PetscMin(l12, l13), l23) / 2.0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  l14 = PetscSqrtReal(PetscSqr(pts[0][0] - pts[3][0]) + PetscSqr(pts[0][1] - pts[3][1]) + PetscSqr(pts[0][2] - pts[3][2]));
  l24 = PetscSqrtReal(PetscSqr(pts[1][0] - pts[3][0]) + PetscSqr(pts[1][1] - pts[3][1]) + PetscSqr(pts[1][2] - pts[3][2]));
  l34 = PetscSqrtReal(PetscSqr(pts[2][0] - pts[3][0]) + PetscSqr(pts[2][1] - pts[3][1]) + PetscSqr(pts[1][2] - pts[3][2]));
  *eps = 9.81 * PetscMin(PetscMin(PetscMin(l12, l13), PetscMin(l23, l14)), PetscMin(l24, l34)) / 2.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroAssemble(HydroCtx *ctx, Vec pw, Mat A, Vec rhs, PetscBool nonlinear)
{
  DM              dm = ctx->dm;
  P4Basis        *basis = ctx->basis;
  PetscSection    lsec = NULL, gsec = NULL;
  Vec             pw_loc = NULL, rhs_loc = NULL, prescribed_loc = NULL;
  PetscInt        cStart, cEnd, ndof = basis->n_basis;
  PetscScalar    *elem_vec = NULL, *elem_mat = NULL;
  PetscLogDouble  t0, t1;
  PetscBool       initial_gravity = (PetscBool)(ctx->opt.dim == 3);

  PetscFunctionBeginUser;
  PetscCall(PetscTime(&t0));
  PetscCall(DMGetLocalSection(dm, &lsec));
  PetscCall(DMGetGlobalSection(dm, &gsec));
  PetscCall(MatZeroEntries(A));
  PetscCall(VecZeroEntries(rhs));
  PetscCall(DMGetLocalVector(dm, &rhs_loc));
  PetscCall(VecZeroEntries(rhs_loc));
  if (pw) {
    PetscCall(DMGetLocalVector(dm, &pw_loc));
    PetscCall(DMGlobalToLocalBegin(dm, pw, INSERT_VALUES, pw_loc));
    PetscCall(DMGlobalToLocalEnd(dm, pw, INSERT_VALUES, pw_loc));
  }
  if (!nonlinear && ctx->prescribed) {
    PetscCall(DMGetLocalVector(dm, &prescribed_loc));
    PetscCall(DMGlobalToLocalBegin(dm, ctx->prescribed, INSERT_VALUES, prescribed_loc));
    PetscCall(DMGlobalToLocalEnd(dm, ctx->prescribed, INSERT_VALUES, prescribed_loc));
  }
  PetscCall(PetscCalloc1(ndof, &elem_vec));
  PetscCall(PetscCalloc1(ndof * ndof, &elem_mat));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
    PetscReal     v0[3], J[9], invJ[9], detJ, eps;
    PetscScalar  *pw_cell = NULL;
    PetscInt      pw_size = 0, nidx = 0, *idx = NULL;
    PetscReal     grad[35][3], pw_vals[35], pwD_vals[35];

    PetscCall(PetscArrayzero(elem_vec, ndof));
    PetscCall(PetscArrayzero(elem_mat, ndof * ndof));
    PetscCall(DMPlexComputeCellGeometryFEM(dm, cell, NULL, v0, J, invJ, &detJ));
    detJ = PetscAbsReal(detJ);
    PetscCall(HydroPenaltyEps(dm, cell, &eps));
    if (pw_loc) {
      PetscCall(DMPlexVecGetClosure(dm, lsec, pw_loc, cell, &pw_size, &pw_cell));
      PetscCheck(pw_size == ndof, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Unexpected hydro closure size %" PetscInt_FMT " != %" PetscInt_FMT, pw_size, ndof);
      for (PetscInt a = 0; a < ndof; ++a) pw_vals[a] = PetscRealPart(pw_cell[a]);
    } else {
      for (PetscInt a = 0; a < ndof; ++a) pw_vals[a] = 0.0;
    }
    if (prescribed_loc) {
      PetscScalar *pd_cell = NULL;
      PetscInt     pd_size = 0;
      PetscCall(DMPlexVecGetClosure(dm, lsec, prescribed_loc, cell, &pd_size, &pd_cell));
      PetscCheck(pd_size == ndof, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Unexpected hydro prescribed closure size %" PetscInt_FMT " != %" PetscInt_FMT, pd_size, ndof);
      for (PetscInt a = 0; a < ndof; ++a) pwD_vals[a] = PetscRealPart(pd_cell[a]);
      PetscCall(DMPlexVecRestoreClosure(dm, lsec, prescribed_loc, cell, &pd_size, &pd_cell));
    } else {
      for (PetscInt a = 0; a < ndof; ++a) pwD_vals[a] = 0.0;
    }
    for (PetscInt q = 0; q < basis->n_qp; ++q) {
      PetscReal w = basis->weights[q] * detJ, pw_q = 0.0, grad_pw[3] = {0, 0, 0}, grad_pwD[3] = {0, 0, 0};
      PetscReal perm = 1.0, perm_der = 0.0;

      for (PetscInt a = 0; a < ndof; ++a) {
        HydroGradPhys(basis, q, a, invJ, grad[a]);
        if (!nonlinear) {
          for (PetscInt d = 0; d < basis->dim; ++d) grad_pwD[d] += grad[a][d] * pwD_vals[a];
        }
        if (nonlinear) {
          const PetscReal phi = basis->basis[q * ndof + a];
          pw_q += phi * pw_vals[a];
          for (PetscInt d = 0; d < basis->dim; ++d) grad_pw[d] += grad[a][d] * pw_vals[a];
        }
      }
      if (nonlinear) {
        if (pw_q < eps && pw_q > 0.0) {
          perm     = pw_q / eps;
          perm_der = 1.0 / eps;
        } else if (pw_q <= 0.0) {
          perm = 0.0;
        }
      }
      for (PetscInt a = 0; a < ndof; ++a) {
        if (nonlinear) {
          PetscReal flux = 0.0;
          for (PetscInt d = 0; d < basis->dim; ++d) flux += grad[a][d] * grad_pw[d];
          flux += grad[a][1] * ctx->opt.grho * perm;
          elem_vec[a] += -w * flux;
        } else {
          PetscReal flux = 0.0;
          for (PetscInt d = 0; d < basis->dim; ++d) flux += grad[a][d] * grad_pwD[d];
          if (initial_gravity) flux += ctx->opt.grho * grad[a][1];
          elem_vec[a] += -w * flux;
        }
        for (PetscInt b = 0; b < ndof; ++b) {
          PetscReal val = 0.0;
          for (PetscInt d = 0; d < basis->dim; ++d) val += grad[a][d] * grad[b][d];
          if (nonlinear) val += perm_der * ctx->opt.grho * grad[a][1] * basis->basis[q * ndof + b];
          elem_mat[a * ndof + b] += w * val;
        }
      }
    }
    if (pw_cell) PetscCall(DMPlexVecRestoreClosure(dm, lsec, pw_loc, cell, &pw_size, &pw_cell));
    PetscCall(DMPlexVecSetClosure(dm, lsec, rhs_loc, cell, elem_vec, ADD_VALUES));
    PetscCall(DMPlexGetClosureIndices(dm, lsec, gsec, cell, PETSC_TRUE, &nidx, &idx, NULL, NULL));
    for (PetscInt i = 0; i < nidx; ++i) {
      if (idx[i] < 0) continue;
      for (PetscInt j = 0; j < nidx; ++j) {
        if (idx[j] < 0) continue;
        PetscCall(MatSetValue(A, idx[i], idx[j], elem_mat[i * nidx + j], ADD_VALUES));
      }
    }
    PetscCall(DMPlexRestoreClosureIndices(dm, lsec, gsec, cell, PETSC_TRUE, &nidx, &idx, NULL, NULL));
  }
  PetscCall(DMLocalToGlobalBegin(dm, rhs_loc, ADD_VALUES, rhs));
  PetscCall(DMLocalToGlobalEnd(dm, rhs_loc, ADD_VALUES, rhs));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscFree(elem_vec));
  PetscCall(PetscFree(elem_mat));
  if (pw_loc) PetscCall(DMRestoreLocalVector(dm, &pw_loc));
  if (prescribed_loc) PetscCall(DMRestoreLocalVector(dm, &prescribed_loc));
  PetscCall(DMRestoreLocalVector(dm, &rhs_loc));
  PetscCall(PetscTime(&t1));
  ctx->assembly_time += t1 - t0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroBuildBasisReferencePoints(P4Basis *basis, PetscReal **points_out)
{
  PetscDualSpace dual;
  PetscReal     *points;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(basis->dim * basis->n_basis, &points));
  PetscCall(PetscFEGetDualSpace(basis->fe_scalar, &dual));
  for (PetscInt b = 0; b < basis->n_basis; ++b) {
    PetscQuadrature  q;
    PetscInt         dim, Nc, npoints;
    const PetscReal *p;
    PetscCall(PetscDualSpaceGetFunctional(dual, b, &q));
    PetscCall(PetscQuadratureGetData(q, &dim, &Nc, &npoints, &p, NULL));
    PetscCheck(dim == basis->dim && Nc == 1 && npoints >= 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected scalar dual functional shape");
    for (PetscInt d = 0; d < basis->dim; ++d) points[basis->dim * b + d] = p[d];
  }
  *points_out = points;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroCreateSameMeshScalarDM(DM fine_dm, P4Basis *basis, DM *level_dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMClone(fine_dm, level_dm));
  PetscCall(DMClearDS(*level_dm));
  PetscCall(DMClearFields(*level_dm));
  PetscCall(DMSetField(*level_dm, 0, NULL, (PetscObject)basis->fe_scalar));
  PetscCall(DMCreateDS(*level_dm));
  PetscCall(DMGetCoordinatesLocalSetUp(*level_dm));
  PetscCall(DMSetMatType(*level_dm, MATAIJ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroBuildInterpolationMatrix(DM fine_dm, P4Basis *fine_basis, DM coarse_dm, P4Basis *coarse_basis, Mat *P)
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
  PetscCall(HydroBuildBasisReferencePoints(fine_basis, &fine_points));
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
    for (PetscInt fb = 0; fb < fine_n; ++fb) {
      const PetscInt row = fine_idx[fb];
      if (row < rlo || row >= rhi) continue;
      for (PetscInt cb = 0; cb < coarse_n; ++cb) {
        const PetscInt    col = coarse_idx[cb];
        const PetscScalar val = phi[fb * coarse_basis->n_basis + cb];
        if (col < 0 || PetscAbsScalar(val) <= 1.0e-12) continue;
        PetscCall(MatSetValue(mat, row, col, val, INSERT_VALUES));
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

static PetscErrorCode HydroConfigurePC(KSP ksp, DM dm, P4Basis *fine_basis, const HydroOptions *opt)
{
  MPI_Comm  comm = PetscObjectComm((PetscObject)dm);
  PC        pc;
  PetscBool is_pmg, is_gamg, is_none;

  PetscFunctionBeginUser;
  PetscCall(KSPSetType(ksp, KSPFGMRES));
  PetscCall(KSPSetTolerances(ksp, opt->ksp_rtol, PETSC_DEFAULT, PETSC_DEFAULT, opt->ksp_max_it));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PetscStrcasecmp(opt->pc_variant, "pmg", &is_pmg));
  PetscCall(PetscStrcasecmp(opt->pc_variant, "gamg", &is_gamg));
  PetscCall(PetscStrcasecmp(opt->pc_variant, "none", &is_none));
  PetscCheck(is_pmg || is_gamg || is_none, comm, PETSC_ERR_ARG_WRONG, "-hydro_pc_variant must be pmg, gamg, or none");
  if (is_none) {
    PetscCall(PCSetType(pc, PCNONE));
  } else if (is_gamg || fine_basis->degree == 1) {
    PetscCall(PCSetType(pc, PCGAMG));
    PetscCall(PCGAMGSetType(pc, PCGAMGAGG));
  } else {
    P4Basis coarse_basis;
    DM      coarse_dm = NULL;
    Mat     P = NULL, R = NULL;
    KSP     smooth = NULL, coarse = NULL;
    PC      smooth_pc = NULL, coarse_pc = NULL;

    PetscCall(P4BasisCreateDegreeDim(PETSC_COMM_SELF, 1, fine_basis->dim, 1, &coarse_basis));
    PetscCall(HydroCreateSameMeshScalarDM(dm, &coarse_basis, &coarse_dm));
    PetscCall(HydroBuildInterpolationMatrix(dm, fine_basis, coarse_dm, &coarse_basis, &P));
    PetscCall(MatTranspose(P, MAT_INITIAL_MATRIX, &R));
    PetscCall(PCSetType(pc, PCMG));
    PetscCall(PCMGSetLevels(pc, 2, NULL));
    PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
    PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));
    PetscCall(PCMGSetInterpolation(pc, 1, P));
    PetscCall(PCMGSetRestriction(pc, 1, R));
    PetscCall(PCMGGetSmoother(pc, 1, &smooth));
    PetscCall(KSPSetType(smooth, KSPCHEBYSHEV));
    PetscCall(KSPSetTolerances(smooth, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, opt->pmg_smoother_max_it));
    PetscCall(KSPGetPC(smooth, &smooth_pc));
    PetscCall(PCSetType(smooth_pc, PCJACOBI));
    PetscCall(PCMGGetCoarseSolve(pc, &coarse));
    PetscCall(KSPSetType(coarse, KSPFGMRES));
    PetscCall(KSPSetTolerances(coarse, 1.0e-8, PETSC_DEFAULT, PETSC_DEFAULT, opt->pmg_coarse_max_it));
    PetscCall(KSPGetPC(coarse, &coarse_pc));
    PetscCall(PCSetType(coarse_pc, opt->pmg_coarse_pc_type));
    PetscCall(PetscPrintf(comm, "HYDRO_PMG_CONFIG levels=2 fine_degree=%" PetscInt_FMT " coarse_degree=1 smoother=chebyshev+jacobi smoother_max_it=%" PetscInt_FMT " coarse_ksp=fgmres coarse_pc=%s coarse_max_it=%" PetscInt_FMT "\n",
                          fine_basis->degree, opt->pmg_smoother_max_it, opt->pmg_coarse_pc_type, opt->pmg_coarse_max_it));
    PetscCall(MatDestroy(&P));
    PetscCall(MatDestroy(&R));
    PetscCall(DMDestroy(&coarse_dm));
    PetscCall(P4BasisDestroy(&coarse_basis));
  }
  PetscCall(KSPSetFromOptions(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroApplyDirichletInitial(HydroCtx *ctx, Mat A, Vec rhs)
{
  PetscFunctionBeginUser;
  PetscCall(MatZeroRowsColumnsIS(A, ctx->constrained_is, 1.0, ctx->prescribed, rhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroApplyDirichletCorrection(HydroCtx *ctx, Mat A, Vec rhs)
{
  Vec zero = NULL;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(ctx->prescribed, &zero));
  PetscCall(VecZeroEntries(zero));
  PetscCall(MatZeroRowsColumnsIS(A, ctx->constrained_is, 1.0, zero, rhs));
  PetscCall(VecDestroy(&zero));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroSolveLogged(KSP ksp, const char phase[], PetscInt step, Vec rhs, Vec x, PetscInt *its, PetscLogDouble *time)
{
  KSPConvergedReason reason;
  PetscLogDouble     t0, t1;

  PetscFunctionBeginUser;
  PetscCall(PetscTime(&t0));
  PetscCall(KSPSolve(ksp, rhs, x));
  PetscCall(PetscTime(&t1));
  PetscCall(KSPGetIterationNumber(ksp, its));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  *time = t1 - t0;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ksp),
                        "HYDRO_LINEAR phase=%s step=%" PetscInt_FMT " ksp_its=%" PetscInt_FMT " reason=%" PetscInt_FMT " solve_time=%.6g\n",
                        phase, step, *its, (PetscInt)reason, (double)*time));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroCountSaturated(HydroCtx *ctx, Vec pw, PetscInt *global_sat)
{
  DM           dm = ctx->dm;
  P4Basis     *basis = ctx->basis;
  PetscSection lsec = NULL;
  Vec          pw_loc = NULL;
  PetscInt     cStart, cEnd, local = 0;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalSection(dm, &lsec));
  PetscCall(DMGetLocalVector(dm, &pw_loc));
  PetscCall(DMGlobalToLocalBegin(dm, pw, INSERT_VALUES, pw_loc));
  PetscCall(DMGlobalToLocalEnd(dm, pw, INSERT_VALUES, pw_loc));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
    PetscScalar *pw_cell = NULL;
    PetscInt     n = 0;
    PetscReal    avg_num = 0.0, avg_den = 0.0, eps;
    PetscReal    v0[3], J[9], invJ[9], detJ;
    PetscCall(DMPlexComputeCellGeometryFEM(dm, cell, NULL, v0, J, invJ, &detJ));
    detJ = PetscAbsReal(detJ);
    PetscCall(HydroPenaltyEps(dm, cell, &eps));
    PetscCall(DMPlexVecGetClosure(dm, lsec, pw_loc, cell, &n, &pw_cell));
    for (PetscInt q = 0; q < basis->n_qp; ++q) {
      PetscReal pwq = 0.0, w = basis->weights[q] * detJ;
      for (PetscInt a = 0; a < n; ++a) pwq += basis->basis[q * basis->n_basis + a] * PetscRealPart(pw_cell[a]);
      avg_num += pwq * w;
      avg_den += w;
    }
    PetscCall(DMPlexVecRestoreClosure(dm, lsec, pw_loc, cell, &n, &pw_cell));
    if (avg_den > 0.0 && avg_num / avg_den >= 0.1 * eps) local++;
  }
  PetscCall(DMRestoreLocalVector(dm, &pw_loc));
  PetscCallMPI(MPI_Allreduce(&local, global_sat, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)dm)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroWriteSummary(const HydroOptions *opt, PetscInt ranks, PetscInt degree, PetscInt global_dofs, PetscInt cells, PetscInt constrained, PetscInt newton_its,
                                        PetscInt total_linear_its, PetscReal final_crit, PetscReal pmin, PetscReal pmax, PetscInt saturated, PetscLogDouble assembly_time,
                                        PetscLogDouble solve_time, PetscLogDouble wall_time)
{
  PetscMPIInt rank;
  FILE       *fh;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank != 0 || !opt->summary_json[0]) PetscFunctionReturn(PETSC_SUCCESS);
  fh = fopen(opt->summary_json, "w");
  PetscCheck(fh, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot write %s", opt->summary_json);
  fprintf(fh,
          "{\n"
          "  \"backend\": \"petsc\",\n"
          "  \"pc_variant\": \"%s\",\n"
          "  \"ranks\": %d,\n"
          "  \"degree\": %" PetscInt_FMT ",\n"
          "  \"global_dofs\": %" PetscInt_FMT ",\n"
          "  \"cells\": %" PetscInt_FMT ",\n"
          "  \"constrained_dofs\": %" PetscInt_FMT ",\n"
          "  \"newton_iterations\": %" PetscInt_FMT ",\n"
          "  \"linear_iterations\": %" PetscInt_FMT ",\n"
          "  \"final_criterion\": %.17g,\n"
          "  \"pressure_min\": %.17g,\n"
          "  \"pressure_max\": %.17g,\n"
          "  \"pressure_binary\": \"%s\",\n"
          "  \"saturated_elements\": %" PetscInt_FMT ",\n"
          "  \"assembly_time\": %.17g,\n"
          "  \"solve_time\": %.17g,\n"
          "  \"wall_time\": %.17g\n"
          "}\n",
          opt->pc_variant, ranks, degree, global_dofs, cells, constrained, newton_its, total_linear_its, (double)final_crit, (double)pmin, (double)pmax, opt->pressure_binary, saturated,
          (double)assembly_time, (double)solve_time, (double)wall_time);
  fclose(fh);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroWritePressureBinary(const HydroOptions *opt, Vec pressure)
{
  PetscViewer viewer = NULL;

  PetscFunctionBeginUser;
  if (!opt->pressure_binary[0]) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectSetName((PetscObject)pressure, "hydraulic_pressure"));
  PetscCall(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)pressure), opt->pressure_binary, FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(pressure, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pressure), "HYDRO_PRESSURE_OUTPUT binary=%s\n", opt->pressure_binary));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroWriteDofCoordsCSV(const HydroOptions *opt, DM dm)
{
  PetscViewer  viewer = NULL;
  PetscSection gsec = NULL;
  PetscInt     pStart, pEnd;
  PetscMPIInt  rank;

  PetscFunctionBeginUser;
  if (!opt->dof_coords_csv[0]) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(PetscViewerASCIIOpen(PetscObjectComm((PetscObject)dm), opt->dof_coords_csv, &viewer));
  if (rank == 0) PetscCall(PetscViewerASCIIPrintf(viewer, "global,x,y,z\n"));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  PetscCall(DMGetGlobalSection(dm, &gsec));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  for (PetscInt p = pStart; p < pEnd; ++p) {
    PetscInt  dof = 0, off = 0;
    PetscReal x[3];

    PetscCall(PetscSectionGetDof(gsec, p, &dof));
    if (dof <= 0) continue;
    PetscCall(PetscSectionGetOffset(gsec, p, &off));
    if (off < 0) continue;
    PetscCall(HydroPointCentroid(dm, p, x));
    for (PetscInt d = 0; d < dof; ++d) {
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%" PetscInt_FMT ",%.17g,%.17g,%.17g\n", off + d, (double)x[0], (double)x[1], (double)x[2]));
    }
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "HYDRO_DOF_COORDS csv=%s\n", opt->dof_coords_csv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode HydroSeepageRunFromOptions(void)
{
  MPI_Comm       comm = PETSC_COMM_WORLD;
  HydroOptions  opt;
  HydroHeadSets sets;
  P4Basis       basis;
  DM            dm = NULL;
  HydroCtx      ctx;
  Mat           A = NULL;
  Vec           rhs = NULL, pw = NULL, dp = NULL;
  KSP           ksp = NULL;
  PetscInt      cells_local, cells_global, global_dofs, total_linear = 0, saturated = 0;
  PetscReal     norm0, final_crit = PETSC_MAX_REAL, pmin = 0.0, pmax = 0.0;
  PetscLogDouble t0, t1, solve_time = 0.0, st;
  PetscMPIInt   ranks;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &ranks));
  PetscCall(PetscTime(&t0));
  PetscCall(HydroParseOptions(comm, &opt));
  PetscCall(PetscPrintf(comm, "HYDRO_OPTIONS mesh=%s dim=%" PetscInt_FMT " elem_type=%s pc_variant=%s head_mode=%s\n", opt.mesh, opt.dim, opt.elem_type, opt.pc_variant, opt.head_mode));
  PetscCall(HydroParseGmshHeadSets(comm, opt.mesh, &sets));
  PetscCall(PetscPrintf(comm, "HYDRO_HEADSET_PARSE dry=%" PetscInt_FMT " porous=%" PetscInt_FMT " free=%" PetscInt_FMT " support=%" PetscInt_FMT "\n", sets.dry.n, sets.porous.n, sets.free_head.n, sets.support.n));
  PetscCall(P4BasisCreateDegreeDim(PETSC_COMM_SELF, opt.degree, opt.dim, 1, &basis));
  PetscCall(HydroCreateScalarDM(comm, &opt, &basis, &dm));
  PetscCall(PetscPrintf(comm, "HYDRO_DM_CREATE degree=%" PetscInt_FMT " complete=true\n", opt.degree));
  PetscCall(HydroCtxCreate(dm, &basis, &sets, &opt, &ctx));
  PetscCall(PetscPrintf(comm, "HYDRO_CONSTRAINTS constrained_dofs=%" PetscInt_FMT " local_rank0=%" PetscInt_FMT "\n", ctx.n_constrained_global, ctx.n_constrained_local));
  PetscCall(DMCreateMatrix(dm, &A));
  PetscCall(DMCreateGlobalVector(dm, &rhs));
  PetscCall(VecDuplicate(rhs, &pw));
  PetscCall(VecDuplicate(rhs, &dp));
  PetscCall(VecGetSize(pw, &global_dofs));
  PetscCall(DMPlexGetHeightStratum(dm, 0, NULL, &cells_local));
  {
    PetscInt cStart, cEnd, local;
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    local = cEnd - cStart;
    PetscCallMPI(MPI_Allreduce(&local, &cells_global, 1, MPIU_INT, MPI_SUM, comm));
  }
  PetscCall(PetscPrintf(comm, "HYDRO_PETSC_MESH backend=petsc degree=%" PetscInt_FMT " ranks=%d global_dofs=%" PetscInt_FMT " cells=%" PetscInt_FMT " constrained_dofs=%" PetscInt_FMT " mesh=%s\n",
                        opt.degree, ranks, global_dofs, cells_global, ctx.n_constrained_global, opt.mesh));
  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(HydroConfigurePC(ksp, dm, &basis, &opt));

  PetscCall(HydroAssemble(&ctx, NULL, A, rhs, PETSC_FALSE));
  PetscCall(HydroApplyDirichletInitial(&ctx, A, rhs));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(HydroSolveLogged(ksp, "initial", 0, rhs, pw, &cells_local, &st));
  total_linear += cells_local;
  solve_time += st;
  PetscCall(VecNorm(pw, NORM_2, &norm0));
  if (norm0 <= 1.0e-14) norm0 = 1.0e-14;

  for (PetscInt it = 1; it <= opt.newton_max_it; ++it) {
    PetscReal dpnorm;
    PetscInt  its;

    PetscCall(HydroAssemble(&ctx, pw, A, rhs, PETSC_TRUE));
    PetscCall(HydroApplyDirichletCorrection(&ctx, A, rhs));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(VecZeroEntries(dp));
    PetscCall(HydroSolveLogged(ksp, "newton", it, rhs, dp, &its, &st));
    total_linear += its;
    solve_time += st;
    PetscCall(VecAXPY(pw, 1.0, dp));
    PetscCall(VecNorm(dp, NORM_2, &dpnorm));
    final_crit = dpnorm / norm0;
    PetscCall(PetscPrintf(comm, "HYDRO_NEWTON it=%" PetscInt_FMT " correction_norm=%.8e criterion=%.8e linear_its=%" PetscInt_FMT "\n", it, (double)dpnorm, (double)final_crit, its));
    if (final_crit < opt.newton_tol) {
      PetscCall(HydroCountSaturated(&ctx, pw, &saturated));
      PetscCall(VecMin(pw, NULL, &pmin));
      PetscCall(VecMax(pw, NULL, &pmax));
      PetscCall(PetscTime(&t1));
      PetscCall(HydroWritePressureBinary(&opt, pw));
      PetscCall(HydroWriteDofCoordsCSV(&opt, dm));
      PetscCall(PetscPrintf(comm,
                            "HYDRO_RESULT backend=petsc pc_variant=%s degree=%" PetscInt_FMT " ranks=%d global_dofs=%" PetscInt_FMT " cells=%" PetscInt_FMT " constrained_dofs=%" PetscInt_FMT " newton_iterations=%" PetscInt_FMT " total_linear_iterations=%" PetscInt_FMT " final_criterion=%.8e pressure_min=%.8e pressure_max=%.8e saturated_elements=%" PetscInt_FMT " assembly_time=%.6g solve_time=%.6g wall_time=%.6g summary_json=%s pressure_binary=%s\n",
                            opt.pc_variant, opt.degree, ranks, global_dofs, cells_global, ctx.n_constrained_global, it, total_linear, (double)final_crit, (double)pmin,
                            (double)pmax, saturated, (double)ctx.assembly_time, (double)solve_time, (double)(t1 - t0), opt.summary_json, opt.pressure_binary));
      PetscCall(HydroWriteSummary(&opt, ranks, opt.degree, global_dofs, cells_global, ctx.n_constrained_global, it, total_linear, final_crit, pmin, pmax, saturated, ctx.assembly_time,
                                  solve_time, t1 - t0));
      goto cleanup;
    }
  }
  PetscCall(HydroCountSaturated(&ctx, pw, &saturated));
  PetscCall(VecMin(pw, NULL, &pmin));
  PetscCall(VecMax(pw, NULL, &pmax));
  PetscCall(PetscTime(&t1));
  PetscCall(HydroWritePressureBinary(&opt, pw));
  PetscCall(HydroWriteDofCoordsCSV(&opt, dm));
  PetscCall(PetscPrintf(comm,
                        "HYDRO_RESULT backend=petsc pc_variant=%s degree=%" PetscInt_FMT " ranks=%d global_dofs=%" PetscInt_FMT " cells=%" PetscInt_FMT " constrained_dofs=%" PetscInt_FMT " newton_iterations=%" PetscInt_FMT " total_linear_iterations=%" PetscInt_FMT " final_criterion=%.8e pressure_min=%.8e pressure_max=%.8e saturated_elements=%" PetscInt_FMT " assembly_time=%.6g solve_time=%.6g wall_time=%.6g summary_json=%s pressure_binary=%s\n",
                        opt.pc_variant, opt.degree, ranks, global_dofs, cells_global, ctx.n_constrained_global, opt.newton_max_it, total_linear, (double)final_crit, (double)pmin,
                        (double)pmax, saturated, (double)ctx.assembly_time, (double)solve_time, (double)(t1 - t0), opt.summary_json, opt.pressure_binary));
  PetscCall(HydroWriteSummary(&opt, ranks, opt.degree, global_dofs, cells_global, ctx.n_constrained_global, opt.newton_max_it, total_linear, final_crit, pmin, pmax, saturated,
                              ctx.assembly_time, solve_time, t1 - t0));

cleanup:
  if (opt.log_view) PetscCall(PetscLogView(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&dp));
  PetscCall(VecDestroy(&pw));
  PetscCall(VecDestroy(&rhs));
  PetscCall(MatDestroy(&A));
  PetscCall(HydroCtxDestroy(&ctx));
  PetscCall(DMDestroy(&dm));
  PetscCall(P4BasisDestroy(&basis));
  PetscCall(HydroHeadSetsDestroy(&sets));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode HydroSeepageRunOptionsString(const char options[])
{
  PetscFunctionBeginUser;
  if (options && options[0]) PetscCall(PetscOptionsInsertString(NULL, options));
  PetscCall(HydroSeepageRunFromOptions());
  PetscFunctionReturn(PETSC_SUCCESS);
}
