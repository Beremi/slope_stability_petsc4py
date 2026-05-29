#include "assembly.h"
#include "petsc_ssr_algorithms.h"

#include <petscdualspace.h>
#include <petscdmplex.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  long long k[3];
  PetscReal x[3];
  PetscReal p;
} PressurePoint;

typedef struct {
  PetscReal x[3];
  PetscBool constrained[3];
} ConstraintPoint;

typedef struct {
  char      support_kind[32];
  char      support_name[128];
  char      dm_label[64];
  PetscInt  tag;
  PetscBool constrained[3];
} LabelConstraintRule;

static PetscErrorCode CopyConstrainedGlobalIndices(AssemblyCtx *ctx);
static PetscErrorCode RestrictConstrainedISOwned(DM dm, AssemblyCtx *ctx);

static long long PressureCoordKey(PetscReal x, PetscReal tol)
{
  return llround((double)(x / tol));
}

static int PressurePointCompare(const void *ap, const void *bp)
{
  const PressurePoint *a = (const PressurePoint *)ap;
  const PressurePoint *b = (const PressurePoint *)bp;
  for (PetscInt d = 0; d < 3; ++d) {
    if (a->k[d] < b->k[d]) return -1;
    if (a->k[d] > b->k[d]) return 1;
  }
  return 0;
}

static PetscErrorCode MechanicsPointCentroid(DM dm, PetscInt point, PetscReal x[3])
{
  DM           cdm = NULL;
  Vec          coords = NULL;
  PetscSection cs = NULL;
  PetscInt     nclosure = 0, *closure = NULL, nverts = 0;

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

static PetscBool ConstraintPointMatches(const ConstraintPoint *point, const PetscReal x[3], PetscReal tol)
{
  return (PetscAbsReal(point->x[0] - x[0]) <= tol && PetscAbsReal(point->x[1] - x[1]) <= tol && PetscAbsReal(point->x[2] - x[2]) <= tol) ? PETSC_TRUE : PETSC_FALSE;
}

static char *TrimField(char *text)
{
  char *end;

  while (*text && isspace((unsigned char)*text)) ++text;
  end = text + strlen(text);
  while (end > text && isspace((unsigned char)end[-1])) --end;
  *end = '\0';
  return text;
}

static PetscInt SplitCsvFields(char *line, char *fields[], PetscInt max_fields)
{
  PetscInt n = 0;
  char    *start = line;

  for (char *p = line;; ++p) {
    if (*p == ',' || *p == '\n' || *p == '\r' || *p == '\0') {
      const char sep = *p;

      *p = '\0';
      if (n < max_fields) fields[n++] = start;
      if (sep == '\0' || sep == '\n' || sep == '\r') break;
      start = p + 1;
    }
  }
  return n;
}

static PetscErrorCode SplitQuotedCsvFields(char *line, char *fields[], PetscInt max_fields, PetscInt *nfields, const char context[])
{
  PetscInt  n = 0;
  char     *src = line;
  char     *dst = line;
  PetscBool in_quotes = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCheck(nfields, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "SplitQuotedCsvFields requires an output field count");
  if (max_fields > 0) fields[n++] = dst;
  for (;;) {
    const char c = *src++;

    if (c == '\0') {
      PetscCheck(!in_quotes, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unterminated quoted %s CSV field", context);
      *dst = '\0';
      break;
    }
    if (c == '"') {
      if (in_quotes && *src == '"') {
        *dst++ = '"';
        ++src;
      } else {
        in_quotes = in_quotes ? PETSC_FALSE : PETSC_TRUE;
      }
      continue;
    }
    if (!in_quotes && (c == ',' || c == '\n' || c == '\r')) {
      *dst++ = '\0';
      if (c == '\n' || c == '\r') break;
      if (n < max_fields) fields[n] = dst;
      ++n;
      continue;
    }
    *dst++ = c;
  }
  *nfields = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ParseConstraintComponents(const char text[], PetscBool constrained[3])
{
  PetscBool any = PETSC_FALSE;

  PetscFunctionBeginUser;
  for (PetscInt c = 0; c < 3; ++c) constrained[c] = PETSC_FALSE;
  for (const char *p = text; *p;) {
    while (*p && (isspace((unsigned char)*p) || *p == ';' || *p == '|' || *p == '+')) ++p;
    if (!*p) break;
    if (*p == 'x' || *p == 'X' || *p == '0') {
      constrained[0] = PETSC_TRUE;
      any = PETSC_TRUE;
    } else if (*p == 'y' || *p == 'Y' || *p == '1') {
      constrained[1] = PETSC_TRUE;
      any = PETSC_TRUE;
    } else if (*p == 'z' || *p == 'Z' || *p == '2') {
      constrained[2] = PETSC_TRUE;
      any = PETSC_TRUE;
    } else {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported mechanics constraint component token starting at '%s'", p);
    }
    while (*p && !(isspace((unsigned char)*p) || *p == ';' || *p == '|' || *p == '+')) ++p;
  }
  PetscCheck(any, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mechanics label constraint row has no constrained components");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ParseLabelConstraintRule(char line[], LabelConstraintRule *rule, PetscBool *is_header)
{
  char    *fields[8];
  PetscInt nfields;
  char    *support_kind, *support_name, *dm_label, *tag_text, *components, *values;

  PetscFunctionBeginUser;
  *is_header = PETSC_FALSE;
  nfields = SplitCsvFields(line, fields, 8);
  PetscCheck(nfields >= 5, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mechanics label constraint row has %" PetscInt_FMT " fields; expected at least 5", nfields);
  support_kind = TrimField(fields[0]);
  support_name = TrimField(fields[1]);
  dm_label = TrimField(fields[2]);
  tag_text = TrimField(fields[3]);
  components = TrimField(fields[4]);
  values = nfields > 5 ? TrimField(fields[5]) : (char *)"";
  if (!strcmp(support_kind, "support_kind")) {
    *is_header = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(!values[0], PETSC_COMM_SELF, PETSC_ERR_SUP,
             "Nonzero mechanics label constraint values are not supported yet; row target=%s values=%s", support_name, values);
  PetscCall(PetscStrncpy(rule->support_kind, support_kind, sizeof(rule->support_kind)));
  PetscCall(PetscStrncpy(rule->support_name, support_name, sizeof(rule->support_name)));
  PetscCall(PetscStrncpy(rule->dm_label, dm_label, sizeof(rule->dm_label)));
  {
    char *end = NULL;
    long  tag = strtol(tag_text, &end, 10);

    PetscCheck(end && end != tag_text && !TrimField(end)[0], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Invalid mechanics label constraint tag %s", tag_text);
    rule->tag = (PetscInt)tag;
  }
  PetscCall(ParseConstraintComponents(components, rule->constrained));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscBool SortedIntArrayContains(const PetscInt idx[], PetscInt n, PetscInt value)
{
  PetscInt lo = 0, hi = n;

  while (lo < hi) {
    PetscInt mid = lo + (hi - lo) / 2;
    if (idx[mid] == value) return PETSC_TRUE;
    if (idx[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  return PETSC_FALSE;
}

static PetscErrorCode AppendPoint(PetscInt point, PetscInt **points, PetscInt *npoints, PetscInt *cap_points)
{
  PetscFunctionBeginUser;
  if (*npoints >= *cap_points) {
    *cap_points = *cap_points ? 2 * *cap_points : 128;
    PetscCall(PetscRealloc(sizeof(**points) * *cap_points, points));
  }
  (*points)[(*npoints)++] = point;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PointDepthFromStrata(DM dm, PetscInt point, PetscInt dim, PetscInt *depth)
{
  PetscFunctionBeginUser;
  *depth = -1;
  for (PetscInt d = 0; d <= dim; ++d) {
    PetscInt pStart, pEnd;

    PetscCall(DMPlexGetDepthStratum(dm, d, &pStart, &pEnd));
    if (point >= pStart && point < pEnd) {
      *depth = d;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PointClosureVerticesInSet(DM dm, PetscInt point, PetscInt dim, const PetscInt selected[], PetscInt nselected, PetscBool *all_selected, PetscInt *nverts)
{
  PetscInt nclosure = 0, *closure = NULL;

  PetscFunctionBeginUser;
  *all_selected = PETSC_TRUE;
  *nverts = 0;
  PetscCall(DMPlexGetTransitiveClosure(dm, point, PETSC_TRUE, &nclosure, &closure));
  for (PetscInt i = 0; i < nclosure; ++i) {
    PetscInt p = closure[2 * i], depth;

    PetscCall(PointDepthFromStrata(dm, p, dim, &depth));
    if (depth != 0) continue;
    ++*nverts;
    if (!SortedIntArrayContains(selected, nselected, p)) {
      *all_selected = PETSC_FALSE;
      break;
    }
  }
  PetscCall(DMPlexRestoreTransitiveClosure(dm, point, PETSC_TRUE, &nclosure, &closure));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PointTouchesBoundary(DM dm, PetscInt point, PetscInt depth, PetscInt dim, PetscBool *touches)
{
  PetscFunctionBeginUser;
  *touches = PETSC_FALSE;
  if (depth == dim - 1) {
    PetscInt support_size = 0;

    PetscCall(DMPlexGetSupportSize(dm, point, &support_size));
    *touches = support_size == 1 ? PETSC_TRUE : PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (depth > 0 && depth < dim - 1) {
    PetscInt nclosure = 0, *closure = NULL;

    PetscCall(DMPlexGetTransitiveClosure(dm, point, PETSC_FALSE, &nclosure, &closure));
    for (PetscInt i = 0; i < nclosure; ++i) {
      PetscInt p = closure[2 * i], pdepth;

      PetscCall(PointDepthFromStrata(dm, p, dim, &pdepth));
      if (pdepth == dim - 1) {
        PetscInt support_size = 0;

        PetscCall(DMPlexGetSupportSize(dm, p, &support_size));
        if (support_size == 1) {
          *touches = PETSC_TRUE;
          break;
        }
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, point, PETSC_FALSE, &nclosure, &closure));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppendConstraintDofsForPoint(AssemblyCtx *ctx, PetscSection gsec, PetscInt point, const PetscBool flags[3], PetscInt **idx, PetscInt *nidx, PetscInt *cap_idx, PetscBool *added)
{
  MPI_Comm comm;
  PetscInt dof = 0, off = 0, scalar_dofs;

  PetscFunctionBeginUser;
  *added = PETSC_FALSE;
  if (!flags[0] && !flags[1] && !flags[2]) PetscFunctionReturn(PETSC_SUCCESS);
  comm = PetscObjectComm((PetscObject)ctx->dm);
  PetscCall(PetscSectionGetDof(gsec, point, &dof));
  if (dof <= 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSectionGetOffset(gsec, point, &off));
  if (off < 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(dof % ctx->basis->components == 0, comm, PETSC_ERR_PLIB,
             "Point %" PetscInt_FMT " dof %" PetscInt_FMT " is not divisible by component count %" PetscInt_FMT,
             point, dof, ctx->basis->components);
  scalar_dofs = dof / ctx->basis->components;
  for (PetscInt s = 0; s < scalar_dofs; ++s) {
    for (PetscInt c = 0; c < ctx->basis->components && c < 3; ++c) {
      if (!flags[c]) continue;
      if (*nidx >= *cap_idx) {
        *cap_idx = *cap_idx ? 2 * *cap_idx : 1024;
        PetscCall(PetscRealloc(sizeof(**idx) * *cap_idx, idx));
      }
      (*idx)[(*nidx)++] = off + s * ctx->basis->components + c;
      *added = PETSC_TRUE;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppendExistingSectionConstraintDofsForComponents(AssemblyCtx *ctx, PetscSection gsec, const PetscBool flags[3], PetscInt **idx, PetscInt *nidx, PetscInt *cap_idx, PetscInt *matched_points)
{
  MPI_Comm comm;
  PetscSection lsec = NULL;
  PetscInt pStart, pEnd;

  PetscFunctionBeginUser;
  comm = PetscObjectComm((PetscObject)ctx->dm);
  PetscCall(DMGetLocalSection(ctx->dm, &lsec));
  PetscCall(DMPlexGetChart(ctx->dm, &pStart, &pEnd));
  for (PetscInt p = pStart; p < pEnd; ++p) {
    const PetscInt *constraint_indices = NULL;
    PetscInt        ldof = 0, gdof = 0, cdof = 0, off = 0;
    PetscBool       point_added = PETSC_FALSE;

    PetscCall(PetscSectionGetDof(lsec, p, &ldof));
    if (ldof <= 0) continue;
    PetscCall(PetscSectionGetConstraintDof(lsec, p, &cdof));
    if (cdof <= 0) continue;
    PetscCall(PetscSectionGetDof(gsec, p, &gdof));
    if (gdof <= 0) continue;
    PetscCall(PetscSectionGetOffset(gsec, p, &off));
    if (off < 0) continue;
    PetscCheck(ldof == gdof, comm, PETSC_ERR_PLIB,
               "Point %" PetscInt_FMT " local dof %" PetscInt_FMT " differs from global-section dof %" PetscInt_FMT,
               p, ldof, gdof);
    PetscCheck(gdof % ctx->basis->components == 0, comm, PETSC_ERR_PLIB,
               "Point %" PetscInt_FMT " dof %" PetscInt_FMT " is not divisible by component count %" PetscInt_FMT,
               p, gdof, ctx->basis->components);
    PetscCall(PetscSectionGetConstraintIndices(lsec, p, &constraint_indices));
    PetscCheck(constraint_indices || cdof == ldof, comm, PETSC_ERR_PLIB,
               "Point %" PetscInt_FMT " has partial constrained dofs but no section constraint indices", p);
    for (PetscInt k = 0; k < cdof; ++k) {
      const PetscInt local_dof = constraint_indices ? constraint_indices[k] : k;
      const PetscInt component = local_dof % ctx->basis->components;

      PetscCheck(local_dof >= 0 && local_dof < gdof, comm, PETSC_ERR_PLIB,
                 "Point %" PetscInt_FMT " constraint local dof %" PetscInt_FMT " is outside dof range %" PetscInt_FMT,
                 p, local_dof, gdof);
      if (component < 0 || component >= 3 || !flags[component]) continue;
      if (*nidx >= *cap_idx) {
        *cap_idx = *cap_idx ? 2 * *cap_idx : 1024;
        PetscCall(PetscRealloc(sizeof(**idx) * *cap_idx, idx));
      }
      (*idx)[(*nidx)++] = off + local_dof;
      point_added = PETSC_TRUE;
    }
    if (point_added) (*matched_points)++;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReplaceExtraConstraints(AssemblyCtx *ctx, PetscInt nidx, PetscInt idx[])
{
  MPI_Comm comm = PetscObjectComm((PetscObject)ctx->dm);

  PetscFunctionBeginUser;
  PetscCall(PetscFree(ctx->constrained_all));
  ctx->constrained_all = NULL;
  ctx->n_constrained_all = 0;
  PetscCall(ISDestroy(&ctx->constrained_all_is));
  PetscCall(ISDestroy(&ctx->constrained_is));
  PetscCall(ISCreateGeneral(comm, nidx, idx, PETSC_OWN_POINTER, &ctx->constrained_is));
  ctx->n_constrained_local = nidx;
  PetscCall(CopyConstrainedGlobalIndices(ctx));
  PetscCall(RestrictConstrainedISOwned(ctx->dm, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StagedBoundaryCopyField(MPI_Comm comm, const char path[], const char support_name[], const char field_name[], const char text[], char dest[], size_t dest_size)
{
  const size_t len = text ? strlen(text) : 0;

  PetscFunctionBeginUser;
  PetscCheck(dest && dest_size > 0, comm, PETSC_ERR_ARG_NULL, "StagedBoundaryCopyField requires a destination buffer");
  PetscCheck(text, comm, PETSC_ERR_ARG_NULL, "Boundary field %s for target %s in %s is NULL", field_name, support_name, path);
  PetscCheck(len < dest_size, comm, PETSC_ERR_ARG_SIZ,
             "Boundary field %s for target %s in %s is too long (%lu >= %lu)", field_name, support_name, path,
             (unsigned long)len, (unsigned long)dest_size);
  PetscCall(PetscStrncpy(dest, text, dest_size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AssemblyCtxClearSeepageBoundaryRules(AssemblyCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Assembly context is NULL");
  PetscCall(PetscFree(ctx->seepage_boundary_rules));
  ctx->seepage_boundary_rule_count = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ParseOptionalBoundaryGeometryOrder(MPI_Comm comm, const char path[], const char support_name[], const char geometry[], const char text[], PetscInt *order)
{
  char *end = NULL;
  long  parsed;

  PetscFunctionBeginUser;
  PetscCheck(order, comm, PETSC_ERR_ARG_NULL, "ParseOptionalBoundaryGeometryOrder requires an output order");
  *order = 0;
  if (!text || !text[0]) {
    PetscCheck(!geometry || !geometry[0], comm, PETSC_ERR_ARG_WRONG,
               "Boundary target %s in %s declares geometry but no positive geometry_order", support_name, path);
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  parsed = strtol(text, &end, 10);
  PetscCheck(end && end != text && !TrimField(end)[0], comm, PETSC_ERR_ARG_WRONG,
             "Invalid boundary geometry_order %s for target %s in %s", text, support_name, path);
  PetscCheck(parsed > 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "Boundary geometry_order must be positive for target %s in %s", support_name, path);
  PetscCheck(geometry && geometry[0], comm, PETSC_ERR_ARG_WRONG,
             "Boundary target %s in %s declares geometry_order %ld but no geometry patch", support_name, path, parsed);
  *order = (PetscInt)parsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ValidateSeepageBoundaryNativeStatus(MPI_Comm comm, const char path[], const char support_name[], const char field[], const char status[])
{
  PetscBool head_ready = PETSC_FALSE, flux_pending = PETSC_FALSE, is_head = PETSC_FALSE, is_flux = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCheck(status && status[0], comm, PETSC_ERR_ARG_WRONG, "Seepage boundary target %s in %s has empty native_status", support_name, path);
  PetscCall(PetscStrcasecmp(field, "head", &is_head));
  PetscCall(PetscStrcasecmp(field, "flux", &is_flux));
  PetscCall(PetscStrcasecmp(status, "label_ready_coordinate_pressure_bridge_active", &head_ready));
  PetscCall(PetscStrcasecmp(status, "pending_native_face_quadrature", &flux_pending));
  PetscCheck((is_head && head_ready) || (is_flux && flux_pending), comm, PETSC_ERR_SUP,
             "Seepage %s target %s in %s has unsupported native_status %s", field, support_name, path, status);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AssemblyCtxAppendSeepageBoundaryRule(AssemblyCtx *ctx, PetscInt *capacity, const char path[], const char field[], const char support_kind[],
                                                           const char support_name[], const char dm_label[], PetscInt tag, const char kind[], const char geometry[],
                                                           PetscInt geometry_order, const char value_model[], const char native_status[], PetscInt matched_points)
{
  MPI_Comm                    comm;
  AssemblySeepageBoundaryRule *rule = NULL;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Assembly context is NULL");
  PetscCheck(capacity, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "AssemblyCtxAppendSeepageBoundaryRule requires a capacity pointer");
  comm = PetscObjectComm((PetscObject)ctx->dm);
  if (ctx->seepage_boundary_rule_count >= *capacity) {
    *capacity = *capacity ? 2 * (*capacity) : 4;
    PetscCall(PetscRealloc(sizeof(*ctx->seepage_boundary_rules) * (*capacity), &ctx->seepage_boundary_rules));
  }
  rule = &ctx->seepage_boundary_rules[ctx->seepage_boundary_rule_count++];
  PetscCall(PetscMemzero(rule, sizeof(*rule)));
  PetscCall(StagedBoundaryCopyField(comm, path, support_name, "field", field, rule->field, sizeof(rule->field)));
  PetscCall(StagedBoundaryCopyField(comm, path, support_name, "support_kind", support_kind, rule->support_kind, sizeof(rule->support_kind)));
  PetscCall(StagedBoundaryCopyField(comm, path, support_name, "support_name", support_name, rule->support_name, sizeof(rule->support_name)));
  PetscCall(StagedBoundaryCopyField(comm, path, support_name, "dm_label", dm_label, rule->dm_label, sizeof(rule->dm_label)));
  PetscCall(StagedBoundaryCopyField(comm, path, support_name, "kind", kind, rule->kind, sizeof(rule->kind)));
  PetscCall(StagedBoundaryCopyField(comm, path, support_name, "geometry", geometry, rule->geometry, sizeof(rule->geometry)));
  PetscCall(StagedBoundaryCopyField(comm, path, support_name, "value_model", value_model, rule->value_model, sizeof(rule->value_model)));
  PetscCall(StagedBoundaryCopyField(comm, path, support_name, "native_status", native_status, rule->native_status, sizeof(rule->native_status)));
  rule->tag            = tag;
  rule->geometry_order = geometry_order;
  rule->matched_points = matched_points;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void GradPhys(const P4Basis *basis, PetscInt q, PetscInt b, const PetscReal invJ[9], PetscReal grad[3])
{
  const PetscReal *dref = &basis->basis_der[(q * basis->n_basis + b) * basis->dim];

  for (PetscInt d = 0; d < 3; ++d) grad[d] = 0.0;
  for (PetscInt d = 0; d < basis->dim; ++d) {
    for (PetscInt r = 0; r < basis->dim; ++r) grad[d] += invJ[r * basis->dim + d] * dref[r];
  }
}

static PetscErrorCode BuildBasisReferencePoints(P4Basis *basis, PetscReal **points_out)
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

static void ReferenceToPhysical(const P4Basis *basis, const PetscReal ref[], const PetscReal v0[3], const PetscReal J[9], PetscReal x[3])
{
  for (PetscInt d = 0; d < 3; ++d) x[d] = 0.0;
  for (PetscInt d = 0; d < basis->dim; ++d) {
    x[d] = v0[d];
    for (PetscInt r = 0; r < basis->dim; ++r) x[d] += J[d * basis->dim + r] * (ref[r] + 1.0);
  }
}

static PetscErrorCode PressureLookup(const AssemblyCtx *ctx, const PetscReal x[3], PetscReal *pressure)
{
  const PressurePoint *points = (const PressurePoint *)ctx->seepage_points;
  PressurePoint        key;
  PetscInt             lo = 0, hi = ctx->seepage_npoints;
  PetscReal            best = PETSC_MAX_REAL, best_x[3] = {0.0, 0.0, 0.0}, best_p = 0.0;

  PetscFunctionBeginUser;
  for (PetscInt d = 0; d < 3; ++d) {
    key.k[d] = PressureCoordKey(x[d], ctx->seepage_tol);
    key.x[d] = x[d];
  }
  while (lo < hi) {
    PetscInt mid = lo + (hi - lo) / 2;
    int      cmp = PressurePointCompare(&key, &points[mid]);
    if (cmp == 0) {
      *pressure = points[mid].p;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (cmp < 0) hi = mid;
    else lo = mid + 1;
  }
  for (PetscInt i = 0; i < ctx->seepage_npoints; ++i) {
    const PetscReal dx = PetscAbsReal(points[i].x[0] - x[0]);
    const PetscReal dy = PetscAbsReal(points[i].x[1] - x[1]);
    const PetscReal dz = PetscAbsReal(points[i].x[2] - x[2]);
    const PetscReal dist = PetscMax(dx, PetscMax(dy, dz));
    if (dist < best) {
      best = dist;
      best_x[0] = points[i].x[0];
      best_x[1] = points[i].x[1];
      best_x[2] = points[i].x[2];
      best_p = points[i].p;
    }
  }
  if (best <= 1.0e-6) {
    *pressure = best_p;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ctx->dm),
                        "SEEPAGE_PRESSURE_LOOKUP_FAILED x=(%.17g,%.17g,%.17g) nearest=(%.17g,%.17g,%.17g) nearest_linf=%.6e points=%" PetscInt_FMT " tol=%.3e\n",
                        (double)x[0], (double)x[1], (double)x[2], (double)best_x[0], (double)best_x[1], (double)best_x[2],
                        (double)best, ctx->seepage_npoints, (double)ctx->seepage_tol));
  SETERRQ(PetscObjectComm((PetscObject)ctx->dm), PETSC_ERR_ARG_WRONGSTATE,
          "Could not match seepage pressure point at x=(%.17g,%.17g,%.17g); table points=%" PetscInt_FMT " tol=%.3e",
          (double)x[0], (double)x[1], (double)x[2], ctx->seepage_npoints, (double)ctx->seepage_tol);
}

static PetscErrorCode CellPressureValues(AssemblyCtx *ctx, const PetscReal v0[3], const PetscReal J[9], PetscReal values[])
{
  PetscFunctionBeginUser;
  for (PetscInt a = 0; a < ctx->basis->n_basis; ++a) {
    PetscReal x[3];
    ReferenceToPhysical(ctx->basis, &ctx->basis_ref_points[ctx->basis->dim * a], v0, J, x);
    PetscCall(PressureLookup(ctx, x, &values[a]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void ReferenceVertex(PetscInt dim, PetscInt vertex, PetscReal ref[3])
{
  ref[0] = ref[1] = ref[2] = -1.0;
  if (vertex > 0 && vertex <= dim) ref[vertex - 1] = 1.0;
}

static PetscReal Distance3(const PetscReal a[3], const PetscReal b[3])
{
  return PetscSqrtReal(PetscSqr(a[0] - b[0]) + PetscSqr(a[1] - b[1]) + PetscSqr(a[2] - b[2]));
}

static PetscErrorCode CellSeepageEps(const AssemblyCtx *ctx, const PetscReal v0[3], const PetscReal J[9], PetscReal *eps)
{
  PetscReal verts[4][3], min_edge = PETSC_MAX_REAL;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < ctx->basis->dim + 1; ++i) {
    PetscReal ref[3];
    ReferenceVertex(ctx->basis->dim, i, ref);
    ReferenceToPhysical(ctx->basis, ref, v0, J, verts[i]);
  }
  for (PetscInt i = 0; i < ctx->basis->dim + 1; ++i) {
    for (PetscInt j = i + 1; j < ctx->basis->dim + 1; ++j) min_edge = PetscMin(min_edge, Distance3(verts[i], verts[j]));
  }
  *eps = ctx->seepage_grho * min_edge / 2.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void AddBTransposeStress3D(PetscScalar elem[], PetscInt a, const PetscReal grad[3], const PetscReal stress[6], PetscReal weight)
{
  elem[3 * a + 0] += weight * (grad[0] * stress[0] + grad[1] * stress[3] + grad[2] * stress[5]);
  elem[3 * a + 1] += weight * (grad[1] * stress[1] + grad[0] * stress[3] + grad[2] * stress[4]);
  elem[3 * a + 2] += weight * (grad[2] * stress[2] + grad[1] * stress[4] + grad[0] * stress[5]);
}

static void AddBTransposeStress2D(PetscScalar elem[], PetscInt a, const PetscReal grad[3], const PetscReal stress[3], PetscReal weight)
{
  elem[2 * a + 0] += weight * (grad[0] * stress[0] + grad[1] * stress[2]);
  elem[2 * a + 1] += weight * (grad[1] * stress[1] + grad[0] * stress[2]);
}

static void BColumn3D(PetscInt comp, const PetscReal grad[3], PetscReal bcol[6])
{
  for (PetscInt i = 0; i < 6; ++i) bcol[i] = 0.0;
  if (comp == 0) {
    bcol[0] = grad[0];
    bcol[3] = grad[1];
    bcol[5] = grad[2];
  } else if (comp == 1) {
    bcol[3] = grad[0];
    bcol[1] = grad[1];
    bcol[4] = grad[2];
  } else {
    bcol[5] = grad[0];
    bcol[4] = grad[1];
    bcol[2] = grad[2];
  }
}

static void BColumn2D(PetscInt comp, const PetscReal grad[3], PetscReal bcol[3])
{
  for (PetscInt i = 0; i < 3; ++i) bcol[i] = 0.0;
  if (comp == 0) {
    bcol[0] = grad[0];
    bcol[2] = grad[1];
  } else {
    bcol[1] = grad[1];
    bcol[2] = grad[0];
  }
}

static PetscErrorCode CellRegion(AssemblyCtx *ctx, PetscInt cell, PetscInt *region)
{
  PetscFunctionBeginUser;
  *region = 1;
  if (ctx->cell_sets) {
    PetscCall(DMLabelGetValue(ctx->cell_sets, cell, region));
    if (*region < 0) *region = 1;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CopyConstrainedGlobalIndices(AssemblyCtx *ctx)
{
  const PetscInt *idx;
  PetscBool       debug = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(ISAllGather(ctx->constrained_is, &ctx->constrained_all_is));
  PetscCall(ISSortRemoveDups(ctx->constrained_all_is));
  PetscCall(ISGetLocalSize(ctx->constrained_all_is, &ctx->n_constrained_all));
  if (ctx->n_constrained_all > 0) {
    PetscCall(PetscMalloc1(ctx->n_constrained_all, &ctx->constrained_all));
    PetscCall(ISGetIndices(ctx->constrained_all_is, &idx));
    PetscCall(PetscArraycpy(ctx->constrained_all, idx, ctx->n_constrained_all));
    PetscCall(ISRestoreIndices(ctx->constrained_all_is, &idx));
  }
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-debug_constraints", &debug, NULL));
  if (debug) {
    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)ctx->dm), &rank));
    PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)ctx->dm), "[%d] global constrained dofs:", rank));
    for (PetscInt i = 0; i < ctx->n_constrained_all; ++i) PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)ctx->dm), " %" PetscInt_FMT, ctx->constrained_all[i]));
    PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)ctx->dm), "\n"));
    PetscCall(PetscSynchronizedFlush(PetscObjectComm((PetscObject)ctx->dm), PETSC_STDOUT));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RestrictConstrainedISOwned(DM dm, AssemblyCtx *ctx)
{
  Vec       v;
  PetscInt lo, hi, nowned = 0, *owned = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(dm, &v));
  PetscCall(VecGetOwnershipRange(v, &lo, &hi));
  PetscCall(VecDestroy(&v));

  if (ctx->n_constrained_all > 0) PetscCall(PetscMalloc1(ctx->n_constrained_all, &owned));
  for (PetscInt i = 0; i < ctx->n_constrained_all; ++i) {
    if (ctx->constrained_all[i] >= lo && ctx->constrained_all[i] < hi) owned[nowned++] = ctx->constrained_all[i];
  }
  PetscCall(ISDestroy(&ctx->constrained_is));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), nowned, owned, PETSC_OWN_POINTER, &ctx->constrained_is));
  ctx->n_constrained_local = nowned;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscBool IsConstrainedGlobalDof(const AssemblyCtx *ctx, PetscInt idx)
{
  PetscInt lo = 0, hi = ctx->n_constrained_all;

  if (idx < 0 || ctx->n_constrained_all == 0) return PETSC_FALSE;
  while (lo < hi) {
    const PetscInt mid = lo + (hi - lo) / 2;
    if (ctx->constrained_all[mid] == idx) return PETSC_TRUE;
    if (ctx->constrained_all[mid] < idx) lo = mid + 1;
    else hi = mid;
  }
  return PETSC_FALSE;
}

PetscErrorCode BuildConstrainedGlobalIS(DM dm, IS *is, PetscInt *nlocal)
{
  PetscFunctionBeginUser;
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), 0, NULL, PETSC_COPY_VALUES, is));
  if (nlocal) *nlocal = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssemblyCtxCreate(DM dm, P4Basis *basis, AssemblyCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(ctx, sizeof(*ctx)));
  ctx->dm        = dm;
  ctx->basis     = basis;
  PetscCheck(basis->components == basis->dim, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP,
             "Mechanics assembly expects vector components == spatial dimension, got components=%" PetscInt_FMT " dim=%" PetscInt_FMT,
             basis->components, basis->dim);
  ctx->cell_dofs = basis->components * basis->n_basis;
  PetscCall(DMGetLabel(dm, "Cell Sets", &ctx->cell_sets));
  PetscCall(BuildConstrainedGlobalIS(dm, &ctx->constrained_is, &ctx->n_constrained_local));
  PetscCall(CopyConstrainedGlobalIndices(ctx));
  PetscCall(RestrictConstrainedISOwned(dm, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssemblyCtxLoadLabelConstraintsCSV(AssemblyCtx *ctx, const char path[], PetscBool *loaded, AssemblyLabelConstraintStats *stats)
{
  MPI_Comm            comm;
  FILE               *fh = NULL;
  char                line[2048];
  LabelConstraintRule rule;
  PetscSection        gsec = NULL;
  PetscInt           *idx = NULL, nidx = 0, cap_idx = 0;
  PetscInt            nrows = 0, local_matched = 0, global_matched = 0, global_rows = 0;
  PetscInt            dim = 0, pStart, pEnd;

  PetscFunctionBeginUser;
  if (loaded) *loaded = PETSC_FALSE;
  if (stats) PetscCall(PetscMemzero(stats, sizeof(*stats)));
  if (!path || !path[0]) PetscFunctionReturn(PETSC_SUCCESS);
  comm = PetscObjectComm((PetscObject)ctx->dm);
  fh = fopen(path, "r");
  PetscCheck(fh, comm, PETSC_ERR_FILE_OPEN, "Cannot open mechanics label constraint CSV %s", path);
  PetscCall(DMGetDimension(ctx->dm, &dim));
  PetscCall(DMGetGlobalSection(ctx->dm, &gsec));
  PetscCall(DMPlexGetChart(ctx->dm, &pStart, &pEnd));
  while (fgets(line, sizeof(line), fh)) {
    PetscBool is_header = PETSC_FALSE, is_face_label = PETSC_FALSE, is_vertex_label = PETSC_FALSE;
    DMLabel   label = NULL;
    IS        points = NULL;
    PetscInt  local_label_points = 0, global_label_points = 0;

    if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
    PetscCall(ParseLabelConstraintRule(line, &rule, &is_header));
    if (is_header) continue;
    nrows++;
    PetscCheck(rule.tag > 0, comm, PETSC_ERR_ARG_WRONG, "Mechanics label constraint row %s has invalid tag %" PetscInt_FMT, rule.support_name, rule.tag);
    for (PetscInt c = dim; c < 3; ++c) {
      PetscCheck(!rule.constrained[c], comm, PETSC_ERR_ARG_WRONG,
                 "Mechanics label constraint row %s constrains component %" PetscInt_FMT " but mesh dimension is %" PetscInt_FMT,
                 rule.support_name, c, dim);
    }
    PetscCall(PetscStrcasecmp(rule.dm_label, "Face Sets", &is_face_label));
    PetscCall(PetscStrcasecmp(rule.dm_label, "Vertex Sets", &is_vertex_label));
    PetscCall(DMGetLabel(ctx->dm, rule.dm_label, &label));
    if (!label && is_vertex_label) {
      PetscInt compat_matched = 0, global_compat_matched = 0;

      PetscCall(AppendExistingSectionConstraintDofsForComponents(ctx, gsec, rule.constrained, &idx, &nidx, &cap_idx, &compat_matched));
      PetscCallMPI(MPI_Allreduce(&compat_matched, &global_compat_matched, 1, MPIU_INT, MPI_SUM, comm));
      PetscCheck(global_compat_matched > 0, comm, PETSC_ERR_ARG_WRONGSTATE,
                 "Mesh has no DMPlex label %s for mechanics constraint target %s and no existing PETSc section constraints for the requested components",
                 rule.dm_label, rule.support_name);
      local_matched += compat_matched;
      PetscCall(PetscPrintf(comm,
                            "MECHANICS_BC_LABELS_COMPAT target=%s label=%s status=missing_vertex_label_using_section_constraints matched_points=%" PetscInt_FMT "\n",
                            rule.support_name, rule.dm_label, global_compat_matched));
      continue;
    }
    PetscCheck(label, comm, PETSC_ERR_ARG_WRONGSTATE, "Mesh has no DMPlex label %s for mechanics constraint target %s", rule.dm_label, rule.support_name);
    PetscCall(DMLabelGetStratumIS(label, rule.tag, &points));
    if (points) PetscCall(ISGetLocalSize(points, &local_label_points));
    PetscCallMPI(MPI_Allreduce(&local_label_points, &global_label_points, 1, MPIU_INT, MPI_SUM, comm));
    PetscCheck(global_label_points > 0, comm, PETSC_ERR_ARG_WRONGSTATE,
               "DMPlex label %s has no stratum tag %" PetscInt_FMT " for mechanics constraint target %s",
               rule.dm_label, rule.tag, rule.support_name);
    if (is_face_label) {
      const PetscInt *face_idx = NULL;
      PetscInt        nfaces = 0;

      if (points) {
        PetscCall(ISGetLocalSize(points, &nfaces));
        PetscCall(ISGetIndices(points, &face_idx));
        for (PetscInt i = 0; i < nfaces; ++i) {
          PetscInt nclosure = 0, *closure = NULL;

          PetscCall(DMPlexGetTransitiveClosure(ctx->dm, face_idx[i], PETSC_TRUE, &nclosure, &closure));
          for (PetscInt j = 0; j < nclosure; ++j) {
            PetscBool added = PETSC_FALSE;

            PetscCall(AppendConstraintDofsForPoint(ctx, gsec, closure[2 * j], rule.constrained, &idx, &nidx, &cap_idx, &added));
            if (added) local_matched++;
          }
          PetscCall(DMPlexRestoreTransitiveClosure(ctx->dm, face_idx[i], PETSC_TRUE, &nclosure, &closure));
        }
        PetscCall(ISRestoreIndices(points, &face_idx));
      }
    } else if (is_vertex_label) {
      IS              expanded_points = NULL;
      const PetscInt *selected = NULL;
      PetscInt        nselected = 0;

      if (points) {
        const PetscInt *raw_points = NULL;
        PetscInt       *expanded = NULL;
        PetscInt        nraw = 0, nexpanded = 0, cap_expanded = 0;

        PetscCall(ISGetLocalSize(points, &nraw));
        PetscCall(ISGetIndices(points, &raw_points));
        for (PetscInt i = 0; i < nraw; ++i) {
          PetscInt nclosure = 0, *closure = NULL;

          PetscCall(AppendPoint(raw_points[i], &expanded, &nexpanded, &cap_expanded));
          PetscCall(DMPlexGetTransitiveClosure(ctx->dm, raw_points[i], PETSC_TRUE, &nclosure, &closure));
          for (PetscInt j = 0; j < nclosure; ++j) {
            PetscInt p = closure[2 * j], depth = -1;

            PetscCall(PointDepthFromStrata(ctx->dm, p, dim, &depth));
            if (depth == 0) PetscCall(AppendPoint(p, &expanded, &nexpanded, &cap_expanded));
          }
          PetscCall(DMPlexRestoreTransitiveClosure(ctx->dm, raw_points[i], PETSC_TRUE, &nclosure, &closure));
        }
        PetscCall(ISRestoreIndices(points, &raw_points));
        PetscCall(ISCreateGeneral(comm, nexpanded, expanded, PETSC_OWN_POINTER, &expanded_points));
        PetscCall(ISSortRemoveDups(expanded_points));
        PetscCall(ISGetLocalSize(expanded_points, &nselected));
        PetscCall(ISGetIndices(expanded_points, &selected));
      }
      for (PetscInt p = pStart; p < pEnd; ++p) {
        PetscInt  depth = -1, nverts = 0;
        PetscBool all_selected = PETSC_FALSE, touches_boundary = PETSC_FALSE, use_point = PETSC_FALSE;

        PetscCall(PointDepthFromStrata(ctx->dm, p, dim, &depth));
        if (depth < 0 || depth >= dim) continue;
        if (depth == 0) {
          use_point = SortedIntArrayContains(selected, nselected, p);
        } else {
          PetscCall(PointTouchesBoundary(ctx->dm, p, depth, dim, &touches_boundary));
          if (touches_boundary) {
            PetscCall(PointClosureVerticesInSet(ctx->dm, p, dim, selected, nselected, &all_selected, &nverts));
            use_point = (PetscBool)(nverts > 0 && all_selected);
          }
        }
        if (use_point) {
          PetscBool added = PETSC_FALSE;

          PetscCall(AppendConstraintDofsForPoint(ctx, gsec, p, rule.constrained, &idx, &nidx, &cap_idx, &added));
          if (added) local_matched++;
        }
      }
      if (expanded_points) PetscCall(ISRestoreIndices(expanded_points, &selected));
      PetscCall(ISDestroy(&expanded_points));
    } else {
      SETERRQ(comm, PETSC_ERR_SUP, "Unsupported mechanics constraint DMPlex label %s for target %s", rule.dm_label, rule.support_name);
    }
    PetscCall(ISDestroy(&points));
  }
  fclose(fh);
  PetscCheck(nrows > 0, comm, PETSC_ERR_ARG_WRONGSTATE, "No mechanics label constraint rows were read from %s", path);

  PetscCallMPI(MPI_Allreduce(&local_matched, &global_matched, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&nidx, &global_rows, 1, MPIU_INT, MPI_SUM, comm));
  PetscCheck(global_rows > 0, comm, PETSC_ERR_ARG_WRONGSTATE, "Mechanics label constraint CSV %s did not resolve to any algebraic rows", path);
  PetscCall(ReplaceExtraConstraints(ctx, nidx, idx));
  if (stats) {
    stats->rows = nrows;
    stats->matched_points = global_matched;
    stats->raw_constrained_rows = global_rows;
    stats->owned_constraints = ctx->n_constrained_local;
    stats->global_constraints = ctx->n_constrained_all;
  }
  if (loaded) *loaded = PETSC_TRUE;
  PetscCall(PetscPrintf(comm,
                        "MECHANICS_BC_LABELS_CONFIG enabled=true path=%s rows=%" PetscInt_FMT " matched_points=%" PetscInt_FMT " raw_constrained_rows=%" PetscInt_FMT " owned_constraints=%" PetscInt_FMT " global_constraints=%" PetscInt_FMT "\n",
                        path, nrows, global_matched, global_rows, ctx->n_constrained_local, ctx->n_constrained_all));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssemblyCtxValidateSeepageBoundaryLabelsCSV(AssemblyCtx *ctx, const char path[], PetscInt expected_head_rows, PetscInt expected_flux_rows)
{
  MPI_Comm comm;
  FILE    *fh = NULL;
  char     line[4096];
  PetscInt rows = 0, head_rows = 0, flux_rows = 0, matched_points = 0, cap_rules = 0;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Assembly context is NULL");
  PetscCall(AssemblyCtxClearSeepageBoundaryRules(ctx));
  comm = PetscObjectComm((PetscObject)ctx->dm);
  if (!path || !path[0]) {
    PetscCheck(expected_head_rows <= 0 && expected_flux_rows <= 0, comm, PETSC_ERR_ARG_WRONG,
               "Native problem manifest declares seepage boundary rows but no seepage boundary label table is available");
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  fh = fopen(path, "r");
  PetscCheck(fh, comm, PETSC_ERR_FILE_OPEN, "Cannot open seepage boundary label CSV %s", path);
  while (fgets(line, sizeof(line), fh)) {
    char    *fields[10];
    PetscInt nfields;
    char    *field, *support_kind, *support_name, *dm_label, *tag_text, *kind, *geometry, *geometry_order_text, *value_model, *native_status;
    long     tag;
    char    *end = NULL;
    DMLabel  label = NULL;
    IS       points = NULL;
    PetscInt npoints = 0, global_points = 0;
    PetscInt geometry_order = 0;
    PetscBool is_head = PETSC_FALSE, is_flux = PETSC_FALSE, is_vertex_label = PETSC_FALSE;

    if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
    PetscCall(SplitQuotedCsvFields(line, fields, 10, &nfields, "seepage boundary label"));
    PetscCheck(nfields == 10, comm, PETSC_ERR_ARG_WRONG, "Seepage boundary label row in %s has %" PetscInt_FMT " fields; expected exactly 10", path, nfields);
    field = TrimField(fields[0]);
    if (!strcmp(field, "field")) continue;
    support_kind = TrimField(fields[1]);
    support_name = TrimField(fields[2]);
    dm_label = TrimField(fields[3]);
    tag_text = TrimField(fields[4]);
    kind = TrimField(fields[5]);
    geometry = TrimField(fields[6]);
    geometry_order_text = TrimField(fields[7]);
    value_model = TrimField(fields[8]);
    native_status = TrimField(fields[9]);
    tag = strtol(tag_text, &end, 10);
    PetscCheck(end && end != tag_text && !TrimField(end)[0], comm, PETSC_ERR_ARG_WRONG, "Invalid seepage boundary tag %s in %s", tag_text, path);
    PetscCheck(kind[0], comm, PETSC_ERR_ARG_WRONG, "Seepage boundary target %s in %s has empty kind", support_name, path);
    PetscCall(ParseOptionalBoundaryGeometryOrder(comm, path, support_name, geometry, geometry_order_text, &geometry_order));
    PetscCall(PetscStrcasecmp(field, "head", &is_head));
    PetscCall(PetscStrcasecmp(field, "flux", &is_flux));
    PetscCall(PetscStrcasecmp(dm_label, "Vertex Sets", &is_vertex_label));
    PetscCheck(is_head || is_flux, comm, PETSC_ERR_SUP, "Unsupported seepage boundary field %s in %s", field, path);
    PetscCall(ValidateSeepageBoundaryNativeStatus(comm, path, support_name, field, native_status));
    PetscCheck(!strcmp(support_kind, "boundary") || !strcmp(support_kind, "nodeset"), comm, PETSC_ERR_SUP,
               "Unsupported seepage support kind %s for target %s", support_kind, support_name);
    PetscCheck(!is_flux || !strcmp(support_kind, "boundary"), comm, PETSC_ERR_SUP, "Seepage flux target %s must use boundary support, got %s", support_name, support_kind);
    PetscCall(DMGetLabel(ctx->dm, dm_label, &label));
    if (!label && is_vertex_label && !strcmp(support_kind, "nodeset") && is_head && strstr(native_status, "coordinate_pressure_bridge_active")) {
      PetscCall(PetscPrintf(comm,
                            "SEEPAGE_BOUNDARY_LABELS_COMPAT target=%s label=%s status=missing_vertex_label_coordinate_pressure_bridge_active\n",
                            support_name, dm_label));
    } else {
      PetscCheck(label, comm, PETSC_ERR_ARG_WRONGSTATE, "Mesh has no DMPlex label %s for seepage %s target %s", dm_label, field, support_name);
      PetscCall(DMLabelGetStratumIS(label, (PetscInt)tag, &points));
      if (points) PetscCall(ISGetLocalSize(points, &npoints));
      PetscCallMPI(MPI_Allreduce(&npoints, &global_points, 1, MPIU_INT, MPI_SUM, comm));
      PetscCheck(global_points > 0, comm, PETSC_ERR_ARG_WRONGSTATE, "DMPlex label %s has no stratum tag %ld for seepage %s target %s", dm_label, tag, field, support_name);
      PetscCall(ISDestroy(&points));
    }
    PetscCall(AssemblyCtxAppendSeepageBoundaryRule(ctx, &cap_rules, path, field, support_kind, support_name, dm_label, (PetscInt)tag, kind, geometry, geometry_order,
                                                   value_model, native_status, global_points));
    rows++;
    matched_points += global_points;
    if (is_head) head_rows++;
    if (is_flux) flux_rows++;
  }
  fclose(fh);
  if (expected_head_rows >= 0) {
    PetscCheck(head_rows == expected_head_rows, comm, PETSC_ERR_ARG_WRONG,
               "Native problem manifest declares %" PetscInt_FMT " seepage head row(s), but label table %s contains %" PetscInt_FMT,
               expected_head_rows, path, head_rows);
  }
  if (expected_flux_rows >= 0) {
    PetscCheck(flux_rows == expected_flux_rows, comm, PETSC_ERR_ARG_WRONG,
               "Native problem manifest declares %" PetscInt_FMT " seepage flux row(s), but label table %s contains %" PetscInt_FMT,
               expected_flux_rows, path, flux_rows);
  }
  if (rows > 0) {
    PetscCall(PetscPrintf(comm,
                          "SEEPAGE_BOUNDARY_LABELS_CONFIG enabled=true path=%s rows=%" PetscInt_FMT " head=%" PetscInt_FMT " flux=%" PetscInt_FMT " matched_points=%" PetscInt_FMT " status=metadata_only_pressure_csv_active staged_rules=%" PetscInt_FMT "\n",
                          path, rows, head_rows, flux_rows, matched_points, ctx->seepage_boundary_rule_count));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssemblyCtxLoadCoordinateConstraintsCSV(AssemblyCtx *ctx, const char path[])
{
  MPI_Comm         comm;
  FILE            *fh = NULL;
  char             line[1024];
  ConstraintPoint *points = NULL;
  PetscInt         npoints = 0, cap_points = 0;
  PetscInt        *idx = NULL, nidx = 0, cap_idx = 0;
  PetscInt         pStart, pEnd, local_matched = 0, global_matched = 0, global_rows = 0;
  PetscSection     gsec = NULL;
  const PetscReal  tol = 1.0e-8;

  PetscFunctionBeginUser;
  if (!path || !path[0]) PetscFunctionReturn(PETSC_SUCCESS);
  comm = PetscObjectComm((PetscObject)ctx->dm);
  fh = fopen(path, "r");
  PetscCheck(fh, comm, PETSC_ERR_FILE_OPEN, "Cannot open mechanics constraint CSV %s", path);
  while (fgets(line, sizeof(line), fh)) {
    double x = 0.0, y = 0.0, z = 0.0;
    int    cx = 0, cy = 0, cz = 0;

    if (line[0] == '#' || strncmp(line, "x,", 2) == 0) continue;
    if (sscanf(line, " %lf , %lf , %lf , %d , %d , %d", &x, &y, &z, &cx, &cy, &cz) != 6) continue;
    if (!cx && !cy && !cz) continue;
    if (npoints >= cap_points) {
      cap_points = cap_points ? 2 * cap_points : 1024;
      PetscCall(PetscRealloc(sizeof(*points) * cap_points, &points));
    }
    points[npoints].x[0] = (PetscReal)x;
    points[npoints].x[1] = (PetscReal)y;
    points[npoints].x[2] = (PetscReal)z;
    points[npoints].constrained[0] = cx ? PETSC_TRUE : PETSC_FALSE;
    points[npoints].constrained[1] = cy ? PETSC_TRUE : PETSC_FALSE;
    points[npoints].constrained[2] = cz ? PETSC_TRUE : PETSC_FALSE;
    npoints++;
  }
  fclose(fh);
  PetscCheck(npoints > 0, comm, PETSC_ERR_ARG_WRONGSTATE, "No constrained mechanics rows were read from %s", path);

  PetscCall(DMGetGlobalSection(ctx->dm, &gsec));
  PetscCall(DMPlexGetChart(ctx->dm, &pStart, &pEnd));
  for (PetscInt p = pStart; p < pEnd; ++p) {
    PetscInt  dof = 0, cdof = 0, off = 0, scalar_dofs;
    PetscReal x[3];
    PetscBool flags[3] = {PETSC_FALSE, PETSC_FALSE, PETSC_FALSE};

    PetscCall(PetscSectionGetDof(gsec, p, &dof));
    if (dof <= 0) continue;
    PetscCall(PetscSectionGetConstraintDof(gsec, p, &cdof));
    if (cdof > 0) continue;
    PetscCall(PetscSectionGetOffset(gsec, p, &off));
    if (off < 0) continue;
    PetscCheck(dof % ctx->basis->components == 0, comm, PETSC_ERR_PLIB,
               "Point %" PetscInt_FMT " dof %" PetscInt_FMT " is not divisible by component count %" PetscInt_FMT,
               p, dof, ctx->basis->components);
    PetscCall(MechanicsPointCentroid(ctx->dm, p, x));
    for (PetscInt i = 0; i < npoints; ++i) {
      if (!ConstraintPointMatches(&points[i], x, tol)) continue;
      for (PetscInt c = 0; c < ctx->basis->components && c < 3; ++c) flags[c] = (PetscBool)(flags[c] || points[i].constrained[c]);
    }
    if (!flags[0] && !flags[1] && !flags[2]) continue;
    local_matched++;
    scalar_dofs = dof / ctx->basis->components;
    for (PetscInt s = 0; s < scalar_dofs; ++s) {
      for (PetscInt c = 0; c < ctx->basis->components && c < 3; ++c) {
        if (!flags[c]) continue;
        if (nidx >= cap_idx) {
          cap_idx = cap_idx ? 2 * cap_idx : 1024;
          PetscCall(PetscRealloc(sizeof(*idx) * cap_idx, &idx));
        }
        idx[nidx++] = off + s * ctx->basis->components + c;
      }
    }
  }

  PetscCallMPI(MPI_Allreduce(&local_matched, &global_matched, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&nidx, &global_rows, 1, MPIU_INT, MPI_SUM, comm));
  PetscCall(ReplaceExtraConstraints(ctx, nidx, idx));
  PetscCall(PetscPrintf(comm,
                        "MECHANICS_BC_NODES_CONFIG enabled=true path=%s rows=%" PetscInt_FMT " matched_points=%" PetscInt_FMT " raw_constrained_rows=%" PetscInt_FMT " owned_constraints=%" PetscInt_FMT " global_constraints=%" PetscInt_FMT "\n",
                        path, npoints, global_matched, global_rows, ctx->n_constrained_local, ctx->n_constrained_all));
  PetscCall(PetscFree(points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssemblyCtxDestroy(AssemblyCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscFree(ctx->constrained_all));
  PetscCall(PetscFree(ctx->seepage_points));
  PetscCall(PetscFree(ctx->basis_ref_points));
  PetscCall(PetscFree(ctx->neumann_rules));
  ctx->neumann_rule_count = 0;
  PetscCall(PetscFree(ctx->seepage_boundary_rules));
  ctx->seepage_boundary_rule_count = 0;
  PetscCall(ISDestroy(&ctx->constrained_all_is));
  PetscCall(ISDestroy(&ctx->constrained_is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssemblyCtxLoadSeepagePressureCSV(AssemblyCtx *ctx, const char path[], PetscReal grho)
{
  MPI_Comm       comm;
  FILE          *fh = NULL;
  char           line[4096];
  PressurePoint *points = NULL;
  PetscInt       n = 0, cap = 0, skipped = 0;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Assembly context is NULL");
  if (!path || !path[0]) PetscFunctionReturn(PETSC_SUCCESS);
  comm = PetscObjectComm((PetscObject)ctx->dm);
  fh = fopen(path, "r");
  PetscCheck(fh, comm, PETSC_ERR_FILE_OPEN, "Cannot open seepage pressure CSV %s", path);
  ctx->seepage_tol  = 1.0e-8;
  ctx->seepage_grho = grho;
  while (fgets(line, sizeof(line), fh)) {
    double x = 0.0, y = 0.0, z = 0.0, p = 0.0;
    char  *s = line;
    while (*s == ' ' || *s == '\t') s++;
    if (!*s || *s == '#' || (*s < '0' && *s != '-' && *s != '+' && *s != '.')) {
      skipped++;
      continue;
    }
    if (sscanf(s, "%lf,%lf,%lf,%lf", &x, &y, &z, &p) != 4) {
      skipped++;
      continue;
    }
    if (n >= cap) {
      cap = cap ? 2 * cap : 1024;
      PetscCall(PetscRealloc(sizeof(*points) * cap, &points));
    }
    points[n].x[0] = (PetscReal)x;
    points[n].x[1] = (PetscReal)y;
    points[n].x[2] = (PetscReal)z;
    points[n].p    = (PetscReal)p;
    for (PetscInt d = 0; d < 3; ++d) points[n].k[d] = PressureCoordKey(points[n].x[d], ctx->seepage_tol);
    n++;
  }
  fclose(fh);
  PetscCheck(n > 0, comm, PETSC_ERR_ARG_WRONGSTATE, "No pressure rows were read from %s (skipped=%" PetscInt_FMT ")", path, skipped);
  qsort(points, (size_t)n, sizeof(*points), PressurePointCompare);
  PetscCall(PetscFree(ctx->seepage_points));
  ctx->seepage_points  = points;
  ctx->seepage_npoints = n;
  ctx->seepage_enabled = PETSC_TRUE;
  if (!ctx->basis_ref_points) PetscCall(BuildBasisReferencePoints(ctx->basis, &ctx->basis_ref_points));
  PetscCall(PetscPrintf(comm, "SEEPAGE_COUPLING_CONFIG enabled=true pressure_csv=%s points=%" PetscInt_FMT " grho=%.12g tol=%.3e\n",
                        path, n, (double)ctx->seepage_grho, (double)ctx->seepage_tol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ZeroConstrainedVector(IS is, Vec v)
{
  const PetscInt *idx;
  PetscInt        n, lo, hi, nf = 0;
  PetscScalar    *zeros;
  PetscInt       *filtered;

  PetscFunctionBeginUser;
  PetscCall(ISGetLocalSize(is, &n));
  if (n > 0) {
    PetscCall(ISGetIndices(is, &idx));
    PetscCall(VecGetOwnershipRange(v, &lo, &hi));
    PetscCall(PetscMalloc1(n, &filtered));
    for (PetscInt i = 0; i < n; ++i) {
      if (idx[i] >= lo && idx[i] < hi) filtered[nf++] = idx[i];
    }
    if (nf > 0) {
      PetscCall(PetscCalloc1(nf, &zeros));
      PetscCall(VecSetValues(v, nf, filtered, zeros, INSERT_VALUES));
      PetscCall(PetscFree(zeros));
    }
    PetscCall(PetscFree(filtered));
    PetscCall(ISRestoreIndices(is, &idx));
  }
  PetscCall(VecAssemblyBegin(v));
  PetscCall(VecAssemblyEnd(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ApplyZeroDirichlet(IS is, Mat A, Vec rhs)
{
  PetscFunctionBeginUser;
  (void)A;
  PetscCall(ZeroConstrainedVector(is, rhs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssemblyWriteDisplacementPointCSV(AssemblyCtx *ctx, Vec u, const char path[])
{
  MPI_Comm       comm;
  PetscMPIInt    rank;
  DM             dm;
  PetscSection   lsec = NULL, gsec = NULL;
  Vec            u_loc = NULL;
  PetscViewer    viewer = NULL;
  PetscInt       cStart, cEnd, lo, hi, nloc, nseen = 0;
  PetscBool     *seen = NULL;

  PetscFunctionBeginUser;
  if (!path || !path[0]) PetscFunctionReturn(PETSC_SUCCESS);
  dm   = ctx->dm;
  comm = PetscObjectComm((PetscObject)dm);
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (!ctx->basis_ref_points) PetscCall(BuildBasisReferencePoints(ctx->basis, &ctx->basis_ref_points));
  PetscCall(DMGetLocalSection(dm, &lsec));
  PetscCall(DMGetGlobalSection(dm, &gsec));
  PetscCall(VecGetOwnershipRange(u, &lo, &hi));
  PetscCall(VecGetLocalSize(u, &nloc));
  PetscCall(PetscCalloc1(nloc, &seen));
  PetscCall(DMGetLocalVector(dm, &u_loc));
  PetscCall(DMGlobalToLocalBegin(dm, u, INSERT_VALUES, u_loc));
  PetscCall(DMGlobalToLocalEnd(dm, u, INSERT_VALUES, u_loc));
  PetscCall(PetscViewerASCIIOpen(comm, path, &viewer));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  if (rank == 0) PetscCall(PetscViewerASCIIPrintf(viewer, "x,y,z,ux,uy,uz\n"));

  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
    PetscReal     v0[3], J[9], invJ[9], detJ;
    PetscScalar  *u_cell = NULL;
    PetscInt     *gidx = NULL;
    PetscInt      u_size = 0, nidx = 0;

    PetscCall(DMPlexComputeCellGeometryFEM(dm, cell, NULL, v0, J, invJ, &detJ));
    PetscCall(DMPlexVecGetClosure(dm, lsec, u_loc, cell, &u_size, &u_cell));
    PetscCall(DMPlexGetClosureIndices(dm, lsec, gsec, cell, PETSC_TRUE, &nidx, &gidx, NULL, NULL));
    PetscCheck(u_size == ctx->cell_dofs && nidx == ctx->cell_dofs, comm, PETSC_ERR_PLIB,
               "Unexpected solution closure size on cell %" PetscInt_FMT ": values=%" PetscInt_FMT " indices=%" PetscInt_FMT " expected=%" PetscInt_FMT,
               cell, u_size, nidx, ctx->cell_dofs);
    for (PetscInt b = 0; b < ctx->basis->n_basis; ++b) {
      PetscInt  key = -1;
      PetscReal x[3], disp[3] = {0.0, 0.0, 0.0};

      for (PetscInt c = 0; c < ctx->basis->components; ++c) {
        const PetscInt idx = b * ctx->basis->components + c;
        if (c < 3) disp[c] = PetscRealPart(u_cell[idx]);
        if (key < 0 && gidx[idx] >= lo && gidx[idx] < hi) key = gidx[idx];
      }
      if (key < lo || key >= hi) continue;
      if (seen[key - lo]) continue;
      seen[key - lo] = PETSC_TRUE;
      nseen++;
      ReferenceToPhysical(ctx->basis, &ctx->basis_ref_points[ctx->basis->dim * b], v0, J, x);
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                                                   (double)x[0], (double)x[1], (double)x[2],
                                                   (double)disp[0], (double)disp[1], (double)disp[2]));
    }
    PetscCall(DMPlexRestoreClosureIndices(dm, lsec, gsec, cell, PETSC_TRUE, &nidx, &gidx, NULL, NULL));
    PetscCall(DMPlexVecRestoreClosure(dm, lsec, u_loc, cell, &u_size, &u_cell));
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIIPopSynchronized(viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(DMRestoreLocalVector(dm, &u_loc));
  PetscCall(PetscPrintf(comm, "SSR_SOLUTION_OUTPUT kind=point_csv path=%s local_points=%" PetscInt_FMT "\n", path, nseen));
  PetscCall(PetscFree(seen));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetClosureFreeDofs(AssemblyCtx *ctx, PetscSection lsec, PetscSection gsec, Mat A, PetscInt cell, PetscScalar elem_mat[], PetscScalar free_mat[], PetscInt free_idx[], PetscInt free_pos[])
{
  DM           dm = ctx->dm;
  PetscInt    num_indices = 0, *global_indices = NULL, nfree = 0;
  PetscScalar *values = elem_mat, *values_orig = elem_mat;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetClosureIndices(dm, lsec, gsec, cell, PETSC_TRUE, &num_indices, &global_indices, NULL, &values));
  PetscCheck(num_indices == ctx->cell_dofs, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB,
             "Unexpected matrix closure size %" PetscInt_FMT " != %" PetscInt_FMT " on cell %" PetscInt_FMT, num_indices, ctx->cell_dofs, cell);

  for (PetscInt i = 0; i < num_indices; ++i) {
    if (global_indices[i] < 0 || IsConstrainedGlobalDof(ctx, global_indices[i])) continue;
    free_pos[nfree] = i;
    free_idx[nfree] = global_indices[i];
    ++nfree;
  }

  if (nfree > 0) {
    for (PetscInt i = 0; i < nfree; ++i) {
      const PetscInt ii = free_pos[i];
      for (PetscInt j = 0; j < nfree; ++j) {
        const PetscInt jj = free_pos[j];
        free_mat[i * nfree + j] = values[ii * num_indices + jj];
      }
    }
    PetscCall(MatSetValues(A, nfree, free_idx, nfree, free_idx, free_mat, ADD_VALUES));
  }

  PetscCall(DMPlexRestoreClosureIndices(dm, lsec, gsec, cell, PETSC_TRUE, &num_indices, &global_indices, NULL, &values));
  if (values != values_orig) PetscCall(DMRestoreWorkArray(dm, 0, MPIU_SCALAR, &values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetConstrainedDiagonals(AssemblyCtx *ctx, PetscBool use_local_indices, PetscSection lsec, PetscSection gsec, Mat A)
{
  const PetscInt *idx = NULL;
  PetscInt        n = 0;
  PetscScalar     one = 1.0;

  PetscFunctionBeginUser;
  (void)lsec;
  (void)gsec;
  if (use_local_indices) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(ISGetLocalSize(ctx->constrained_is, &n));
  if (n > 0) {
    PetscCall(ISGetIndices(ctx->constrained_is, &idx));
    for (PetscInt i = 0; i < n; ++i) PetscCall(MatSetValue(A, idx[i], idx[i], one, ADD_VALUES));
    PetscCall(ISRestoreIndices(ctx->constrained_is, &idx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatEnsureDiagonalSlots(DM dm, PetscBool use_local_indices, PetscSection lsec, Mat A)
{
  PetscScalar zero = 0.0;

  PetscFunctionBeginUser;
  (void)use_local_indices;
  (void)lsec;
  {
    Vec      v = NULL;
    PetscInt lo, hi;

    PetscCall(DMCreateGlobalVector(dm, &v));
    PetscCall(VecGetOwnershipRange(v, &lo, &hi));
    PetscCall(VecDestroy(&v));
    for (PetscInt i = lo; i < hi; ++i) PetscCall(MatSetValue(A, i, i, zero, ADD_VALUES));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AssembleCells(AssemblyCtx *ctx, PetscReal lambda, Vec u, Vec f_ext, Mat A, Vec residual, PetscBool plastic, PetscBool assemble_jacobian)
{
  DM              dm = ctx->dm;
  P4Basis        *basis = ctx->basis;
  PetscSection    lsec, gsec;
  Vec             u_loc = NULL, probe_loc = NULL, out_loc = NULL;
  Vec             out = residual ? residual : f_ext;
  PetscInt        cStart, cEnd;
  PetscScalar    *elem_vec = NULL, *elem_mat = NULL, *free_mat = NULL;
  PetscInt       *free_idx = NULL, *free_pos = NULL;
  const PetscInt  ndof = ctx->cell_dofs;
  const PetscInt  dim = basis->dim;
  const PetscInt  ncomp = basis->components;
  const PetscInt  nstrain = (dim == 2) ? 3 : 6;
  const PetscInt  gravity_comp = 1;
  const SsrMaterialOps *material_ops = NULL;
  PetscBool       use_local_mat_indices = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCheck(dim == 2 || dim == 3, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Mechanics assembly supports only 2D or 3D");
  PetscCheck(ndof <= 105 && nstrain <= 6, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Element workspace is too small for this FE space");
  PetscCall(DMGetLocalSection(dm, &lsec));
  PetscCall(DMGetGlobalSection(dm, &gsec));
  if (u) {
    PetscCall(DMGetLocalVector(dm, &u_loc));
    PetscCall(DMGlobalToLocalBegin(dm, u, INSERT_VALUES, u_loc));
    PetscCall(DMGlobalToLocalEnd(dm, u, INSERT_VALUES, u_loc));
    PetscCall(DMPlexInsertBoundaryValues(dm, PETSC_TRUE, u_loc, 0.0, NULL, NULL, NULL));
  } else {
    PetscCall(DMGetLocalVector(dm, &probe_loc));
    PetscCall(VecZeroEntries(probe_loc));
  }
  if (A && assemble_jacobian) {
    PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &use_local_mat_indices));
    PetscCall(MatZeroEntries(A));
  }
  PetscCall(SsrMaterialRegistryFind("mohr_coulomb", &material_ops));
  PetscCheck(material_ops->evaluate, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Material model mohr_coulomb has no native evaluator");
  if (out) {
    PetscCall(VecZeroEntries(out));
    PetscCall(DMGetLocalVector(dm, &out_loc));
    PetscCall(VecZeroEntries(out_loc));
  }
  PetscCall(PetscCalloc1(ndof, &elem_vec));
  if (A && assemble_jacobian) {
    PetscCall(PetscCalloc1(ndof * ndof, &elem_mat));
    PetscCall(PetscMalloc1(ndof * ndof, &free_mat));
    PetscCall(PetscMalloc1(ndof, &free_idx));
    PetscCall(PetscMalloc1(ndof, &free_pos));
  }

  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (PetscInt cell = cStart; cell < cEnd; ++cell) {
    PetscReal     v0[3], J[9], invJ[9], detJ;
    PetscScalar  *u_cell = NULL;
    PetscInt      u_size = 0, region;
    PetscReal     grad[35][3];
    PetscReal     pressure_vals[35], cell_pressure_avg = 0.0, cell_weight = 0.0, cell_eps = 0.0;
    PetscReal     cell_gamma_sat = 0.0, cell_gamma_unsat = 0.0;
    PetscBool     have_cell_material = PETSC_FALSE;

    PetscCall(PetscArrayzero(elem_vec, ndof));
    if (elem_mat) PetscCall(PetscArrayzero(elem_mat, ndof * ndof));
    PetscCall(DMPlexComputeCellGeometryFEM(dm, cell, NULL, v0, J, invJ, &detJ));
    detJ = PetscAbsReal(detJ);
    PetscCall(CellRegion(ctx, cell, &region));
    if (u_loc || probe_loc) {
      PetscCall(DMPlexVecGetClosure(dm, lsec, u_loc ? u_loc : probe_loc, cell, &u_size, &u_cell));
      PetscCheck(u_size == ndof, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Unexpected closure size %" PetscInt_FMT " != %" PetscInt_FMT, u_size, ndof);
    }
    if (ctx->seepage_enabled && !residual) {
      PetscCall(CellPressureValues(ctx, v0, J, pressure_vals));
      PetscCall(CellSeepageEps(ctx, v0, J, &cell_eps));
    }

    for (PetscInt q = 0; q < basis->n_qp; ++q) {
      PetscReal strain[6] = {0, 0, 0, 0, 0, 0};
      PetscReal stress[6], tangent[36];
      PetscReal B[105][6], DB[105][6];
      PetscReal w = basis->weights[q] * detJ;
      PetscReal pw_q = 0.0, grad_pw[3] = {0.0, 0.0, 0.0};
      SsrMaterialPointInput  material_input;
      SsrMaterialPointResult material_result;

      for (PetscInt a = 0; a < basis->n_basis; ++a) {
        GradPhys(basis, q, a, invJ, grad[a]);
        if (ctx->seepage_enabled && !residual) {
          const PetscReal phi = basis->basis[q * basis->n_basis + a];
          pw_q += phi * pressure_vals[a];
          for (PetscInt d = 0; d < dim; ++d) grad_pw[d] += grad[a][d] * pressure_vals[a];
        }
        for (PetscInt ca = 0; ca < ncomp; ++ca) {
          if (dim == 2) BColumn2D(ca, grad[a], B[ncomp * a + ca]);
          else BColumn3D(ca, grad[a], B[ncomp * a + ca]);
        }
        if (u_cell) {
          const PetscReal ux = PetscRealPart(u_cell[ncomp * a + 0]);
          const PetscReal uy = PetscRealPart(u_cell[ncomp * a + 1]);
          if (dim == 2) {
            strain[0] += grad[a][0] * ux;
            strain[1] += grad[a][1] * uy;
            strain[2] += grad[a][1] * ux + grad[a][0] * uy;
          } else {
            const PetscReal uz = PetscRealPart(u_cell[ncomp * a + 2]);
            strain[0] += grad[a][0] * ux;
            strain[1] += grad[a][1] * uy;
            strain[2] += grad[a][2] * uz;
            strain[3] += grad[a][1] * ux + grad[a][0] * uy;
            strain[4] += grad[a][2] * uy + grad[a][1] * uz;
            strain[5] += grad[a][2] * ux + grad[a][0] * uz;
          }
        }
      }
      if (ctx->seepage_enabled && !residual) {
        cell_pressure_avg += pw_q * w;
        cell_weight += w;
      }
      material_input.dim     = dim;
      material_input.region  = region;
      material_input.plastic = plastic;
      material_input.lambda  = lambda;
      material_input.strain  = strain;
      material_result.stress = stress;
      material_result.tangent = tangent;
      material_result.gamma_sat = 0.0;
      material_result.gamma_unsat = 0.0;
      PetscCall(material_ops->evaluate(NULL, &material_input, &material_result));
      if (!have_cell_material) {
        cell_gamma_sat     = material_result.gamma_sat;
        cell_gamma_unsat   = material_result.gamma_unsat;
        have_cell_material = PETSC_TRUE;
      }

      if (residual) {
        for (PetscInt a = 0; a < basis->n_basis; ++a) {
          if (dim == 2) AddBTransposeStress2D(elem_vec, a, grad[a], stress, w);
          else AddBTransposeStress3D(elem_vec, a, grad[a], stress, w);
        }
      } else {
        const PetscReal gamma = ctx->seepage_enabled ? 0.0 : material_result.gamma_sat;
        for (PetscInt a = 0; a < basis->n_basis; ++a) {
          const PetscReal phi = basis->basis[q * basis->n_basis + a];
          if (ctx->seepage_enabled) {
            for (PetscInt d = 0; d < dim; ++d) elem_vec[ncomp * a + d] += w * phi * (-grad_pw[d]);
          } else {
            elem_vec[ncomp * a + gravity_comp] += w * phi * (-gamma);
          }
        }
      }

      if (elem_mat) {
        for (PetscInt j = 0; j < ndof; ++j) {
          for (PetscInt r = 0; r < nstrain; ++r) {
            DB[j][r] = 0.0;
            for (PetscInt s = 0; s < nstrain; ++s) DB[j][r] += tangent[r + nstrain * s] * B[j][s];
          }
        }
        for (PetscInt i = 0; i < ndof; ++i) {
          for (PetscInt j = 0; j < ndof; ++j) {
            PetscReal val = 0.0;
            for (PetscInt r = 0; r < nstrain; ++r) val += B[i][r] * DB[j][r];
            elem_mat[i * ndof + j] += w * val;
          }
        }
      }
    }
    if (ctx->seepage_enabled && !residual) {
      const PetscReal gamma = (cell_weight > 0.0 && cell_pressure_avg / cell_weight >= 0.1 * cell_eps) ? cell_gamma_sat : cell_gamma_unsat;
      for (PetscInt q = 0; q < basis->n_qp; ++q) {
        PetscReal w = basis->weights[q] * detJ;
        for (PetscInt a = 0; a < basis->n_basis; ++a) {
          const PetscReal phi = basis->basis[q * basis->n_basis + a];
          elem_vec[ncomp * a + gravity_comp] += w * phi * (-gamma);
        }
      }
    }
    if (elem_mat) {
      /* Keep the tangent exactly symmetric for CG/GAMG despite roundoff-order noise. */
      for (PetscInt i = 0; i < ndof; ++i) {
        for (PetscInt j = i + 1; j < ndof; ++j) {
          const PetscScalar v = 0.5 * (elem_mat[i * ndof + j] + elem_mat[j * ndof + i]);
          elem_mat[i * ndof + j] = v;
          elem_mat[j * ndof + i] = v;
        }
      }
    }
    if (u_cell) PetscCall(DMPlexVecRestoreClosure(dm, lsec, u_loc ? u_loc : probe_loc, cell, &u_size, &u_cell));
    if (residual) {
      PetscCall(DMPlexVecSetClosure(dm, lsec, out_loc, cell, elem_vec, ADD_VALUES));
    } else if (f_ext) {
      PetscCall(DMPlexVecSetClosure(dm, lsec, out_loc, cell, elem_vec, ADD_VALUES));
    }
    if (elem_mat) PetscCall(MatSetClosureFreeDofs(ctx, lsec, gsec, A, cell, elem_mat, free_mat, free_idx, free_pos));
  }

  if (out) {
    PetscCall(DMLocalToGlobalBegin(dm, out_loc, ADD_VALUES, out));
    PetscCall(DMLocalToGlobalEnd(dm, out_loc, ADD_VALUES, out));
  }
  if (A && assemble_jacobian) {
    /*
      PETSc section constraints remove essential DOFs from the global algebraic
      system. Keep owned diagonal slots explicit for preconditioners without
      adding artificial constrained unit rows.
    */
    PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    PetscCall(MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE));
    PetscCall(MatEnsureDiagonalSlots(dm, use_local_mat_indices, lsec, A));
    PetscCall(MatSetConstrainedDiagonals(ctx, use_local_mat_indices, lsec, gsec, A));
    PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  if (residual) {
    PetscCall(VecAXPY(residual, -1.0, f_ext));
  }
  PetscCall(PetscFree(elem_vec));
  PetscCall(PetscFree(elem_mat));
  PetscCall(PetscFree(free_mat));
  PetscCall(PetscFree(free_idx));
  PetscCall(PetscFree(free_pos));
  if (u_loc) PetscCall(DMRestoreLocalVector(dm, &u_loc));
  if (probe_loc) PetscCall(DMRestoreLocalVector(dm, &probe_loc));
  if (out_loc) PetscCall(DMRestoreLocalVector(dm, &out_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssembleElasticProblem(AssemblyCtx *ctx, Mat A, Vec f_ext)
{
  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(f_ext));
  PetscCall(AssembleCells(ctx, 1.0, NULL, f_ext, A, NULL, PETSC_FALSE, PETSC_TRUE));
  PetscCall(AssemblyCtxAssembleNeumannResidual(ctx, f_ext));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssemblePlasticResidualJacobian(AssemblyCtx *ctx, PetscReal lambda, Vec u, Vec f_ext, Mat A, Vec residual, PetscBool assemble_jacobian)
{
  PetscFunctionBeginUser;
  PetscCall(AssembleCells(ctx, lambda, u, f_ext, A, residual, PETSC_TRUE, assemble_jacobian));
  PetscFunctionReturn(PETSC_SUCCESS);
}
