#include "assembly.h"
#include "petsc_ssr_algorithms.h"
#include "petsc_ssr_stats.h"

#include <petscdualspace.h>
#include <petscdmplex.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct _p_SsrNeumannValueCtx {
  PetscInt  dim;
  PetscReal constant_traction[3];
};

static char *NeumannTrimField(char *text)
{
  char *end;

  while (*text && isspace((unsigned char)*text)) ++text;
  end = text + strlen(text);
  while (end > text && isspace((unsigned char)end[-1])) --end;
  *end = '\0';
  return text;
}

static PetscErrorCode NeumannSplitCsvFields(char *line, char *fields[], PetscInt max_fields, PetscInt *nfields)
{
  PetscInt  n = 0;
  char     *src = line;
  char     *dst = line;
  PetscBool in_quotes = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCheck(nfields, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "NeumannSplitCsvFields requires an output field count");
  if (max_fields > 0) fields[n++] = dst;
  for (;;) {
    const char c = *src++;

    if (c == '\0') {
      PetscCheck(!in_quotes, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unterminated quoted mechanics Neumann CSV field");
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

static PetscErrorCode NeumannExtractJsonString(const char json[], const char key[], char value[], size_t value_size, PetscBool *found)
{
  const char *p;

  PetscFunctionBeginUser;
  *found = PETSC_FALSE;
  if (value_size > 0) value[0] = '\0';
  if (!json || !json[0]) PetscFunctionReturn(PETSC_SUCCESS);
  p = strstr(json, key);
  if (!p) PetscFunctionReturn(PETSC_SUCCESS);
  p += strlen(key);
  while (*p && isspace((unsigned char)*p)) ++p;
  PetscCheck(*p == ':', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed Neumann value_model near key %s: %s", key, json);
  ++p;
  while (*p && isspace((unsigned char)*p)) ++p;
  PetscCheck(*p == '"', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Neumann value_model key %s must be a string in %s", key, json);
  ++p;
  {
    size_t n = 0;

    while (*p && *p != '"') {
      PetscCheck(n + 1 < value_size, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Neumann value_model key %s is too long", key);
      value[n++] = *p++;
    }
    PetscCheck(*p == '"', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unterminated Neumann value_model string for key %s in %s", key, json);
    value[n] = '\0';
  }
  *found = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NeumannValueRegistryName(const char kind[], const char value_model[], char name[], size_t name_size)
{
  char      type[64] = "";
  PetscBool found = PETSC_FALSE;
  PetscBool constant = PETSC_FALSE;
  PetscBool traction = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(NeumannExtractJsonString(value_model, "\"type\"", type, sizeof(type), &found));
  if (!found || !type[0]) PetscCall(PetscStrncpy(type, kind, sizeof(type)));
  PetscCall(PetscStrcasecmp(type, "constant", &constant));
  PetscCall(PetscStrcasecmp(kind, "traction", &traction));
  if (constant && traction) PetscCall(PetscStrncpy(name, "constant-traction", name_size));
  else PetscCall(PetscStrncpy(name, type, name_size));
  for (char *p = name; *p; ++p) {
    if (*p == '_') *p = '-';
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NeumannValidateValueModel(const char path[], const char support_name[], const char kind[], const char value_model[], char model_name[], size_t model_name_size)
{
  const SsrNeumannValueOps *ops = NULL;

  PetscFunctionBeginUser;
  PetscCall(NeumannValueRegistryName(kind, value_model, model_name, model_name_size));
  PetscCheck(model_name[0], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mechanics Neumann target %s in %s has no value model name", support_name, path);
  PetscCall(SsrNeumannValueRegistryFind(model_name, &ops));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NeumannParseGeometryOrder(MPI_Comm comm, const char path[], const char support_name[], const char geometry[], const char text[], PetscInt *order)
{
  char *end = NULL;
  long  parsed;

  PetscFunctionBeginUser;
  PetscCheck(order, comm, PETSC_ERR_ARG_NULL, "NeumannParseGeometryOrder requires an output order");
  *order = 0;
  if (!text || !text[0]) {
    PetscCheck(!geometry || !geometry[0], comm, PETSC_ERR_ARG_WRONG,
               "Mechanics Neumann target %s in %s declares geometry but no positive geometry_order", support_name, path);
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  parsed = strtol(text, &end, 10);
  PetscCheck(end && end != text && !NeumannTrimField(end)[0], comm, PETSC_ERR_ARG_WRONG,
             "Invalid mechanics Neumann geometry_order %s for target %s in %s", text, support_name, path);
  PetscCheck(parsed > 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "Mechanics Neumann geometry_order must be positive for target %s in %s", support_name, path);
  PetscCheck(geometry && geometry[0], comm, PETSC_ERR_ARG_WRONG,
             "Mechanics Neumann target %s in %s declares geometry_order %ld but no geometry patch", support_name, path, parsed);
  *order = (PetscInt)parsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NeumannValidateNativeStatus(MPI_Comm comm, const char path[], const char support_name[], const char status[], PetscInt geometry_order)
{
  PetscBool affine = PETSC_FALSE, curved = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCheck(status && status[0], comm, PETSC_ERR_ARG_WRONG, "Mechanics Neumann target %s in %s has empty native_status", support_name, path);
  PetscCall(PetscStrcasecmp(status, "native_face_quadrature_affine", &affine));
  PetscCall(PetscStrcasecmp(status, "pending_native_curved_face_quadrature", &curved));
  if (geometry_order > 0) {
    PetscCheck(curved, comm, PETSC_ERR_SUP,
               "Mechanics Neumann target %s in %s has geometry_order %" PetscInt_FMT " but native_status %s is not pending_native_curved_face_quadrature",
               support_name, path, geometry_order, status);
  } else {
    PetscCheck(affine, comm, PETSC_ERR_SUP,
               "Mechanics Neumann target %s in %s has unsupported native_status %s for affine native face quadrature",
               support_name, path, status);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SsrNeumannConstantTractionEvaluate(SsrNeumannValueCtx ctx, const SsrNeumannValueInput *input, SsrNeumannValueResult *result)
{
  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Constant-traction Neumann value model requires a context");
  PetscCheck(input, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Constant-traction Neumann value model requires input");
  PetscCheck(result, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Constant-traction Neumann value model requires result");
  PetscCheck(result->traction, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Constant-traction Neumann value model requires traction output");
  PetscCheck(input->dim == ctx->dim, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
             "Constant-traction Neumann value model input dim %" PetscInt_FMT " does not match context dim %" PetscInt_FMT, input->dim, ctx->dim);
  PetscCheck(input->dim > 0 && input->dim <= 3, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
             "Constant-traction Neumann value model dimension %" PetscInt_FMT " is not supported", input->dim);
  for (PetscInt d = 0; d < input->dim; ++d) result->traction[d] = ctx->constant_traction[d];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NeumannParseConstantTraction(MPI_Comm comm, const AssemblyNeumannRule *rule, PetscInt dim, SsrNeumannValueCtx value_ctx)
{
  const char *p;
  PetscBool  constant_traction = PETSC_FALSE;
  PetscReal  values[3] = {0.0, 0.0, 0.0};
  PetscInt   n = 0;

  PetscFunctionBeginUser;
  PetscCheck(value_ctx, comm, PETSC_ERR_ARG_NULL, "NeumannParseConstantTraction requires a value context");
  value_ctx->dim = dim;
  for (PetscInt d = 0; d < 3; ++d) value_ctx->constant_traction[d] = 0.0;
  PetscCall(PetscStrcasecmp(rule->value_model_name, "constant-traction", &constant_traction));
  PetscCheck(constant_traction, comm, PETSC_ERR_SUP,
             "Mechanics Neumann target %s uses value model %s; native affine face quadrature currently supports constant-traction only",
             rule->support_name, rule->value_model_name);
  p = strstr(rule->value_model, "\"value\"");
  PetscCheck(p, comm, PETSC_ERR_ARG_WRONG, "Mechanics Neumann target %s constant-traction value_model has no value array: %s", rule->support_name, rule->value_model);
  p += strlen("\"value\"");
  while (*p && isspace((unsigned char)*p)) ++p;
  PetscCheck(*p == ':', comm, PETSC_ERR_ARG_WRONG, "Malformed constant-traction value_model for target %s: %s", rule->support_name, rule->value_model);
  ++p;
  while (*p && isspace((unsigned char)*p)) ++p;
  PetscCheck(*p == '[', comm, PETSC_ERR_ARG_WRONG, "Constant-traction value for target %s must be an array: %s", rule->support_name, rule->value_model);
  ++p;
  while (*p && *p != ']') {
    char *end = NULL;

    while (*p && (isspace((unsigned char)*p) || *p == ',')) ++p;
    PetscCheck(*p && *p != ']', comm, PETSC_ERR_ARG_WRONG, "Unexpected end of constant-traction value array for target %s", rule->support_name);
    PetscCheck(n < 3, comm, PETSC_ERR_ARG_OUTOFRANGE, "Constant-traction value for target %s has more than three components", rule->support_name);
    values[n++] = (PetscReal)strtod(p, &end);
    PetscCheck(end && end != p, comm, PETSC_ERR_ARG_WRONG, "Invalid constant-traction component near %s for target %s", p, rule->support_name);
    p = end;
    while (*p && isspace((unsigned char)*p)) ++p;
    PetscCheck(*p == ',' || *p == ']', comm, PETSC_ERR_ARG_WRONG, "Expected comma or ] in constant-traction value for target %s: %s", rule->support_name, p);
  }
  PetscCheck(*p == ']', comm, PETSC_ERR_ARG_WRONG, "Unterminated constant-traction value array for target %s", rule->support_name);
  PetscCheck(n == dim || n == 3, comm, PETSC_ERR_ARG_WRONG,
             "Constant-traction value for target %s has %" PetscInt_FMT " components, expected %" PetscInt_FMT " or 3",
             rule->support_name, n, dim);
  for (PetscInt d = dim; d < n; ++d) {
    PetscCheck(PetscAbsReal(values[d]) <= PETSC_SMALL, comm, PETSC_ERR_ARG_WRONG,
               "Constant-traction value for %" PetscInt_FMT "D target %s has nonzero extra component %" PetscInt_FMT,
               dim, rule->support_name, d);
  }
  for (PetscInt d = 0; d < dim; ++d) value_ctx->constant_traction[d] = values[d];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NeumannPrepareValueModel(MPI_Comm comm, const AssemblyNeumannRule *rule, PetscInt dim, const SsrNeumannValueOps **ops, SsrNeumannValueCtx value_ctx)
{
  PetscFunctionBeginUser;
  PetscCheck(ops, comm, PETSC_ERR_ARG_NULL, "NeumannPrepareValueModel requires operations output");
  PetscCheck(rule, comm, PETSC_ERR_ARG_NULL, "NeumannPrepareValueModel requires a rule");
  PetscCheck(value_ctx, comm, PETSC_ERR_ARG_NULL, "NeumannPrepareValueModel requires a value context");
  *ops = NULL;
  PetscCall(SsrNeumannValueRegistryFind(rule->value_model_name, ops));
  PetscCheck((*ops)->evaluate, comm, PETSC_ERR_SUP,
             "Mechanics Neumann target %s uses value model %s; native affine face quadrature has no evaluator for that model",
             rule->support_name, rule->value_model_name);
  PetscCall(NeumannParseConstantTraction(comm, rule, dim, value_ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NeumannPointCoordinates(DM dm, PetscInt point, PetscInt dim, PetscReal x[3])
{
  DM           cdm = NULL;
  Vec          coords = NULL;
  PetscSection cs = NULL;
  PetscScalar *vals = NULL;
  PetscInt     n = 0;

  PetscFunctionBeginUser;
  for (PetscInt d = 0; d < 3; ++d) x[d] = 0.0;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCall(DMGetLocalSection(cdm, &cs));
  PetscCall(DMPlexVecGetClosure(cdm, cs, coords, point, &n, &vals));
  PetscCheck(n == dim, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE,
             "Expected coordinate dof %" PetscInt_FMT " on boundary vertex %" PetscInt_FMT ", got %" PetscInt_FMT, dim, point, n);
  for (PetscInt d = 0; d < dim; ++d) x[d] = PetscRealPart(vals[d]);
  PetscCall(DMPlexVecRestoreClosure(cdm, cs, coords, point, &n, &vals));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NeumannFaceVertices(DM dm, PetscInt face, PetscInt dim, PetscReal vertices[3][3], PetscInt *nverts)
{
  PetscInt vStart, vEnd, nclosure = 0, *closure = NULL;

  PetscFunctionBeginUser;
  *nverts = 0;
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMPlexGetTransitiveClosure(dm, face, PETSC_TRUE, &nclosure, &closure));
  for (PetscInt i = 0; i < nclosure; ++i) {
    const PetscInt point = closure[2 * i];

    if (point < vStart || point >= vEnd) continue;
    PetscCheck(*nverts < 3, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE,
               "Boundary face %" PetscInt_FMT " has more than three vertices", face);
    PetscCall(NeumannPointCoordinates(dm, point, dim, vertices[*nverts]));
    ++(*nverts);
  }
  PetscCall(DMPlexRestoreTransitiveClosure(dm, face, PETSC_TRUE, &nclosure, &closure));
  PetscCheck(*nverts == dim, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE,
             "Boundary point %" PetscInt_FMT " has %" PetscInt_FMT " vertices; expected %" PetscInt_FMT " for %" PetscInt_FMT "D mechanics Neumann assembly",
             face, *nverts, dim, dim);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscReal NeumannFaceMeasure(PetscInt dim, const PetscReal vertices[3][3])
{
  if (dim == 2) {
    const PetscReal dx = vertices[1][0] - vertices[0][0];
    const PetscReal dy = vertices[1][1] - vertices[0][1];

    return PetscSqrtReal(dx * dx + dy * dy);
  } else {
    PetscReal a[3], b[3], cross[3];

    for (PetscInt d = 0; d < 3; ++d) {
      a[d] = vertices[1][d] - vertices[0][d];
      b[d] = vertices[2][d] - vertices[0][d];
    }
    cross[0] = a[1] * b[2] - a[2] * b[1];
    cross[1] = a[2] * b[0] - a[0] * b[2];
    cross[2] = a[0] * b[1] - a[1] * b[0];
    return 0.5 * PetscSqrtReal(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
  }
}

static PetscInt NeumannFaceQuadratureCount(PetscInt dim)
{
  return dim == 2 ? 3 : 7;
}

static void NeumannFaceQuadrature(PetscInt dim, PetscInt q, PetscReal xi[2], PetscReal *weight)
{
  static const PetscReal edge_x[3] = {0.1127016653792583, 0.5, 0.8872983346207417};
  static const PetscReal edge_w[3] = {5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0};
  static const PetscReal tri_x[7][2] = {
    {0.1012865073235, 0.1012865073235},
    {0.7974269853531, 0.1012865073235},
    {0.1012865073235, 0.7974269853531},
    {0.4701420641051, 0.0597158717898},
    {0.4701420641051, 0.4701420641051},
    {0.0597158717898, 0.4701420641051},
    {1.0 / 3.0, 1.0 / 3.0},
  };
  static const PetscReal tri_w[7] = {
    0.1259391805448 / 2.0,
    0.1259391805448 / 2.0,
    0.1259391805448 / 2.0,
    0.1323941527885 / 2.0,
    0.1323941527885 / 2.0,
    0.1323941527885 / 2.0,
    0.2250000000000 / 2.0,
  };

  if (dim == 2) {
    xi[0] = edge_x[q];
    xi[1] = 0.0;
    *weight = edge_w[q];
  } else {
    xi[0] = tri_x[q][0];
    xi[1] = tri_x[q][1];
    *weight = tri_w[q];
  }
}

static void NeumannFacePhysicalPoint(PetscInt dim, const PetscReal vertices[3][3], const PetscReal xi[2], PetscReal x[3])
{
  for (PetscInt d = 0; d < 3; ++d) x[d] = 0.0;
  if (dim == 2) {
    for (PetscInt d = 0; d < dim; ++d) x[d] = (1.0 - xi[0]) * vertices[0][d] + xi[0] * vertices[1][d];
  } else {
    const PetscReal l0 = 1.0 - xi[0] - xi[1];

    for (PetscInt d = 0; d < dim; ++d) x[d] = l0 * vertices[0][d] + xi[0] * vertices[1][d] + xi[1] * vertices[2][d];
  }
}

static void NeumannPhysicalToReference(const P4Basis *basis, const PetscReal v0[3], const PetscReal invJ[9], const PetscReal x[3], PetscReal ref[3])
{
  for (PetscInt r = 0; r < basis->dim; ++r) {
    PetscReal ref_plus_one = 0.0;

    for (PetscInt d = 0; d < basis->dim; ++d) ref_plus_one += invJ[r * basis->dim + d] * (x[d] - v0[d]);
    ref[r] = ref_plus_one - 1.0;
  }
}

static void NeumannReferenceBarycentric(PetscInt dim, const PetscReal ref[3], PetscReal lambda[4])
{
  PetscReal sum = 0.0;

  for (PetscInt d = 0; d < dim; ++d) {
    lambda[d + 1] = 0.5 * (ref[d] + 1.0);
    sum += lambda[d + 1];
  }
  lambda[0] = 1.0 - sum;
  for (PetscInt d = dim + 1; d < 4; ++d) lambda[d] = 0.0;
}

static PetscErrorCode NeumannBuildBasisAlphas(P4Basis *basis, PetscInt **alphas_out)
{
  PetscDualSpace dual = NULL;
  PetscInt      *alphas = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1((basis->dim + 1) * basis->n_basis, &alphas));
  PetscCall(PetscFEGetDualSpace(basis->fe_scalar, &dual));
  for (PetscInt b = 0; b < basis->n_basis; ++b) {
    PetscQuadrature  q = NULL;
    PetscInt         qdim, Nc, npoints, sum = 0;
    const PetscReal *p = NULL;
    PetscReal        lambda[4];

    PetscCall(PetscDualSpaceGetFunctional(dual, b, &q));
    PetscCall(PetscQuadratureGetData(q, &qdim, &Nc, &npoints, &p, NULL));
    PetscCheck(qdim == basis->dim && Nc == 1 && npoints >= 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected scalar dual functional shape");
    NeumannReferenceBarycentric(basis->dim, p, lambda);
    for (PetscInt d = 0; d <= basis->dim; ++d) {
      const PetscReal scaled = (PetscReal)basis->degree * lambda[d];
      PetscInt        alpha  = (PetscInt)PetscFloorReal(scaled + 0.5);

      PetscCheck(PetscAbsReal(scaled - (PetscReal)alpha) <= 100.0 * PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_PLIB,
                 "Could not identify equispaced Lagrange node %" PetscInt_FMT " for Neumann face basis", b);
      alphas[(basis->dim + 1) * b + d] = alpha;
      sum += alpha;
    }
    PetscCheck(sum == basis->degree, PETSC_COMM_SELF, PETSC_ERR_PLIB,
               "Invalid Neumann basis multi-index sum %" PetscInt_FMT " for basis %" PetscInt_FMT ", expected degree %" PetscInt_FMT,
               sum, b, basis->degree);
  }
  *alphas_out = alphas;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscReal NeumannSimplexLagrange(PetscInt dim, PetscInt degree, const PetscInt alpha[], const PetscReal lambda[])
{
  PetscReal value = 1.0;

  for (PetscInt d = 0; d <= dim; ++d) {
    for (PetscInt m = 0; m < alpha[d]; ++m) value *= (((PetscReal)degree) * lambda[d] - (PetscReal)m) / (PetscReal)(alpha[d] - m);
  }
  return value;
}

static void NeumannEvaluateBasis(P4Basis *basis, const PetscInt alphas[], const PetscReal ref[3], PetscReal phi[])
{
  PetscReal lambda[4];

  NeumannReferenceBarycentric(basis->dim, ref, lambda);
  for (PetscInt b = 0; b < basis->n_basis; ++b) {
    phi[b] = NeumannSimplexLagrange(basis->dim, basis->degree, &alphas[(basis->dim + 1) * b], lambda);
  }
}

static PetscErrorCode NeumannCopyField(MPI_Comm comm, const char path[], const char support_name[], const char field_name[], const char text[], char dest[], size_t dest_size)
{
  const size_t len = text ? strlen(text) : 0;

  PetscFunctionBeginUser;
  PetscCheck(dest && dest_size > 0, comm, PETSC_ERR_ARG_NULL, "NeumannCopyField requires a destination buffer");
  PetscCheck(text, comm, PETSC_ERR_ARG_NULL, "Mechanics Neumann field %s for target %s in %s is NULL", field_name, support_name, path);
  PetscCheck(len < dest_size, comm, PETSC_ERR_ARG_SIZ,
             "Mechanics Neumann field %s for target %s in %s is too long (%lu >= %lu)", field_name, support_name, path,
             (unsigned long)len, (unsigned long)dest_size);
  PetscCall(PetscStrncpy(dest, text, dest_size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AssemblyCtxClearNeumannRules(AssemblyCtx *ctx)
{
  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Assembly context is NULL");
  PetscCall(PetscFree(ctx->neumann_rules));
  ctx->neumann_rule_count = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AssemblyCtxAppendNeumannRule(AssemblyCtx *ctx, PetscInt *capacity, const char path[], const char support_kind[], const char support_name[],
                                                  const char dm_label[], PetscInt tag, const char kind[], const char geometry[], PetscInt geometry_order,
                                                  const char value_model[], const char value_model_name[], const char native_status[], PetscInt matched_points)
{
  MPI_Comm             comm;
  AssemblyNeumannRule *rule = NULL;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Assembly context is NULL");
  PetscCheck(capacity, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "AssemblyCtxAppendNeumannRule requires a capacity pointer");
  comm = PetscObjectComm((PetscObject)ctx->dm);
  if (ctx->neumann_rule_count >= *capacity) {
    *capacity = *capacity ? 2 * (*capacity) : 4;
    PetscCall(PetscRealloc(sizeof(*ctx->neumann_rules) * (*capacity), &ctx->neumann_rules));
  }
  rule = &ctx->neumann_rules[ctx->neumann_rule_count++];
  PetscCall(PetscMemzero(rule, sizeof(*rule)));
  PetscCall(NeumannCopyField(comm, path, support_name, "support_kind", support_kind, rule->support_kind, sizeof(rule->support_kind)));
  PetscCall(NeumannCopyField(comm, path, support_name, "support_name", support_name, rule->support_name, sizeof(rule->support_name)));
  PetscCall(NeumannCopyField(comm, path, support_name, "dm_label", dm_label, rule->dm_label, sizeof(rule->dm_label)));
  PetscCall(NeumannCopyField(comm, path, support_name, "kind", kind, rule->kind, sizeof(rule->kind)));
  PetscCall(NeumannCopyField(comm, path, support_name, "geometry", geometry, rule->geometry, sizeof(rule->geometry)));
  PetscCall(NeumannCopyField(comm, path, support_name, "value_model", value_model, rule->value_model, sizeof(rule->value_model)));
  PetscCall(NeumannCopyField(comm, path, support_name, "value_model_name", value_model_name, rule->value_model_name, sizeof(rule->value_model_name)));
  PetscCall(NeumannCopyField(comm, path, support_name, "native_status", native_status, rule->native_status, sizeof(rule->native_status)));
  rule->tag            = tag;
  rule->geometry_order = geometry_order;
  rule->matched_points = matched_points;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssemblyCtxLoadNeumannLabelsCSV(AssemblyCtx *ctx, const char path[], AssemblyNeumannLabelStats *stats)
{
  MPI_Comm comm;
  FILE    *fh = NULL;
  char     line[4096];
  PetscInt cap_rules = 0;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Assembly context is NULL");
  if (stats) PetscCall(PetscMemzero(stats, sizeof(*stats)));
  PetscCall(AssemblyCtxClearNeumannRules(ctx));
  if (!path || !path[0]) PetscFunctionReturn(PETSC_SUCCESS);
  comm = PetscObjectComm((PetscObject)ctx->dm);
  fh = fopen(path, "r");
  PetscCheck(fh, comm, PETSC_ERR_FILE_OPEN, "Cannot open mechanics Neumann label CSV %s", path);
  while (fgets(line, sizeof(line), fh)) {
    char    *fields[9];
    PetscInt nfields;
    char    *support_kind, *support_name, *dm_label, *tag_text, *kind, *geometry, *geometry_order_text, *value_model, *native_status;
    char     value_model_name[64] = "";
    long     tag;
    char    *end = NULL;
    DMLabel  label = NULL;
    IS       points = NULL;
    PetscInt npoints = 0, global_points = 0;
    PetscInt geometry_order = 0;

    if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
    PetscCall(NeumannSplitCsvFields(line, fields, 9, &nfields));
    PetscCheck(nfields == 9, comm, PETSC_ERR_ARG_WRONG, "Mechanics Neumann label row in %s has %" PetscInt_FMT " fields; expected exactly 9", path, nfields);
    support_kind = NeumannTrimField(fields[0]);
    if (!strcmp(support_kind, "support_kind")) continue;
    support_name = NeumannTrimField(fields[1]);
    dm_label = NeumannTrimField(fields[2]);
    tag_text = NeumannTrimField(fields[3]);
    kind = NeumannTrimField(fields[4]);
    geometry = NeumannTrimField(fields[5]);
    geometry_order_text = NeumannTrimField(fields[6]);
    value_model = NeumannTrimField(fields[7]);
    native_status = NeumannTrimField(fields[8]);
    tag = strtol(tag_text, &end, 10);
    PetscCheck(end && end != tag_text && !NeumannTrimField(end)[0], comm, PETSC_ERR_ARG_WRONG, "Invalid mechanics Neumann tag %s in %s", tag_text, path);
    PetscCheck(!strcmp(support_kind, "boundary"), comm, PETSC_ERR_SUP, "Mechanics Neumann target %s must use boundary support, got %s", support_name, support_kind);
    PetscCheck(kind[0], comm, PETSC_ERR_ARG_WRONG, "Mechanics Neumann target %s in %s has empty kind", support_name, path);
    PetscCall(NeumannParseGeometryOrder(comm, path, support_name, geometry, geometry_order_text, &geometry_order));
    PetscCall(NeumannValidateNativeStatus(comm, path, support_name, native_status, geometry_order));
    PetscCall(DMGetLabel(ctx->dm, dm_label, &label));
    PetscCheck(label, comm, PETSC_ERR_ARG_WRONGSTATE, "Mesh has no DMPlex label %s for mechanics Neumann target %s", dm_label, support_name);
    PetscCall(NeumannValidateValueModel(path, support_name, kind, value_model, value_model_name, sizeof(value_model_name)));
    PetscCall(DMLabelGetStratumIS(label, (PetscInt)tag, &points));
    if (points) PetscCall(ISGetLocalSize(points, &npoints));
    PetscCallMPI(MPI_Allreduce(&npoints, &global_points, 1, MPIU_INT, MPI_SUM, comm));
    PetscCheck(global_points > 0, comm, PETSC_ERR_ARG_WRONGSTATE, "DMPlex label %s has no stratum tag %ld for mechanics Neumann target %s", dm_label, tag, support_name);
    PetscCall(ISDestroy(&points));
    PetscCall(AssemblyCtxAppendNeumannRule(ctx, &cap_rules, path, support_kind, support_name, dm_label, (PetscInt)tag, kind, geometry, geometry_order, value_model,
                                           value_model_name, native_status, global_points));
    if (stats) {
      stats->rows++;
      if (geometry_order > 0) stats->curved_rows++;
      else stats->affine_rows++;
      stats->matched_points += global_points;
      stats->last_geometry_order = geometry_order;
      PetscCall(PetscStrncpy(stats->last_kind, kind, sizeof(stats->last_kind)));
      PetscCall(PetscStrncpy(stats->last_geometry, geometry, sizeof(stats->last_geometry)));
      PetscCall(PetscStrncpy(stats->last_value_model, value_model_name, sizeof(stats->last_value_model)));
      PetscCall(PetscStrncpy(stats->last_native_status, native_status, sizeof(stats->last_native_status)));
    }
  }
  fclose(fh);
  if (stats && stats->rows > 0) {
    PetscCall(PetscPrintf(comm,
                          "MECHANICS_NEUMANN_LABELS_CONFIG enabled=true path=%s rows=%" PetscInt_FMT " matched_points=%" PetscInt_FMT " last_kind=%s last_geometry=%s last_geometry_order=%" PetscInt_FMT " last_value_model=%s status=label_table_validated native_status=%s staged_rules=%" PetscInt_FMT "\n",
                          path, stats->rows, stats->matched_points, stats->last_kind, stats->last_geometry[0] ? stats->last_geometry : "none", stats->last_geometry_order,
                          stats->last_value_model, stats->last_native_status, ctx->neumann_rule_count));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssemblyCtxAssembleNeumannResidual(AssemblyCtx *ctx, Vec rhs)
{
  MPI_Comm     comm;
  DM           dm;
  PetscSection lsec = NULL;
  Vec          rhs_loc = NULL;
  PetscScalar *elem_vec = NULL;
  PetscReal   *phi = NULL;
  PetscInt    *alphas = NULL;
  PetscInt     local_faces = 0, global_faces = 0;
  PetscInt     local_qpoints = 0, global_qpoints = 0;
  PetscLogDouble elapsed = 0.0;
  SsrProfileTimer profile_timer;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Assembly context is NULL");
  if (ctx->neumann_rule_count == 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(rhs, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "AssemblyCtxAssembleNeumannResidual requires an RHS vector");
  dm   = ctx->dm;
  comm = PetscObjectComm((PetscObject)dm);
  PetscCall(DMGetLocalSection(dm, &lsec));
  PetscCall(DMGetLocalVector(dm, &rhs_loc));
  PetscCall(VecZeroEntries(rhs_loc));
  PetscCall(PetscCalloc1(ctx->cell_dofs, &elem_vec));
  PetscCall(PetscMalloc1(ctx->basis->n_basis, &phi));
  PetscCall(NeumannBuildBasisAlphas(ctx->basis, &alphas));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ASSEMBLE_NEUMANN, dm, rhs, &profile_timer);
  for (PetscInt r = 0; r < ctx->neumann_rule_count; ++r) {
    const AssemblyNeumannRule *rule = &ctx->neumann_rules[r];
    DMLabel                    label = NULL;
    IS                         faces = NULL;
    const PetscInt            *face_idx = NULL;
    PetscInt                   nfaces = 0;
    struct _p_SsrNeumannValueCtx value_ctx;
    const SsrNeumannValueOps    *value_ops = NULL;

    PetscCheck(rule->geometry_order <= 0, comm, PETSC_ERR_SUP,
               "Mechanics Neumann target %s declares curved geometry %s(order=%" PetscInt_FMT
               "); native curved face quadrature is not implemented yet",
               rule->support_name, rule->geometry[0] ? rule->geometry : "none", rule->geometry_order);
    PetscCall(NeumannPrepareValueModel(comm, rule, ctx->basis->dim, &value_ops, (SsrNeumannValueCtx)&value_ctx));
    PetscCall(DMGetLabel(dm, rule->dm_label, &label));
    PetscCheck(label, comm, PETSC_ERR_ARG_WRONGSTATE, "Mesh has no DMPlex label %s for mechanics Neumann target %s", rule->dm_label, rule->support_name);
    PetscCall(DMLabelGetStratumIS(label, rule->tag, &faces));
    if (!faces) continue;
    PetscCall(ISGetLocalSize(faces, &nfaces));
    PetscCall(ISGetIndices(faces, &face_idx));
    for (PetscInt i = 0; i < nfaces; ++i) {
      const PetscInt  face = face_idx[i];
      const PetscInt *support = NULL;
      PetscInt        support_size = 0, cell;
      PetscReal       vertices[3][3], measure, v0[3], J[9], invJ[9], detJ, scale;
      PetscInt        nverts = 0, nq = NeumannFaceQuadratureCount(ctx->basis->dim);

      PetscCall(DMPlexGetSupportSize(dm, face, &support_size));
      PetscCheck(support_size == 1, comm, PETSC_ERR_ARG_WRONGSTATE,
                 "Mechanics Neumann target %s face %" PetscInt_FMT " has support size %" PetscInt_FMT "; expected one boundary cell",
                 rule->support_name, face, support_size);
      PetscCall(DMPlexGetSupport(dm, face, &support));
      cell = support[0];
      PetscCall(NeumannFaceVertices(dm, face, ctx->basis->dim, vertices, &nverts));
      measure = NeumannFaceMeasure(ctx->basis->dim, (const PetscReal (*)[3])vertices);
      PetscCheck(measure > PETSC_SMALL, comm, PETSC_ERR_ARG_WRONGSTATE,
                 "Mechanics Neumann target %s face %" PetscInt_FMT " has near-zero measure %.8e",
                 rule->support_name, face, (double)measure);
      PetscCall(DMPlexComputeCellGeometryFEM(dm, cell, NULL, v0, J, invJ, &detJ));
      PetscCall(PetscArrayzero(elem_vec, ctx->cell_dofs));
      scale = ctx->basis->dim == 2 ? measure : 2.0 * measure;
      for (PetscInt q = 0; q < nq; ++q) {
        PetscReal              xi[2], x[3], ref[3] = {0.0, 0.0, 0.0}, normal[3] = {0.0, 0.0, 0.0}, weight;
        PetscReal              traction[3] = {0.0, 0.0, 0.0};
        SsrNeumannValueInput   input;
        SsrNeumannValueResult  result;

        NeumannFaceQuadrature(ctx->basis->dim, q, xi, &weight);
        NeumannFacePhysicalPoint(ctx->basis->dim, (const PetscReal (*)[3])vertices, xi, x);
        input.dim    = ctx->basis->dim;
        input.time   = 0.0;
        input.point  = x;
        input.normal = normal;
        result.traction = traction;
        PetscCall(value_ops->evaluate((SsrNeumannValueCtx)&value_ctx, &input, &result));
        NeumannPhysicalToReference(ctx->basis, v0, invJ, x, ref);
        NeumannEvaluateBasis(ctx->basis, alphas, ref, phi);
        for (PetscInt a = 0; a < ctx->basis->n_basis; ++a) {
          for (PetscInt c = 0; c < ctx->basis->components; ++c) elem_vec[ctx->basis->components * a + c] += scale * weight * phi[a] * traction[c];
        }
        local_qpoints++;
      }
      PetscCall(DMPlexVecSetClosure(dm, lsec, rhs_loc, cell, elem_vec, ADD_VALUES));
      local_faces++;
    }
    PetscCall(ISRestoreIndices(faces, &face_idx));
    PetscCall(ISDestroy(&faces));
  }
  PetscCall(DMLocalToGlobalBegin(dm, rhs_loc, ADD_VALUES, rhs));
  PetscCall(DMLocalToGlobalEnd(dm, rhs_loc, ADD_VALUES, rhs));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  PetscCallMPI(MPI_Allreduce(&local_faces, &global_faces, 1, MPIU_INT, MPI_SUM, comm));
  PetscCallMPI(MPI_Allreduce(&local_qpoints, &global_qpoints, 1, MPIU_INT, MPI_SUM, comm));
  PetscCall(SsrStatsAddNeumannAssembly(&ctx->neumann_stats, ctx->neumann_rule_count, global_faces, global_qpoints, elapsed));
  PetscCall(PetscPrintf(comm,
                        "MECHANICS_NEUMANN_ASSEMBLY enabled=true rules=%" PetscInt_FMT " faces=%" PetscInt_FMT " quadrature_points=%" PetscInt_FMT " assembly_time=%.6g status=native_face_quadrature_affine\n",
                        ctx->neumann_rule_count, global_faces, global_qpoints, (double)elapsed));
  PetscCall(PetscFree(alphas));
  PetscCall(PetscFree(phi));
  PetscCall(PetscFree(elem_vec));
  PetscCall(DMRestoreLocalVector(dm, &rhs_loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode AssemblyCtxValidateNeumannLabelsCSV(AssemblyCtx *ctx, const char path[], PetscInt expected_rows)
{
  MPI_Comm                  comm;
  AssemblyNeumannLabelStats stats;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Assembly context is NULL");
  comm = PetscObjectComm((PetscObject)ctx->dm);
  if (!path || !path[0]) {
    PetscCheck(expected_rows <= 0, comm, PETSC_ERR_ARG_WRONG,
               "Native problem manifest declares mechanics Neumann rows but no mechanics Neumann label table is available");
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(AssemblyCtxLoadNeumannLabelsCSV(ctx, path, &stats));
  if (expected_rows >= 0) {
    PetscCheck(stats.rows == expected_rows, comm, PETSC_ERR_ARG_WRONG,
               "Native problem manifest declares %" PetscInt_FMT " mechanics Neumann row(s), but label table %s contains %" PetscInt_FMT,
               expected_rows, path, stats.rows);
  }
  if (stats.curved_rows > 0) {
    SETERRQ(comm, PETSC_ERR_SUP,
            "Mechanics Neumann label table %s contains %" PetscInt_FMT
            " curved rule(s); native curved face quadrature is not implemented yet and Neumann loads will not be approximated",
            path, stats.curved_rows);
  }
  if (stats.affine_rows > 0) {
    PetscCall(PetscPrintf(comm,
                          "MECHANICS_NEUMANN_LABELS_READY path=%s affine_rows=%" PetscInt_FMT " curved_rows=%" PetscInt_FMT " status=native_face_quadrature_affine\n",
                          path, stats.affine_rows, stats.curved_rows));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
