#include "petsc_ssr_problem.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* Manifest kind literal: "petsc_ssr_native_problem_manifest"; see petsc_ssr_problem.h. */

static PetscErrorCode NativeManifestReadFile(const char path[], char **text)
{
  MPI_Comm comm = PETSC_COMM_WORLD;
  FILE    *fh = NULL;
  long     size;
  size_t   nread;

  PetscFunctionBeginUser;
  *text = NULL;
  if (!path || !path[0]) PetscFunctionReturn(PETSC_SUCCESS);
  fh = fopen(path, "rb");
  PetscCheck(fh, comm, PETSC_ERR_FILE_OPEN, "Cannot open native problem manifest %s", path);
  PetscCheck(fseek(fh, 0, SEEK_END) == 0, comm, PETSC_ERR_FILE_READ, "Cannot seek native problem manifest %s", path);
  size = ftell(fh);
  PetscCheck(size >= 0, comm, PETSC_ERR_FILE_READ, "Cannot size native problem manifest %s", path);
  PetscCheck(fseek(fh, 0, SEEK_SET) == 0, comm, PETSC_ERR_FILE_READ, "Cannot rewind native problem manifest %s", path);
  PetscCall(PetscMalloc1((size_t)size + 1, text));
  nread = fread(*text, 1, (size_t)size, fh);
  PetscCheck(nread == (size_t)size, comm, PETSC_ERR_FILE_READ, "Could not read full native problem manifest %s", path);
  (*text)[size] = '\0';
  fclose(fh);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void NativeManifestSkipWhitespace(const char **p)
{
  while (**p && (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r')) ++(*p);
}

static PetscBool NativeManifestValueIsNull(const char p[])
{
  return (PetscBool)(strncmp(p, "null", 4) == 0 && (p[4] == ',' || p[4] == '}' || p[4] == ']' || p[4] == ' ' || p[4] == '\t' || p[4] == '\n' || p[4] == '\r' || p[4] == '\0'));
}

static PetscErrorCode NativeManifestFindKeyValueBounded(const char begin[], const char end[], const char key[], const char **value, PetscBool *found)
{
  char        needle[128];
  const char *p          = NULL;
  size_t      needle_len = 0;

  PetscFunctionBeginUser;
  *found = PETSC_FALSE;
  *value = NULL;
  PetscCheck(strlen(key) + 3 < sizeof(needle), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Native manifest key %s is too long", key);
  snprintf(needle, sizeof(needle), "\"%s\"", key);
  needle_len = strlen(needle);
  for (p = begin; p + needle_len <= end; ++p) {
    const char *q = NULL;

    if (strncmp(p, needle, needle_len) != 0) continue;
    q = p + needle_len;
    NativeManifestSkipWhitespace(&q);
    PetscCheck(q < end && *q == ':', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest key %s: missing ':'", key);
    ++q;
    NativeManifestSkipWhitespace(&q);
    PetscCheck(q < end, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest key %s: missing value", key);
    *value = q;
    *found = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeManifestFindKeyValue(const char json[], const char key[], const char **value, PetscBool *found)
{
  PetscFunctionBeginUser;
  PetscCall(NativeManifestFindKeyValueBounded(json, json + strlen(json), key, value, found));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeManifestValueEnd(const char value[], const char **end);

static PetscErrorCode NativeManifestFindMemberValueBounded(const char begin[], const char end[], const char key[], const char **value, PetscBool *found)
{
  const char *p       = begin;
  size_t      key_len = strlen(key);

  PetscFunctionBeginUser;
  *found = PETSC_FALSE;
  *value = NULL;
  NativeManifestSkipWhitespace(&p);
  PetscCheck(p < end && *p == '{', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Expected native manifest object while looking for member %s", key);
  ++p;
  NativeManifestSkipWhitespace(&p);
  if (p < end && *p == '}') PetscFunctionReturn(PETSC_SUCCESS);
  while (p < end && *p) {
    const char *member_begin = NULL;
    const char *member_end   = NULL;
    const char *value_end    = NULL;
    PetscBool   same         = PETSC_FALSE;

    PetscCheck(*p == '"', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest object key");
    member_begin = ++p;
    while (p < end && *p) {
      if (*p == '\\') {
        ++p;
        PetscCheck(p < end && *p, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest object key escape");
      } else if (*p == '"') {
        member_end = p;
        break;
      }
      ++p;
    }
    PetscCheck(member_end, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest object key");
    same = (PetscBool)((size_t)(member_end - member_begin) == key_len && strncmp(member_begin, key, key_len) == 0);
    ++p;
    NativeManifestSkipWhitespace(&p);
    PetscCheck(p < end && *p == ':', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest object delimiter");
    ++p;
    NativeManifestSkipWhitespace(&p);
    PetscCheck(p < end, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest key %s: missing value", key);
    if (same) {
      *value = p;
      *found = PETSC_TRUE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscCall(NativeManifestValueEnd(p, &value_end));
    p = value_end;
    NativeManifestSkipWhitespace(&p);
    if (p < end && *p == ',') {
      ++p;
      NativeManifestSkipWhitespace(&p);
      continue;
    }
    PetscCheck(p < end && *p == '}', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest object delimiter");
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest object");
}

static PetscErrorCode NativeManifestValueEnd(const char value[], const char **end)
{
  const char *p = value;

  PetscFunctionBeginUser;
  NativeManifestSkipWhitespace(&p);
  PetscCheck(*p, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest JSON value");
  if (*p == '{' || *p == '[') {
    PetscInt  object_depth = 0;
    PetscInt  array_depth  = 0;
    PetscBool in_string    = PETSC_FALSE;
    PetscBool escape       = PETSC_FALSE;

    for (; *p; ++p) {
      char c = *p;

      if (in_string) {
        if (escape) escape = PETSC_FALSE;
        else if (c == '\\') escape = PETSC_TRUE;
        else if (c == '"') in_string = PETSC_FALSE;
        continue;
      }
      if (c == '"') in_string = PETSC_TRUE;
      else if (c == '{') ++object_depth;
      else if (c == '[') ++array_depth;
      else if (c == '}') {
        --object_depth;
        PetscCheck(object_depth >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest object nesting");
      } else if (c == ']') {
        --array_depth;
        PetscCheck(array_depth >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest array nesting");
      }
      if (object_depth == 0 && array_depth == 0) {
        *end = p + 1;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    }
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest JSON: unterminated object or array");
  }
  if (*p == '"') {
    ++p;
    while (*p) {
      if (*p == '\\') {
        ++p;
        PetscCheck(*p, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest string escape");
      } else if (*p == '"') {
        *end = p + 1;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      ++p;
    }
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest JSON: unterminated string");
  }
  while (*p && *p != ',' && *p != '}' && *p != ']') ++p;
  *end = p;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeManifestFindStringBounded(const char begin[], const char end[], const char key[], char value[], size_t value_len, PetscBool *found)
{
  const char *p         = NULL;
  PetscBool   key_found = PETSC_FALSE;
  size_t      n         = 0;

  PetscFunctionBeginUser;
  *found = PETSC_FALSE;
  if (value_len > 0) value[0] = '\0';
  PetscCall(NativeManifestFindKeyValueBounded(begin, end, key, &p, &key_found));
  if (!key_found || NativeManifestValueIsNull(p)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(*p == '"', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest key %s: expected string value", key);
  ++p;
  while (*p && *p != '"') {
    char c = *p++;

    if (c == '\\') {
      PetscCheck(*p, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest key %s: truncated escape", key);
      c = *p++;
      if (c == 'n') c = '\n';
      else if (c == 'r') c = '\r';
      else if (c == 't') c = '\t';
    }
    PetscCheck(n + 1 < value_len, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Native manifest value for %s is too long", key);
    value[n++] = c;
  }
  PetscCheck(*p == '"', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest key %s: unterminated string", key);
  value[n] = '\0';
  *found = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeManifestFindString(const char json[], const char key[], char value[], size_t value_len, PetscBool *found)
{
  PetscFunctionBeginUser;
  PetscCall(NativeManifestFindStringBounded(json, json + strlen(json), key, value, value_len, found));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeManifestFindIntBounded(const char begin[], const char bound[], const char key[], PetscInt *value, PetscBool *found)
{
  const char *p         = NULL;
  char       *parse_end = NULL;
  long        parsed;
  PetscBool   key_found = PETSC_FALSE;

  PetscFunctionBeginUser;
  *found = PETSC_FALSE;
  *value = 0;
  PetscCall(NativeManifestFindKeyValueBounded(begin, bound, key, &p, &key_found));
  if (!key_found || NativeManifestValueIsNull(p)) PetscFunctionReturn(PETSC_SUCCESS);
  parsed = strtol(p, &parse_end, 10);
  PetscCheck(parse_end != p, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest key %s: expected integer value", key);
  while (parse_end < bound && (*parse_end == ' ' || *parse_end == '\t' || *parse_end == '\n' || *parse_end == '\r')) ++parse_end;
  PetscCheck(parse_end < bound && (*parse_end == ',' || *parse_end == '}' || *parse_end == ']'), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
             "Malformed native manifest key %s: bad integer delimiter", key);
  *value = (PetscInt)parsed;
  *found = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeManifestFindInt(const char json[], const char key[], PetscInt *value, PetscBool *found)
{
  PetscFunctionBeginUser;
  PetscCall(NativeManifestFindIntBounded(json, json + strlen(json), key, value, found));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeManifestFindBoolBounded(const char begin[], const char end[], const char key[], PetscBool *value, PetscBool *found)
{
  const char *p         = NULL;
  PetscBool   key_found = PETSC_FALSE;

  PetscFunctionBeginUser;
  *found = PETSC_FALSE;
  *value = PETSC_FALSE;
  PetscCall(NativeManifestFindKeyValueBounded(begin, end, key, &p, &key_found));
  if (!key_found || NativeManifestValueIsNull(p)) PetscFunctionReturn(PETSC_SUCCESS);
  if (strncmp(p, "true", 4) == 0) {
    const char *q = p + 4;

    while (q < end && (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r')) ++q;
    PetscCheck(q < end && (*q == ',' || *q == '}' || *q == ']'), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Malformed native manifest key %s: bad boolean delimiter", key);
    *value = PETSC_TRUE;
    *found = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (strncmp(p, "false", 5) == 0) {
    const char *q = p + 5;

    while (q < end && (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r')) ++q;
    PetscCheck(q < end && (*q == ',' || *q == '}' || *q == ']'), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Malformed native manifest key %s: bad boolean delimiter", key);
    *value = PETSC_FALSE;
    *found = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest key %s: expected boolean value", key);
}

static PetscErrorCode NativeManifestFindBool(const char json[], const char key[], PetscBool *value, PetscBool *found)
{
  PetscFunctionBeginUser;
  PetscCall(NativeManifestFindBoolBounded(json, json + strlen(json), key, value, found));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeManifestFindObjectBounded(const char begin[], const char end[], const char key[], const char **object_begin, const char **object_end, PetscBool *found)
{
  const char *p         = NULL;
  const char *value_end = NULL;
  PetscBool   key_found = PETSC_FALSE;

  PetscFunctionBeginUser;
  *found        = PETSC_FALSE;
  *object_begin = NULL;
  *object_end   = NULL;
  PetscCall(NativeManifestFindKeyValueBounded(begin, end, key, &p, &key_found));
  if (!key_found || NativeManifestValueIsNull(p)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(*p == '{', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Native problem manifest key %s must be an object", key);
  PetscCall(NativeManifestValueEnd(p, &value_end));
  *object_begin = p;
  *object_end   = value_end;
  *found        = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeManifestFindObject(const char json[], const char key[], const char **object_begin, const char **object_end, PetscBool *found)
{
  PetscFunctionBeginUser;
  PetscCall(NativeManifestFindObjectBounded(json, json + strlen(json), key, object_begin, object_end, found));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeManifestFindObjectMemberBounded(const char begin[], const char end[], const char key[], const char **object_begin, const char **object_end, PetscBool *found)
{
  const char *p         = NULL;
  const char *value_end = NULL;
  PetscBool   key_found = PETSC_FALSE;

  PetscFunctionBeginUser;
  *found        = PETSC_FALSE;
  *object_begin = NULL;
  *object_end   = NULL;
  PetscCall(NativeManifestFindMemberValueBounded(begin, end, key, &p, &key_found));
  if (!key_found || NativeManifestValueIsNull(p)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(*p == '{', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Native problem manifest key %s must be an object", key);
  PetscCall(NativeManifestValueEnd(p, &value_end));
  *object_begin = p;
  *object_end   = value_end;
  *found        = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeManifestFindObjectMember(const char json[], const char key[], const char **object_begin, const char **object_end, PetscBool *found)
{
  PetscFunctionBeginUser;
  PetscCall(NativeManifestFindObjectMemberBounded(json, json + strlen(json), key, object_begin, object_end, found));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeManifestCountArrayEntries(const char value[], PetscInt *count)
{
  const char *p = value;

  PetscFunctionBeginUser;
  *count = 0;
  NativeManifestSkipWhitespace(&p);
  PetscCheck(*p == '[', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Expected native manifest array");
  ++p;
  NativeManifestSkipWhitespace(&p);
  if (*p == ']') PetscFunctionReturn(PETSC_SUCCESS);
  while (*p) {
    const char *value_end = NULL;

    PetscCall(NativeManifestValueEnd(p, &value_end));
    ++(*count);
    p = value_end;
    NativeManifestSkipWhitespace(&p);
    if (*p == ',') {
      ++p;
      NativeManifestSkipWhitespace(&p);
      continue;
    }
    PetscCheck(*p == ']', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest array delimiter");
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest array");
}

static PetscErrorCode NativeManifestCountObjectMembers(const char value[], PetscInt *count)
{
  const char *p = value;

  PetscFunctionBeginUser;
  *count = 0;
  NativeManifestSkipWhitespace(&p);
  PetscCheck(*p == '{', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Expected native manifest object");
  ++p;
  NativeManifestSkipWhitespace(&p);
  if (*p == '}') PetscFunctionReturn(PETSC_SUCCESS);
  while (*p) {
    const char *value_end = NULL;

    PetscCheck(*p == '"', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest object key");
    ++p;
    while (*p) {
      if (*p == '\\') {
        ++p;
        PetscCheck(*p, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest object key escape");
      } else if (*p == '"') {
        ++p;
        break;
      }
      ++p;
    }
    PetscCheck(*p, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest object key");
    NativeManifestSkipWhitespace(&p);
    PetscCheck(*p == ':', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest object delimiter");
    ++p;
    NativeManifestSkipWhitespace(&p);
    PetscCall(NativeManifestValueEnd(p, &value_end));
    ++(*count);
    p = value_end;
    NativeManifestSkipWhitespace(&p);
    if (*p == ',') {
      ++p;
      NativeManifestSkipWhitespace(&p);
      continue;
    }
    PetscCheck(*p == '}', PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest object delimiter");
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Malformed native manifest object");
}

static PetscErrorCode NativeManifestCountArrayInObject(const char begin[], const char end[], const char key[], PetscInt *count, PetscBool *found)
{
  const char *p         = NULL;
  PetscBool   key_found = PETSC_FALSE;

  PetscFunctionBeginUser;
  *count = 0;
  *found = PETSC_FALSE;
  PetscCall(NativeManifestFindKeyValueBounded(begin, end, key, &p, &key_found));
  if (!key_found || NativeManifestValueIsNull(p)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(NativeManifestCountArrayEntries(p, count));
  *found = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeManifestCountObjectInObject(const char begin[], const char end[], const char key[], PetscInt *count, PetscBool *found)
{
  const char *object_begin = NULL;
  const char *object_end   = NULL;

  PetscFunctionBeginUser;
  *count = 0;
  *found = PETSC_FALSE;
  PetscCall(NativeManifestFindObjectBounded(begin, end, key, &object_begin, &object_end, found));
  if (!*found) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(NativeManifestCountObjectMembers(object_begin, count));
  (void)object_end;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeManifestElementDegree(const char element[], PetscInt *degree, PetscBool *known)
{
  char *end = NULL;
  long  parsed;

  PetscFunctionBeginUser;
  *known = PETSC_FALSE;
  *degree = 0;
  if (!element || !element[0]) PetscFunctionReturn(PETSC_SUCCESS);
  if (element[0] == 'P' || element[0] == 'p') {
    parsed = strtol(element + 1, &end, 10);
    PetscCheck(end && end != element + 1 && !*end, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Native manifest element value %s is not a supported polynomial degree", element);
  } else {
    parsed = strtol(element, &end, 10);
    PetscCheck(end && end != element && !*end, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
               "Native manifest element value %s is not a supported polynomial degree", element);
  }
  *degree = (PetscInt)parsed;
  *known = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeProblemManifestValidateMetadata(AppCtx *app, const char json[])
{
  MPI_Comm    comm = PETSC_COMM_WORLD;
  char        text[PETSC_MAX_PATH_LEN];
  PetscBool   found = PETSC_FALSE;
  PetscBool   flg = PETSC_FALSE;
  PetscBool   known_degree = PETSC_FALSE;
  PetscInt    value = 0;
  PetscInt    degree = 0;
  PetscInt    resolved_world_size = 0;
  PetscMPIInt comm_size = 1;
  char        analysis[32] = "";
  char        element[32] = "";
  char        solver_profile[128] = "";

  PetscFunctionBeginUser;
  PetscCall(NativeManifestFindString(json, "kind", text, sizeof(text), &found));
  PetscCheck(found, comm, PETSC_ERR_ARG_WRONG, "Native problem manifest %s has no string kind", app->native_problem_manifest);
  PetscCall(PetscStrcmp(text, PETSC_SSR_NATIVE_PROBLEM_MANIFEST_KIND, &flg));
  PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "Native problem manifest %s has unsupported kind %s", app->native_problem_manifest, text);

  PetscCall(NativeManifestFindInt(json, "schema_version", &value, &found));
  PetscCheck(found, comm, PETSC_ERR_ARG_WRONG, "Native problem manifest %s has no integer schema_version", app->native_problem_manifest);
  PetscCheck(value == PETSC_SSR_NATIVE_PROBLEM_MANIFEST_SCHEMA_VERSION, comm, PETSC_ERR_ARG_WRONG, "Native problem manifest %s schema_version %" PetscInt_FMT " is not supported", app->native_problem_manifest, value);

  PetscCall(NativeManifestFindString(json, "analysis", analysis, sizeof(analysis), &found));
  if (found && analysis[0]) {
    PetscCall(PetscStrcasecmp(analysis, app->analysis, &flg));
    PetscCheck(flg, comm, PETSC_ERR_ARG_WRONG, "Native problem manifest analysis %s does not match option -analysis %s", analysis, app->analysis);
  }

  PetscCall(NativeManifestFindString(json, "element", element, sizeof(element), &found));
  if (found && element[0]) {
    PetscCall(NativeManifestElementDegree(element, &degree, &known_degree));
    PetscCheck(!known_degree || degree == app->element_degree, comm, PETSC_ERR_ARG_WRONG,
               "Native problem manifest element %s does not match option -element_degree %" PetscInt_FMT, element, app->element_degree);
  }

  PetscCall(NativeManifestFindInt(json, "resolved_world_size", &resolved_world_size, &found));
  if (found && resolved_world_size > 0) {
    PetscCallMPI(MPI_Comm_size(comm, &comm_size));
    PetscCheck(resolved_world_size == (PetscInt)comm_size, comm, PETSC_ERR_ARG_WRONG,
               "Native problem manifest resolved_world_size %" PetscInt_FMT " does not match MPI size %d", resolved_world_size, comm_size);
  }

  PetscCall(NativeManifestFindString(json, "solver_profile", solver_profile, sizeof(solver_profile), &found));
  PetscCall(PetscPrintf(comm, "NATIVE_PROBLEM_MANIFEST_METADATA schema_version=1 analysis=%s element=%s resolved_world_size=%" PetscInt_FMT " solver_profile=%s\n",
                        analysis[0] ? analysis : "unspecified", element[0] ? element : "unspecified", resolved_world_size,
                        solver_profile[0] ? solver_profile : "unspecified"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeProblemManifestValidateRequiredLabel(const char begin[], const char end[], const char key[], const char expected[], char value[], size_t value_len)
{
  PetscBool found = PETSC_FALSE;
  PetscBool same  = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(NativeManifestFindStringBounded(begin, end, key, value, value_len, &found));
  PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Native problem manifest dmplex has no string %s", key);
  PetscCall(PetscStrcmp(value, expected, &same));
  PetscCheck(same, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
             "Native problem manifest dmplex.%s=%s is not supported by this native startup; expected %s", key, value, expected);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeProblemManifestValidateDeclaredCount(const char begin[], const char end[], const char section[], const char key[], PetscInt actual)
{
  MPI_Comm  comm = PETSC_COMM_WORLD;
  PetscInt  expected = 0;
  PetscBool found = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(NativeManifestFindIntBounded(begin, end, key, &expected, &found));
  PetscCheck(found, comm, PETSC_ERR_ARG_WRONG, "Native problem manifest %s has no integer %s.%s", section, section, key);
  PetscCheck(expected == actual, comm, PETSC_ERR_ARG_WRONG,
             "Native problem manifest %s.%s expected %" PetscInt_FMT " entries but parsed %" PetscInt_FMT, section, key, expected, actual);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeProblemManifestValidateTopology(AppCtx *app, const char json[], SsrNativeProblemTopologyStats *stats)
{
  MPI_Comm    comm            = PETSC_COMM_WORLD;
  const char *dmplex_begin    = NULL;
  const char *dmplex_end      = NULL;
  const char *counts_begin    = NULL;
  const char *counts_end      = NULL;
  const char *supports_begin  = NULL;
  const char *supports_end    = NULL;
  const char *geometry_begin  = NULL;
  const char *geometry_end    = NULL;
  PetscBool   found           = PETSC_FALSE;
  PetscBool   count_found     = PETSC_FALSE;

  PetscFunctionBeginUser;
  memset(stats, 0, sizeof(*stats));
  PetscCall(NativeManifestFindObjectMember(json, "dmplex", &dmplex_begin, &dmplex_end, &found));
  PetscCheck(found, comm, PETSC_ERR_ARG_WRONG, "Native problem manifest %s has no object dmplex", app->native_problem_manifest);
  PetscCall(NativeProblemManifestValidateRequiredLabel(dmplex_begin, dmplex_end, "region_label", PETSC_SSR_DMPLEX_REGION_LABEL, stats->region_label, sizeof(stats->region_label)));
  PetscCall(NativeProblemManifestValidateRequiredLabel(dmplex_begin, dmplex_end, "boundary_label", PETSC_SSR_DMPLEX_BOUNDARY_LABEL, stats->boundary_label, sizeof(stats->boundary_label)));
  PetscCall(NativeProblemManifestValidateRequiredLabel(dmplex_begin, dmplex_end, "native_boundary_marker_label", PETSC_SSR_DMPLEX_NATIVE_BOUNDARY_MARKER_LABEL, stats->native_boundary_marker_label, sizeof(stats->native_boundary_marker_label)));

  PetscCall(NativeManifestFindObjectBounded(dmplex_begin, dmplex_end, "supports", &supports_begin, &supports_end, &found));
  PetscCheck(found, comm, PETSC_ERR_ARG_WRONG, "Native problem manifest %s has no dmplex.supports object", app->native_problem_manifest);
  PetscCall(NativeManifestCountObjectInObject(supports_begin, supports_end, "regions", &stats->regions, &count_found));
  PetscCheck(count_found, comm, PETSC_ERR_ARG_WRONG, "Native problem manifest %s has no dmplex.supports.regions object", app->native_problem_manifest);
  PetscCall(NativeManifestCountObjectInObject(supports_begin, supports_end, "boundaries", &stats->boundaries, &count_found));
  PetscCheck(count_found, comm, PETSC_ERR_ARG_WRONG, "Native problem manifest %s has no dmplex.supports.boundaries object", app->native_problem_manifest);
  PetscCall(NativeManifestCountObjectInObject(supports_begin, supports_end, "nodesets", &stats->nodesets, &count_found));
  PetscCheck(count_found, comm, PETSC_ERR_ARG_WRONG, "Native problem manifest %s has no dmplex.supports.nodesets object", app->native_problem_manifest);

  PetscCall(NativeManifestFindObjectMember(json, "boundary_geometry", &geometry_begin, &geometry_end, &found));
  if (found) PetscCall(NativeManifestCountObjectMembers(geometry_begin, &stats->boundary_geometry));
  (void)geometry_end;

  PetscCall(NativeManifestFindObjectBounded(dmplex_begin, dmplex_end, "support_counts", &counts_begin, &counts_end, &found));
  PetscCheck(found, comm, PETSC_ERR_ARG_WRONG, "Native problem manifest %s has no dmplex.support_counts object", app->native_problem_manifest);
  PetscCall(NativeProblemManifestValidateDeclaredCount(counts_begin, counts_end, "dmplex.support_counts", "regions", stats->regions));
  PetscCall(NativeProblemManifestValidateDeclaredCount(counts_begin, counts_end, "dmplex.support_counts", "boundaries", stats->boundaries));
  PetscCall(NativeProblemManifestValidateDeclaredCount(counts_begin, counts_end, "dmplex.support_counts", "nodesets", stats->nodesets));
  PetscCall(NativeProblemManifestValidateDeclaredCount(counts_begin, counts_end, "dmplex.support_counts", "boundary_geometry", stats->boundary_geometry));
  PetscCall(PetscPrintf(comm,
                        "NATIVE_PROBLEM_MANIFEST_LABELS region_label=\"%s\" boundary_label=\"%s\" native_boundary_marker_label=%s regions=%" PetscInt_FMT " boundaries=%" PetscInt_FMT " nodesets=%" PetscInt_FMT " boundary_geometry=%" PetscInt_FMT "\n",
                        stats->region_label, stats->boundary_label, stats->native_boundary_marker_label, stats->regions, stats->boundaries, stats->nodesets, stats->boundary_geometry));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeProblemManifestValidateBoundaryRules(AppCtx *app, const char json[], SsrNativeProblemRuleStats *stats)
{
  MPI_Comm    comm          = PETSC_COMM_WORLD;
  const char *section_begin = NULL;
  const char *section_end   = NULL;
  const char *counts_begin  = NULL;
  const char *counts_end    = NULL;
  PetscBool   found         = PETSC_FALSE;
  PetscBool   count_found   = PETSC_FALSE;

  PetscFunctionBeginUser;
  memset(stats, 0, sizeof(*stats));
  PetscCall(NativeManifestFindObjectMember(json, "mechanics", &section_begin, &section_end, &found));
  if (found) {
    PetscCall(NativeManifestCountArrayInObject(section_begin, section_end, "dirichlet", &stats->mechanics_dirichlet, &count_found));
    PetscCheck(count_found, comm, PETSC_ERR_ARG_WRONG, "Native problem manifest %s mechanics object has no dirichlet array", app->native_problem_manifest);
    PetscCall(NativeManifestCountArrayInObject(section_begin, section_end, "neumann", &stats->mechanics_neumann, &count_found));
    PetscCheck(count_found, comm, PETSC_ERR_ARG_WRONG, "Native problem manifest %s mechanics object has no neumann array", app->native_problem_manifest);
  }
  PetscCall(NativeManifestFindObjectMember(json, "seepage", &section_begin, &section_end, &found));
  if (found) {
    PetscCall(NativeManifestCountArrayInObject(section_begin, section_end, "head_bcs", &stats->seepage_head, &count_found));
    PetscCheck(count_found, comm, PETSC_ERR_ARG_WRONG, "Native problem manifest %s seepage object has no head_bcs array", app->native_problem_manifest);
    PetscCall(NativeManifestCountArrayInObject(section_begin, section_end, "flux_bcs", &stats->seepage_flux, &count_found));
    PetscCheck(count_found, comm, PETSC_ERR_ARG_WRONG, "Native problem manifest %s seepage object has no flux_bcs array", app->native_problem_manifest);
  }
  PetscCall(NativeManifestFindObjectMember(json, "rule_counts", &counts_begin, &counts_end, &found));
  PetscCheck(found, comm, PETSC_ERR_ARG_WRONG, "Native problem manifest %s has no rule_counts object", app->native_problem_manifest);
  PetscCall(NativeProblemManifestValidateDeclaredCount(counts_begin, counts_end, "rule_counts", "mechanics_dirichlet", stats->mechanics_dirichlet));
  PetscCall(NativeProblemManifestValidateDeclaredCount(counts_begin, counts_end, "rule_counts", "mechanics_neumann", stats->mechanics_neumann));
  PetscCall(NativeProblemManifestValidateDeclaredCount(counts_begin, counts_end, "rule_counts", "seepage_head", stats->seepage_head));
  PetscCall(NativeProblemManifestValidateDeclaredCount(counts_begin, counts_end, "rule_counts", "seepage_flux", stats->seepage_flux));
  PetscCall(PetscPrintf(comm,
                        "NATIVE_PROBLEM_MANIFEST_RULES mechanics_dirichlet=%" PetscInt_FMT " mechanics_neumann=%" PetscInt_FMT " seepage_head=%" PetscInt_FMT " seepage_flux=%" PetscInt_FMT "\n",
                        stats->mechanics_dirichlet, stats->mechanics_neumann, stats->seepage_head, stats->seepage_flux));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeProblemManifestApplyNativeInputPath(const char json[], const char native_inputs_begin[], const char native_inputs_end[], PetscBool native_inputs_found, const char key[], char dest[], size_t dest_len)
{
  MPI_Comm  comm  = PETSC_COMM_WORLD;
  char      path[PETSC_MAX_PATH_LEN];
  PetscBool found = PETSC_FALSE;

  PetscFunctionBeginUser;
  if (native_inputs_found) PetscCall(NativeManifestFindStringBounded(native_inputs_begin, native_inputs_end, key, path, sizeof(path), &found));
  else PetscCall(NativeManifestFindString(json, key, path, sizeof(path), &found));
  if (found) {
    if (dest[0]) {
      PetscBool same = PETSC_FALSE;

      PetscCall(PetscStrcmp(dest, path, &same));
      PetscCheck(same, comm, PETSC_ERR_ARG_WRONG,
                 "Native problem manifest native_inputs.%s=%s conflicts with already configured value %s", key, path, dest);
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscCall(PetscStrncpy(dest, path, dest_len));
    PetscCall(PetscPrintf(comm, "NATIVE_PROBLEM_MANIFEST_APPLY key=%s path=%s\n", key, dest));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeProblemManifestApplyNativeInputBool(const char json[], const char native_inputs_begin[], const char native_inputs_end[], PetscBool native_inputs_found, const char key[], PetscBool *dest)
{
  MPI_Comm  comm  = PETSC_COMM_WORLD;
  PetscBool value = PETSC_FALSE;
  PetscBool found = PETSC_FALSE;

  PetscFunctionBeginUser;
  if (native_inputs_found) PetscCall(NativeManifestFindBoolBounded(native_inputs_begin, native_inputs_end, key, &value, &found));
  else PetscCall(NativeManifestFindBool(json, key, &value, &found));
  if (found) {
    PetscCheck(!*dest || value, comm, PETSC_ERR_ARG_WRONG,
               "Native problem manifest native_inputs.%s=false conflicts with already configured true value", key);
    *dest = value;
    PetscCall(PetscPrintf(comm, "NATIVE_PROBLEM_MANIFEST_APPLY key=%s value=%s\n", key, value ? "true" : "false"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeManifestFileFingerprint(const char path[], char fingerprint[], size_t fingerprint_len)
{
  MPI_Comm comm = PETSC_COMM_WORLD;
  FILE    *fh = NULL;
  unsigned char buffer[4096];
  size_t   nread = 0;
  uint64_t value = 0xcbf29ce484222325ULL;
  PetscBool read_error = PETSC_FALSE;
  int      nwritten = 0;

  PetscFunctionBeginUser;
  PetscCheck(path && path[0], comm, PETSC_ERR_ARG_NULL, "Native manifest fingerprint requires a label-table path");
  PetscCheck(fingerprint && fingerprint_len > 0, comm, PETSC_ERR_ARG_NULL, "Native manifest fingerprint requires output storage");
  fingerprint[0] = '\0';
  fh = fopen(path, "rb");
  PetscCheck(fh, comm, PETSC_ERR_FILE_OPEN, "Cannot open native manifest label table %s for row fingerprint validation", path);
  while ((nread = fread(buffer, 1, sizeof(buffer), fh)) > 0) {
    for (size_t i = 0; i < nread; ++i) {
      value ^= (uint64_t)buffer[i];
      value *= 0x100000001b3ULL;
    }
  }
  read_error = ferror(fh) ? PETSC_TRUE : PETSC_FALSE;
  fclose(fh);
  PetscCheck(!read_error, comm, PETSC_ERR_FILE_READ, "Could not read native manifest label table %s for row fingerprint validation", path);
  nwritten = snprintf(fingerprint, fingerprint_len, "fnv1a64:%016llx", (unsigned long long)value);
  PetscCheck(nwritten > 0 && (size_t)nwritten < fingerprint_len, comm, PETSC_ERR_ARG_SIZ,
             "Native manifest label-table fingerprint buffer is too small for %s", path);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeProblemManifestValidateLabelTableFingerprint(const char json[], const char native_inputs_begin[], const char native_inputs_end[],
                                                                        PetscBool native_inputs_found, const char key[], const char path[])
{
  MPI_Comm  comm = PETSC_COMM_WORLD;
  char      expected[64];
  char      actual[64];
  PetscBool found = PETSC_FALSE;
  PetscBool same = PETSC_FALSE;

  PetscFunctionBeginUser;
  if (native_inputs_found) PetscCall(NativeManifestFindStringBounded(native_inputs_begin, native_inputs_end, key, expected, sizeof(expected), &found));
  else PetscCall(NativeManifestFindString(json, key, expected, sizeof(expected), &found));
  if (!found) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(strncmp(expected, "fnv1a64:", 8) == 0, comm, PETSC_ERR_ARG_WRONG,
             "Native problem manifest native_inputs.%s has unsupported row fingerprint %s", key, expected);
  PetscCheck(path && path[0], comm, PETSC_ERR_ARG_WRONG,
             "Native problem manifest native_inputs.%s is present but no label-table path is configured", key);
  PetscCall(NativeManifestFileFingerprint(path, actual, sizeof(actual)));
  PetscCall(PetscStrcmp(expected, actual, &same));
  PetscCheck(same, comm, PETSC_ERR_ARG_WRONG,
             "Native problem manifest native_inputs.%s=%s does not match label table %s row fingerprint %s", key, expected, path, actual);
  PetscCall(PetscPrintf(comm, "NATIVE_PROBLEM_MANIFEST_ROW_FINGERPRINT key=%s path=%s value=%s\n", key, path, actual));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode NativeProblemManifestApply(AppCtx *app)
{
  MPI_Comm    comm                = PETSC_COMM_WORLD;
  char       *json                = NULL;
  const char *native_inputs_begin = NULL;
  const char *native_inputs_end   = NULL;
  PetscBool   native_inputs_found = PETSC_FALSE;
  SsrNativeProblemTopologyStats topology_stats;
  SsrNativeProblemRuleStats     rule_stats;

  PetscFunctionBeginUser;
  if (!app->native_problem_manifest[0]) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(NativeManifestReadFile(app->native_problem_manifest, &json));
  PetscCall(NativeProblemManifestValidateMetadata(app, json));
  PetscCall(NativeProblemManifestValidateTopology(app, json, &topology_stats));
  PetscCall(NativeProblemManifestValidateBoundaryRules(app, json, &rule_stats));
  app->native_problem_manifest_loaded = PETSC_TRUE;
  app->native_manifest_mechanics_dirichlet = rule_stats.mechanics_dirichlet;
  app->native_manifest_mechanics_neumann = rule_stats.mechanics_neumann;
  app->native_manifest_seepage_head = rule_stats.seepage_head;
  app->native_manifest_seepage_flux = rule_stats.seepage_flux;
  PetscCall(NativeManifestFindObjectMember(json, "native_inputs", &native_inputs_begin, &native_inputs_end, &native_inputs_found));

  PetscCall(NativeProblemManifestApplyNativeInputPath(json, native_inputs_begin, native_inputs_end, native_inputs_found,
                                                      "mechanics_label_constraints_csv", app->mechanics_bc_labels_csv, sizeof(app->mechanics_bc_labels_csv)));
  PetscCall(NativeProblemManifestApplyNativeInputBool(json, native_inputs_begin, native_inputs_end, native_inputs_found,
                                                      "debug_coordinate_bc_table", &app->debug_coordinate_bc_table));
  PetscCall(NativeProblemManifestApplyNativeInputPath(json, native_inputs_begin, native_inputs_end, native_inputs_found,
                                                      "mechanics_coordinate_constraints_csv", app->mechanics_bc_nodes_csv, sizeof(app->mechanics_bc_nodes_csv)));
  PetscCall(NativeProblemManifestApplyNativeInputPath(json, native_inputs_begin, native_inputs_end, native_inputs_found,
                                                      "mechanics_neumann_labels_csv", app->mechanics_neumann_labels_csv, sizeof(app->mechanics_neumann_labels_csv)));
  PetscCall(NativeProblemManifestApplyNativeInputPath(json, native_inputs_begin, native_inputs_end, native_inputs_found,
                                                      "seepage_boundary_labels_csv", app->seepage_boundary_labels_csv, sizeof(app->seepage_boundary_labels_csv)));
  PetscCall(NativeProblemManifestApplyNativeInputPath(json, native_inputs_begin, native_inputs_end, native_inputs_found,
                                                      "seepage_pressure_source", app->seepage_pressure_source, sizeof(app->seepage_pressure_source)));
  PetscCall(NativeProblemManifestApplyNativeInputPath(json, native_inputs_begin, native_inputs_end, native_inputs_found,
                                                      "seepage_pressure_csv", app->seepage_pressure_csv, sizeof(app->seepage_pressure_csv)));

  PetscCheck(rule_stats.mechanics_dirichlet == 0 || app->mechanics_bc_labels_csv[0], comm, PETSC_ERR_ARG_WRONG,
             "Native problem manifest %s declares mechanics Dirichlet rules but no mechanics label constraint table is available", app->native_problem_manifest);
  PetscCheck(rule_stats.mechanics_neumann == 0 || app->mechanics_neumann_labels_csv[0], comm, PETSC_ERR_ARG_WRONG,
             "Native problem manifest %s declares mechanics Neumann rules but no mechanics Neumann label table is available", app->native_problem_manifest);
  PetscCheck(rule_stats.seepage_head + rule_stats.seepage_flux == 0 || app->seepage_boundary_labels_csv[0], comm, PETSC_ERR_ARG_WRONG,
             "Native problem manifest %s declares seepage boundary rules but no seepage boundary label table is available", app->native_problem_manifest);
  PetscCall(NativeProblemManifestValidateLabelTableFingerprint(json, native_inputs_begin, native_inputs_end, native_inputs_found,
                                                               "mechanics_label_constraints_row_fingerprint", app->mechanics_bc_labels_csv));
  PetscCall(NativeProblemManifestValidateLabelTableFingerprint(json, native_inputs_begin, native_inputs_end, native_inputs_found,
                                                               "mechanics_neumann_labels_row_fingerprint", app->mechanics_neumann_labels_csv));
  PetscCall(NativeProblemManifestValidateLabelTableFingerprint(json, native_inputs_begin, native_inputs_end, native_inputs_found,
                                                               "seepage_boundary_labels_row_fingerprint", app->seepage_boundary_labels_csv));

  PetscCall(PetscFree(json));
  PetscFunctionReturn(PETSC_SUCCESS);
}
