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
