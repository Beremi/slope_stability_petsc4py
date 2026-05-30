static PetscErrorCode BuildBasisReferencePoints(P4Basis *basis, PetscReal **points_out)
{
  PetscDualSpace dual;
  PetscReal     *points_out_local;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(basis->dim * basis->n_basis, &points_out_local));
  PetscCall(PetscFEGetDualSpace(basis->fe_scalar, &dual));
  for (PetscInt b = 0; b < basis->n_basis; ++b) {
    PetscQuadrature  q;
    PetscInt         dim, Nc, npoints;
    const PetscReal *points;

    PetscCall(PetscDualSpaceGetFunctional(dual, b, &q));
    PetscCall(PetscQuadratureGetData(q, &dim, &Nc, &npoints, &points, NULL));
    PetscCheck(dim == basis->dim && Nc == 1 && npoints >= 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected scalar dual functional shape");
    for (PetscInt d = 0; d < basis->dim; ++d) points_out_local[basis->dim * b + d] = points[d];
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

static PetscErrorCode BuildInterpolationMatrixWithLayouts(DM fine_dm, P4Basis *fine_basis, DM coarse_dm, P4Basis *coarse_basis,
                                                          PetscInt mat_mlocal, PetscInt mat_nlocal, PetscInt mat_M, PetscInt mat_N, Mat *P)
{
  MPI_Comm        comm = PetscObjectComm((PetscObject)fine_dm);
  PetscSection    fine_lsec, fine_gsec, coarse_lsec, coarse_gsec;
  Vec             fine_vec = NULL;
  PetscInt        rlo, rhi, cStart, cEnd;
  PetscReal      *fine_points = NULL;
  PetscTabulation coarse_at_fine = NULL;
  const PetscReal *phi;
  Mat             mat;
  const PetscInt  ncomp = fine_basis->components;

  PetscFunctionBeginUser;
  PetscCheck(fine_basis->dim == coarse_basis->dim, comm, PETSC_ERR_ARG_INCOMP, "Fine and coarse PMG bases have different dimensions");
  PetscCheck(fine_basis->components == coarse_basis->components, comm, PETSC_ERR_ARG_INCOMP, "Fine and coarse PMG bases have different component counts");
  PetscCall(DMCreateGlobalVector(fine_dm, &fine_vec));
  PetscCall(VecGetOwnershipRange(fine_vec, &rlo, &rhi));
  PetscCall(VecDestroy(&fine_vec));

  PetscCall(BuildBasisReferencePoints(fine_basis, &fine_points));
  PetscCall(PetscFECreateTabulation(coarse_basis->fe_scalar, 1, fine_basis->n_basis, fine_points, 0, &coarse_at_fine));
  phi = coarse_at_fine->T[0];

  PetscCall(MatCreateAIJ(comm, mat_mlocal, mat_nlocal, mat_M, mat_N, coarse_basis->n_basis, NULL, coarse_basis->n_basis, NULL, &mat));
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
    PetscCheck(fine_n == ncomp * fine_basis->n_basis, comm, PETSC_ERR_PLIB, "Unexpected fine transfer closure size %" PetscInt_FMT, fine_n);
    PetscCheck(coarse_n == ncomp * coarse_basis->n_basis, comm, PETSC_ERR_PLIB, "Unexpected coarse transfer closure size %" PetscInt_FMT, coarse_n);
    for (PetscInt fb = 0; fb < fine_basis->n_basis; ++fb) {
      for (PetscInt comp = 0; comp < ncomp; ++comp) {
        const PetscInt row = fine_idx[ncomp * fb + comp];

        if (row < 0) continue;
        if (row < rlo || row >= rhi) continue;
        for (PetscInt cb = 0; cb < coarse_basis->n_basis; ++cb) {
          const PetscScalar val = phi[fb * coarse_basis->n_basis + cb];
          const PetscInt    col = coarse_idx[ncomp * cb + comp];

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

static PetscErrorCode BuildInterpolationMatrix(DM fine_dm, P4Basis *fine_basis, DM coarse_dm, P4Basis *coarse_basis, Mat *P)
{
  Vec      fine_vec = NULL, coarse_vec = NULL;
  PetscInt mlocal, nlocal, M, N;

  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(fine_dm, &fine_vec));
  PetscCall(DMCreateGlobalVector(coarse_dm, &coarse_vec));
  PetscCall(VecGetLocalSize(fine_vec, &mlocal));
  PetscCall(VecGetLocalSize(coarse_vec, &nlocal));
  PetscCall(VecGetSize(fine_vec, &M));
  PetscCall(VecGetSize(coarse_vec, &N));
  PetscCall(VecDestroy(&fine_vec));
  PetscCall(VecDestroy(&coarse_vec));
  PetscCall(BuildInterpolationMatrixWithLayouts(fine_dm, fine_basis, coarse_dm, coarse_basis, mlocal, nlocal, M, N, P));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ChoosePMGCoarsePC(const SsrPmgOptions *pmg, DM coarse_dm, char coarse_pc[], size_t coarse_pc_len)
{
  MPI_Comm  comm = PetscObjectComm((PetscObject)coarse_dm);
  Vec       v = NULL;
  PetscInt  coarse_dofs;
  PetscMPIInt size;
  PetscBool flg, direct_set = PETSC_FALSE;
  char      direct_pc[32] = "";

  PetscFunctionBeginUser;
  PetscCheck(pmg, comm, PETSC_ERR_ARG_NULL, "PMG coarse PC selection requires resolved PMG options");
  PetscCall(DMCreateGlobalVector(coarse_dm, &v));
  PetscCall(VecGetSize(v, &coarse_dofs));
  PetscCall(VecDestroy(&v));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-mg_coarse_pc_type", direct_pc, sizeof(direct_pc), &direct_set));
  if (direct_set) {
    PetscCall(PetscStrcasecmp(direct_pc, "lu", &flg));
    PetscCheck(!flg || coarse_dofs <= pmg->coarse_lu_max_dofs, comm, PETSC_ERR_ARG_WRONG,
               "Refusing PMG coarse LU for %" PetscInt_FMT " coarse DOFs above -pmg_coarse_lu_max_dofs %" PetscInt_FMT, coarse_dofs, pmg->coarse_lu_max_dofs);
#if !defined(PETSC_HAVE_HYPRE)
    PetscCall(PetscStrcasecmp(direct_pc, "hypre", &flg));
    PetscCheck(!flg, comm, PETSC_ERR_SUP, "This PETSc build has no HYPRE support");
#endif
    PetscCall(PetscStrncpy(coarse_pc, direct_pc, coarse_pc_len));
    PetscCall(PetscPrintf(comm, "PMG coarse space: dofs=%" PetscInt_FMT " selected_pc=%s source=mg_coarse_pc_type lu_limit=%" PetscInt_FMT "\n", coarse_dofs, coarse_pc,
                          pmg->coarse_lu_max_dofs));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscStrcasecmp(pmg->coarse_pc_type, "auto", &flg));
  if (flg) {
    if (size == 1 && coarse_dofs <= pmg->coarse_lu_max_dofs) PetscCall(PetscStrncpy(coarse_pc, "lu", coarse_pc_len));
    else PetscCall(PetscStrncpy(coarse_pc, "gamg", coarse_pc_len));
  } else {
    PetscBool is_lu, is_hypre, is_gamg;

    PetscCall(PetscStrcasecmp(pmg->coarse_pc_type, "lu", &is_lu));
    PetscCall(PetscStrcasecmp(pmg->coarse_pc_type, "hypre", &is_hypre));
    PetscCall(PetscStrcasecmp(pmg->coarse_pc_type, "gamg", &is_gamg));
    PetscCheck(is_lu || is_hypre || is_gamg, comm, PETSC_ERR_ARG_WRONG, "-pmg_coarse_pc_type must be auto, hypre, gamg, or lu");
    PetscCheck(!is_lu || coarse_dofs <= pmg->coarse_lu_max_dofs, comm, PETSC_ERR_ARG_WRONG,
               "Refusing PMG coarse LU for %" PetscInt_FMT " coarse DOFs above -pmg_coarse_lu_max_dofs %" PetscInt_FMT, coarse_dofs, pmg->coarse_lu_max_dofs);
#if !defined(PETSC_HAVE_HYPRE)
    PetscCheck(!is_hypre, comm, PETSC_ERR_SUP, "This PETSc build has no HYPRE support");
#endif
    PetscCall(PetscStrncpy(coarse_pc, pmg->coarse_pc_type, coarse_pc_len));
  }
  PetscCall(PetscPrintf(comm, "PMG coarse space: dofs=%" PetscInt_FMT " selected_pc=%s lu_limit=%" PetscInt_FMT "\n", coarse_dofs, coarse_pc, pmg->coarse_lu_max_dofs));
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

static PetscErrorCode SetPMGTelescopeDefaults(const SsrPmgOptions *pmg, MPI_Comm comm)
{
  PetscMPIInt ranks;
  PetscBool   pc_set;
  char        value[64];

  PetscFunctionBeginUser;
  PetscCheck(pmg, comm, PETSC_ERR_ARG_NULL, "PMG telescope defaults require resolved PMG options");
  PetscCallMPI(MPI_Comm_size(comm, &ranks));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-mg_coarse_pc_type", &pc_set));
  if (pmg->coarse_telescope_active_ranks <= 0 || ranks <= pmg->coarse_telescope_active_ranks || pc_set) {
    PetscCall(PetscPrintf(comm,
                          "PMG_TELESCOPE_CONFIG level=p1 enabled=false option_override=%s active_ranks=%" PetscInt_FMT " reduction_factor=1 subcomm=%s inner_ksp=%s inner_pc=%s\n",
                          pc_set ? "true" : "false", pmg->coarse_telescope_active_ranks, pmg->coarse_telescope_subcomm_type,
                          pmg->coarse_ksp_type, pmg->coarse_inner_pc_type));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(ranks % pmg->coarse_telescope_active_ranks == 0, comm, PETSC_ERR_ARG_WRONG,
             "-pmg_coarse_telescope_active_ranks %" PetscInt_FMT " must divide MPI ranks %d", pmg->coarse_telescope_active_ranks, ranks);

  PetscCall(SetDefaultOption("-mg_coarse_pc_type", "telescope"));
  PetscCall(PetscSNPrintf(value, sizeof(value), "%" PetscInt_FMT, (PetscInt)ranks / pmg->coarse_telescope_active_ranks));
  PetscCall(SetDefaultOption("-mg_coarse_pc_telescope_reduction_factor", value));
  PetscCall(SetDefaultOption("-mg_coarse_pc_telescope_subcomm_type", pmg->coarse_telescope_subcomm_type));
  PetscCall(SetDefaultOption("-mg_coarse_telescope_ksp_type", pmg->coarse_ksp_type));
  PetscCall(PetscSNPrintf(value, sizeof(value), "%.16g", (double)pmg->coarse_ksp_rtol));
  PetscCall(SetDefaultOption("-mg_coarse_telescope_ksp_rtol", value));
  PetscCall(PetscSNPrintf(value, sizeof(value), "%" PetscInt_FMT, pmg->coarse_ksp_max_it));
  PetscCall(SetDefaultOption("-mg_coarse_telescope_ksp_max_it", value));
  PetscCall(SetDefaultOption("-mg_coarse_telescope_pc_type", pmg->coarse_inner_pc_type));
  PetscCall(PetscPrintf(comm,
                        "PMG_TELESCOPE_CONFIG level=p1 enabled=true option_override=false active_ranks=%" PetscInt_FMT " reduction_factor=%" PetscInt_FMT " subcomm=%s inner_ksp=%s inner_pc=%s\n",
                        pmg->coarse_telescope_active_ranks, (PetscInt)ranks / pmg->coarse_telescope_active_ranks, pmg->coarse_telescope_subcomm_type,
                        pmg->coarse_ksp_type, pmg->coarse_inner_pc_type));
  PetscCall(PetscPrintf(comm,
                        "PMG_TELESCOPE active_ranks=%" PetscInt_FMT " reduction_factor=%" PetscInt_FMT " subcomm=%s inner_ksp=%s inner_pc=%s\n",
                        pmg->coarse_telescope_active_ranks, (PetscInt)ranks / pmg->coarse_telescope_active_ranks, pmg->coarse_telescope_subcomm_type,
                        pmg->coarse_ksp_type, pmg->coarse_inner_pc_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetPMGP2TelescopeDefaults(const SsrPmgOptions *pmg, MPI_Comm comm)
{
  PetscMPIInt ranks;
  PetscBool   pc_set;
  char        value[64];

  PetscFunctionBeginUser;
  PetscCheck(pmg, comm, PETSC_ERR_ARG_NULL, "PMG P2 telescope defaults require resolved PMG options");
  PetscCallMPI(MPI_Comm_size(comm, &ranks));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-mg_levels_1_pc_type", &pc_set));
  if (pmg->p2_telescope_active_ranks <= 0 || ranks <= pmg->p2_telescope_active_ranks || pc_set) {
    PetscCall(PetscPrintf(comm,
                          "PMG_TELESCOPE_CONFIG level=p2 enabled=false option_override=%s active_ranks=%" PetscInt_FMT " reduction_factor=1 subcomm=%s inner_ksp=%s inner_pc=%s\n",
                          pc_set ? "true" : "false", pmg->p2_telescope_active_ranks, pmg->p2_telescope_subcomm_type,
                          pmg->p2_telescope_ksp_type, pmg->p2_telescope_pc_type));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCheck(ranks % pmg->p2_telescope_active_ranks == 0, comm, PETSC_ERR_ARG_WRONG,
             "-pmg_p2_telescope_active_ranks %" PetscInt_FMT " must divide MPI ranks %d", pmg->p2_telescope_active_ranks, ranks);

  PetscCall(SetDefaultOption("-mg_levels_1_pc_type", "telescope"));
  PetscCall(PetscSNPrintf(value, sizeof(value), "%" PetscInt_FMT, (PetscInt)ranks / pmg->p2_telescope_active_ranks));
  PetscCall(SetDefaultOption("-mg_levels_1_pc_telescope_reduction_factor", value));
  PetscCall(SetDefaultOption("-mg_levels_1_pc_telescope_subcomm_type", pmg->p2_telescope_subcomm_type));
  PetscCall(SetDefaultOption("-mg_levels_1_telescope_ksp_type", pmg->p2_telescope_ksp_type));
  PetscCall(PetscSNPrintf(value, sizeof(value), "%.16g", (double)pmg->p2_telescope_ksp_rtol));
  PetscCall(SetDefaultOption("-mg_levels_1_telescope_ksp_rtol", value));
  PetscCall(PetscSNPrintf(value, sizeof(value), "%" PetscInt_FMT, pmg->p2_telescope_ksp_max_it));
  PetscCall(SetDefaultOption("-mg_levels_1_telescope_ksp_max_it", value));
  PetscCall(SetDefaultOption("-mg_levels_1_telescope_pc_type", pmg->p2_telescope_pc_type));
  {
    PetscBool is_gamg = PETSC_FALSE;

    PetscCall(PetscStrcasecmp(pmg->p2_telescope_pc_type, "gamg", &is_gamg));
    if (is_gamg) PetscCall(SetDefaultOption("-mg_levels_1_telescope_pc_gamg_aggressive_square_graph", pmg->coarse_gamg_aggressive_square_graph ? "true" : "false"));
  }
  PetscCall(PetscPrintf(comm,
                        "PMG_TELESCOPE_CONFIG level=p2 enabled=true option_override=false active_ranks=%" PetscInt_FMT " reduction_factor=%" PetscInt_FMT " subcomm=%s inner_ksp=%s inner_pc=%s\n",
                        pmg->p2_telescope_active_ranks, (PetscInt)ranks / pmg->p2_telescope_active_ranks, pmg->p2_telescope_subcomm_type,
                        pmg->p2_telescope_ksp_type, pmg->p2_telescope_pc_type));
  PetscCall(PetscPrintf(comm,
                        "PMG_P2_TELESCOPE active_ranks=%" PetscInt_FMT " reduction_factor=%" PetscInt_FMT " subcomm=%s inner_ksp=%s inner_pc=%s\n",
                        pmg->p2_telescope_active_ranks, (PetscInt)ranks / pmg->p2_telescope_active_ranks, pmg->p2_telescope_subcomm_type,
                        pmg->p2_telescope_ksp_type, pmg->p2_telescope_pc_type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

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

static PetscErrorCode PMGResolveActiveRanks(MPI_Comm comm, PetscInt requested_active_ranks, PetscInt *active_ranks, PetscInt *reduction_factor)
{
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (requested_active_ranks <= 0 || requested_active_ranks >= (PetscInt)size) *active_ranks = (PetscInt)size;
  else *active_ranks = requested_active_ranks;
  PetscCheck(*active_ranks > 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "Active-rank count must be positive");
  PetscCheck((PetscInt)size % *active_ranks == 0, comm, PETSC_ERR_ARG_WRONG,
             "Active-rank count %" PetscInt_FMT " must divide MPI ranks %d", *active_ranks, size);
  *reduction_factor = (PetscInt)size / *active_ranks;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGIsActiveRank(MPI_Comm comm, PetscInt active_ranks, PetscInt reduction_factor, PetscSubcommType subcomm_type, PetscBool *active)
{
  PetscMPIInt rank, size;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (active_ranks == (PetscInt)size) *active = PETSC_TRUE;
  else if (subcomm_type == PETSC_SUBCOMM_CONTIGUOUS) *active = (PetscBool)(rank < active_ranks);
  else *active = (PetscBool)(rank % reduction_factor == 0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscBool       active;
  PetscInt        requested_active_ranks;
  PetscInt        active_ranks;
  PetscInt        inactive_ranks;
  PetscInt        reduction_factor;
  PetscInt        global_dofs;
  PetscInt        local_dofs;
  PetscInt        local_min;
  PetscInt        local_max;
  PetscInt        ownership_start;
  PetscInt        ownership_end;
  PetscInt        block_size;
  PetscSubcommType subcomm_type;
  char            subcomm_type_name[32];
  MPI_Comm        subcomm;
  IS              isrow;
  Vec             full_template;
  Vec             sub_template;
  VecScatter      original_to_active;
} PMGActiveLayout;

static PetscErrorCode PMGActiveLayoutCreate(MPI_Comm comm, Vec original_template, PetscInt requested_active_ranks, const char subcomm_type_name[], PMGActiveLayout *layout)
{
  PetscMPIInt rank, size;
  VecType     vec_type = NULL;
  PetscInt    local_for_min;
  PetscInt    original_local, original_start, original_end;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(layout, sizeof(*layout)));
  layout->subcomm = MPI_COMM_NULL;
  layout->requested_active_ranks = requested_active_ranks;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PMGParseSubcommType(comm, subcomm_type_name, &layout->subcomm_type));
  PetscCall(PetscStrncpy(layout->subcomm_type_name, subcomm_type_name, sizeof(layout->subcomm_type_name)));
  PetscCall(VecGetSize(original_template, &layout->global_dofs));
  PetscCall(VecGetLocalSize(original_template, &original_local));
  PetscCall(VecGetOwnershipRange(original_template, &original_start, &original_end));
  PetscCall(VecGetBlockSize(original_template, &layout->block_size));
  PetscCall(VecGetType(original_template, &vec_type));
  PetscCall(PMGResolveActiveRanks(comm, requested_active_ranks, &layout->active_ranks, &layout->reduction_factor));
  layout->inactive_ranks   = (PetscInt)size - layout->active_ranks;
  PetscCall(PMGIsActiveRank(comm, layout->active_ranks, layout->reduction_factor, layout->subcomm_type, &layout->active));

  PetscCallMPI(MPI_Comm_split(comm, layout->active ? 0 : MPI_UNDEFINED, rank, &layout->subcomm));
  if (layout->active) {
    PetscCall(VecCreate(layout->subcomm, &layout->sub_template));
    PetscCall(VecSetSizes(layout->sub_template, layout->active_ranks == (PetscInt)size ? original_local : PETSC_DECIDE, layout->global_dofs));
    PetscCall(VecSetBlockSize(layout->sub_template, layout->block_size));
    if (vec_type) PetscCall(VecSetType(layout->sub_template, vec_type));
    PetscCall(VecSetFromOptions(layout->sub_template));
    PetscCall(VecGetLocalSize(layout->sub_template, &layout->local_dofs));
    PetscCall(VecGetOwnershipRange(layout->sub_template, &layout->ownership_start, &layout->ownership_end));
    if (layout->active_ranks == (PetscInt)size) {
      PetscCheck(layout->local_dofs == original_local && layout->ownership_start == original_start && layout->ownership_end == original_end, comm, PETSC_ERR_PLIB,
                 "All-active PMG layout failed to preserve original ownership: original [%" PetscInt_FMT ",%" PetscInt_FMT ") local %" PetscInt_FMT
                 ", active [%" PetscInt_FMT ",%" PetscInt_FMT ") local %" PetscInt_FMT,
                 original_start, original_end, original_local, layout->ownership_start, layout->ownership_end, layout->local_dofs);
    }
  } else {
    layout->local_dofs      = 0;
    layout->ownership_start = 0;
    layout->ownership_end   = 0;
  }

  PetscCall(VecCreate(comm, &layout->full_template));
  PetscCall(VecSetSizes(layout->full_template, layout->local_dofs, layout->global_dofs));
  PetscCall(VecSetBlockSize(layout->full_template, layout->block_size));
  if (vec_type) PetscCall(VecSetType(layout->full_template, vec_type));
  PetscCall(VecSetFromOptions(layout->full_template));
  PetscCall(ISCreateStride(comm, layout->local_dofs, layout->ownership_start, 1, &layout->isrow));
  PetscCall(ISSetBlockSize(layout->isrow, layout->block_size));
  PetscCall(VecScatterCreate(original_template, layout->isrow, layout->full_template, NULL, &layout->original_to_active));

  local_for_min = layout->active ? layout->local_dofs : PETSC_MAX_INT;
  PetscCallMPI(MPI_Allreduce(&local_for_min, &layout->local_min, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPI_Allreduce(&layout->local_dofs, &layout->local_max, 1, MPIU_INT, MPI_MAX, comm));
  if (layout->local_min == PETSC_MAX_INT) layout->local_min = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGActiveLayoutDestroy(PMGActiveLayout *layout)
{
  PetscFunctionBeginUser;
  PetscCall(VecScatterDestroy(&layout->original_to_active));
  PetscCall(VecDestroy(&layout->full_template));
  PetscCall(VecDestroy(&layout->sub_template));
  PetscCall(ISDestroy(&layout->isrow));
  if (layout->subcomm != MPI_COMM_NULL) PetscCallMPI(MPI_Comm_free(&layout->subcomm));
  PetscCall(PetscMemzero(layout, sizeof(*layout)));
  layout->subcomm = MPI_COMM_NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGActiveLayoutDuplicateFull(PMGActiveLayout *layout, Vec *v)
{
  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(layout->full_template, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGActiveLayoutDuplicateSub(PMGActiveLayout *layout, Vec *v)
{
  PetscFunctionBeginUser;
  if (layout->active) PetscCall(VecDuplicate(layout->sub_template, v));
  else *v = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGActiveLayoutCopyFullToSub(PMGActiveLayout *layout, Vec full, Vec sub)
{
  PetscFunctionBeginUser;
  if (layout->active) {
    const PetscScalar *full_array;
    PetscScalar       *sub_array;
    PetscInt           nfull, nsub;

    PetscCall(VecGetLocalSize(full, &nfull));
    PetscCall(VecGetLocalSize(sub, &nsub));
    PetscCheck(nfull == nsub, layout->subcomm, PETSC_ERR_PLIB, "Active full/sub vector sizes differ: %" PetscInt_FMT " != %" PetscInt_FMT, nfull, nsub);
    PetscCall(VecGetArrayRead(full, &full_array));
    PetscCall(VecGetArray(sub, &sub_array));
    for (PetscInt i = 0; i < nsub; ++i) sub_array[i] = full_array[i];
    PetscCall(VecRestoreArray(sub, &sub_array));
    PetscCall(VecRestoreArrayRead(full, &full_array));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGActiveLayoutCopySubToFull(PMGActiveLayout *layout, Vec sub, Vec full)
{
  PetscFunctionBeginUser;
  if (layout->active) {
    const PetscScalar *sub_array;
    PetscScalar       *full_array;
    PetscInt           nfull, nsub;

    PetscCall(VecGetLocalSize(full, &nfull));
    PetscCall(VecGetLocalSize(sub, &nsub));
    PetscCheck(nfull == nsub, layout->subcomm, PETSC_ERR_PLIB, "Active full/sub vector sizes differ: %" PetscInt_FMT " != %" PetscInt_FMT, nfull, nsub);
    PetscCall(VecGetArrayRead(sub, &sub_array));
    PetscCall(VecGetArray(full, &full_array));
    for (PetscInt i = 0; i < nfull; ++i) full_array[i] = sub_array[i];
    PetscCall(VecRestoreArray(full, &full_array));
    PetscCall(VecRestoreArrayRead(sub, &sub_array));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGActiveLayoutScatterOriginalToFull(PMGActiveLayout *layout, Vec original, Vec active_full)
{
  PetscFunctionBeginUser;
  PetscCall(VecScatterBegin(layout->original_to_active, original, active_full, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(layout->original_to_active, original, active_full, INSERT_VALUES, SCATTER_FORWARD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellRedistributeActiveMatrix(MPI_Comm comm, PMGActiveLayout *layout, Mat B, MatReuse reuse, Mat *Bred,
                                                       PetscLogDouble *submatrix_time, PetscLogDouble *concatenate_time)
{
  Mat           *submats = NULL;
  Mat            Blocal = NULL;
  IS             iscol = NULL;
  PetscInt       nc, nr, bs;
  SsrProfileTimer profile_timer;

  PetscFunctionBeginUser;
  *submatrix_time   = 0.0;
  *concatenate_time = 0.0;
  PetscCall(MatGetSize(B, &nr, &nc));
  PetscCall(MatGetBlockSizes(B, NULL, &bs));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, nc, 0, 1, &iscol));
  PetscCall(ISSetIdentity(iscol));
  PetscCall(ISSetBlockSize(iscol, bs));
  PetscCall(MatSetOption(B, MAT_SUBMAT_SINGLEIS, PETSC_TRUE));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_SUBMATRIX, B, iscol, &profile_timer);
  PetscCall(MatCreateSubMatrices(B, 1, &layout->isrow, &iscol, MAT_INITIAL_MATRIX, &submats));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, submatrix_time);
  Blocal = submats[0];
  PetscCall(PetscFree(submats));
  if (layout->active) {
    PetscInt mm;

    PetscCall(MatGetSize(Blocal, &mm, NULL));
    SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_CONCATENATE, Blocal, NULL, &profile_timer);
    PetscCall(MatCreateMPIMatConcatenateSeqMat(layout->subcomm, Blocal, mm, reuse, Bred));
    SSR_PROFILE_TIMER_END(NULL, &profile_timer, concatenate_time);
  }
  PetscCall(MatDestroy(&Blocal));
  PetscCall(ISDestroy(&iscol));
  (void)comm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellAttachSubcommNearNullspace(DM dm, PMGActiveLayout *layout, Mat Ared)
{
  DM           subdm = NULL;
  MatNullSpace ns = NULL;
  PetscInt     field = 0;
  PetscBool    has_const = PETSC_FALSE;
  PetscInt     nvec = 0;
  const Vec   *vecs = NULL;
  Vec         *sub_vecs = NULL;
  Vec          active_tmp = NULL;

  PetscFunctionBeginUser;
  PetscCall(DMCreateSubDM(dm, 1, &field, NULL, &subdm));
  PetscCall(DMPlexCreateRigidBody(subdm, 0, &ns));
  PetscCall(MatNullSpaceGetVecs(ns, &has_const, &nvec, &vecs));
  PetscCall(PMGActiveLayoutDuplicateFull(layout, &active_tmp));
  if (layout->active && nvec > 0) PetscCall(VecDuplicateVecs(layout->sub_template, nvec, &sub_vecs));
  for (PetscInt i = 0; i < nvec; ++i) {
    PetscCall(PMGActiveLayoutScatterOriginalToFull(layout, vecs[i], active_tmp));
    PetscCall(PMGActiveLayoutCopyFullToSub(layout, active_tmp, layout->active ? sub_vecs[i] : NULL));
  }
  if (layout->active) {
    MatNullSpace sub_ns = NULL;

    PetscCall(MatNullSpaceCreate(layout->subcomm, has_const, nvec, sub_vecs, &sub_ns));
    PetscCall(MatSetNearNullSpace(Ared, sub_ns));
    PetscCall(MatNullSpaceDestroy(&sub_ns));
    if (nvec > 0) PetscCall(VecDestroyVecs(nvec, &sub_vecs));
  }
  PetscCall(VecDestroy(&active_tmp));
  PetscCall(MatNullSpaceDestroy(&ns));
  PetscCall(DMDestroy(&subdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  AppCtx        *app;
  const SsrPmgOptions *pmg;
  DM             dm;
  AssemblyCtx   *actx;
  LinearSolverCtx *owner;
  Mat            A4;
  PetscBool      setup_done;
  PetscBool      p1_basis_created;
  PetscBool      p2_basis_created;
  P4Basis        p1_basis;
  P4Basis        p2_basis;
  DM             dm_p1;
  DM             dm_p2;
  PMGActiveLayout p2_layout;
  PMGActiveLayout p1_layout;
  Mat            P42;
  Mat            P21;
  Mat            A2_active;
  Mat            A1_active;
  Mat            A2_sub;
  Mat            A1_sub;
  KSP            smooth4;
  KSP            smooth2;
  KSP            coarse1;
  Vec            r4;
  Vec            tmp4;
  Vec            z4;
  Vec            p2_rhs_full;
  Vec            p2_x_full;
  Vec            p2_r_full;
  Vec            p2_corr_full;
  Vec            p1_rhs_full;
  Vec            p1_x_full;
  Vec            p2_rhs;
  Vec            p2_x;
  Vec            p2_r;
  Vec            p2_delta;
  Vec            p1_rhs;
  Vec            p1_x;
  PetscBool      debug_capture;
  Vec            debug_fine_pre;
  Vec            debug_fine_residual;
  Vec            debug_p2_rhs;
  Vec            debug_p2_pre;
  Vec            debug_p2_residual;
  Vec            debug_p1_rhs;
  Vec            debug_p1_x;
  Vec            debug_p2_post;
  PetscInt       apply_calls;
  PetscInt       operator_updates;
  PetscLogDouble fine_smooth_time;
  PetscLogDouble p2_smooth_time;
  PetscLogDouble restrict_time;
  PetscLogDouble prolong_time;
  PetscLogDouble coarse_solve_time;
  PetscLogDouble residual_time;
  PetscLogDouble operator_update_time;
} PMGShellVCycleCtx;

static PetscErrorCode LinearSolverDestroyPMGHierarchy(LinearSolverCtx *solver);
static PetscErrorCode CheckPMGCoarseTransfers(LinearSolverCtx *solver);
static PetscErrorCode LinearSolverSetupPMGTransferChecks(LinearSolverCtx *solver, DM dm, const AppCtx *app, const SsrPmgOptions *pmg);

static PetscErrorCode PMGShellConfigureSmootherKSP(KSP ksp, const SsrPmgOptions *pmg, const char prefix[], Mat A)
{
  PC pc = NULL;

  PetscFunctionBeginUser;
  PetscCheck(pmg, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_NULL, "PMG shell smoother configuration requires resolved PMG options");
  PetscCall(KSPSetOptionsPrefix(ksp, prefix));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, pmg->smoother_ksp_type));
  PetscCall(KSPSetNormType(ksp, KSP_NORM_NONE));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(KSPSetTolerances(ksp, 0.0, 0.0, PETSC_CURRENT, pmg->smoother_max_it));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, pmg->smoother_pc_type));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellConfigureCoarseKSP(KSP ksp, const SsrPmgOptions *pmg, const char prefix[], Mat A)
{
  PC pc = NULL;

  PetscFunctionBeginUser;
  PetscCheck(pmg, PetscObjectComm((PetscObject)ksp), PETSC_ERR_ARG_NULL, "PMG shell coarse configuration requires resolved PMG options");
  PetscCall(KSPSetOptionsPrefix(ksp, prefix));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, pmg->coarse_ksp_type));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(KSPSetTolerances(ksp, pmg->coarse_ksp_rtol, PETSC_DEFAULT, PETSC_DEFAULT, pmg->coarse_ksp_max_it));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(ConfigurePMGBasePC(pc, pmg->coarse_inner_pc_type, pmg->coarse_gamg_aggressive_square_graph));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellReportFineLevel(DM dm, PetscInt degree)
{
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);
  Vec      v = NULL;
  PetscMPIInt size;
  PetscInt local_dofs, global_dofs, local_min, local_max;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMCreateGlobalVector(dm, &v));
  PetscCall(VecGetLocalSize(v, &local_dofs));
  PetscCall(VecGetSize(v, &global_dofs));
  PetscCallMPI(MPI_Allreduce(&local_dofs, &local_min, 1, MPIU_INT, MPI_MIN, comm));
  PetscCallMPI(MPI_Allreduce(&local_dofs, &local_max, 1, MPIU_INT, MPI_MAX, comm));
  PetscCall(PetscPrintf(comm,
                        "PMG_SHELL_LEVEL level=2 degree=%" PetscInt_FMT " active_ranks=%d global_dofs=%" PetscInt_FMT " local_min=%" PetscInt_FMT " local_max=%" PetscInt_FMT " inactive_ranks=0\n",
                        degree, size, global_dofs, local_min, local_max));
  PetscCall(VecDestroy(&v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellReportActiveLevel(MPI_Comm comm, PetscInt level, PetscInt degree, PMGActiveLayout *layout)
{
  PetscFunctionBeginUser;
  PetscCall(PetscPrintf(comm,
                        "PMG_SHELL_LEVEL level=%" PetscInt_FMT " degree=%" PetscInt_FMT " active_ranks=%" PetscInt_FMT " global_dofs=%" PetscInt_FMT " local_min=%" PetscInt_FMT " local_max=%" PetscInt_FMT " inactive_ranks=%" PetscInt_FMT "\n",
                        level, degree, layout->active_ranks, layout->global_dofs, layout->local_min, layout->local_max, layout->inactive_ranks));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellCreateHierarchy(PC pc, PMGShellVCycleCtx *ctx, Mat A4)
{
  MPI_Comm  comm = PetscObjectComm((PetscObject)pc);
  const SsrPmgOptions *pmg = ctx->pmg;
  Vec       fine_vec = NULL, p2_vec = NULL, p1_vec = NULL;
  PetscInt  fine_local, fine_global;
  char      prefix4[128];

  PetscFunctionBeginUser;
  PetscCheck(pmg, comm, PETSC_ERR_ARG_NULL, "PMG shell hierarchy requires resolved PMG options");
  PetscCall(P4BasisCreateDegreeDim(PETSC_COMM_SELF, 1, ctx->actx->basis->dim, ctx->actx->basis->components, &ctx->p1_basis));
  ctx->p1_basis_created = PETSC_TRUE;
  PetscCall(P4BasisCreateDegreeDim(PETSC_COMM_SELF, 2, ctx->actx->basis->dim, ctx->actx->basis->components, &ctx->p2_basis));
  ctx->p2_basis_created = PETSC_TRUE;
  PetscCall(CreateSameMeshLevelDM(ctx->dm, &ctx->p1_basis, ctx->app, &ctx->dm_p1));
  PetscCall(CreateSameMeshLevelDM(ctx->dm, &ctx->p2_basis, ctx->app, &ctx->dm_p2));

  PetscCall(ReportPMGLevelDofs(ctx->dm_p1, 0, 1));
  PetscCall(ReportPMGLevelDofs(ctx->dm_p2, 1, 2));
  PetscCall(ReportPMGLevelDofs(ctx->dm, 2, ctx->actx->basis->degree));

  PetscCall(DMCreateGlobalVector(ctx->dm, &fine_vec));
  PetscCall(DMCreateGlobalVector(ctx->dm_p2, &p2_vec));
  PetscCall(DMCreateGlobalVector(ctx->dm_p1, &p1_vec));
  PetscCall(PMGActiveLayoutCreate(comm, p2_vec, pmg->p2_active_ranks, pmg->shell_subcomm_type, &ctx->p2_layout));
  PetscCall(PMGActiveLayoutCreate(comm, p1_vec, pmg->p1_active_ranks, pmg->shell_subcomm_type, &ctx->p1_layout));
  PetscCall(PMGShellReportFineLevel(ctx->dm, ctx->actx->basis->degree));
  PetscCall(PMGShellReportActiveLevel(comm, 1, 2, &ctx->p2_layout));
  PetscCall(PMGShellReportActiveLevel(comm, 0, 1, &ctx->p1_layout));

  PetscCall(VecGetLocalSize(fine_vec, &fine_local));
  PetscCall(VecGetSize(fine_vec, &fine_global));
  PetscCall(BuildInterpolationMatrixWithLayouts(ctx->dm_p2, &ctx->p2_basis, ctx->dm_p1, &ctx->p1_basis, ctx->p2_layout.local_dofs,
                                                ctx->p1_layout.local_dofs, ctx->p2_layout.global_dofs, ctx->p1_layout.global_dofs, &ctx->P21));
  PetscCall(BuildInterpolationMatrixWithLayouts(ctx->dm, ctx->actx->basis, ctx->dm_p2, &ctx->p2_basis, fine_local, ctx->p2_layout.local_dofs,
                                                fine_global, ctx->p2_layout.global_dofs, &ctx->P42));
  PetscCall(PetscPrintf(comm,
                        "PMG_COARSE_OPERATOR_CONFIG type=galerkin_shell_vcycle p1_quadrature_points=%" PetscInt_FMT " p2_quadrature_points=%" PetscInt_FMT "\n",
                        ctx->p1_basis.n_qp, ctx->p2_basis.n_qp));
  if (pmg->check_coarse_transfers) {
    LinearSolverCtx transfer_solver;

    PetscCall(PetscMemzero(&transfer_solver, sizeof(transfer_solver)));
    transfer_solver.dm   = ctx->dm;
    transfer_solver.actx = ctx->actx;
    transfer_solver.app  = ctx->app;
    PetscCall(LinearSolverSetupPMGTransferChecks(&transfer_solver, ctx->dm, ctx->app, pmg));
    PetscCall(LinearSolverDestroyPMGHierarchy(&transfer_solver));
  }

  PetscCall(MatCreateVecs(A4, &ctx->tmp4, NULL));
  PetscCall(VecDuplicate(ctx->tmp4, &ctx->r4));
  PetscCall(VecDuplicate(ctx->tmp4, &ctx->z4));
  PetscCall(PMGActiveLayoutDuplicateFull(&ctx->p2_layout, &ctx->p2_rhs_full));
  PetscCall(PMGActiveLayoutDuplicateFull(&ctx->p2_layout, &ctx->p2_x_full));
  PetscCall(PMGActiveLayoutDuplicateFull(&ctx->p2_layout, &ctx->p2_r_full));
  PetscCall(PMGActiveLayoutDuplicateFull(&ctx->p2_layout, &ctx->p2_corr_full));
  PetscCall(PMGActiveLayoutDuplicateFull(&ctx->p1_layout, &ctx->p1_rhs_full));
  PetscCall(PMGActiveLayoutDuplicateFull(&ctx->p1_layout, &ctx->p1_x_full));
  PetscCall(PMGActiveLayoutDuplicateSub(&ctx->p2_layout, &ctx->p2_rhs));
  PetscCall(PMGActiveLayoutDuplicateSub(&ctx->p2_layout, &ctx->p2_x));
  PetscCall(PMGActiveLayoutDuplicateSub(&ctx->p2_layout, &ctx->p2_r));
  PetscCall(PMGActiveLayoutDuplicateSub(&ctx->p2_layout, &ctx->p2_delta));
  PetscCall(PMGActiveLayoutDuplicateSub(&ctx->p1_layout, &ctx->p1_rhs));
  PetscCall(PMGActiveLayoutDuplicateSub(&ctx->p1_layout, &ctx->p1_x));

  PetscCall(KSPCreate(comm, &ctx->smooth4));
  PetscCall(PetscSNPrintf(prefix4, sizeof(prefix4), "pmg_shell_fine_"));
  PetscCall(PMGShellConfigureSmootherKSP(ctx->smooth4, pmg, prefix4, A4));
  PetscCall(ReportPMGLevelSolver(comm, 2, ctx->smooth4));
  if (ctx->p2_layout.active) {
    PetscCall(KSPCreate(ctx->p2_layout.subcomm, &ctx->smooth2));
  }
  if (ctx->p1_layout.active) {
    PetscCall(KSPCreate(ctx->p1_layout.subcomm, &ctx->coarse1));
  }

  PetscCall(VecDestroy(&fine_vec));
  PetscCall(VecDestroy(&p2_vec));
  PetscCall(VecDestroy(&p1_vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellUpdateOperators(PC pc, PMGShellVCycleCtx *ctx, Mat A4)
{
  MPI_Comm       comm = PetscObjectComm((PetscObject)pc);
  const SsrPmgOptions *pmg = ctx->pmg;
  MatReuse       reuse = ctx->operator_updates ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX;
  const char    *reuse_label = ctx->operator_updates ? "reuse" : "initial";
  SsrProfileTimer update_timer, phase_timer;
  PetscLogDouble  sub_time, cat_time, p2_ptap, p1_ptap, p2_redist, p1_redist, update_elapsed;

  PetscFunctionBeginUser;
  PetscCheck(pmg, comm, PETSC_ERR_ARG_NULL, "PMG shell operator update requires resolved PMG options");
  ctx->A4 = A4;
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_OPERATOR_UPDATE, pc, A4, &update_timer);
  PetscCall(KSPSetOperators(ctx->smooth4, A4, A4));

  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_GALERKIN_PRODUCT, A4, ctx->P42, &phase_timer);
  PetscCall(MatPtAP(A4, ctx->P42, reuse, PETSC_DETERMINE, &ctx->A2_active));
  SSR_PROFILE_TIMER_END(NULL, &phase_timer, &p2_ptap);
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_REDISTRIBUTE, ctx->A2_active, NULL, &phase_timer);
  PetscCall(PMGShellRedistributeActiveMatrix(comm, &ctx->p2_layout, ctx->A2_active, reuse, &ctx->A2_sub, &sub_time, &cat_time));
  SSR_PROFILE_TIMER_END(NULL, &phase_timer, &p2_redist);
  if (reuse == MAT_INITIAL_MATRIX) PetscCall(PMGShellAttachSubcommNearNullspace(ctx->dm_p2, &ctx->p2_layout, ctx->A2_sub));
  if (ctx->p2_layout.active) {
    if (!ctx->smooth2) PetscCall(KSPCreate(ctx->p2_layout.subcomm, &ctx->smooth2));
    if (reuse == MAT_INITIAL_MATRIX) {
      PetscCall(PMGShellConfigureSmootherKSP(ctx->smooth2, pmg, "pmg_shell_p2_", ctx->A2_sub));
      PetscCall(ReportPMGLevelSolver(ctx->p2_layout.subcomm, 1, ctx->smooth2));
    } else PetscCall(KSPSetOperators(ctx->smooth2, ctx->A2_sub, ctx->A2_sub));
  }
  PetscCall(PetscPrintf(comm, "PMG_SHELL_OPERATOR_UPDATE level=1 reuse=%s ptap_time=%.6g redistribute_time=%.6g submatrix_time=%.6g concatenate_time=%.6g\n",
                        reuse_label, (double)p2_ptap, (double)p2_redist, (double)sub_time, (double)cat_time));

  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_GALERKIN_PRODUCT, ctx->A2_active, ctx->P21, &phase_timer);
  PetscCall(MatPtAP(ctx->A2_active, ctx->P21, reuse, PETSC_DETERMINE, &ctx->A1_active));
  SSR_PROFILE_TIMER_END(NULL, &phase_timer, &p1_ptap);
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_REDISTRIBUTE, ctx->A1_active, NULL, &phase_timer);
  PetscCall(PMGShellRedistributeActiveMatrix(comm, &ctx->p1_layout, ctx->A1_active, reuse, &ctx->A1_sub, &sub_time, &cat_time));
  SSR_PROFILE_TIMER_END(NULL, &phase_timer, &p1_redist);
  if (reuse == MAT_INITIAL_MATRIX) PetscCall(PMGShellAttachSubcommNearNullspace(ctx->dm_p1, &ctx->p1_layout, ctx->A1_sub));
  if (ctx->p1_layout.active) {
    if (!ctx->coarse1) PetscCall(KSPCreate(ctx->p1_layout.subcomm, &ctx->coarse1));
    if (reuse == MAT_INITIAL_MATRIX) {
      PetscCall(PMGShellConfigureCoarseKSP(ctx->coarse1, pmg, "pmg_shell_p1_", ctx->A1_sub));
      PetscCall(ReportPMGLevelSolver(ctx->p1_layout.subcomm, 0, ctx->coarse1));
    } else PetscCall(KSPSetOperators(ctx->coarse1, ctx->A1_sub, ctx->A1_sub));
  }
  PetscCall(PetscPrintf(comm, "PMG_SHELL_OPERATOR_UPDATE level=0 reuse=%s ptap_time=%.6g redistribute_time=%.6g submatrix_time=%.6g concatenate_time=%.6g\n",
                        reuse_label, (double)p1_ptap, (double)p1_redist, (double)sub_time, (double)cat_time));
  ctx->operator_updates++;
  if (ctx->owner) ctx->owner->pmg_operator_updates++;
  SSR_PROFILE_TIMER_END(NULL, &update_timer, &update_elapsed);
  PetscCall(SsrStatsAccumulateElapsed(&ctx->operator_update_time, update_elapsed));
  if (ctx->owner) PetscCall(SsrStatsAccumulateElapsed(&ctx->owner->pmg_operator_update_time, update_elapsed));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellResidual(Mat A, Vec rhs, Vec x, Vec r, Vec tmp, PetscLogDouble *accum_time, PetscLogDouble *owner_accum_time)
{
  SsrProfileTimer profile_timer;
  PetscLogDouble  elapsed;

  PetscFunctionBeginUser;
  SSR_PROFILE_STAGE_PUSH(SSR_STAGE_PMG_SHELL_RESIDUAL);
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_RESIDUAL, A, x, &profile_timer);
  PetscCall(MatMult(A, x, tmp));
  PetscCall(VecWAXPY(r, -1.0, tmp, rhs));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  PetscCall(SsrStatsAccumulateElapsed(accum_time, elapsed));
  PetscCall(SsrStatsAccumulateElapsed(owner_accum_time, elapsed));
  SSR_PROFILE_STAGE_POP(SSR_STAGE_PMG_SHELL_RESIDUAL);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellDebugCapture(PMGShellVCycleCtx *ctx, Vec src, Vec *dst)
{
  PetscFunctionBeginUser;
  if (!ctx->debug_capture || !src) PetscFunctionReturn(PETSC_SUCCESS);
  if (!*dst) PetscCall(VecDuplicate(src, dst));
  PetscCall(VecCopy(src, *dst));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellVCycleSetUp(PC pc)
{
  PMGShellVCycleCtx *ctx = NULL;
  Mat                A = NULL;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, (void **)&ctx));
  PetscCall(PCGetOperators(pc, NULL, &A));
  PetscCheck(A, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONGSTATE, "PMG shell V-cycle requires a fine operator");
  SSR_PROFILE_BEGIN(NULL, SSR_EVENT_PMG_SETUP, pc, A);
  if (!ctx->setup_done) {
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pc), "PMG_BACKEND backend=shell_vcycle\n"));
    PetscCall(PMGShellCreateHierarchy(pc, ctx, A));
    ctx->setup_done = PETSC_TRUE;
  }
  PetscCall(PMGShellUpdateOperators(pc, ctx, A));
  SSR_PROFILE_END(NULL, SSR_EVENT_PMG_SETUP, pc, A);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellVCycleApply(PC pc, Vec x, Vec y)
{
  PMGShellVCycleCtx *ctx = NULL;
  SsrProfileTimer    profile_timer;
  PetscLogDouble     elapsed;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, (void **)&ctx));
  PetscCall(VecZeroEntries(y));
  SSR_PROFILE_BEGIN(NULL, SSR_EVENT_PMG_APPLY, pc, x);

  SSR_PROFILE_STAGE_PUSH(SSR_STAGE_PMG_SHELL_FINE_SMOOTH);
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_FINE_SMOOTH, ctx->smooth4, x, &profile_timer);
  PetscCall(KSPSolve(ctx->smooth4, x, y));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  PetscCall(SsrStatsAccumulateElapsed(&ctx->fine_smooth_time, elapsed));
  if (ctx->owner) PetscCall(SsrStatsAccumulateElapsed(&ctx->owner->pmg_fine_smooth_time, elapsed));
  PetscCall(PMGShellDebugCapture(ctx, y, &ctx->debug_fine_pre));
  SSR_PROFILE_STAGE_POP(SSR_STAGE_PMG_SHELL_FINE_SMOOTH);

  PetscCall(PMGShellResidual(ctx->A4, x, y, ctx->r4, ctx->tmp4, &ctx->residual_time, ctx->owner ? &ctx->owner->pmg_residual_time : NULL));
  PetscCall(PMGShellDebugCapture(ctx, ctx->r4, &ctx->debug_fine_residual));
  SSR_PROFILE_STAGE_PUSH(SSR_STAGE_PMG_SHELL_TRANSFER);
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_TRANSFER, ctx->P42, ctx->r4, &profile_timer);
  PetscCall(MatMultTranspose(ctx->P42, ctx->r4, ctx->p2_rhs_full));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  PetscCall(SsrStatsAccumulateElapsed(&ctx->restrict_time, elapsed));
  if (ctx->owner) PetscCall(SsrStatsAccumulateElapsed(&ctx->owner->pmg_restrict_time, elapsed));
  PetscCall(PMGActiveLayoutCopyFullToSub(&ctx->p2_layout, ctx->p2_rhs_full, ctx->p2_rhs));
  PetscCall(PMGShellDebugCapture(ctx, ctx->p2_rhs, &ctx->debug_p2_rhs));
  SSR_PROFILE_STAGE_POP(SSR_STAGE_PMG_SHELL_TRANSFER);

  if (ctx->p2_layout.active) {
    PetscCall(VecZeroEntries(ctx->p2_x));
    SSR_PROFILE_STAGE_PUSH(SSR_STAGE_PMG_SHELL_P2);
    SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_P2_SMOOTH, ctx->smooth2, ctx->p2_rhs, &profile_timer);
    PetscCall(KSPSolve(ctx->smooth2, ctx->p2_rhs, ctx->p2_x));
    SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
    PetscCall(SsrStatsAccumulateElapsed(&ctx->p2_smooth_time, elapsed));
    if (ctx->owner) PetscCall(SsrStatsAccumulateElapsed(&ctx->owner->pmg_p2_smooth_time, elapsed));
    PetscCall(PMGShellDebugCapture(ctx, ctx->p2_x, &ctx->debug_p2_pre));
    SSR_PROFILE_STAGE_POP(SSR_STAGE_PMG_SHELL_P2);
    PetscCall(PMGShellResidual(ctx->A2_sub, ctx->p2_rhs, ctx->p2_x, ctx->p2_r, ctx->p2_delta, &ctx->residual_time, ctx->owner ? &ctx->owner->pmg_residual_time : NULL));
    PetscCall(PMGShellDebugCapture(ctx, ctx->p2_r, &ctx->debug_p2_residual));
  }
  SSR_PROFILE_STAGE_PUSH(SSR_STAGE_PMG_SHELL_TRANSFER);
  PetscCall(PMGActiveLayoutCopySubToFull(&ctx->p2_layout, ctx->p2_r, ctx->p2_r_full));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_TRANSFER, ctx->P21, ctx->p2_r_full, &profile_timer);
  PetscCall(MatMultTranspose(ctx->P21, ctx->p2_r_full, ctx->p1_rhs_full));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  PetscCall(SsrStatsAccumulateElapsed(&ctx->restrict_time, elapsed));
  if (ctx->owner) PetscCall(SsrStatsAccumulateElapsed(&ctx->owner->pmg_restrict_time, elapsed));
  PetscCall(PMGActiveLayoutCopyFullToSub(&ctx->p1_layout, ctx->p1_rhs_full, ctx->p1_rhs));
  PetscCall(PMGShellDebugCapture(ctx, ctx->p1_rhs, &ctx->debug_p1_rhs));
  SSR_PROFILE_STAGE_POP(SSR_STAGE_PMG_SHELL_TRANSFER);

  if (ctx->p1_layout.active) {
    PetscCall(VecZeroEntries(ctx->p1_x));
    SSR_PROFILE_STAGE_PUSH(SSR_STAGE_PMG_SHELL_P1);
    SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_COARSE_SOLVE, ctx->coarse1, ctx->p1_rhs, &profile_timer);
    PetscCall(KSPSolve(ctx->coarse1, ctx->p1_rhs, ctx->p1_x));
    SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
    PetscCall(SsrStatsAccumulateElapsed(&ctx->coarse_solve_time, elapsed));
    if (ctx->owner) PetscCall(SsrStatsAccumulateElapsed(&ctx->owner->pmg_coarse_solve_time, elapsed));
    PetscCall(PMGShellDebugCapture(ctx, ctx->p1_x, &ctx->debug_p1_x));
    SSR_PROFILE_STAGE_POP(SSR_STAGE_PMG_SHELL_P1);
  }
  SSR_PROFILE_STAGE_PUSH(SSR_STAGE_PMG_SHELL_TRANSFER);
  PetscCall(PMGActiveLayoutCopySubToFull(&ctx->p1_layout, ctx->p1_x, ctx->p1_x_full));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_TRANSFER, ctx->P21, ctx->p1_x_full, &profile_timer);
  PetscCall(MatMult(ctx->P21, ctx->p1_x_full, ctx->p2_corr_full));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  PetscCall(SsrStatsAccumulateElapsed(&ctx->prolong_time, elapsed));
  if (ctx->owner) PetscCall(SsrStatsAccumulateElapsed(&ctx->owner->pmg_prolong_time, elapsed));
  PetscCall(PMGActiveLayoutCopyFullToSub(&ctx->p2_layout, ctx->p2_corr_full, ctx->p2_delta));
  SSR_PROFILE_STAGE_POP(SSR_STAGE_PMG_SHELL_TRANSFER);
  if (ctx->p2_layout.active) {
    PetscCall(VecAXPY(ctx->p2_x, 1.0, ctx->p2_delta));
    SSR_PROFILE_STAGE_PUSH(SSR_STAGE_PMG_SHELL_P2);
    SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_P2_SMOOTH, ctx->smooth2, ctx->p2_rhs, &profile_timer);
    PetscCall(KSPSolve(ctx->smooth2, ctx->p2_rhs, ctx->p2_x));
    SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
    PetscCall(SsrStatsAccumulateElapsed(&ctx->p2_smooth_time, elapsed));
    if (ctx->owner) PetscCall(SsrStatsAccumulateElapsed(&ctx->owner->pmg_p2_smooth_time, elapsed));
    PetscCall(PMGShellDebugCapture(ctx, ctx->p2_x, &ctx->debug_p2_post));
    SSR_PROFILE_STAGE_POP(SSR_STAGE_PMG_SHELL_P2);
  }
  SSR_PROFILE_STAGE_PUSH(SSR_STAGE_PMG_SHELL_TRANSFER);
  PetscCall(PMGActiveLayoutCopySubToFull(&ctx->p2_layout, ctx->p2_x, ctx->p2_corr_full));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_TRANSFER, ctx->P42, ctx->p2_corr_full, &profile_timer);
  PetscCall(MatMult(ctx->P42, ctx->p2_corr_full, ctx->z4));
  PetscCall(VecAXPY(y, 1.0, ctx->z4));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  PetscCall(SsrStatsAccumulateElapsed(&ctx->prolong_time, elapsed));
  if (ctx->owner) PetscCall(SsrStatsAccumulateElapsed(&ctx->owner->pmg_prolong_time, elapsed));
  SSR_PROFILE_STAGE_POP(SSR_STAGE_PMG_SHELL_TRANSFER);

  SSR_PROFILE_STAGE_PUSH(SSR_STAGE_PMG_SHELL_FINE_SMOOTH);
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_PMG_FINE_SMOOTH, ctx->smooth4, x, &profile_timer);
  PetscCall(KSPSolve(ctx->smooth4, x, y));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  PetscCall(SsrStatsAccumulateElapsed(&ctx->fine_smooth_time, elapsed));
  if (ctx->owner) PetscCall(SsrStatsAccumulateElapsed(&ctx->owner->pmg_fine_smooth_time, elapsed));
  SSR_PROFILE_STAGE_POP(SSR_STAGE_PMG_SHELL_FINE_SMOOTH);
  SSR_PROFILE_END(NULL, SSR_EVENT_PMG_APPLY, pc, x);
  ctx->apply_calls++;
  if (ctx->owner) ctx->owner->pmg_apply_calls++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PMGShellVCycleDestroy(PC pc)
{
  PMGShellVCycleCtx *ctx = NULL;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, (void **)&ctx));
  if (!ctx) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)pc),
                        "PMG_SHELL_APPLY_SUMMARY apply_calls=%" PetscInt_FMT " operator_updates=%" PetscInt_FMT
                        " fine_smooth=%.6g p2_smooth=%.6g restrict=%.6g prolong=%.6g coarse_solve=%.6g residual=%.6g operator_update=%.6g\n",
                        ctx->apply_calls, ctx->operator_updates, (double)ctx->fine_smooth_time, (double)ctx->p2_smooth_time,
                        (double)ctx->restrict_time, (double)ctx->prolong_time, (double)ctx->coarse_solve_time, (double)ctx->residual_time,
                        (double)ctx->operator_update_time));
  PetscCall(KSPDestroy(&ctx->smooth4));
  PetscCall(KSPDestroy(&ctx->smooth2));
  PetscCall(KSPDestroy(&ctx->coarse1));
  PetscCall(MatDestroy(&ctx->P42));
  PetscCall(MatDestroy(&ctx->P21));
  PetscCall(MatDestroy(&ctx->A2_active));
  PetscCall(MatDestroy(&ctx->A1_active));
  PetscCall(MatDestroy(&ctx->A2_sub));
  PetscCall(MatDestroy(&ctx->A1_sub));
  PetscCall(VecDestroy(&ctx->r4));
  PetscCall(VecDestroy(&ctx->tmp4));
  PetscCall(VecDestroy(&ctx->z4));
  PetscCall(VecDestroy(&ctx->p2_rhs_full));
  PetscCall(VecDestroy(&ctx->p2_x_full));
  PetscCall(VecDestroy(&ctx->p2_r_full));
  PetscCall(VecDestroy(&ctx->p2_corr_full));
  PetscCall(VecDestroy(&ctx->p1_rhs_full));
  PetscCall(VecDestroy(&ctx->p1_x_full));
  PetscCall(VecDestroy(&ctx->p2_rhs));
  PetscCall(VecDestroy(&ctx->p2_x));
  PetscCall(VecDestroy(&ctx->p2_r));
  PetscCall(VecDestroy(&ctx->p2_delta));
  PetscCall(VecDestroy(&ctx->p1_rhs));
  PetscCall(VecDestroy(&ctx->p1_x));
  PetscCall(VecDestroy(&ctx->debug_fine_pre));
  PetscCall(VecDestroy(&ctx->debug_fine_residual));
  PetscCall(VecDestroy(&ctx->debug_p2_rhs));
  PetscCall(VecDestroy(&ctx->debug_p2_pre));
  PetscCall(VecDestroy(&ctx->debug_p2_residual));
  PetscCall(VecDestroy(&ctx->debug_p1_rhs));
  PetscCall(VecDestroy(&ctx->debug_p1_x));
  PetscCall(VecDestroy(&ctx->debug_p2_post));
  PetscCall(PMGActiveLayoutDestroy(&ctx->p2_layout));
  PetscCall(PMGActiveLayoutDestroy(&ctx->p1_layout));
  PetscCall(DMDestroy(&ctx->dm_p1));
  PetscCall(DMDestroy(&ctx->dm_p2));
  if (ctx->p1_basis_created) PetscCall(P4BasisDestroy(&ctx->p1_basis));
  if (ctx->p2_basis_created) PetscCall(P4BasisDestroy(&ctx->p2_basis));
  PetscCall(PetscFree(ctx));
  PetscCall(PCShellSetContext(pc, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigurePMGShellVCycle(PC pc, DM dm, AssemblyCtx *actx, AppCtx *app, LinearSolverCtx *solver)
{
  PMGShellVCycleCtx *ctx = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&ctx));
  ctx->app  = app;
  ctx->pmg  = &app->resolved_profile.linear.pmg;
  ctx->dm   = dm;
  ctx->actx = actx;
  ctx->owner = solver;
  ctx->p2_layout.subcomm = MPI_COMM_NULL;
  ctx->p1_layout.subcomm = MPI_COMM_NULL;
  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetContext(pc, ctx));
  PetscCall(PCShellSetName(pc, "pmg_shell_vcycle"));
  PetscCall(PCShellSetSetUp(pc, PMGShellVCycleSetUp));
  PetscCall(PCShellSetApply(pc, PMGShellVCycleApply));
  PetscCall(PCShellSetDestroy(pc, PMGShellVCycleDestroy));
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
  PetscCheck(Nc == dim, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Expected displacement components to match coordinate dimension");
  u[0] = 1.25;
  u[1] = -2.0;
  if (Nc > 2) u[2] = 0.5;
  return PETSC_SUCCESS;
}

static PetscErrorCode PMGTransferAffine(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  (void)dim; (void)time; (void)ctx;
  PetscCheck(Nc == dim, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Expected displacement components to match coordinate dimension");
  u[0] = 1.0 + 2.0 * x[0] - 0.5 * x[1] + (dim > 2 ? 0.25 * x[2] : 0.0);
  u[1] = -0.25 + x[0] + 0.75 * x[1] + (dim > 2 ? -0.1 * x[2] : 0.0);
  if (Nc > 2) u[2] = 0.5 - 0.2 * x[0] + 0.4 * x[1] + 1.3 * x[2];
  return PETSC_SUCCESS;
}

static PetscErrorCode PMGTransferQuadratic(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  (void)dim; (void)time; (void)ctx;
  PetscCheck(Nc == dim, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Expected displacement components to match coordinate dimension");
  u[0] = 1.0 + x[0] * x[0] + (dim > 2 ? 2.0 * x[1] * x[2] : 0.25 * x[1] * x[1]);
  u[1] = -0.3 + x[0] * x[1] + (dim > 2 ? x[2] * x[2] : 0.0);
  if (Nc > 2) u[2] = x[0] * x[2] + x[1] * x[1];
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

static PetscErrorCode LinearSolverSetupPMGTransferChecks(LinearSolverCtx *solver, DM dm, const AppCtx *app, const SsrPmgOptions *pmg)
{
  MPI_Comm comm = PetscObjectComm((PetscObject)dm);
  Vec      v4 = NULL, v2 = NULL, v1 = NULL;
  PetscInt n4, n2, n1;

  PetscFunctionBeginUser;
  PetscCheck(pmg, comm, PETSC_ERR_ARG_NULL, "PMG transfer checks require resolved PMG options");
  PetscCall(LinearSolverDestroyPMGHierarchy(solver));
  PetscCall(P4BasisCreateDegreeDim(PETSC_COMM_SELF, 1, solver->actx->basis->dim, solver->actx->basis->components, &solver->pmg_p1_basis));
  solver->pmg_p1_basis_created = PETSC_TRUE;
  PetscCall(P4BasisCreateDegreeDim(PETSC_COMM_SELF, 2, solver->actx->basis->dim, solver->actx->basis->components, &solver->pmg_p2_basis));
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
  if (pmg->check_coarse_transfers) PetscCall(CheckPMGCoarseTransfers(solver));
  PetscCall(PetscPrintf(comm,
                        "PMG_COARSE_TRANSFER_CONFIG p1_quadrature_points=%" PetscInt_FMT " p2_quadrature_points=%" PetscInt_FMT " p1_global_dofs=%" PetscInt_FMT " p2_global_dofs=%" PetscInt_FMT "\n",
                        solver->pmg_p1_basis.n_qp, solver->pmg_p2_basis.n_qp, n1, n2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ConfigurePMG(PC pc, DM dm, AssemblyCtx *actx, AppCtx *app, LinearSolverCtx *solver)
{
  const SsrPmgOptions *pmg = &app->resolved_profile.linear.pmg;
  P4Basis             p1_basis, p2_basis;
  DM                  dm_p1 = NULL, dm_p2 = NULL;
  Mat                 P21 = NULL, P42 = NULL;
  KSP                 coarse = NULL, smoother = NULL;
  PC                  coarse_pc = NULL, smoother_pc = NULL;
  char                coarse_pc_type[32];
  MPI_Comm            comm = PetscObjectComm((PetscObject)dm);
  PetscMPIInt ranks;
  PetscBool coarse_pc_from_options = PETSC_FALSE, coarse_is_lu = PETSC_FALSE, coarse_is_telescope = PETSC_FALSE, coarse_is_shell = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCheck(pmg, comm, PETSC_ERR_ARG_NULL, "PMG setup requires resolved PMG options");
  PetscCallMPI(MPI_Comm_size(comm, &ranks));
  PetscCall(PetscPrintf(comm, "PMG_BACKEND backend=pcmg\n"));
  PetscCall(P4BasisCreateDegreeDim(PETSC_COMM_SELF, 1, actx->basis->dim, actx->basis->components, &p1_basis));
  PetscCall(P4BasisCreateDegreeDim(PETSC_COMM_SELF, 2, actx->basis->dim, actx->basis->components, &p2_basis));
  PetscCall(CreateSameMeshLevelDM(dm, &p1_basis, app, &dm_p1));
  PetscCall(CreateSameMeshLevelDM(dm, &p2_basis, app, &dm_p2));
  PetscCall(ReportPMGLevelDofs(dm_p1, 0, 1));
  PetscCall(ReportPMGLevelDofs(dm_p2, 1, 2));
  PetscCall(ReportPMGLevelDofs(dm, 2, actx->basis->degree));
  PetscCall(BuildInterpolationMatrix(dm_p2, &p2_basis, dm_p1, &p1_basis, &P21));
  PetscCall(BuildInterpolationMatrix(dm, actx->basis, dm_p2, &p2_basis, &P42));
  PetscCall(SetPMGTelescopeDefaults(pmg, comm));
  PetscCall(SetPMGP2TelescopeDefaults(pmg, comm));
  PetscCall(ChoosePMGCoarsePC(pmg, dm_p1, coarse_pc_type, sizeof(coarse_pc_type)));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-mg_coarse_pc_type", &coarse_pc_from_options));
  PetscCall(PetscStrcasecmp(coarse_pc_type, "lu", &coarse_is_lu));
  PetscCall(PetscStrcasecmp(coarse_pc_type, "telescope", &coarse_is_telescope));
  PetscCall(PetscStrcasecmp(coarse_pc_type, "shell", &coarse_is_shell));
  PetscCheck(!coarse_is_shell, comm, PETSC_ERR_ARG_WRONG,
             "The removed PMG coarse shell experiment is no longer supported; use -pmg_apply_backend shell_vcycle for the maintained shell V-cycle backend");

  PetscCall(PCSetType(pc, PCMG));
  PetscCall(PCMGSetLevels(pc, 3, NULL));
  PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
  PetscCall(PCMGSetCycleType(pc, PC_MG_CYCLE_V));
  PetscCall(PCMGSetInterpolation(pc, 1, P21));
  PetscCall(PCMGSetInterpolation(pc, 2, P42));
  PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));
  PetscCall(PetscPrintf(comm, "PMG_COARSE_OPERATOR_CONFIG type=galerkin p1_quadrature_points=%" PetscInt_FMT " p2_quadrature_points=%" PetscInt_FMT "\n",
                        p1_basis.n_qp, p2_basis.n_qp));
  if (pmg->check_coarse_transfers) PetscCall(LinearSolverSetupPMGTransferChecks(solver, dm, app, pmg));

  PetscCall(PCMGGetCoarseSolve(pc, &coarse));
  PetscCall(KSPGetPC(coarse, &coarse_pc));
  if (!coarse_pc_from_options && !coarse_is_lu && pmg->coarse_redundant_group_size > 0 && ranks > pmg->coarse_redundant_group_size) {
    KSP      inner_ksp;
    PC       inner_pc;
    PetscInt copies = ((PetscInt)ranks + pmg->coarse_redundant_group_size - 1) / pmg->coarse_redundant_group_size;

    PetscCall(KSPSetType(coarse, KSPPREONLY));
    PetscCall(PCSetType(coarse_pc, PCREDUNDANT));
    PetscCall(PCRedundantSetNumber(coarse_pc, copies));
    PetscCall(PCRedundantGetKSP(coarse_pc, &inner_ksp));
    PetscCall(KSPSetType(inner_ksp, pmg->coarse_ksp_type));
    PetscCall(KSPSetTolerances(inner_ksp, pmg->coarse_ksp_rtol, PETSC_DEFAULT, PETSC_DEFAULT, pmg->coarse_ksp_max_it));
    PetscCall(KSPGetPC(inner_ksp, &inner_pc));
    PetscCall(ConfigurePMGBasePC(inner_pc, coarse_pc_type, pmg->coarse_gamg_aggressive_square_graph));
    PetscCall(PetscPrintf(comm, "PMG_REDUNDANT_CONFIG enabled=true group_size=%" PetscInt_FMT " copies=%" PetscInt_FMT " inner_ksp=%s inner_pc=%s\n",
                          pmg->coarse_redundant_group_size, copies, pmg->coarse_ksp_type, coarse_pc_type));
    PetscCall(PetscPrintf(comm,
                          "PMG_COARSE_SOLVE type=redundant group_size=%" PetscInt_FMT " copies=%" PetscInt_FMT " inner_pc=%s aggressive_square_graph=%s\n",
                          pmg->coarse_redundant_group_size, copies, coarse_pc_type, pmg->coarse_gamg_aggressive_square_graph ? "true" : "false"));
  } else {
    PetscCall(PetscPrintf(comm, "PMG_REDUNDANT_CONFIG enabled=false group_size=%" PetscInt_FMT " copies=0 inner_ksp=none inner_pc=none\n",
                          pmg->coarse_redundant_group_size));
    if (coarse_is_lu || coarse_is_telescope) {
      PetscCall(KSPSetType(coarse, KSPPREONLY));
    } else {
      PetscBool coarse_is_gmres_type = PETSC_FALSE, coarse_is_fgmres_type = PETSC_FALSE;

      PetscCall(KSPSetType(coarse, pmg->coarse_ksp_type));
      PetscCall(KSPSetTolerances(coarse, pmg->coarse_ksp_rtol, PETSC_DEFAULT, PETSC_DEFAULT, pmg->coarse_ksp_max_it));
      PetscCall(PetscStrcasecmp(pmg->coarse_ksp_type, KSPGMRES, &coarse_is_gmres_type));
      PetscCall(PetscStrcasecmp(pmg->coarse_ksp_type, KSPFGMRES, &coarse_is_fgmres_type));
      if (coarse_is_gmres_type || coarse_is_fgmres_type) PetscCall(KSPGMRESSetRestart(coarse, pmg->coarse_ksp_max_it));
    }
    PetscCall(ConfigurePMGBasePC(coarse_pc, coarse_pc_type, pmg->coarse_gamg_aggressive_square_graph));
    PetscCall(PetscPrintf(comm, "PMG_COARSE_SOLVE type=%s group_size=%" PetscInt_FMT " aggressive_square_graph=%s\n", coarse_pc_type,
                          pmg->coarse_redundant_group_size, pmg->coarse_gamg_aggressive_square_graph ? "true" : "false"));
  }
  for (PetscInt level = 1; level < 3; ++level) {
    PetscCall(PCMGGetSmoother(pc, level, &smoother));
    PetscCall(KSPSetType(smoother, pmg->smoother_ksp_type));
    PetscCall(KSPSetTolerances(smoother, 0.0, 0.0, PETSC_CURRENT, pmg->smoother_max_it));
    PetscCall(KSPGetPC(smoother, &smoother_pc));
    PetscCall(PCSetType(smoother_pc, pmg->smoother_pc_type));
  }

  PetscCall(MatDestroy(&P21));
  PetscCall(MatDestroy(&P42));
  PetscCall(DMDestroy(&dm_p1));
  PetscCall(DMDestroy(&dm_p2));
  PetscCall(P4BasisDestroy(&p1_basis));
  PetscCall(P4BasisDestroy(&p2_basis));
  PetscFunctionReturn(PETSC_SUCCESS);
}
