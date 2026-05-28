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
      PetscBool use_shell_vcycle = PETSC_FALSE;

      PetscCall(PMGApplyBackendIsShell(app, &use_shell_vcycle));
      if (use_shell_vcycle) PetscCall(ConfigurePMGShellVCycle(pc, dm, actx, app));
      else PetscCall(ConfigurePMG(pc, dm, actx, app, solver));
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
static PetscErrorCode LinearSolverClearRecycleBasis(LinearSolverCtx *solver);
static PetscErrorCode LinearSolverPrepareKSP(LinearSolverCtx *solver, PetscBool nonlinear_tangent, KSP *ksp_out, PetscBool *owned_out);
static PetscErrorCode LinearSolverSolvePrepared(LinearSolverCtx *solver, KSP ksp, Vec rhs, Vec x, const char *label, PetscBool nonlinear_tangent, PetscBool basis_already_orthogonalized, PetscInt *its);
static PetscErrorCode LinearSolverFinishPreparedKSP(KSP *ksp, PetscBool owned);

static PetscErrorCode LinearSolverInit(LinearSolverCtx *solver, DM dm, AssemblyCtx *actx, AppCtx *app, Mat A)
{
  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(solver, sizeof(*solver)));
  solver->dm    = dm;
  solver->actx  = actx;
  solver->app   = app;
  solver->A     = A;
  solver->reuse = app->reuse_linear_solver;
  solver->recycle_temp_start_raw = -1;
  solver->force_reuse_preconditioner = PETSC_FALSE;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "LINEAR_SOLVER_REUSE enabled=%s\n", solver->reuse ? "true" : "false"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverDestroy(LinearSolverCtx *solver)
{
  PetscFunctionBeginUser;
  PetscCall(LinearSolverClearOrthBasis(solver));
  PetscCall(LinearSolverClearRecycleBasis(solver));
  for (PetscInt i = 0; i < solver->n_raw_basis; ++i) PetscCall(VecDestroy(&solver->raw_basis[i]));
  PetscCall(PetscFree(solver->raw_basis));
  PetscCall(PetscFree(solver->raw_basis_tol));
  PetscCall(PetscFree(solver->orth_basis));
  PetscCall(PetscFree(solver->left_basis));
  PetscCall(PetscFree(solver->Aorth_basis));
  PetscCall(PetscFree(solver->recycle_basis));
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
  for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
    PetscCall(VecDestroy(&solver->orth_basis[i]));
    PetscCall(VecDestroy(&solver->left_basis[i]));
    PetscCall(VecDestroy(&solver->Aorth_basis[i]));
  }
  solver->n_orth_basis = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static Vec LinearSolverTestBasis(const LinearSolverCtx *solver, PetscInt i)
{
  if (solver->app->deflation_projector == DEFLATION_PROJECTOR_BIORTHOGONAL && solver->left_basis && solver->left_basis[i]) return solver->left_basis[i];
  return solver->orth_basis[i];
}

static PetscErrorCode LinearSolverAppendRawBasisWithTol(LinearSolverCtx *solver, Vec v, const char label[], PetscReal basis_tol)
{
  Vec copy = NULL;

  PetscFunctionBeginUser;
  if (!solver->app->use_deflation) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecDuplicate(v, &copy));
  PetscCall(VecCopy(v, copy));
  if (solver->raw_basis_cap == solver->n_raw_basis) {
    const PetscInt new_cap = solver->raw_basis_cap ? 2 * solver->raw_basis_cap : 8;

    PetscCall(PetscRealloc((size_t)new_cap * sizeof(Vec), &solver->raw_basis));
    PetscCall(PetscRealloc((size_t)new_cap * sizeof(PetscReal), &solver->raw_basis_tol));
    for (PetscInt i = solver->raw_basis_cap; i < new_cap; ++i) solver->raw_basis[i] = NULL;
    solver->raw_basis_cap = new_cap;
  }
  solver->raw_basis[solver->n_raw_basis]     = copy;
  solver->raw_basis_tol[solver->n_raw_basis] = basis_tol;
  solver->n_raw_basis++;
  copy                                = NULL;
  if (solver->app->deflation_max_vectors > 0 && solver->n_raw_basis > solver->app->deflation_max_vectors) {
    PetscCall(VecDestroy(&solver->raw_basis[0]));
    for (PetscInt i = 1; i < solver->n_raw_basis; ++i) {
      solver->raw_basis[i - 1]     = solver->raw_basis[i];
      solver->raw_basis_tol[i - 1] = solver->raw_basis_tol[i];
    }
    solver->raw_basis[--solver->n_raw_basis] = NULL;
  }
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_BASIS_ADD label=\"%s\" raw_cols=%" PetscInt_FMT " max_cols=%" PetscInt_FMT " basis_tol=%.6e\n",
                        label, solver->n_raw_basis, solver->app->deflation_max_vectors, (double)basis_tol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverAppendRawBasis(LinearSolverCtx *solver, Vec v, const char label[])
{
  PetscFunctionBeginUser;
  PetscCall(LinearSolverAppendRawBasisWithTol(solver, v, label, solver->app->deflation_basis_tol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverTruncateRawBasis(LinearSolverCtx *solver, PetscInt n_keep)
{
  PetscFunctionBeginUser;
  if (n_keep < 0) n_keep = 0;
  if (n_keep >= solver->n_raw_basis) PetscFunctionReturn(PETSC_SUCCESS);
  for (PetscInt i = n_keep; i < solver->n_raw_basis; ++i) PetscCall(VecDestroy(&solver->raw_basis[i]));
  solver->n_raw_basis = n_keep;
  for (PetscInt i = n_keep; i < solver->raw_basis_cap; ++i) {
    if (solver->raw_basis) solver->raw_basis[i] = NULL;
  }
  solver->recycle_temp_basis_active = PETSC_FALSE;
  solver->recycle_temp_start_raw    = -1;
  PetscCall(LinearSolverClearOrthBasis(solver));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_BASIS_RESTORE raw_cols=%" PetscInt_FMT "\n", solver->n_raw_basis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverClearRecycleBasis(LinearSolverCtx *solver)
{
  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < solver->n_recycle_basis; ++i) PetscCall(VecDestroy(&solver->recycle_basis[i]));
  solver->n_recycle_basis = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverCaptureRecycleVector(LinearSolverCtx *solver, Vec v)
{
  Vec copy = NULL;

  PetscFunctionBeginUser;
  if (!solver->capture_recycle_basis) PetscFunctionReturn(PETSC_SUCCESS);
  if (solver->app->deflation_recycle_max_vectors > 0 && solver->n_recycle_basis >= solver->app->deflation_recycle_max_vectors) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecDuplicate(v, &copy));
  PetscCall(VecCopy(v, copy));
  if (solver->recycle_basis_cap == solver->n_recycle_basis) {
    const PetscInt new_cap = solver->recycle_basis_cap ? 2 * solver->recycle_basis_cap : 8;

    PetscCall(PetscRealloc((size_t)new_cap * sizeof(Vec), &solver->recycle_basis));
    for (PetscInt i = solver->recycle_basis_cap; i < new_cap; ++i) solver->recycle_basis[i] = NULL;
    solver->recycle_basis_cap = new_cap;
  }
  solver->recycle_basis[solver->n_recycle_basis++] = copy;
  copy = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverAppendPersistentKrylovVector(LinearSolverCtx *solver, Vec v, const char solve_label[], PetscInt krylov_it)
{
  char label[256];

  PetscFunctionBeginUser;
  if (!solver->app->deflation_krylov_persistent) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSNPrintf(label, sizeof(label), "%s persistent Krylov direction %" PetscInt_FMT, solve_label, krylov_it + 1));
  PetscCall(LinearSolverAppendRawBasisWithTol(solver, v, label, solver->app->deflation_krylov_basis_tol));
  solver->deflation_krylov_persistent_added++;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_KRYLOV_PERSISTENT_ADD solve=\"%s\" krylov_it=%" PetscInt_FMT " raw_cols=%" PetscInt_FMT " basis_tol=%.6e total_added=%" PetscInt_FMT "\n",
                        solve_label, krylov_it + 1, solver->n_raw_basis, (double)solver->app->deflation_krylov_basis_tol,
                        solver->deflation_krylov_persistent_added));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverBeginRecycleCapture(LinearSolverCtx *solver)
{
  PetscFunctionBeginUser;
  PetscCall(LinearSolverClearRecycleBasis(solver));
  solver->capture_recycle_basis = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverEndRecycleCapture(LinearSolverCtx *solver)
{
  PetscFunctionBeginUser;
  solver->capture_recycle_basis = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverAppendTemporaryRecycleBasis(LinearSolverCtx *solver, const char label[], PetscInt *added)
{
  PetscFunctionBeginUser;
  *added = 0;
  solver->recycle_temp_start_raw    = solver->n_raw_basis;
  solver->recycle_temp_basis_active = (PetscBool)(solver->n_recycle_basis > 0);
  for (PetscInt i = 0; i < solver->n_recycle_basis; ++i) {
    PetscCall(LinearSolverAppendRawBasisWithTol(solver, solver->recycle_basis[i], label, solver->app->deflation_recycle_basis_tol));
    ++(*added);
  }
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_INTRANEWTON_TEMP label=\"%s\" captured=%" PetscInt_FMT " added=%" PetscInt_FMT " max=%" PetscInt_FMT " temp_start_raw=%" PetscInt_FMT " recycle_tol=%.6e\n",
                        label, solver->n_recycle_basis, *added, solver->app->deflation_recycle_max_vectors,
                        solver->recycle_temp_start_raw, (double)solver->app->deflation_recycle_basis_tol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverStoreOrthVector(LinearSolverCtx *solver, Vec v, Vec Av, Vec w)
{
  Vec copy = NULL, Acopy = NULL, wcopy = NULL;

  PetscFunctionBeginUser;
  if (solver->orth_basis_cap == solver->n_orth_basis) {
    const PetscInt new_cap = solver->orth_basis_cap ? 2 * solver->orth_basis_cap : PetscMax(1, solver->n_raw_basis);

    PetscCall(PetscRealloc((size_t)new_cap * sizeof(Vec), &solver->orth_basis));
    PetscCall(PetscRealloc((size_t)new_cap * sizeof(Vec), &solver->left_basis));
    PetscCall(PetscRealloc((size_t)new_cap * sizeof(Vec), &solver->Aorth_basis));
    for (PetscInt i = solver->orth_basis_cap; i < new_cap; ++i) {
      solver->orth_basis[i]  = NULL;
      solver->left_basis[i]  = NULL;
      solver->Aorth_basis[i] = NULL;
    }
    solver->orth_basis_cap = new_cap;
  }
  PetscCall(VecDuplicate(v, &copy));
  PetscCall(VecCopy(v, copy));
  if (w) {
    PetscCall(VecDuplicate(w, &wcopy));
    PetscCall(VecCopy(w, wcopy));
  }
  PetscCall(VecDuplicate(Av, &Acopy));
  PetscCall(VecCopy(Av, Acopy));
  solver->orth_basis[solver->n_orth_basis]  = copy;
  solver->left_basis[solver->n_orth_basis]   = wcopy;
  solver->Aorth_basis[solver->n_orth_basis] = Acopy;
  solver->n_orth_basis++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverDropOrthVector(LinearSolverCtx *solver, PetscInt idx)
{
  PetscFunctionBeginUser;
  PetscCheck(idx >= 0 && idx < solver->n_orth_basis, PetscObjectComm((PetscObject)solver->dm), PETSC_ERR_ARG_OUTOFRANGE,
             "Invalid orthogonal basis index %" PetscInt_FMT " for %" PetscInt_FMT " columns", idx, solver->n_orth_basis);
  PetscCall(VecDestroy(&solver->orth_basis[idx]));
  PetscCall(VecDestroy(&solver->left_basis[idx]));
  PetscCall(VecDestroy(&solver->Aorth_basis[idx]));
  for (PetscInt i = idx + 1; i < solver->n_orth_basis; ++i) {
    solver->orth_basis[i - 1]  = solver->orth_basis[i];
    solver->left_basis[i - 1]  = solver->left_basis[i];
    solver->Aorth_basis[i - 1] = solver->Aorth_basis[i];
  }
  solver->n_orth_basis--;
  solver->orth_basis[solver->n_orth_basis]  = NULL;
  solver->left_basis[solver->n_orth_basis]  = NULL;
  solver->Aorth_basis[solver->n_orth_basis] = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverRenormalizeOrthVector(LinearSolverCtx *solver, PetscInt idx, PetscInt *dropped)
{
  PetscScalar norm_scalar;
  PetscReal   norm_a;

  PetscFunctionBeginUser;
  *dropped = 0;
  PetscCall(VecDot(solver->orth_basis[idx], solver->Aorth_basis[idx], &norm_scalar));
  norm_a = PetscRealPart(norm_scalar);
  if (norm_a <= 0.0) {
    PetscCall(LinearSolverDropOrthVector(solver, idx));
    *dropped = 1;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  {
    const PetscReal inv_norm = 1.0 / PetscSqrtReal(norm_a);

    PetscCall(VecScale(solver->orth_basis[idx], inv_norm));
    PetscCall(VecScale(solver->Aorth_basis[idx], inv_norm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverReorthogonalizeBasis(LinearSolverCtx *solver, const char label[])
{
  PetscInt dropped_total = 0;

  PetscFunctionBeginUser;
  if (solver->app->deflation_reorthogonalize_sweeps <= 0 || solver->n_orth_basis <= 1) PetscFunctionReturn(PETSC_SUCCESS);
  for (PetscInt sweep = 0; sweep < solver->app->deflation_reorthogonalize_sweeps; ++sweep) {
    PetscInt dropped_sweep = 0;

    for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
      for (PetscInt j = 0; j < i; ++j) {
        PetscScalar coeff;

        PetscCall(VecDot(solver->orth_basis[j], solver->Aorth_basis[i], &coeff));
        PetscCall(VecAXPY(solver->orth_basis[i], -coeff, solver->orth_basis[j]));
        PetscCall(VecAXPY(solver->Aorth_basis[i], -coeff, solver->Aorth_basis[j]));
      }
      {
        PetscInt dropped = 0;

        PetscCall(LinearSolverRenormalizeOrthVector(solver, i, &dropped));
        if (dropped) {
          ++dropped_sweep;
          --i;
        }
      }
    }
    for (PetscInt i = solver->n_orth_basis; i-- > 0;) {
      for (PetscInt j = solver->n_orth_basis; j-- > i + 1;) {
        PetscScalar coeff;

        PetscCall(VecDot(solver->orth_basis[j], solver->Aorth_basis[i], &coeff));
        PetscCall(VecAXPY(solver->orth_basis[i], -coeff, solver->orth_basis[j]));
        PetscCall(VecAXPY(solver->Aorth_basis[i], -coeff, solver->Aorth_basis[j]));
      }
      {
        PetscInt dropped = 0;

        PetscCall(LinearSolverRenormalizeOrthVector(solver, i, &dropped));
        if (dropped) ++dropped_sweep;
      }
    }
    dropped_total += dropped_sweep;
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                          "DEFLATION_REORTHO_SWEEP label=\"%s\" sweep=%" PetscInt_FMT " cols=%" PetscInt_FMT " dropped=%" PetscInt_FMT "\n",
                          label, sweep + 1, solver->n_orth_basis, dropped_sweep));
  }
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_REORTHO_SUMMARY label=\"%s\" sweeps=%" PetscInt_FMT " cols=%" PetscInt_FMT " dropped=%" PetscInt_FMT "\n",
                        label, solver->app->deflation_reorthogonalize_sweeps, solver->n_orth_basis, dropped_total));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverCheckAOrthonormality(LinearSolverCtx *solver, const char label[])
{
  PetscReal max_diag_err = 0.0, max_offdiag = 0.0, max_lower = 0.0, max_upper = 0.0, max_asym = 0.0;
  PetscReal min_diag = PETSC_MAX_REAL, max_diag = -PETSC_MAX_REAL;
  PetscInt  max_diag_i = -1, max_off_i = -1, max_off_j = -1, max_asym_i = -1, max_asym_j = -1;
  const char *mode = solver->app->deflation_projector == DEFLATION_PROJECTOR_BIORTHOGONAL ? "biorthogonal" : "a_orthonormal";

  PetscFunctionBeginUser;
  if (!solver->app->deflation_check_orthonormality) PetscFunctionReturn(PETSC_SUCCESS);
  for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
    PetscScalar gii;
    PetscReal   diag, diag_err;

    PetscCall(VecDot(LinearSolverTestBasis(solver, i), solver->Aorth_basis[i], &gii));
    diag     = PetscRealPart(gii);
    diag_err = PetscAbsScalar(gii - 1.0);
    if (diag < min_diag) min_diag = diag;
    if (diag > max_diag) max_diag = diag;
    if (diag_err > max_diag_err) {
      max_diag_err = diag_err;
      max_diag_i   = i;
    }
    for (PetscInt j = i + 1; j < solver->n_orth_basis; ++j) {
      PetscScalar gij, gji;
      PetscReal   abs_ij, abs_ji, asym;

      PetscCall(VecDot(LinearSolverTestBasis(solver, i), solver->Aorth_basis[j], &gij));
      PetscCall(VecDot(LinearSolverTestBasis(solver, j), solver->Aorth_basis[i], &gji));
      abs_ij = PetscAbsScalar(gij);
      abs_ji = PetscAbsScalar(gji);
      asym   = PetscAbsScalar(gij - gji);
      if (abs_ij > max_upper) max_upper = abs_ij;
      if (abs_ji > max_lower) max_lower = abs_ji;
      if (abs_ij > max_offdiag) {
        max_offdiag = abs_ij;
        max_off_i   = i;
        max_off_j   = j;
      }
      if (abs_ji > max_offdiag) {
        max_offdiag = abs_ji;
        max_off_i   = j;
        max_off_j   = i;
      }
      if (asym > max_asym) {
        max_asym   = asym;
        max_asym_i = i;
        max_asym_j = j;
      }
    }
  }
  if (!solver->n_orth_basis) {
    min_diag = 0.0;
    max_diag = 0.0;
  }
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_GRAM_CHECK label=\"%s\" mode=%s cols=%" PetscInt_FMT " max_diag_err=%.6e max_offdiag=%.6e max_lower=%.6e max_upper=%.6e max_asym=%.6e min_diag=%.6e max_diag=%.6e max_diag_i=%" PetscInt_FMT " max_off_i=%" PetscInt_FMT " max_off_j=%" PetscInt_FMT " max_asym_i=%" PetscInt_FMT " max_asym_j=%" PetscInt_FMT " warn_tol=%.6e ok=%s\n",
                        label, mode, solver->n_orth_basis, (double)max_diag_err, (double)max_offdiag, (double)max_lower,
                        (double)max_upper, (double)max_asym, (double)min_diag, (double)max_diag, max_diag_i, max_off_i,
                        max_off_j, max_asym_i, max_asym_j, (double)solver->app->deflation_orthonormality_warn_tol,
                        (max_diag_err <= solver->app->deflation_orthonormality_warn_tol &&
                         max_offdiag <= solver->app->deflation_orthonormality_warn_tol) ? "true" : "false"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverBiorthogonalizeBasis(LinearSolverCtx *solver, Mat A, const char label[])
{
  Vec            v = NULL, w = NULL, Av = NULL;
  PetscInt       skipped_small = 0, skipped_pivot = 0;
  PetscLogDouble t0, t1;
  const PetscInt passes = PetscMax((PetscInt)1, solver->app->deflation_reorthogonalize_sweeps + 1);

  PetscFunctionBeginUser;
  PetscCall(PetscLogStagePush(log_stage_deflation_orthogonalize));
  PetscCall(PetscTime(&t0));
  PetscCall(LinearSolverClearOrthBasis(solver));
  if (!solver->n_raw_basis) {
    PetscCall(PetscTime(&t1));
    solver->deflation_orthogonalization_time += t1 - t0;
    PetscCall(PetscLogStagePop());
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecDuplicate(solver->raw_basis[0], &v));
  PetscCall(VecDuplicate(solver->raw_basis[0], &w));
  PetscCall(VecDuplicate(solver->raw_basis[0], &Av));

  /*
    Build an oblique projector basis V,W with W^T A V = I.  V is the
    correction space used in the coarse component; W is only the test space
    used for coefficients.  As in the one-sided path, recent raw vectors get
    priority when near-dependent candidates are skipped.
  */
  for (PetscInt src = solver->n_raw_basis; src-- > 0;) {
    PetscScalar gamma;
    PetscReal   abs_gamma, basis_tol, w_norm, Av_norm, pivot_tol, pivot_threshold;

    PetscCall(VecCopy(solver->raw_basis[src], v));
    PetscCall(VecCopy(solver->raw_basis[src], w));
    PetscCall(MatMult(A, v, Av));

    for (PetscInt pass = 0; pass < passes; ++pass) {
      for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
        PetscScalar alpha, beta;

        PetscCall(VecDot(LinearSolverTestBasis(solver, i), Av, &alpha));
        PetscCall(VecAXPY(v, -alpha, solver->orth_basis[i]));
        PetscCall(VecAXPY(Av, -alpha, solver->Aorth_basis[i]));
        PetscCall(VecDot(w, solver->Aorth_basis[i], &beta));
        PetscCall(VecAXPY(w, -beta, LinearSolverTestBasis(solver, i)));
      }
    }

    PetscCall(VecDot(w, Av, &gamma));
    PetscCall(VecNorm(w, NORM_2, &w_norm));
    PetscCall(VecNorm(Av, NORM_2, &Av_norm));
    abs_gamma = PetscAbsScalar(gamma);
    basis_tol = solver->raw_basis_tol ? solver->raw_basis_tol[src] : solver->app->deflation_basis_tol;
    pivot_tol = solver->app->deflation_biorthogonal_pivot_tol * w_norm * Av_norm;
    pivot_threshold = PetscMax(PetscMax(basis_tol, pivot_tol), 1.0e-30);
    if (PetscIsInfOrNanReal(abs_gamma) || abs_gamma <= pivot_threshold) {
      ++skipped_small;
      if (abs_gamma <= PetscMax(pivot_tol, 1.0e-30)) ++skipped_pivot;
      continue;
    }
    {
      const PetscReal sign = PetscRealPart(gamma) < 0.0 ? -1.0 : 1.0;
      const PetscReal scale_right = sign / PetscSqrtReal(abs_gamma);
      const PetscReal scale_left  = 1.0 / PetscSqrtReal(abs_gamma);

      PetscCall(VecScale(v, scale_right));
      PetscCall(VecScale(Av, scale_right));
      PetscCall(VecScale(w, scale_left));
    }
    PetscCall(LinearSolverStoreOrthVector(solver, v, Av, w));
  }
  for (PetscInt i = 0; i < solver->n_orth_basis / 2; ++i) {
    Vec tmp = solver->orth_basis[i];
    Vec wtmp = solver->left_basis[i];
    Vec Atmp = solver->Aorth_basis[i];

    solver->orth_basis[i] = solver->orth_basis[solver->n_orth_basis - 1 - i];
    solver->orth_basis[solver->n_orth_basis - 1 - i] = tmp;
    solver->left_basis[i] = solver->left_basis[solver->n_orth_basis - 1 - i];
    solver->left_basis[solver->n_orth_basis - 1 - i] = wtmp;
    solver->Aorth_basis[i] = solver->Aorth_basis[solver->n_orth_basis - 1 - i];
    solver->Aorth_basis[solver->n_orth_basis - 1 - i] = Atmp;
  }
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&w));
  PetscCall(VecDestroy(&Av));
  PetscCall(PetscTime(&t1));
  solver->deflation_orthogonalization_time += t1 - t0;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_BIORTHO label=\"%s\" raw_cols=%" PetscInt_FMT " orth_cols=%" PetscInt_FMT " skipped_small=%" PetscInt_FMT " skipped_pivot=%" PetscInt_FMT " passes=%" PetscInt_FMT " default_tol=%.6e pivot_tol=%.6e recycle_active=%s recycle_start=%" PetscInt_FMT " recycle_tol=%.6e krylov_tol=%.6e time=%.6g\n",
                        label, solver->n_raw_basis, solver->n_orth_basis, skipped_small, skipped_pivot, passes,
                        (double)solver->app->deflation_basis_tol, (double)solver->app->deflation_biorthogonal_pivot_tol,
                        solver->recycle_temp_basis_active ? "true" : "false", solver->recycle_temp_start_raw, (double)solver->app->deflation_recycle_basis_tol,
                        (double)solver->app->deflation_krylov_basis_tol, (double)(t1 - t0)));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_CACHE_CONFIG orth_cols=%" PetscInt_FMT " cached_A_cols=%" PetscInt_FMT " left_cols=%" PetscInt_FMT " projector=biorthogonal\n",
                        solver->n_orth_basis, solver->n_orth_basis, solver->n_orth_basis));
  PetscCall(LinearSolverCheckAOrthonormality(solver, label));
  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverAOrthogonalizeBasis(LinearSolverCtx *solver, Mat A, const char label[])
{
  Vec            v = NULL, Av = NULL;
  PetscInt       skipped_small = 0, skipped_nonpositive = 0;
  PetscLogDouble t0, t1;

  PetscFunctionBeginUser;
  if (solver->app->deflation_projector == DEFLATION_PROJECTOR_BIORTHOGONAL) {
    PetscCall(LinearSolverBiorthogonalizeBasis(solver, A, label));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscLogStagePush(log_stage_deflation_orthogonalize));
  PetscCall(PetscTime(&t0));
  PetscCall(LinearSolverClearOrthBasis(solver));
  if (!solver->n_raw_basis) {
    PetscCall(PetscTime(&t1));
    solver->deflation_orthogonalization_time += t1 - t0;
    PetscCall(PetscLogStagePop());
    PetscFunctionReturn(PETSC_SUCCESS);
  }
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
    PetscCall(MatMult(A, v, Av));
    for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
      PetscScalar coeff;

      PetscCall(VecDot(solver->orth_basis[i], Av, &coeff));
      PetscCall(VecAXPY(v, -coeff, solver->orth_basis[i]));
      PetscCall(VecAXPY(Av, -coeff, solver->Aorth_basis[i]));
    }
    PetscCall(VecDot(v, Av, &norm_scalar));
    norm_a = PetscRealPart(norm_scalar);
    if (norm_a <= 0.0) {
      ++skipped_nonpositive;
      continue;
    }
    {
      const PetscReal basis_tol = solver->raw_basis_tol ? solver->raw_basis_tol[src] : solver->app->deflation_basis_tol;

      if (norm_a <= basis_tol) {
        ++skipped_small;
        continue;
      }
    }
    {
      const PetscReal inv_norm = 1.0 / PetscSqrtReal(norm_a);

      PetscCall(VecScale(v, inv_norm));
      PetscCall(VecScale(Av, inv_norm));
    }
    PetscCall(LinearSolverStoreOrthVector(solver, v, Av, NULL));
  }
  for (PetscInt i = 0; i < solver->n_orth_basis / 2; ++i) {
    Vec tmp = solver->orth_basis[i];
    Vec Atmp = solver->Aorth_basis[i];

    solver->orth_basis[i] = solver->orth_basis[solver->n_orth_basis - 1 - i];
    solver->orth_basis[solver->n_orth_basis - 1 - i] = tmp;
    solver->Aorth_basis[i] = solver->Aorth_basis[solver->n_orth_basis - 1 - i];
    solver->Aorth_basis[solver->n_orth_basis - 1 - i] = Atmp;
  }
  PetscCall(LinearSolverReorthogonalizeBasis(solver, label));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&Av));
  PetscCall(PetscTime(&t1));
  solver->deflation_orthogonalization_time += t1 - t0;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_ORTHO label=\"%s\" raw_cols=%" PetscInt_FMT " orth_cols=%" PetscInt_FMT " skipped_small=%" PetscInt_FMT " skipped_nonpositive=%" PetscInt_FMT " default_tol=%.6e recycle_active=%s recycle_start=%" PetscInt_FMT " recycle_tol=%.6e krylov_tol=%.6e time=%.6g\n",
                        label, solver->n_raw_basis, solver->n_orth_basis, skipped_small, skipped_nonpositive,
                        (double)solver->app->deflation_basis_tol, solver->recycle_temp_basis_active ? "true" : "false",
                        solver->recycle_temp_start_raw, (double)solver->app->deflation_recycle_basis_tol,
                        (double)solver->app->deflation_krylov_basis_tol, (double)(t1 - t0)));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_CACHE_CONFIG orth_cols=%" PetscInt_FMT " cached_A_cols=%" PetscInt_FMT " left_cols=0 projector=a_orthonormal\n",
                        solver->n_orth_basis, solver->n_orth_basis));
  PetscCall(LinearSolverCheckAOrthonormality(solver, label));
  PetscCall(PetscLogStagePop());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DeflationCoarseInitialGuess(LinearSolverCtx *solver, Mat A, Vec rhs, Vec x, Vec r, Vec work)
{
  PetscLogDouble t0, t1;

  PetscFunctionBeginUser;
  PetscCall(PetscLogStagePush(log_stage_deflation_initial_guess));
  PetscCall(PetscTime(&t0));
  PetscCall(VecZeroEntries(x));
  PetscCall(VecCopy(rhs, r));
  for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
    PetscScalar coeff;

    PetscCall(VecDot(LinearSolverTestBasis(solver, i), rhs, &coeff));
    PetscCall(VecAXPY(x, coeff, solver->orth_basis[i]));
    PetscCall(VecAXPY(r, -coeff, solver->Aorth_basis[i]));
  }
  PetscCall(PetscTime(&t1));
  solver->deflation_coarse_time += t1 - t0;
  ++solver->deflation_coarse_calls;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATION_COARSE_INITIAL basis_cols=%" PetscInt_FMT " call=%" PetscInt_FMT " time=%.6g cumulative_time=%.6g\n",
                        solver->n_orth_basis, solver->deflation_coarse_calls, (double)(t1 - t0), (double)solver->deflation_coarse_time));
  PetscCall(PetscLogStagePop());
  (void)A;
  (void)work;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DeflationApplyProjectedPC(LinearSolverCtx *solver, Mat A, PC pc, Vec v, Vec z, Vec Az, PetscBool return_Az)
{
  PetscLogDouble t0, t1, t2, pc_time, projector_time = 0.0;

  PetscFunctionBeginUser;
  PetscCall(PetscTime(&t0));
  PetscCall(PCApply(pc, v, z));
  PetscCall(PetscTime(&t1));
  if (solver->n_orth_basis || return_Az) {
    PetscCall(PetscLogStagePush(log_stage_deflation_projector));
    PetscCall(MatMult(A, z, Az));
    for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
      PetscScalar coeff;

      PetscCall(VecDot(LinearSolverTestBasis(solver, i), Az, &coeff));
      PetscCall(VecAXPY(z, -coeff, solver->orth_basis[i]));
      PetscCall(VecAXPY(Az, -coeff, solver->Aorth_basis[i]));
    }
    PetscCall(PetscLogStagePop());
  }
  PetscCall(PetscTime(&t2));
  pc_time        = t1 - t0;
  projector_time = t2 - t1;
  solver->deflation_pc_apply_time += pc_time;
  solver->deflation_projector_time += projector_time;
  ++solver->deflation_projected_pc_calls;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DeflationApplyMatlabProjectedPC(LinearSolverCtx *solver, Mat A, PC pc, Vec v, Vec z, Vec work)
{
  PetscLogDouble t0, t1, t2, pc_time, projector_time = 0.0;

  PetscFunctionBeginUser;
  PetscCall(PetscTime(&t0));
  PetscCall(PCApply(pc, v, z));
  PetscCall(PetscTime(&t1));
  if (solver->n_orth_basis) {
    PetscCall(PetscLogStagePush(log_stage_deflation_projector));
    PetscCall(MatMult(A, z, work));
    for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
      PetscScalar coeff;

      PetscCall(VecDot(LinearSolverTestBasis(solver, i), work, &coeff));
      PetscCall(VecAXPY(z, -coeff, solver->orth_basis[i]));
    }
    PetscCall(PetscLogStagePop());
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

static PetscErrorCode DenseLeastSquaresLAPACK(PetscInt ldh, const PetscScalar H[], PetscReal beta, PetscInt rows, PetscInt cols, PetscScalar y[], PetscReal *residual)
{
  PetscScalar  *A = NULL, *B = NULL, *work = NULL;
  PetscBLASInt  m, n, nrhs = 1, lda, ldb, info, lwork = -1;
  PetscScalar   work_query;
  char          trans = 'N';

  PetscFunctionBeginUser;
  if (!cols) {
    *residual = beta;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscBLASIntCast(rows, &m));
  PetscCall(PetscBLASIntCast(cols, &n));
  PetscCall(PetscBLASIntCast(PetscMax(rows, (PetscInt)1), &lda));
  PetscCall(PetscBLASIntCast(PetscMax(rows, cols), &ldb));
  PetscCall(PetscCalloc2((size_t)lda * (size_t)n, &A, (size_t)ldb, &B));
  for (PetscInt c = 0; c < cols; ++c) {
    for (PetscInt r = 0; r < rows; ++r) A[r + c * rows] = H[r * ldh + c];
  }
  B[0] = beta;
  LAPACKgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, &work_query, &lwork, &info);
  PetscCheck(!info, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACKgels workspace query failed with info=%d", (int)info);
  PetscCall(PetscBLASIntCast(PetscMax((PetscInt)1, (PetscInt)PetscRealPart(work_query)), &lwork));
  PetscCall(PetscMalloc1((size_t)lwork, &work));
  LAPACKgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, work, &lwork, &info);
  PetscCheck(!info, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACKgels least-squares solve failed with info=%d", (int)info);
  for (PetscInt i = 0; i < cols; ++i) y[i] = B[i];
  {
    PetscReal r2 = 0.0;

    for (PetscInt r = 0; r < rows; ++r) {
      PetscScalar hy = 0.0;
      PetscScalar g  = (r == 0) ? beta : 0.0;

      for (PetscInt c = 0; c < cols; ++c) hy += H[r * ldh + c] * y[c];
      r2 += PetscSqr(PetscAbsScalar(g - hy));
    }
    *residual = PetscSqrtReal(r2);
  }
  PetscCall(PetscFree(work));
  PetscCall(PetscFree2(A, B));
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
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATED_FGMRES_INITIAL label=\"%s\" basis_cols=%" PetscInt_FMT " rhs_norm=%.17e beta=%.17e rel=%.17e\n",
                        label, solver->n_orth_basis, (double)rhs_norm, (double)beta, (double)rel));
  if (rel <= rtol) {
    converged = PETSC_TRUE;
  } else {
    PetscCall(VecCopy(r, V[0]));
    PetscCall(VecScale(V[0], 1.0 / beta));
  }

  for (PetscInt j = 0; !converged && j < max_it; ++j) {
    PetscReal hnext, res_norm;

    PetscCall(DeflationApplyProjectedPC(solver, A, pc, V[j], Z[j], Az, PETSC_TRUE));
    PetscCall(LinearSolverCaptureRecycleVector(solver, Z[j]));
    PetscCall(LinearSolverAppendPersistentKrylovVector(solver, Z[j], label, j));
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

static PetscErrorCode DeflatedMatlabDFGMRESSolve(LinearSolverCtx *solver, KSP ksp, Vec rhs, Vec x, const char label[], PetscInt *its, PetscReal *reported_rel)
{
  Mat          A = solver->A;
  PC           pc;
  Vec         *V = NULL, *Z = NULL;
  Vec          r = NULL, work = NULL;
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
  PetscCall(VecDuplicate(rhs, &work));
  PetscCall(PetscCalloc2((max_it + 1) * max_it, &H, max_it, &y));

  PetscCall(PetscLogStagePush(log_stage_deflation_initial_guess));
  PetscCall(VecZeroEntries(x));
  for (PetscInt i = 0; i < solver->n_orth_basis; ++i) {
    PetscScalar coeff;

    PetscCall(VecDot(LinearSolverTestBasis(solver, i), rhs, &coeff));
    PetscCall(VecAXPY(x, coeff, solver->orth_basis[i]));
  }
  PetscCall(MatMult(A, x, work));
  PetscCall(VecWAXPY(r, -1.0, work, rhs));
  PetscCall(PetscLogStagePop());
  ++solver->deflation_coarse_calls;

  PetscCall(VecNorm(rhs, NORM_2, &rhs_norm));
  if (rhs_norm == 0.0) rhs_norm = 1.0;
  PetscCall(VecNorm(r, NORM_2, &beta));
  rel = beta / rhs_norm;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATED_MATLAB_DFGMRES_INITIAL label=\"%s\" basis_cols=%" PetscInt_FMT " rhs_norm=%.17e beta=%.17e rel=%.17e\n",
                        label, solver->n_orth_basis, (double)rhs_norm, (double)beta, (double)rel));
  if (rel <= rtol) {
    converged = PETSC_TRUE;
  } else {
    PetscCall(VecCopy(r, V[0]));
    PetscCall(VecScale(V[0], 1.0 / beta));
  }

  for (PetscInt j = 0; !converged && j < max_it; ++j) {
    PetscReal hnext, res_norm;

    PetscCall(DeflationApplyMatlabProjectedPC(solver, A, pc, V[j], Z[j], work));
    PetscCall(MatMult(A, Z[j], work));
    PetscCall(LinearSolverCaptureRecycleVector(solver, Z[j]));
    PetscCall(LinearSolverAppendPersistentKrylovVector(solver, Z[j], label, j));
    for (PetscInt i = 0; i <= j; ++i) {
      PetscScalar hij;

      PetscCall(VecDot(V[i], work, &hij));
      H[i * max_it + j] = hij;
      PetscCall(VecAXPY(work, -hij, V[i]));
    }
    PetscCall(VecNorm(work, NORM_2, &hnext));
    H[(j + 1) * max_it + j] = hnext;
    if (j == 0) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                            "DEFLATED_MATLAB_DFGMRES_SAMPLE label=\"%s\" it=1 h00=%.17e h10=%.17e\n",
                            label, (double)PetscRealPart(H[0]), (double)hnext));
    }
    final_its = j + 1;
    if (hnext <= 1.0e-14) {
      PetscCall(DenseLeastSquaresLAPACK(max_it, H, beta, final_its + 1, final_its, y, &res_norm));
      rel = res_norm / rhs_norm;
      break;
    }
    PetscCall(VecCopy(work, V[j + 1]));
    PetscCall(VecScale(V[j + 1], 1.0 / hnext));
    PetscCall(DenseLeastSquaresLAPACK(max_it, H, beta, j + 2, j + 1, y, &res_norm));
    rel = res_norm / rhs_norm;
    if (solver->app->deflation_monitor) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm), "DEFLATED_MATLAB_DFGMRES label=\"%s\" it=%" PetscInt_FMT " rel=%.6e\n", label, final_its, (double)rel));
    }
    if (rel <= rtol) converged = PETSC_TRUE;
  }
  for (PetscInt i = 0; i < final_its; ++i) PetscCall(VecAXPY(x, y[i], Z[i]));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                        "DEFLATED_SOLVE label=\"%s\" method=matlab_dfgmres basis_cols=%" PetscInt_FMT " iterations=%" PetscInt_FMT " reported_rel=%.6e converged=%s\n",
                        label, solver->n_orth_basis, final_its, (double)rel, converged ? "true" : "false"));
  *its          = final_its;
  *reported_rel = rel;

  PetscCall(PetscFree2(H, y));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&work));
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

    PetscCall(DeflationApplyProjectedPC(solver, A, pc, r, z, Ap, PETSC_FALSE));
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
    PetscCall(DeflationApplyProjectedPC(solver, A, pc, r, z, Ap, PETSC_FALSE));
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
  PetscBool owned = PETSC_FALSE;

  PetscFunctionBeginUser;
  (void)current_u;
  PetscCall(LinearSolverPrepareKSP(solver, nonlinear_tangent, &ksp, &owned));
  PetscCall(LinearSolverSolvePrepared(solver, ksp, rhs, x, label, nonlinear_tangent, PETSC_FALSE, its));
  PetscCall(LinearSolverFinishPreparedKSP(&ksp, owned));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverPrepareKSP(LinearSolverCtx *solver, PetscBool nonlinear_tangent, KSP *ksp_out, PetscBool *owned_out)
{
  KSP ksp = NULL;

  PetscFunctionBeginUser;
  PetscCheck(solver, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Linear solver pointer is NULL");
  PetscCheck(ksp_out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "KSP output pointer is NULL");
  PetscCheck(owned_out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "KSP ownership output pointer is NULL");
  *ksp_out  = NULL;
  *owned_out = PETSC_FALSE;
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
        if (solver->force_reuse_preconditioner) reuse_pc = PETSC_TRUE;
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)solver->dm),
                              "PMG_LAG_PRECONDITIONER solve_index=%" PetscInt_FMT " lag=%" PetscInt_FMT " rebuild=%s forced_reuse=%s reuse_preconditioner=%s\n",
                              idx, lag, rebuild ? "true" : "false", solver->force_reuse_preconditioner ? "true" : "false", reuse_pc ? "true" : "false"));
        solver->pmg_lag_solve_index++;
      }
      if (solver->force_reuse_preconditioner) reuse_pc = PETSC_TRUE;
      PetscCall(RefreshKSPOperators(solver->ksp, solver->app, solver->A, PETSC_FALSE, reuse_pc));
    }
    ksp = solver->ksp;
  } else {
    PetscCall(KSPCreate(PetscObjectComm((PetscObject)solver->dm), &ksp));
    PetscCall(ConfigureKSP(ksp, solver->dm, solver->actx, solver->app, solver->A, nonlinear_tangent, solver));
    *owned_out = PETSC_TRUE;
  }
  *ksp_out = ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverSolvePrepared(LinearSolverCtx *solver, KSP ksp, Vec rhs, Vec x, const char *label, PetscBool nonlinear_tangent, PetscBool basis_already_orthogonalized, PetscInt *its)
{
  PetscFunctionBeginUser;
  PetscCheck(solver, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Linear solver pointer is NULL");
  PetscCheck(ksp, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "KSP is not set up");
  if (solver->app->use_deflation && nonlinear_tangent && solver->n_raw_basis > 0) {
    PetscReal reported_rel = PETSC_MAX_REAL;

    if (!basis_already_orthogonalized) PetscCall(LinearSolverAOrthogonalizeBasis(solver, solver->A, label));
    PetscCall(VecZeroEntries(x));
    if (solver->app->deflation_solver == DEFLATION_SOLVER_FGMRES) {
      PetscCall(DeflatedFGMRESSolve(solver, ksp, rhs, x, label, its, &reported_rel));
    } else if (solver->app->deflation_solver == DEFLATION_SOLVER_MATLAB_DFGMRES) {
      PetscCall(DeflatedMatlabDFGMRESSolve(solver, ksp, rhs, x, label, its, &reported_rel));
    } else {
      PetscCall(DeflatedCGSolve(solver, ksp, rhs, x, label, its, &reported_rel));
    }
    PetscCall(CheckLinearSolutionExplicit(solver->dm, solver->app, solver->A, rhs, x, label, *its, reported_rel));
  } else {
    PetscCall(KSPSolve(ksp, rhs, x));
    PetscCall(CheckLinearSolve(solver->dm, solver->actx, solver->app, solver->A, rhs, x, label, ksp, its));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode LinearSolverFinishPreparedKSP(KSP *ksp, PetscBool owned)
{
  PetscFunctionBeginUser;
  if (owned) PetscCall(KSPDestroy(ksp));
  else if (ksp) *ksp = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
