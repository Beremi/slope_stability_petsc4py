/*
   Coupled indirect SSR Newton solve.

   For a prescribed omega target, each iteration assembles the current tangent
   and lambda derivative, solves the W and V directions with the same
   regularized operator, combines them into a displacement/lambda correction,
   runs ALG5-style damping, then accepts or regularizes the next trial. The
   optional replay diagnostics live here only because they compare this loop
   iteration by iteration against exported petsc4py states.
*/
static PetscErrorCode IndirectNewtonLineSearch(AssemblyCtx *actx, AppCtx *app, Vec f_ext, Vec u, PetscReal lambda, Vec dU, PetscReal d_lambda, PetscReal omega_target, PetscReal current_rel, PetscReal rhs_norm, Vec u_trial, Vec r_trial, PetscReal *alpha_out, PetscReal *trial_rel_out, PetscInt *ls_its)
{
  PetscReal alpha = 1.0, trial_rel = PETSC_MAX_REAL, last_alpha = 0.0, last_trial_rel = PETSC_MAX_REAL;

  PetscFunctionBeginUser;
  (void)omega_target;
  *ls_its = 0;
  if (PetscIsInfOrNanReal(d_lambda)) {
    *alpha_out = 0.0;
    *trial_rel_out = PETSC_MAX_REAL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  for (PetscInt damp = 0; damp < PetscMax(1, app->it_damp_max); ++damp) {
    if (lambda + alpha * d_lambda <= 0.0) {
      alpha *= 0.5;
      (*ls_its)++;
      continue;
    }
    PetscCall(VecWAXPY(u_trial, alpha, dU, u));
    PetscCall(AssembleResidualRel(actx, lambda + alpha * d_lambda, u_trial, f_ext, r_trial, rhs_norm, &trial_rel));
    (*ls_its)++;
    if (!PetscIsInfOrNanReal(trial_rel)) {
      last_alpha     = alpha;
      last_trial_rel = trial_rel;
      if (trial_rel < current_rel) break;
    }
    alpha *= 0.5;
  }
  *alpha_out     = last_alpha;
  *trial_rel_out = last_trial_rel;
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscInt  iteration;
  PetscReal lambda;
  PetscReal r;
  PetscReal alpha;
  PetscReal delta_lambda;
  PetscReal accepted_delta_lambda;
  PetscReal rel_residual;
  PetscReal rel_correction;
  PetscReal stopping_value;
  PetscInt  dW_iterations;
  PetscInt  dV_iterations;
  PetscInt  linear_iterations;
  PetscInt  line_search_iterations;
  PetscInt  deflation_basis_dim_solve;
  PetscInt  deflation_basis_dim_end;
  char      status[32];
} StepReplayExpectedRow;

typedef struct {
  PetscBool              loaded;
  PetscInt               nrows;
  StepReplayExpectedRow *rows;
} StepReplayExpected;

static PetscErrorCode StepReplayExpectedLoad(MPI_Comm comm, const char dir[], StepReplayExpected *expected)
{
  char path[PETSC_MAX_PATH_LEN], line[1024];
  FILE *fh = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(expected, sizeof(*expected)));
  if (!dir || !dir[0]) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSNPrintf(path, sizeof(path), "%s/step_expected.csv", dir));
  fh = fopen(path, "r");
  if (!fh) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscCalloc1(1024, &expected->rows));
  if (!fgets(line, sizeof(line), fh)) {
    fclose(fh);
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  while (fgets(line, sizeof(line), fh)) {
    StepReplayExpectedRow row;
    int                   nscan;
    long long             iteration, dW_iterations, dV_iterations, linear_iterations, line_search_iterations, basis_solve, basis_end;

    PetscCall(PetscMemzero(&row, sizeof(row)));
    nscan = sscanf(line,
                   "%lld,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lld,%lld,%lld,%lld,%lld,%lld,%31[^,\n]",
                   &iteration, &row.lambda, &row.r, &row.alpha, &row.delta_lambda, &row.accepted_delta_lambda,
                   &row.rel_residual, &row.rel_correction, &row.stopping_value, &dW_iterations, &dV_iterations,
                   &linear_iterations, &line_search_iterations, &basis_solve, &basis_end, row.status);
    if (nscan < 15) continue;
    row.iteration                   = (PetscInt)iteration;
    row.dW_iterations               = (PetscInt)dW_iterations;
    row.dV_iterations               = (PetscInt)dV_iterations;
    row.linear_iterations           = (PetscInt)linear_iterations;
    row.line_search_iterations      = (PetscInt)line_search_iterations;
    row.deflation_basis_dim_solve   = (PetscInt)basis_solve;
    row.deflation_basis_dim_end     = (PetscInt)basis_end;
    PetscCheck(expected->nrows < 1024, comm, PETSC_ERR_ARG_SIZ, "Too many step replay expected rows in %s", path);
    expected->rows[expected->nrows++] = row;
  }
  fclose(fh);
  expected->loaded = PETSC_TRUE;
  PetscCall(PetscPrintf(comm, "STEP_REPLAY_EXPECTED file=%s rows=%" PetscInt_FMT "\n", path, expected->nrows));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static const StepReplayExpectedRow *StepReplayExpectedFind(const StepReplayExpected *expected, PetscInt iteration)
{
  if (!expected || !expected->loaded) return NULL;
  for (PetscInt i = 0; i < expected->nrows; ++i) {
    if (expected->rows[i].iteration == iteration) return &expected->rows[i];
  }
  return NULL;
}

static PetscErrorCode StepReplayExpectedDestroy(StepReplayExpected *expected)
{
  PetscFunctionBeginUser;
  PetscCall(PetscFree(expected->rows));
  expected->loaded = PETSC_FALSE;
  expected->nrows  = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IndirectNewtonSolve(DM dm, AssemblyCtx *actx, AppCtx *app, LinearSolverCtx *solver, Mat Areg, Mat Kelastic, Mat Ktangent, Vec f_ext, Vec u, PetscReal *lambda, PetscReal omega_target, PetscReal rhs_norm, PetscReal initial_r, const char criterion[], PetscReal stop_tol, NewtonStats *stats)
{
  Vec            residual = NULL, residual_eps = NULL, G = NULL, rhsW = NULL, rhsV = NULL, dW = NULL, dV = NULL, dU = NULL, u_trial = NULL, r_trial = NULL;
  PetscReal      rel = PETSC_MAX_REAL, rel_corr = PETSC_MAX_REAL, r = PetscMax(initial_r, app->r_min), abs_dlambda = PETSC_MAX_REAL;
  PetscInt       prev_itsW = -1, prev_itsV = -1;
  PetscLogDouble wall_time = 0.0, elapsed;
  PetscBool      stop = PETSC_FALSE, compute_diffs = PETSC_TRUE;
  PetscBool      have_pair_matrix = PETSC_FALSE;
  PetscInt       pair_matrix_start_it = -1;
  PetscReal      pair_matrix_r = r;
  const PetscInt krylov_persistent_start_added = solver->deflation_krylov_persistent_added;
  StepReplayExpected expected;
  SsrProfileTimer run_timer, profile_timer;
  const SsrNativeLinearOps *linear_ops = NULL;
  struct _p_SsrLinearCtx    linear_ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(stats, sizeof(*stats)));
  PetscCall(SsrNativeLinearRegistryFind(app->linear_algorithm, &linear_ops));
  PetscCall(SsrNativeLinearCtxBind(solver, &app->resolved_profile.linear, &linear_ctx));
  if (!(*lambda > 0.0) || PetscIsInfOrNanReal(*lambda)) {
    stats->final_rel            = PETSC_MAX_REAL;
    stats->final_rel_correction = PETSC_MAX_REAL;
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                          "SSR_NEWTON_REJECT omega=%.8e reason=invalid_initial_lambda lambda=%.8e\n",
                          (double)omega_target, (double)*lambda));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(StepReplayExpectedLoad(PetscObjectComm((PetscObject)dm), app->step_replay_dir, &expected));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_NEWTON_SOLVE, dm, f_ext, &run_timer);
  PetscCall(VecDuplicate(f_ext, &residual));
  PetscCall(VecDuplicate(f_ext, &residual_eps));
  PetscCall(VecDuplicate(f_ext, &G));
  PetscCall(VecDuplicate(f_ext, &rhsW));
  PetscCall(VecDuplicate(f_ext, &rhsV));
  PetscCall(VecDuplicate(f_ext, &dW));
  PetscCall(VecDuplicate(f_ext, &dV));
  PetscCall(VecDuplicate(f_ext, &dU));
  PetscCall(VecDuplicate(f_ext, &u_trial));
  PetscCall(VecDuplicate(f_ext, &r_trial));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                        "DEFLATION_KRYLOV_PERSISTENT_CONFIG enabled=%s basis_tol=%.6e raw_cols_start=%" PetscInt_FMT " total_added_start=%" PetscInt_FMT "\n",
                        app->deflation_krylov_persistent ? "true" : "false", (double)app->deflation_krylov_basis_tol,
                        solver->n_raw_basis, krylov_persistent_start_added));

  for (PetscInt it = 0; it < app->newton_max_it; ++it) {
    PetscInt  itsW = 0, itsV = 0, ls_its = 0;
    PetscReal denom, numer, alpha = 0.0, trial_rel = PETSC_MAX_REAL, dU_norm, u_norm;
    PetscScalar dot;
    const PetscReal lambda_start = *lambda;
    const PetscReal r_start = r;
    const PetscInt  basis_start = solver->n_raw_basis;

    const PetscBool freeze_pair_matrix =
      (PetscBool)(app->indirect_newton_pair_freeze_matrix && have_pair_matrix && (it % 2 == 1));
    const SsrEvent assembly_event = freeze_pair_matrix ? SSR_EVENT_ASSEMBLE_RESIDUAL : SSR_EVENT_ASSEMBLE_TANGENT;

    SSR_PROFILE_TIMER_BEGIN(NULL, assembly_event, dm, residual, &profile_timer);
    if (freeze_pair_matrix) {
      PetscCall(AssemblePlasticResidualJacobian(actx, *lambda, u, f_ext, NULL, residual, PETSC_FALSE));
    } else {
      PetscCall(AssemblePlasticResidualJacobian(actx, *lambda, u, f_ext, Ktangent, residual, PETSC_TRUE));
    }
    PetscCall(ResidualNormFree(actx, residual, rhs_norm, &rel));
    if (compute_diffs) PetscCall(ComputeLambdaDerivativeFD(actx, app, *lambda, u, f_ext, residual, residual_eps, G));
    if (!freeze_pair_matrix) {
      PetscCall(BuildRegularizedOperator(Areg, Kelastic, Ktangent, r));
      have_pair_matrix    = PETSC_TRUE;
      pair_matrix_start_it = it;
      pair_matrix_r        = r;
    }
    SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
    PetscCall(SsrStatsAddNewtonStepAssembly(stats, elapsed));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                          "SSR_PAIR_MATRIX omega=%.8e it=%" PetscInt_FMT " enabled=%s mode=%s pair_start_it=%" PetscInt_FMT " current_r=%.6e matrix_r=%.6e force_pc_reuse=%s\n",
                          (double)omega_target, it, app->indirect_newton_pair_freeze_matrix ? "true" : "false",
                          freeze_pair_matrix ? "frozen" : "fresh", pair_matrix_start_it, (double)r, (double)pair_matrix_r,
                          freeze_pair_matrix ? "true" : "false"));

    PetscCall(StoppingByCriterion(criterion, rel, rel_corr, abs_dlambda, stop_tol, PETSC_FALSE, &stop));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                          "SSR_NEWTON omega=%.8e it=%" PetscInt_FMT " lambda=%.8e rel_res=%.6e rel_corr=%.6e r=%.6e compute_diffs=%s stop=%s\n",
                          (double)omega_target, it, (double)*lambda, (double)rel, (double)rel_corr, (double)r, compute_diffs ? "true" : "false", stop ? "true" : "false"));
    if (stop || (rel <= app->newton_rtol && it > 0)) {
      stats->converged = PETSC_TRUE;
      break;
    }

    PetscCall(VecCopy(G, rhsW));
    PetscCall(VecScale(rhsW, -1.0));
    PetscCall(VecCopy(residual, rhsV));
    PetscCall(VecScale(rhsV, -1.0));
    PetscCall(ZeroConstrainedVector(actx->constrained_is, rhsW));
    PetscCall(ZeroConstrainedVector(actx->constrained_is, rhsV));
    solver->A = Areg;
    PetscCall(VecZeroEntries(dW));
    PetscCall(VecZeroEntries(dV));
    SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_KSP_SOLVE, Areg, rhsW, &profile_timer);
    {
      const PetscBool saved_force_reuse = solver->force_reuse_preconditioner;

    if (app->deflation_intra_newton_recycle && app->use_deflation && solver->n_raw_basis > 0) {
      const PetscBool cheap_is_v = (PetscBool)(prev_itsW >= 0 && prev_itsV >= 0 && prev_itsV < prev_itsW);
      const PetscInt  raw_snapshot = solver->n_raw_basis;
      PetscInt        temp_added = 0;

      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                            "DEFLATION_INTRANEWTON_SELECT omega=%.8e it=%" PetscInt_FMT " enabled=true previous_w=%" PetscInt_FMT " previous_v=%" PetscInt_FMT " first=%s target=%s raw_cols=%" PetscInt_FMT "\n",
                            (double)omega_target, it, prev_itsW, prev_itsV, cheap_is_v ? "dV" : "dW", cheap_is_v ? "dW" : "dV",
                            raw_snapshot));
      if (cheap_is_v) {
        solver->force_reuse_preconditioner = freeze_pair_matrix;
        PetscCall(LinearSolverBeginRecycleCapture(solver));
        PetscCall(SsrNativeLinearSolve(linear_ops, &linear_ctx, rhsV, dV, "SSR indirect dV", u, PETSC_TRUE, &itsV));
        PetscCall(LinearSolverEndRecycleCapture(solver));
        PetscCall(LinearSolverAppendTemporaryRecycleBasis(solver, "SSR intra-newton dV-to-dW", &temp_added));
        solver->force_reuse_preconditioner = PETSC_TRUE;
        PetscCall(SsrNativeLinearSolve(linear_ops, &linear_ctx, rhsW, dW, "SSR indirect dW", u, PETSC_TRUE, &itsW));
      } else {
        solver->force_reuse_preconditioner = freeze_pair_matrix;
        PetscCall(LinearSolverBeginRecycleCapture(solver));
        PetscCall(SsrNativeLinearSolve(linear_ops, &linear_ctx, rhsW, dW, "SSR indirect dW", u, PETSC_TRUE, &itsW));
        PetscCall(LinearSolverEndRecycleCapture(solver));
        PetscCall(LinearSolverAppendTemporaryRecycleBasis(solver, "SSR intra-newton dW-to-dV", &temp_added));
        solver->force_reuse_preconditioner = PETSC_TRUE;
        PetscCall(SsrNativeLinearSolve(linear_ops, &linear_ctx, rhsV, dV, "SSR indirect dV", u, PETSC_TRUE, &itsV));
      }
      PetscCall(LinearSolverTruncateRawBasis(solver, raw_snapshot));
      PetscCall(LinearSolverClearRecycleBasis(solver));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                            "DEFLATION_INTRANEWTON_RESULT omega=%.8e it=%" PetscInt_FMT " first=%s temp_added=%" PetscInt_FMT " linear_w=%" PetscInt_FMT " linear_v=%" PetscInt_FMT "\n",
                            (double)omega_target, it, cheap_is_v ? "dV" : "dW", temp_added, itsW, itsV));
    } else {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                            "DEFLATION_INTRANEWTON_SELECT omega=%.8e it=%" PetscInt_FMT " enabled=false previous_w=%" PetscInt_FMT " previous_v=%" PetscInt_FMT " raw_cols=%" PetscInt_FMT "\n",
                            (double)omega_target, it, prev_itsW, prev_itsV, solver->n_raw_basis));
      solver->force_reuse_preconditioner = freeze_pair_matrix;
      PetscCall(SsrNativeLinearSolve(linear_ops, &linear_ctx, rhsW, dW, "SSR indirect dW", u, PETSC_TRUE, &itsW));
      solver->force_reuse_preconditioner = PETSC_TRUE;
      PetscCall(SsrNativeLinearSolve(linear_ops, &linear_ctx, rhsV, dV, "SSR indirect dV", u, PETSC_TRUE, &itsV));
    }
      solver->force_reuse_preconditioner = saved_force_reuse;
    }
    SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
    PetscCall(SsrStatsAddNewtonStepLinearSolve(stats, itsW + itsV, elapsed));
    prev_itsW = itsW;
    prev_itsV = itsV;
    PetscCall(SsrStatsAddNewtonStepIteration(stats));

    PetscCall(VecDot(f_ext, dW, &dot));
    denom = PetscRealPart(dot);
    PetscCall(VecDot(f_ext, dV, &dot));
    numer = PetscRealPart(dot);
    PetscCheck(PetscAbsReal(denom) > 1.0e-30, PetscObjectComm((PetscObject)dm), PETSC_ERR_NOT_CONVERGED,
               "Indirect Newton has singular constraint denominator f_ext^T dW=%.6e", (double)denom);
    abs_dlambda = PetscAbsReal(-numer / denom);
    PetscCall(VecCopy(dV, dU));
    PetscCall(VecAXPY(dU, -numer / denom, dW));
    SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_LINE_SEARCH, u, dU, &profile_timer);
    PetscCall(IndirectNewtonLineSearch(actx, app, f_ext, u, *lambda, dU, -numer / denom, omega_target, rel, rhs_norm, u_trial, r_trial, &alpha, &trial_rel, &ls_its));
    SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
    PetscCall(SsrStatsAddNewtonStepLineSearch(stats, ls_its));
    PetscCall(VecNorm(dU, NORM_2, &dU_norm));
    PetscCall(VecNorm(u, NORM_2, &u_norm));
    rel_corr = (alpha * dU_norm) / PetscMax(u_norm, 1.0);

    PetscCall(StoppingByCriterion(criterion, rel, rel_corr, abs_dlambda, stop_tol, (PetscBool)(alpha > 0.0), &stop));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                          "SSR_NEWTON_ACCEPT omega=%.8e it=%" PetscInt_FMT " alpha=%.6e lambda=%.8e d_lambda=%.6e rel_res=%.6e trial_rel=%.6e rel_corr=%.6e linear_w=%" PetscInt_FMT " linear_v=%" PetscInt_FMT " linear_its=%" PetscInt_FMT " stop=%s\n",
                          (double)omega_target, it, (double)alpha, (double)*lambda, (double)(-numer / denom), (double)rel, (double)trial_rel,
                          (double)rel_corr, itsW, itsV, itsW + itsV, stop ? "true" : "false"));
    {
      const StepReplayExpectedRow *erow = StepReplayExpectedFind(&expected, it + 1);

      if (erow) {
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                              "STEP_REPLAY_NEWTON_COMPARE it=%" PetscInt_FMT " expected_w=%" PetscInt_FMT " c_w=%" PetscInt_FMT " expected_v=%" PetscInt_FMT " c_v=%" PetscInt_FMT " expected_total=%" PetscInt_FMT " c_total=%" PetscInt_FMT " expected_alpha=%.17e c_alpha=%.17e expected_lambda=%.17e c_lambda=%.17e expected_r=%.17e c_r=%.17e expected_dlambda=%.17e c_dlambda=%.17e expected_rel_res=%.17e c_rel_res=%.17e expected_rel_corr=%.17e c_rel_corr=%.17e expected_ls=%" PetscInt_FMT " c_ls=%" PetscInt_FMT " expected_basis=%" PetscInt_FMT " c_basis=%" PetscInt_FMT " expected_status=%s c_stop=%s\n",
                              it + 1, erow->dW_iterations, itsW, erow->dV_iterations, itsV, erow->linear_iterations,
                              itsW + itsV, (double)erow->alpha, (double)alpha, (double)erow->lambda, (double)lambda_start,
                              (double)erow->r, (double)r_start, (double)erow->delta_lambda, (double)(-numer / denom),
                              (double)erow->rel_residual, (double)rel, (double)erow->rel_correction, (double)rel_corr,
                              erow->line_search_iterations, ls_its, erow->deflation_basis_dim_solve, basis_start,
                              erow->status, stop ? "true" : "false"));
      }
    }
    if (alpha > 0.0) {
      PetscCall(VecCopy(u_trial, u));
      PetscCall(ScaleToOmega(f_ext, u, omega_target));
      *lambda += alpha * (-numer / denom);
      rel = trial_rel;
    }
    if (stop) {
      stats->converged = PETSC_TRUE;
      break;
    }
    compute_diffs = PETSC_TRUE;
    {
      PetscBool appended_pair_solutions = PETSC_FALSE;

      if (app->indirect_newton_pair_freeze_matrix && alpha > 0.0) {
        PetscCall(LinearSolverAppendRawBasis(solver, dW, "SSR indirect pair dW"));
        PetscCall(LinearSolverAppendRawBasis(solver, dV, "SSR indirect pair dV"));
        appended_pair_solutions = PETSC_TRUE;
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                              "DEFLATION_PAIR_SOLUTIONS_ADD omega=%.8e it=%" PetscInt_FMT " raw_cols=%" PetscInt_FMT " alpha=%.6e\n",
                              (double)omega_target, it, solver->n_raw_basis, (double)alpha));
      }

    if (alpha < 0.1) {
      if (alpha == 0.0) {
        compute_diffs = PETSC_FALSE;
        r *= 2.0;
      } else {
        r *= PetscPowReal(2.0, 0.25);
      }
    } else if (alpha > 0.5) {
      if (!appended_pair_solutions) {
        PetscCall(LinearSolverAppendRawBasis(solver, dW, "SSR indirect dW"));
        PetscCall(LinearSolverAppendRawBasis(solver, dV, "SSR indirect dV"));
      }
      r = PetscMax(r / PetscSqrtReal(2.0), app->r_min);
    } else {
      if (!appended_pair_solutions) {
        PetscCall(LinearSolverAppendRawBasis(solver, dW, "SSR indirect dW"));
        PetscCall(LinearSolverAppendRawBasis(solver, dV, "SSR indirect dV"));
      }
    }
    }
    if (alpha == 0.0 && r > 1.0) break;
  }

  PetscCall(AssembleResidualRel(actx, *lambda, u, f_ext, residual, rhs_norm, &rel));
  SSR_PROFILE_TIMER_END(NULL, &run_timer, &wall_time);
  stats->final_rel             = rel;
  stats->final_rel_correction  = rel_corr;
  stats->wall_time             = wall_time;
  if (!stats->converged && rel <= 10.0 * app->newton_rtol) stats->converged = PETSC_TRUE;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                        "SSR_NEWTON_SUMMARY omega=%.8e converged=%s lambda=%.8e final_rel=%.6e final_rel_correction=%.6e newton_its=%" PetscInt_FMT " linear_its=%" PetscInt_FMT " line_search_its=%" PetscInt_FMT " assembly_time=%.6g solve_time=%.6g wall_time=%.6g\n",
                        (double)omega_target, stats->converged ? "true" : "false", (double)*lambda, (double)stats->final_rel,
                        (double)stats->final_rel_correction, stats->newton_its, stats->total_linear_its, stats->line_search_its,
                        (double)stats->assembly_time, (double)stats->solve_time, (double)stats->wall_time));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                        "DEFLATION_KRYLOV_PERSISTENT_SUMMARY enabled=%s added_this_newton=%" PetscInt_FMT " added_total=%" PetscInt_FMT " raw_cols_end=%" PetscInt_FMT "\n",
                        app->deflation_krylov_persistent ? "true" : "false",
                        solver->deflation_krylov_persistent_added - krylov_persistent_start_added,
                        solver->deflation_krylov_persistent_added, solver->n_raw_basis));
  PetscCall(StepReplayExpectedDestroy(&expected));

  PetscCall(VecDestroy(&residual));
  PetscCall(VecDestroy(&residual_eps));
  PetscCall(VecDestroy(&G));
  PetscCall(VecDestroy(&rhsW));
  PetscCall(VecDestroy(&rhsV));
  PetscCall(VecDestroy(&dW));
  PetscCall(VecDestroy(&dV));
  PetscCall(VecDestroy(&dU));
  PetscCall(VecDestroy(&u_trial));
  PetscCall(VecDestroy(&r_trial));
  PetscFunctionReturn(PETSC_SUCCESS);
}
