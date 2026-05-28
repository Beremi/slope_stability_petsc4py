/*
   Classic indirect SSR continuation.

   The control flow mirrors the reference implementation: solve two fixed-lambda
   seed points with backoff, predict the next omega point by a secant step,
   solve the indirect Newton problem, halve omega on failure, and accept the
   point into both the curve and deflation history on success.
*/
static PetscErrorCode SSRContinuationSolve(DM dm, AssemblyCtx *actx, AppCtx *app, LinearSolverCtx *solver, Mat Areg, Mat Kelastic, Mat Ktangent, Vec f_ext, PetscReal rhs_norm, Vec u_out, SSRStats *stats)
{
  MPI_Comm       comm = PetscObjectComm((PetscObject)dm);
  PetscMPIInt    rank;
  FILE          *csv = NULL;
  Vec            u_prev = NULL, u_cur = NULL, u_guess = NULL, u_tmp = NULL;
  PetscReal      lambda_prev, lambda_cur, lambda_guess, omega_prev, omega_cur, d_lambda = app->d_lambda_init, d_omega;
  PetscReal      lambda_init = app->lambda_init;
  PetscInt       init_basis_base;
  PetscInt       attempts = 0, step = 0, omega_reductions = 0;
  PetscLogDouble t_start, t_end;
  NewtonStats    nstats;
  SSRCurveRow    row;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(stats, sizeof(*stats)));
  PetscCall(PetscStrncpy(stats->stop_reason, "running", sizeof(stats->stop_reason)));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    csv = fopen(app->curve_csv, "w");
    PetscCheck(csv, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Failed to open continuation CSV %s", app->curve_csv);
  }
  PetscCall(CurveWriteHeader(csv));
  PetscCall(PetscTime(&t_start));
  init_basis_base = solver->n_raw_basis;
  PetscCall(VecDuplicate(f_ext, &u_prev));
  PetscCall(VecDuplicate(f_ext, &u_cur));
  PetscCall(VecDuplicate(f_ext, &u_guess));
  PetscCall(VecDuplicate(f_ext, &u_tmp));

  while (PETSC_TRUE) {
    const PetscInt basis_snapshot = solver->n_raw_basis;

    PetscCall(VecZeroEntries(u_prev));
    PetscCall(PetscPrintf(comm, "SSR_INIT phase=seed lambda=%.8e d_lambda=%.8e basis_snapshot=%" PetscInt_FMT "\n", (double)lambda_init, (double)d_lambda, basis_snapshot));
    PetscCall(FixedLambdaNewtonSolve(dm, actx, app, solver, Areg, Kelastic, Ktangent, f_ext, u_prev, lambda_init, rhs_norm, app->init_newton_stopping_criterion, app->init_newton_stopping_tol, &nstats));
    attempts++;
    if (nstats.converged) break;
    PetscCall(LinearSolverTruncateRawBasis(solver, basis_snapshot));
    lambda_init *= 0.5;
    d_lambda *= 0.5;
    PetscCheck(d_lambda >= app->d_lambda_min, comm, PETSC_ERR_NOT_CONVERGED, "SSR initialization failed before d_lambda_min %.6e", (double)app->d_lambda_min);
  }
  lambda_prev = lambda_init;
  PetscCall(DotOmega(f_ext, u_prev, &omega_prev));
  PetscCall(LinearSolverAppendRawBasis(solver, u_prev, "SSR init seed"));
  PetscCall(FillCurveRow(&row, step, "init", omega_prev, lambda_prev, 0.0, 0.0, u_prev, attempts, &nstats, "accepted"));
  PetscCall(CurveWriteRow(csv, &row));
  PetscCall(PetscPrintf(comm, "SSR_STEP step=%" PetscInt_FMT " phase=init omega=%.8e lambda=%.8e accepted=true newton_its=%" PetscInt_FMT " linear_its=%" PetscInt_FMT "\n",
                        step, (double)omega_prev, (double)lambda_prev, nstats.newton_its, nstats.total_linear_its));
  stats->accepted_steps++;
  stats->total_newton_its += nstats.newton_its;
  stats->total_linear_its += nstats.total_linear_its;
  stats->total_line_search_its += nstats.line_search_its;
  stats->final_rel            = nstats.final_rel;
  stats->final_rel_correction = nstats.final_rel_correction;
  step++;

  while (PETSC_TRUE) {
    const PetscInt basis_snapshot = solver->n_raw_basis;

    lambda_cur = lambda_prev + d_lambda;
    PetscCall(VecCopy(u_prev, u_cur));
    PetscCall(PetscPrintf(comm, "SSR_INIT phase=advance lambda=%.8e d_lambda=%.8e basis_snapshot=%" PetscInt_FMT "\n", (double)lambda_cur, (double)d_lambda, basis_snapshot));
    PetscCall(FixedLambdaNewtonSolve(dm, actx, app, solver, Areg, Kelastic, Ktangent, f_ext, u_cur, lambda_cur, rhs_norm, app->init_newton_stopping_criterion, app->init_newton_stopping_tol, &nstats));
    attempts++;
    if (!nstats.converged) {
      PetscCall(LinearSolverTruncateRawBasis(solver, basis_snapshot));
      d_lambda *= 0.5;
      PetscCheck(d_lambda >= app->d_lambda_min, comm, PETSC_ERR_NOT_CONVERGED, "SSR second initialization point failed before d_lambda_min %.6e", (double)app->d_lambda_min);
      continue;
    }
    PetscCall(DotOmega(f_ext, u_cur, &omega_cur));
    if ((omega_cur - omega_prev) / PetscMax(1.0, PetscAbsReal(omega_prev)) < 1.0e-5) {
      PetscCall(PetscPrintf(comm, "SSR_INIT phase=advance tiny_omega_increment=true omega_prev=%.8e omega_cur=%.8e shifting_seed=true\n", (double)omega_prev, (double)omega_cur));
      PetscCall(VecCopy(u_cur, u_prev));
      lambda_prev = lambda_cur;
      omega_prev  = omega_cur;
      continue;
    }
    break;
  }
  PetscCall(LinearSolverTruncateRawBasis(solver, init_basis_base));
  PetscCall(LinearSolverAppendRawBasis(solver, u_prev, "SSR accepted init previous"));
  PetscCall(FillCurveRow(&row, step, "init", omega_cur, lambda_cur, omega_cur - omega_prev, lambda_cur - lambda_prev, u_cur, attempts, &nstats, "accepted"));
  PetscCall(CurveWriteRow(csv, &row));
  PetscCall(PetscPrintf(comm, "SSR_STEP step=%" PetscInt_FMT " phase=init omega=%.8e lambda=%.8e d_omega=%.8e d_lambda=%.8e accepted=true newton_its=%" PetscInt_FMT " linear_its=%" PetscInt_FMT "\n",
                        step, (double)omega_cur, (double)lambda_cur, (double)(omega_cur - omega_prev), (double)(lambda_cur - lambda_prev), nstats.newton_its, nstats.total_linear_its));
  stats->accepted_steps++;
  stats->total_newton_its += nstats.newton_its;
  stats->total_linear_its += nstats.total_linear_its;
  stats->total_line_search_its += nstats.line_search_its;
  stats->final_rel            = nstats.final_rel;
  stats->final_rel_correction = nstats.final_rel_correction;
  step++;

  d_omega = omega_cur - omega_prev;
  PetscCheck(d_omega > 0.0, comm, PETSC_ERR_NOT_CONVERGED, "SSR initialization did not produce increasing omega: %.8e -> %.8e", (double)omega_prev, (double)omega_cur);
  if (omega_cur >= app->omega_max || step >= app->continuation_step_max) {
    PetscCall(PetscStrncpy(stats->stop_reason, omega_cur >= app->omega_max ? "omega_max" : "step_max", sizeof(stats->stop_reason)));
  }

  while (PETSC_TRUE) {
    PetscBool      reason_running;
    const PetscInt basis_snapshot = solver->n_raw_basis;
    PetscReal      target = PetscMin(omega_cur + d_omega, app->omega_max);
    PetscReal      alpha_sec = (target - omega_cur) / (omega_cur - omega_prev);
    PetscReal      omega_old = omega_cur;
    PetscBool      branch_double = PETSC_FALSE;

    PetscCall(PetscStrcmp(stats->stop_reason, "running", &reason_running));
    if (!reason_running || step >= app->continuation_step_max || omega_cur >= app->omega_max) break;
    PetscCall(VecCopy(u_cur, u_guess));
    PetscCall(VecWAXPY(u_tmp, -1.0, u_prev, u_cur));
    PetscCall(VecAXPY(u_guess, alpha_sec, u_tmp));
    lambda_guess = lambda_cur + alpha_sec * (lambda_cur - lambda_prev);
    PetscCall(PetscPrintf(comm, "SSR_ATTEMPT step=%" PetscInt_FMT " target_omega=%.8e d_omega=%.8e lambda_predict=%.8e basis_snapshot=%" PetscInt_FMT "\n",
                          step, (double)target, (double)d_omega, (double)lambda_guess, basis_snapshot));
    if (!(lambda_guess > 0.0) || PetscIsInfOrNanReal(lambda_guess)) {
      d_omega *= 0.5;
      omega_reductions++;
      PetscCall(PetscPrintf(comm,
                            "SSR_ATTEMPT step=%" PetscInt_FMT " accepted=false reason=invalid_lambda_predict lambda_predict=%.8e reductions=%" PetscInt_FMT " next_d_omega=%.8e\n",
                            step, (double)lambda_guess, omega_reductions, (double)d_omega));
      if (omega_reductions >= 5) PetscCall(PetscStrncpy(stats->stop_reason, "omega_reduction_limit", sizeof(stats->stop_reason)));
      continue;
    }
    PetscCall(IndirectNewtonSolve(dm, actx, app, solver, Areg, Kelastic, Ktangent, f_ext, u_guess, &lambda_guess, target, rhs_norm, app->r_min, app->newton_stopping_criterion, app->newton_stopping_tol, &nstats));
    attempts++;
    if (!nstats.converged) {
      PetscCall(LinearSolverTruncateRawBasis(solver, basis_snapshot));
      d_omega *= 0.5;
      omega_reductions++;
      PetscCall(PetscPrintf(comm, "SSR_ATTEMPT step=%" PetscInt_FMT " accepted=false reductions=%" PetscInt_FMT " next_d_omega=%.8e\n", step, omega_reductions, (double)d_omega));
      if (omega_reductions >= 5) PetscCall(PetscStrncpy(stats->stop_reason, "omega_reduction_limit", sizeof(stats->stop_reason)));
      continue;
    }
    PetscCall(LinearSolverTruncateRawBasis(solver, basis_snapshot));

    branch_double = (PetscBool)((lambda_guess - lambda_cur) < 0.9 * (lambda_cur - lambda_prev));
    lambda_prev = lambda_cur;
    omega_prev  = omega_cur;
    PetscCall(VecCopy(u_cur, u_prev));
    lambda_cur = lambda_guess;
    omega_cur  = target;
    PetscCall(VecCopy(u_guess, u_cur));
    PetscCall(LinearSolverAppendRawBasis(solver, u_cur, "SSR accepted continuation state"));
    omega_reductions = 0;
    PetscCall(FillCurveRow(&row, step, "cont", omega_cur, lambda_cur, omega_cur - omega_prev, lambda_cur - lambda_prev, u_cur, attempts, &nstats, "accepted"));
    PetscCall(CurveWriteRow(csv, &row));
    PetscCall(PetscPrintf(comm,
                          "SSR_STEP step=%" PetscInt_FMT " phase=cont omega=%.8e lambda=%.8e d_omega=%.8e d_lambda=%.8e accepted=true newton_its=%" PetscInt_FMT " linear_its=%" PetscInt_FMT " rel_res=%.6e\n",
                          step, (double)omega_cur, (double)lambda_cur, (double)(omega_cur - omega_prev), (double)(lambda_cur - lambda_prev),
                          nstats.newton_its, nstats.total_linear_its, (double)nstats.final_rel));
    stats->accepted_steps++;
    stats->total_newton_its += nstats.newton_its;
    stats->total_linear_its += nstats.total_linear_its;
    stats->total_line_search_its += nstats.line_search_its;
    stats->final_rel            = nstats.final_rel;
    stats->final_rel_correction = nstats.final_rel_correction;
    step++;

    if (branch_double) d_omega *= 2.0;
    if (app->d_lambda_diff_scaled_min > 0.0) {
      const PetscReal slope_scaled = PetscAbsReal((lambda_cur - lambda_prev) / PetscMax(omega_cur - omega_prev, 1.0e-30)) * PetscMax(omega_cur, 1.0);
      if (slope_scaled <= app->d_lambda_diff_scaled_min) PetscCall(PetscStrncpy(stats->stop_reason, "d_lambda_diff_scaled_min", sizeof(stats->stop_reason)));
    }
    if (omega_cur >= app->omega_max) PetscCall(PetscStrncpy(stats->stop_reason, "omega_max", sizeof(stats->stop_reason)));
    (void)omega_old;
  }
  {
    PetscBool reason_running;

    PetscCall(PetscStrcmp(stats->stop_reason, "running", &reason_running));
    if (omega_cur >= app->omega_max && reason_running) PetscCall(PetscStrncpy(stats->stop_reason, "omega_max", sizeof(stats->stop_reason)));
    if (step >= app->continuation_step_max && reason_running) PetscCall(PetscStrncpy(stats->stop_reason, "step_max", sizeof(stats->stop_reason)));
  }
  stats->omega_last  = omega_cur;
  stats->lambda_last = lambda_cur;
  if (u_out) PetscCall(VecCopy(u_cur, u_out));
  PetscCall(PetscTime(&t_end));
  stats->wall_time = t_end - t_start;
  PetscCall(PetscPrintf(comm,
                        "SSR_RESULT omega_last=%.8e lambda_last=%.8e accepted_steps=%" PetscInt_FMT " total_newton_iterations=%" PetscInt_FMT " total_linear_iterations=%" PetscInt_FMT " total_line_search_iterations=%" PetscInt_FMT " wall_time=%.6g stop_reason=%s curve_csv=%s\n",
                        (double)stats->omega_last, (double)stats->lambda_last, stats->accepted_steps, stats->total_newton_its,
                        stats->total_linear_its, stats->total_line_search_its, (double)stats->wall_time, stats->stop_reason, app->curve_csv));

  if (csv) fclose(csv);
  PetscCall(VecDestroy(&u_prev));
  PetscCall(VecDestroy(&u_cur));
  PetscCall(VecDestroy(&u_guess));
  PetscCall(VecDestroy(&u_tmp));
  PetscFunctionReturn(PETSC_SUCCESS);
}
