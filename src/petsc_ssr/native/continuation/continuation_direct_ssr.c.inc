/*
   Direct SSR continuation.

   This path advances lambda directly and solves one fixed-lambda nonlinear
   problem per accepted point. It deliberately reuses FixedLambdaNewtonSolve so
   DMPlex assembly, regularized operators, PMG, deflation, and damping stay on
   the same maintained C path as the indirect initialization solves.
*/
static PetscErrorCode SSRDirectContinuationSolve(DM dm, AssemblyCtx *actx, AppCtx *app, LinearSolverCtx *solver, Mat Areg, Mat Kelastic, Mat Ktangent, Vec f_ext, PetscReal rhs_norm, Vec u_out, SSRStats *stats)
{
  MPI_Comm       comm = PetscObjectComm((PetscObject)dm);
  PetscMPIInt    rank;
  FILE          *csv = NULL;
  Vec            u_prev = NULL, u_cur = NULL, u_trial = NULL;
  PetscReal      lambda_prev = 0.0, lambda_cur = 0.0, lambda_trial = 0.0, omega_prev = 0.0, omega_cur = 0.0, omega_trial = 0.0;
  PetscReal      d_lambda = app->d_lambda_init, d_omega_ref = 0.0;
  PetscInt       init_basis_base, attempts = 0, step = 0, lambda_reductions = 0;
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
  PetscCall(VecDuplicate(f_ext, &u_trial));

  lambda_cur = app->lambda_init;
  PetscCall(VecZeroEntries(u_prev));
  PetscCall(PetscPrintf(comm, "SSR_DIRECT_INIT phase=seed lambda=%.8e d_lambda=%.8e basis_snapshot=%" PetscInt_FMT "\n", (double)lambda_cur, (double)d_lambda, init_basis_base));
  PetscCall(FixedLambdaNewtonSolve(dm, actx, app, solver, Areg, Kelastic, Ktangent, f_ext, u_prev, lambda_cur, rhs_norm, app->newton_stopping_criterion, app->newton_stopping_tol, &nstats));
  attempts++;
  PetscCheck(nstats.converged, comm, PETSC_ERR_NOT_CONVERGED, "Direct SSR seed solve failed at lambda %.8e", (double)lambda_cur);
  PetscCall(DotOmega(f_ext, u_prev, &omega_cur));
  PetscCall(LinearSolverTruncateRawBasis(solver, init_basis_base));
  PetscCall(LinearSolverAppendRawBasis(solver, u_prev, "SSR direct accepted seed"));
  PetscCall(FillCurveRow(&row, step, "init", omega_cur, lambda_cur, 0.0, 0.0, u_prev, attempts, &nstats, "accepted"));
  PetscCall(CurveWriteRow(csv, &row));
  PetscCall(PetscPrintf(comm,
                        "SSR_STEP step=%" PetscInt_FMT " phase=direct_init omega=%.8e lambda=%.8e accepted=true newton_its=%" PetscInt_FMT " linear_its=%" PetscInt_FMT "\n",
                        step, (double)omega_cur, (double)lambda_cur, nstats.newton_its, nstats.total_linear_its));
  stats->accepted_steps++;
  stats->total_newton_its += nstats.newton_its;
  stats->total_linear_its += nstats.total_linear_its;
  stats->total_line_search_its += nstats.line_search_its;
  stats->final_rel            = nstats.final_rel;
  stats->final_rel_correction = nstats.final_rel_correction;
  step++;

  while (PETSC_TRUE) {
    const PetscInt basis_snapshot = solver->n_raw_basis;

    lambda_trial = lambda_cur + d_lambda;
    PetscCall(VecCopy(u_prev, u_cur));
    PetscCall(PetscPrintf(comm, "SSR_DIRECT_INIT phase=advance lambda=%.8e d_lambda=%.8e basis_snapshot=%" PetscInt_FMT "\n", (double)lambda_trial, (double)d_lambda, basis_snapshot));
    PetscCall(FixedLambdaNewtonSolve(dm, actx, app, solver, Areg, Kelastic, Ktangent, f_ext, u_cur, lambda_trial, rhs_norm, app->newton_stopping_criterion, app->newton_stopping_tol, &nstats));
    attempts++;
    if (!nstats.converged) {
      PetscCall(LinearSolverTruncateRawBasis(solver, basis_snapshot));
      d_lambda *= 0.5;
      PetscCheck(d_lambda >= app->d_lambda_min, comm, PETSC_ERR_NOT_CONVERGED, "Direct SSR second initialization point failed before d_lambda_min %.6e", (double)app->d_lambda_min);
      continue;
    }
    PetscCall(DotOmega(f_ext, u_cur, &omega_trial));
    if ((omega_trial - omega_cur) / PetscMax(1.0, PetscAbsReal(omega_cur)) < 1.0e-5) {
      PetscCall(PetscPrintf(comm, "SSR_DIRECT_INIT phase=advance tiny_omega_increment=true omega_prev=%.8e omega_cur=%.8e shifting_seed=true\n", (double)omega_cur, (double)omega_trial));
      PetscCall(LinearSolverTruncateRawBasis(solver, init_basis_base));
      PetscCall(VecCopy(u_cur, u_prev));
      lambda_cur = lambda_trial;
      omega_cur  = omega_trial;
      PetscCall(LinearSolverAppendRawBasis(solver, u_prev, "SSR direct shifted seed"));
      continue;
    }
    break;
  }

  lambda_prev = lambda_cur;
  omega_prev  = omega_cur;
  lambda_cur  = lambda_trial;
  omega_cur   = omega_trial;
  PetscCall(LinearSolverTruncateRawBasis(solver, init_basis_base));
  PetscCall(LinearSolverAppendRawBasis(solver, u_prev, "SSR direct accepted init previous"));
  PetscCall(FillCurveRow(&row, step, "init", omega_cur, lambda_cur, omega_cur - omega_prev, lambda_cur - lambda_prev, u_cur, attempts, &nstats, "accepted"));
  PetscCall(CurveWriteRow(csv, &row));
  PetscCall(PetscPrintf(comm,
                        "SSR_STEP step=%" PetscInt_FMT " phase=direct_init omega=%.8e lambda=%.8e d_omega=%.8e d_lambda=%.8e accepted=true newton_its=%" PetscInt_FMT " linear_its=%" PetscInt_FMT "\n",
                        step, (double)omega_cur, (double)lambda_cur, (double)(omega_cur - omega_prev), (double)(lambda_cur - lambda_prev), nstats.newton_its, nstats.total_linear_its));
  stats->accepted_steps++;
  stats->total_newton_its += nstats.newton_its;
  stats->total_linear_its += nstats.total_linear_its;
  stats->total_line_search_its += nstats.line_search_its;
  stats->final_rel            = nstats.final_rel;
  stats->final_rel_correction = nstats.final_rel_correction;
  step++;
  d_omega_ref = omega_cur - omega_prev;
  PetscCall(VecCopy(u_cur, u_trial));

  if (omega_cur >= app->omega_max) PetscCall(PetscStrncpy(stats->stop_reason, "omega_max", sizeof(stats->stop_reason)));
  else if (step >= app->continuation_step_max) PetscCall(PetscStrncpy(stats->stop_reason, "step_max", sizeof(stats->stop_reason)));
  PetscCheck(d_omega_ref > 0.0, comm, PETSC_ERR_NOT_CONVERGED, "Direct SSR initialization did not produce increasing omega: %.8e -> %.8e", (double)omega_prev, (double)omega_cur);

  while (PETSC_TRUE) {
    PetscBool      reason_running;
    const PetscInt basis_snapshot = solver->n_raw_basis;
    PetscReal      d_omega_trial;

    PetscCall(PetscStrcmp(stats->stop_reason, "running", &reason_running));
    if (!reason_running || step >= app->continuation_step_max || omega_cur >= app->omega_max) break;
    lambda_trial = lambda_cur + d_lambda;
    PetscCall(VecCopy(u_cur, u_trial));
    PetscCall(PetscPrintf(comm,
                          "SSR_ATTEMPT step=%" PetscInt_FMT " method=direct lambda_trial=%.8e d_lambda=%.8e basis_snapshot=%" PetscInt_FMT "\n",
                          step, (double)lambda_trial, (double)d_lambda, basis_snapshot));
    PetscCall(FixedLambdaNewtonSolve(dm, actx, app, solver, Areg, Kelastic, Ktangent, f_ext, u_trial, lambda_trial, rhs_norm, app->newton_stopping_criterion, app->newton_stopping_tol, &nstats));
    attempts++;
    PetscCall(LinearSolverTruncateRawBasis(solver, basis_snapshot));
    if (!nstats.converged) {
      d_lambda *= 0.5;
      lambda_reductions++;
      PetscCall(PetscPrintf(comm,
                            "SSR_ATTEMPT step=%" PetscInt_FMT " method=direct accepted=false reason=newton_failed reductions=%" PetscInt_FMT " next_d_lambda=%.8e\n",
                            step, lambda_reductions, (double)d_lambda));
      if (d_lambda < app->d_lambda_min) PetscCall(PetscStrncpy(stats->stop_reason, "d_lambda_min", sizeof(stats->stop_reason)));
      continue;
    }
    PetscCall(DotOmega(f_ext, u_trial, &omega_trial));
    d_omega_trial = omega_trial - omega_cur;
    if (!(d_omega_trial >= 0.0) || PetscIsInfOrNanReal(d_omega_trial)) {
      d_lambda *= 0.5;
      lambda_reductions++;
      PetscCall(PetscPrintf(comm,
                            "SSR_ATTEMPT step=%" PetscInt_FMT " method=direct accepted=false reason=negative_omega_increment d_omega=%.8e reductions=%" PetscInt_FMT " next_d_lambda=%.8e\n",
                            step, (double)d_omega_trial, lambda_reductions, (double)d_lambda));
      if (d_lambda < app->d_lambda_min) PetscCall(PetscStrncpy(stats->stop_reason, "d_lambda_min", sizeof(stats->stop_reason)));
      continue;
    }

    lambda_prev = lambda_cur;
    omega_prev  = omega_cur;
    PetscCall(VecCopy(u_cur, u_prev));
    lambda_cur = lambda_trial;
    omega_cur  = omega_trial;
    PetscCall(VecCopy(u_trial, u_cur));
    PetscCall(LinearSolverAppendRawBasis(solver, u_cur, "SSR direct accepted continuation state"));
    lambda_reductions = 0;
    PetscCall(FillCurveRow(&row, step, "direct", omega_cur, lambda_cur, omega_cur - omega_prev, lambda_cur - lambda_prev, u_cur, attempts, &nstats, "accepted"));
    PetscCall(CurveWriteRow(csv, &row));
    PetscCall(PetscPrintf(comm,
                          "SSR_STEP step=%" PetscInt_FMT " phase=direct omega=%.8e lambda=%.8e d_omega=%.8e d_lambda=%.8e accepted=true newton_its=%" PetscInt_FMT " linear_its=%" PetscInt_FMT " rel_res=%.6e\n",
                          step, (double)omega_cur, (double)lambda_cur, (double)(omega_cur - omega_prev), (double)(lambda_cur - lambda_prev),
                          nstats.newton_its, nstats.total_linear_its, (double)nstats.final_rel));
    stats->accepted_steps++;
    stats->total_newton_its += nstats.newton_its;
    stats->total_linear_its += nstats.total_linear_its;
    stats->total_line_search_its += nstats.line_search_its;
    stats->final_rel            = nstats.final_rel;
    stats->final_rel_correction = nstats.final_rel_correction;
    step++;

    if (d_omega_trial > 1.5 * d_omega_ref) d_lambda *= 0.5;
    if (app->d_lambda_diff_scaled_min > 0.0) {
      const PetscReal slope_scaled = PetscAbsReal((lambda_cur - lambda_prev) / PetscMax(omega_cur - omega_prev, 1.0e-30)) * PetscMax(omega_cur, 1.0);
      if (slope_scaled <= app->d_lambda_diff_scaled_min) PetscCall(PetscStrncpy(stats->stop_reason, "d_lambda_diff_scaled_min", sizeof(stats->stop_reason)));
    }
    if (omega_cur >= app->omega_max) PetscCall(PetscStrncpy(stats->stop_reason, "omega_max", sizeof(stats->stop_reason)));
    if (d_lambda < app->d_lambda_min) PetscCall(PetscStrncpy(stats->stop_reason, "d_lambda_min", sizeof(stats->stop_reason)));
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
                        "SSR_RESULT method=direct omega_last=%.8e lambda_last=%.8e accepted_steps=%" PetscInt_FMT " total_newton_iterations=%" PetscInt_FMT " total_linear_iterations=%" PetscInt_FMT " total_line_search_iterations=%" PetscInt_FMT " wall_time=%.6g stop_reason=%s curve_csv=%s\n",
                        (double)stats->omega_last, (double)stats->lambda_last, stats->accepted_steps, stats->total_newton_its,
                        stats->total_linear_its, stats->total_line_search_its, (double)stats->wall_time, stats->stop_reason, app->curve_csv));

  if (csv) fclose(csv);
  PetscCall(VecDestroy(&u_prev));
  PetscCall(VecDestroy(&u_cur));
  PetscCall(VecDestroy(&u_trial));
  PetscFunctionReturn(PETSC_SUCCESS);
}
