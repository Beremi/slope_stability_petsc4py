/*
   Fixed-lambda initialization Newton.

   This is the seed solve used by indirect SSR before continuation starts. The
   loop is intentionally kept in algorithm order: assemble residual/tangent,
   form the regularized operator, solve the correction, damp, accept, and update
   the deflation history. Timing and parseable output enter SsrStats helpers so
   the surrounding continuation code only sees one solve result.
*/
static PetscErrorCode FixedLambdaNewtonSolve(DM dm, AssemblyCtx *actx, AppCtx *app, LinearSolverCtx *solver, Mat Areg, Mat Kelastic, Mat Ktangent, Vec f_ext, Vec u, PetscReal lambda, PetscReal rhs_norm, const char criterion[], PetscReal stop_tol, NewtonStats *stats)
{
  Vec            residual = NULL, rhs = NULL, du = NULL, u_trial = NULL, r_trial = NULL;
  PetscReal      rel = PETSC_MAX_REAL, rel_corr = PETSC_MAX_REAL, r = app->r_min;
  PetscLogDouble wall_time = 0.0, elapsed;
  PetscBool      stop = PETSC_FALSE;
  SsrProfileTimer run_timer, profile_timer;
  const SsrNativeLinearOps *linear_ops = NULL;
  struct _p_SsrLinearCtx    linear_ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(stats, sizeof(*stats)));
  PetscCall(SsrNativeLinearRegistryFind(app->linear_algorithm, &linear_ops));
  PetscCall(SsrNativeLinearCtxBind(solver, &app->resolved_profile.linear, &linear_ctx));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_NEWTON_SOLVE, dm, f_ext, &run_timer);
  PetscCall(VecDuplicate(f_ext, &residual));
  PetscCall(VecDuplicate(f_ext, &rhs));
  PetscCall(VecDuplicate(f_ext, &du));
  PetscCall(VecDuplicate(f_ext, &u_trial));
  PetscCall(VecDuplicate(f_ext, &r_trial));

  for (PetscInt it = 0; it < app->newton_max_it; ++it) {
    PetscInt  linear_its = 0;
    PetscReal alpha = 1.0, du_norm, u_norm, initial_decrease = PETSC_MAX_REAL;
    PetscInt  ls_its = 0;

    SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ASSEMBLE_TANGENT, dm, residual, &profile_timer);
    PetscCall(AssemblePlasticResidualJacobian(actx, lambda, u, f_ext, Ktangent, residual, PETSC_TRUE));
    PetscCall(BuildRegularizedOperator(Areg, Kelastic, Ktangent, r));
    SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
    PetscCall(SsrStatsAddNewtonStepAssembly(stats, elapsed));
    PetscCall(ResidualNormFree(actx, residual, rhs_norm, &rel));
    PetscCall(StoppingByCriterion(criterion, rel, rel_corr, PETSC_MAX_REAL, stop_tol, PETSC_FALSE, &stop));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "SSR_INIT_NEWTON lambda=%.8e it=%" PetscInt_FMT " rel_res=%.6e rel_corr=%.6e stop=%s\n",
                          (double)lambda, it, (double)rel, (double)rel_corr, stop ? "true" : "false"));
    if (stop || rel <= app->newton_rtol) {
      stats->converged = PETSC_TRUE;
      break;
    }

    PetscCall(VecCopy(residual, rhs));
    PetscCall(VecScale(rhs, -1.0));
    PetscCall(ApplyZeroDirichlet(actx->constrained_is, Areg, rhs));
    PetscCall(VecZeroEntries(du));
    solver->A = Areg;
    SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_KSP_SOLVE, Areg, rhs, &profile_timer);
    PetscCall(SsrNativeLinearSolve(linear_ops, &linear_ctx, rhs, du, "SSR fixed-lambda Newton correction", u, PETSC_TRUE, &linear_its));
    SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
    PetscCall(SsrStatsAddNewtonStepLinearSolve(stats, linear_its, elapsed));
    PetscCall(SsrStatsAddNewtonStepIteration(stats));

    PetscCall(VecNorm(du, NORM_2, &du_norm));
    PetscCall(VecNorm(u, NORM_2, &u_norm));
    SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_LINE_SEARCH, u, du, &profile_timer);
    PetscCall(FixedLambdaDirectionalDamping(actx, app, lambda, u, du, residual, f_ext, u_trial, r_trial, &alpha, &ls_its, &initial_decrease));
    SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
    PetscCall(SsrStatsAddNewtonStepLineSearch(stats, ls_its));
    rel_corr = (alpha * du_norm) / PetscMax(u_norm, 1.0e-30);
    PetscCall(VecScale(du, alpha));
    PetscCall(VecAXPY(u, 1.0, du));
    PetscCall(StoppingByCriterion(criterion, rel, rel_corr, PETSC_MAX_REAL, stop_tol, (PetscBool)(alpha > 0.0), &stop));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "SSR_INIT_NEWTON_ACCEPT lambda=%.8e it=%" PetscInt_FMT " alpha=%.6e rel_res=%.6e rel_corr=%.6e r=%.6e directional=%.6e stop=%s\n",
                          (double)lambda, it, (double)alpha, (double)rel, (double)rel_corr, (double)r, (double)initial_decrease, stop ? "true" : "false"));
    if (stop) {
      stats->converged = PETSC_TRUE;
      break;
    }
    if (alpha < 0.1) {
      if (alpha == 0.0) r *= 2.0;
      else r *= PetscPowReal(2.0, 0.25);
    } else {
      PetscCall(LinearSolverAppendRawBasis(solver, du, "SSR fixed-lambda Newton correction"));
      if (alpha > 0.5) r = PetscMax(r / PetscSqrtReal(2.0), app->r_min);
    }
    if (alpha == 0.0 && r > 1.0) break;
  }
  SSR_PROFILE_TIMER_END(NULL, &run_timer, &wall_time);
  stats->final_rel             = rel;
  stats->final_rel_correction  = rel_corr;
  stats->wall_time             = wall_time;
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm),
                        "SSR_INIT_NEWTON_SUMMARY lambda=%.8e converged=%s final_rel=%.6e final_rel_correction=%.6e newton_its=%" PetscInt_FMT " linear_its=%" PetscInt_FMT " wall_time=%.6g\n",
                        (double)lambda, stats->converged ? "true" : "false", (double)stats->final_rel, (double)stats->final_rel_correction,
                        stats->newton_its, stats->total_linear_its, (double)stats->wall_time));

  PetscCall(VecDestroy(&residual));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&du));
  PetscCall(VecDestroy(&u_trial));
  PetscCall(VecDestroy(&r_trial));
  PetscFunctionReturn(PETSC_SUCCESS);
}
