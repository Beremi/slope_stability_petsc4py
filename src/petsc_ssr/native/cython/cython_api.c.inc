static PetscErrorCode P4SsrEngineCheckSlot(P4SsrEngine *ctx, PetscInt slot)
{
  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "P4SsrEngine pointer is NULL");
  PetscCheck(slot >= 0 && slot < 4, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "State slot %" PetscInt_FMT " outside [0,4)", slot);
  PetscCheck(ctx->states[slot], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "State slot %" PetscInt_FMT " is not initialized", slot);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineCreateOptionsString(const char options[], P4SsrEngine **out)
{
  P4SsrEngine   *ctx = NULL;
  PetscInt       cStart, cEnd, nStart, nEnd, local_dofs, reported_owned_constraints = 0, reported_global_constraints = 0;
  PetscLogDouble  elapsed;
  SsrProfileTimer create_timer, profile_timer;

  PetscFunctionBeginUser;
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Output engine pointer is NULL");
  *out = NULL;
  if (options && options[0]) PetscCall(PetscOptionsInsertString(NULL, options));
  PetscCall(RegisterLogStages());
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ENGINE_CREATE, NULL, NULL, &create_timer);
  PetscCall(PetscCalloc1(1, &ctx));
  PetscCall(ParseOptions(PETSC_COMM_WORLD, &ctx->app));
  PetscCall(NativeProblemManifestApply(&ctx->app));
  PetscCall(MaterialMCConfigureFromOptions(PETSC_COMM_WORLD));
  PetscCall(SetDDPartitionerDefault(PETSC_COMM_WORLD, &ctx->app));
  PetscCall(P4BasisCreateDegreeDim(PETSC_COMM_SELF, ctx->app.element_degree, ctx->app.mechanics_dim, ctx->app.mechanics_dim, &ctx->basis));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &ctx->app, &ctx->basis, &ctx->dm));
  PetscCall(DMPlexGetHeightStratum(ctx->dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetDepthStratum(ctx->dm, 0, &nStart, &nEnd));
  PetscCall(DMCreateMatrix(ctx->dm, &ctx->Areg));
  PetscCall(DMCreateMatrix(ctx->dm, &ctx->Kelastic));
  PetscCall(DMCreateMatrix(ctx->dm, &ctx->Ktangent));
  PetscCall(MatSetOption(ctx->Areg, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(MatSetOption(ctx->Kelastic, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(MatSetOption(ctx->Ktangent, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(DMCreateGlobalVector(ctx->dm, &ctx->f_ext));
  for (PetscInt i = 0; i < 4; ++i) PetscCall(DMCreateGlobalVector(ctx->dm, &ctx->states[i]));
  PetscCall(VecDuplicate(ctx->f_ext, &ctx->residual));
  PetscCall(VecDuplicate(ctx->f_ext, &ctx->residual_eps));
  PetscCall(VecDuplicate(ctx->f_ext, &ctx->G));
  PetscCall(VecDuplicate(ctx->f_ext, &ctx->rhs));
  PetscCall(VecDuplicate(ctx->f_ext, &ctx->rhsW));
  PetscCall(VecDuplicate(ctx->f_ext, &ctx->rhsV));
  PetscCall(VecDuplicate(ctx->f_ext, &ctx->du));
  PetscCall(VecDuplicate(ctx->f_ext, &ctx->dW));
  PetscCall(VecDuplicate(ctx->f_ext, &ctx->dV));
  PetscCall(VecDuplicate(ctx->f_ext, &ctx->dU));
  PetscCall(VecDuplicate(ctx->f_ext, &ctx->u_trial));
  PetscCall(VecDuplicate(ctx->f_ext, &ctx->r_trial));
  PetscCall(VecGetLocalSize(ctx->states[0], &local_dofs));
  PetscCall(VecGetSize(ctx->states[0], &ctx->global_dofs));
  PetscCall(AssemblyCtxCreate(ctx->dm, &ctx->basis, &ctx->actx));
  reported_owned_constraints = ctx->actx.n_constrained_local;
  reported_global_constraints = ctx->actx.n_constrained_all;
  {
    PetscBool                    label_constraints_loaded = PETSC_FALSE;
    AssemblyLabelConstraintStats label_stats;

    PetscCall(AssemblyCtxLoadLabelConstraintsCSV(&ctx->actx, ctx->app.mechanics_bc_labels_csv, &label_constraints_loaded, &label_stats));
    if (label_constraints_loaded) {
      reported_owned_constraints = label_stats.owned_constraints;
      reported_global_constraints = label_stats.global_constraints;
    }
    if (ctx->app.native_problem_manifest_loaded) {
      PetscCheck(label_stats.rows == ctx->app.native_manifest_mechanics_dirichlet, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
                 "Native problem manifest declares %" PetscInt_FMT " mechanics Dirichlet row(s), but label table contains %" PetscInt_FMT,
                 ctx->app.native_manifest_mechanics_dirichlet, label_stats.rows);
    }
    PetscCheck(!ctx->app.mechanics_bc_nodes_csv[0] || ctx->app.debug_coordinate_bc_table, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "-mechanics_bc_nodes_csv requires -debug_coordinate_bc_table true");
    if (ctx->app.mechanics_bc_nodes_csv[0]) {
      if (label_constraints_loaded) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MECHANICS_BC_NODES_CONFIG path=%s status=debug_coordinate_override\n", ctx->app.mechanics_bc_nodes_csv));
      PetscCall(AssemblyCtxLoadCoordinateConstraintsCSV(&ctx->actx, ctx->app.mechanics_bc_nodes_csv));
      reported_owned_constraints = ctx->actx.n_constrained_local;
      reported_global_constraints = ctx->actx.n_constrained_all;
    } else if (!label_constraints_loaded) PetscCall(AssemblyCtxLoadCoordinateConstraintsCSV(&ctx->actx, ctx->app.mechanics_bc_nodes_csv));
  }
  if (ctx->app.seepage_pressure_csv[0]) {
    PetscBool pressure_source_ok = PETSC_FALSE;

    PetscCall(PetscStrcasecmp(ctx->app.seepage_pressure_source, "hydro_prepass_coordinate_bridge", &pressure_source_ok));
    PetscCheck(pressure_source_ok, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "-seepage_pressure_csv requires -seepage_pressure_source hydro_prepass_coordinate_bridge");
  }
  PetscCall(AssemblyCtxLoadSeepagePressureCSV(&ctx->actx, ctx->app.seepage_pressure_csv, ctx->app.seepage_grho));
  PetscCall(AssemblyCtxValidateSeepageBoundaryLabelsCSV(&ctx->actx, ctx->app.seepage_boundary_labels_csv,
                                                        ctx->app.native_problem_manifest_loaded ? ctx->app.native_manifest_seepage_head : -1,
                                                        ctx->app.native_problem_manifest_loaded ? ctx->app.native_manifest_seepage_flux : -1));
  PetscCall(AssemblyCtxValidateNeumannLabelsCSV(&ctx->actx, ctx->app.mechanics_neumann_labels_csv,
                                                ctx->app.native_problem_manifest_loaded ? ctx->app.native_manifest_mechanics_neumann : -1));
  if (ctx->app.native_problem_manifest[0]) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "NATIVE_PROBLEM_MANIFEST path=%s\n", ctx->app.native_problem_manifest));
  PetscCall(LinearSolverInit(&ctx->solver, ctx->dm, &ctx->actx, &ctx->app, ctx->Areg));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "PY_ENGINE_CREATE mesh=%s%s refine_levels=%" PetscInt_FMT " mechanics_dim=%" PetscInt_FMT " element_degree=%" PetscInt_FMT " local_cells=%" PetscInt_FMT " local_vertices=%" PetscInt_FMT " local_dofs=%" PetscInt_FMT " global_dofs=%" PetscInt_FMT " basis_functions=%" PetscInt_FMT " owned_constraints=%" PetscInt_FMT " global_constraints=%" PetscInt_FMT " pc_variant=%s pmg_backend=%s bc=%s omega_max=%.6g lambda_init=%.6g d_lambda_init=%.6g\n",
                        ctx->app.use_box_mesh ? "generated-box:" : "", ctx->app.use_box_mesh ? "unit" : ctx->app.mesh, ctx->app.refine_levels,
                        ctx->basis.dim, ctx->basis.degree, cEnd - cStart, nEnd - nStart, local_dofs, ctx->global_dofs, ctx->basis.n_basis,
                        reported_owned_constraints, reported_global_constraints, ctx->app.variant_name,
                        ctx->app.pmg_apply_backend, ctx->app.mesh_bc_mode, (double)ctx->app.omega_max,
                        (double)ctx->app.lambda_init, (double)ctx->app.d_lambda_init));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ASSEMBLE_ELASTIC, ctx->Kelastic, ctx->f_ext, &profile_timer);
  PetscCall(AssembleElasticProblem(&ctx->actx, ctx->Kelastic, ctx->f_ext));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &ctx->elastic_assembly_time);
  PetscCall(ApplyZeroDirichlet(ctx->actx.constrained_is, ctx->Kelastic, ctx->f_ext));
  PetscCall(VecNorm(ctx->f_ext, NORM_2, &ctx->rhs_norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "PY_ENGINE_ELASTIC assembly_time=%.6g rhs_norm=%.6e\n", (double)ctx->elastic_assembly_time, (double)ctx->rhs_norm));
  for (PetscInt i = 0; i < 4; ++i) PetscCall(VecZeroEntries(ctx->states[i]));
  SSR_PROFILE_TIMER_END(NULL, &create_timer, &elapsed);
  ctx->create_time = elapsed;
  *out = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineDestroy(P4SsrEngine **ctxp)
{
  P4SsrEngine *ctx;

  PetscFunctionBeginUser;
  if (!ctxp || !*ctxp) PetscFunctionReturn(PETSC_SUCCESS);
  ctx = *ctxp;
  PetscCall(VecDestroy(&ctx->r_trial));
  PetscCall(VecDestroy(&ctx->u_trial));
  PetscCall(VecDestroy(&ctx->dU));
  PetscCall(VecDestroy(&ctx->dV));
  PetscCall(VecDestroy(&ctx->dW));
  PetscCall(VecDestroy(&ctx->du));
  PetscCall(VecDestroy(&ctx->rhsV));
  PetscCall(VecDestroy(&ctx->rhsW));
  PetscCall(VecDestroy(&ctx->rhs));
  PetscCall(VecDestroy(&ctx->G));
  PetscCall(VecDestroy(&ctx->residual_eps));
  PetscCall(VecDestroy(&ctx->residual));
  for (PetscInt i = 0; i < 4; ++i) PetscCall(VecDestroy(&ctx->states[i]));
  PetscCall(VecDestroy(&ctx->f_ext));
  if (ctx->prepared_ksp_owned) PetscCall(KSPDestroy(&ctx->prepared_ksp));
  else ctx->prepared_ksp = NULL;
  PetscCall(LinearSolverDestroy(&ctx->solver));
  PetscCall(AssemblyCtxDestroy(&ctx->actx));
  PetscCall(MatDestroy(&ctx->Ktangent));
  PetscCall(MatDestroy(&ctx->Kelastic));
  PetscCall(MatDestroy(&ctx->Areg));
  PetscCall(DMDestroy(&ctx->dm));
  PetscCall(P4BasisDestroy(&ctx->basis));
  PetscCall(PetscFree(ctx));
  *ctxp = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineGetInfo(P4SsrEngine *ctx, P4SsrEngineInfo *info)
{
  PetscMPIInt size = 1;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "P4SsrEngine pointer is NULL");
  PetscCheck(info, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Info pointer is NULL");
  PetscCall(PetscMemzero(info, sizeof(*info)));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  info->ranks = (PetscInt)size;
  info->global_dofs = ctx->global_dofs;
  info->basis_cols = ctx->solver.n_raw_basis;
  info->rhs_norm = ctx->rhs_norm;
  info->elastic_assembly_time = ctx->elastic_assembly_time;
  info->create_time = ctx->create_time;
  info->deflation_orthogonalization_time = ctx->solver.deflation_orthogonalization_time;
  info->deflation_coarse_initial_time = ctx->solver.deflation_coarse_time;
  info->deflation_pc_apply_time = ctx->solver.deflation_pc_apply_time;
  info->deflation_projector_time = ctx->solver.deflation_projector_time;
  info->deflation_coarse_initial_calls = ctx->solver.deflation_coarse_calls;
  info->deflation_projected_pc_calls = ctx->solver.deflation_projected_pc_calls;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineBasisCols(P4SsrEngine *ctx, PetscInt *cols)
{
  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "P4SsrEngine pointer is NULL");
  PetscCheck(cols, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Output pointer is NULL");
  *cols = ctx->solver.n_raw_basis;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineTruncateBasis(P4SsrEngine *ctx, PetscInt n_keep)
{
  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "P4SsrEngine pointer is NULL");
  PetscCall(LinearSolverTruncateRawBasis(&ctx->solver, n_keep));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineAppendBasisFromSlot(P4SsrEngine *ctx, PetscInt slot, const char label[])
{
  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCall(LinearSolverAppendRawBasis(&ctx->solver, ctx->states[slot], label ? label : "python-loop state"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineVecZero(P4SsrEngine *ctx, PetscInt slot)
{
  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCall(VecZeroEntries(ctx->states[slot]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineVecCopy(P4SsrEngine *ctx, PetscInt src, PetscInt dst)
{
  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, src));
  PetscCall(P4SsrEngineCheckSlot(ctx, dst));
  PetscCall(VecCopy(ctx->states[src], ctx->states[dst]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineVecWAXPY(P4SsrEngine *ctx, PetscInt dst, PetscReal alpha, PetscInt x, PetscInt y)
{
  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, dst));
  PetscCall(P4SsrEngineCheckSlot(ctx, x));
  PetscCall(P4SsrEngineCheckSlot(ctx, y));
  PetscCall(VecWAXPY(ctx->states[dst], alpha, ctx->states[x], ctx->states[y]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineVecAXPY(P4SsrEngine *ctx, PetscInt y, PetscReal alpha, PetscInt x)
{
  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, y));
  PetscCall(P4SsrEngineCheckSlot(ctx, x));
  PetscCall(VecAXPY(ctx->states[y], alpha, ctx->states[x]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineDotOmega(P4SsrEngine *ctx, PetscInt slot, PetscReal *omega)
{
  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCall(DotOmega(ctx->f_ext, ctx->states[slot], omega));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineScaleToOmega(P4SsrEngine *ctx, PetscInt slot, PetscReal omega)
{
  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCall(ScaleToOmega(ctx->f_ext, ctx->states[slot], omega));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineDisplacementMax(P4SsrEngine *ctx, PetscInt slot, PetscReal *u_max)
{
  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCall(DisplacementMax(ctx->states[slot], u_max));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineWriteSolutionFromSlot(P4SsrEngine *ctx, PetscInt slot)
{
  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCall(WriteFinalSolutionOutputs(&ctx->app, ctx->dm, &ctx->actx, ctx->states[slot]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineFixedLambdaStep(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda, const char criterion[], PetscReal stop_tol, PetscReal rel_corr_in, PetscReal r_in, P4SsrStepResult *out)
{
  PetscReal      rel = PETSC_MAX_REAL, rel_corr = rel_corr_in, r = r_in;
  PetscReal      alpha = 1.0, du_norm, u_norm, initial_decrease = PETSC_MAX_REAL;
  PetscInt       linear_its = 0, ls_its = 0;
  PetscLogDouble  elapsed;
  SsrProfileTimer run_timer, profile_timer;
  PetscBool      stop = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_NEWTON_SOLVE, ctx->dm, ctx->states[slot], &run_timer);
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ASSEMBLE_TANGENT, ctx->Ktangent, ctx->residual, &profile_timer);
  PetscCall(AssemblePlasticResidualJacobian(&ctx->actx, lambda, ctx->states[slot], ctx->f_ext, ctx->Ktangent, ctx->residual, PETSC_TRUE));
  PetscCall(BuildRegularizedOperator(ctx->Areg, ctx->Kelastic, ctx->Ktangent, r));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  out->assembly_time = elapsed;
  PetscCall(ResidualNormFree(&ctx->actx, ctx->residual, ctx->rhs_norm, &rel));
  PetscCall(StoppingByCriterion(criterion, rel, rel_corr, PETSC_MAX_REAL, stop_tol, PETSC_FALSE, &stop));
  if (stop || rel <= ctx->app.newton_rtol) {
    out->converged = PETSC_TRUE;
    out->stop = PETSC_TRUE;
    out->rel_residual = rel;
    out->rel_correction = rel_corr;
    out->r_out = r;
    SSR_PROFILE_TIMER_END(NULL, &run_timer, &elapsed);
    out->wall_time = elapsed;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecCopy(ctx->residual, ctx->rhs));
  PetscCall(VecScale(ctx->rhs, -1.0));
  PetscCall(ApplyZeroDirichlet(ctx->actx.constrained_is, ctx->Areg, ctx->rhs));
  PetscCall(VecZeroEntries(ctx->du));
  ctx->solver.A = ctx->Areg;
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_KSP_SOLVE, ctx->Areg, ctx->rhs, &profile_timer);
  PetscCall(SolveLinearSystem(&ctx->solver, ctx->rhs, ctx->du, "SSR fixed-lambda Newton correction", ctx->states[slot], PETSC_TRUE, &linear_its));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  out->solve_time = elapsed;
  PetscCall(VecNorm(ctx->du, NORM_2, &du_norm));
  PetscCall(VecNorm(ctx->states[slot], NORM_2, &u_norm));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_LINE_SEARCH, ctx->states[slot], ctx->du, &profile_timer);
  PetscCall(FixedLambdaDirectionalDamping(&ctx->actx, &ctx->app, lambda, ctx->states[slot], ctx->du, ctx->residual, ctx->f_ext, ctx->u_trial, ctx->r_trial, &alpha, &ls_its, &initial_decrease));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  rel_corr = (alpha * du_norm) / PetscMax(u_norm, 1.0e-30);
  PetscCall(VecScale(ctx->du, alpha));
  PetscCall(VecAXPY(ctx->states[slot], 1.0, ctx->du));
  PetscCall(StoppingByCriterion(criterion, rel, rel_corr, PETSC_MAX_REAL, stop_tol, (PetscBool)(alpha > 0.0), &stop));
  if (stop) out->converged = PETSC_TRUE;
  else if (alpha < 0.1) {
    if (alpha == 0.0) r *= 2.0;
    else r *= PetscPowReal(2.0, 0.25);
  } else {
    PetscCall(LinearSolverAppendRawBasis(&ctx->solver, ctx->du, "SSR fixed-lambda Newton correction"));
    if (alpha > 0.5) r = PetscMax(r / PetscSqrtReal(2.0), ctx->app.r_min);
  }
  out->failed = (PetscBool)(!out->converged && alpha == 0.0 && r > 1.0);
  out->solved = PETSC_TRUE;
  out->stop = stop;
  out->rel_residual = rel;
  out->rel_correction = rel_corr;
  out->alpha = alpha;
  out->r_out = r;
  out->linear_its = linear_its;
  out->line_search_its = ls_its;
  out->newton_its = 1;
  out->initial_decrease = initial_decrease;
  SSR_PROFILE_TIMER_END(NULL, &run_timer, &elapsed);
  out->wall_time = elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineIndirectStep(P4SsrEngine *ctx, PetscInt slot, PetscInt iteration, PetscReal lambda_in, PetscReal omega_target, const char criterion[], PetscReal stop_tol, PetscReal rel_corr_in, PetscReal abs_dlambda_in, PetscReal r_in, PetscBool compute_diffs, P4SsrStepResult *out)
{
  PetscReal      rel = PETSC_MAX_REAL, rel_corr = rel_corr_in, r = r_in, abs_dlambda = abs_dlambda_in;
  PetscReal      denom, numer, alpha = 0.0, trial_rel = PETSC_MAX_REAL, dU_norm, u_norm, d_lambda;
  PetscScalar    dot;
  PetscInt       itsW = 0, itsV = 0, ls_its = 0;
  PetscLogDouble  elapsed;
  SsrProfileTimer run_timer, profile_timer;
  PetscBool      stop = PETSC_FALSE;
  const SsrEvent  assembly_event = compute_diffs ? SSR_EVENT_ASSEMBLE_RESIDUAL : SSR_EVENT_ASSEMBLE_TANGENT;

  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCheck(!ctx->app.indirect_newton_pair_freeze_matrix, PetscObjectComm((PetscObject)ctx->dm), PETSC_ERR_SUP, "Python-loop engine does not expose pair-freeze matrix mode");
  PetscCheck(!ctx->app.deflation_intra_newton_recycle, PetscObjectComm((PetscObject)ctx->dm), PETSC_ERR_SUP, "Python-loop engine does not expose intra-Newton recycle mode");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  out->lambda_out = lambda_in;
  out->compute_diffs_out = PETSC_TRUE;
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_NEWTON_SOLVE, ctx->dm, ctx->states[slot], &run_timer);
  SSR_PROFILE_TIMER_BEGIN(NULL, assembly_event, ctx->Ktangent, ctx->residual, &profile_timer);
  PetscCall(AssemblePlasticResidualJacobian(&ctx->actx, lambda_in, ctx->states[slot], ctx->f_ext, ctx->Ktangent, ctx->residual, PETSC_TRUE));
  PetscCall(ResidualNormFree(&ctx->actx, ctx->residual, ctx->rhs_norm, &rel));
  if (compute_diffs) PetscCall(ComputeLambdaDerivativeFD(&ctx->actx, &ctx->app, lambda_in, ctx->states[slot], ctx->f_ext, ctx->residual, ctx->residual_eps, ctx->G));
  PetscCall(BuildRegularizedOperator(ctx->Areg, ctx->Kelastic, ctx->Ktangent, r));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  out->assembly_time = elapsed;
  PetscCall(StoppingByCriterion(criterion, rel, rel_corr, abs_dlambda, stop_tol, PETSC_FALSE, &stop));
  if (stop || (rel <= ctx->app.newton_rtol && iteration > 0)) {
    out->converged = PETSC_TRUE;
    out->stop = PETSC_TRUE;
    out->rel_residual = rel;
    out->rel_correction = rel_corr;
    out->r_out = r;
    SSR_PROFILE_TIMER_END(NULL, &run_timer, &elapsed);
    out->wall_time = elapsed;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecCopy(ctx->G, ctx->rhsW));
  PetscCall(VecScale(ctx->rhsW, -1.0));
  PetscCall(VecCopy(ctx->residual, ctx->rhsV));
  PetscCall(VecScale(ctx->rhsV, -1.0));
  PetscCall(ZeroConstrainedVector(ctx->actx.constrained_is, ctx->rhsW));
  PetscCall(ZeroConstrainedVector(ctx->actx.constrained_is, ctx->rhsV));
  ctx->solver.A = ctx->Areg;
  PetscCall(VecZeroEntries(ctx->dW));
  PetscCall(VecZeroEntries(ctx->dV));
  {
    const PetscBool saved_force_reuse = ctx->solver.force_reuse_preconditioner;

    SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_KSP_SOLVE, ctx->Areg, ctx->rhsW, &profile_timer);
    ctx->solver.force_reuse_preconditioner = PETSC_FALSE;
    PetscCall(SolveLinearSystem(&ctx->solver, ctx->rhsW, ctx->dW, "SSR indirect dW", ctx->states[slot], PETSC_TRUE, &itsW));
    ctx->solver.force_reuse_preconditioner = PETSC_TRUE;
    PetscCall(SolveLinearSystem(&ctx->solver, ctx->rhsV, ctx->dV, "SSR indirect dV", ctx->states[slot], PETSC_TRUE, &itsV));
    ctx->solver.force_reuse_preconditioner = saved_force_reuse;
    SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  }
  out->solve_time = elapsed;
  PetscCall(VecDot(ctx->f_ext, ctx->dW, &dot));
  denom = PetscRealPart(dot);
  PetscCall(VecDot(ctx->f_ext, ctx->dV, &dot));
  numer = PetscRealPart(dot);
  PetscCheck(PetscAbsReal(denom) > 1.0e-30, PetscObjectComm((PetscObject)ctx->dm), PETSC_ERR_NOT_CONVERGED,
             "Indirect Newton has singular constraint denominator f_ext^T dW=%.6e", (double)denom);
  d_lambda = -numer / denom;
  abs_dlambda = PetscAbsReal(d_lambda);
  PetscCall(VecCopy(ctx->dV, ctx->dU));
  PetscCall(VecAXPY(ctx->dU, d_lambda, ctx->dW));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_LINE_SEARCH, ctx->states[slot], ctx->dU, &profile_timer);
  PetscCall(IndirectNewtonLineSearch(&ctx->actx, &ctx->app, ctx->f_ext, ctx->states[slot], lambda_in, ctx->dU, d_lambda, omega_target, rel, ctx->rhs_norm, ctx->u_trial, ctx->r_trial, &alpha, &trial_rel, &ls_its));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  PetscCall(VecNorm(ctx->dU, NORM_2, &dU_norm));
  PetscCall(VecNorm(ctx->states[slot], NORM_2, &u_norm));
  rel_corr = (alpha * dU_norm) / PetscMax(u_norm, 1.0);
  PetscCall(StoppingByCriterion(criterion, rel, rel_corr, abs_dlambda, stop_tol, (PetscBool)(alpha > 0.0), &stop));
  if (alpha > 0.0) {
    PetscCall(VecCopy(ctx->u_trial, ctx->states[slot]));
    PetscCall(ScaleToOmega(ctx->f_ext, ctx->states[slot], omega_target));
    out->lambda_out = lambda_in + alpha * d_lambda;
    rel = trial_rel;
  }
  if (stop) out->converged = PETSC_TRUE;
  else {
    out->compute_diffs_out = PETSC_TRUE;
    if (alpha < 0.1) {
      if (alpha == 0.0) {
        out->compute_diffs_out = PETSC_FALSE;
        r *= 2.0;
      } else {
        r *= PetscPowReal(2.0, 0.25);
      }
    } else if (alpha > 0.5) {
      PetscCall(LinearSolverAppendRawBasis(&ctx->solver, ctx->dW, "SSR indirect dW"));
      PetscCall(LinearSolverAppendRawBasis(&ctx->solver, ctx->dV, "SSR indirect dV"));
      r = PetscMax(r / PetscSqrtReal(2.0), ctx->app.r_min);
    } else {
      PetscCall(LinearSolverAppendRawBasis(&ctx->solver, ctx->dW, "SSR indirect dW"));
      PetscCall(LinearSolverAppendRawBasis(&ctx->solver, ctx->dV, "SSR indirect dV"));
    }
  }
  out->failed = (PetscBool)(!out->converged && alpha == 0.0 && r > 1.0);
  out->solved = PETSC_TRUE;
  out->stop = stop;
  out->rel_residual = rel;
  out->rel_correction = rel_corr;
  out->alpha = alpha;
  out->r_out = r;
  out->delta_lambda = d_lambda;
  out->abs_delta_lambda = abs_dlambda;
  out->trial_rel = trial_rel;
  out->linear_its = itsW + itsV;
  out->linear_its_w = itsW;
  out->linear_its_v = itsV;
  out->line_search_its = ls_its;
  out->newton_its = 1;
  SSR_PROFILE_TIMER_END(NULL, &run_timer, &elapsed);
  out->wall_time = elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineAssembleFixed(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda, PetscReal r, P4SsrStepResult *out)
{
  PetscLogDouble  elapsed;
  SsrProfileTimer profile_timer;
  PetscReal      rel = PETSC_MAX_REAL;

  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ASSEMBLE_TANGENT, ctx->Ktangent, ctx->residual, &profile_timer);
  PetscCall(AssemblePlasticResidualJacobian(&ctx->actx, lambda, ctx->states[slot], ctx->f_ext, ctx->Ktangent, ctx->residual, PETSC_TRUE));
  PetscCall(BuildRegularizedOperator(ctx->Areg, ctx->Kelastic, ctx->Ktangent, r));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  PetscCall(ResidualNormFree(&ctx->actx, ctx->residual, ctx->rhs_norm, &rel));
  out->assembly_time = elapsed;
  out->rel_residual  = rel;
  out->r_out         = r;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineSolveFixedCorrection(P4SsrEngine *ctx, PetscInt slot, P4SsrStepResult *out)
{
  PetscInt       linear_its = 0;
  PetscLogDouble  elapsed;
  SsrProfileTimer profile_timer;

  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  PetscCall(VecCopy(ctx->residual, ctx->rhs));
  PetscCall(VecScale(ctx->rhs, -1.0));
  PetscCall(ApplyZeroDirichlet(ctx->actx.constrained_is, ctx->Areg, ctx->rhs));
  PetscCall(VecZeroEntries(ctx->du));
  ctx->solver.A = ctx->Areg;
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_KSP_SOLVE, ctx->Areg, ctx->rhs, &profile_timer);
  PetscCall(SolveLinearSystem(&ctx->solver, ctx->rhs, ctx->du, "SSR fixed-lambda Newton correction", ctx->states[slot], PETSC_TRUE, &linear_its));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  out->solved     = PETSC_TRUE;
  out->solve_time = elapsed;
  out->linear_its = linear_its;
  out->newton_its = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineFixedLineSearch(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda, P4SsrStepResult *out)
{
  PetscReal      alpha = 1.0, du_norm, u_norm, initial_decrease = PETSC_MAX_REAL, rel_corr;
  PetscInt       ls_its = 0;
  PetscLogDouble  elapsed;
  SsrProfileTimer profile_timer;

  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_LINE_SEARCH, ctx->states[slot], ctx->du, &profile_timer);
  PetscCall(VecNorm(ctx->du, NORM_2, &du_norm));
  PetscCall(VecNorm(ctx->states[slot], NORM_2, &u_norm));
  PetscCall(FixedLambdaDirectionalDamping(&ctx->actx, &ctx->app, lambda, ctx->states[slot], ctx->du, ctx->residual, ctx->f_ext, ctx->u_trial, ctx->r_trial, &alpha, &ls_its, &initial_decrease));
  rel_corr = (alpha * du_norm) / PetscMax(u_norm, 1.0e-30);
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  out->alpha              = alpha;
  out->rel_correction     = rel_corr;
  out->line_search_its    = ls_its;
  out->initial_decrease   = initial_decrease;
  out->wall_time          = elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineApplyFixedCorrection(P4SsrEngine *ctx, PetscInt slot, PetscReal alpha, PetscReal r_in, PetscBool update_basis, P4SsrStepResult *out)
{
  PetscReal r = r_in;

  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  PetscCall(VecScale(ctx->du, alpha));
  PetscCall(VecAXPY(ctx->states[slot], 1.0, ctx->du));
  if (update_basis) {
    if (alpha < 0.1) {
      if (alpha == 0.0) r *= 2.0;
      else r *= PetscPowReal(2.0, 0.25);
    } else {
      PetscCall(LinearSolverAppendRawBasis(&ctx->solver, ctx->du, "SSR fixed-lambda Newton correction"));
      if (alpha > 0.5) r = PetscMax(r / PetscSqrtReal(2.0), ctx->app.r_min);
    }
  }
  out->alpha  = alpha;
  out->r_out  = r;
  out->failed = (PetscBool)(update_basis && alpha == 0.0 && r > 1.0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode P4SsrEngineClearPreparedKSP(P4SsrEngine *ctx)
{
  PetscFunctionBeginUser;
  if (ctx->prepared_ksp_owned) PetscCall(KSPDestroy(&ctx->prepared_ksp));
  else ctx->prepared_ksp = NULL;
  ctx->prepared_ksp_owned        = PETSC_FALSE;
  ctx->prepared_basis_current    = PETSC_FALSE;
  ctx->prepared_nonlinear_tangent = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineAssembleResidualJacobian(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda, P4SsrStepResult *out)
{
  PetscLogDouble  elapsed;
  SsrProfileTimer profile_timer;
  PetscReal      rel = PETSC_MAX_REAL;

  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ASSEMBLE_TANGENT, ctx->Ktangent, ctx->residual, &profile_timer);
  PetscCall(AssemblePlasticResidualJacobian(&ctx->actx, lambda, ctx->states[slot], ctx->f_ext, ctx->Ktangent, ctx->residual, PETSC_TRUE));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  PetscCall(ResidualNormFree(&ctx->actx, ctx->residual, ctx->rhs_norm, &rel));
  out->assembly_time = elapsed;
  out->rel_residual  = rel;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineComputeLambdaDerivative(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda, P4SsrStepResult *out)
{
  PetscLogDouble  elapsed;
  SsrProfileTimer profile_timer;

  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ASSEMBLE_RESIDUAL, ctx->residual_eps, ctx->G, &profile_timer);
  PetscCall(ComputeLambdaDerivativeFD(&ctx->actx, &ctx->app, lambda, ctx->states[slot], ctx->f_ext, ctx->residual, ctx->residual_eps, ctx->G));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  out->wall_time = elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineBuildRegularizedOperator(P4SsrEngine *ctx, PetscReal r, P4SsrStepResult *out)
{
  PetscLogDouble  elapsed;
  SsrProfileTimer profile_timer;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "P4SsrEngine pointer is NULL");
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_OPERATOR_BUILD, ctx->Areg, ctx->Ktangent, &profile_timer);
  PetscCall(BuildRegularizedOperator(ctx->Areg, ctx->Kelastic, ctx->Ktangent, r));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  out->wall_time = elapsed;
  out->r_out     = r;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineBuildFixedCorrectionRHS(P4SsrEngine *ctx, P4SsrStepResult *out)
{
  PetscLogDouble  elapsed;
  SsrProfileTimer profile_timer;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "P4SsrEngine pointer is NULL");
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_BUILD_RHS, ctx->Areg, ctx->rhs, &profile_timer);
  PetscCall(VecCopy(ctx->residual, ctx->rhs));
  PetscCall(VecScale(ctx->rhs, -1.0));
  PetscCall(ApplyZeroDirichlet(ctx->actx.constrained_is, ctx->Areg, ctx->rhs));
  PetscCall(VecZeroEntries(ctx->du));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  out->wall_time = elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineBuildIndirectRHS(P4SsrEngine *ctx, P4SsrStepResult *out)
{
  PetscLogDouble  elapsed;
  SsrProfileTimer profile_timer;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "P4SsrEngine pointer is NULL");
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_BUILD_RHS, ctx->Areg, ctx->rhsW, &profile_timer);
  PetscCall(VecCopy(ctx->G, ctx->rhsW));
  PetscCall(VecScale(ctx->rhsW, -1.0));
  PetscCall(VecCopy(ctx->residual, ctx->rhsV));
  PetscCall(VecScale(ctx->rhsV, -1.0));
  PetscCall(ZeroConstrainedVector(ctx->actx.constrained_is, ctx->rhsW));
  PetscCall(ZeroConstrainedVector(ctx->actx.constrained_is, ctx->rhsV));
  PetscCall(VecZeroEntries(ctx->dW));
  PetscCall(VecZeroEntries(ctx->dV));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  out->wall_time = elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineKSPSetup(P4SsrEngine *ctx, PetscBool force_reuse_preconditioner, P4SsrStepResult *out)
{
  PetscBool      saved_force_reuse;
  PetscLogDouble  elapsed;
  SsrProfileTimer profile_timer;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "P4SsrEngine pointer is NULL");
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  PetscCall(P4SsrEngineClearPreparedKSP(ctx));
  ctx->solver.A = ctx->Areg;
  saved_force_reuse = ctx->solver.force_reuse_preconditioner;
  ctx->solver.force_reuse_preconditioner = force_reuse_preconditioner;
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_KSP_SETUP, ctx->Areg, NULL, &profile_timer);
  PetscCall(LinearSolverPrepareKSP(&ctx->solver, PETSC_TRUE, &ctx->prepared_ksp, &ctx->prepared_ksp_owned));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  ctx->solver.force_reuse_preconditioner = saved_force_reuse;
  ctx->prepared_nonlinear_tangent = PETSC_TRUE;
  out->wall_time = elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineAOrthogonalize(P4SsrEngine *ctx, const char label[], P4SsrStepResult *out)
{
  PetscLogDouble  elapsed;
  SsrProfileTimer profile_timer;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "P4SsrEngine pointer is NULL");
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_DEFLATION_ORTHO, ctx->Areg, NULL, &profile_timer);
  if (ctx->app.use_deflation && ctx->solver.n_raw_basis > 0) PetscCall(LinearSolverAOrthogonalizeBasis(&ctx->solver, ctx->Areg, label));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  ctx->prepared_basis_current = PETSC_TRUE;
  out->wall_time = elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode P4SsrEngineSolvePreparedVector(P4SsrEngine *ctx, Vec rhs, Vec x, const char label[], PetscInt *its, PetscLogDouble *solve_time)
{
  PetscLogDouble  elapsed;
  SsrProfileTimer profile_timer;

  PetscFunctionBeginUser;
  PetscCheck(ctx->prepared_ksp, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Call P4SsrEngineKSPSetup before ksp_solve");
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_KSP_SOLVE, ctx->prepared_ksp, rhs, &profile_timer);
  PetscCall(LinearSolverSolvePrepared(&ctx->solver, ctx->prepared_ksp, rhs, x, label, ctx->prepared_nonlinear_tangent, ctx->prepared_basis_current, its));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  PetscCall(SsrStatsAddLinearSolve(NULL, *its, KSP_CONVERGED_ITERATING));
  PetscCall(P4SsrEngineClearPreparedKSP(ctx));
  *solve_time = elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineKSPSolveFixedCorrection(P4SsrEngine *ctx, P4SsrStepResult *out)
{
  PetscInt       its = 0;
  PetscLogDouble solve_time = 0.0;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "P4SsrEngine pointer is NULL");
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  PetscCall(P4SsrEngineSolvePreparedVector(ctx, ctx->rhs, ctx->du, "SSR fixed-lambda Newton correction", &its, &solve_time));
  out->solved     = PETSC_TRUE;
  out->solve_time = solve_time;
  out->linear_its = its;
  out->newton_its = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineKSPSolveIndirectW(P4SsrEngine *ctx, P4SsrStepResult *out)
{
  PetscInt       its = 0;
  PetscLogDouble solve_time = 0.0;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "P4SsrEngine pointer is NULL");
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  PetscCall(P4SsrEngineSolvePreparedVector(ctx, ctx->rhsW, ctx->dW, "SSR indirect dW", &its, &solve_time));
  out->solved       = PETSC_TRUE;
  out->solve_time   = solve_time;
  out->linear_its_w = its;
  out->linear_its   = its;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineKSPSolveIndirectV(P4SsrEngine *ctx, P4SsrStepResult *out)
{
  PetscInt       its = 0;
  PetscLogDouble solve_time = 0.0;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "P4SsrEngine pointer is NULL");
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  PetscCall(P4SsrEngineSolvePreparedVector(ctx, ctx->rhsV, ctx->dV, "SSR indirect dV", &its, &solve_time));
  out->solved       = PETSC_TRUE;
  out->solve_time   = solve_time;
  out->linear_its_v = its;
  out->linear_its   = its;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineAssembleIndirect(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda, PetscReal r, PetscBool compute_diffs, P4SsrStepResult *out)
{
  PetscLogDouble  elapsed;
  SsrProfileTimer profile_timer;
  PetscReal      rel = PETSC_MAX_REAL;
  const SsrEvent assembly_event = compute_diffs ? SSR_EVENT_ASSEMBLE_RESIDUAL : SSR_EVENT_ASSEMBLE_TANGENT;

  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  SSR_PROFILE_TIMER_BEGIN(NULL, assembly_event, ctx->Ktangent, ctx->residual, &profile_timer);
  PetscCall(AssemblePlasticResidualJacobian(&ctx->actx, lambda, ctx->states[slot], ctx->f_ext, ctx->Ktangent, ctx->residual, PETSC_TRUE));
  PetscCall(ResidualNormFree(&ctx->actx, ctx->residual, ctx->rhs_norm, &rel));
  if (compute_diffs) PetscCall(ComputeLambdaDerivativeFD(&ctx->actx, &ctx->app, lambda, ctx->states[slot], ctx->f_ext, ctx->residual, ctx->residual_eps, ctx->G));
  PetscCall(BuildRegularizedOperator(ctx->Areg, ctx->Kelastic, ctx->Ktangent, r));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  out->assembly_time     = elapsed;
  out->rel_residual      = rel;
  out->r_out             = r;
  out->compute_diffs_out = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineSolveIndirectDirections(P4SsrEngine *ctx, PetscInt slot, P4SsrStepResult *out)
{
  PetscInt       itsW = 0, itsV = 0;
  PetscLogDouble  elapsed;
  SsrProfileTimer profile_timer;
  PetscBool      saved_force_reuse;

  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  PetscCall(VecCopy(ctx->G, ctx->rhsW));
  PetscCall(VecScale(ctx->rhsW, -1.0));
  PetscCall(VecCopy(ctx->residual, ctx->rhsV));
  PetscCall(VecScale(ctx->rhsV, -1.0));
  PetscCall(ZeroConstrainedVector(ctx->actx.constrained_is, ctx->rhsW));
  PetscCall(ZeroConstrainedVector(ctx->actx.constrained_is, ctx->rhsV));
  ctx->solver.A = ctx->Areg;
  PetscCall(VecZeroEntries(ctx->dW));
  PetscCall(VecZeroEntries(ctx->dV));
  saved_force_reuse = ctx->solver.force_reuse_preconditioner;
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_KSP_SOLVE, ctx->Areg, ctx->rhsW, &profile_timer);
  ctx->solver.force_reuse_preconditioner = PETSC_FALSE;
  PetscCall(SolveLinearSystem(&ctx->solver, ctx->rhsW, ctx->dW, "SSR indirect dW", ctx->states[slot], PETSC_TRUE, &itsW));
  ctx->solver.force_reuse_preconditioner = PETSC_TRUE;
  PetscCall(SolveLinearSystem(&ctx->solver, ctx->rhsV, ctx->dV, "SSR indirect dV", ctx->states[slot], PETSC_TRUE, &itsV));
  ctx->solver.force_reuse_preconditioner = saved_force_reuse;
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  out->solved       = PETSC_TRUE;
  out->solve_time   = elapsed;
  out->linear_its_w = itsW;
  out->linear_its_v = itsV;
  out->linear_its   = itsW + itsV;
  out->newton_its   = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineFormIndirectUpdate(P4SsrEngine *ctx, P4SsrStepResult *out)
{
  PetscReal   denom, numer, d_lambda;
  PetscScalar dot;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "P4SsrEngine pointer is NULL");
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  PetscCall(VecDot(ctx->f_ext, ctx->dW, &dot));
  denom = PetscRealPart(dot);
  PetscCall(VecDot(ctx->f_ext, ctx->dV, &dot));
  numer = PetscRealPart(dot);
  PetscCheck(PetscAbsReal(denom) > 1.0e-30, PetscObjectComm((PetscObject)ctx->dm), PETSC_ERR_NOT_CONVERGED,
             "Indirect Newton has singular constraint denominator f_ext^T dW=%.6e", (double)denom);
  d_lambda = -numer / denom;
  PetscCall(VecCopy(ctx->dV, ctx->dU));
  PetscCall(VecAXPY(ctx->dU, d_lambda, ctx->dW));
  out->delta_lambda     = d_lambda;
  out->abs_delta_lambda = PetscAbsReal(d_lambda);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineIndirectLineSearch(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda, PetscReal omega_target, PetscReal current_rel, PetscReal d_lambda, P4SsrStepResult *out)
{
  PetscReal      alpha = 0.0, trial_rel = PETSC_MAX_REAL, dU_norm, u_norm, rel_corr;
  PetscInt       ls_its = 0;
  PetscLogDouble  elapsed;
  SsrProfileTimer profile_timer;

  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_LINE_SEARCH, ctx->states[slot], ctx->dU, &profile_timer);
  PetscCall(IndirectNewtonLineSearch(&ctx->actx, &ctx->app, ctx->f_ext, ctx->states[slot], lambda, ctx->dU, d_lambda, omega_target, current_rel, ctx->rhs_norm, ctx->u_trial, ctx->r_trial, &alpha, &trial_rel, &ls_its));
  PetscCall(VecNorm(ctx->dU, NORM_2, &dU_norm));
  PetscCall(VecNorm(ctx->states[slot], NORM_2, &u_norm));
  rel_corr = (alpha * dU_norm) / PetscMax(u_norm, 1.0);
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  out->alpha           = alpha;
  out->trial_rel       = trial_rel;
  out->rel_correction  = rel_corr;
  out->line_search_its = ls_its;
  out->wall_time       = elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineAcceptIndirectUpdate(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda_in, PetscReal omega_target, PetscReal alpha, PetscReal d_lambda, PetscReal r_in, PetscBool update_basis, P4SsrStepResult *out)
{
  PetscReal r = r_in;

  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  out->lambda_out        = lambda_in;
  out->compute_diffs_out = PETSC_TRUE;
  if (alpha > 0.0) {
    PetscCall(VecCopy(ctx->u_trial, ctx->states[slot]));
    PetscCall(ScaleToOmega(ctx->f_ext, ctx->states[slot], omega_target));
    out->lambda_out = lambda_in + alpha * d_lambda;
  }
  if (update_basis) {
    if (alpha < 0.1) {
      if (alpha == 0.0) {
        out->compute_diffs_out = PETSC_FALSE;
        r *= 2.0;
      } else {
        r *= PetscPowReal(2.0, 0.25);
      }
    } else if (alpha > 0.5) {
      PetscCall(LinearSolverAppendRawBasis(&ctx->solver, ctx->dW, "SSR indirect dW"));
      PetscCall(LinearSolverAppendRawBasis(&ctx->solver, ctx->dV, "SSR indirect dV"));
      r = PetscMax(r / PetscSqrtReal(2.0), ctx->app.r_min);
    } else {
      PetscCall(LinearSolverAppendRawBasis(&ctx->solver, ctx->dW, "SSR indirect dW"));
      PetscCall(LinearSolverAppendRawBasis(&ctx->solver, ctx->dV, "SSR indirect dV"));
    }
  }
  out->alpha        = alpha;
  out->delta_lambda = d_lambda;
  out->r_out        = r;
  out->failed       = (PetscBool)(update_basis && alpha == 0.0 && r > 1.0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineResidualRel(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda, PetscReal *rel)
{
  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCall(AssembleResidualRel(&ctx->actx, lambda, ctx->states[slot], ctx->f_ext, ctx->residual, ctx->rhs_norm, rel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineSolveElasticInitialGuess(P4SsrEngine *ctx, PetscInt slot, PetscReal scale, P4SsrStepResult *out)
{
  PetscInt       its = 0;
  PetscLogDouble  elapsed;
  SsrProfileTimer profile_timer;

  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  ctx->solver.A = ctx->Kelastic;
  PetscCall(VecZeroEntries(ctx->states[slot]));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_KSP_SOLVE, ctx->Kelastic, ctx->f_ext, &profile_timer);
  PetscCall(SolveLinearSystem(&ctx->solver, ctx->f_ext, ctx->states[slot], "LL elastic initial guess", NULL, PETSC_FALSE, &its));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  PetscCall(LinearSolverAppendRawBasis(&ctx->solver, ctx->states[slot], "LL elastic initial guess"));
  PetscCall(VecScale(ctx->states[slot], scale));
  out->solved     = PETSC_TRUE;
  out->solve_time = elapsed;
  out->linear_its = its;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineAssembleLimitLoad(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda_ell, PetscReal load_t, PetscReal r, P4SsrStepResult *out)
{
  PetscLogDouble  elapsed;
  SsrProfileTimer profile_timer;
  PetscReal      rel = PETSC_MAX_REAL;

  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_ASSEMBLE_TANGENT, ctx->Ktangent, ctx->residual, &profile_timer);
  PetscCall(AssemblePlasticResidualJacobian(&ctx->actx, lambda_ell, ctx->states[slot], ctx->f_ext, ctx->Ktangent, ctx->residual, PETSC_TRUE));
  PetscCall(VecAXPY(ctx->residual, 1.0 - load_t, ctx->f_ext));
  PetscCall(ZeroConstrainedVector(ctx->actx.constrained_is, ctx->residual));
  PetscCall(BuildRegularizedOperator(ctx->Areg, ctx->Kelastic, ctx->Ktangent, r));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  PetscCall(ResidualNormFree(&ctx->actx, ctx->residual, ctx->rhs_norm, &rel));
  out->assembly_time = elapsed;
  out->rel_residual  = rel;
  out->r_out         = r;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineBuildLimitLoadRHS(P4SsrEngine *ctx, PetscReal load_t, P4SsrStepResult *out)
{
  PetscLogDouble  elapsed;
  SsrProfileTimer profile_timer;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "P4SsrEngine pointer is NULL");
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  (void)load_t;
  PetscCall(PetscMemzero(out, sizeof(*out)));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_BUILD_RHS, ctx->Areg, ctx->rhsW, &profile_timer);
  PetscCall(VecCopy(ctx->f_ext, ctx->rhsW));
  PetscCall(VecCopy(ctx->residual, ctx->rhsV));
  PetscCall(VecScale(ctx->rhsV, -1.0));
  PetscCall(ZeroConstrainedVector(ctx->actx.constrained_is, ctx->rhsW));
  PetscCall(ZeroConstrainedVector(ctx->actx.constrained_is, ctx->rhsV));
  PetscCall(VecZeroEntries(ctx->dW));
  PetscCall(VecZeroEntries(ctx->dV));
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  out->wall_time = elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineFormLimitLoadUpdate(P4SsrEngine *ctx, P4SsrStepResult *out)
{
  PetscReal   denom, numer, d_t;
  PetscScalar dot;

  PetscFunctionBeginUser;
  PetscCheck(ctx, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "P4SsrEngine pointer is NULL");
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  PetscCall(VecDot(ctx->f_ext, ctx->dW, &dot));
  denom = PetscRealPart(dot);
  PetscCall(VecDot(ctx->f_ext, ctx->dV, &dot));
  numer = PetscRealPart(dot);
  PetscCheck(PetscAbsReal(denom) > 1.0e-30, PetscObjectComm((PetscObject)ctx->dm), PETSC_ERR_NOT_CONVERGED,
             "Limit-load Newton has singular constraint denominator f_ext^T dW=%.6e", (double)denom);
  d_t = -numer / denom;
  PetscCall(VecCopy(ctx->dV, ctx->dU));
  PetscCall(VecAXPY(ctx->dU, d_t, ctx->dW));
  out->delta_lambda     = d_t;
  out->abs_delta_lambda = PetscAbsReal(d_t);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineLimitLoadLineSearch(P4SsrEngine *ctx, PetscInt slot, PetscReal lambda_ell, PetscReal load_t, P4SsrStepResult *out)
{
  PetscReal      alpha = 1.0, alpha_min = 0.0, alpha_max = 1.0;
  PetscReal      dU_norm, u_norm, rel_corr, initial_decrease = PETSC_MAX_REAL;
  PetscScalar    dot;
  PetscInt       ls_its = 0;
  PetscLogDouble  elapsed;
  SsrProfileTimer profile_timer;

  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  SSR_PROFILE_TIMER_BEGIN(NULL, SSR_EVENT_LINE_SEARCH, ctx->states[slot], ctx->dU, &profile_timer);
  PetscCall(VecCopy(ctx->residual, ctx->rhs));
  PetscCall(VecAXPY(ctx->rhs, load_t, ctx->f_ext)); /* internal force F_int = (F_int - t f) + t f */
  PetscCall(ZeroConstrainedVector(ctx->actx.constrained_is, ctx->rhs));
  PetscCall(VecDot(ctx->rhs, ctx->dU, &dot));
  initial_decrease = PetscRealPart(dot);
  PetscCall(VecNorm(ctx->dU, NORM_2, &dU_norm));
  PetscCall(VecNorm(ctx->states[slot], NORM_2, &u_norm));
  if (PetscIsInfOrNanReal(initial_decrease) || PetscIsInfOrNanReal(dU_norm) || initial_decrease >= 0.0) {
    alpha = 0.0;
  } else if (ctx->app.it_damp_max == 0) {
    alpha = 1.0;
  } else {
    for (PetscInt damp = 0; damp < ctx->app.it_damp_max; ++damp) {
      PetscReal decrease;

      ls_its++;
      PetscCall(VecWAXPY(ctx->u_trial, alpha, ctx->dU, ctx->states[slot]));
      PetscCall(AssemblePlasticResidualJacobian(&ctx->actx, lambda_ell, ctx->u_trial, ctx->f_ext, NULL, ctx->r_trial, PETSC_FALSE));
      PetscCall(VecAXPY(ctx->r_trial, 1.0, ctx->f_ext));
      PetscCall(ZeroConstrainedVector(ctx->actx.constrained_is, ctx->r_trial));
      PetscCall(VecDot(ctx->r_trial, ctx->dU, &dot));
      decrease = PetscRealPart(dot);
      if (!PetscIsInfOrNanReal(decrease) && decrease < 0.0) {
        if (alpha == 1.0) break;
        alpha_min = alpha;
      } else {
        alpha_max = alpha;
      }
      alpha = 0.5 * (alpha_min + alpha_max);
    }
  }
  if (alpha > 0.0) PetscCall(VecWAXPY(ctx->u_trial, alpha, ctx->dU, ctx->states[slot]));
  rel_corr = (alpha * dU_norm) / PetscMax(u_norm, 1.0e-30);
  SSR_PROFILE_TIMER_END(NULL, &profile_timer, &elapsed);
  out->alpha            = alpha;
  out->rel_correction   = rel_corr;
  out->line_search_its  = ls_its;
  out->initial_decrease = initial_decrease;
  out->wall_time        = elapsed;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode P4SsrEngineAcceptLimitLoadUpdate(P4SsrEngine *ctx, PetscInt slot, PetscReal load_t, PetscReal omega_target, PetscReal alpha, PetscReal d_t, PetscReal r_in, PetscBool update_basis, P4SsrStepResult *out)
{
  PetscReal r = r_in;

  PetscFunctionBeginUser;
  PetscCall(P4SsrEngineCheckSlot(ctx, slot));
  PetscCheck(out, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Step result pointer is NULL");
  PetscCall(PetscMemzero(out, sizeof(*out)));
  out->lambda_out = load_t + d_t;
  if (alpha > 0.0) {
    PetscCall(VecCopy(ctx->u_trial, ctx->states[slot]));
  }
  PetscCall(ScaleToOmega(ctx->f_ext, ctx->states[slot], omega_target));
  if (update_basis) {
    if (alpha < 0.1) {
      if (alpha == 0.0) r *= 2.0;
      else r *= PetscPowReal(2.0, 0.25);
    } else {
      PetscCall(LinearSolverAppendRawBasis(&ctx->solver, ctx->dW, "LL indirect dW"));
      PetscCall(LinearSolverAppendRawBasis(&ctx->solver, ctx->dV, "LL indirect dV"));
      if (alpha > 0.5) r = PetscMax(r / PetscSqrtReal(2.0), ctx->app.r_min);
    }
  }
  out->alpha        = alpha;
  out->delta_lambda = d_t;
  out->r_out        = r;
  out->failed       = (PetscBool)(update_basis && alpha == 0.0 && r > 1.0);
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  char      kind[64];
  PetscReal lambda;
  PetscReal omega;
  PetscReal r;
  PetscReal rel_residual;
  PetscInt  sample_id;
  PetscInt  newton_iteration;
  PetscInt  basis_cols;
  PetscInt  expected_dW_iterations;
  PetscInt  expected_dV_iterations;
  PetscReal expected_dW_reported_residual_final;
  PetscReal expected_dV_reported_residual_final;
  PetscInt  expected_iterations;
  PetscReal expected_reported_residual_final;
  PetscReal expected_true_residual_final;
  PetscReal expected_alpha;
  PetscInt  expected_line_search_iterations;
  PetscReal expected_initial_decrease;
  PetscReal expected_rel_correction;
  PetscBool expected_stop;
} LinearReplayMeta;
