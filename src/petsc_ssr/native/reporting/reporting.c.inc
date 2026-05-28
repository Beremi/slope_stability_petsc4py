typedef struct {
  PetscInt       step;
  char           phase[16];
  PetscReal      omega;
  PetscReal      lambda;
  PetscReal      d_omega;
  PetscReal      d_lambda;
  PetscReal      u_max;
  PetscInt       attempts;
  PetscInt       newton_iterations;
  PetscInt       linear_iterations;
  PetscInt       line_search_iterations;
  PetscReal      rel_residual;
  PetscReal      rel_correction;
  PetscLogDouble step_wall_time;
  char           stop_reason[64];
} SSRCurveRow;

typedef struct {
  PetscInt       accepted_steps;
  PetscInt       total_newton_its;
  PetscInt       total_linear_its;
  PetscInt       total_line_search_its;
  PetscReal      omega_last;
  PetscReal      lambda_last;
  PetscReal      final_rel;
  PetscReal      final_rel_correction;
  PetscLogDouble wall_time;
  char           stop_reason[64];
} SSRStats;

typedef struct {
  AppCtx          app;
  P4Basis         basis;
  DM              dm;
  AssemblyCtx     actx;
  LinearSolverCtx solver;
  Mat             Areg;
  Mat             Kelastic;
  Mat             Ktangent;
  Vec             f_ext;
  Vec             states[4];
  Vec             residual;
  Vec             residual_eps;
  Vec             G;
  Vec             rhs;
  Vec             rhsW;
  Vec             rhsV;
  Vec             du;
  Vec             dW;
  Vec             dV;
  Vec             dU;
  Vec             u_trial;
  Vec             r_trial;
  KSP             prepared_ksp;
  PetscBool       prepared_ksp_owned;
  PetscBool       prepared_basis_current;
  PetscBool       prepared_nonlinear_tangent;
  PetscReal       rhs_norm;
  PetscInt        global_dofs;
  PetscLogDouble  elastic_assembly_time;
  PetscLogDouble  create_time;
} P4SsrEngine;

typedef struct {
  PetscInt      ranks;
  PetscInt      global_dofs;
  PetscInt      basis_cols;
  PetscReal     rhs_norm;
  PetscLogDouble elastic_assembly_time;
  PetscLogDouble create_time;
  PetscLogDouble deflation_orthogonalization_time;
  PetscLogDouble deflation_coarse_initial_time;
  PetscLogDouble deflation_pc_apply_time;
  PetscLogDouble deflation_projector_time;
  PetscInt       deflation_coarse_initial_calls;
  PetscInt       deflation_projected_pc_calls;
} P4SsrEngineInfo;

typedef struct {
  PetscBool     converged;
  PetscBool     solved;
  PetscBool     failed;
  PetscBool     stop;
  PetscBool     compute_diffs_out;
  PetscReal     rel_residual;
  PetscReal     rel_correction;
  PetscReal     alpha;
  PetscReal     r_out;
  PetscReal     lambda_out;
  PetscReal     delta_lambda;
  PetscReal     trial_rel;
  PetscReal     abs_delta_lambda;
  PetscReal     initial_decrease;
  PetscInt      linear_its;
  PetscInt      linear_its_w;
  PetscInt      linear_its_v;
  PetscInt      line_search_its;
  PetscInt      newton_its;
  PetscLogDouble assembly_time;
  PetscLogDouble solve_time;
  PetscLogDouble wall_time;
} P4SsrStepResult;

static PetscErrorCode DotOmega(Vec f_ext, Vec u, PetscReal *omega)
{
  PetscScalar dot;

  PetscFunctionBeginUser;
  PetscCall(VecDot(f_ext, u, &dot));
  *omega = PetscRealPart(dot);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DisplacementMax(Vec u, PetscReal *u_max)
{
  PetscFunctionBeginUser;
  PetscCall(VecNorm(u, NORM_INFINITY, u_max));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ScaleToOmega(Vec f_ext, Vec u, PetscReal omega_target)
{
  PetscReal omega;

  PetscFunctionBeginUser;
  PetscCall(DotOmega(f_ext, u, &omega));
  PetscCheck(PetscAbsReal(omega) > 1.0e-30, PetscObjectComm((PetscObject)u), PETSC_ERR_NOT_CONVERGED,
             "Cannot rescale displacement to omega %.6e because f_ext^T u is %.6e", (double)omega_target, (double)omega);
  PetscCall(VecScale(u, omega_target / omega));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AssembleResidualRel(AssemblyCtx *actx, PetscReal lambda, Vec u, Vec f_ext, Vec residual, PetscReal rhs_norm, PetscReal *rel)
{
  PetscFunctionBeginUser;
  PetscCall(AssemblePlasticResidualJacobian(actx, lambda, u, f_ext, NULL, residual, PETSC_FALSE));
  PetscCall(ResidualNormFree(actx, residual, rhs_norm, rel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FixedLambdaDirectionalDamping(AssemblyCtx *actx, AppCtx *app, PetscReal lambda, Vec u, Vec du, Vec current_residual, Vec f_ext, Vec u_trial, Vec r_trial, PetscReal *alpha_out, PetscInt *ls_its, PetscReal *initial_decrease)
{
  PetscReal alpha = 1.0, alpha_min = 0.0, alpha_max = 1.0;
  PetscScalar dot;

  PetscFunctionBeginUser;
  *alpha_out       = 0.0;
  *ls_its          = 0;
  *initial_decrease = PETSC_MAX_REAL;

  PetscCall(VecDot(current_residual, du, &dot));
  *initial_decrease = PetscRealPart(dot);
  if (PetscIsInfOrNanReal(*initial_decrease) || *initial_decrease >= 0.0) PetscFunctionReturn(PETSC_SUCCESS);
  if (app->it_damp_max == 0) {
    *alpha_out = 1.0;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  for (PetscInt damp = 0; damp < app->it_damp_max; ++damp) {
    PetscReal decrease;

    (*ls_its)++;
    PetscCall(VecWAXPY(u_trial, alpha, du, u));
    PetscCall(AssemblePlasticResidualJacobian(actx, lambda, u_trial, f_ext, NULL, r_trial, PETSC_FALSE));
    PetscCall(ZeroConstrainedVector(actx->constrained_is, r_trial));
    PetscCall(VecDot(r_trial, du, &dot));
    decrease = PetscRealPart(dot);
    if (!PetscIsInfOrNanReal(decrease) && decrease < 0.0) {
      if (alpha == 1.0) break;
      alpha_min = alpha;
    } else {
      alpha_max = alpha;
    }
    alpha = 0.5 * (alpha_min + alpha_max);
  }
  *alpha_out = alpha;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StoppingByCriterion(const char criterion[], PetscReal rel_residual, PetscReal rel_correction, PetscReal abs_delta_lambda, PetscReal tol, PetscBool have_correction, PetscBool *stop)
{
  PetscBool is_rel_res, is_rel_corr, is_abs_lambda;

  PetscFunctionBeginUser;
  *stop = PETSC_FALSE;
  PetscCall(PetscStrcasecmp(criterion, "relative_residual", &is_rel_res));
  PetscCall(PetscStrcasecmp(criterion, "relative_correction", &is_rel_corr));
  PetscCall(PetscStrcasecmp(criterion, "absolute_delta_lambda", &is_abs_lambda));
  if (is_rel_res) *stop = (PetscBool)(rel_residual <= tol);
  else if (is_rel_corr) *stop = (PetscBool)(have_correction && rel_correction <= tol);
  else if (is_abs_lambda) *stop = (PetscBool)(have_correction && abs_delta_lambda <= tol);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeLambdaDerivativeFD(AssemblyCtx *actx, AppCtx *app, PetscReal lambda, Vec u, Vec f_ext, Vec residual, Vec residual_eps, Vec G)
{
  PetscReal eps = PetscMax(1.0e-12, app->newton_rtol / 1000.0);

  PetscFunctionBeginUser;
  PetscCall(AssemblePlasticResidualJacobian(actx, lambda + eps, u, f_ext, NULL, residual_eps, PETSC_FALSE));
  PetscCall(VecWAXPY(G, -1.0, residual, residual_eps));
  PetscCall(VecScale(G, 1.0 / eps));
  PetscCall(ZeroConstrainedVector(actx->constrained_is, G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildRegularizedOperator(Mat Areg, Mat Kelastic, Mat Ktangent, PetscReal r)
{
  PetscFunctionBeginUser;
  PetscCall(MatSetOption(Areg, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatZeroEntries(Areg));
  PetscCall(MatAXPY(Areg, 1.0 - r, Ktangent, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatAXPY(Areg, r, Kelastic, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatAssemblyBegin(Areg, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Areg, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(Areg, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CurveWriteHeader(FILE *fp)
{
  PetscFunctionBeginUser;
  if (!fp) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(fprintf(fp, "step,phase,omega,lambda,d_omega,d_lambda,u_max,attempts,newton_iterations,linear_iterations,line_search_iterations,rel_residual,rel_correction,step_wall_time,stop_reason\n") > 0,
             PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Failed to write continuation CSV header");
  fflush(fp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CurveWriteRow(FILE *fp, const SSRCurveRow *row)
{
  PetscFunctionBeginUser;
  if (!fp) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(fprintf(fp,
                     "%" PetscInt_FMT ",%s,%.16e,%.16e,%.16e,%.16e,%.16e,%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ",%" PetscInt_FMT ",%.16e,%.16e,%.16e,%s\n",
                     row->step, row->phase, (double)row->omega, (double)row->lambda, (double)row->d_omega, (double)row->d_lambda,
                     (double)row->u_max, row->attempts, row->newton_iterations, row->linear_iterations, row->line_search_iterations,
                     (double)row->rel_residual, (double)row->rel_correction, (double)row->step_wall_time, row->stop_reason) > 0,
             PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Failed to write continuation CSV row");
  fflush(fp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteFinalSolutionOutputs(const AppCtx *app, DM dm, AssemblyCtx *actx, Vec u)
{
  PetscViewer viewer = NULL;

  PetscFunctionBeginUser;
  if (!app->solution_binary[0] && !app->solution_vtk[0] && !app->solution_points_csv[0]) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectSetName((PetscObject)u, "displacement"));
  if (app->solution_binary[0]) {
    PetscCall(PetscViewerBinaryOpen(PetscObjectComm((PetscObject)u), app->solution_binary, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(u, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)u), "SSR_SOLUTION_OUTPUT kind=binary path=%s\n", app->solution_binary));
  }
  if (app->solution_vtk[0]) {
    PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)dm), app->solution_vtk, FILE_MODE_WRITE, &viewer));
    PetscCall(DMView(dm, viewer));
    PetscCall(VecView(u, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)u), "SSR_SOLUTION_OUTPUT kind=vtk path=%s\n", app->solution_vtk));
  }
  if (app->solution_points_csv[0]) {
    PetscCall(AssemblyWriteDisplacementPointCSV(actx, u, app->solution_points_csv));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteSummaryJSON(const AppCtx *app, PetscMPIInt ranks, const char part_type[], PetscInt global_dofs, PetscLogDouble elastic_assembly_time, PetscLogDouble wall_time, const SSRStats *stats, const LinearSolverCtx *solver)
{
  PetscMPIInt rank;
  FILE       *fp = NULL;

  PetscFunctionBeginUser;
  if (!app->summary_json[0]) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank) PetscFunctionReturn(PETSC_SUCCESS);

  fp = fopen(app->summary_json, "w");
  PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Failed to open summary JSON %s", app->summary_json);
  PetscCheck(fprintf(fp,
                     "{\n"
                     "  \"engine\": \"petsc_ssr_full_c\",\n"
                     "  \"variant\": \"%s\",\n"
                     "  \"continuation_method\": \"%s\",\n"
                     "  \"ranks\": %d,\n"
                     "  \"partitioner\": \"%s\",\n"
                     "  \"global_dofs\": %" PetscInt_FMT ",\n"
                     "  \"accepted_steps\": %" PetscInt_FMT ",\n"
                     "  \"total_newton_its\": %" PetscInt_FMT ",\n"
                     "  \"total_linear_its\": %" PetscInt_FMT ",\n"
                     "  \"total_line_search_its\": %" PetscInt_FMT ",\n"
                     "  \"elastic_assembly_time\": %.17g,\n"
                     "  \"continuation_wall_time\": %.17g,\n"
                     "  \"wall_time\": %.17g,\n"
                     "  \"omega_last\": %.17g,\n"
                     "  \"lambda_last\": %.17g,\n"
                     "  \"final_rel\": %.17g,\n"
                     "  \"final_rel_correction\": %.17g,\n"
                     "  \"stop_reason\": \"%s\",\n"
                     "  \"deflation\": %s,\n"
                     "  \"deflation_solver\": \"%s\",\n"
                     "  \"deflation_basis_cols\": %" PetscInt_FMT ",\n"
                     "  \"deflation_orthogonalization_time\": %.17g,\n"
                     "  \"deflation_coarse_initial_time\": %.17g,\n"
                     "  \"deflation_coarse_initial_calls\": %" PetscInt_FMT ",\n"
                     "  \"deflation_pc_apply_time\": %.17g,\n"
                     "  \"deflation_projector_time\": %.17g,\n"
                     "  \"deflation_projected_pc_calls\": %" PetscInt_FMT ",\n"
                     "  \"curve_csv\": \"%s\",\n"
                     "  \"solution_binary\": \"%s\",\n"
                     "  \"solution_vtk\": \"%s\",\n"
                     "  \"solution_points_csv\": \"%s\"\n"
                     "}\n",
                     app->variant_name, app->continuation_method, (int)ranks, part_type ? part_type : "unknown", global_dofs,
                     stats->accepted_steps, stats->total_newton_its, stats->total_linear_its, stats->total_line_search_its,
                     (double)elastic_assembly_time, (double)stats->wall_time, (double)wall_time,
                     (double)stats->omega_last, (double)stats->lambda_last, (double)stats->final_rel,
                     (double)stats->final_rel_correction, stats->stop_reason, app->use_deflation ? "true" : "false",
                     app->deflation_solver_name, solver->n_raw_basis, (double)solver->deflation_orthogonalization_time,
                     (double)solver->deflation_coarse_time, solver->deflation_coarse_calls,
                     (double)solver->deflation_pc_apply_time, (double)solver->deflation_projector_time,
                     solver->deflation_projected_pc_calls, app->curve_csv, app->solution_binary, app->solution_vtk, app->solution_points_csv) > 0,
             PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Failed to write summary JSON %s", app->summary_json);
  fclose(fp);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FillCurveRow(SSRCurveRow *row, PetscInt step, const char phase[], PetscReal omega, PetscReal lambda, PetscReal d_omega, PetscReal d_lambda, Vec u, PetscInt attempts, const NewtonStats *stats, const char reason[])
{
  PetscFunctionBeginUser;
  PetscCall(PetscMemzero(row, sizeof(*row)));
  row->step     = step;
  row->omega    = omega;
  row->lambda   = lambda;
  row->d_omega  = d_omega;
  row->d_lambda = d_lambda;
  row->attempts = attempts;
  PetscCall(PetscStrncpy(row->phase, phase, sizeof(row->phase)));
  PetscCall(PetscStrncpy(row->stop_reason, reason, sizeof(row->stop_reason)));
  PetscCall(DisplacementMax(u, &row->u_max));
  if (stats) {
    row->newton_iterations     = stats->newton_its;
    row->linear_iterations     = stats->total_linear_its;
    row->line_search_iterations = stats->line_search_its;
    row->rel_residual          = stats->final_rel;
    row->rel_correction        = stats->final_rel_correction;
    row->step_wall_time        = stats->wall_time;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
